"""
lacuna.models.reconstruction.heads

Concrete reconstruction head implementations.

Head Types:
    MCARHead: Simple MLP, no cross-column structure exploited.
    MARHead: Cross-attention to observed cells in the same row.
    MNARSelfCensoringHead: Predicts with censoring score for extreme values.
    MNARThresholdHead: Learns soft thresholds for truncation patterns.
    MNARLatentHead: Infers latent confounder driving missingness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from lacuna.models.reconstruction.base import BaseReconstructionHead, ReconstructionConfig
from lacuna.data.tokenization import IDX_VALUE, IDX_OBSERVED


# =============================================================================
# MCAR Head: Simple MLP (no special structure)
# =============================================================================

class MCARHead(BaseReconstructionHead):
    """
    MCAR reconstruction head.
    
    Architecture:
        Simple MLP applied independently to each token representation.
    
    Inductive bias:
        Missing values are random; we can only use the global distribution
        learned from training to make predictions. No special cross-column
        structure is exploited.
    
    Expected behavior:
        - Under MCAR: Moderate error (predicting population mean-ish values)
        - Under MAR: Higher error (missing context that would help)
        - Under MNAR: Variable error (depends on the specific pattern)
    """
    
    def __init__(self, config: ReconstructionConfig):
        super().__init__(config)
        
        # Simple MLP: hidden_dim -> head_hidden_dim -> ... -> 1
        layers = []
        in_dim = config.hidden_dim
        
        for _ in range(config.n_head_layers - 1):
            layers.extend([
                nn.Linear(in_dim, config.head_hidden_dim),
                nn.LayerNorm(config.head_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = config.head_hidden_dim
        
        # Final prediction layer
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        token_repr: torch.Tensor,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict values using simple MLP on each token independently."""
        # token_repr: [B, max_rows, max_cols, hidden_dim]
        predictions = self.mlp(token_repr).squeeze(-1)  # [B, max_rows, max_cols]
        return predictions


# =============================================================================
# MAR Head: Cross-Attention to Observed Values
# =============================================================================

class MARHead(BaseReconstructionHead):
    """
    MAR reconstruction head.
    
    Architecture:
        Cross-attention from each cell to observed cells in the same row,
        then MLP to predict the value.
    
    Inductive bias:
        Missing values can be predicted from other observed values in the
        same row. This is exactly what MAR assumes - missingness depends on
        observed variables, so observed variables carry information about
        missing ones.
    
    Expected behavior:
        - Under MCAR: Moderate error (cross-attention doesn't help much)
        - Under MAR: LOW error (observed values predict missing ones)
        - Under MNAR: Higher error (missing value info not in observed context)
    """
    
    def __init__(self, config: ReconstructionConfig):
        super().__init__(config)
        
        self.hidden_dim = config.hidden_dim
        self.head_hidden_dim = config.head_hidden_dim
        
        # Cross-attention: query from target cell, key/value from observed cells
        self.query_proj = nn.Linear(config.hidden_dim, config.head_hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.head_hidden_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.head_hidden_dim)
        
        # Output projection after attention
        self.out_proj = nn.Linear(config.head_hidden_dim, config.head_hidden_dim)
        
        # Final prediction MLP
        self.predictor = nn.Sequential(
            nn.Linear(config.head_hidden_dim + config.hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = config.head_hidden_dim ** 0.5
    
    def forward(
        self,
        token_repr: torch.Tensor,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict values using cross-attention to observed cells."""
        B, max_rows, max_cols, hidden_dim = token_repr.shape
        
        # Get observation mask from tokens
        is_observed = tokens[..., IDX_OBSERVED]  # [B, max_rows, max_cols]
        
        # Reshape for row-wise processing
        # [B, max_rows, max_cols, hidden_dim] -> [B * max_rows, max_cols, hidden_dim]
        token_repr_flat = token_repr.view(B * max_rows, max_cols, hidden_dim)
        is_observed_flat = is_observed.view(B * max_rows, max_cols)
        
        # Expand col_mask: [B, max_cols] -> [B * max_rows, max_cols]
        col_mask_flat = col_mask.unsqueeze(1).expand(B, max_rows, max_cols)
        col_mask_flat = col_mask_flat.reshape(B * max_rows, max_cols)
        
        # Project to Q, K, V
        Q = self.query_proj(token_repr_flat)  # [B*max_rows, max_cols, head_hidden_dim]
        K = self.key_proj(token_repr_flat)
        V = self.value_proj(token_repr_flat)
        
        # Attention scores: [B*max_rows, max_cols, max_cols]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Mask: only attend to OBSERVED and VALID cells
        attn_mask = is_observed_flat.bool() & col_mask_flat.bool()
        attn_mask = attn_mask.unsqueeze(1)  # [B*max_rows, 1, max_cols]
        
        # Apply mask
        mask_value = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~attn_mask, mask_value)
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)  # [B*max_rows, max_cols, head_hidden_dim]
        context = self.out_proj(context)
        
        # Concatenate with original representation and predict
        combined = torch.cat([context, token_repr_flat], dim=-1)
        predictions_flat = self.predictor(combined).squeeze(-1)  # [B*max_rows, max_cols]
        
        # Reshape back
        predictions = predictions_flat.view(B, max_rows, max_cols)
        
        return predictions


# =============================================================================
# MNAR Self-Censoring Head
# =============================================================================

class MNARSelfCensoringHead(BaseReconstructionHead):
    """
    MNAR Self-Censoring reconstruction head.
    
    Architecture:
        Predicts value AND estimates a "censoring score" indicating how likely
        this value is to be missing due to its own magnitude.
    
    Inductive bias:
        High or low values are systematically missing (e.g., high income,
        extreme health measures). The head learns to recognize distributional
        asymmetry and adjust predictions accordingly.
    
    Expected behavior:
        - Under MCAR/MAR: Censoring score is uninformative
        - Under MNAR self-censoring: Censoring score correlates with missingness
    """
    
    def __init__(self, config: ReconstructionConfig):
        super().__init__(config)
        
        # Value predictor
        self.value_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )
        
        # Censoring score predictor (auxiliary output)
        # Predicts log-odds of value being "extreme" (censored)
        self.censoring_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )
        
        # Bias adjustment based on censoring score
        self.bias_adjustment = nn.Sequential(
            nn.Linear(1, config.head_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.head_hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        token_repr: torch.Tensor,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict values with self-censoring adjustment."""
        # Base prediction
        base_pred = self.value_predictor(token_repr).squeeze(-1)  # [B, max_rows, max_cols]
        
        # Censoring score
        censoring_score = self.censoring_predictor(token_repr)  # [B, max_rows, max_cols, 1]
        
        # Adjust prediction based on censoring (missing values might be extreme)
        bias = self.bias_adjustment(censoring_score).squeeze(-1)  # [B, max_rows, max_cols]
        
        # Final prediction: base + learned bias for potential censoring
        predictions = base_pred + bias
        
        return predictions
    
    def get_censoring_scores(self, token_repr: torch.Tensor) -> torch.Tensor:
        """Get censoring scores for analysis/interpretability."""
        return torch.sigmoid(self.censoring_predictor(token_repr).squeeze(-1))


# =============================================================================
# MNAR Threshold Head
# =============================================================================

class MNARThresholdHead(BaseReconstructionHead):
    """
    MNAR Threshold reconstruction head.
    
    Architecture:
        Learns a soft threshold function; values beyond the threshold are
        predicted differently (acknowledging they're from a truncated distribution).
    
    Inductive bias:
        Values above or below some threshold are systematically missing
        (e.g., lab values below detection limit, income above reporting threshold).
    
    Expected behavior:
        - Learns to identify truncation patterns
        - Predicts boundary values for threshold-missing cells
    """
    
    def __init__(self, config: ReconstructionConfig):
        super().__init__(config)
        
        # Threshold estimator (per-column, learned from context)
        self.threshold_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 2),  # [lower_threshold, upper_threshold]
        )
        
        # Value predictor for "normal" range
        self.normal_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )
        
        # Value predictor for "truncated" range
        self.truncated_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim + 2, config.head_hidden_dim),  # +2 for thresholds
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )
        
        # Soft selection between normal and truncated prediction
        self.selection_gate = nn.Sequential(
            nn.Linear(config.hidden_dim, config.head_hidden_dim),
            nn.GELU(),
            nn.Linear(config.head_hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        token_repr: torch.Tensor,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict values with threshold-aware adjustment."""
        # Estimate thresholds from context
        thresholds = self.threshold_estimator(token_repr)  # [B, max_rows, max_cols, 2]
        
        # Normal prediction
        normal_pred = self.normal_predictor(token_repr).squeeze(-1)  # [B, max_rows, max_cols]
        
        # Truncated prediction (includes threshold info)
        truncated_input = torch.cat([token_repr, thresholds], dim=-1)
        truncated_pred = self.truncated_predictor(truncated_input).squeeze(-1)
        
        # Gate between normal and truncated
        gate = self.selection_gate(token_repr).squeeze(-1)  # [B, max_rows, max_cols]
        
        # Blend predictions
        predictions = gate * truncated_pred + (1 - gate) * normal_pred
        
        return predictions
    
    def get_thresholds(self, token_repr: torch.Tensor) -> torch.Tensor:
        """Get estimated thresholds for analysis."""
        return self.threshold_estimator(token_repr)


# =============================================================================
# MNAR Latent Head
# =============================================================================

class MNARLatentHead(BaseReconstructionHead):
    """
    MNAR Latent reconstruction head.
    
    Architecture:
        Infers a latent variable from the observed pattern, then conditions
        predictions on this latent. The latent captures unobserved confounders
        that drive both values and missingness.
    
    Inductive bias:
        There's an unobserved factor (e.g., "health status", "engagement level")
        that influences both which values are observed and what those values would be.
    
    Expected behavior:
        - Under MCAR/MAR: Latent provides minimal help
        - Under MNAR-latent: Latent captures confounding structure
    """
    
    def __init__(self, config: ReconstructionConfig, latent_dim: int = 16):
        super().__init__(config)
        
        self.latent_dim = latent_dim
        
        # Latent encoder: pool row representation into latent vector
        self.latent_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, latent_dim * 2),  # mean and log_var
        )
        
        # Value predictor conditioned on latent
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim + latent_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )
    
    def forward(
        self,
        token_repr: torch.Tensor,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict values conditioned on inferred latent."""
        B, max_rows, max_cols, hidden_dim = token_repr.shape
        
        # Get observation mask
        is_observed = tokens[..., IDX_OBSERVED]  # [B, max_rows, max_cols]
        
        # Pool observed tokens within each row to infer latent
        # Mask: [B, max_rows, max_cols] -> [B, max_rows, max_cols, 1]
        obs_mask = is_observed.unsqueeze(-1) * col_mask.unsqueeze(1).unsqueeze(-1)
        
        # Masked mean of token representations
        masked_repr = token_repr * obs_mask
        sum_repr = masked_repr.sum(dim=2)  # [B, max_rows, hidden_dim]
        count = obs_mask.sum(dim=2).clamp(min=1.0)  # [B, max_rows, 1]
        row_summary = sum_repr / count  # [B, max_rows, hidden_dim]
        
        # Encode to latent (per row)
        latent_params = self.latent_encoder(row_summary)  # [B, max_rows, latent_dim * 2]
        latent_mean = latent_params[..., :self.latent_dim]
        latent_logvar = latent_params[..., self.latent_dim:]
        
        # Reparameterization (only during training)
        if self.training:
            std = torch.exp(0.5 * latent_logvar)
            eps = torch.randn_like(std)
            latent = latent_mean + eps * std
        else:
            latent = latent_mean
        
        # Expand to all columns: [B, max_rows, latent_dim] -> [B, max_rows, max_cols, latent_dim]
        latent_expanded = latent.unsqueeze(2).expand(B, max_rows, max_cols, self.latent_dim)
        
        # Concatenate with token representation
        conditioned = torch.cat([token_repr, latent_expanded], dim=-1)
        
        # Predict
        predictions = self.predictor(conditioned).squeeze(-1)  # [B, max_rows, max_cols]
        
        return predictions
    
    def get_latent(
        self,
        token_repr: torch.Tensor,
        tokens: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get latent mean and logvar for analysis."""
        B, max_rows, max_cols, hidden_dim = token_repr.shape
        
        is_observed = tokens[..., IDX_OBSERVED]
        obs_mask = is_observed.unsqueeze(-1) * col_mask.unsqueeze(1).unsqueeze(-1)
        
        masked_repr = token_repr * obs_mask
        sum_repr = masked_repr.sum(dim=2)
        count = obs_mask.sum(dim=2).clamp(min=1.0)
        row_summary = sum_repr / count
        
        latent_params = self.latent_encoder(row_summary)
        latent_mean = latent_params[..., :self.latent_dim]
        latent_logvar = latent_params[..., self.latent_dim:]
        
        return latent_mean, latent_logvar


# =============================================================================
# Head Registry
# =============================================================================

HEAD_REGISTRY = {
    "mcar": MCARHead,
    "mar": MARHead,
    "self_censoring": MNARSelfCensoringHead,
    "threshold": MNARThresholdHead,
    "latent": MNARLatentHead,
}


def create_head(name: str, config: ReconstructionConfig, **kwargs) -> BaseReconstructionHead:
    """
    Factory function to create a reconstruction head by name.
    
    Args:
        name: Head name (from HEAD_REGISTRY).
        config: ReconstructionConfig with architecture parameters.
        **kwargs: Additional arguments for specific head types.
    
    Returns:
        Configured reconstruction head instance.
    
    Raises:
        ValueError: If name is not in HEAD_REGISTRY.
    """
    if name not in HEAD_REGISTRY:
        raise ValueError(
            f"Unknown head type: {name}. "
            f"Available: {list(HEAD_REGISTRY.keys())}"
        )
    
    return HEAD_REGISTRY[name](config, **kwargs)