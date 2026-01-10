"""
lacuna.models.reconstruction.heads

Concrete reconstruction head implementations.

CRITICAL FIX (2026-01-10):
--------------------------
The original MARHead used cross-attention over token REPRESENTATIONS (the output
of the encoder's transformer). This doesn't work for MAR/MNAR discrimination
because by that point the raw values have been transformed into abstract features.

The fix: MARHead now uses the RAW OBSERVED VALUES from `tokens[..., IDX_VALUE]`
as the attention values. This means:
    - Query: from target cell's representation (what are we predicting?)
    - Key: from observed cells' representations (which cells to attend to?)
    - Value: the ACTUAL OBSERVED VALUES (what to use for prediction!)

Under MAR, the missing values ARE predictable from observed values in other
columns. Under MNAR, they are NOT. This is the discriminative signal.

Head Types:
    MCARHead: Simple MLP, no cross-column structure exploited.
    MARHead: Cross-attention using RAW observed values (FIXED).
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
# MAR Head: Cross-Attention to RAW Observed Values (FIXED)
# =============================================================================

class MARHead(BaseReconstructionHead):
    """
    MAR reconstruction head using RAW observed values.
    
    CRITICAL FIX:
    -------------
    The original implementation used cross-attention over token representations.
    This doesn't work because by that point the raw values are transformed.
    
    The fix: Use the ACTUAL OBSERVED VALUES from tokens[..., IDX_VALUE] as the
    attention values. The attention mechanism learns WHICH observed columns
    to attend to, and then uses their RAW VALUES to predict the missing value.
    
    Architecture:
        1. Query projection from target cell's representation
        2. Key projection from all cells' representations  
        3. Attention mask: only attend to OBSERVED cells
        4. Value: the RAW OBSERVED VALUES (not representations!)
        5. Weighted sum of raw values + MLP to predict
    
    Inductive bias:
        Under MAR, missing values can be predicted as a (learned) weighted
        combination of observed values in the same row. This is exactly what
        imputation methods assume and what makes MAR "ignorable".
    
    Expected behavior:
        - Under MCAR: Moderate error (random missingness, weak correlations)
        - Under MAR: LOW error (observed values predict missing ones!)
        - Under MNAR: Higher error (the missing value itself matters)
    """
    
    def __init__(self, config: ReconstructionConfig):
        super().__init__(config)
        
        self.hidden_dim = config.hidden_dim
        self.head_hidden_dim = config.head_hidden_dim
        
        # Query and Key projections from token representations
        # These learn WHICH cells to attend to
        self.query_proj = nn.Linear(config.hidden_dim, config.head_hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.head_hidden_dim)
        
        # Value projection: from raw observed value (scalar) to hidden dim
        # This learns how to weight the raw values
        self.value_proj = nn.Linear(1, config.head_hidden_dim)
        
        # Output projection after attention
        self.out_proj = nn.Linear(config.head_hidden_dim, config.head_hidden_dim)
        
        # Final prediction MLP
        # Input: attention output + original representation
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
        """
        Predict values using cross-attention to RAW observed values.
        
        The key insight: we use token representations to compute attention
        weights (which columns are relevant?), but we apply those weights
        to the RAW OBSERVED VALUES (what to use for prediction).
        
        Under MAR, this should work well because observed values in other
        columns actually predict the missing value.
        Under MNAR, this should fail because the missing value's own
        magnitude determines missingness, not other columns.
        """
        B, max_rows, max_cols, hidden_dim = token_repr.shape
        
        # Extract raw values and observation mask from tokens
        raw_values = tokens[..., IDX_VALUE]      # [B, max_rows, max_cols]
        is_observed = tokens[..., IDX_OBSERVED]  # [B, max_rows, max_cols]
        
        # Reshape for row-wise processing
        # [B, max_rows, max_cols, hidden_dim] -> [B * max_rows, max_cols, hidden_dim]
        token_repr_flat = token_repr.view(B * max_rows, max_cols, hidden_dim)
        raw_values_flat = raw_values.view(B * max_rows, max_cols, 1)  # Add dim for projection
        is_observed_flat = is_observed.view(B * max_rows, max_cols)
        
        # Expand col_mask: [B, max_cols] -> [B * max_rows, max_cols]
        col_mask_flat = col_mask.unsqueeze(1).expand(B, max_rows, max_cols)
        col_mask_flat = col_mask_flat.reshape(B * max_rows, max_cols)
        
        # Project query and key from token representations
        # These determine WHICH observed cells to attend to
        Q = self.query_proj(token_repr_flat)  # [B*max_rows, max_cols, head_hidden_dim]
        K = self.key_proj(token_repr_flat)    # [B*max_rows, max_cols, head_hidden_dim]
        
        # Project raw values
        # This transforms the scalar raw values into a vector for weighted sum
        V = self.value_proj(raw_values_flat)  # [B*max_rows, max_cols, head_hidden_dim]
        
        # Attention scores: [B*max_rows, max_cols, max_cols]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Mask: only attend to OBSERVED and VALID cells
        attn_mask = is_observed_flat.bool() & col_mask_flat.bool()
        attn_mask = attn_mask.unsqueeze(1)  # [B*max_rows, 1, max_cols]
        
        # Apply mask (set non-observed to -inf before softmax)
        mask_value = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~attn_mask, mask_value)
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Handle rows with NO observed values (all -inf -> NaN after softmax)
        # Replace NaN with zeros (no information from attention)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Apply attention to VALUES (not representations!)
        # This computes a weighted sum of raw values from observed columns
        context = torch.matmul(attn_weights, V)  # [B*max_rows, max_cols, head_hidden_dim]
        context = self.out_proj(context)
        
        # Concatenate attention output with original representation and predict
        combined = torch.cat([context, token_repr_flat], dim=-1)
        predictions_flat = self.predictor(combined).squeeze(-1)  # [B*max_rows, max_cols]
        
        # Reshape back to [B, max_rows, max_cols]
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
        - Under MCAR: Moderate error (no systematic censoring)
        - Under MAR: Higher error (censoring adjustment not helpful)
        - Under MNAR self-censoring: Lower error (matches the pattern)
    """
    
    def __init__(self, config: ReconstructionConfig):
        super().__init__(config)
        
        # Main value prediction
        self.value_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )
        
        # Censoring direction estimator: learns whether high or low values are censored
        # Output is a scalar that shifts the prediction
        self.censoring_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
            nn.Tanh(),  # Bounded shift in [-1, 1]
        )
        
        # Learnable scale for censoring adjustment
        self.censoring_scale = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        token_repr: torch.Tensor,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict values with censoring adjustment."""
        # Base value prediction
        base_pred = self.value_predictor(token_repr).squeeze(-1)  # [B, max_rows, max_cols]
        
        # Estimate censoring direction and magnitude
        censoring_adj = self.censoring_estimator(token_repr).squeeze(-1)
        censoring_adj = censoring_adj * self.censoring_scale
        
        # Get observation indicator
        is_observed = tokens[..., IDX_OBSERVED]  # [B, max_rows, max_cols]
        
        # Apply censoring adjustment only to MISSING values
        # (observed values should use base prediction)
        predictions = base_pred + censoring_adj * (1.0 - is_observed)
        
        return predictions


# =============================================================================
# MNAR Threshold Head
# =============================================================================

class MNARThresholdHead(BaseReconstructionHead):
    """
    MNAR Threshold reconstruction head.
    
    Architecture:
        Learns soft thresholds for each column and adjusts predictions
        based on which side of the threshold values are likely to be missing.
    
    Inductive bias:
        Values above or below certain thresholds are systematically missing
        (e.g., income below poverty line, tests above detection limit).
    
    Expected behavior:
        - Under MCAR: Moderate error (thresholds don't help)
        - Under MAR: Higher error (threshold adjustment not helpful)
        - Under MNAR threshold: Lower error (matches the pattern)
    """
    
    def __init__(self, config: ReconstructionConfig):
        super().__init__(config)
        
        # Main value prediction
        self.value_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )
        
        # Threshold estimator: estimates soft threshold for this position
        self.threshold_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 2),  # (threshold, direction)
        )
    
    def forward(
        self,
        token_repr: torch.Tensor,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict values with threshold adjustment."""
        # Base value prediction
        base_pred = self.value_predictor(token_repr).squeeze(-1)
        
        # Estimate threshold parameters
        thresh_params = self.threshold_estimator(token_repr)  # [B, max_rows, max_cols, 2]
        threshold = thresh_params[..., 0]  # Threshold value
        direction = torch.sigmoid(thresh_params[..., 1])  # Direction (0=below, 1=above)
        
        # Get observation indicator
        is_observed = tokens[..., IDX_OBSERVED]
        
        # Adjust predictions for missing values based on threshold
        # If direction > 0.5: values above threshold are missing -> predict above threshold
        # If direction < 0.5: values below threshold are missing -> predict below threshold
        adjustment = (direction - 0.5) * 2.0  # Scale to [-1, 1]
        
        # Apply adjustment only to missing values
        predictions = base_pred + adjustment * (1.0 - is_observed)
        
        return predictions


# =============================================================================
# MNAR Latent Head
# =============================================================================

class MNARLatentHead(BaseReconstructionHead):
    """
    MNAR Latent reconstruction head.
    
    Architecture:
        Infers a latent variable that explains both the value and the
        missingness, then uses that to adjust predictions.
    
    Inductive bias:
        There's an unobserved confounder that causes both the value and
        its missingness (e.g., underlying health status affects both
        test values and willingness to get tested).
    
    Expected behavior:
        - Under MCAR: Moderate error (no confounding)
        - Under MAR: Higher error (confounder adjustment not helpful)
        - Under MNAR latent: Lower error (captures confounding)
    """
    
    def __init__(self, config: ReconstructionConfig):
        super().__init__(config)
        
        self.latent_dim = config.head_hidden_dim // 2
        
        # Latent variable encoder
        self.latent_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, self.latent_dim * 2),  # Mean and log-var
        )
        
        # Value prediction conditioned on latent
        self.value_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim + self.latent_dim, config.head_hidden_dim),
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
        """Predict values via latent variable inference."""
        # Encode to latent distribution
        latent_params = self.latent_encoder(token_repr)
        mu = latent_params[..., :self.latent_dim]
        log_var = latent_params[..., self.latent_dim:]
        
        # Reparameterization trick (sample if training, use mean if eval)
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        # Decode to value prediction
        decoder_input = torch.cat([token_repr, z], dim=-1)
        predictions = self.value_decoder(decoder_input).squeeze(-1)
        
        return predictions


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


def create_head(name: str, config: ReconstructionConfig) -> BaseReconstructionHead:
    """
    Create a reconstruction head by name.
    
    Args:
        name: Head type name (one of HEAD_REGISTRY keys)
        config: ReconstructionConfig for the head
    
    Returns:
        Instantiated reconstruction head
    
    Raises:
        KeyError: If name is not in HEAD_REGISTRY
    """
    if name not in HEAD_REGISTRY:
        raise KeyError(
            f"Unknown head type: {name}. "
            f"Available types: {list(HEAD_REGISTRY.keys())}"
        )
    
    return HEAD_REGISTRY[name](config)