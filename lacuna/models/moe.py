"""
lacuna.models.moe

Mixture of Experts for mechanism classification.

Enhanced with explicit missingness pattern features for better MAR/MCAR discrimination.

The MoE receives three types of input signals:
1. Evidence vector: Learned representation from transformer encoder
2. Reconstruction errors: How well each head predicts missing values
3. Missingness features: Explicit statistical features of missingness patterns

The missingness features are critical for MCAR vs MAR discrimination:
- MCAR: Low variance in missing rates, low cross-column correlation
- MAR: High point-biserial correlations, structured missingness patterns
- MNAR: Distributional distortions (skewness, kurtosis)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field

from lacuna.core.types import MoEOutput, MCAR, MAR, MNAR


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts layer."""
    
    # Input dimensions
    evidence_dim: int = 64
    hidden_dim: int = 128
    
    # Expert structure
    n_mechanism_classes: int = 3
    mnar_variants: List[str] = None
    
    # Gating architecture
    gate_hidden_dim: int = 64
    gate_n_layers: int = 2
    gate_dropout: float = 0.1
    
    # Gating mode
    gating_level: str = "dataset"
    use_reconstruction_errors: bool = True
    n_reconstruction_heads: int = 5
    
    # Missingness pattern features (NEW)
    use_missingness_features: bool = True
    n_missingness_features: int = 16  # From MissingnessFeatureConfig.n_features
    
    # Expert heads
    use_expert_heads: bool = False
    expert_hidden_dim: int = 32
    
    # Calibration
    temperature: float = 1.0
    learn_temperature: bool = False
    
    # Class aggregation
    class_aggregation: str = "mean"
    
    # Regularization
    load_balance_weight: float = 0.0
    entropy_weight: float = 0.0
    
    def __post_init__(self):
        if self.mnar_variants is None:
            self.mnar_variants = ["self_censoring", "threshold", "latent"]
        
        if self.gating_level not in ("dataset", "row"):
            raise ValueError(f"gating_level must be 'dataset' or 'row', got {self.gating_level}")
        
        if self.class_aggregation not in ("mean", "sum", "learned"):
            raise ValueError(f"class_aggregation must be 'mean', 'sum', or 'learned'")
    
    @property
    def n_experts(self) -> int:
        """Total number of experts."""
        return 2 + len(self.mnar_variants)
    
    @property
    def expert_names(self) -> List[str]:
        """Ordered list of expert names."""
        return ["mcar", "mar"] + list(self.mnar_variants)
    
    @property
    def gate_input_dim(self) -> int:
        """Dimension of input to gating network."""
        base_dim = self.evidence_dim if self.gating_level == "dataset" else self.hidden_dim
        
        if self.use_reconstruction_errors:
            base_dim += self.n_reconstruction_heads
        
        if self.use_missingness_features:
            base_dim += self.n_missingness_features
        
        return base_dim


# =============================================================================
# Gating Network
# =============================================================================

class GatingNetwork(nn.Module):
    """
    Gating network that produces expert mixture weights.
    
    Takes evidence vector, reconstruction errors, and missingness features
    to produce logits for each expert.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        
        self.config = config
        
        # Build MLP layers
        layers = []
        in_dim = config.gate_input_dim
        
        for i in range(config.gate_n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, config.gate_hidden_dim),
                nn.LayerNorm(config.gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.gate_dropout),
            ])
            in_dim = config.gate_hidden_dim
        
        layers.append(nn.Linear(in_dim, config.n_experts))
        
        self.mlp = nn.Sequential(*layers)
        
        # Temperature parameter
        if config.learn_temperature:
            init_log_temp = torch.log(torch.tensor(config.temperature))
            self.log_temperature = nn.Parameter(init_log_temp)
        else:
            self.register_buffer("log_temperature", torch.log(torch.tensor(config.temperature)))
    
    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature value."""
        return torch.exp(self.log_temperature)
    
    def forward(
        self,
        evidence: torch.Tensor,
        reconstruction_errors: Optional[torch.Tensor] = None,
        missingness_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gating logits and probabilities.
        
        Args:
            evidence: Evidence vector from encoder. [B, evidence_dim]
            reconstruction_errors: Reconstruction errors per head. [B, n_heads]
            missingness_features: Explicit missingness pattern features. [B, n_miss_features]
        
        Returns:
            logits: Raw gating logits. [B, n_experts]
            probs: Gating probabilities. [B, n_experts]
        """
        # Build gate input by concatenating available features
        gate_inputs = [evidence]
        
        if self.config.use_reconstruction_errors and reconstruction_errors is not None:
            gate_inputs.append(reconstruction_errors)
        
        if self.config.use_missingness_features and missingness_features is not None:
            gate_inputs.append(missingness_features)
        
        gate_input = torch.cat(gate_inputs, dim=-1)
        
        # Compute logits
        logits = self.mlp(gate_input)
        
        # Temperature-scaled softmax
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        return logits, probs


# =============================================================================
# Expert Heads
# =============================================================================

class ExpertHead(nn.Module):
    """Lightweight expert head for mechanism-specific refinement."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ExpertHeads(nn.Module):
    """Container for all expert heads."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        
        self.config = config
        
        input_dim = config.evidence_dim if config.gating_level == "dataset" else config.hidden_dim
        
        self.experts = nn.ModuleDict()
        for name in config.expert_names:
            self.experts[name] = ExpertHead(
                input_dim=input_dim,
                hidden_dim=config.expert_hidden_dim,
                dropout=config.gate_dropout,
            )
    
    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        adjustments = []
        for name in self.config.expert_names:
            adj = self.experts[name](evidence)
            adjustments.append(adj)
        
        return torch.stack(adjustments, dim=-1)


# =============================================================================
# Main MoE Layer
# =============================================================================

class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer for mechanism classification.
    
    Now receives three input signals:
    1. evidence: Learned representation from encoder
    2. reconstruction_errors: How well each head predicts missing values
    3. missingness_features: Explicit statistical features of missingness patterns
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        
        self.config = config
        
        # Gating network
        self.gating = GatingNetwork(config)
        
        # Expert heads (optional)
        if config.use_expert_heads:
            self.experts = ExpertHeads(config)
        else:
            self.experts = None
        
        # Mapping from expert index to mechanism class
        self.register_buffer(
            "expert_to_class",
            torch.tensor([MCAR, MAR] + [MNAR] * len(config.mnar_variants))
        )
        
        # Precompute experts per class
        experts_per_class = torch.zeros(3)
        for expert_idx in range(config.n_experts):
            class_idx = self.expert_to_class[expert_idx].item()
            experts_per_class[class_idx] += 1
        self.register_buffer("experts_per_class", experts_per_class)
        
        # Learnable class bias
        if config.class_aggregation == "learned":
            n_mnar = len(config.mnar_variants)
            init_bias = torch.tensor([
                0.0,
                0.0,
                -torch.log(torch.tensor(float(n_mnar))).item(),
            ])
            self.class_bias = nn.Parameter(init_bias)
        else:
            self.class_bias = None
    
    def forward(
        self,
        evidence: torch.Tensor,
        reconstruction_errors: Optional[torch.Tensor] = None,
        missingness_features: Optional[torch.Tensor] = None,
        row_mask: Optional[torch.Tensor] = None,
    ) -> MoEOutput:
        """
        Compute mechanism posterior via gated experts.
        
        Args:
            evidence: Evidence vector from encoder. [B, evidence_dim]
            reconstruction_errors: Reconstruction errors per head. [B, n_heads]
            missingness_features: Explicit missingness pattern features. [B, n_miss_features]
            row_mask: Optional mask for valid rows (row-level gating).
        
        Returns:
            MoEOutput with gating probabilities and auxiliary info.
        """
        # Compute gating probabilities
        gate_logits, gate_probs = self.gating(
            evidence,
            reconstruction_errors=reconstruction_errors,
            missingness_features=missingness_features,
        )
        
        # Apply expert heads if configured
        expert_adjustments = None
        if self.experts is not None:
            expert_adjustments = self.experts(evidence)
            combined_logits = gate_logits + expert_adjustments
            gate_probs = F.softmax(combined_logits / self.gating.temperature, dim=-1)
        
        return MoEOutput(
            gate_logits=gate_logits,
            gate_probs=gate_probs,
            expert_outputs=[expert_adjustments] if expert_adjustments is not None else None,
            combined_output=gate_logits if expert_adjustments is None else (gate_logits + expert_adjustments),
        )
    
    def _compute_load_balance_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to encourage uniform expert usage."""
        if self.config.load_balance_weight <= 0:
            return torch.tensor(0.0, device=gate_probs.device)
        
        # Average probability per expert across batch
        avg_probs = gate_probs.mean(dim=0)
        
        # Target is uniform distribution
        target = torch.ones_like(avg_probs) / self.config.n_experts
        
        # KL divergence from uniform
        loss = F.kl_div(avg_probs.log(), target, reduction='sum')
        
        return self.config.load_balance_weight * loss
    
    def _compute_entropy(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of gating distribution."""
        log_probs = torch.log(gate_probs.clamp(min=1e-8))
        entropy = -(gate_probs * log_probs).sum(dim=-1)
        return entropy.mean()
    
    def get_class_posterior(self, output: MoEOutput) -> torch.Tensor:
        """
        Aggregate expert probabilities to class probabilities.
        
        Uses balanced aggregation by default to avoid MNAR bias.
        """
        gate_probs = output.gate_probs
        B = gate_probs.shape[0]
        device = gate_probs.device
        
        p_class = torch.zeros(B, 3, device=device)
        
        for expert_idx in range(self.config.n_experts):
            class_idx = self.expert_to_class[expert_idx].item()
            p_class[:, class_idx] += gate_probs[:, expert_idx]
        
        # Apply aggregation method
        if self.config.class_aggregation == "mean":
            # Normalize by number of experts per class
            p_class = p_class / self.experts_per_class.unsqueeze(0).clamp(min=1)
            p_class = p_class / p_class.sum(dim=-1, keepdim=True)
        
        elif self.config.class_aggregation == "learned":
            # Apply learned bias correction
            if self.class_bias is not None:
                log_p = torch.log(p_class.clamp(min=1e-8)) + self.class_bias.unsqueeze(0)
                p_class = F.softmax(log_p, dim=-1)
        
        # "sum" mode: just normalize to sum to 1
        else:
            p_class = p_class / p_class.sum(dim=-1, keepdim=True)
        
        return p_class
    
    def get_mnar_variant_posterior(self, output: MoEOutput) -> torch.Tensor:
        """Get posterior over MNAR variants (conditional on MNAR)."""
        gate_probs = output.gate_probs
        
        # Extract MNAR expert probabilities (indices 2+)
        mnar_probs = gate_probs[:, 2:]
        
        # Normalize to get conditional distribution
        mnar_total = mnar_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        p_variant = mnar_probs / mnar_total
        
        return p_variant
    
    def get_auxiliary_losses(self, output: MoEOutput) -> Dict[str, torch.Tensor]:
        """Get auxiliary losses for training."""
        losses = {}
        
        if self.config.load_balance_weight > 0:
            losses["load_balance"] = self._compute_load_balance_loss(output.gate_probs)
        
        if self.config.entropy_weight > 0:
            losses["entropy"] = self.config.entropy_weight * self._compute_entropy(output.gate_probs)
        
        return losses


# =============================================================================
# Row-Level Aggregation
# =============================================================================

class RowToDatasetAggregator(nn.Module):
    """Aggregate row-level gating decisions to dataset-level."""
    
    def __init__(self, hidden_dim: int, n_experts: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        row_logits: torch.Tensor,
        row_repr: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate row-level logits to dataset-level.
        
        Args:
            row_logits: [B, max_rows, n_experts]
            row_repr: [B, max_rows, hidden_dim]
            row_mask: [B, max_rows]
        
        Returns:
            dataset_logits: [B, n_experts]
        """
        # Compute attention weights
        attn_scores = self.attention(row_repr).squeeze(-1)
        
        # Mask invalid rows
        mask_value = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~row_mask, mask_value)
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted average
        dataset_logits = torch.einsum('br,bre->be', attn_weights, row_logits)
        
        return dataset_logits


# =============================================================================
# Factory Function
# =============================================================================

def create_moe(
    evidence_dim: int = 64,
    hidden_dim: int = 128,
    mnar_variants: Optional[List[str]] = None,
    gate_hidden_dim: int = 64,
    gate_n_layers: int = 2,
    gating_level: str = "dataset",
    use_reconstruction_errors: bool = True,
    n_reconstruction_heads: int = 5,
    use_missingness_features: bool = True,
    n_missingness_features: int = 16,
    use_expert_heads: bool = False,
    temperature: float = 1.0,
    learn_temperature: bool = False,
    class_aggregation: str = "mean",
    load_balance_weight: float = 0.0,
    entropy_weight: float = 0.0,
    dropout: float = 0.1,
) -> MixtureOfExperts:
    """Factory function to create MixtureOfExperts."""
    
    if mnar_variants is None:
        mnar_variants = ["self_censoring", "threshold", "latent"]
    
    config = MoEConfig(
        evidence_dim=evidence_dim,
        hidden_dim=hidden_dim,
        mnar_variants=mnar_variants,
        gate_hidden_dim=gate_hidden_dim,
        gate_n_layers=gate_n_layers,
        gate_dropout=dropout,
        gating_level=gating_level,
        use_reconstruction_errors=use_reconstruction_errors,
        n_reconstruction_heads=n_reconstruction_heads,
        use_missingness_features=use_missingness_features,
        n_missingness_features=n_missingness_features,
        use_expert_heads=use_expert_heads,
        temperature=temperature,
        learn_temperature=learn_temperature,
        class_aggregation=class_aggregation,
        load_balance_weight=load_balance_weight,
        entropy_weight=entropy_weight,
    )
    
    return MixtureOfExperts(config)