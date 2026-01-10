"""
lacuna.models.moe

Mixture of Experts (MoE) layer for mechanism classification.

Architecture Overview:
    Instead of a simple classifier, Lacuna uses a gating network that routes
    inputs to specialized expert heads. Each expert corresponds to a mechanism
    class or MNAR variant.
    
    The gating network learns to recognize which "world" (mechanism) generated
    the data, outputting mixture weights that become the mechanism posterior.

Gating Granularities:
    The MoE can operate at different levels depending on the analysis goal:
    
    1. Dataset-level gating (default):
       - One mechanism dominates the entire dataset
       - Gate takes dataset evidence vector, outputs [B, n_experts]
       - Most common use case for mechanism classification
    
    2. Row-level gating:
       - Mechanism mixture varies by subpopulation/row
       - Gate takes row representations, outputs [B, max_rows, n_experts]
       - Useful for detecting heterogeneous missingness patterns
    
    3. Feature-block gating (future):
       - Different mechanisms for different variable groups
       - Not yet implemented

Expert Structure:
    Experts are lightweight heads that refine the gating decision:
    
    - MCAR expert: Confirms random scatter pattern
    - MAR expert: Confirms cross-column predictability
    - MNAR experts (per variant): Confirm specific MNAR signatures
    
    The final output combines gating probabilities with expert outputs:
        p(mechanism) = softmax(gate_logits + expert_adjustments)
    
    Or in "pure gating" mode:
        p(mechanism) = softmax(gate_logits)

Integration with Reconstruction:
    The reconstruction errors from each head can inform the gating:
    
    - Low MAR reconstruction error → increase MAR gate weight
    - High MNAR reconstruction error → decrease MNAR gate weight
    
    This creates a feedback loop where reconstruction quality guides
    mechanism classification.

Design Decisions:
    1. Gating network is an MLP on the evidence/row representation
    2. Experts are optional refinement heads (can be identity)
    3. Reconstruction errors can be concatenated to gate input
    4. Temperature scaling for calibrated probabilities
    5. Load balancing loss to prevent expert collapse (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field

from lacuna.core.types import MoEOutput, MCAR, MAR, MNAR


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts layer."""
    
    # Input dimensions
    evidence_dim: int = 64         # Dimension of dataset evidence vector
    hidden_dim: int = 128          # Dimension of row representations (for row-level gating)
    
    # Expert structure
    n_mechanism_classes: int = 3   # MCAR, MAR, MNAR (base classes)
    mnar_variants: List[str] = None  # MNAR variant names
    
    # Gating architecture
    gate_hidden_dim: int = 64      # Hidden dimension in gating network
    gate_n_layers: int = 2         # Depth of gating network
    gate_dropout: float = 0.1      # Dropout in gating network
    
    # Gating mode
    gating_level: str = "dataset"  # "dataset" or "row"
    use_reconstruction_errors: bool = True  # Include recon errors in gate input
    n_reconstruction_heads: int = 5  # Number of reconstruction heads (if used)
    
    # Expert heads
    use_expert_heads: bool = False  # If False, pure gating (experts are identity)
    expert_hidden_dim: int = 32    # Hidden dimension in expert heads
    
    # Calibration
    temperature: float = 1.0       # Temperature for softmax calibration
    learn_temperature: bool = False  # Learn temperature as parameter
    
    # Regularization
    load_balance_weight: float = 0.0  # Weight for load balancing loss (0 = disabled)
    entropy_weight: float = 0.0    # Weight for entropy regularization (0 = disabled)
    
    def __post_init__(self):
        if self.mnar_variants is None:
            self.mnar_variants = ["self_censoring", "threshold", "latent"]
        
        if self.gating_level not in ("dataset", "row"):
            raise ValueError(f"gating_level must be 'dataset' or 'row', got {self.gating_level}")
    
    @property
    def n_experts(self) -> int:
        """Total number of experts: MCAR + MAR + MNAR variants."""
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
            return base_dim + self.n_reconstruction_heads
        return base_dim


# =============================================================================
# Gating Network
# =============================================================================

class GatingNetwork(nn.Module):
    """
    Gating network that produces expert mixture weights.
    
    Takes evidence (dataset-level) or row representations (row-level) and
    outputs logits for each expert. Optionally incorporates reconstruction
    errors as additional input features.
    
    Architecture:
        [evidence_dim (+ n_recon_heads)] -> MLP -> [n_experts]
    
    The output logits are converted to probabilities via temperature-scaled softmax.
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
        
        # Final layer outputs expert logits
        layers.append(nn.Linear(in_dim, config.n_experts))
        
        self.mlp = nn.Sequential(*layers)
        
        # Temperature parameter
        if config.learn_temperature:
            # Initialize log_temperature so that exp(log_temp) = config.temperature
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
        evidence: torch.Tensor,                    # [B, evidence_dim] or [B, max_rows, hidden_dim]
        reconstruction_errors: Optional[torch.Tensor] = None,  # [B, n_heads] or [B, max_rows, n_heads]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gating logits and probabilities.
        
        Args:
            evidence: Evidence vector(s) from encoder.
                     Dataset-level: [B, evidence_dim]
                     Row-level: [B, max_rows, hidden_dim]
            reconstruction_errors: Optional reconstruction errors per head.
                                  Dataset-level: [B, n_heads]
                                  Row-level: [B, max_rows, n_heads]
        
        Returns:
            logits: Raw gating logits. Shape matches evidence batch dims + [n_experts]
            probs: Gating probabilities (temperature-scaled softmax of logits).
        """
        # Concatenate reconstruction errors if provided and configured
        if self.config.use_reconstruction_errors and reconstruction_errors is not None:
            gate_input = torch.cat([evidence, reconstruction_errors], dim=-1)
        else:
            gate_input = evidence
        
        # Compute logits
        logits = self.mlp(gate_input)  # [..., n_experts]
        
        # Temperature-scaled softmax
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        return logits, probs


# =============================================================================
# Expert Heads
# =============================================================================

class ExpertHead(nn.Module):
    """
    Lightweight expert head for mechanism-specific refinement.
    
    Each expert takes the evidence/row representation and produces a scalar
    adjustment to the gating logit. This allows experts to learn mechanism-specific
    patterns that refine the base gating decision.
    
    In "pure gating" mode (use_expert_heads=False), these are not used.
    """
    
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
        """
        Compute expert adjustment.
        
        Args:
            x: Input representation. Shape: [..., input_dim]
        
        Returns:
            adjustment: Scalar adjustment. Shape: [...]
        """
        return self.net(x).squeeze(-1)


class ExpertHeads(nn.Module):
    """
    Container for all expert heads.
    
    Manages one head per expert (mechanism class / MNAR variant).
    """
    
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
        """
        Compute all expert adjustments.
        
        Args:
            evidence: Input representation. Shape: [..., input_dim]
        
        Returns:
            adjustments: Expert adjustments. Shape: [..., n_experts]
        """
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
    
    Combines a gating network with optional expert heads to produce
    mechanism posteriors. The gating network learns to recognize which
    mechanism generated the data based on the evidence vector and
    optionally reconstruction errors.
    
    Architecture:
        evidence [B, evidence_dim]
             │
             ├──────────────────────────────┐
             │                              │
             ▼                              ▼
        GatingNetwork                  ExpertHeads (optional)
             │                              │
             ▼                              ▼
        gate_logits [B, n_experts]     expert_adj [B, n_experts]
             │                              │
             └──────────┬───────────────────┘
                        │
                        ▼
                combined_logits = gate_logits + expert_adj
                        │
                        ▼
                p(mechanism) = softmax(combined_logits / T)
    
    Attributes:
        config: MoEConfig with architecture parameters.
        gating: GatingNetwork that produces mixture weights.
        experts: Optional ExpertHeads for mechanism-specific refinement.
    
    Example:
        >>> config = MoEConfig(evidence_dim=64)
        >>> moe = MixtureOfExperts(config)
        >>> output = moe(evidence)
        >>> output.gate_probs.shape
        torch.Size([B, 5])  # 5 experts
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
        # Experts 0, 1 are MCAR, MAR; rest are MNAR variants
        self.register_buffer(
            "expert_to_class",
            torch.tensor([MCAR, MAR] + [MNAR] * len(config.mnar_variants))
        )
    
    def forward(
        self,
        evidence: torch.Tensor,
        reconstruction_errors: Optional[torch.Tensor] = None,
        row_mask: Optional[torch.Tensor] = None,
    ) -> MoEOutput:
        """
        Compute mechanism posterior via gated experts.
        
        Args:
            evidence: Evidence vector(s) from encoder.
                     Dataset-level: [B, evidence_dim]
                     Row-level: [B, max_rows, hidden_dim]
            reconstruction_errors: Optional reconstruction errors per head.
                                  Dataset-level: [B, n_heads]
                                  Row-level: [B, max_rows, n_heads]
            row_mask: Optional mask for valid rows (row-level gating only).
                     Shape: [B, max_rows]
        
        Returns:
            MoEOutput containing:
                - gate_logits: Raw gating logits
                - gate_probs: Gating probabilities (mechanism posterior)
                - expert_outputs: Expert adjustments (if use_expert_heads)
                - combined_output: Final combined logits
        """
        # Get base gating logits
        gate_logits, gate_probs = self.gating(evidence, reconstruction_errors)
        
        # Apply expert refinement if configured
        if self.experts is not None:
            expert_adj = self.experts(evidence)
            combined_logits = gate_logits + expert_adj
            combined_probs = F.softmax(combined_logits / self.gating.temperature, dim=-1)
        else:
            expert_adj = None
            combined_logits = gate_logits
            combined_probs = gate_probs
        
        # For row-level gating, aggregate to dataset level if needed
        # (store row-level in expert_outputs for analysis)
        
        return MoEOutput(
            gate_logits=gate_logits,
            gate_probs=combined_probs,
            expert_outputs=[expert_adj] if expert_adj is not None else None,
            combined_output=combined_logits,
        )
    
    def get_class_posterior(
        self,
        moe_output: MoEOutput,
    ) -> torch.Tensor:
        """
        Aggregate expert posteriors into mechanism class posterior.
        
        Combines MNAR variant probabilities into single MNAR probability.
        
        Args:
            moe_output: Output from forward().
        
        Returns:
            p_class: Class posterior [B, 3] for MCAR, MAR, MNAR.
        """
        # gate_probs: [B, n_experts]
        probs = moe_output.gate_probs
        B = probs.shape[0]
        device = probs.device
        
        # Sum probabilities by class
        # expert_to_class maps each expert to its class (0=MCAR, 1=MAR, 2=MNAR)
        p_class = torch.zeros(B, 3, device=device)
        
        for expert_idx in range(self.config.n_experts):
            class_idx = self.expert_to_class[expert_idx].item()
            p_class[:, class_idx] += probs[:, expert_idx]
        
        return p_class
    
    def get_mnar_variant_posterior(
        self,
        moe_output: MoEOutput,
    ) -> torch.Tensor:
        """
        Extract MNAR variant posterior (conditioned on MNAR).
        
        Args:
            moe_output: Output from forward().
        
        Returns:
            p_variant: Variant posterior [B, n_mnar_variants].
                      Normalized to sum to 1 (conditional on MNAR).
        """
        # gate_probs: [B, n_experts]
        probs = moe_output.gate_probs
        
        # MNAR variants are experts 2, 3, 4, ...
        mnar_probs = probs[:, 2:]  # [B, n_mnar_variants]
        
        # Normalize to get conditional probabilities
        mnar_sum = mnar_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        p_variant = mnar_probs / mnar_sum
        
        return p_variant
    
    def compute_load_balance_loss(
        self,
        moe_output: MoEOutput,
    ) -> torch.Tensor:
        """
        Compute load balancing loss to prevent expert collapse.
        
        Encourages uniform usage of experts across the batch.
        
        Args:
            moe_output: Output from forward().
        
        Returns:
            loss: Load balancing loss (scalar).
        """
        # gate_probs: [B, n_experts]
        probs = moe_output.gate_probs
        
        # Average probability per expert across batch
        avg_prob = probs.mean(dim=0)  # [n_experts]
        
        # Fraction of batch where each expert has highest probability
        hard_assignments = probs.argmax(dim=-1)  # [B]
        expert_counts = torch.zeros(self.config.n_experts, device=probs.device)
        for i in range(self.config.n_experts):
            expert_counts[i] = (hard_assignments == i).float().sum()
        expert_fractions = expert_counts / probs.shape[0]
        
        # Load balance loss: product of avg_prob and expert_fraction
        # Minimizing this encourages uniform distribution
        # See Switch Transformer paper for derivation
        n_experts = self.config.n_experts
        loss = n_experts * (avg_prob * expert_fractions).sum()
        
        return loss
    
    def compute_entropy_loss(
        self,
        moe_output: MoEOutput,
    ) -> torch.Tensor:
        """
        Compute entropy regularization loss.
        
        Encourages confident (low entropy) or uncertain (high entropy)
        predictions depending on the sign of entropy_weight.
        
        Args:
            moe_output: Output from forward().
        
        Returns:
            loss: Negative entropy (scalar). Minimize to maximize entropy.
        """
        # gate_probs: [B, n_experts]
        probs = moe_output.gate_probs
        
        # Entropy per sample
        log_probs = torch.log(probs.clamp(min=1e-8))
        entropy = -(probs * log_probs).sum(dim=-1)  # [B]
        
        # Return negative mean entropy (minimizing this maximizes entropy)
        return -entropy.mean()
    
    def get_auxiliary_losses(
        self,
        moe_output: MoEOutput,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all auxiliary losses for MoE training.
        
        Args:
            moe_output: Output from forward().
        
        Returns:
            Dict with loss names and values.
        """
        losses = {}
        
        if self.config.load_balance_weight > 0:
            losses["load_balance"] = (
                self.config.load_balance_weight * self.compute_load_balance_loss(moe_output)
            )
        
        if self.config.entropy_weight != 0:
            losses["entropy"] = (
                self.config.entropy_weight * self.compute_entropy_loss(moe_output)
            )
        
        return losses


# =============================================================================
# Row-Level Aggregation
# =============================================================================

class RowToDatasetAggregator(nn.Module):
    """
    Aggregates row-level MoE outputs to dataset-level.
    
    When using row-level gating, this module combines the per-row mechanism
    posteriors into a single dataset-level posterior.
    
    Aggregation methods:
        - "mean": Simple average across rows
        - "attention": Learned attention-weighted average
        - "max": Max pooling (take most confident prediction)
    """
    
    def __init__(
        self,
        n_experts: int,
        hidden_dim: int,
        method: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.method = method
        self.n_experts = n_experts
        
        if method == "attention":
            # Attention over rows, conditioned on row gating
            self.attention = nn.Sequential(
                nn.Linear(n_experts, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.dropout = nn.Dropout(dropout)
        elif method not in ("mean", "max"):
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def forward(
        self,
        row_probs: torch.Tensor,   # [B, max_rows, n_experts]
        row_mask: torch.Tensor,     # [B, max_rows]
    ) -> torch.Tensor:
        """
        Aggregate row-level probabilities to dataset level.
        
        Args:
            row_probs: Row-level mechanism posteriors.
            row_mask: Valid row mask.
        
        Returns:
            dataset_probs: Dataset-level posterior. Shape: [B, n_experts]
        """
        # Expand mask for broadcasting
        mask = row_mask.unsqueeze(-1)  # [B, max_rows, 1]
        
        if self.method == "mean":
            # Masked mean
            masked_probs = row_probs * mask.float()
            sum_probs = masked_probs.sum(dim=1)  # [B, n_experts]
            count = row_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            dataset_probs = sum_probs / count
        
        elif self.method == "max":
            # Masked max (take row with highest max probability)
            masked_probs = row_probs.masked_fill(~mask, float('-inf'))
            dataset_probs, _ = masked_probs.max(dim=1)
        
        elif self.method == "attention":
            # Attention-weighted average
            scores = self.attention(row_probs).squeeze(-1)  # [B, max_rows]
            scores = scores.masked_fill(~row_mask, float('-inf'))
            weights = F.softmax(scores, dim=-1)  # [B, max_rows]
            weights = self.dropout(weights)
            dataset_probs = torch.einsum("br,bre->be", weights, row_probs)
        
        return dataset_probs


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
    use_expert_heads: bool = False,
    temperature: float = 1.0,
    learn_temperature: bool = False,
    load_balance_weight: float = 0.0,
    entropy_weight: float = 0.0,
    dropout: float = 0.1,
) -> MixtureOfExperts:
    """
    Factory function to create a MixtureOfExperts layer.
    
    Args:
        evidence_dim: Dimension of dataset evidence vector.
        hidden_dim: Dimension of row representations (for row-level gating).
        mnar_variants: List of MNAR variant names.
        gate_hidden_dim: Hidden dimension in gating network.
        gate_n_layers: Depth of gating network.
        gating_level: "dataset" or "row".
        use_reconstruction_errors: Include reconstruction errors in gate input.
        n_reconstruction_heads: Number of reconstruction heads.
        use_expert_heads: Use expert refinement heads.
        temperature: Temperature for softmax calibration.
        learn_temperature: Learn temperature as parameter.
        load_balance_weight: Weight for load balancing loss.
        entropy_weight: Weight for entropy regularization.
        dropout: Dropout probability.
    
    Returns:
        Configured MixtureOfExperts instance.
    
    Example:
        >>> moe = create_moe(evidence_dim=64, use_reconstruction_errors=True)
        >>> output = moe(evidence, reconstruction_errors)
        >>> p_class = moe.get_class_posterior(output)
    """
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
        use_expert_heads=use_expert_heads,
        temperature=temperature,
        learn_temperature=learn_temperature,
        load_balance_weight=load_balance_weight,
        entropy_weight=entropy_weight,
    )
    
    return MixtureOfExperts(config)