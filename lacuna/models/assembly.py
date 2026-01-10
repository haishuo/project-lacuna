"""
lacuna.models.assembly

Complete Lacuna model assembly.

This module wires together all components of the BERT-inspired architecture:
    1. Encoder: Tokenizes data and produces evidence vectors
    2. Reconstruction Heads: Predict masked values (self-supervised pretraining)
    3. Mixture of Experts: Produces mechanism posteriors
    4. Decision Rule: Converts posteriors to actionable decisions

Architecture Overview:

    Input: TokenBatch
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                        LacunaEncoder                            │
    │  tokens → TokenEmbedding → Transformer → RowPool → DatasetPool  │
    └─────────────────────────────────────────────────────────────────┘
        │                           │
        │ evidence [B, evidence_dim]│ token_repr [B, max_rows, max_cols, hidden_dim]
        │                           │
        │                           ▼
        │              ┌─────────────────────────────┐
        │              │    ReconstructionHeads      │
        │              │  MCAR | MAR | MNAR variants │
        │              └─────────────────────────────┘
        │                           │
        │                           │ errors [B, n_heads]
        │                           │
        ▼                           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      MixtureOfExperts                           │
    │  evidence + errors → GatingNetwork → (ExpertHeads) → posterior  │
    └─────────────────────────────────────────────────────────────────┘
        │
        │ p_class [B, 3], p_mechanism [B, n_experts]
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      BayesOptimalDecision                       │
    │  p_class + loss_matrix → action (Green/Yellow/Red)              │
    └─────────────────────────────────────────────────────────────────┘
        │
        ▼
    LacunaOutput (posterior, decision, reconstruction, moe_output, evidence)

Training Modes:
    1. Pretraining: Reconstruction loss only (self-supervised)
    2. Classification: Mechanism loss only (supervised on synthetic data)
    3. Joint: Reconstruction + Mechanism loss (full training)

Usage:
    # Create model
    model = create_lacuna_model(config)
    
    # Forward pass
    output = model(batch)
    
    # Access components
    posterior = output.posterior       # PosteriorResult
    decision = output.decision         # Decision
    recon = output.reconstruction      # Dict[str, ReconstructionResult]
    moe = output.moe_output           # MoEOutput
    evidence = output.evidence        # [B, evidence_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field

from lacuna.core.types import (
    TokenBatch,
    PosteriorResult,
    Decision,
    ReconstructionResult,
    MoEOutput,
    LacunaOutput,
    MCAR,
    MAR,
    MNAR,
    CLASS_NAMES,
)
from lacuna.core.exceptions import ValidationError
from lacuna.models.encoder import LacunaEncoder, EncoderConfig, create_encoder
from lacuna.models.reconstruction import (
    ReconstructionHeads,
    ReconstructionConfig,
    create_reconstruction_heads,
)
from lacuna.models.moe import MixtureOfExperts, MoEConfig, create_moe
from lacuna.data.tokenization import TOKEN_DIM


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LacunaModelConfig:
    """Complete configuration for the Lacuna model."""
    
    # === Encoder ===
    hidden_dim: int = 128          # Transformer hidden dimension
    evidence_dim: int = 64         # Final evidence vector dimension
    n_layers: int = 4              # Number of transformer layers
    n_heads: int = 4               # Number of attention heads
    max_cols: int = 32             # Maximum number of columns
    row_pooling: str = "attention" # Row pooling method
    dataset_pooling: str = "attention"  # Dataset pooling method
    
    # === Reconstruction ===
    recon_head_hidden_dim: int = 64   # Hidden dimension in reconstruction heads
    recon_n_head_layers: int = 2      # Depth of reconstruction heads
    mnar_variants: List[str] = None   # MNAR variant names
    
    # === MoE ===
    gate_hidden_dim: int = 64         # Gating network hidden dimension
    gate_n_layers: int = 2            # Gating network depth
    gating_level: str = "dataset"     # "dataset" or "row"
    use_reconstruction_errors: bool = True  # Feed recon errors to gate
    use_expert_heads: bool = False    # Use expert refinement heads
    temperature: float = 1.0          # Softmax temperature
    learn_temperature: bool = False   # Learn temperature
    load_balance_weight: float = 0.0  # MoE load balancing loss weight
    
    # === Decision ===
    # Loss matrix for Bayes-optimal decision rule
    # Rows: actions (Green, Yellow, Red)
    # Cols: true states (MCAR, MAR, MNAR)
    # Entry (a, s) = cost of taking action a when true state is s
    loss_matrix: List[float] = None
    
    # === Regularization ===
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.mnar_variants is None:
            self.mnar_variants = ["self_censoring", "threshold", "latent"]
        
        if self.loss_matrix is None:
            # Default loss matrix:
            #          MCAR  MAR  MNAR
            # Green:    0     0    10   (high cost for ignoring MNAR)
            # Yellow:   1     1     2   (moderate cost, conservative)
            # Red:      3     2     0   (high cost for over-reacting to MCAR/MAR)
            self.loss_matrix = [
                0.0, 0.0, 10.0,  # Green
                1.0, 1.0,  2.0,  # Yellow
                3.0, 2.0,  0.0,  # Red
            ]
    
    @property
    def n_experts(self) -> int:
        """Total number of mechanism experts."""
        return 2 + len(self.mnar_variants)  # MCAR + MAR + MNAR variants
    
    @property
    def n_reconstruction_heads(self) -> int:
        """Total number of reconstruction heads."""
        return self.n_experts  # Same as experts: MCAR, MAR, MNAR variants
    
    def get_encoder_config(self) -> EncoderConfig:
        """Create EncoderConfig from model config."""
        return EncoderConfig(
            hidden_dim=self.hidden_dim,
            evidence_dim=self.evidence_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            max_cols=self.max_cols,
            dropout=self.dropout,
            row_pooling=self.row_pooling,
            dataset_pooling=self.dataset_pooling,
        )
    
    def get_reconstruction_config(self) -> ReconstructionConfig:
        """Create ReconstructionConfig from model config."""
        return ReconstructionConfig(
            hidden_dim=self.hidden_dim,
            head_hidden_dim=self.recon_head_hidden_dim,
            n_head_layers=self.recon_n_head_layers,
            dropout=self.dropout,
            mnar_variants=self.mnar_variants,
        )
    
    def get_moe_config(self) -> MoEConfig:
        """Create MoEConfig from model config."""
        return MoEConfig(
            evidence_dim=self.evidence_dim,
            hidden_dim=self.hidden_dim,
            mnar_variants=self.mnar_variants,
            gate_hidden_dim=self.gate_hidden_dim,
            gate_n_layers=self.gate_n_layers,
            gate_dropout=self.dropout,
            gating_level=self.gating_level,
            use_reconstruction_errors=self.use_reconstruction_errors,
            n_reconstruction_heads=self.n_reconstruction_heads,
            use_expert_heads=self.use_expert_heads,
            temperature=self.temperature,
            learn_temperature=self.learn_temperature,
            load_balance_weight=self.load_balance_weight,
        )


# =============================================================================
# Bayes-Optimal Decision Rule
# =============================================================================

class BayesOptimalDecision(nn.Module):
    """
    Bayes-optimal decision rule for mechanism classification.
    
    Given a posterior over mechanism classes and a loss matrix specifying
    the cost of each action under each true state, computes the action
    that minimizes expected risk.
    
    Actions:
        0 = Green: Proceed with standard analysis (assume MCAR/MAR)
        1 = Yellow: Proceed with caution, sensitivity analysis recommended
        2 = Red: Stop, mechanism likely MNAR, standard methods invalid
    
    Decision rule:
        action* = argmin_a E[L(a, s)] = argmin_a sum_s p(s) * L(a, s)
    
    Attributes:
        loss_matrix: [n_actions, n_classes] loss matrix.
        action_names: Tuple of action names for output.
    """
    
    def __init__(
        self,
        loss_matrix: torch.Tensor,
        action_names: Tuple[str, ...] = ("Green", "Yellow", "Red"),
    ):
        super().__init__()
        
        # loss_matrix: [n_actions, n_classes]
        self.register_buffer("loss_matrix", loss_matrix)
        self.action_names = action_names
        self.n_actions = loss_matrix.shape[0]
        self.n_classes = loss_matrix.shape[1]
    
    def forward(self, p_class: torch.Tensor) -> Decision:
        """
        Compute Bayes-optimal decision.
        
        Args:
            p_class: Class posterior. Shape: [B, n_classes]
        
        Returns:
            Decision with action_ids and expected_risks.
        """
        # Compute expected risk for each action
        # p_class: [B, n_classes]
        # loss_matrix: [n_actions, n_classes]
        # expected_risk[b, a] = sum_s p_class[b, s] * loss_matrix[a, s]
        expected_risks = torch.matmul(p_class, self.loss_matrix.T)  # [B, n_actions]
        
        # Select action with minimum expected risk
        action_ids = expected_risks.argmin(dim=-1)  # [B]
        min_risks = expected_risks.gather(1, action_ids.unsqueeze(-1)).squeeze(-1)  # [B]
        
        # Compute confidence as 1 - (risk ratio)
        # Lower min_risk relative to max_risk = higher confidence
        max_risks = expected_risks.max(dim=-1).values
        risk_range = (max_risks - min_risks).clamp(min=1e-8)
        confidence = risk_range / max_risks.clamp(min=1e-8)
        
        return Decision(
            action_ids=action_ids,
            action_names=self.action_names,
            expected_risks=min_risks,
            confidence=confidence,
        )
    
    def forward_with_all_risks(
        self,
        p_class: torch.Tensor,
    ) -> Tuple[Decision, torch.Tensor]:
        """
        Compute decision and return all expected risks.
        
        Args:
            p_class: Class posterior. Shape: [B, n_classes]
        
        Returns:
            decision: Decision object.
            all_risks: Expected risks for all actions. Shape: [B, n_actions]
        """
        expected_risks = torch.matmul(p_class, self.loss_matrix.T)
        action_ids = expected_risks.argmin(dim=-1)
        min_risks = expected_risks.gather(1, action_ids.unsqueeze(-1)).squeeze(-1)
        
        max_risks = expected_risks.max(dim=-1).values
        risk_range = (max_risks - min_risks).clamp(min=1e-8)
        confidence = risk_range / max_risks.clamp(min=1e-8)
        
        decision = Decision(
            action_ids=action_ids,
            action_names=self.action_names,
            expected_risks=min_risks,
            confidence=confidence,
        )
        
        return decision, expected_risks


# =============================================================================
# Entropy Computation
# =============================================================================

def compute_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute entropy of probability distribution.
    
    Args:
        probs: Probability tensor (should sum to 1 along dim).
        dim: Dimension along which to compute entropy.
    
    Returns:
        entropy: Entropy values. Shape is probs.shape with dim removed.
    """
    # Clamp to avoid log(0)
    log_probs = torch.log(probs.clamp(min=1e-8))
    entropy = -(probs * log_probs).sum(dim=dim)
    return entropy


# =============================================================================
# Main Model
# =============================================================================

class LacunaModel(nn.Module):
    """
    Complete Lacuna model for missing data mechanism classification.
    
    Combines:
        1. LacunaEncoder: Tokenizes data, produces evidence vectors
        2. ReconstructionHeads: Predict masked values (self-supervised)
        3. MixtureOfExperts: Produces mechanism posteriors
        4. BayesOptimalDecision: Converts posteriors to decisions
    
    The model can operate in three modes:
        - Pretraining: Only reconstruction loss, learns data structure
        - Classification: Only mechanism loss, learns to classify
        - Joint: Both losses, full training
    
    Attributes:
        config: LacunaModelConfig with all architecture parameters.
        encoder: LacunaEncoder for tokenization and evidence extraction.
        reconstruction: ReconstructionHeads for self-supervised pretraining.
        moe: MixtureOfExperts for mechanism classification.
        decision_rule: BayesOptimalDecision for action selection.
    
    Example:
        >>> config = LacunaModelConfig()
        >>> model = LacunaModel(config)
        >>> batch = TokenBatch(...)
        >>> output = model(batch)
        >>> print(output.posterior.p_class)  # [B, 3]
        >>> print(output.decision.action_ids)  # [B]
    """
    
    def __init__(self, config: LacunaModelConfig):
        super().__init__()
        
        self.config = config
        
        # === Encoder ===
        self.encoder = LacunaEncoder(config.get_encoder_config())
        
        # === Reconstruction Heads ===
        self.reconstruction = ReconstructionHeads(config.get_reconstruction_config())
        
        # === Mixture of Experts ===
        self.moe = MixtureOfExperts(config.get_moe_config())
        
        # === Decision Rule ===
        # Parse loss matrix from flat list to [n_actions, n_classes] tensor
        loss_matrix = torch.tensor(config.loss_matrix).reshape(3, 3)
        self.decision_rule = BayesOptimalDecision(loss_matrix)
        
        # === Class mapping for backward compatibility ===
        # Maps expert index to class index (MCAR=0, MAR=1, MNAR=2)
        self.register_buffer(
            "expert_to_class",
            torch.tensor([MCAR, MAR] + [MNAR] * len(config.mnar_variants))
        )
    
    def forward(
        self,
        batch: TokenBatch,
        compute_reconstruction: bool = True,
        compute_decision: bool = True,
    ) -> LacunaOutput:
        """
        Full forward pass through Lacuna.
        
        Args:
            batch: TokenBatch containing tokenized datasets.
            compute_reconstruction: Whether to compute reconstruction predictions.
                                   Set False for faster inference if not needed.
            compute_decision: Whether to compute Bayes-optimal decision.
                             Set False if only posteriors are needed.
        
        Returns:
            LacunaOutput containing all model outputs:
                - posterior: PosteriorResult with mechanism probabilities
                - decision: Decision with recommended action (if compute_decision)
                - reconstruction: Dict of ReconstructionResult (if compute_reconstruction)
                - moe_output: MoEOutput with gating details
                - evidence: Evidence vector [B, evidence_dim]
        """
        # Move batch to correct device if needed
        device = next(self.parameters()).device
        batch = batch.to(device)
        
        # === 1. Encode ===
        # Get both evidence vector and token representations
        encoder_output = self.encoder(
            batch.tokens,
            batch.row_mask,
            batch.col_mask,
            return_intermediates=True,
        )
        evidence = encoder_output["evidence"]  # [B, evidence_dim]
        token_repr = encoder_output["token_representations"]  # [B, max_rows, max_cols, hidden_dim]
        
        # === 2. Reconstruction ===
        reconstruction_results = None
        reconstruction_errors = None
        
        if compute_reconstruction:
            reconstruction_results = self.reconstruction(
                token_repr=token_repr,
                tokens=batch.tokens,
                row_mask=batch.row_mask,
                col_mask=batch.col_mask,
                original_values=batch.original_values,
                reconstruction_mask=batch.reconstruction_mask,
            )
            
            # Get error tensor for MoE input
            reconstruction_errors = self.reconstruction.get_error_tensor(reconstruction_results)
        
        # === 3. Mixture of Experts ===
        moe_output = self.moe(
            evidence=evidence,
            reconstruction_errors=reconstruction_errors,
        )
        
        # === 4. Build PosteriorResult ===
        # Get class posterior (aggregates MNAR variants)
        p_class = self.moe.get_class_posterior(moe_output)
        
        # Get MNAR variant posterior (conditional on MNAR)
        p_mnar_variant = self.moe.get_mnar_variant_posterior(moe_output)
        
        # Full mechanism posterior (MCAR, MAR, MNAR_variant1, ...)
        p_mechanism = moe_output.gate_probs
        
        # Compute entropies
        entropy_class = compute_entropy(p_class)
        entropy_mechanism = compute_entropy(p_mechanism)
        
        # Build reconstruction errors dict for PosteriorResult
        recon_errors_dict = {}
        if reconstruction_results is not None:
            for name in self.reconstruction.head_names:
                recon_errors_dict[name] = reconstruction_results[name].errors
        
        posterior = PosteriorResult(
            p_class=p_class,
            p_mnar_variant=p_mnar_variant,
            p_mechanism=p_mechanism,
            entropy_class=entropy_class,
            entropy_mechanism=entropy_mechanism,
            logits_class=None,  # We don't have class-level logits directly
            logits_mnar_variant=None,
            gate_probs=moe_output.gate_probs,
            reconstruction_errors=recon_errors_dict if recon_errors_dict else None,
        )
        
        # === 5. Decision ===
        decision = None
        if compute_decision:
            decision = self.decision_rule(p_class)
        
        # === 6. Assemble Output ===
        return LacunaOutput(
            posterior=posterior,
            decision=decision,
            reconstruction=reconstruction_results,
            moe_output=moe_output,
            evidence=evidence,
        )
    
    def forward_classification_only(
        self,
        batch: TokenBatch,
    ) -> PosteriorResult:
        """
        Simplified forward for classification only (no reconstruction).
        
        Faster inference when reconstruction is not needed.
        
        Args:
            batch: TokenBatch containing tokenized datasets.
        
        Returns:
            PosteriorResult with mechanism probabilities.
        """
        output = self.forward(
            batch,
            compute_reconstruction=False,
            compute_decision=False,
        )
        return output.posterior
    
    def forward_with_decision(
        self,
        batch: TokenBatch,
    ) -> Tuple[PosteriorResult, Decision]:
        """
        Forward pass returning posterior and decision.
        
        Convenience method for inference workflows.
        
        Args:
            batch: TokenBatch containing tokenized datasets.
        
        Returns:
            posterior: PosteriorResult with mechanism probabilities.
            decision: Decision with recommended action.
        """
        output = self.forward(
            batch,
            compute_reconstruction=False,
            compute_decision=True,
        )
        return output.posterior, output.decision
    
    def get_auxiliary_losses(
        self,
        output: LacunaOutput,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary losses for training.
        
        Includes MoE regularization losses (load balancing, entropy).
        
        Args:
            output: Output from forward().
        
        Returns:
            Dict with loss names and values.
        """
        return self.moe.get_auxiliary_losses(output.moe_output)
    
    def encode(self, batch: TokenBatch) -> torch.Tensor:
        """
        Get evidence vector only (for analysis/visualization).
        
        Args:
            batch: TokenBatch containing tokenized datasets.
        
        Returns:
            evidence: Evidence vector. Shape: [B, evidence_dim]
        """
        device = next(self.parameters()).device
        batch = batch.to(device)
        
        return self.encoder(
            batch.tokens,
            batch.row_mask,
            batch.col_mask,
            return_intermediates=False,
        )
    
    def get_token_representations(
        self,
        batch: TokenBatch,
    ) -> torch.Tensor:
        """
        Get token-level representations (for reconstruction analysis).
        
        Args:
            batch: TokenBatch containing tokenized datasets.
        
        Returns:
            token_repr: Token representations.
                       Shape: [B, max_rows, max_cols, hidden_dim]
        """
        device = next(self.parameters()).device
        batch = batch.to(device)
        
        return self.encoder.get_token_representations(
            batch.tokens,
            batch.row_mask,
            batch.col_mask,
        )
    
    def get_row_representations(
        self,
        batch: TokenBatch,
    ) -> torch.Tensor:
        """
        Get row-level representations (for row-level analysis).
        
        Args:
            batch: TokenBatch containing tokenized datasets.
        
        Returns:
            row_repr: Row representations.
                     Shape: [B, max_rows, hidden_dim]
        """
        device = next(self.parameters()).device
        batch = batch.to(device)
        
        return self.encoder.get_row_representations(
            batch.tokens,
            batch.row_mask,
            batch.col_mask,
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_lacuna_model(
    # Encoder
    hidden_dim: int = 128,
    evidence_dim: int = 64,
    n_layers: int = 4,
    n_heads: int = 4,
    max_cols: int = 32,
    row_pooling: str = "attention",
    dataset_pooling: str = "attention",
    # Reconstruction
    recon_head_hidden_dim: int = 64,
    recon_n_head_layers: int = 2,
    mnar_variants: Optional[List[str]] = None,
    # MoE
    gate_hidden_dim: int = 64,
    gate_n_layers: int = 2,
    gating_level: str = "dataset",
    use_reconstruction_errors: bool = True,
    use_expert_heads: bool = False,
    temperature: float = 1.0,
    learn_temperature: bool = False,
    load_balance_weight: float = 0.0,
    # Decision
    loss_matrix: Optional[List[float]] = None,
    # Regularization
    dropout: float = 0.1,
) -> LacunaModel:
    """
    Factory function to create a LacunaModel.
    
    Args:
        hidden_dim: Transformer hidden dimension.
        evidence_dim: Evidence vector dimension.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        max_cols: Maximum number of columns.
        row_pooling: Row pooling method ("mean", "max", "attention").
        dataset_pooling: Dataset pooling method.
        recon_head_hidden_dim: Reconstruction head hidden dimension.
        recon_n_head_layers: Reconstruction head depth.
        mnar_variants: List of MNAR variant names.
        gate_hidden_dim: Gating network hidden dimension.
        gate_n_layers: Gating network depth.
        gating_level: "dataset" or "row".
        use_reconstruction_errors: Feed reconstruction errors to gate.
        use_expert_heads: Use expert refinement heads.
        temperature: Softmax temperature for calibration.
        learn_temperature: Learn temperature as parameter.
        load_balance_weight: MoE load balancing loss weight.
        loss_matrix: Bayes decision loss matrix (flat list, row-major).
        dropout: Dropout probability.
    
    Returns:
        Configured LacunaModel instance.
    
    Example:
        >>> model = create_lacuna_model(hidden_dim=256, n_layers=6)
        >>> output = model(batch)
    """
    if mnar_variants is None:
        mnar_variants = ["self_censoring", "threshold", "latent"]
    
    config = LacunaModelConfig(
        hidden_dim=hidden_dim,
        evidence_dim=evidence_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_cols=max_cols,
        row_pooling=row_pooling,
        dataset_pooling=dataset_pooling,
        recon_head_hidden_dim=recon_head_hidden_dim,
        recon_n_head_layers=recon_n_head_layers,
        mnar_variants=mnar_variants,
        gate_hidden_dim=gate_hidden_dim,
        gate_n_layers=gate_n_layers,
        gating_level=gating_level,
        use_reconstruction_errors=use_reconstruction_errors,
        use_expert_heads=use_expert_heads,
        temperature=temperature,
        learn_temperature=learn_temperature,
        load_balance_weight=load_balance_weight,
        loss_matrix=loss_matrix,
        dropout=dropout,
    )
    
    return LacunaModel(config)


def create_lacuna_mini(
    max_cols: int = 32,
    mnar_variants: Optional[List[str]] = None,
) -> LacunaModel:
    """
    Create a minimal Lacuna model for testing and fast iteration.
    
    Smaller architecture suitable for:
        - Unit tests
        - Quick experiments
        - CPU-only environments
        - Debugging
    
    Args:
        max_cols: Maximum number of columns.
        mnar_variants: List of MNAR variant names.
    
    Returns:
        Small LacunaModel instance.
    """
    if mnar_variants is None:
        mnar_variants = ["self_censoring", "threshold"]  # Only 2 variants
    
    return create_lacuna_model(
        hidden_dim=64,
        evidence_dim=32,
        n_layers=2,
        n_heads=2,
        max_cols=max_cols,
        row_pooling="mean",
        dataset_pooling="mean",
        recon_head_hidden_dim=32,
        recon_n_head_layers=1,
        mnar_variants=mnar_variants,
        gate_hidden_dim=32,
        gate_n_layers=1,
        use_reconstruction_errors=True,
        use_expert_heads=False,
        dropout=0.1,
    )


def create_lacuna_base(
    max_cols: int = 32,
    mnar_variants: Optional[List[str]] = None,
) -> LacunaModel:
    """
    Create the standard Lacuna model configuration.
    
    Balanced architecture suitable for:
        - Production use
        - Full experiments
        - GPU training
    
    Args:
        max_cols: Maximum number of columns.
        mnar_variants: List of MNAR variant names.
    
    Returns:
        Standard LacunaModel instance.
    """
    return create_lacuna_model(
        hidden_dim=128,
        evidence_dim=64,
        n_layers=4,
        n_heads=4,
        max_cols=max_cols,
        row_pooling="attention",
        dataset_pooling="attention",
        recon_head_hidden_dim=64,
        recon_n_head_layers=2,
        mnar_variants=mnar_variants,
        gate_hidden_dim=64,
        gate_n_layers=2,
        use_reconstruction_errors=True,
        use_expert_heads=False,
        temperature=1.0,
        learn_temperature=False,
        dropout=0.1,
    )


def create_lacuna_large(
    max_cols: int = 64,
    mnar_variants: Optional[List[str]] = None,
) -> LacunaModel:
    """
    Create a large Lacuna model for maximum accuracy.
    
    Larger architecture suitable for:
        - Final production models
        - When compute is not a constraint
        - Complex datasets with many features
    
    Args:
        max_cols: Maximum number of columns.
        mnar_variants: List of MNAR variant names.
    
    Returns:
        Large LacunaModel instance.
    """
    return create_lacuna_model(
        hidden_dim=256,
        evidence_dim=128,
        n_layers=6,
        n_heads=8,
        max_cols=max_cols,
        row_pooling="attention",
        dataset_pooling="attention",
        recon_head_hidden_dim=128,
        recon_n_head_layers=3,
        mnar_variants=mnar_variants,
        gate_hidden_dim=128,
        gate_n_layers=3,
        use_reconstruction_errors=True,
        use_expert_heads=True,
        temperature=1.0,
        learn_temperature=True,
        load_balance_weight=0.01,
        dropout=0.1,
    )