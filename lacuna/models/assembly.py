"""
lacuna.models.assembly

Complete Lacuna model assembling all components.

Enhanced with explicit missingness pattern features for improved MAR/MCAR discrimination.

Architecture:
    1. Encoder: Tokenized data -> evidence vector + token representations
    2. Reconstruction Heads: Predict masked values (natural error = discrimination signal)
    3. Missingness Feature Extractor: Explicit statistical features of missingness patterns
    4. Mixture of Experts: Combine all signals for mechanism classification
    5. Bayes-Optimal Decision: Convert posteriors to recommended actions

The key enhancement is the MissingnessFeatureExtractor which computes 16 statistical
features that strongly discriminate between mechanisms:
    - Missing rate statistics (variance, range) -> MCAR vs MAR (d > 9.0)
    - Little's test approximation -> MAR vs MNAR (d > 2.8)
    - Cross-column correlations -> Additional discrimination signal

Training Modes:
    1. Pretraining: Reconstruction loss only (self-supervised)
    2. Classification: Mechanism loss only (supervised on synthetic data)
    3. Joint: Reconstruction + Mechanism loss (full training)
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
from lacuna.data.missingness_features import (
    MissingnessFeatureExtractor,
    MissingnessFeatureConfig,
    DEFAULT_CONFIG as DEFAULT_MISSINGNESS_CONFIG,
)


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
    
    # === Missingness Features ===
    use_missingness_features: bool = True  # Extract explicit missingness features
    
    # === MoE ===
    gate_hidden_dim: int = 64         # Gating network hidden dimension
    gate_n_layers: int = 2            # Gating network depth
    gating_level: str = "dataset"     # "dataset" or "row"
    use_reconstruction_errors: bool = True  # Feed recon errors to gate
    use_expert_heads: bool = False    # Use expert refinement heads
    
    # === Calibration ===
    temperature: float = 1.0          # Softmax temperature
    learn_temperature: bool = False   # Learn temperature as parameter
    
    # === Aggregation ===
    class_aggregation: str = "mean"   # "mean", "sum", or "learned"
    
    # === Regularization ===
    load_balance_weight: float = 0.0  # MoE load balancing loss weight
    dropout: float = 0.1              # Dropout probability
    
    # === Decision Rule ===
    loss_matrix: List[float] = field(default_factory=lambda: [
        # Green action (assume MCAR)
        0.0, 0.3, 1.0,   # True: MCAR, MAR, MNAR
        # Yellow action (assume MAR)
        0.2, 0.0, 0.2,   # True: MCAR, MAR, MNAR
        # Red action (assume MNAR)
        1.0, 0.3, 0.0,   # True: MCAR, MAR, MNAR
    ])
    
    def __post_init__(self):
        if self.mnar_variants is None:
            self.mnar_variants = ["self_censoring", "threshold", "latent"]
        
        if self.hidden_dim % self.n_heads != 0:
            raise ValidationError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})"
            )
    
    def get_encoder_config(self) -> EncoderConfig:
        """Create EncoderConfig from this config."""
        return EncoderConfig(
            hidden_dim=self.hidden_dim,
            evidence_dim=self.evidence_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            max_cols=self.max_cols,
            row_pooling=self.row_pooling,
            dataset_pooling=self.dataset_pooling,
            dropout=self.dropout,
        )
    
    def get_reconstruction_config(self) -> ReconstructionConfig:
        """Create ReconstructionConfig from this config."""
        return ReconstructionConfig(
            hidden_dim=self.hidden_dim,
            head_hidden_dim=self.recon_head_hidden_dim,
            n_head_layers=self.recon_n_head_layers,
            dropout=self.dropout,
            mnar_variants=self.mnar_variants,
        )
    
    def get_moe_config(self) -> MoEConfig:
        """Create MoEConfig from this config."""
        # Calculate number of reconstruction heads
        n_recon_heads = 2 + len(self.mnar_variants)  # mcar + mar + mnar variants
        
        # Calculate number of missingness features
        n_miss_features = DEFAULT_MISSINGNESS_CONFIG.n_features if self.use_missingness_features else 0
        
        return MoEConfig(
            evidence_dim=self.evidence_dim,
            hidden_dim=self.hidden_dim,
            mnar_variants=self.mnar_variants,
            gate_hidden_dim=self.gate_hidden_dim,
            gate_n_layers=self.gate_n_layers,
            gating_level=self.gating_level,
            use_reconstruction_errors=self.use_reconstruction_errors,
            n_reconstruction_heads=n_recon_heads,
            use_missingness_features=self.use_missingness_features,
            n_missingness_features=n_miss_features,
            use_expert_heads=self.use_expert_heads,
            temperature=self.temperature,
            learn_temperature=self.learn_temperature,
            class_aggregation=self.class_aggregation,
            load_balance_weight=self.load_balance_weight,
            gate_dropout=self.dropout,
        )


# =============================================================================
# Bayes-Optimal Decision Rule
# =============================================================================

def compute_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute entropy of probability distribution."""
    log_probs = torch.log(probs.clamp(min=eps))
    return -(probs * log_probs).sum(dim=-1)


class BayesOptimalDecision(nn.Module):
    """
    Bayes-optimal decision rule for mechanism classification.
    
    Given posterior P(class | data) and loss matrix L[action, true_class],
    choose action that minimizes expected loss:
    
        action* = argmin_a sum_c P(c | data) * L[a, c]
    
    Default loss matrix encodes:
        - Green (assume MCAR): High loss if MNAR, moderate if MAR
        - Yellow (assume MAR): Moderate loss if wrong
        - Red (assume MNAR): High loss if MCAR (over-conservative)
    """
    
    def __init__(self, loss_matrix: torch.Tensor):
        """
        Initialize decision rule.
        
        Args:
            loss_matrix: [n_actions, n_classes] loss matrix.
        """
        super().__init__()
        
        if loss_matrix.shape != (3, 3):
            raise ValueError(f"loss_matrix must be [3, 3], got {loss_matrix.shape}")
        
        self.register_buffer("loss_matrix", loss_matrix)
    
    def forward(self, p_class: torch.Tensor) -> Decision:
        """
        Compute Bayes-optimal decision.
        
        Args:
            p_class: [B, 3] posterior over classes.
        
        Returns:
            Decision with action IDs and expected risks.
        """
        # Expected risk for each action: [B, n_actions]
        # risk[b, a] = sum_c P(c | data_b) * L[a, c]
        expected_risks = torch.matmul(p_class, self.loss_matrix.T)
        
        # Bayes-optimal action: minimize expected risk
        action_ids = expected_risks.argmin(dim=-1)
        
        # Minimum expected risk (for confidence)
        min_risks = expected_risks.min(dim=-1).values
        
        return Decision(
            action_ids=action_ids,
            expected_risks=min_risks,
            all_risks=expected_risks,
        )


# =============================================================================
# Decision Data Class
# =============================================================================

@dataclass
class Decision:
    """Output of Bayes-optimal decision rule."""
    
    action_ids: torch.Tensor      # [B] action indices (0=Green, 1=Yellow, 2=Red)
    expected_risks: torch.Tensor  # [B] minimum expected risk
    all_risks: torch.Tensor       # [B, 3] expected risk for each action
    
    ACTION_NAMES = ["Green", "Yellow", "Red"]
    ACTION_DESCRIPTIONS = [
        "Assume MCAR - proceed with complete-case or simple imputation",
        "Assume MAR - use multiple imputation or likelihood methods",
        "Assume MNAR - use sensitivity analysis or selection models",
    ]
    
    def get_actions(self) -> List[str]:
        """Get action names for batch."""
        return [self.ACTION_NAMES[i] for i in self.action_ids.tolist()]
    
    def get_descriptions(self) -> List[str]:
        """Get action descriptions for batch."""
        return [self.ACTION_DESCRIPTIONS[i] for i in self.action_ids.tolist()]


# =============================================================================
# Main Model
# =============================================================================

class LacunaModel(nn.Module):
    """
    Complete Lacuna model for missing data mechanism classification.
    
    This model combines:
        1. Transformer encoder for learning data representations
        2. Reconstruction heads for self-supervised pretraining
        3. Missingness feature extractor for explicit pattern statistics
        4. Mixture of Experts for mechanism classification
        5. Bayes-optimal decision rule for action recommendations
    
    The MoE gating network receives three types of signals:
        - Evidence vector: Learned representation from encoder
        - Reconstruction errors: How well each head predicts missing values
        - Missingness features: Explicit statistical patterns (critical for MCAR vs MAR)
    
    CRITICAL: The MoE uses BALANCED class aggregation (default "mean") to
    avoid structural prior bias toward MNAR from having more MNAR experts.
    """
    
    def __init__(self, config: LacunaModelConfig):
        super().__init__()
        
        self.config = config
        
        # === Encoder ===
        self.encoder = LacunaEncoder(config.get_encoder_config())
        
        # === Reconstruction Heads ===
        self.reconstruction = ReconstructionHeads(config.get_reconstruction_config())
        
        # === Missingness Feature Extractor ===
        if config.use_missingness_features:
            self.missingness_extractor = MissingnessFeatureExtractor()
        else:
            self.missingness_extractor = None
        
        # === Mixture of Experts ===
        self.moe = MixtureOfExperts(config.get_moe_config())
        
        # === Decision Rule ===
        loss_matrix = torch.tensor(config.loss_matrix).reshape(3, 3)
        self.decision_rule = BayesOptimalDecision(loss_matrix)
        
        # === Class mapping for backward compatibility ===
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
        
        The MoE gating now receives THREE types of signals:
            1. Evidence vector from encoder
            2. Natural reconstruction errors (MAR vs MNAR discrimination)
            3. Missingness pattern features (MCAR vs MAR discrimination)
        
        Args:
            batch: TokenBatch containing tokenized datasets.
            compute_reconstruction: Whether to compute reconstruction predictions.
            compute_decision: Whether to compute Bayes-optimal decision.
        
        Returns:
            LacunaOutput containing all model outputs.
        """
        # Move batch to correct device if needed
        device = next(self.parameters()).device
        batch = batch.to(device)
        
        # === 1. Encode ===
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
        reconstruction_errors_for_moe = None
        
        if compute_reconstruction:
            reconstruction_results = self.reconstruction(
                token_repr=token_repr,
                tokens=batch.tokens,
                row_mask=batch.row_mask,
                col_mask=batch.col_mask,
                original_values=batch.original_values,
                reconstruction_mask=batch.reconstruction_mask,
                compute_natural_errors=True,
            )
            
            # Use natural errors for MoE discrimination
            natural_errors = self.reconstruction.get_natural_error_tensor(reconstruction_results)
            
            if natural_errors is not None:
                reconstruction_errors_for_moe = natural_errors
            else:
                reconstruction_errors_for_moe = self.reconstruction.get_error_tensor(reconstruction_results)
        
        # === 3. Missingness Features ===
        missingness_features = None
        if self.missingness_extractor is not None:
            missingness_features = self.missingness_extractor(
                batch.tokens,
                batch.row_mask,
                batch.col_mask,
            )
            # Safety check for NaN/Inf
            if torch.isnan(missingness_features).any() or torch.isinf(missingness_features).any():
                missingness_features = torch.where(
                    torch.isnan(missingness_features) | torch.isinf(missingness_features),
                    torch.zeros_like(missingness_features),
                    missingness_features
                )
        
        # === 4. Mixture of Experts ===
        moe_output = self.moe(
            evidence=evidence,
            reconstruction_errors=reconstruction_errors_for_moe,
            missingness_features=missingness_features,
        )
        
        # === 5. Build PosteriorResult ===
        p_class = self.moe.get_class_posterior(moe_output)
        p_mnar_variant = self.moe.get_mnar_variant_posterior(moe_output)
        p_mechanism = moe_output.gate_probs
        
        # Compute entropies
        entropy_class = compute_entropy(p_class)
        entropy_mechanism = compute_entropy(p_mechanism)
        
        # Build reconstruction errors dict for PosteriorResult
        recon_errors_dict = {}
        if reconstruction_results is not None:
            for name in self.reconstruction.head_names:
                result = reconstruction_results[name]
                if hasattr(result, 'natural_errors') and result.natural_errors is not None:
                    recon_errors_dict[name] = result.natural_errors
                else:
                    recon_errors_dict[name] = result.errors
        
        posterior = PosteriorResult(
            p_class=p_class,
            p_mnar_variant=p_mnar_variant,
            p_mechanism=p_mechanism,
            entropy_class=entropy_class,
            entropy_mechanism=entropy_mechanism,
            logits_class=None,
            logits_mnar_variant=None,
            gate_probs=moe_output.gate_probs,
            reconstruction_errors=recon_errors_dict if recon_errors_dict else None,
        )
        
        # === 6. Decision ===
        decision = None
        if compute_decision:
            decision = self.decision_rule(p_class)
        
        # === 7. Assemble Output ===
        return LacunaOutput(
            posterior=posterior,
            decision=decision,
            reconstruction=reconstruction_results,
            moe=moe_output,
            evidence=evidence,
        )
    
    def forward_classification_only(
        self,
        batch: TokenBatch,
    ) -> PosteriorResult:
        """
        Simplified forward for classification only (no reconstruction).
        
        Note: This still computes missingness features as they don't require
        reconstruction and are critical for MCAR vs MAR discrimination.
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
        """Forward pass returning posterior and decision."""
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
        """Compute auxiliary losses for training."""
        return self.moe.get_auxiliary_losses(output.moe)
    
    def encode(self, batch: TokenBatch) -> torch.Tensor:
        """Get evidence vector only."""
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
        """Get token-level representations."""
        device = next(self.parameters()).device
        batch = batch.to(device)
        
        return self.encoder.get_token_representations(
            batch.tokens,
            batch.row_mask,
            batch.col_mask,
        )
    
    def get_missingness_features(
        self,
        batch: TokenBatch,
    ) -> Optional[torch.Tensor]:
        """Get explicit missingness pattern features."""
        if self.missingness_extractor is None:
            return None
        
        device = next(self.parameters()).device
        batch = batch.to(device)
        
        return self.missingness_extractor(
            batch.tokens,
            batch.row_mask,
            batch.col_mask,
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_lacuna_model(
    hidden_dim: int = 128,
    evidence_dim: int = 64,
    n_layers: int = 4,
    n_heads: int = 4,
    max_cols: int = 32,
    row_pooling: str = "attention",
    dataset_pooling: str = "attention",
    recon_head_hidden_dim: int = 64,
    recon_n_head_layers: int = 2,
    mnar_variants: Optional[List[str]] = None,
    use_missingness_features: bool = True,
    gate_hidden_dim: int = 64,
    gate_n_layers: int = 2,
    gating_level: str = "dataset",
    use_reconstruction_errors: bool = True,
    use_expert_heads: bool = False,
    temperature: float = 1.0,
    learn_temperature: bool = False,
    class_aggregation: str = "mean",
    load_balance_weight: float = 0.0,
    loss_matrix: Optional[List[float]] = None,
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
        row_pooling: Row pooling method.
        dataset_pooling: Dataset pooling method.
        recon_head_hidden_dim: Reconstruction head hidden dimension.
        recon_n_head_layers: Reconstruction head depth.
        mnar_variants: List of MNAR variant names.
        use_missingness_features: Extract explicit missingness pattern features.
        gate_hidden_dim: Gating network hidden dimension.
        gate_n_layers: Gating network depth.
        gating_level: "dataset" or "row".
        use_reconstruction_errors: Feed reconstruction errors to gate.
        use_expert_heads: Use expert refinement heads.
        temperature: Softmax temperature for calibration.
        learn_temperature: Learn temperature as parameter.
        class_aggregation: Method for aggregating expert probs to class probs.
        load_balance_weight: MoE load balancing loss weight.
        loss_matrix: Bayes decision loss matrix (flat list, row-major).
        dropout: Dropout probability.
    
    Returns:
        Configured LacunaModel instance.
    """
    if mnar_variants is None:
        mnar_variants = ["self_censoring", "threshold", "latent"]
    
    if loss_matrix is None:
        loss_matrix = [
            0.0, 0.3, 1.0,   # Green
            0.2, 0.0, 0.2,   # Yellow
            1.0, 0.3, 0.0,   # Red
        ]
    
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
        use_missingness_features=use_missingness_features,
        gate_hidden_dim=gate_hidden_dim,
        gate_n_layers=gate_n_layers,
        gating_level=gating_level,
        use_reconstruction_errors=use_reconstruction_errors,
        use_expert_heads=use_expert_heads,
        temperature=temperature,
        learn_temperature=learn_temperature,
        class_aggregation=class_aggregation,
        load_balance_weight=load_balance_weight,
        dropout=dropout,
        loss_matrix=loss_matrix,
    )
    
    return LacunaModel(config)


def create_lacuna_mini(
    max_cols: int = 32,
    mnar_variants: Optional[List[str]] = None,
    use_missingness_features: bool = True,
) -> LacunaModel:
    """
    Create a minimal Lacuna model for testing and fast iteration.
    """
    if mnar_variants is None:
        mnar_variants = ["self_censoring", "threshold"]
    
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
        use_missingness_features=use_missingness_features,
        gate_hidden_dim=32,
        gate_n_layers=1,
        use_reconstruction_errors=True,
        use_expert_heads=False,
        class_aggregation="mean",
        dropout=0.1,
    )


def create_lacuna_base(
    max_cols: int = 32,
    mnar_variants: Optional[List[str]] = None,
    use_missingness_features: bool = True,
) -> LacunaModel:
    """
    Create the standard Lacuna model configuration.
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
        use_missingness_features=use_missingness_features,
        gate_hidden_dim=64,
        gate_n_layers=2,
        use_reconstruction_errors=True,
        use_expert_heads=False,
        temperature=1.0,
        learn_temperature=False,
        class_aggregation="mean",
        dropout=0.1,
    )


def create_lacuna_large(
    max_cols: int = 64,
    mnar_variants: Optional[List[str]] = None,
    use_missingness_features: bool = True,
) -> LacunaModel:
    """
    Create a large Lacuna model for maximum accuracy.
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
        use_missingness_features=use_missingness_features,
        gate_hidden_dim=128,
        gate_n_layers=3,
        use_reconstruction_errors=True,
        use_expert_heads=True,
        temperature=1.0,
        learn_temperature=True,
        class_aggregation="mean",
        load_balance_weight=0.01,
        dropout=0.1,
    )