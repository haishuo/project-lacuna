"""
lacuna.core.types

Core data types for Project Lacuna.

All types are immutable dataclasses with validation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import torch

# Mechanism class IDs (base classes)
MCAR = 0
MAR = 1
MNAR = 2

CLASS_NAMES = ("MCAR", "MAR", "MNAR")

# MNAR variant IDs (sub-types of MNAR)
MNAR_SELF_CENSORING = 0  # Missingness depends on value itself (high/low values missing)
MNAR_THRESHOLD = 1       # Values beyond threshold systematically missing
MNAR_LATENT = 2          # Missingness depends on unobserved latent factor
MNAR_MIXTURE = 3         # Multiple MNAR behaviors coexist

MNAR_VARIANT_NAMES = (
    "MNAR-SelfCensoring",
    "MNAR-Threshold",
    "MNAR-Latent",
    "MNAR-Mixture",
)

# Default number of MNAR variants
DEFAULT_N_MNAR_VARIANTS = 4


@dataclass(frozen=True)
class ObservedDataset:
    """An observed dataset with missingness.
    
    Attributes:
        x: [n, d] data tensor. Missing values should be 0 (or any value).
        r: [n, d] bool tensor. True = observed, False = missing.
        n: Number of rows.
        d: Number of columns.
        feature_names: Optional column names.
        dataset_id: Unique identifier.
        meta: Additional metadata.
    """
    x: torch.Tensor
    r: torch.Tensor
    n: int
    d: int
    feature_names: Optional[Tuple[str, ...]] = None
    dataset_id: str = "unnamed"
    meta: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.x.shape != (self.n, self.d):
            raise ValueError(f"x shape {self.x.shape} != expected ({self.n}, {self.d})")
        if self.r.shape != (self.n, self.d):
            raise ValueError(f"r shape {self.r.shape} != expected ({self.n}, {self.d})")
        if self.r.dtype != torch.bool:
            raise TypeError(f"r.dtype must be bool, got {self.r.dtype}")
    
    @property
    def missing_rate(self) -> float:
        return 1.0 - self.r.float().mean().item()
    
    @property
    def n_observed(self) -> int:
        return self.r.sum().item()


@dataclass(frozen=True)
class TokenBatch:
    """Batch of tokenized datasets.
    
    Row-level tokenization: each dataset is [n, d, token_dim].
    Batched with padding on both rows and columns.
    
    Token structure per cell:
        - Channel 0: normalized value (0 if missing)
        - Channel 1: observation indicator (1=observed, 0=missing)
        - Channel 2: masking indicator (1=masked for reconstruction, 0=not masked)
    
    Attributes:
        tokens: [B, max_rows, max_cols, token_dim] token tensor.
        row_mask: [B, max_rows] bool. True = real row, False = padding.
        col_mask: [B, max_cols] bool. True = real column, False = padding.
        generator_ids: [B] generator labels (training only).
        class_ids: [B] class labels (derived from generator_ids).
        variant_ids: [B] MNAR variant labels (training only, for MNAR samples).
        original_values: [B, max_rows, max_cols] original values before masking (for reconstruction loss).
        reconstruction_mask: [B, max_rows, max_cols] bool. True = this cell was masked for reconstruction.
    """
    tokens: torch.Tensor
    row_mask: torch.Tensor
    col_mask: torch.Tensor
    generator_ids: Optional[torch.Tensor] = None
    class_ids: Optional[torch.Tensor] = None
    variant_ids: Optional[torch.Tensor] = None
    original_values: Optional[torch.Tensor] = None
    reconstruction_mask: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        B, max_rows, max_cols, token_dim = self.tokens.shape
        
        if self.row_mask.shape != (B, max_rows):
            raise ValueError(f"row_mask shape {self.row_mask.shape} != ({B}, {max_rows})")
        if self.col_mask.shape != (B, max_cols):
            raise ValueError(f"col_mask shape {self.col_mask.shape} != ({B}, {max_cols})")
        
        if self.generator_ids is not None and self.generator_ids.shape != (B,):
            raise ValueError(f"generator_ids shape {self.generator_ids.shape} != ({B},)")
        if self.class_ids is not None and self.class_ids.shape != (B,):
            raise ValueError(f"class_ids shape {self.class_ids.shape} != ({B},)")
        if self.variant_ids is not None and self.variant_ids.shape != (B,):
            raise ValueError(f"variant_ids shape {self.variant_ids.shape} != ({B},)")
        if self.original_values is not None and self.original_values.shape != (B, max_rows, max_cols):
            raise ValueError(f"original_values shape {self.original_values.shape} != ({B}, {max_rows}, {max_cols})")
        if self.reconstruction_mask is not None and self.reconstruction_mask.shape != (B, max_rows, max_cols):
            raise ValueError(f"reconstruction_mask shape {self.reconstruction_mask.shape} != ({B}, {max_rows}, {max_cols})")
    
    @property
    def batch_size(self) -> int:
        return self.tokens.shape[0]
    
    @property
    def max_rows(self) -> int:
        return self.tokens.shape[1]
    
    @property
    def max_cols(self) -> int:
        return self.tokens.shape[2]
    
    @property
    def token_dim(self) -> int:
        return self.tokens.shape[3]
    
    def to(self, device: str) -> "TokenBatch":
        """Move batch to device."""
        return TokenBatch(
            tokens=self.tokens.to(device),
            row_mask=self.row_mask.to(device),
            col_mask=self.col_mask.to(device),
            generator_ids=self.generator_ids.to(device) if self.generator_ids is not None else None,
            class_ids=self.class_ids.to(device) if self.class_ids is not None else None,
            variant_ids=self.variant_ids.to(device) if self.variant_ids is not None else None,
            original_values=self.original_values.to(device) if self.original_values is not None else None,
            reconstruction_mask=self.reconstruction_mask.to(device) if self.reconstruction_mask is not None else None,
        )


@dataclass(frozen=True)
class ReconstructionResult:
    """Output from reconstruction heads.
    
    Attributes:
        predictions: [B, max_rows, max_cols] predicted values for masked cells.
        errors: [B] reconstruction error per sample (e.g., MSE over masked cells).
        per_cell_errors: [B, max_rows, max_cols] per-cell reconstruction errors.
    """
    predictions: torch.Tensor
    errors: torch.Tensor
    per_cell_errors: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class MoEOutput:
    """Output from Mixture of Experts layer.
    
    Attributes:
        gate_logits: [B, n_experts] raw gating logits.
        gate_probs: [B, n_experts] gating probabilities (softmax of logits).
        expert_outputs: List of expert-specific outputs.
        combined_output: Combined output after gating (if applicable).
    """
    gate_logits: torch.Tensor
    gate_probs: torch.Tensor
    expert_outputs: Optional[List[torch.Tensor]] = None
    combined_output: Optional[torch.Tensor] = None
    
    @property
    def n_experts(self) -> int:
        return self.gate_probs.shape[-1]


@dataclass(frozen=True)
class PosteriorResult:
    """Model output posteriors.
    
    Attributes:
        p_class: [B, 3] class posterior (MCAR/MAR/MNAR).
        p_mnar_variant: [B, n_variants] MNAR variant posterior (given MNAR).
        p_mechanism: [B, 2 + n_variants] full mechanism posterior (MCAR, MAR, MNAR_v1, ...).
        entropy_class: [B] entropy of class posterior.
        entropy_mechanism: [B] entropy of full mechanism posterior.
        logits_class: [B, 3] raw class logits.
        logits_mnar_variant: [B, n_variants] raw MNAR variant logits.
        gate_probs: [B, n_experts] MoE gating probabilities (optional).
        reconstruction_errors: Dict[str, Tensor] reconstruction errors by head name.
    """
    p_class: torch.Tensor
    p_mnar_variant: Optional[torch.Tensor] = None
    p_mechanism: Optional[torch.Tensor] = None
    entropy_class: Optional[torch.Tensor] = None
    entropy_mechanism: Optional[torch.Tensor] = None
    logits_class: Optional[torch.Tensor] = None
    logits_mnar_variant: Optional[torch.Tensor] = None
    gate_probs: Optional[torch.Tensor] = None
    reconstruction_errors: Optional[Dict[str, torch.Tensor]] = None
    
    # Legacy fields for backward compatibility with generator-based training
    p_generator: Optional[torch.Tensor] = None
    entropy_generator: Optional[torch.Tensor] = None
    logits_generator: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class Decision:
    """Bayes-optimal decision output.
    
    Attributes:
        action_ids: [B] action indices in {0, 1, 2}.
        action_names: Names for actions ("Green", "Yellow", "Red").
        expected_risks: [B] expected loss under optimal action.
        confidence: [B] confidence score (optional, e.g., 1 - entropy).
    """
    action_ids: torch.Tensor
    expected_risks: torch.Tensor
    action_names: Tuple[str, str, str] = ("Green", "Yellow", "Red")
    confidence: Optional[torch.Tensor] = None
    
    @property
    def batch_size(self) -> int:
        return self.action_ids.shape[0]
    
    def get_actions(self) -> List[str]:
        """Get action names for batch."""
        return [self.action_names[i] for i in self.action_ids.tolist()]


@dataclass(frozen=True)
class LacunaOutput:
    """Complete model output combining all components.
    
    Attributes:
        posterior: Mechanism posterior probabilities.
        decision: Bayes-optimal decision.
        reconstruction: Reconstruction results (if reconstruction heads used).
        moe: MoE gating details (if MoE used).
        evidence: [B, evidence_dim] evidence embedding from encoder.
    """
    posterior: PosteriorResult
    decision: Optional[Decision] = None
    reconstruction: Optional[Dict[str, ReconstructionResult]] = None
    moe: Optional[MoEOutput] = None
    evidence: Optional[torch.Tensor] = None