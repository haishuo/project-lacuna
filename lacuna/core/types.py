"""
lacuna.core.types

Core dataclasses for Lacuna.

All types are frozen (immutable) for safety.
Validation happens in __post_init__.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import torch


@dataclass(frozen=True)
class ObservedDataset:
    """One dataset with missingness.
    
    This is the fundamental input type for Lacuna.
    
    Attributes:
        x: [n, d] float32 tensor. Missing values are set to 0.0.
        r: [n, d] bool tensor. True = observed, False = missing.
        n: Number of rows (observations).
        d: Number of columns (features).
        feature_names: Tuple of unique feature names.
        dataset_id: Unique identifier for this dataset.
        meta: Optional metadata dict.
    """
    x: torch.Tensor
    r: torch.Tensor
    n: int
    d: int
    feature_names: Tuple[str, ...]
    dataset_id: str
    meta: Optional[dict] = None
    
    def __post_init__(self):
        # Shape checks
        if self.x.shape != (self.n, self.d):
            raise ValueError(
                f"x shape {self.x.shape} != expected ({self.n}, {self.d})"
            )
        if self.r.shape != (self.n, self.d):
            raise ValueError(
                f"r shape {self.r.shape} != expected ({self.n}, {self.d})"
            )
        
        # Type checks
        if self.r.dtype != torch.bool:
            raise TypeError(f"r.dtype must be bool, got {self.r.dtype}")
        
        # Feature name checks
        if len(self.feature_names) != self.d:
            raise ValueError(
                f"len(feature_names)={len(self.feature_names)} != d={self.d}"
            )
        if len(set(self.feature_names)) != self.d:
            raise ValueError("feature_names must be unique")
        
        # At least one observed value
        if self.r.sum() < 1:
            raise ValueError("Dataset must have at least one observed value")
    
    @property
    def missing_rate(self) -> float:
        """Fraction of values that are missing."""
        return 1.0 - self.r.float().mean().item()
    
    @property
    def n_missing(self) -> int:
        """Total count of missing values."""
        return (~self.r).sum().item()
    
    @property
    def n_observed(self) -> int:
        """Total count of observed values."""
        return self.r.sum().item()


@dataclass(frozen=True)
class TokenBatch:
    """Batch of tokenized datasets.
    
    Ready for input to the transformer encoder.
    
    Attributes:
        tokens: [B, max_cols, q] column token features.
        col_mask: [B, max_cols] bool. True = column exists (not padding).
        generator_ids: [B] long tensor. Generator labels (training only).
        class_ids: [B] long tensor. Class labels (derived from generator_ids).
    """
    tokens: torch.Tensor
    col_mask: torch.Tensor
    generator_ids: Optional[torch.Tensor] = None
    class_ids: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        B, max_cols, q = self.tokens.shape
        
        if self.col_mask.shape != (B, max_cols):
            raise ValueError(
                f"col_mask shape {self.col_mask.shape} != expected ({B}, {max_cols})"
            )
        if self.col_mask.dtype != torch.bool:
            raise TypeError(f"col_mask.dtype must be bool, got {self.col_mask.dtype}")
        
        if self.generator_ids is not None:
            if self.generator_ids.shape != (B,):
                raise ValueError(
                    f"generator_ids shape {self.generator_ids.shape} != expected ({B},)"
                )
        
        if self.class_ids is not None:
            if self.class_ids.shape != (B,):
                raise ValueError(
                    f"class_ids shape {self.class_ids.shape} != expected ({B},)"
                )
    
    @property
    def batch_size(self) -> int:
        return self.tokens.shape[0]
    
    @property
    def max_cols(self) -> int:
        return self.tokens.shape[1]
    
    @property
    def token_dim(self) -> int:
        return self.tokens.shape[2]
    
    def to(self, device: str) -> "TokenBatch":
        """Move batch to device."""
        return TokenBatch(
            tokens=self.tokens.to(device),
            col_mask=self.col_mask.to(device),
            generator_ids=self.generator_ids.to(device) if self.generator_ids is not None else None,
            class_ids=self.class_ids.to(device) if self.class_ids is not None else None,
        )


@dataclass(frozen=True)
class PosteriorResult:
    """Model output posteriors.
    
    Contains both generator-level and class-level posteriors,
    plus uncertainty diagnostics.
    
    Attributes:
        p_generator: [B, K] generator posterior probabilities.
        p_class: [B, 3] class posterior (MCAR/MAR/MNAR).
        entropy_generator: [B] entropy of generator posterior.
        entropy_class: [B] entropy of class posterior.
        logits_generator: [B, K] raw logits before softmax.
    """
    p_generator: torch.Tensor
    p_class: torch.Tensor
    entropy_generator: torch.Tensor
    entropy_class: torch.Tensor
    logits_generator: torch.Tensor
    
    def __post_init__(self):
        B = self.p_generator.shape[0]
        
        if self.p_class.shape[0] != B:
            raise ValueError("Batch size mismatch in p_class")
        if self.p_class.shape[1] != 3:
            raise ValueError(f"p_class must have 3 classes, got {self.p_class.shape[1]}")
        if self.entropy_generator.shape != (B,):
            raise ValueError("entropy_generator shape mismatch")
        if self.entropy_class.shape != (B,):
            raise ValueError("entropy_class shape mismatch")
        if self.logits_generator.shape != self.p_generator.shape:
            raise ValueError("logits_generator shape must match p_generator")
    
    @property
    def batch_size(self) -> int:
        return self.p_generator.shape[0]
    
    @property
    def n_generators(self) -> int:
        return self.p_generator.shape[1]
    
    @property
    def confidence(self) -> torch.Tensor:
        """Confidence score: 1 - normalized entropy."""
        max_entropy = torch.log(torch.tensor(3.0))
        return 1.0 - self.entropy_class / max_entropy


@dataclass(frozen=True)
class Decision:
    """Bayes-optimal decision output.
    
    Attributes:
        action_ids: [B] int tensor in {0, 1, 2}.
        action_names: Tuple of action names ("Green", "Yellow", "Red").
        expected_risks: [B] expected loss under optimal action.
    """
    action_ids: torch.Tensor
    action_names: Tuple[str, str, str]
    expected_risks: torch.Tensor
    
    def __post_init__(self):
        if len(self.action_names) != 3:
            raise ValueError("Must have exactly 3 action names")
        if self.action_ids.shape != self.expected_risks.shape:
            raise ValueError("action_ids and expected_risks shape mismatch")
    
    @property
    def batch_size(self) -> int:
        return self.action_ids.shape[0]
    
    def get_actions(self) -> list:
        """Return list of action name strings."""
        return [self.action_names[i] for i in self.action_ids.tolist()]


# Class ID constants
MCAR = 0
MAR = 1
MNAR = 2

CLASS_NAMES = ("MCAR", "MAR", "MNAR")
