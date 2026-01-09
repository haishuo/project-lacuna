"""
lacuna.core.types

Core data types for Project Lacuna.

All types are immutable dataclasses with validation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import torch

# Mechanism class IDs
MCAR = 0
MAR = 1
MNAR = 2

CLASS_NAMES = ("MCAR", "MAR", "MNAR")


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
    
    Attributes:
        tokens: [B, max_rows, max_cols, token_dim] token tensor.
        row_mask: [B, max_rows] bool. True = real row, False = padding.
        col_mask: [B, max_cols] bool. True = real column, False = padding.
        generator_ids: [B] generator labels (training only).
        class_ids: [B] class labels (derived from generator_ids).
    """
    tokens: torch.Tensor
    row_mask: torch.Tensor
    col_mask: torch.Tensor
    generator_ids: Optional[torch.Tensor] = None
    class_ids: Optional[torch.Tensor] = None
    
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
        )


@dataclass(frozen=True)
class PosteriorResult:
    """Model output posteriors.
    
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


@dataclass(frozen=True)
class Decision:
    """Bayes-optimal decision output.
    
    Attributes:
        action_ids: [B] action indices in {0, 1, 2}.
        action_names: Names for actions ("Green", "Yellow", "Red").
        expected_risks: [B] expected loss under optimal action.
    """
    action_ids: torch.Tensor
    action_names: Tuple[str, str, str] = ("Green", "Yellow", "Red")
    expected_risks: torch.Tensor = None
    
    @property
    def batch_size(self) -> int:
        return self.action_ids.shape[0]
    
    def get_actions(self) -> list:
        return [self.action_names[i] for i in self.action_ids.tolist()]
