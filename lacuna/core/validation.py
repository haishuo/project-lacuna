"""
lacuna.core.validation

Boundary validation functions.

Design: Validate at API boundaries, trust internally.
All validation functions raise ValidationError on failure.
"""

import torch
import numpy as np
from typing import Any, Optional, Sequence

from .exceptions import ValidationError, NumericalError


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: tuple,
    name: str = "tensor"
) -> None:
    """Validate tensor has expected shape.
    
    Args:
        tensor: Tensor to validate.
        expected_shape: Expected shape tuple. Use -1 for any size.
        name: Name for error messages.
    
    Raises:
        ValidationError: If shape doesn't match.
    """
    if len(tensor.shape) != len(expected_shape):
        raise ValidationError(
            f"{name} has {len(tensor.shape)} dims, expected {len(expected_shape)}"
        )
    
    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise ValidationError(
                f"{name} dim {i} is {actual}, expected {expected}"
            )


def validate_tensor_dtype(
    tensor: torch.Tensor,
    expected_dtype: torch.dtype,
    name: str = "tensor"
) -> None:
    """Validate tensor has expected dtype."""
    if tensor.dtype != expected_dtype:
        raise ValidationError(
            f"{name} dtype is {tensor.dtype}, expected {expected_dtype}"
        )


def validate_no_nan_inf(
    tensor: torch.Tensor,
    name: str = "tensor"
) -> None:
    """Validate tensor contains no NaN or Inf values.
    
    Raises:
        NumericalError: If NaN or Inf found.
    """
    if torch.isnan(tensor).any():
        n_nan = torch.isnan(tensor).sum().item()
        raise NumericalError(f"{name} contains {n_nan} NaN values")
    
    if torch.isinf(tensor).any():
        n_inf = torch.isinf(tensor).sum().item()
        raise NumericalError(f"{name} contains {n_inf} Inf values")


def validate_positive(
    value: float,
    name: str = "value"
) -> None:
    """Validate value is strictly positive."""
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_non_negative(
    value: float,
    name: str = "value"
) -> None:
    """Validate value is non-negative."""
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")


def validate_probability(
    value: float,
    name: str = "probability"
) -> None:
    """Validate value is in [0, 1]."""
    if not (0 <= value <= 1):
        raise ValidationError(f"{name} must be in [0, 1], got {value}")


def validate_in_range(
    value: float,
    low: float,
    high: float,
    name: str = "value"
) -> None:
    """Validate value is in [low, high]."""
    if not (low <= value <= high):
        raise ValidationError(f"{name} must be in [{low}, {high}], got {value}")


def validate_positive_int(
    value: int,
    name: str = "value"
) -> None:
    """Validate value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be positive int, got {value}")


def validate_non_empty_sequence(
    seq: Sequence,
    name: str = "sequence"
) -> None:
    """Validate sequence is non-empty."""
    if len(seq) == 0:
        raise ValidationError(f"{name} must be non-empty")


def validate_unique_elements(
    seq: Sequence,
    name: str = "sequence"
) -> None:
    """Validate all elements in sequence are unique."""
    if len(seq) != len(set(seq)):
        raise ValidationError(f"{name} contains duplicate elements")


def validate_class_id(class_id: int) -> None:
    """Validate class_id is 0, 1, or 2."""
    if class_id not in (0, 1, 2):
        raise ValidationError(f"class_id must be 0, 1, or 2, got {class_id}")


def validate_probabilities_sum_to_one(
    probs: torch.Tensor,
    dim: int = -1,
    tol: float = 1e-5,
    name: str = "probabilities"
) -> None:
    """Validate probabilities sum to 1 along given dimension."""
    sums = probs.sum(dim=dim)
    if not torch.allclose(sums, torch.ones_like(sums), atol=tol):
        max_diff = (sums - 1.0).abs().max().item()
        raise ValidationError(
            f"{name} do not sum to 1 (max diff: {max_diff:.2e})"
        )
