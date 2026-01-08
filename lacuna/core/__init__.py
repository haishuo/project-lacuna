"""
lacuna.core

Core infrastructure for Project Lacuna.

Exports:
- Exception classes
- RNG management
- Core data types
- Validation utilities
"""

from .exceptions import (
    LacunaError,
    ValidationError,
    ConfigError,
    RegistryError,
    CheckpointError,
    NumericalError,
)

from .rng import RNGState

from .types import (
    ObservedDataset,
    TokenBatch,
    PosteriorResult,
    Decision,
    MCAR,
    MAR,
    MNAR,
    CLASS_NAMES,
)

from .validation import (
    validate_tensor_shape,
    validate_tensor_dtype,
    validate_no_nan_inf,
    validate_positive,
    validate_non_negative,
    validate_probability,
    validate_in_range,
    validate_positive_int,
    validate_non_empty_sequence,
    validate_unique_elements,
    validate_class_id,
    validate_probabilities_sum_to_one,
)

__all__ = [
    # Exceptions
    "LacunaError",
    "ValidationError", 
    "ConfigError",
    "RegistryError",
    "CheckpointError",
    "NumericalError",
    # RNG
    "RNGState",
    # Types
    "ObservedDataset",
    "TokenBatch",
    "PosteriorResult",
    "Decision",
    "MCAR",
    "MAR",
    "MNAR",
    "CLASS_NAMES",
    # Validation
    "validate_tensor_shape",
    "validate_tensor_dtype",
    "validate_no_nan_inf",
    "validate_positive",
    "validate_non_negative",
    "validate_probability",
    "validate_in_range",
    "validate_positive_int",
    "validate_non_empty_sequence",
    "validate_unique_elements",
    "validate_class_id",
    "validate_probabilities_sum_to_one",
]
