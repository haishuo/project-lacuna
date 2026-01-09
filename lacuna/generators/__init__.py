"""
lacuna.generators

Generator system for synthetic data with controlled missingness mechanisms.

This module provides:
- Abstract Generator base class
- Concrete generator implementations (MCAR, MAR, MNAR)
- Registry for managing generator sets
- Prior distributions over generators
"""

from .base import Generator
from .params import GeneratorParams
from .registry import GeneratorRegistry
from .priors import GeneratorPrior

from .families import (
    # Base data
    sample_gaussian,
    sample_gaussian_correlated,
    sample_uniform,
    sample_mixed,
    # MCAR
    MCARUniform,
    MCARColumnwise,
    # MAR
    MARLogistic,
    MARMultiPredictor,
    # MNAR
    MNARLogistic,
    MNARSelfCensoring,
)

__all__ = [
    # Core classes
    "Generator",
    "GeneratorParams",
    "GeneratorRegistry",
    "GeneratorPrior",
    # Base data
    "sample_gaussian",
    "sample_gaussian_correlated",
    "sample_uniform",
    "sample_mixed",
    # MCAR
    "MCARUniform",
    "MCARColumnwise",
    # MAR
    "MARLogistic",
    "MARMultiPredictor",
    # MNAR
    "MNARLogistic",
    "MNARSelfCensoring",
]


def create_minimal_registry() -> GeneratorRegistry:
    """Create minimal 6-generator registry for testing.
    
    Returns registry with 2 generators per class:
    - MCAR: Uniform 10%, Uniform 30%
    - MAR: Logistic weak, Logistic strong
    - MNAR: Logistic weak, Logistic strong
    """
    generators = (
        # MCAR
        MCARUniform(0, "MCAR-Uniform-10", GeneratorParams(miss_rate=0.10)),
        MCARUniform(1, "MCAR-Uniform-30", GeneratorParams(miss_rate=0.30)),
        # MAR
        MARLogistic(2, "MAR-Logistic-Weak", GeneratorParams(alpha0=0.0, alpha1=1.0)),
        MARLogistic(3, "MAR-Logistic-Strong", GeneratorParams(alpha0=0.0, alpha1=3.0)),
        # MNAR
        MNARLogistic(4, "MNAR-Logistic-Weak", GeneratorParams(beta0=0.0, beta1=0.0, beta2=1.0)),
        MNARLogistic(5, "MNAR-Logistic-Strong", GeneratorParams(beta0=0.0, beta1=0.0, beta2=3.0)),
    )
    return GeneratorRegistry(generators)
