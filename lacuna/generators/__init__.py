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
from .families.mar import MARLogistic, MARMultiPredictor, MARMultiColumn

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
    """Create minimal 6-generator registry for training.
    
    Returns registry with 2 generators per class:
    - MCAR: Uniform 10%, Uniform 30%
    - MAR: MultiColumn weak, MultiColumn strong (affects ~30% of columns)
    - MNAR: SelfCensoring weak, SelfCensoring strong
    
    Using MARMultiColumn instead of MARLogistic creates a much stronger
    MAR signal by affecting multiple columns with the same predictor-based
    missingness pattern.
    """
    generators = (
        # MCAR - random missingness independent of values
        MCARUniform(0, "MCAR-Uniform-10", GeneratorParams(miss_rate=0.10)),
        MCARUniform(1, "MCAR-Uniform-30", GeneratorParams(miss_rate=0.30)),
        
        # MAR - missingness depends on OBSERVED values (predictor column)
        # Using MARMultiColumn to affect multiple columns for stronger signal
        MARMultiColumn(
            2, "MAR-MultiCol-Weak",
            GeneratorParams(
                alpha0=0.0,      # Baseline ~50% missing
                alpha1=1.5,      # Moderate dependence on predictor
                target_frac=0.3, # Affect 30% of columns
            )
        ),
        MARMultiColumn(
            3, "MAR-MultiCol-Strong",
            GeneratorParams(
                alpha0=0.0,      # Baseline ~50% missing  
                alpha1=3.0,      # Strong dependence on predictor
                target_frac=0.4, # Affect 40% of columns
            )
        ),
        
        # MNAR - missingness depends on the UNOBSERVED value itself
        MNARSelfCensoring(
            4, "MNAR-SelfCensor-Weak",
            GeneratorParams(
                beta0=-0.5,       # Slight baseline toward observed
                beta1=1.5,        # Moderate self-censoring
                affected_frac=0.4 # Affect 40% of columns
            )
        ),
        MNARSelfCensoring(
            5, "MNAR-SelfCensor-Strong",
            GeneratorParams(
                beta0=-0.5,       # Slight baseline toward observed
                beta1=3.0,        # Strong self-censoring
                affected_frac=0.5 # Affect 50% of columns
            )
        ),
    )
    return GeneratorRegistry(generators)