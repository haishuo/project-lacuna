"""
lacuna.generators

Generator system for synthetic data with controlled missingness mechanisms.
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
    "Generator",
    "GeneratorParams",
    "GeneratorRegistry",
    "GeneratorPrior",
    "sample_gaussian",
    "sample_gaussian_correlated",
    "sample_uniform",
    "sample_mixed",
    "MCARUniform",
    "MCARColumnwise",
    "MARLogistic",
    "MARMultiPredictor",
    "MARMultiColumn",
    "MNARLogistic",
    "MNARSelfCensoring",
]


def create_minimal_registry() -> GeneratorRegistry:
    """Create minimal 6-generator registry for training.
    
    Returns registry with 2 generators per class:
    - MCAR: Uniform 10%, Uniform 30%
    - MAR: MultiColumn with VERY STRONG signal (alpha1=4.0 and 6.0)
    - MNAR: SelfCensoring weak, SelfCensoring strong
    
    ENHANCED: Using stronger alpha1 values (4.0 and 6.0) to create
    more distinctive MAR patterns that are easier to distinguish from MNAR.
    Also affecting more columns (50-60%) to amplify the signal.
    """
    generators = (
        # MCAR - random missingness independent of values
        MCARUniform(0, "MCAR-Uniform-10", GeneratorParams(miss_rate=0.10)),
        MCARUniform(1, "MCAR-Uniform-30", GeneratorParams(miss_rate=0.30)),
        
        # MAR - missingness depends on OBSERVED values (predictor column)
        # Using STRONGER alpha1 values for more distinctive MAR patterns
        MARMultiColumn(
            2, "MAR-MultiCol-Strong",
            GeneratorParams(
                alpha0=-0.5,     # Baseline toward observed (fewer missing)
                alpha1=4.0,      # STRONG dependence on predictor
                target_frac=0.5, # Affect 50% of columns
            )
        ),
        MARMultiColumn(
            3, "MAR-MultiCol-VeryStrong",
            GeneratorParams(
                alpha0=-0.5,     # Baseline toward observed
                alpha1=6.0,      # VERY STRONG dependence on predictor
                target_frac=0.6, # Affect 60% of columns
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