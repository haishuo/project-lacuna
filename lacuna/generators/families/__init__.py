"""
lacuna.generators.families

Generator family implementations.

Families:
- MCAR: Missing Completely At Random
- MAR: Missing At Random
- MNAR: Missing Not At Random
"""

from .base_data import (
    sample_gaussian,
    sample_gaussian_correlated,
    sample_uniform,
    sample_mixed,
)

from .mcar import (
    MCARUniform,
    MCARColumnwise,
)

from .mar import (
    MARLogistic,
    MARMultiPredictor,
)

from .mnar import (
    MNARLogistic,
    MNARSelfCensoring,
)

__all__ = [
    # Base data samplers
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
