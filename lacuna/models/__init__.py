"""
lacuna.models

Model architectures for Project Lacuna.

Components:
- EvidenceEncoder: Transformer for evidence extraction
- GeneratorHead: Classifier for generator prediction
- Aggregator: Generator -> class posterior mapping
- Decision: Bayes-optimal decision rule
- LacunaModel: Full model assembly
"""

from .encoder import EvidenceEncoder
from .heads import GeneratorHead, ClassHead
from .aggregator import (
    aggregate_to_class_posterior,
    aggregate_to_class_posterior_efficient,
    compute_entropy,
    compute_confidence,
    get_predicted_class,
)
from .decision import (
    DEFAULT_LOSS_MATRIX,
    compute_expected_loss,
    bayes_optimal_decision,
    make_decision,
    interpret_decision,
)
from .assembly import LacunaModel

__all__ = [
    # Encoder
    "EvidenceEncoder",
    # Heads
    "GeneratorHead",
    "ClassHead",
    # Aggregator
    "aggregate_to_class_posterior",
    "aggregate_to_class_posterior_efficient",
    "compute_entropy",
    "compute_confidence",
    "get_predicted_class",
    # Decision
    "DEFAULT_LOSS_MATRIX",
    "compute_expected_loss",
    "bayes_optimal_decision",
    "make_decision",
    "interpret_decision",
    # Full model
    "LacunaModel",
]
