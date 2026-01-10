"""
lacuna.training

Training infrastructure for Project Lacuna.

Exports:
- Loss functions
- Trainer
- Checkpointing
- Logging
"""

from .loss import (
    generator_cross_entropy,
    class_cross_entropy,
    combined_loss,
    compute_accuracy,
    compute_topk_accuracy,
)

from .trainer import (
    Trainer,
    TrainerConfig,
    TrainerState,
)

from .checkpoint import (
    CheckpointData,
    save_checkpoint,
    load_checkpoint,
    load_model_from_checkpoint,
    load_trainer_from_checkpoint,
    CHECKPOINT_VERSION,
)

from .logging import (
    create_logger,
)

__all__ = [
    # Loss
    "generator_cross_entropy",
    "class_cross_entropy",
    "combined_loss",
    "compute_accuracy",
    "compute_topk_accuracy",
    # Trainer
    "Trainer",
    "TrainerConfig",
    "TrainerState",
    # Checkpoint
    "CheckpointData",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_from_checkpoint",
    "load_trainer_from_checkpoint",
    "CHECKPOINT_VERSION",
    # Logging
    "create_logger",
]