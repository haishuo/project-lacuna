"""
lacuna.config

Configuration management for Project Lacuna.

Exports:
- Config schemas
- Loading/saving utilities
- Hashing for reproducibility
"""

from .schema import (
    LacunaConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    GeneratorConfig,
)

from .load import (
    load_config,
    save_config,
    config_from_dict,
    config_to_dict,
)

from .hashing import (
    hash_config,
    hash_dict,
    config_signature,
)

__all__ = [
    # Schemas
    "LacunaConfig",
    "DataConfig",
    "ModelConfig", 
    "TrainingConfig",
    "GeneratorConfig",
    # Load/save
    "load_config",
    "save_config",
    "config_from_dict",
    "config_to_dict",
    # Hashing
    "hash_config",
    "hash_dict",
    "config_signature",
]
