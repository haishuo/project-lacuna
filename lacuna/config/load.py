"""
lacuna.config.load

Config loading and validation.
"""

import yaml
from pathlib import Path
from typing import Union, Dict, Any

from .schema import LacunaConfig, DataConfig, ModelConfig, TrainingConfig, GeneratorConfig
from ..core.exceptions import ConfigError


def load_config(path: Union[str, Path]) -> LacunaConfig:
    """Load configuration from YAML file.
    
    Args:
        path: Path to YAML config file.
    
    Returns:
        Validated LacunaConfig object.
    
    Raises:
        ConfigError: If file not found or validation fails.
    """
    path = Path(path)
    
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")
    
    return config_from_dict(raw)


def config_from_dict(d: Dict[str, Any]) -> LacunaConfig:
    """Create LacunaConfig from dictionary.
    
    Args:
        d: Dictionary with config values.
    
    Returns:
        Validated LacunaConfig object.
    """
    try:
        data = DataConfig(**d.get("data", {}))
        model = ModelConfig(**d.get("model", {}))
        training = TrainingConfig(**d.get("training", {}))
        generator = GeneratorConfig(**d.get("generator", {}))
        
        return LacunaConfig(
            data=data,
            model=model,
            training=training,
            generator=generator,
            seed=d.get("seed", 42),
            device=d.get("device", "cuda"),
            output_dir=d.get("output_dir", "/mnt/artifacts/project_lacuna/runs"),
            loss_matrix=d.get("loss_matrix", LacunaConfig().loss_matrix),
        )
    except (TypeError, ValueError) as e:
        raise ConfigError(f"Invalid config: {e}")


def save_config(config: LacunaConfig, path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: LacunaConfig to save.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    d = config_to_dict(config)
    
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False)


def config_to_dict(config: LacunaConfig) -> Dict[str, Any]:
    """Convert LacunaConfig to dictionary."""
    return {
        "data": {
            "max_cols": config.data.max_cols,
            "n_range": list(config.data.n_range),
            "d_range": list(config.data.d_range),
            "normalization": config.data.normalization,
        },
        "model": {
            "hidden_dim": config.model.hidden_dim,
            "evidence_dim": config.model.evidence_dim,
            "n_layers": config.model.n_layers,
            "n_heads": config.model.n_heads,
            "dropout": config.model.dropout,
        },
        "training": {
            "batch_size": config.training.batch_size,
            "lr": config.training.lr,
            "weight_decay": config.training.weight_decay,
            "epochs": config.training.epochs,
            "warmup_steps": config.training.warmup_steps,
            "patience": config.training.patience,
            "grad_clip": config.training.grad_clip,
        },
        "generator": {
            "n_generators": config.generator.n_generators,
            "class_balance": list(config.generator.class_balance),
        },
        "seed": config.seed,
        "device": config.device,
        "output_dir": config.output_dir,
        "loss_matrix": config.loss_matrix,
    }
