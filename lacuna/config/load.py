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
    """Load configuration from YAML file."""
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
    """Create LacunaConfig from dictionary."""
    try:
        data_dict = d.get("data", {})
        
        # Convert lists to tuples for dataclass
        if "n_range" in data_dict and isinstance(data_dict["n_range"], list):
            data_dict["n_range"] = tuple(data_dict["n_range"])
        if "d_range" in data_dict and isinstance(data_dict["d_range"], list):
            data_dict["d_range"] = tuple(data_dict["d_range"])
        
        # Handle optional dataset lists (for semi-synthetic)
        # Keep as lists if present, None otherwise
        if "train_datasets" in data_dict and data_dict["train_datasets"] is None:
            del data_dict["train_datasets"]
        if "val_datasets" in data_dict and data_dict["val_datasets"] is None:
            del data_dict["val_datasets"]
        
        data = DataConfig(**data_dict)
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
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    d = config_to_dict(config)
    
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False)


def config_to_dict(config: LacunaConfig) -> Dict[str, Any]:
    """Convert LacunaConfig to dictionary."""
    data_dict = {
        "n_range": list(config.data.n_range),
        "d_range": list(config.data.d_range),
        "max_cols": config.data.max_cols,
        "max_rows": config.data.max_rows,
    }
    
    # Include dataset lists if present
    if config.data.train_datasets is not None:
        data_dict["train_datasets"] = config.data.train_datasets
    if config.data.val_datasets is not None:
        data_dict["val_datasets"] = config.data.val_datasets
    
    return {
        "data": data_dict,
        "model": {
            "hidden_dim": config.model.hidden_dim,
            "evidence_dim": config.model.evidence_dim,
            "n_layers": config.model.n_layers,
            "n_heads": config.model.n_heads,
            "dropout": config.model.dropout,
        },
        "training": {
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "batches_per_epoch": config.training.batches_per_epoch,
            "val_batches": config.training.val_batches,
            "lr": config.training.lr,
            "weight_decay": config.training.weight_decay,
            "grad_clip": config.training.grad_clip,
            "warmup_steps": config.training.warmup_steps,
            "patience": config.training.patience,
            "min_delta": config.training.min_delta,
        },
        "generator": {
            "n_generators": config.generator.n_generators,
        },
        "seed": config.seed,
        "device": config.device,
        "output_dir": config.output_dir,
        "loss_matrix": config.loss_matrix,
    }