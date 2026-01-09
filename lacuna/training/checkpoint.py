"""
lacuna.training.checkpoint

Checkpoint saving and loading with validation.

Design:
- Explicit data structure (CheckpointData)
- Validation on load
- Version compatibility checking
"""

import torch
from dataclasses import dataclass
from typing import Any, Optional, Dict
from pathlib import Path

from lacuna.core.exceptions import CheckpointError


CHECKPOINT_VERSION = "1.0"


@dataclass
class CheckpointData:
    """Checkpoint contents."""
    
    model_state: Dict[str, torch.Tensor]
    optimizer_state: Optional[Dict[str, Any]] = None
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


def save_checkpoint(data: CheckpointData, path: Path) -> None:
    """Save checkpoint to disk.
    
    Args:
        data: CheckpointData to save.
        path: Output path.
    
    Raises:
        CheckpointError: If save fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "model_state": data.model_state,
        "optimizer_state": data.optimizer_state,
        "step": data.step,
        "epoch": data.epoch,
        "best_val_loss": data.best_val_loss,
        "config": data.config,
        "metadata": data.metadata,
    }
    
    try:
        # Save to temp file first, then rename (atomic)
        temp_path = path.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        temp_path.rename(path)
    except Exception as e:
        raise CheckpointError(f"Failed to save checkpoint to {path}: {e}")


def load_checkpoint(
    path: Path,
    map_location: Optional[str] = None,
) -> CheckpointData:
    """Load checkpoint from disk.
    
    Args:
        path: Checkpoint file path.
        map_location: Device to map tensors to (e.g., "cpu", "cuda").
    
    Returns:
        CheckpointData with loaded state.
    
    Raises:
        CheckpointError: If load fails or checkpoint is invalid.
    """
    path = Path(path)
    
    if not path.exists():
        raise CheckpointError(f"Checkpoint not found: {path}")
    
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint from {path}: {e}")
    
    # Version check
    version = checkpoint.get("version", "unknown")
    if version != CHECKPOINT_VERSION:
        # For now, just warn - could add migration logic
        import warnings
        warnings.warn(
            f"Checkpoint version mismatch: got {version}, expected {CHECKPOINT_VERSION}"
        )
    
    # Validate required fields
    if "model_state" not in checkpoint:
        raise CheckpointError("Checkpoint missing model_state")
    
    return CheckpointData(
        model_state=checkpoint["model_state"],
        optimizer_state=checkpoint.get("optimizer_state"),
        step=checkpoint.get("step", 0),
        epoch=checkpoint.get("epoch", 0),
        best_val_loss=checkpoint.get("best_val_loss", float("inf")),
        config=checkpoint.get("config"),
        metadata=checkpoint.get("metadata"),
    )


def load_model_from_checkpoint(
    model: torch.nn.Module,
    path: Path,
    strict: bool = True,
    map_location: Optional[str] = None,
) -> CheckpointData:
    """Load model weights from checkpoint.
    
    Args:
        model: Model to load weights into.
        path: Checkpoint file path.
        strict: Whether to require exact key match.
        map_location: Device mapping.
    
    Returns:
        CheckpointData (for accessing metadata).
    
    Raises:
        CheckpointError: If load fails.
    """
    data = load_checkpoint(path, map_location)
    
    try:
        model.load_state_dict(data.model_state, strict=strict)
    except Exception as e:
        raise CheckpointError(f"Failed to load model state: {e}")
    
    return data


def load_trainer_from_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    map_location: Optional[str] = None,
) -> CheckpointData:
    """Load full training state from checkpoint.
    
    Args:
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        path: Checkpoint file path.
        map_location: Device mapping.
    
    Returns:
        CheckpointData (for accessing step, epoch, etc.).
    
    Raises:
        CheckpointError: If load fails.
    """
    data = load_checkpoint(path, map_location)
    
    try:
        model.load_state_dict(data.model_state, strict=True)
    except Exception as e:
        raise CheckpointError(f"Failed to load model state: {e}")
    
    if data.optimizer_state is not None:
        try:
            optimizer.load_state_dict(data.optimizer_state)
        except Exception as e:
            raise CheckpointError(f"Failed to load optimizer state: {e}")
    
    return data
