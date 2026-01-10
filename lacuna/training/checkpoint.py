"""
lacuna.training.checkpoint

Checkpoint management for training state persistence.

This module provides:
    - CheckpointData: Structured checkpoint with model, optimizer, and metadata
    - save_checkpoint, load_checkpoint: Basic checkpoint I/O
    - load_model_weights: Convenience function for weight loading
    - CheckpointManager: Multi-checkpoint management with cleanup
    - Utilities: validation, comparison, export, hashing

Note on PyTorch 2.6+ Compatibility:
    PyTorch 2.6 changed the default for torch.load to weights_only=True,
    which only allows loading tensor data. Since our checkpoints include
    metadata (step, epoch, config dicts, etc.), we always use weights_only=False.
    This is safe because we only load checkpoints we created ourselves.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from lacuna.core.exceptions import CheckpointError

# Version for checkpoint compatibility checking
__version__ = "0.1.0"


# =============================================================================
# CheckpointData
# =============================================================================

@dataclass
class CheckpointData:
    """
    Structured checkpoint data.

    Contains all state needed to resume training or perform inference:
        - Model weights (required)
        - Optimizer state (optional, for resuming training)
        - Scheduler state (optional)
        - Training progress (step, epoch)
        - Best metrics (for model selection)
        - Configuration (for reproducibility)

    Attributes:
        model_state: Model state dict (required).
        optimizer_state: Optimizer state dict (optional).
        scheduler_state: LR scheduler state dict (optional).
        step: Global training step.
        epoch: Current epoch.
        best_val_loss: Best validation loss seen.
        best_val_acc: Best validation accuracy seen.
        config: Training configuration dict.
        model_config: Model configuration dict.
        metrics: Additional metrics dict.
        timestamp: ISO format timestamp of checkpoint creation.
        lacuna_version: Lacuna version string.

    Example:
        >>> checkpoint = CheckpointData(
        ...     model_state=model.state_dict(),
        ...     optimizer_state=optimizer.state_dict(),
        ...     step=1000,
        ...     epoch=5,
        ... )
        >>> save_checkpoint(checkpoint, "checkpoints/epoch5.pt")
    """

    # Required
    model_state: Dict[str, torch.Tensor]

    # Optional training state
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None

    # Progress tracking
    step: int = 0
    epoch: int = 0

    # Best metrics
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0

    # Configuration
    config: Optional[Dict[str, Any]] = None
    model_config: Optional[Dict[str, Any]] = None

    # Metrics
    metrics: Optional[Dict[str, Any]] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    lacuna_version: str = __version__

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "step": self.step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "config": self.config,
            "model_config": self.model_config,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "lacuna_version": self.lacuna_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointData":
        """Create from dictionary."""
        return cls(
            model_state=data["model_state"],
            optimizer_state=data.get("optimizer_state"),
            scheduler_state=data.get("scheduler_state"),
            step=data.get("step", 0),
            epoch=data.get("epoch", 0),
            best_val_loss=data.get("best_val_loss", float("inf")),
            best_val_acc=data.get("best_val_acc", 0.0),
            config=data.get("config"),
            model_config=data.get("model_config"),
            metrics=data.get("metrics"),
            timestamp=data.get("timestamp", ""),
            lacuna_version=data.get("lacuna_version", "unknown"),
        )


# =============================================================================
# Save and Load Functions
# =============================================================================

def save_checkpoint(
    checkpoint: CheckpointData,
    path: Union[str, Path],
    include_optimizer: bool = True,
) -> Path:
    """
    Save checkpoint to disk.

    Args:
        checkpoint: CheckpointData to save.
        path: Path to save checkpoint to.
        include_optimizer: Whether to include optimizer state (can be large).

    Returns:
        Path where checkpoint was saved.

    Raises:
        CheckpointError: If save fails.

    Example:
        >>> checkpoint = CheckpointData(model_state=model.state_dict())
        >>> save_checkpoint(checkpoint, "checkpoints/best.pt")
    """
    path = Path(path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build save dict
    save_dict = checkpoint.to_dict()

    # Optionally exclude optimizer state (can be large)
    if not include_optimizer:
        save_dict["optimizer_state"] = None

    try:
        # Save with pickle protocol 4 for better compatibility
        torch.save(save_dict, path, pickle_protocol=4)
    except Exception as e:
        raise CheckpointError(f"Failed to save checkpoint to {path}: {e}")

    return path


def load_checkpoint(
    path: Union[str, Path],
    device: Optional[str] = None,
) -> CheckpointData:
    """
    Load checkpoint from disk.

    Args:
        path: Path to checkpoint file.
        device: Device to map tensors to (e.g., "cuda", "cpu").
                If None, uses default device.

    Returns:
        CheckpointData with loaded state.

    Raises:
        CheckpointError: If load fails or checkpoint is invalid.

    Note:
        Always uses weights_only=False because our checkpoint format
        includes non-tensor metadata (dicts, strings, floats).
        This is safe for checkpoints we create ourselves.

    Example:
        >>> checkpoint = load_checkpoint("checkpoints/best.pt", device="cuda")
        >>> model.load_state_dict(checkpoint.model_state)
    """
    path = Path(path)

    if not path.exists():
        raise CheckpointError(f"Checkpoint not found: {path}")

    try:
        # Set up device mapping
        if device is not None:
            map_location = torch.device(device)
        else:
            map_location = None

        # Load checkpoint with weights_only=False
        # Required because our checkpoint contains non-tensor metadata
        # (step, epoch, config dicts, timestamp strings, etc.)
        data = torch.load(path, map_location=map_location, weights_only=False)

    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint from {path}: {e}")

    # Validate checkpoint structure
    if "model_state" not in data:
        raise CheckpointError(f"Invalid checkpoint: missing 'model_state' key")

    return CheckpointData.from_dict(data)


def load_model_weights(
    model: torch.nn.Module,
    path: Union[str, Path],
    device: Optional[str] = None,
    strict: bool = True,
) -> torch.nn.Module:
    """
    Load model weights from checkpoint.

    Convenience function that handles the common case of just loading
    weights into a model.

    Args:
        model: Model to load weights into.
        path: Path to checkpoint file.
        device: Device to map tensors to.
        strict: Whether to require exact key matching.

    Returns:
        Model with loaded weights (same instance, modified in-place).

    Raises:
        CheckpointError: If load fails.

    Example:
        >>> model = create_lacuna_model(config)
        >>> model = load_model_weights(model, "checkpoints/best.pt", device="cuda")
    """
    checkpoint = load_checkpoint(path, device=device)

    try:
        model.load_state_dict(checkpoint.model_state, strict=strict)
    except RuntimeError as e:
        if strict:
            raise CheckpointError(
                f"Failed to load model weights (strict mode): {e}\n"
                f"Try setting strict=False to allow partial loading."
            )
        else:
            # Log warning about missing/unexpected keys
            result = model.load_state_dict(checkpoint.model_state, strict=False)
            if result.missing_keys:
                print(f"Warning: Missing keys in checkpoint: {result.missing_keys[:5]}...")
            if result.unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {result.unexpected_keys[:5]}...")

    return model


# =============================================================================
# Checkpoint Validation
# =============================================================================

def validate_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate checkpoint file and return metadata.

    Args:
        path: Path to checkpoint file.

    Returns:
        Dict with checkpoint metadata and validation status.

    Raises:
        CheckpointError: If checkpoint is invalid.
    """
    path = Path(path)

    if not path.exists():
        raise CheckpointError(f"Checkpoint not found: {path}")

    try:
        # Load without device mapping for validation
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint: {e}")

    # Check required fields
    if "model_state" not in data:
        raise CheckpointError("Missing required field: model_state")

    # Count parameters
    num_params = sum(
        v.numel() for v in data["model_state"].values()
        if isinstance(v, torch.Tensor)
    )

    return {
        "valid": True,
        "path": str(path),
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        "num_parameters": num_params,
        "step": data.get("step", 0),
        "epoch": data.get("epoch", 0),
        "best_val_loss": data.get("best_val_loss", float("inf")),
        "best_val_acc": data.get("best_val_acc", 0.0),
        "has_optimizer": data.get("optimizer_state") is not None,
        "has_scheduler": data.get("scheduler_state") is not None,
        "timestamp": data.get("timestamp", "unknown"),
        "lacuna_version": data.get("lacuna_version", "unknown"),
    }


# =============================================================================
# Checkpoint Management
# =============================================================================

class CheckpointManager:
    """
    Manages multiple checkpoints with automatic cleanup.

    Features:
        - Tracks best and periodic checkpoints
        - Automatic cleanup of old checkpoints
        - Checkpoint metadata and listing

    Attributes:
        checkpoint_dir: Directory for checkpoints.
        keep_best: Number of best checkpoints to keep.
        keep_last: Number of recent checkpoints to keep.

    Example:
        >>> manager = CheckpointManager("checkpoints/", keep_best=1, keep_last=3)
        >>> manager.save(checkpoint, is_best=True)
        >>> best = manager.load_best()
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        keep_best: int = 1,
        keep_last: int = 3,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints.
            keep_best: Number of best checkpoints to keep.
            keep_last: Number of recent periodic checkpoints to keep.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_best = keep_best
        self.keep_last = keep_last

        # Track saved checkpoints
        self.best_checkpoints: List[Path] = []
        self.periodic_checkpoints: List[Path] = []

        # Load existing checkpoints
        self._scan_existing()

    def _scan_existing(self):
        """Scan directory for existing checkpoints."""
        for path in self.checkpoint_dir.glob("best_*.pt"):
            self.best_checkpoints.append(path)
        for path in self.checkpoint_dir.glob("checkpoint_*.pt"):
            self.periodic_checkpoints.append(path)

        # Sort by modification time
        self.best_checkpoints.sort(key=lambda p: p.stat().st_mtime)
        self.periodic_checkpoints.sort(key=lambda p: p.stat().st_mtime)

    def save(
        self,
        checkpoint: CheckpointData,
        is_best: bool = False,
    ) -> Path:
        """
        Save checkpoint with automatic cleanup.

        Args:
            checkpoint: CheckpointData to save.
            is_best: Whether this is the best model so far.

        Returns:
            Path to saved checkpoint.
        """
        if is_best:
            # Save as best checkpoint
            filename = f"best_epoch{checkpoint.epoch}_step{checkpoint.step}.pt"
            path = self.checkpoint_dir / filename
            save_checkpoint(checkpoint, path)

            self.best_checkpoints.append(path)
            self._cleanup_best()
        else:
            # Save as periodic checkpoint
            filename = f"checkpoint_epoch{checkpoint.epoch}_step{checkpoint.step}.pt"
            path = self.checkpoint_dir / filename
            save_checkpoint(checkpoint, path)

            self.periodic_checkpoints.append(path)
            self._cleanup_periodic()

        return path

    def _cleanup_best(self):
        """Remove old best checkpoints."""
        while len(self.best_checkpoints) > self.keep_best:
            old_path = self.best_checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()

    def _cleanup_periodic(self):
        """Remove old periodic checkpoints."""
        while len(self.periodic_checkpoints) > self.keep_last:
            old_path = self.periodic_checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()

    def load_best(self) -> CheckpointData:
        """
        Load the best checkpoint.

        Returns:
            CheckpointData from best checkpoint.

        Raises:
            CheckpointError: If no best checkpoint exists.
        """
        if not self.best_checkpoints:
            raise CheckpointError("No best checkpoint found")

        # Load most recent best
        return load_checkpoint(self.best_checkpoints[-1])

    def load_latest(self) -> CheckpointData:
        """
        Load the most recent checkpoint (best or periodic).

        Returns:
            CheckpointData from latest checkpoint.

        Raises:
            CheckpointError: If no checkpoints exist.
        """
        all_checkpoints = self.best_checkpoints + self.periodic_checkpoints

        if not all_checkpoints:
            raise CheckpointError("No checkpoints found")

        # Find most recent by modification time
        latest = max(all_checkpoints, key=lambda p: p.stat().st_mtime)
        return load_checkpoint(latest)

    def list_checkpoints(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all available checkpoints with metadata.

        Returns:
            Dict with 'best' and 'periodic' lists of checkpoint info.
        """
        result = {"best": [], "periodic": []}

        for path in self.best_checkpoints:
            if path.exists():
                result["best"].append({
                    "path": str(path),
                    "name": path.name,
                    "size_mb": path.stat().st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                })

        for path in self.periodic_checkpoints:
            if path.exists():
                result["periodic"].append({
                    "path": str(path),
                    "name": path.name,
                    "size_mb": path.stat().st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                })

        return result


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_checkpoints(
    path1: Union[str, Path],
    path2: Union[str, Path],
) -> Dict[str, Any]:
    """
    Compare two checkpoints for differences.

    Args:
        path1: Path to first checkpoint.
        path2: Path to second checkpoint.

    Returns:
        Dict with comparison results including:
            - checkpoint1/checkpoint2: metadata for each
            - keys_only_in_1/2: keys unique to each checkpoint
            - common_keys: number of shared keys
            - differing_weights: number of keys with different values
            - weight_diffs: details of weight differences
    """
    cp1 = load_checkpoint(path1)
    cp2 = load_checkpoint(path2)

    keys1 = set(cp1.model_state.keys())
    keys2 = set(cp2.model_state.keys())

    # Compare weights
    weight_diffs = {}
    common_keys = keys1 & keys2

    for key in common_keys:
        w1 = cp1.model_state[key]
        w2 = cp2.model_state[key]

        if w1.shape != w2.shape:
            weight_diffs[key] = {
                "type": "shape_mismatch",
                "shape1": tuple(w1.shape),
                "shape2": tuple(w2.shape),
            }
        else:
            diff = (w1.float() - w2.float()).abs().mean().item()
            if diff > 1e-6:
                weight_diffs[key] = {"type": "value_diff", "mean_diff": diff}

    return {
        "checkpoint1": {
            "step": cp1.step,
            "epoch": cp1.epoch,
            "best_val_loss": cp1.best_val_loss,
        },
        "checkpoint2": {
            "step": cp2.step,
            "epoch": cp2.epoch,
            "best_val_loss": cp2.best_val_loss,
        },
        "keys_only_in_1": list(keys1 - keys2),
        "keys_only_in_2": list(keys2 - keys1),
        "common_keys": len(common_keys),
        "differing_weights": len(weight_diffs),
        "weight_diffs": weight_diffs,
    }


# =============================================================================
# Export Utilities
# =============================================================================

def export_for_inference(
    checkpoint_path: Union[str, Path],
    output_path: Union[str, Path],
    device: str = "cpu",
) -> Path:
    """
    Export checkpoint for inference (model weights only).

    Creates a minimal checkpoint without optimizer state for
    deployment or sharing. Reduces file size significantly.

    Args:
        checkpoint_path: Path to full checkpoint.
        output_path: Path for inference checkpoint.
        device: Device to map tensors to.

    Returns:
        Path to exported checkpoint.
    """
    checkpoint = load_checkpoint(checkpoint_path, device=device)

    # Create minimal checkpoint without optimizer
    inference_checkpoint = CheckpointData(
        model_state=checkpoint.model_state,
        step=checkpoint.step,
        epoch=checkpoint.epoch,
        best_val_loss=checkpoint.best_val_loss,
        best_val_acc=checkpoint.best_val_acc,
        model_config=checkpoint.model_config,
        metrics=checkpoint.metrics,
    )

    return save_checkpoint(inference_checkpoint, output_path, include_optimizer=False)


def compute_checkpoint_hash(path: Union[str, Path]) -> str:
    """
    Compute SHA256 hash of checkpoint file for integrity verification.

    Args:
        path: Path to checkpoint file.

    Returns:
        SHA256 hash as hex string (64 characters).

    Example:
        >>> hash1 = compute_checkpoint_hash("model_v1.pt")
        >>> hash2 = compute_checkpoint_hash("model_v2.pt")
        >>> if hash1 == hash2:
        ...     print("Checkpoints are identical")
    """
    path = Path(path)

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in chunks for memory efficiency with large files
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()