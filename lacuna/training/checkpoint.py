"""
lacuna.training.checkpoint

Checkpoint management for saving and loading model state.

Features:
    - Save complete training state (model, optimizer, scheduler, metrics)
    - Load checkpoints with device mapping
    - Checkpoint validation and integrity checking
    - Support for best-model and periodic checkpoints
    - Metadata tracking for reproducibility
"""

import torch
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Union
from datetime import datetime
import json
import hashlib

from lacuna.core.exceptions import CheckpointError


# =============================================================================
# Checkpoint Data Structure
# =============================================================================

@dataclass
class CheckpointData:
    """
    Complete checkpoint data structure.
    
    Contains all information needed to resume training or load
    a trained model for inference.
    
    Attributes:
        model_state: Model state dict from model.state_dict().
        optimizer_state: Optimizer state dict (optional, for resume).
        scheduler_state: LR scheduler state dict (optional).
        step: Global training step.
        epoch: Current epoch.
        best_val_loss: Best validation loss seen.
        best_val_acc: Best validation accuracy seen.
        config: Training configuration dict.
        metrics: Latest metrics dict.
        model_config: Model architecture configuration (optional).
        timestamp: When checkpoint was created.
        lacuna_version: Version string for compatibility.
    """
    
    # Required fields
    model_state: Dict[str, torch.Tensor]
    
    # Training state (optional for inference-only checkpoints)
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    
    # Progress tracking
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0
    
    # Configuration
    config: Optional[Dict[str, Any]] = None
    model_config: Optional[Dict[str, Any]] = None
    
    # Metrics
    metrics: Optional[Dict[str, float]] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    lacuna_version: str = "2.0.0"
    
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
        include_optimizer: Whether to include optimizer state (large).
    
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
    weights_only: bool = False,
) -> CheckpointData:
    """
    Load checkpoint from disk.
    
    Args:
        path: Path to checkpoint file.
        device: Device to map tensors to (e.g., "cuda", "cpu").
                If None, uses default device.
        weights_only: If True, only load model weights (safer, recommended
                     for untrusted checkpoints).
    
    Returns:
        CheckpointData with loaded state.
    
    Raises:
        CheckpointError: If load fails or checkpoint is invalid.
    
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
        
        # Load checkpoint
        if weights_only:
            # Safer loading that doesn't execute arbitrary code
            data = torch.load(path, map_location=map_location, weights_only=True)
        else:
            data = torch.load(path, map_location=map_location)
        
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
        Model with loaded weights.
    
    Raises:
        CheckpointError: If load fails.
    
    Example:
        >>> model = create_lacuna_model(config)
        >>> model = load_model_weights(model, "checkpoints/best.pt", device="cuda")
    """
    checkpoint = load_checkpoint(path, device=device, weights_only=True)
    
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
            missing, unexpected = model.load_state_dict(
                checkpoint.model_state, strict=False
            )
            if missing:
                print(f"Warning: Missing keys in checkpoint: {missing[:5]}...")
            if unexpected:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected[:5]}...")
    
    return model


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
        self.best_checkpoints: list[Path] = []
        self.periodic_checkpoints: list[Path] = []
        
        # Load existing checkpoints
        self._scan_existing()
    
    def _scan_existing(self):
        """Scan directory for existing checkpoints."""
        for path in self.checkpoint_dir.glob("*.pt"):
            if "best" in path.name:
                self.best_checkpoints.append(path)
            else:
                self.periodic_checkpoints.append(path)
        
        # Sort by modification time
        self.best_checkpoints.sort(key=lambda p: p.stat().st_mtime)
        self.periodic_checkpoints.sort(key=lambda p: p.stat().st_mtime)
    
    def save(
        self,
        checkpoint: CheckpointData,
        is_best: bool = False,
        name: Optional[str] = None,
    ) -> Path:
        """
        Save checkpoint with automatic naming and cleanup.
        
        Args:
            checkpoint: CheckpointData to save.
            is_best: Whether this is a new best checkpoint.
            name: Optional custom name (without extension).
        
        Returns:
            Path where checkpoint was saved.
        """
        if name is not None:
            filename = f"{name}.pt"
        elif is_best:
            filename = f"best_epoch{checkpoint.epoch}_step{checkpoint.step}.pt"
        else:
            filename = f"checkpoint_epoch{checkpoint.epoch}_step{checkpoint.step}.pt"
        
        path = self.checkpoint_dir / filename
        save_checkpoint(checkpoint, path)
        
        # Track and cleanup
        if is_best:
            self.best_checkpoints.append(path)
            self._cleanup_best()
        else:
            self.periodic_checkpoints.append(path)
            self._cleanup_periodic()
        
        return path
    
    def _cleanup_best(self):
        """Remove old best checkpoints beyond keep_best limit."""
        while len(self.best_checkpoints) > self.keep_best:
            old_path = self.best_checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
    
    def _cleanup_periodic(self):
        """Remove old periodic checkpoints beyond keep_last limit."""
        while len(self.periodic_checkpoints) > self.keep_last:
            old_path = self.periodic_checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
    
    def load_best(self, device: Optional[str] = None) -> CheckpointData:
        """
        Load the best checkpoint.
        
        Args:
            device: Device to map tensors to.
        
        Returns:
            CheckpointData from best checkpoint.
        
        Raises:
            CheckpointError: If no best checkpoint exists.
        """
        if not self.best_checkpoints:
            raise CheckpointError("No best checkpoint found")
        
        return load_checkpoint(self.best_checkpoints[-1], device=device)
    
    def load_latest(self, device: Optional[str] = None) -> CheckpointData:
        """
        Load the most recent checkpoint (best or periodic).
        
        Args:
            device: Device to map tensors to.
        
        Returns:
            CheckpointData from latest checkpoint.
        
        Raises:
            CheckpointError: If no checkpoint exists.
        """
        all_checkpoints = self.best_checkpoints + self.periodic_checkpoints
        
        if not all_checkpoints:
            raise CheckpointError("No checkpoints found")
        
        # Sort by modification time and get latest
        latest = max(all_checkpoints, key=lambda p: p.stat().st_mtime)
        return load_checkpoint(latest, device=device)
    
    def list_checkpoints(self) -> Dict[str, list[Dict[str, Any]]]:
        """
        List all available checkpoints with metadata.
        
        Returns:
            Dict with "best" and "periodic" lists of checkpoint info.
        """
        def get_info(path: Path) -> Dict[str, Any]:
            stat = path.stat()
            return {
                "path": str(path),
                "name": path.name,
                "size_mb": stat.st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        
        return {
            "best": [get_info(p) for p in self.best_checkpoints],
            "periodic": [get_info(p) for p in self.periodic_checkpoints],
        }
    
    def get_best_path(self) -> Optional[Path]:
        """Get path to best checkpoint, or None if none exists."""
        if self.best_checkpoints:
            return self.best_checkpoints[-1]
        return None
    
    def get_latest_path(self) -> Optional[Path]:
        """Get path to latest checkpoint, or None if none exists."""
        all_checkpoints = self.best_checkpoints + self.periodic_checkpoints
        if not all_checkpoints:
            return None
        return max(all_checkpoints, key=lambda p: p.stat().st_mtime)


# =============================================================================
# Checkpoint Comparison and Validation
# =============================================================================

def validate_checkpoint(
    path: Union[str, Path],
    expected_keys: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Validate checkpoint file and return metadata.
    
    Args:
        path: Path to checkpoint file.
        expected_keys: Optional list of required model state keys.
    
    Returns:
        Dict with checkpoint metadata and validation results.
    
    Raises:
        CheckpointError: If checkpoint is invalid.
    """
    path = Path(path)
    
    if not path.exists():
        raise CheckpointError(f"Checkpoint not found: {path}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(path)
    
    # Basic validation
    info = {
        "path": str(path),
        "valid": True,
        "step": checkpoint.step,
        "epoch": checkpoint.epoch,
        "best_val_loss": checkpoint.best_val_loss,
        "best_val_acc": checkpoint.best_val_acc,
        "timestamp": checkpoint.timestamp,
        "version": checkpoint.lacuna_version,
        "has_optimizer": checkpoint.optimizer_state is not None,
        "has_scheduler": checkpoint.scheduler_state is not None,
        "has_config": checkpoint.config is not None,
        "num_model_keys": len(checkpoint.model_state),
    }
    
    # Check expected keys
    if expected_keys is not None:
        model_keys = set(checkpoint.model_state.keys())
        expected_set = set(expected_keys)
        
        info["missing_keys"] = list(expected_set - model_keys)
        info["extra_keys"] = list(model_keys - expected_set)
        
        if info["missing_keys"]:
            info["valid"] = False
    
    return info


def compare_checkpoints(
    path1: Union[str, Path],
    path2: Union[str, Path],
) -> Dict[str, Any]:
    """
    Compare two checkpoints.
    
    Args:
        path1: Path to first checkpoint.
        path2: Path to second checkpoint.
    
    Returns:
        Dict with comparison results.
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
            weight_diffs[key] = {"type": "shape_mismatch", "shape1": w1.shape, "shape2": w2.shape}
        else:
            diff = (w1 - w2).abs().mean().item()
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
    deployment or sharing.
    
    Args:
        checkpoint_path: Path to full checkpoint.
        output_path: Path for inference checkpoint.
        device: Device to map tensors to.
    
    Returns:
        Path to exported checkpoint.
    """
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    
    # Create minimal checkpoint
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
    Compute hash of checkpoint for integrity verification.
    
    Args:
        path: Path to checkpoint file.
    
    Returns:
        SHA256 hash of checkpoint file.
    """
    path = Path(path)
    
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    
    return sha256.hexdigest()