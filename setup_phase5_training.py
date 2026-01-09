#!/usr/bin/env python3
"""
Phase 5: Training Infrastructure

Creates:
- lacuna/training/loss.py - Loss functions (CE on generators, optional class auxiliary)
- lacuna/training/trainer.py - Training loop with validation
- lacuna/training/checkpoint.py - Save/load with validation
- tests/unit/training/test_loss.py
- tests/unit/training/test_trainer.py
- tests/unit/training/test_checkpoint.py

Run: python setup_phase5_training.py
Then: pytest tests/unit/training/ -v
"""

from pathlib import Path

def write_file(path: str, content: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    print(f"  Created: {path}")

# =============================================================================
# lacuna/training/loss.py
# =============================================================================
LOSS_PY = '''"""
lacuna.training.loss

Loss functions for Lacuna training.

Primary loss: Cross-entropy on generator classification.
Auxiliary loss: Cross-entropy on class posterior (optional regularization).
"""

import torch
import torch.nn.functional as F
from typing import Optional

from lacuna.core.types import PosteriorResult


def generator_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy loss on generator logits.
    
    Args:
        logits: [B, K] raw generator logits.
        targets: [B] generator IDs (long tensor).
        reduction: "mean", "sum", or "none".
    
    Returns:
        Loss tensor (scalar if reduction != "none").
    """
    return F.cross_entropy(logits, targets, reduction=reduction)


def class_cross_entropy(
    p_class: torch.Tensor,
    class_targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy on class posterior (auxiliary loss).
    
    Note: Uses log of probabilities, not logits.
    
    Args:
        p_class: [B, 3] class posterior probabilities.
        class_targets: [B] class IDs (0=MCAR, 1=MAR, 2=MNAR).
        reduction: "mean", "sum", or "none".
    
    Returns:
        Loss tensor.
    """
    # Add epsilon for numerical stability
    log_p = torch.log(p_class + 1e-10)
    return F.nll_loss(log_p, class_targets, reduction=reduction)


def combined_loss(
    posterior: PosteriorResult,
    generator_targets: torch.Tensor,
    class_targets: torch.Tensor,
    class_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined loss with optional class auxiliary term.
    
    Total loss = CE(generator) + class_weight * CE(class)
    
    Args:
        posterior: Model output containing logits and p_class.
        generator_targets: [B] generator IDs.
        class_targets: [B] class IDs.
        class_weight: Weight for auxiliary class loss (0 = disabled).
    
    Returns:
        total_loss: Combined loss tensor.
        metrics: Dict with individual loss components.
    """
    gen_loss = generator_cross_entropy(posterior.logits_generator, generator_targets)
    
    metrics = {"loss_generator": gen_loss.item()}
    
    if class_weight > 0:
        cls_loss = class_cross_entropy(posterior.p_class, class_targets)
        total_loss = gen_loss + class_weight * cls_loss
        metrics["loss_class"] = cls_loss.item()
        metrics["loss_total"] = total_loss.item()
    else:
        total_loss = gen_loss
        metrics["loss_total"] = gen_loss.item()
    
    return total_loss, metrics


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute classification accuracy.
    
    Args:
        logits: [B, K] prediction logits.
        targets: [B] ground truth labels.
    
    Returns:
        Accuracy as float in [0, 1].
    """
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def compute_topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 3,
) -> float:
    """Compute top-k accuracy.
    
    Args:
        logits: [B, K] prediction logits.
        targets: [B] ground truth labels.
        k: Number of top predictions to consider.
    
    Returns:
        Top-k accuracy as float in [0, 1].
    """
    B, K = logits.shape
    k = min(k, K)
    
    _, topk_preds = logits.topk(k, dim=-1)  # [B, k]
    correct = (topk_preds == targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()
'''

# =============================================================================
# lacuna/training/trainer.py
# =============================================================================
TRAINER_PY = '''"""
lacuna.training.trainer

Training loop for Lacuna models.

Design:
- Explicit configuration (no hidden defaults)
- Validation at each epoch
- Early stopping with patience
- Gradient clipping
- Learning rate scheduling
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional, Callable
import time

from lacuna.core.types import TokenBatch, PosteriorResult
from lacuna.core.exceptions import NumericalError
from lacuna.core.validation import validate_no_nan_inf
from lacuna.models.assembly import LacunaModel
from lacuna.training.loss import (
    generator_cross_entropy,
    combined_loss,
    compute_accuracy,
)
from lacuna.training.checkpoint import save_checkpoint, CheckpointData


@dataclass
class TrainerConfig:
    """Training configuration."""
    
    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # Schedule
    epochs: int = 20
    warmup_steps: int = 100
    
    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4
    
    # Auxiliary loss
    class_loss_weight: float = 0.0
    
    # Logging
    log_every: int = 50
    eval_every: int = 500
    
    # Checkpointing
    checkpoint_dir: Optional[str] = None
    save_best_only: bool = True


@dataclass
class TrainerState:
    """Mutable training state."""
    
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0
    patience_counter: int = 0
    should_stop: bool = False
    
    # Running metrics for current epoch
    train_loss_sum: float = 0.0
    train_correct: int = 0
    train_total: int = 0


class Trainer:
    """Training loop manager.
    
    Usage:
        trainer = Trainer(model, config, device="cuda")
        trainer.fit(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: LacunaModel,
        config: TrainerConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        self.state = TrainerState()
        self._log_fn: Optional[Callable] = None
    
    def set_logger(self, log_fn: Callable[[dict], None]):
        """Set logging callback."""
        self._log_fn = log_fn
    
    def _log(self, metrics: dict):
        """Log metrics if logger is set."""
        if self._log_fn is not None:
            self._log_fn(metrics)
    
    def _get_lr_scale(self) -> float:
        """Compute learning rate scale for warmup."""
        if self.state.step < self.config.warmup_steps:
            return (self.state.step + 1) / self.config.warmup_steps
        return 1.0
    
    def _apply_lr_scale(self):
        """Apply learning rate scaling."""
        scale = self._get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.config.lr * scale
    
    def train_step(self, batch: TokenBatch) -> dict:
        """Execute single training step.
        
        Args:
            batch: TokenBatch with generator_ids and class_ids.
        
        Returns:
            Dict with loss and accuracy metrics.
        """
        self.model.train()
        batch = batch.to(self.device)
        
        # Apply LR warmup
        self._apply_lr_scale()
        
        # Forward
        self.optimizer.zero_grad()
        posterior = self.model(batch)
        
        # Check for NaN
        try:
            validate_no_nan_inf(posterior.logits_generator, "logits")
        except Exception as e:
            raise NumericalError(f"NaN in forward pass at step {self.state.step}: {e}")
        
        # Loss
        loss, metrics = combined_loss(
            posterior,
            batch.generator_ids,
            batch.class_ids,
            class_weight=self.config.class_loss_weight,
        )
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )
            metrics["grad_norm"] = grad_norm.item()
        
        # Step
        self.optimizer.step()
        self.state.step += 1
        
        # Accuracy
        metrics["acc_generator"] = compute_accuracy(
            posterior.logits_generator,
            batch.generator_ids,
        )
        metrics["acc_class"] = compute_accuracy(
            posterior.p_class,
            batch.class_ids,
        )
        
        # Update running stats
        B = batch.batch_size
        self.state.train_loss_sum += metrics["loss_total"] * B
        self.state.train_correct += int(metrics["acc_generator"] * B)
        self.state.train_total += B
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """Run validation.
        
        Args:
            val_loader: Validation data loader.
        
        Returns:
            Dict with validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct_gen = 0
        total_correct_cls = 0
        total_samples = 0
        
        for batch in val_loader:
            batch = batch.to(self.device)
            posterior = self.model(batch)
            
            loss = generator_cross_entropy(
                posterior.logits_generator,
                batch.generator_ids,
            )
            
            B = batch.batch_size
            total_loss += loss.item() * B
            total_correct_gen += int(compute_accuracy(
                posterior.logits_generator,
                batch.generator_ids,
            ) * B)
            total_correct_cls += int(compute_accuracy(
                posterior.p_class,
                batch.class_ids,
            ) * B)
            total_samples += B
        
        return {
            "val_loss": total_loss / total_samples,
            "val_acc_generator": total_correct_gen / total_samples,
            "val_acc_class": total_correct_cls / total_samples,
        }
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check and update early stopping state.
        
        Returns:
            True if should stop training.
        """
        if val_loss < self.state.best_val_loss - self.config.min_delta:
            self.state.best_val_loss = val_loss
            self.state.patience_counter = 0
            return False
        else:
            self.state.patience_counter += 1
            if self.state.patience_counter >= self.config.patience:
                return True
            return False
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> dict:
        """Run full training loop.
        
        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
        
        Returns:
            Dict with final training statistics.
        """
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            self.state.train_loss_sum = 0.0
            self.state.train_correct = 0
            self.state.train_total = 0
            
            for batch in train_loader:
                metrics = self.train_step(batch)
                
                # Log periodically
                if self.state.step % self.config.log_every == 0:
                    metrics["step"] = self.state.step
                    metrics["epoch"] = epoch
                    metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                    self._log(metrics)
            
            # Epoch summary
            epoch_loss = self.state.train_loss_sum / max(self.state.train_total, 1)
            epoch_acc = self.state.train_correct / max(self.state.train_total, 1)
            
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_acc": epoch_acc,
            }
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                epoch_metrics.update(val_metrics)
                
                # Early stopping check
                if self._check_early_stopping(val_metrics["val_loss"]):
                    self.state.should_stop = True
                    epoch_metrics["early_stop"] = True
                
                # Save best model
                if val_metrics["val_acc_generator"] > self.state.best_val_acc:
                    self.state.best_val_acc = val_metrics["val_acc_generator"]
                    if self.config.checkpoint_dir:
                        self._save_checkpoint("best")
            
            self._log(epoch_metrics)
            
            if self.state.should_stop:
                break
        
        # Final checkpoint
        if self.config.checkpoint_dir and not self.config.save_best_only:
            self._save_checkpoint("final")
        
        total_time = time.time() - start_time
        
        return {
            "epochs_completed": self.state.epoch + 1,
            "total_steps": self.state.step,
            "best_val_loss": self.state.best_val_loss,
            "best_val_acc": self.state.best_val_acc,
            "training_time": total_time,
        }
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        if self.config.checkpoint_dir is None:
            return
        
        from pathlib import Path
        path = Path(self.config.checkpoint_dir) / f"{name}.pt"
        
        data = CheckpointData(
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            step=self.state.step,
            epoch=self.state.epoch,
            best_val_loss=self.state.best_val_loss,
            config=None,  # Could add config serialization
        )
        
        save_checkpoint(data, path)
'''

# =============================================================================
# lacuna/training/checkpoint.py
# =============================================================================
CHECKPOINT_PY = '''"""
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
'''

# =============================================================================
# lacuna/training/__init__.py (update)
# =============================================================================
TRAINING_INIT_PY = '''"""
lacuna.training

Training infrastructure for Project Lacuna.

Exports:
- Loss functions
- Trainer
- Checkpointing
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
]
'''

# =============================================================================
# tests/unit/training/__init__.py
# =============================================================================
TEST_TRAINING_INIT = '''"""Tests for lacuna.training"""
'''

# =============================================================================
# tests/unit/training/test_loss.py
# =============================================================================
TEST_LOSS_PY = '''"""
Tests for lacuna.training.loss

Verify loss functions behave correctly.
"""

import pytest
import torch
from lacuna.training.loss import (
    generator_cross_entropy,
    class_cross_entropy,
    combined_loss,
    compute_accuracy,
    compute_topk_accuracy,
)
from lacuna.core.types import PosteriorResult


class TestGeneratorCrossEntropy:
    """Tests for generator_cross_entropy."""
    
    def test_perfect_prediction_low_loss(self):
        B, K = 4, 6
        targets = torch.tensor([0, 1, 2, 3])
        
        # Strong logits for correct classes
        logits = torch.zeros(B, K)
        for i, t in enumerate(targets):
            logits[i, t] = 10.0
        
        loss = generator_cross_entropy(logits, targets)
        assert loss.item() < 0.01
    
    def test_uniform_prediction_high_loss(self):
        B, K = 4, 6
        targets = torch.tensor([0, 1, 2, 3])
        logits = torch.zeros(B, K)  # Uniform
        
        loss = generator_cross_entropy(logits, targets)
        # Should be close to log(K)
        expected = torch.log(torch.tensor(float(K)))
        assert abs(loss.item() - expected.item()) < 0.1
    
    def test_reduction_none(self):
        B, K = 4, 6
        logits = torch.randn(B, K)
        targets = torch.randint(0, K, (B,))
        
        loss = generator_cross_entropy(logits, targets, reduction="none")
        assert loss.shape == (B,)
    
    def test_reduction_sum(self):
        B, K = 4, 6
        logits = torch.randn(B, K)
        targets = torch.randint(0, K, (B,))
        
        loss_mean = generator_cross_entropy(logits, targets, reduction="mean")
        loss_sum = generator_cross_entropy(logits, targets, reduction="sum")
        
        assert abs(loss_sum.item() - loss_mean.item() * B) < 1e-5


class TestClassCrossEntropy:
    """Tests for class_cross_entropy."""
    
    def test_correct_class_low_loss(self):
        B = 4
        targets = torch.tensor([0, 1, 2, 0])
        
        # High probability for correct class
        p_class = torch.zeros(B, 3)
        for i, t in enumerate(targets):
            p_class[i, t] = 0.99
            p_class[i, (t + 1) % 3] = 0.005
            p_class[i, (t + 2) % 3] = 0.005
        
        loss = class_cross_entropy(p_class, targets)
        assert loss.item() < 0.02
    
    def test_uniform_high_loss(self):
        B = 4
        targets = torch.tensor([0, 1, 2, 0])
        p_class = torch.ones(B, 3) / 3
        
        loss = class_cross_entropy(p_class, targets)
        expected = torch.log(torch.tensor(3.0))
        assert abs(loss.item() - expected.item()) < 0.1


class TestCombinedLoss:
    """Tests for combined_loss."""
    
    def test_no_class_weight(self):
        B, K = 4, 6
        posterior = PosteriorResult(
            p_generator=torch.softmax(torch.randn(B, K), dim=-1),
            p_class=torch.softmax(torch.randn(B, 3), dim=-1),
            entropy_generator=torch.rand(B),
            entropy_class=torch.rand(B),
            logits_generator=torch.randn(B, K),
        )
        gen_targets = torch.randint(0, K, (B,))
        cls_targets = torch.randint(0, 3, (B,))
        
        loss, metrics = combined_loss(posterior, gen_targets, cls_targets, class_weight=0.0)
        
        assert "loss_generator" in metrics
        assert "loss_class" not in metrics
        assert abs(loss.item() - metrics["loss_generator"]) < 1e-5
    
    def test_with_class_weight(self):
        B, K = 4, 6
        posterior = PosteriorResult(
            p_generator=torch.softmax(torch.randn(B, K), dim=-1),
            p_class=torch.softmax(torch.randn(B, 3), dim=-1),
            entropy_generator=torch.rand(B),
            entropy_class=torch.rand(B),
            logits_generator=torch.randn(B, K),
        )
        gen_targets = torch.randint(0, K, (B,))
        cls_targets = torch.randint(0, 3, (B,))
        
        loss, metrics = combined_loss(posterior, gen_targets, cls_targets, class_weight=0.5)
        
        assert "loss_generator" in metrics
        assert "loss_class" in metrics
        expected = metrics["loss_generator"] + 0.5 * metrics["loss_class"]
        assert abs(loss.item() - expected) < 1e-5


class TestComputeAccuracy:
    """Tests for compute_accuracy."""
    
    def test_perfect_accuracy(self):
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        targets = torch.tensor([0, 1, 2])
        
        acc = compute_accuracy(logits, targets)
        assert acc == 1.0
    
    def test_zero_accuracy(self):
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        targets = torch.tensor([1, 2, 0])  # All wrong
        
        acc = compute_accuracy(logits, targets)
        assert acc == 0.0
    
    def test_partial_accuracy(self):
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
        targets = torch.tensor([0, 0, 0, 1])  # 2 correct, 2 wrong
        
        acc = compute_accuracy(logits, targets)
        assert acc == 0.5


class TestTopkAccuracy:
    """Tests for compute_topk_accuracy."""
    
    def test_top1_equals_accuracy(self):
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        
        top1 = compute_topk_accuracy(logits, targets, k=1)
        acc = compute_accuracy(logits, targets)
        
        assert abs(top1 - acc) < 1e-6
    
    def test_topk_geq_top1(self):
        logits = torch.randn(100, 10)
        targets = torch.randint(0, 10, (100,))
        
        top1 = compute_topk_accuracy(logits, targets, k=1)
        top3 = compute_topk_accuracy(logits, targets, k=3)
        top5 = compute_topk_accuracy(logits, targets, k=5)
        
        assert top3 >= top1
        assert top5 >= top3
    
    def test_topk_equals_k(self):
        # If k >= K, accuracy should be 1.0
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        
        top10 = compute_topk_accuracy(logits, targets, k=10)
        assert top10 == 1.0
'''

# =============================================================================
# tests/unit/training/test_trainer.py
# =============================================================================
TEST_TRAINER_PY = '''"""
Tests for lacuna.training.trainer

Verify training loop behavior.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lacuna.training.trainer import Trainer, TrainerConfig
from lacuna.models.assembly import LacunaModel
from lacuna.core.types import TokenBatch
from lacuna.config.schema import LacunaConfig


def make_dummy_model():
    """Create small model for testing."""
    cfg = LacunaConfig.minimal()
    K = 6
    class_mapping = torch.tensor([0, 0, 1, 1, 2, 2])
    
    return LacunaModel.from_config(cfg, K, class_mapping)


def make_dummy_dataloader(n_batches: int = 5, batch_size: int = 8):
    """Create dummy data loader."""
    max_cols = 16
    token_dim = 12
    K = 6
    
    batches = []
    for _ in range(n_batches):
        gen_ids = torch.randint(0, K, (batch_size,))
        cls_ids = torch.div(gen_ids, 2, rounding_mode="floor")  # 0,1->0, 2,3->1, 4,5->2
        
        batch = TokenBatch(
            tokens=torch.randn(batch_size, max_cols, token_dim),
            col_mask=torch.ones(batch_size, max_cols, dtype=torch.bool),
            generator_ids=gen_ids,
            class_ids=cls_ids,
        )
        batches.append(batch)
    
    return batches


class DummyLoader:
    """Simple loader that iterates over batches."""
    def __init__(self, batches):
        self.batches = batches
    
    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)


class TestTrainerConfig:
    """Tests for TrainerConfig."""
    
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.lr == 1e-4
        assert cfg.epochs == 20
        assert cfg.patience == 5


class TestTrainerStep:
    """Tests for single training step."""
    
    def test_train_step_reduces_loss(self):
        model = make_dummy_model()
        config = TrainerConfig(lr=0.01)
        trainer = Trainer(model, config, device="cpu")
        
        batches = make_dummy_dataloader(n_batches=1)
        batch = batches[0]
        
        # Get initial loss
        model.eval()
        with torch.no_grad():
            initial_loss = torch.nn.functional.cross_entropy(
                model(batch).logits_generator,
                batch.generator_ids,
            ).item()
        
        # Train for a few steps
        for _ in range(10):
            trainer.train_step(batch)
        
        # Check loss decreased
        model.eval()
        with torch.no_grad():
            final_loss = torch.nn.functional.cross_entropy(
                model(batch).logits_generator,
                batch.generator_ids,
            ).item()
        
        assert final_loss < initial_loss
    
    def test_train_step_returns_metrics(self):
        model = make_dummy_model()
        config = TrainerConfig()
        trainer = Trainer(model, config, device="cpu")
        
        batches = make_dummy_dataloader(n_batches=1)
        metrics = trainer.train_step(batches[0])
        
        assert "loss_total" in metrics
        assert "acc_generator" in metrics
        assert "acc_class" in metrics
    
    def test_gradient_clipping(self):
        model = make_dummy_model()
        config = TrainerConfig(grad_clip=0.1)
        trainer = Trainer(model, config, device="cpu")
        
        batches = make_dummy_dataloader(n_batches=1)
        metrics = trainer.train_step(batches[0])
        
        assert "grad_norm" in metrics
        # Grad norm should be clipped
        assert metrics["grad_norm"] <= 0.1 + 1e-6


class TestTrainerValidation:
    """Tests for validation."""
    
    def test_validate_returns_metrics(self):
        model = make_dummy_model()
        config = TrainerConfig()
        trainer = Trainer(model, config, device="cpu")
        
        val_batches = make_dummy_dataloader(n_batches=3)
        val_loader = DummyLoader(val_batches)
        
        metrics = trainer.validate(val_loader)
        
        assert "val_loss" in metrics
        assert "val_acc_generator" in metrics
        assert "val_acc_class" in metrics


class TestTrainerFit:
    """Tests for full training loop."""
    
    def test_fit_runs(self):
        model = make_dummy_model()
        config = TrainerConfig(epochs=2)
        trainer = Trainer(model, config, device="cpu")
        
        train_batches = make_dummy_dataloader(n_batches=5)
        train_loader = DummyLoader(train_batches)
        
        result = trainer.fit(train_loader)
        
        assert result["epochs_completed"] == 2
        assert result["total_steps"] == 10  # 5 batches * 2 epochs
    
    def test_fit_with_validation(self):
        model = make_dummy_model()
        config = TrainerConfig(epochs=2)
        trainer = Trainer(model, config, device="cpu")
        
        train_batches = make_dummy_dataloader(n_batches=5)
        val_batches = make_dummy_dataloader(n_batches=2)
        
        result = trainer.fit(
            DummyLoader(train_batches),
            DummyLoader(val_batches),
        )
        
        assert "best_val_loss" in result
        assert "best_val_acc" in result
    
    def test_early_stopping(self):
        model = make_dummy_model()
        # Small patience, training on random data won\'t improve
        config = TrainerConfig(epochs=100, patience=2)
        trainer = Trainer(model, config, device="cpu")
        
        train_batches = make_dummy_dataloader(n_batches=2)
        val_batches = make_dummy_dataloader(n_batches=2)
        
        result = trainer.fit(
            DummyLoader(train_batches),
            DummyLoader(val_batches),
        )
        
        # Should stop well before 100 epochs
        assert result["epochs_completed"] < 100


class TestLRWarmup:
    """Tests for learning rate warmup."""
    
    def test_warmup_starts_low(self):
        model = make_dummy_model()
        config = TrainerConfig(lr=1e-3, warmup_steps=100)
        trainer = Trainer(model, config, device="cpu")
        
        # At step 0, LR should be 1/100 of target
        initial_scale = trainer._get_lr_scale()
        assert initial_scale == 0.01
    
    def test_warmup_reaches_full(self):
        model = make_dummy_model()
        config = TrainerConfig(lr=1e-3, warmup_steps=100)
        trainer = Trainer(model, config, device="cpu")
        
        # Simulate 100 steps
        trainer.state.step = 100
        final_scale = trainer._get_lr_scale()
        assert final_scale == 1.0
'''

# =============================================================================
# tests/unit/training/test_checkpoint.py
# =============================================================================
TEST_CHECKPOINT_PY = '''"""
Tests for lacuna.training.checkpoint

Verify checkpoint save/load.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from lacuna.training.checkpoint import (
    CheckpointData,
    save_checkpoint,
    load_checkpoint,
    load_model_from_checkpoint,
    load_trainer_from_checkpoint,
    CHECKPOINT_VERSION,
)
from lacuna.core.exceptions import CheckpointError
from lacuna.models.assembly import LacunaModel
from lacuna.config.schema import LacunaConfig


def make_dummy_model():
    """Create small model for testing."""
    cfg = LacunaConfig.minimal()
    K = 6
    class_mapping = torch.tensor([0, 0, 1, 1, 2, 2])
    return LacunaModel.from_config(cfg, K, class_mapping)


class TestCheckpointData:
    """Tests for CheckpointData."""
    
    def test_construction(self):
        data = CheckpointData(
            model_state={"layer.weight": torch.randn(10, 10)},
            step=100,
            epoch=5,
        )
        assert data.step == 100
        assert data.epoch == 5


class TestSaveLoadCheckpoint:
    """Tests for save/load roundtrip."""
    
    def test_save_load_roundtrip(self):
        model = make_dummy_model()
        
        data = CheckpointData(
            model_state=model.state_dict(),
            step=42,
            epoch=3,
            best_val_loss=0.5,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            
            loaded = load_checkpoint(path)
        
        assert loaded.step == 42
        assert loaded.epoch == 3
        assert loaded.best_val_loss == 0.5
        
        # Check model state matches
        for key in data.model_state:
            assert torch.allclose(
                data.model_state[key],
                loaded.model_state[key],
            )
    
    def test_load_nonexistent_raises(self):
        with pytest.raises(CheckpointError, match="not found"):
            load_checkpoint(Path("/nonexistent/path.pt"))
    
    def test_save_creates_directory(self):
        model = make_dummy_model()
        data = CheckpointData(model_state=model.state_dict())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "deep" / "test.pt"
            save_checkpoint(data, path)
            
            assert path.exists()


class TestLoadModel:
    """Tests for load_model_from_checkpoint."""
    
    def test_load_restores_weights(self):
        model1 = make_dummy_model()
        model2 = make_dummy_model()
        
        # Models should start different (random init)
        param1 = next(model1.parameters())
        param2 = next(model2.parameters())
        assert not torch.allclose(param1, param2)
        
        # Save model1
        data = CheckpointData(model_state=model1.state_dict())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            
            # Load into model2
            load_model_from_checkpoint(model2, path)
        
        # Now they should match
        param2_after = next(model2.parameters())
        assert torch.allclose(param1, param2_after)


class TestLoadTrainer:
    """Tests for load_trainer_from_checkpoint."""
    
    def test_load_restores_optimizer(self):
        model = make_dummy_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Take a step to modify optimizer state
        dummy_loss = sum(p.sum() for p in model.parameters())
        dummy_loss.backward()
        optimizer.step()
        
        data = CheckpointData(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            step=10,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            
            # Create fresh model and optimizer
            model2 = make_dummy_model()
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            
            loaded = load_trainer_from_checkpoint(model2, optimizer2, path)
        
        assert loaded.step == 10
        # Optimizer state should be restored
        assert len(optimizer2.state) > 0
'''

# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("Phase 5: Training Infrastructure")
    print("=" * 60)
    
    # Implementation files
    write_file("lacuna/training/loss.py", LOSS_PY)
    write_file("lacuna/training/trainer.py", TRAINER_PY)
    write_file("lacuna/training/checkpoint.py", CHECKPOINT_PY)
    write_file("lacuna/training/__init__.py", TRAINING_INIT_PY)
    
    # Test files
    write_file("tests/unit/training/__init__.py", TEST_TRAINING_INIT)
    write_file("tests/unit/training/test_loss.py", TEST_LOSS_PY)
    write_file("tests/unit/training/test_trainer.py", TEST_TRAINER_PY)
    write_file("tests/unit/training/test_checkpoint.py", TEST_CHECKPOINT_PY)
    
    print()
    print("=" * 60)
    print("Phase 5 Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run tests: pytest tests/unit/training/ -v")
    print("  2. Delete this script: rm setup_phase5_training.py")
    print()
    print("Files created:")
    print("  - lacuna/training/loss.py")
    print("  - lacuna/training/trainer.py")
    print("  - lacuna/training/checkpoint.py")
    print("  - lacuna/training/__init__.py (updated)")
    print("  - tests/unit/training/test_loss.py")
    print("  - tests/unit/training/test_trainer.py")
    print("  - tests/unit/training/test_checkpoint.py")


if __name__ == "__main__":
    main()