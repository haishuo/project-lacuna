"""
lacuna.training.trainer

Training loop for Lacuna models.

Supports three training modes:
    1. Pretraining: Self-supervised reconstruction only
    2. Classification: Mechanism classification only  
    3. Joint: Both reconstruction and classification

Features:
    - Learning rate warmup and scheduling
    - Gradient clipping
    - Early stopping with patience
    - Checkpointing (best and periodic)
    - Mixed precision training (optional)
    - Multi-task loss balancing
    - Comprehensive logging
    - Validation during training

Design:
    - Explicit configuration via TrainerConfig dataclass
    - Mutable state isolated in TrainerState dataclass
    - Clean separation between training logic and I/O
    - Callbacks for custom logging/monitoring
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple, Any, Union
from pathlib import Path
import time
import json
import math

from lacuna.core.types import TokenBatch, LacunaOutput
from lacuna.core.exceptions import NumericalError, CheckpointError
from lacuna.models.assembly import LacunaModel
from lacuna.training.loss import (
    LacunaLoss,
    LossConfig,
    create_loss_function,
    create_joint_loss,
    compute_class_accuracy,
    compute_mechanism_accuracy,
    compute_per_class_accuracy,
)
from lacuna.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    CheckpointData,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainerConfig:
    """Configuration for the training loop."""
    
    # === Optimization ===
    lr: float = 1e-4                    # Peak learning rate
    min_lr: float = 1e-6                # Minimum learning rate for scheduling
    weight_decay: float = 0.01          # AdamW weight decay
    betas: Tuple[float, float] = (0.9, 0.999)  # Adam betas
    eps: float = 1e-8                   # Adam epsilon
    grad_clip: float = 1.0              # Gradient clipping norm (0 = disabled)
    
    # === Schedule ===
    epochs: int = 20                    # Maximum training epochs
    warmup_steps: int = 100             # Linear warmup steps
    warmup_epochs: float = 0.0          # Alternative: warmup as fraction of epoch
    lr_schedule: str = "cosine"         # "constant", "cosine", "linear"
    
    # === Early Stopping ===
    patience: int = 5                   # Epochs without improvement before stopping
    min_delta: float = 1e-4             # Minimum improvement to count as progress
    early_stop_metric: str = "val_loss" # Metric to monitor ("val_loss" or "val_acc")
    early_stop_mode: str = "min"        # "min" for loss, "max" for accuracy
    
    # === Training Mode ===
    training_mode: str = "joint"        # "pretraining", "classification", "joint"
    
    # === Loss Configuration ===
    mechanism_weight: float = 1.0       # Weight for mechanism classification loss
    reconstruction_weight: float = 0.5  # Weight for reconstruction loss
    class_weight: float = 0.5           # Weight for class-level vs mechanism-level
    label_smoothing: float = 0.0        # Label smoothing
    load_balance_weight: float = 0.01   # MoE load balancing
    
    # === Mixed Precision ===
    use_amp: bool = False               # Use automatic mixed precision
    
    # === Logging ===
    log_every: int = 50                 # Log training metrics every N steps
    eval_every: int = 500               # Evaluate on validation set every N steps
    eval_every_epoch: bool = True       # Also evaluate at end of each epoch
    
    # === Checkpointing ===
    checkpoint_dir: Optional[str] = None  # Directory for checkpoints
    save_best_only: bool = True         # Only save best model
    save_every_epoch: bool = False      # Save checkpoint every epoch
    keep_last_n: int = 3                # Keep last N checkpoints (0 = keep all)
    
    # === Reproducibility ===
    seed: Optional[int] = None          # Random seed
    deterministic: bool = False         # Use deterministic algorithms
    
    def __post_init__(self):
        if self.training_mode not in ("pretraining", "classification", "joint"):
            raise ValueError(f"Unknown training_mode: {self.training_mode}")
        
        if self.lr_schedule not in ("constant", "cosine", "linear"):
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")
        
        if self.early_stop_mode not in ("min", "max"):
            raise ValueError(f"Unknown early_stop_mode: {self.early_stop_mode}")
    
    def get_loss_config(self) -> LossConfig:
        """Create LossConfig from trainer config."""
        # Adjust weights based on training mode
        if self.training_mode == "pretraining":
            mechanism_weight = 0.0
            reconstruction_weight = 1.0
        elif self.training_mode == "classification":
            mechanism_weight = 1.0
            reconstruction_weight = 0.0
        else:  # joint
            mechanism_weight = self.mechanism_weight
            reconstruction_weight = self.reconstruction_weight
        
        return LossConfig(
            mechanism_weight=mechanism_weight,
            reconstruction_weight=reconstruction_weight,
            class_weight=self.class_weight,
            label_smoothing=self.label_smoothing,
            load_balance_weight=self.load_balance_weight,
        )


@dataclass
class TrainerState:
    """Mutable training state."""
    
    # === Progress ===
    step: int = 0                       # Global step counter
    epoch: int = 0                      # Current epoch
    samples_seen: int = 0               # Total samples processed
    
    # === Best Metrics ===
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0
    best_epoch: int = 0
    best_step: int = 0
    
    # === Early Stopping ===
    patience_counter: int = 0
    should_stop: bool = False
    
    # === Epoch Metrics ===
    epoch_loss_sum: float = 0.0
    epoch_samples: int = 0
    epoch_correct: int = 0
    epoch_start_time: float = 0.0
    
    # === Learning Rate ===
    current_lr: float = 0.0
    
    def reset_epoch_metrics(self):
        """Reset metrics at start of epoch."""
        self.epoch_loss_sum = 0.0
        self.epoch_samples = 0
        self.epoch_correct = 0
        self.epoch_start_time = time.time()
    
    def get_epoch_metrics(self) -> Dict[str, float]:
        """Compute epoch-level metrics."""
        n = max(self.epoch_samples, 1)
        return {
            "epoch_loss": self.epoch_loss_sum / n,
            "epoch_acc": self.epoch_correct / n,
            "epoch_time": time.time() - self.epoch_start_time,
        }


# =============================================================================
# Learning Rate Scheduling
# =============================================================================

class LRScheduler:
    """
    Learning rate scheduler with warmup and decay.
    
    Supports:
        - Linear warmup
        - Cosine annealing
        - Linear decay
        - Constant (after warmup)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig,
        total_steps: int,
    ):
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        
        self.base_lr = config.lr
        self.min_lr = config.min_lr
        self.warmup_steps = config.warmup_steps
        
        # If warmup_epochs is set, compute warmup_steps
        # (will be updated when we know steps_per_epoch)
        self.warmup_epochs = config.warmup_epochs
    
    def update_warmup_steps(self, steps_per_epoch: int):
        """Update warmup steps based on epochs."""
        if self.warmup_epochs > 0:
            self.warmup_steps = int(self.warmup_epochs * steps_per_epoch)
    
    def get_lr(self, step: int) -> float:
        """Compute learning rate for given step."""
        # Warmup phase
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        
        # Post-warmup phase
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        
        if self.config.lr_schedule == "constant":
            return self.base_lr
        
        elif self.config.lr_schedule == "cosine":
            # Cosine annealing to min_lr
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        elif self.config.lr_schedule == "linear":
            # Linear decay to min_lr
            return self.base_lr - (self.base_lr - self.min_lr) * progress
        
        return self.base_lr
    
    def step(self, current_step: int):
        """Update optimizer learning rate."""
        lr = self.get_lr(current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """
    Training loop manager for Lacuna models.
    
    Handles:
        - Training loop with validation
        - Learning rate scheduling
        - Gradient clipping
        - Early stopping
        - Checkpointing
        - Logging via callbacks
    
    Attributes:
        model: LacunaModel to train.
        config: TrainerConfig with training parameters.
        device: Device for training (cuda/cpu).
        state: TrainerState with current training progress.
        loss_fn: LacunaLoss for computing losses.
        optimizer: AdamW optimizer.
        scheduler: Learning rate scheduler.
        scaler: GradScaler for mixed precision (if enabled).
    
    Example:
        >>> trainer = Trainer(model, config, device="cuda")
        >>> trainer.fit(train_loader, val_loader)
        >>> # Access best metrics
        >>> print(f"Best val acc: {trainer.state.best_val_acc}")
    """
    
    def __init__(
        self,
        model: LacunaModel,
        config: TrainerConfig,
        device: str = "cuda",
        log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: LacunaModel to train.
            config: TrainerConfig with training parameters.
            device: Device string ("cuda", "cpu", "cuda:0", etc.)
            log_fn: Optional callback for logging metrics.
        """
        self.config = config
        self.device = device
        self.log_fn = log_fn
        
        # Set random seed if specified
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
        
        # Set deterministic mode if requested
        if config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Move model to device
        self.model = model.to(device)
        
        # Initialize state
        self.state = TrainerState()
        
        # Create loss function
        self.loss_fn = LacunaLoss(config.get_loss_config())
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )
        
        # Scheduler will be initialized in fit() when we know total steps
        self.scheduler: Optional[LRScheduler] = None
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # Checkpoint directory
        if config.checkpoint_dir is not None:
            self.checkpoint_dir = Path(config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None
        
        # Track saved checkpoints for cleanup
        self.saved_checkpoints: List[Path] = []
    
    def _log(self, metrics: Dict[str, Any]):
        """Log metrics via callback if set."""
        if self.log_fn is not None:
            self.log_fn(metrics)
    
    def _to_device(self, batch: TokenBatch) -> TokenBatch:
        """Move batch to training device."""
        return batch.to(self.device)
    
    def train_step(self, batch: TokenBatch) -> Dict[str, float]:
        """
        Execute single training step.
        
        Args:
            batch: TokenBatch with training data.
        
        Returns:
            Dict with loss values and metrics.
        """
        self.model.train()
        batch = self._to_device(batch)
        
        self.optimizer.zero_grad()
        
        # Forward pass (with optional mixed precision)
        if self.config.use_amp:
            with autocast():
                output = self.model(batch)
                total_loss, loss_dict = self.loss_fn(output, batch)
        else:
            output = self.model(batch)
            total_loss, loss_dict = self.loss_fn(output, batch)
        
        # Check for NaN
        if torch.isnan(total_loss):
            raise NumericalError("NaN loss detected during training")
        
        # Backward pass
        if self.config.use_amp:
            self.scaler.scale(total_loss).backward()
            
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
            
            self.optimizer.step()
        
        # Update learning rate
        if self.scheduler is not None:
            self.state.current_lr = self.scheduler.step(self.state.step)
        
        # Compute accuracy
        if batch.class_ids is not None:
            acc = compute_class_accuracy(output.posterior.p_class, batch.class_ids)
        else:
            acc = torch.tensor(0.0)
        
        # Update state
        self.state.step += 1
        self.state.samples_seen += batch.batch_size
        self.state.epoch_loss_sum += total_loss.item() * batch.batch_size
        self.state.epoch_samples += batch.batch_size
        if batch.class_ids is not None:
            preds = output.posterior.p_class.argmax(dim=-1)
            self.state.epoch_correct += (preds == batch.class_ids).sum().item()
        
        # Return metrics
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        metrics["acc"] = acc.item()
        metrics["lr"] = self.state.current_lr
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run validation on entire validation set.
        
        Args:
            val_loader: DataLoader for validation data.
        
        Returns:
            Dict with validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        
        # Per-class tracking
        class_correct = {0: 0, 1: 0, 2: 0}
        class_total = {0: 0, 1: 0, 2: 0}
        
        # Loss component tracking
        loss_sums = {}
        
        for batch in val_loader:
            batch = self._to_device(batch)
            
            # Forward pass
            if self.config.use_amp:
                with autocast():
                    output = self.model(batch)
                    loss, loss_dict = self.loss_fn(output, batch)
            else:
                output = self.model(batch)
                loss, loss_dict = self.loss_fn(output, batch)
            
            B = batch.batch_size
            total_loss += loss.item() * B
            total_samples += B
            
            # Track loss components
            for k, v in loss_dict.items():
                val = v.item() if torch.is_tensor(v) else v
                loss_sums[k] = loss_sums.get(k, 0.0) + val * B
            
            # Accuracy
            if batch.class_ids is not None:
                preds = output.posterior.p_class.argmax(dim=-1)
                total_correct += (preds == batch.class_ids).sum().item()
                
                # Per-class accuracy
                for class_idx in range(3):
                    mask = batch.class_ids == class_idx
                    if mask.sum() > 0:
                        class_total[class_idx] += mask.sum().item()
                        class_correct[class_idx] += (preds[mask] == class_idx).sum().item()
        
        # Compute averages
        n = max(total_samples, 1)
        metrics = {
            "val_loss": total_loss / n,
            "val_acc": total_correct / n,
        }
        
        # Per-class accuracy
        class_names = ["mcar", "mar", "mnar"]
        for class_idx, class_name in enumerate(class_names):
            if class_total[class_idx] > 0:
                metrics[f"val_{class_name}_acc"] = class_correct[class_idx] / class_total[class_idx]
            else:
                metrics[f"val_{class_name}_acc"] = 0.0
        
        # Loss components
        for k, v in loss_sums.items():
            if k != "total_loss":
                metrics[f"val_{k}"] = v / n
        
        return metrics
    
    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """
        Check early stopping criteria.
        
        Args:
            val_metrics: Validation metrics from validate().
        
        Returns:
            True if training should stop, False otherwise.
        """
        metric = val_metrics.get(self.config.early_stop_metric, val_metrics["val_loss"])
        
        # Check if improved
        if self.config.early_stop_mode == "min":
            improved = metric < (self.state.best_val_loss - self.config.min_delta)
            if improved:
                self.state.best_val_loss = metric
        else:  # max
            improved = metric > (self.state.best_val_acc + self.config.min_delta)
            if improved:
                self.state.best_val_acc = metric
        
        if improved:
            self.state.patience_counter = 0
            self.state.best_epoch = self.state.epoch
            self.state.best_step = self.state.step
            return False
        else:
            self.state.patience_counter += 1
            if self.state.patience_counter >= self.config.patience:
                return True
            return False
    
    def _save_checkpoint(
        self,
        val_metrics: Dict[str, float],
        is_best: bool = False,
        force: bool = False,
    ):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        # Determine if we should save
        if self.config.save_best_only and not is_best and not force:
            return
        
        # Create checkpoint data
        checkpoint = CheckpointData(
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=None,  # We track LR in TrainerState
            step=self.state.step,
            epoch=self.state.epoch,
            best_val_loss=self.state.best_val_loss,
            best_val_acc=self.state.best_val_acc,
            config=self.config.__dict__,
            metrics=val_metrics,
        )
        
        # Save
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch{self.state.epoch}_step{self.state.step}.pt"
        
        save_checkpoint(checkpoint, path)
        
        # Track and cleanup old checkpoints
        if not is_best:
            self.saved_checkpoints.append(path)
            
            if self.config.keep_last_n > 0:
                while len(self.saved_checkpoints) > self.config.keep_last_n:
                    old_path = self.saved_checkpoints.pop(0)
                    if old_path.exists():
                        old_path.unlink()
    
    def load_checkpoint(self, path: Union[str, Path]):
        """
        Load model and training state from checkpoint.
        
        Args:
            path: Path to checkpoint file.
        """
        checkpoint = load_checkpoint(path, device=self.device)
        
        self.model.load_state_dict(checkpoint.model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        
        self.state.step = checkpoint.step
        self.state.epoch = checkpoint.epoch
        self.state.best_val_loss = checkpoint.best_val_loss
        self.state.best_val_acc = checkpoint.best_val_acc
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """
        Run full training loop.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.
            resume_from: Optional path to checkpoint to resume from.
        
        Returns:
            Dict with final/best metrics.
        """
        # Resume from checkpoint if specified
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            print(f"Resumed from checkpoint: step={self.state.step}, epoch={self.state.epoch}")
        
        # Compute total steps for scheduler
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.config.epochs
        
        # Initialize scheduler
        self.scheduler = LRScheduler(self.optimizer, self.config, total_steps)
        self.scheduler.update_warmup_steps(steps_per_epoch)
        
        # Initialize learning rate
        self.state.current_lr = self.scheduler.get_lr(self.state.step)
        
        print(f"Starting training:")
        print(f"  Mode: {self.config.training_mode}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Steps/epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps}")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.config.use_amp}")
        
        # Training loop
        start_epoch = self.state.epoch
        for epoch in range(start_epoch, self.config.epochs):
            self.state.epoch = epoch
            self.state.reset_epoch_metrics()
            
            # Epoch training loop
            for batch_idx, batch in enumerate(train_loader):
                # Training step
                step_metrics = self.train_step(batch)
                
                # Periodic logging
                if self.state.step % self.config.log_every == 0:
                    step_metrics["epoch"] = epoch
                    step_metrics["step"] = self.state.step
                    self._log(step_metrics)
                    
                    # Print progress
                    loss_str = f"loss={step_metrics['total_loss']:.4f}"
                    acc_str = f"acc={step_metrics['acc']:.3f}"
                    lr_str = f"lr={step_metrics['lr']:.2e}"
                    print(f"  Step {self.state.step}: {loss_str}, {acc_str}, {lr_str}")
                
                # Periodic validation
                if (
                    val_loader is not None
                    and self.config.eval_every > 0
                    and self.state.step % self.config.eval_every == 0
                ):
                    val_metrics = self.validate(val_loader)
                    val_metrics["step"] = self.state.step
                    self._log(val_metrics)
                    
                    print(f"  Validation: loss={val_metrics['val_loss']:.4f}, acc={val_metrics['val_acc']:.3f}")
                    
                    # Check for improvement and save checkpoint
                    is_best = self._is_best(val_metrics)
                    if is_best:
                        self._save_checkpoint(val_metrics, is_best=True)
                    
                    # Early stopping check
                    if self._check_early_stopping(val_metrics):
                        print(f"Early stopping at step {self.state.step}")
                        self.state.should_stop = True
                        break
            
            # End of epoch
            epoch_metrics = self.state.get_epoch_metrics()
            print(f"Epoch {epoch} complete: loss={epoch_metrics['epoch_loss']:.4f}, "
                  f"acc={epoch_metrics['epoch_acc']:.3f}, time={epoch_metrics['epoch_time']:.1f}s")
            
            # End-of-epoch validation
            if val_loader is not None and self.config.eval_every_epoch:
                val_metrics = self.validate(val_loader)
                val_metrics["epoch"] = epoch
                self._log(val_metrics)
                
                print(f"  Epoch validation: loss={val_metrics['val_loss']:.4f}, acc={val_metrics['val_acc']:.3f}")
                
                # Check for improvement
                is_best = self._is_best(val_metrics)
                if is_best:
                    self._save_checkpoint(val_metrics, is_best=True)
                    print(f"  New best model! (val_acc={val_metrics['val_acc']:.4f})")
                
                # Early stopping
                if self._check_early_stopping(val_metrics):
                    print(f"Early stopping at epoch {epoch}")
                    self.state.should_stop = True
            
            # Periodic checkpoint
            if self.config.save_every_epoch:
                self._save_checkpoint(val_metrics if val_loader else {}, force=True)
            
            # Check if should stop
            if self.state.should_stop:
                break
        
        # Final summary
        print(f"\nTraining complete!")
        print(f"  Best val_loss: {self.state.best_val_loss:.4f} (epoch {self.state.best_epoch})")
        print(f"  Best val_acc: {self.state.best_val_acc:.4f}")
        
        return {
            "best_val_loss": self.state.best_val_loss,
            "best_val_acc": self.state.best_val_acc,
            "best_epoch": self.state.best_epoch,
            "final_epoch": self.state.epoch,
            "total_steps": self.state.step,
        }
    
    def _is_best(self, val_metrics: Dict[str, float]) -> bool:
        """Check if current metrics are best so far."""
        if self.config.early_stop_mode == "min":
            metric = val_metrics.get(self.config.early_stop_metric, val_metrics["val_loss"])
            return metric < self.state.best_val_loss
        else:
            metric = val_metrics.get(self.config.early_stop_metric, val_metrics["val_acc"])
            return metric > self.state.best_val_acc
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: DataLoader for test data.
        
        Returns:
            Dict with test metrics.
        """
        metrics = self.validate(test_loader)
        # Rename val_ to test_
        return {k.replace("val_", "test_"): v for k, v in metrics.items()}


# =============================================================================
# Convenience Functions
# =============================================================================

def train_lacuna(
    model: LacunaModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 20,
    lr: float = 1e-4,
    device: str = "cuda",
    training_mode: str = "joint",
    checkpoint_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[Trainer, Dict[str, float]]:
    """
    Convenience function to train a Lacuna model.
    
    Args:
        model: LacunaModel to train.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Device string.
        training_mode: "pretraining", "classification", or "joint".
        checkpoint_dir: Optional checkpoint directory.
        **kwargs: Additional TrainerConfig parameters.
    
    Returns:
        trainer: Trained Trainer instance.
        results: Dict with training results.
    
    Example:
        >>> trainer, results = train_lacuna(model, train_loader, val_loader)
        >>> print(f"Best accuracy: {results['best_val_acc']}")
    """
    config = TrainerConfig(
        epochs=epochs,
        lr=lr,
        training_mode=training_mode,
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )
    
    trainer = Trainer(model, config, device=device)
    results = trainer.fit(train_loader, val_loader)
    
    return trainer, results


def pretrain_lacuna(
    model: LacunaModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cuda",
    **kwargs,
) -> Tuple[Trainer, Dict[str, float]]:
    """
    Pretrain Lacuna model with reconstruction objective only.
    
    Args:
        model: LacunaModel to pretrain.
        train_loader: DataLoader for training data (with artificial masking).
        val_loader: Optional DataLoader for validation.
        epochs: Number of pretraining epochs.
        lr: Learning rate.
        device: Device string.
        **kwargs: Additional TrainerConfig parameters.
    
    Returns:
        trainer: Trainer instance.
        results: Dict with pretraining results.
    """
    return train_lacuna(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        training_mode="pretraining",
        **kwargs,
    )


def finetune_lacuna(
    model: LacunaModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 20,
    lr: float = 1e-5,  # Lower LR for finetuning
    device: str = "cuda",
    **kwargs,
) -> Tuple[Trainer, Dict[str, float]]:
    """
    Finetune pretrained Lacuna model for classification.
    
    Args:
        model: Pretrained LacunaModel.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation.
        epochs: Number of finetuning epochs.
        lr: Learning rate (typically lower than pretraining).
        device: Device string.
        **kwargs: Additional TrainerConfig parameters.
    
    Returns:
        trainer: Trainer instance.
        results: Dict with finetuning results.
    """
    return train_lacuna(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        training_mode="classification",
        **kwargs,
    )