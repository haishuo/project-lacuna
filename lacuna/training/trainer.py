"""
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
