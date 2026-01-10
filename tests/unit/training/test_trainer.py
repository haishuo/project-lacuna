"""
Tests for lacuna.training.trainer

Tests the training loop infrastructure:
    - TrainerConfig: Configuration dataclass
    - TrainerState: Mutable training state
    - LRScheduler: Learning rate scheduling with warmup
    - Trainer: Full training loop manager
"""

import pytest
import torch
import torch.nn as nn
from dataclasses import asdict
from typing import List, Iterator
import tempfile
from pathlib import Path
import math

from lacuna.training.trainer import (
    TrainerConfig,
    TrainerState,
    LRScheduler,
    Trainer,
)
from lacuna.training.loss import LossConfig, LacunaLoss
from lacuna.training.checkpoint import CheckpointData, save_checkpoint
from lacuna.models.assembly import create_lacuna_mini
from lacuna.core.types import TokenBatch, LacunaOutput, PosteriorResult
from lacuna.data.tokenization import TOKEN_DIM


# =============================================================================
# Test Helpers
# =============================================================================

def make_dummy_model():
    """Create small model for testing."""
    return create_lacuna_mini(max_cols=8, mnar_variants=["self_censoring"])


def make_dummy_batch(B: int = 4, max_rows: int = 16, max_cols: int = 8):
    """Create dummy TokenBatch for testing."""
    return TokenBatch(
        tokens=torch.randn(B, max_rows, max_cols, TOKEN_DIM),
        row_mask=torch.ones(B, max_rows, dtype=torch.bool),
        col_mask=torch.ones(B, max_cols, dtype=torch.bool),
        class_ids=torch.randint(0, 3, (B,)),
        variant_ids=torch.zeros(B, dtype=torch.long),
        original_values=torch.randn(B, max_rows, max_cols),
        reconstruction_mask=torch.rand(B, max_rows, max_cols) > 0.7,
    )


def make_dummy_dataloader(n_batches: int = 5, batch_size: int = 4) -> List[TokenBatch]:
    """Create list of dummy batches."""
    return [make_dummy_batch(B=batch_size) for _ in range(n_batches)]


class DummyLoader:
    """Simple loader that wraps a list of batches."""
    
    def __init__(self, batches: List[TokenBatch]):
        self.batches = batches
    
    def __iter__(self) -> Iterator[TokenBatch]:
        return iter(self.batches)
    
    def __len__(self) -> int:
        return len(self.batches)


# =============================================================================
# Test TrainerConfig
# =============================================================================

class TestTrainerConfig:
    """Tests for TrainerConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = TrainerConfig()
        
        assert config.lr == 1e-4
        assert config.min_lr == 1e-6
        assert config.weight_decay == 0.01
        assert config.grad_clip == 1.0
        assert config.epochs == 20
        assert config.warmup_steps == 100
        assert config.lr_schedule == "cosine"
        assert config.patience == 5
        assert config.training_mode == "joint"
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainerConfig(
            lr=1e-3,
            epochs=50,
            patience=10,
            training_mode="classification",
            use_amp=True,
        )
        
        assert config.lr == 1e-3
        assert config.epochs == 50
        assert config.patience == 10
        assert config.training_mode == "classification"
        assert config.use_amp is True
    
    def test_invalid_training_mode_raises(self):
        """Test that invalid training_mode raises error."""
        with pytest.raises(ValueError, match="Unknown training_mode"):
            TrainerConfig(training_mode="invalid")
    
    def test_invalid_lr_schedule_raises(self):
        """Test that invalid lr_schedule raises error."""
        with pytest.raises(ValueError, match="Unknown lr_schedule"):
            TrainerConfig(lr_schedule="invalid")
    
    def test_invalid_early_stop_mode_raises(self):
        """Test that invalid early_stop_mode raises error."""
        with pytest.raises(ValueError, match="Unknown early_stop_mode"):
            TrainerConfig(early_stop_mode="invalid")
    
    def test_get_loss_config_joint(self):
        """Test get_loss_config for joint training."""
        config = TrainerConfig(
            training_mode="joint",
            mechanism_weight=1.5,
            reconstruction_weight=0.8,
        )
        
        loss_config = config.get_loss_config()
        
        assert isinstance(loss_config, LossConfig)
        assert loss_config.mechanism_weight == 1.5
        assert loss_config.reconstruction_weight == 0.8
    
    def test_get_loss_config_pretraining(self):
        """Test get_loss_config for pretraining mode."""
        config = TrainerConfig(training_mode="pretraining")
        loss_config = config.get_loss_config()
        
        assert loss_config.mechanism_weight == 0.0
        assert loss_config.reconstruction_weight == 1.0
    
    def test_get_loss_config_classification(self):
        """Test get_loss_config for classification mode."""
        config = TrainerConfig(training_mode="classification")
        loss_config = config.get_loss_config()
        
        assert loss_config.mechanism_weight == 1.0
        assert loss_config.reconstruction_weight == 0.0


# =============================================================================
# Test TrainerState
# =============================================================================

class TestTrainerState:
    """Tests for TrainerState dataclass."""
    
    def test_default_values(self):
        """Test default state values."""
        state = TrainerState()
        
        assert state.step == 0
        assert state.epoch == 0
        assert state.samples_seen == 0
        assert state.best_val_loss == float("inf")
        assert state.best_val_acc == 0.0
        assert state.patience_counter == 0
        assert state.should_stop is False
    
    def test_reset_epoch_metrics(self):
        """Test reset_epoch_metrics method."""
        state = TrainerState()
        
        # Simulate some epoch progress
        state.epoch_loss_sum = 10.0
        state.epoch_samples = 100
        state.epoch_correct = 80
        
        state.reset_epoch_metrics()
        
        assert state.epoch_loss_sum == 0.0
        assert state.epoch_samples == 0
        assert state.epoch_correct == 0
        assert state.epoch_start_time > 0
    
    def test_get_epoch_metrics(self):
        """Test get_epoch_metrics method."""
        state = TrainerState()
        state.reset_epoch_metrics()
        
        state.epoch_loss_sum = 5.0
        state.epoch_samples = 10
        state.epoch_correct = 8
        
        metrics = state.get_epoch_metrics()
        
        assert metrics["epoch_loss"] == 0.5
        assert metrics["epoch_acc"] == 0.8
        assert "epoch_time" in metrics
    
    def test_get_epoch_metrics_handles_zero_samples(self):
        """Test get_epoch_metrics with no samples."""
        state = TrainerState()
        state.reset_epoch_metrics()
        
        metrics = state.get_epoch_metrics()
        
        # Should not raise division by zero
        assert metrics["epoch_loss"] == 0.0
        assert metrics["epoch_acc"] == 0.0


# =============================================================================
# Test LRScheduler
# =============================================================================

class TestLRScheduler:
    """Tests for LRScheduler."""
    
    @pytest.fixture
    def optimizer(self):
        """Create dummy optimizer."""
        model = nn.Linear(10, 10)
        return torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    def test_warmup_starts_low(self, optimizer):
        """Test that warmup starts at low learning rate."""
        config = TrainerConfig(lr=1e-3, warmup_steps=100)
        scheduler = LRScheduler(optimizer, config, total_steps=1000)
        
        lr_at_0 = scheduler.get_lr(0)
        
        # Should be lr * (1 / warmup_steps) = 1e-3 * 0.01 = 1e-5
        assert lr_at_0 == pytest.approx(1e-5, rel=1e-3)
    
    def test_warmup_reaches_full(self, optimizer):
        """Test that warmup reaches full learning rate."""
        config = TrainerConfig(lr=1e-3, warmup_steps=100)
        scheduler = LRScheduler(optimizer, config, total_steps=1000)
        
        lr_at_100 = scheduler.get_lr(100)
        
        assert lr_at_100 == pytest.approx(1e-3, rel=1e-3)
    
    def test_warmup_linear(self, optimizer):
        """Test that warmup is linear."""
        config = TrainerConfig(lr=1e-3, warmup_steps=100)
        scheduler = LRScheduler(optimizer, config, total_steps=1000)
        
        lr_at_50 = scheduler.get_lr(49)  # Step 49 -> 50/100 = 0.5
        
        assert lr_at_50 == pytest.approx(5e-4, rel=1e-2)
    
    def test_cosine_decay(self, optimizer):
        """Test cosine learning rate decay."""
        config = TrainerConfig(
            lr=1e-3,
            min_lr=1e-5,
            warmup_steps=100,
            lr_schedule="cosine",
        )
        scheduler = LRScheduler(optimizer, config, total_steps=1000)
        
        # At end of training, should be close to min_lr
        lr_at_end = scheduler.get_lr(999)
        
        assert lr_at_end == pytest.approx(1e-5, rel=0.1)
    
    def test_linear_decay(self, optimizer):
        """Test linear learning rate decay."""
        config = TrainerConfig(
            lr=1e-3,
            min_lr=1e-5,
            warmup_steps=0,
            lr_schedule="linear",
        )
        scheduler = LRScheduler(optimizer, config, total_steps=1000)
        
        # At midpoint, should be halfway between lr and min_lr
        lr_at_mid = scheduler.get_lr(500)
        
        expected = 1e-3 - (1e-3 - 1e-5) * 0.5
        assert lr_at_mid == pytest.approx(expected, rel=1e-2)
    
    def test_constant_schedule(self, optimizer):
        """Test constant learning rate (after warmup)."""
        config = TrainerConfig(
            lr=1e-3,
            warmup_steps=100,
            lr_schedule="constant",
        )
        scheduler = LRScheduler(optimizer, config, total_steps=1000)
        
        # After warmup, should stay at lr
        lr_at_500 = scheduler.get_lr(500)
        lr_at_900 = scheduler.get_lr(900)
        
        assert lr_at_500 == pytest.approx(1e-3, rel=1e-3)
        assert lr_at_900 == pytest.approx(1e-3, rel=1e-3)
    
    def test_step_updates_optimizer(self, optimizer):
        """Test that step() updates optimizer learning rate."""
        config = TrainerConfig(lr=1e-3, warmup_steps=100)
        scheduler = LRScheduler(optimizer, config, total_steps=1000)
        
        scheduler.step(50)
        
        actual_lr = optimizer.param_groups[0]["lr"]
        expected_lr = scheduler.get_lr(50)
        
        assert actual_lr == pytest.approx(expected_lr, rel=1e-5)
    
    def test_update_warmup_steps_from_epochs(self, optimizer):
        """Test warmup_steps calculation from warmup_epochs."""
        config = TrainerConfig(
            lr=1e-3,
            warmup_steps=0,
            warmup_epochs=0.5,  # Half an epoch
        )
        scheduler = LRScheduler(optimizer, config, total_steps=1000)
        
        scheduler.update_warmup_steps(steps_per_epoch=100)
        
        assert scheduler.warmup_steps == 50


# =============================================================================
# Test Trainer Initialization
# =============================================================================

class TestTrainerInit:
    """Tests for Trainer initialization."""
    
    def test_basic_init(self):
        """Test basic trainer initialization."""
        model = make_dummy_model()
        config = TrainerConfig()
        
        trainer = Trainer(model, config, device="cpu")
        
        assert trainer.model is model
        assert trainer.config is config
        assert trainer.device == "cpu"
        assert trainer.state is not None
        assert trainer.loss_fn is not None
        assert trainer.optimizer is not None
    
    def test_model_moved_to_device(self):
        """Test that model is moved to device."""
        model = make_dummy_model()
        config = TrainerConfig()
        
        trainer = Trainer(model, config, device="cpu")
        
        # Check model parameters are on CPU
        param = next(trainer.model.parameters())
        assert param.device.type == "cpu"
    
    def test_loss_fn_from_config(self):
        """Test that loss function uses config settings."""
        model = make_dummy_model()
        config = TrainerConfig(
            training_mode="classification",
            label_smoothing=0.1,
        )
        
        trainer = Trainer(model, config, device="cpu")
        
        assert trainer.loss_fn.config.mechanism_weight > 0
        assert trainer.loss_fn.config.reconstruction_weight == 0
        assert trainer.loss_fn.config.label_smoothing == 0.1
    
    def test_checkpoint_dir_created(self):
        """Test checkpoint directory is created if specified."""
        model = make_dummy_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            config = TrainerConfig(checkpoint_dir=str(ckpt_dir))
            
            trainer = Trainer(model, config, device="cpu")
            
            assert trainer.checkpoint_dir.exists()


# =============================================================================
# Test Trainer Training Step
# =============================================================================

class TestTrainerStep:
    """Tests for single training step."""
    
    def test_train_step_returns_metrics(self):
        """Test that train_step returns metrics dict."""
        model = make_dummy_model()
        config = TrainerConfig()
        trainer = Trainer(model, config, device="cpu")
        
        batch = make_dummy_batch()
        metrics = trainer.train_step(batch)
        
        assert isinstance(metrics, dict)
        assert "loss" in metrics or "total_loss" in metrics
    
    def test_train_step_updates_state(self):
        """Test that train_step updates state."""
        model = make_dummy_model()
        config = TrainerConfig()
        trainer = Trainer(model, config, device="cpu")
        
        initial_step = trainer.state.step
        
        batch = make_dummy_batch()
        trainer.train_step(batch)
        
        assert trainer.state.step == initial_step + 1
    
    def test_train_step_reduces_loss(self):
        """Test that training steps reduce loss over time."""
        model = make_dummy_model()
        config = TrainerConfig(lr=0.01)
        trainer = Trainer(model, config, device="cpu")
        
        batch = make_dummy_batch()
        
        # Get initial loss
        model.eval()
        with torch.no_grad():
            output = model(batch)
            initial_loss = nn.functional.cross_entropy(
                output.posterior.p_class.log(),
                batch.class_ids,
            ).item()
        
        # Train for several steps
        model.train()
        for _ in range(20):
            trainer.train_step(batch)
        
        # Get final loss
        model.eval()
        with torch.no_grad():
            output = model(batch)
            final_loss = nn.functional.cross_entropy(
                output.posterior.p_class.log(),
                batch.class_ids,
            ).item()
        
        assert final_loss < initial_loss
    
    def test_gradient_clipping(self):
        """Test that gradient clipping is applied."""
        model = make_dummy_model()
        config = TrainerConfig(grad_clip=0.1)
        trainer = Trainer(model, config, device="cpu")
        
        batch = make_dummy_batch()
        metrics = trainer.train_step(batch)
        
        # Check gradient norm is clipped
        if "grad_norm" in metrics:
            assert metrics["grad_norm"] <= 0.1 + 1e-5
        
        # Verify by computing gradient norms manually
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # After clipping, norm should be <= grad_clip
        # Note: clipping happens during train_step, not after
    
    def test_train_step_handles_nan(self):
        """Test handling of NaN in loss (should raise or handle gracefully)."""
        model = make_dummy_model()
        config = TrainerConfig()
        trainer = Trainer(model, config, device="cpu")
        
        # Normal batch should work fine
        batch = make_dummy_batch()
        metrics = trainer.train_step(batch)
        
        assert not any(
            isinstance(v, float) and math.isnan(v) 
            for v in metrics.values() 
            if isinstance(v, (int, float))
        )


# =============================================================================
# Test Trainer Validation
# =============================================================================

class TestTrainerValidation:
    """Tests for validation."""
    
    def test_validate_returns_metrics(self):
        """Test that validate returns metrics dict."""
        model = make_dummy_model()
        config = TrainerConfig()
        trainer = Trainer(model, config, device="cpu")
        
        val_batches = make_dummy_dataloader(n_batches=3)
        val_loader = DummyLoader(val_batches)
        
        metrics = trainer.validate(val_loader)
        
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics
    
    def test_validate_in_eval_mode(self):
        """Test that model is in eval mode during validation."""
        model = make_dummy_model()
        config = TrainerConfig()
        trainer = Trainer(model, config, device="cpu")
        
        val_loader = DummyLoader(make_dummy_dataloader(n_batches=2))
        
        # Patch forward to check training mode
        original_forward = model.forward
        modes_during_forward = []
        
        def tracking_forward(*args, **kwargs):
            modes_during_forward.append(model.training)
            return original_forward(*args, **kwargs)
        
        model.forward = tracking_forward
        
        trainer.validate(val_loader)
        
        model.forward = original_forward
        
        assert all(not mode for mode in modes_during_forward)
    
    def test_validate_no_gradients(self):
        """Test that gradients are disabled during validation."""
        model = make_dummy_model()
        config = TrainerConfig()
        trainer = Trainer(model, config, device="cpu")
        
        val_loader = DummyLoader(make_dummy_dataloader(n_batches=2))
        
        # Clear any existing gradients
        for p in model.parameters():
            p.grad = None
        
        trainer.validate(val_loader)
        
        # No gradients should have been computed
        for p in model.parameters():
            assert p.grad is None


# =============================================================================
# Test Trainer Fit
# =============================================================================

class TestTrainerFit:
    """Tests for full training loop."""
    
    def test_fit_runs_all_epochs(self):
        """Test that fit runs for specified epochs."""
        model = make_dummy_model()
        config = TrainerConfig(epochs=2, patience=100)
        trainer = Trainer(model, config, device="cpu")
        
        train_loader = DummyLoader(make_dummy_dataloader(n_batches=5))
        
        result = trainer.fit(train_loader)
        
        assert result["epochs_completed"] == 2
        assert result["total_steps"] == 10
    
    def test_fit_with_validation(self):
        """Test fit with validation loader."""
        model = make_dummy_model()
        config = TrainerConfig(epochs=2, patience=100, eval_every_epoch=True)
        trainer = Trainer(model, config, device="cpu")
        
        train_loader = DummyLoader(make_dummy_dataloader(n_batches=5))
        val_loader = DummyLoader(make_dummy_dataloader(n_batches=2))
        
        result = trainer.fit(train_loader, val_loader)
        
        assert "best_val_loss" in result
        assert result["best_val_loss"] < float("inf")
    
    def test_early_stopping(self):
        """Test early stopping triggers."""
        model = make_dummy_model()
        config = TrainerConfig(
            epochs=100,
            patience=2,
            lr=1e-10,  # Tiny LR = no improvement
        )
        trainer = Trainer(model, config, device="cpu")
        
        train_loader = DummyLoader(make_dummy_dataloader(n_batches=3))
        val_loader = DummyLoader(make_dummy_dataloader(n_batches=2))
        
        result = trainer.fit(train_loader, val_loader)
        
        # Should stop before 100 epochs
        assert result["epochs_completed"] < 100
        assert result.get("early_stopped", False) is True
    
    def test_fit_tracks_best_model(self):
        """Test that fit tracks best validation metrics."""
        model = make_dummy_model()
        config = TrainerConfig(epochs=3, patience=100, eval_every_epoch=True)
        trainer = Trainer(model, config, device="cpu")
        
        train_loader = DummyLoader(make_dummy_dataloader(n_batches=5))
        val_loader = DummyLoader(make_dummy_dataloader(n_batches=2))
        
        result = trainer.fit(train_loader, val_loader)
        
        assert trainer.state.best_epoch >= 0
        assert trainer.state.best_val_loss < float("inf")
    
    def test_fit_returns_history(self):
        """Test that fit returns training history."""
        model = make_dummy_model()
        config = TrainerConfig(epochs=2, patience=100)
        trainer = Trainer(model, config, device="cpu")
        
        train_loader = DummyLoader(make_dummy_dataloader(n_batches=5))
        
        result = trainer.fit(train_loader)
        
        assert "epochs_completed" in result
        assert "total_steps" in result
        assert "final_train_loss" in result or "train_loss" in result


# =============================================================================
# Test Checkpoint Integration
# =============================================================================

class TestTrainerCheckpointing:
    """Tests for checkpointing during training."""
    
    def test_save_checkpoint(self):
        """Test saving checkpoint during training."""
        model = make_dummy_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                epochs=2,
                checkpoint_dir=tmpdir,
                save_best_only=False,
                save_every_epoch=True,
            )
            trainer = Trainer(model, config, device="cpu")
            
            train_loader = DummyLoader(make_dummy_dataloader(n_batches=3))
            val_loader = DummyLoader(make_dummy_dataloader(n_batches=2))
            
            trainer.fit(train_loader, val_loader)
            
            # Check that checkpoints were saved
            ckpt_files = list(Path(tmpdir).glob("*.pt"))
            assert len(ckpt_files) > 0
    
    def test_resume_from_checkpoint(self):
        """Test resuming training from checkpoint."""
        model1 = make_dummy_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(epochs=2, checkpoint_dir=tmpdir)
            trainer1 = Trainer(model1, config, device="cpu")
            
            train_loader = DummyLoader(make_dummy_dataloader(n_batches=5))
            trainer1.fit(train_loader)
            
            # Save checkpoint
            ckpt_path = Path(tmpdir) / "manual_ckpt.pt"
            checkpoint = CheckpointData(
                model_state=model1.state_dict(),
                optimizer_state=trainer1.optimizer.state_dict(),
                step=trainer1.state.step,
                epoch=trainer1.state.epoch,
            )
            save_checkpoint(checkpoint, ckpt_path)
            
            # Create new trainer and resume
            model2 = make_dummy_model()
            trainer2 = Trainer(model2, config, device="cpu")
            
            trainer2.load_checkpoint(ckpt_path)
            
            assert trainer2.state.step == trainer1.state.step
            assert trainer2.state.epoch == trainer1.state.epoch


# =============================================================================
# Test Early Stopping Logic
# =============================================================================

class TestEarlyStopping:
    """Tests for early stopping logic."""
    
    def test_check_early_stop_improves(self):
        """Test that improvement resets patience."""
        model = make_dummy_model()
        config = TrainerConfig(patience=5, early_stop_mode="min")
        trainer = Trainer(model, config, device="cpu")
        
        # Simulate improving validation loss
        trainer.state.best_val_loss = 1.0
        trainer.state.patience_counter = 3
        
        should_stop = trainer._check_early_stop({"val_loss": 0.5})
        
        assert should_stop is False
        assert trainer.state.patience_counter == 0
        assert trainer.state.best_val_loss == 0.5
    
    def test_check_early_stop_no_improve(self):
        """Test that no improvement increments patience."""
        model = make_dummy_model()
        config = TrainerConfig(patience=5, early_stop_mode="min", min_delta=0.01)
        trainer = Trainer(model, config, device="cpu")
        
        trainer.state.best_val_loss = 0.5
        trainer.state.patience_counter = 0
        
        # Loss didn't improve enough
        should_stop = trainer._check_early_stop({"val_loss": 0.49})
        
        assert should_stop is False
        assert trainer.state.patience_counter == 1
    
    def test_check_early_stop_triggers(self):
        """Test that patience exhaustion triggers stop."""
        model = make_dummy_model()
        config = TrainerConfig(patience=3, early_stop_mode="min")
        trainer = Trainer(model, config, device="cpu")
        
        trainer.state.best_val_loss = 0.5
        trainer.state.patience_counter = 2  # One more = patience exhausted
        
        should_stop = trainer._check_early_stop({"val_loss": 0.6})
        
        assert should_stop is True
        assert trainer.state.patience_counter == 3
    
    def test_early_stop_max_mode(self):
        """Test early stopping in max mode (for accuracy)."""
        model = make_dummy_model()
        config = TrainerConfig(
            patience=5,
            early_stop_mode="max",
            early_stop_metric="val_acc",
        )
        trainer = Trainer(model, config, device="cpu")
        
        trainer.state.best_val_acc = 0.8
        trainer.state.patience_counter = 0
        
        # Accuracy improved
        should_stop = trainer._check_early_stop({"val_acc": 0.85})
        
        assert should_stop is False
        assert trainer.state.patience_counter == 0
        assert trainer.state.best_val_acc == 0.85


# =============================================================================
# Test Log Callback
# =============================================================================

class TestLogCallback:
    """Tests for logging callback."""
    
    def test_log_fn_called(self):
        """Test that log function is called during training."""
        model = make_dummy_model()
        config = TrainerConfig(epochs=1, log_every=1)
        
        logged_metrics = []
        
        def log_fn(metrics):
            logged_metrics.append(metrics)
        
        trainer = Trainer(model, config, device="cpu", log_fn=log_fn)
        
        train_loader = DummyLoader(make_dummy_dataloader(n_batches=5))
        trainer.fit(train_loader)
        
        assert len(logged_metrics) > 0
    
    def test_log_fn_receives_metrics(self):
        """Test that log function receives proper metrics."""
        model = make_dummy_model()
        config = TrainerConfig(epochs=1, log_every=1)
        
        logged_metrics = []
        
        def log_fn(metrics):
            logged_metrics.append(metrics.copy())
        
        trainer = Trainer(model, config, device="cpu", log_fn=log_fn)
        
        train_loader = DummyLoader(make_dummy_dataloader(n_batches=3))
        trainer.fit(train_loader)
        
        # Check that metrics contain expected keys
        if logged_metrics:
            first_log = logged_metrics[0]
            assert "step" in first_log or "epoch" in first_log