"""
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
        # Small patience, training on random data won't improve
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
