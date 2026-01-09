"""
Integration test: Full training loop with checkpointing.

Tests:
- Training reduces loss
- Checkpoints save and restore correctly
- Validation runs correctly
- Early stopping works
"""

import pytest
import torch
import tempfile
from pathlib import Path

from lacuna.core.types import TokenBatch, MCAR, MAR, MNAR
from lacuna.core.rng import RNGState
from lacuna.config.schema import LacunaConfig
from lacuna.generators import create_minimal_registry
from lacuna.data.batching import tokenize_and_batch
from lacuna.data.features import FEATURE_DIM
from lacuna.models.assembly import LacunaModel
from lacuna.training.trainer import Trainer, TrainerConfig
from lacuna.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_model_from_checkpoint,
    CheckpointData,
)


@pytest.fixture
def registry():
    """Create a minimal generator registry."""
    return create_minimal_registry()


def generate_batch(registry, batch_size: int, rng: RNGState) -> TokenBatch:
    """Generate a batch of data from random generators."""
    datasets = []
    gen_ids = []
    
    K = registry.K
    
    for _ in range(batch_size):
        gen_id = int(rng.randint(0, K, (1,)).item())
        gen = registry[gen_id]
        dataset = gen.sample_observed(rng.spawn(), n=50, d=8, dataset_id=f"batch_{gen_id}")
        
        datasets.append(dataset)
        gen_ids.append(gen.generator_id)
    
    return tokenize_and_batch(
        datasets=datasets,
        max_cols=16,
        generator_ids=gen_ids,
        class_mapping=registry.get_class_mapping(),
    )


class SyntheticDataLoader:
    """On-the-fly synthetic data loader for testing."""
    
    def __init__(self, registry, n_batches: int, batch_size: int, seed: int):
        self.registry = registry
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.seed = seed
    
    def __iter__(self):
        rng = RNGState(seed=self.seed)
        for _ in range(self.n_batches):
            yield generate_batch(self.registry, self.batch_size, rng)
    
    def __len__(self):
        return self.n_batches


class TestTrainingLoop:
    """Test full training loop."""
    
    def test_training_reduces_loss(self, registry):
        """Training should reduce loss over time."""
        cfg = LacunaConfig.minimal()
        class_mapping = registry.get_class_mapping()
        model = LacunaModel.from_config(cfg, class_mapping)
        
        train_loader = SyntheticDataLoader(registry, n_batches=10, batch_size=8, seed=42)
        
        # Get initial loss
        model.eval()
        initial_losses = []
        with torch.no_grad():
            for batch in SyntheticDataLoader(registry, n_batches=5, batch_size=8, seed=99):
                posterior = model(batch)
                loss = torch.nn.functional.cross_entropy(
                    posterior.logits_generator,
                    batch.generator_ids,
                )
                initial_losses.append(loss.item())
        initial_avg = sum(initial_losses) / len(initial_losses)
        
        # Train
        trainer_config = TrainerConfig(
            lr=1e-3,
            epochs=3,
            warmup_steps=10,
            grad_clip=1.0,
        )
        trainer = Trainer(model, trainer_config, device="cpu")
        
        result = trainer.fit(train_loader)
        
        # Get final loss
        model.eval()
        final_losses = []
        with torch.no_grad():
            for batch in SyntheticDataLoader(registry, n_batches=5, batch_size=8, seed=99):
                posterior = model(batch)
                loss = torch.nn.functional.cross_entropy(
                    posterior.logits_generator,
                    batch.generator_ids,
                )
                final_losses.append(loss.item())
        final_avg = sum(final_losses) / len(final_losses)
        
        # Loss should decrease
        assert final_avg < initial_avg, f"Loss did not decrease: {initial_avg:.4f} -> {final_avg:.4f}"
    
    def test_checkpoint_save_load(self, registry):
        """Checkpoints should preserve model state."""
        cfg = LacunaConfig.minimal()
        class_mapping = registry.get_class_mapping()
        
        model1 = LacunaModel.from_config(cfg, class_mapping)
        model2 = LacunaModel.from_config(cfg, class_mapping)
        
        # Train model1 briefly
        train_loader = SyntheticDataLoader(registry, n_batches=5, batch_size=8, seed=42)
        trainer_config = TrainerConfig(lr=1e-3, epochs=1, warmup_steps=5)
        trainer = Trainer(model1, trainer_config, device="cpu")
        trainer.fit(train_loader)
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            
            data = CheckpointData(
                model_state=model1.state_dict(),
                step=trainer.state.step,
                epoch=trainer.state.epoch,
            )
            save_checkpoint(data, ckpt_path)
            
            # Load into model2
            load_model_from_checkpoint(model2, ckpt_path)
        
        # Models should produce identical outputs
        rng = RNGState(seed=123)
        test_batch = generate_batch(registry, batch_size=4, rng=rng)
        
        model1.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model1(test_batch)
            out2 = model2(test_batch)
        
        assert torch.allclose(out1.logits_generator, out2.logits_generator)
        assert torch.allclose(out1.p_class, out2.p_class)
    
    def test_training_with_validation(self, registry):
        """Training with validation should track best model."""
        cfg = LacunaConfig.minimal()
        class_mapping = registry.get_class_mapping()
        model = LacunaModel.from_config(cfg, class_mapping)
        
        train_loader = SyntheticDataLoader(registry, n_batches=10, batch_size=8, seed=42)
        val_loader = SyntheticDataLoader(registry, n_batches=3, batch_size=8, seed=99)
        
        trainer_config = TrainerConfig(
            lr=1e-3,
            epochs=3,
            warmup_steps=10,
            patience=10,  # Don't early stop
        )
        trainer = Trainer(model, trainer_config, device="cpu")
        
        result = trainer.fit(train_loader, val_loader)
        
        assert "best_val_loss" in result
        assert "best_val_acc" in result
        assert result["best_val_loss"] < float("inf")
        assert result["best_val_acc"] >= 0
    
    def test_early_stopping_triggers(self, registry):
        """Early stopping should trigger when validation doesn't improve."""
        cfg = LacunaConfig.minimal()
        class_mapping = registry.get_class_mapping()
        model = LacunaModel.from_config(cfg, class_mapping)
        
        # Use tiny learning rate so model won't improve much
        train_loader = SyntheticDataLoader(registry, n_batches=5, batch_size=4, seed=42)
        val_loader = SyntheticDataLoader(registry, n_batches=2, batch_size=4, seed=99)
        
        trainer_config = TrainerConfig(
            lr=1e-8,  # Tiny LR = no improvement
            epochs=100,
            warmup_steps=0,
            patience=2,
            min_delta=0.001,
        )
        trainer = Trainer(model, trainer_config, device="cpu")
        
        result = trainer.fit(train_loader, val_loader)
        
        # Should stop well before 100 epochs
        assert result["epochs_completed"] < 100


class TestCheckpointIntegrity:
    """Test checkpoint save/load integrity."""
    
    def test_checkpoint_preserves_all_state(self, registry):
        """Checkpoint should preserve step, epoch, and optimizer state."""
        cfg = LacunaConfig.minimal()
        class_mapping = registry.get_class_mapping()
        model = LacunaModel.from_config(cfg, class_mapping)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Simulate some training
        rng = RNGState(seed=42)
        for i in range(5):
            batch = generate_batch(registry, batch_size=4, rng=rng)
            optimizer.zero_grad()
            posterior = model(batch)
            loss = torch.nn.functional.cross_entropy(
                posterior.logits_generator,
                batch.generator_ids,
            )
            loss.backward()
            optimizer.step()
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            
            data = CheckpointData(
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                step=42,
                epoch=3,
                best_val_loss=0.5,
                metadata={"test_key": "test_value"},
            )
            save_checkpoint(data, ckpt_path)
            
            # Load
            loaded = load_checkpoint(ckpt_path)
        
        assert loaded.step == 42
        assert loaded.epoch == 3
        assert loaded.best_val_loss == 0.5
        assert loaded.metadata["test_key"] == "test_value"
        assert loaded.optimizer_state is not None
