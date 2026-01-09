"""
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
