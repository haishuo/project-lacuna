"""
Tests for lacuna.training.checkpoint

Tests checkpoint management:
    - CheckpointData: Data structure for checkpoints
    - save_checkpoint, load_checkpoint: Basic I/O
    - load_model_weights: Load weights into model
    - CheckpointManager: Multi-checkpoint management
    - Utilities: compare_checkpoints, export_for_inference, compute_checkpoint_hash
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
from datetime import datetime

from lacuna.training.checkpoint import (
    CheckpointData,
    save_checkpoint,
    load_checkpoint,
    load_model_weights,
    CheckpointManager,
    compare_checkpoints,
    export_for_inference,
    compute_checkpoint_hash,
)
from lacuna.core.exceptions import CheckpointError
from lacuna.models.assembly import create_lacuna_mini
from lacuna.data.tokenization import TOKEN_DIM


# =============================================================================
# Test Helpers
# =============================================================================

def make_dummy_model():
    """Create small model for testing."""
    return create_lacuna_mini(max_cols=8, mnar_variants=["self_censoring"])


def make_dummy_state_dict():
    """Create a simple state dict for testing."""
    return {
        "layer1.weight": torch.randn(32, 16),
        "layer1.bias": torch.randn(32),
        "layer2.weight": torch.randn(16, 32),
        "layer2.bias": torch.randn(16),
    }


# =============================================================================
# Test CheckpointData
# =============================================================================

class TestCheckpointData:
    """Tests for CheckpointData dataclass."""
    
    def test_minimal_construction(self):
        """Test construction with only required fields."""
        state = make_dummy_state_dict()
        data = CheckpointData(model_state=state)
        
        assert data.model_state is state
        assert data.step == 0
        assert data.epoch == 0
        assert data.optimizer_state is None
    
    def test_full_construction(self):
        """Test construction with all fields."""
        model_state = make_dummy_state_dict()
        optimizer_state = {"param_groups": [], "state": {}}
        
        data = CheckpointData(
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state={"last_epoch": 5},
            step=100,
            epoch=5,
            best_val_loss=0.5,
            best_val_acc=0.9,
            config={"lr": 1e-4},
            model_config={"hidden_dim": 128},
            metrics={"loss": 0.3},
        )
        
        assert data.step == 100
        assert data.epoch == 5
        assert data.best_val_loss == 0.5
        assert data.best_val_acc == 0.9
        assert data.config == {"lr": 1e-4}
        assert data.metrics == {"loss": 0.3}
    
    def test_default_timestamp(self):
        """Test that timestamp is auto-generated."""
        data = CheckpointData(model_state=make_dummy_state_dict())
        
        assert data.timestamp is not None
        assert len(data.timestamp) > 0
        
        # Should be valid ISO format
        datetime.fromisoformat(data.timestamp)
    
    def test_default_version(self):
        """Test that version is set."""
        data = CheckpointData(model_state=make_dummy_state_dict())
        
        assert data.lacuna_version is not None
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        data = CheckpointData(
            model_state=make_dummy_state_dict(),
            step=42,
            epoch=3,
        )
        
        as_dict = data.to_dict()
        
        assert isinstance(as_dict, dict)
        assert as_dict["step"] == 42
        assert as_dict["epoch"] == 3
        assert "model_state" in as_dict
        assert "timestamp" in as_dict
    
    def test_from_dict(self):
        """Test construction from dictionary."""
        original = CheckpointData(
            model_state=make_dummy_state_dict(),
            step=42,
            epoch=3,
            best_val_loss=0.5,
        )
        
        as_dict = original.to_dict()
        restored = CheckpointData.from_dict(as_dict)
        
        assert restored.step == 42
        assert restored.epoch == 3
        assert restored.best_val_loss == 0.5
    
    def test_from_dict_missing_optional(self):
        """Test from_dict handles missing optional fields."""
        minimal_dict = {
            "model_state": make_dummy_state_dict(),
        }
        
        data = CheckpointData.from_dict(minimal_dict)
        
        assert data.step == 0
        assert data.epoch == 0
        assert data.optimizer_state is None


# =============================================================================
# Test save_checkpoint and load_checkpoint
# =============================================================================

class TestSaveLoadCheckpoint:
    """Tests for save_checkpoint and load_checkpoint."""
    
    def test_save_load_roundtrip(self):
        """Test basic save/load roundtrip."""
        data = CheckpointData(
            model_state=make_dummy_state_dict(),
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
    
    def test_save_creates_directory(self):
        """Test that save creates parent directories."""
        data = CheckpointData(model_state=make_dummy_state_dict())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "deep" / "test.pt"
            
            save_checkpoint(data, path)
            
            assert path.exists()
    
    def test_save_returns_path(self):
        """Test that save returns the path."""
        data = CheckpointData(model_state=make_dummy_state_dict())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            
            returned_path = save_checkpoint(data, path)
            
            assert returned_path == path
    
    def test_load_nonexistent_raises(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(CheckpointError, match="not found"):
            load_checkpoint(Path("/nonexistent/path/checkpoint.pt"))
    
    def test_load_with_device_mapping(self):
        """Test loading with device mapping."""
        data = CheckpointData(model_state=make_dummy_state_dict())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            
            loaded = load_checkpoint(path, device="cpu")
        
        # Check tensors are on CPU
        for key, tensor in loaded.model_state.items():
            assert tensor.device.type == "cpu"
    
    def test_save_without_optimizer(self):
        """Test saving without optimizer state."""
        data = CheckpointData(
            model_state=make_dummy_state_dict(),
            optimizer_state={"param_groups": [], "state": {}},
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            
            save_checkpoint(data, path, include_optimizer=False)
            loaded = load_checkpoint(path)
        
        assert loaded.optimizer_state is None
    
    def test_weights_only_load(self):
        """Test weights_only loading for security."""
        data = CheckpointData(model_state=make_dummy_state_dict())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            
            # weights_only=True for secure loading
            loaded = load_checkpoint(path, weights_only=True)
        
        assert "layer1.weight" in loaded.model_state
    
    def test_preserves_tensor_values(self):
        """Test that tensor values are preserved exactly."""
        state = {
            "weights": torch.tensor([1.0, 2.0, 3.0]),
            "int_tensor": torch.tensor([1, 2, 3]),
        }
        data = CheckpointData(model_state=state)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            loaded = load_checkpoint(path)
        
        assert torch.equal(loaded.model_state["weights"], state["weights"])
        assert torch.equal(loaded.model_state["int_tensor"], state["int_tensor"])
    
    def test_preserves_optimizer_state(self):
        """Test that optimizer state is preserved."""
        model = make_dummy_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Run a few steps to populate optimizer state
        for _ in range(3):
            loss = sum(p.sum() for p in model.parameters())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        data = CheckpointData(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            loaded = load_checkpoint(path)
        
        assert loaded.optimizer_state is not None
        assert "state" in loaded.optimizer_state
        assert len(loaded.optimizer_state["state"]) > 0


# =============================================================================
# Test load_model_weights
# =============================================================================

class TestLoadModelWeights:
    """Tests for load_model_weights function."""
    
    def test_load_restores_weights(self):
        """Test that weights are correctly restored."""
        model1 = make_dummy_model()
        model2 = make_dummy_model()
        
        # Verify models are different initially
        param1 = next(model1.parameters())
        param2 = next(model2.parameters())
        assert not torch.allclose(param1, param2)
        
        # Save model1 and load into model2
        data = CheckpointData(model_state=model1.state_dict())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            
            model2 = load_model_weights(model2, path)
        
        # Now they should match
        param2_after = next(model2.parameters())
        assert torch.allclose(param1, param2_after)
    
    def test_load_returns_model(self):
        """Test that load returns the model."""
        model = make_dummy_model()
        data = CheckpointData(model_state=model.state_dict())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            
            returned = load_model_weights(model, path)
        
        assert returned is model
    
    def test_strict_mode_raises_on_mismatch(self):
        """Test that strict mode raises on key mismatch."""
        model = make_dummy_model()
        
        # Create state dict with wrong keys
        bad_state = {"wrong.key": torch.randn(10)}
        data = CheckpointData(model_state=bad_state)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            
            with pytest.raises(CheckpointError, match="strict mode"):
                load_model_weights(model, path, strict=True)
    
    def test_non_strict_mode_allows_mismatch(self):
        """Test that non-strict mode allows partial loading."""
        model = make_dummy_model()
        
        # Get partial state dict
        full_state = model.state_dict()
        partial_state = {k: v for i, (k, v) in enumerate(full_state.items()) if i < 5}
        
        data = CheckpointData(model_state=partial_state)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            
            # Should not raise with strict=False
            model = load_model_weights(model, path, strict=False)
    
    def test_device_mapping(self):
        """Test loading weights with device mapping."""
        model = make_dummy_model()
        data = CheckpointData(model_state=model.state_dict())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            
            model = load_model_weights(model, path, device="cpu")
        
        # Verify model is on CPU
        param = next(model.parameters())
        assert param.device.type == "cpu"


# =============================================================================
# Test CheckpointManager
# =============================================================================

class TestCheckpointManager:
    """Tests for CheckpointManager class."""
    
    def test_initialization(self):
        """Test manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, keep_best=1, keep_last=3)
            
            assert manager.checkpoint_dir.exists()
            assert manager.keep_best == 1
            assert manager.keep_last == 3
    
    def test_creates_directory(self):
        """Test that manager creates checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints" / "experiment1"
            
            manager = CheckpointManager(ckpt_dir)
            
            assert ckpt_dir.exists()
    
    def test_save_best_checkpoint(self):
        """Test saving a best checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, keep_best=2)
            
            data = CheckpointData(
                model_state=make_dummy_state_dict(),
                best_val_loss=0.5,
            )
            
            path = manager.save(data, is_best=True)
            
            assert path.exists()
            assert "best" in path.name.lower()
    
    def test_save_periodic_checkpoint(self):
        """Test saving periodic checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, keep_last=3)
            
            # Save multiple periodic checkpoints
            for i in range(5):
                data = CheckpointData(
                    model_state=make_dummy_state_dict(),
                    step=i * 100,
                    epoch=i,
                )
                manager.save(data, is_best=False)
            
            # Should only keep last 3
            assert len(manager.periodic_checkpoints) <= 3
    
    def test_cleanup_old_checkpoints(self):
        """Test automatic cleanup of old checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, keep_last=2)
            
            paths = []
            for i in range(4):
                data = CheckpointData(
                    model_state=make_dummy_state_dict(),
                    step=i,
                )
                path = manager.save(data, is_best=False)
                paths.append(path)
            
            # First two should be deleted
            assert not paths[0].exists()
            assert not paths[1].exists()
            
            # Last two should exist
            assert paths[2].exists()
            assert paths[3].exists()
    
    def test_load_best(self):
        """Test loading best checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            
            # Save best checkpoint
            data = CheckpointData(
                model_state=make_dummy_state_dict(),
                step=42,
                best_val_loss=0.3,
            )
            manager.save(data, is_best=True)
            
            # Load best
            loaded = manager.load_best()
            
            assert loaded.step == 42
            assert loaded.best_val_loss == 0.3
    
    def test_load_latest(self):
        """Test loading latest checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, keep_last=5)
            
            # Save several checkpoints
            for i in range(3):
                data = CheckpointData(
                    model_state=make_dummy_state_dict(),
                    step=i * 10,
                )
                manager.save(data, is_best=False)
            
            # Load latest
            loaded = manager.load_latest()
            
            assert loaded.step == 20  # Last saved
    
    def test_list_checkpoints(self):
        """Test listing available checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            
            # Save a few
            for i in range(3):
                data = CheckpointData(
                    model_state=make_dummy_state_dict(),
                    step=i,
                )
                manager.save(data, is_best=(i == 1))
            
            listing = manager.list_checkpoints()
            
            assert "best" in listing or len(listing["periodic"]) > 0


# =============================================================================
# Test Utilities
# =============================================================================

class TestCompareCheckpoints:
    """Tests for compare_checkpoints utility."""
    
    def test_identical_checkpoints(self):
        """Test comparing identical checkpoints."""
        state = make_dummy_state_dict()
        data = CheckpointData(model_state=state, step=10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "ckpt1.pt"
            path2 = Path(tmpdir) / "ckpt2.pt"
            
            save_checkpoint(data, path1)
            save_checkpoint(data, path2)
            
            comparison = compare_checkpoints(path1, path2)
        
        assert comparison["differing_weights"] == 0
        assert len(comparison["keys_only_in_1"]) == 0
        assert len(comparison["keys_only_in_2"]) == 0
    
    def test_different_weights(self):
        """Test comparing checkpoints with different weights."""
        state1 = {"layer.weight": torch.randn(10)}
        state2 = {"layer.weight": torch.randn(10)}
        
        data1 = CheckpointData(model_state=state1)
        data2 = CheckpointData(model_state=state2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "ckpt1.pt"
            path2 = Path(tmpdir) / "ckpt2.pt"
            
            save_checkpoint(data1, path1)
            save_checkpoint(data2, path2)
            
            comparison = compare_checkpoints(path1, path2)
        
        assert comparison["differing_weights"] == 1
    
    def test_different_keys(self):
        """Test comparing checkpoints with different keys."""
        state1 = {"layer1.weight": torch.randn(10)}
        state2 = {"layer2.weight": torch.randn(10)}
        
        data1 = CheckpointData(model_state=state1)
        data2 = CheckpointData(model_state=state2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "ckpt1.pt"
            path2 = Path(tmpdir) / "ckpt2.pt"
            
            save_checkpoint(data1, path1)
            save_checkpoint(data2, path2)
            
            comparison = compare_checkpoints(path1, path2)
        
        assert "layer1.weight" in comparison["keys_only_in_1"]
        assert "layer2.weight" in comparison["keys_only_in_2"]


class TestExportForInference:
    """Tests for export_for_inference utility."""
    
    def test_export_removes_optimizer(self):
        """Test that export removes optimizer state."""
        data = CheckpointData(
            model_state=make_dummy_state_dict(),
            optimizer_state={"param_groups": [], "state": {}},
            step=100,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            full_path = Path(tmpdir) / "full.pt"
            export_path = Path(tmpdir) / "inference.pt"
            
            save_checkpoint(data, full_path)
            export_for_inference(full_path, export_path)
            
            exported = load_checkpoint(export_path)
        
        assert exported.optimizer_state is None
        assert exported.step == 100
        assert "layer1.weight" in exported.model_state
    
    def test_export_preserves_weights(self):
        """Test that export preserves model weights."""
        state = make_dummy_state_dict()
        data = CheckpointData(model_state=state)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            full_path = Path(tmpdir) / "full.pt"
            export_path = Path(tmpdir) / "inference.pt"
            
            save_checkpoint(data, full_path)
            export_for_inference(full_path, export_path)
            
            exported = load_checkpoint(export_path)
        
        for key in state:
            assert torch.equal(exported.model_state[key], state[key])


class TestComputeCheckpointHash:
    """Tests for compute_checkpoint_hash utility."""
    
    def test_same_content_same_hash(self):
        """Test that same content produces same hash."""
        data = CheckpointData(model_state=make_dummy_state_dict())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            
            hash1 = compute_checkpoint_hash(path)
            hash2 = compute_checkpoint_hash(path)
        
        assert hash1 == hash2
    
    def test_different_content_different_hash(self):
        """Test that different content produces different hash."""
        data1 = CheckpointData(model_state={"w": torch.tensor([1.0])})
        data2 = CheckpointData(model_state={"w": torch.tensor([2.0])})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "ckpt1.pt"
            path2 = Path(tmpdir) / "ckpt2.pt"
            
            save_checkpoint(data1, path1)
            save_checkpoint(data2, path2)
            
            hash1 = compute_checkpoint_hash(path1)
            hash2 = compute_checkpoint_hash(path2)
        
        assert hash1 != hash2
    
    def test_hash_format(self):
        """Test hash is valid hex string."""
        data = CheckpointData(model_state=make_dummy_state_dict())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            
            hash_value = compute_checkpoint_hash(path)
        
        # Should be valid hex
        int(hash_value, 16)
        
        # SHA256 produces 64 character hex string
        assert len(hash_value) == 64


# =============================================================================
# Test Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_state_dict(self):
        """Test handling empty state dict."""
        data = CheckpointData(model_state={})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            loaded = load_checkpoint(path)
        
        assert loaded.model_state == {}
    
    def test_large_tensors(self):
        """Test handling large tensors."""
        large_state = {
            "big_weights": torch.randn(1000, 1000),
            "big_embeddings": torch.randn(10000, 256),
        }
        data = CheckpointData(model_state=large_state)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            loaded = load_checkpoint(path)
        
        assert loaded.model_state["big_weights"].shape == (1000, 1000)
        assert loaded.model_state["big_embeddings"].shape == (10000, 256)
    
    def test_special_float_values(self):
        """Test handling special float values."""
        special_state = {
            "inf_tensor": torch.tensor([float("inf"), float("-inf")]),
            "nan_tensor": torch.tensor([float("nan")]),
            "zeros": torch.zeros(10),
        }
        data = CheckpointData(model_state=special_state)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            loaded = load_checkpoint(path)
        
        assert torch.isinf(loaded.model_state["inf_tensor"]).any()
        assert torch.isnan(loaded.model_state["nan_tensor"]).any()
    
    def test_various_dtypes(self):
        """Test handling various tensor dtypes."""
        mixed_state = {
            "float32": torch.randn(10, dtype=torch.float32),
            "float64": torch.randn(10, dtype=torch.float64),
            "int32": torch.randint(0, 100, (10,), dtype=torch.int32),
            "int64": torch.randint(0, 100, (10,), dtype=torch.int64),
            "bool": torch.tensor([True, False, True]),
        }
        data = CheckpointData(model_state=mixed_state)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            loaded = load_checkpoint(path)
        
        assert loaded.model_state["float32"].dtype == torch.float32
        assert loaded.model_state["float64"].dtype == torch.float64
        assert loaded.model_state["int32"].dtype == torch.int32
        assert loaded.model_state["bool"].dtype == torch.bool
    
    def test_nested_config_preserved(self):
        """Test that nested config dicts are preserved."""
        config = {
            "model": {
                "hidden_dim": 128,
                "layers": [1, 2, 3],
            },
            "training": {
                "lr": 1e-4,
                "schedule": "cosine",
            },
        }
        
        data = CheckpointData(
            model_state=make_dummy_state_dict(),
            config=config,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            loaded = load_checkpoint(path)
        
        assert loaded.config["model"]["hidden_dim"] == 128
        assert loaded.config["training"]["lr"] == 1e-4