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

        # Should have a non-empty timestamp
        assert data.timestamp != ""
        # Should be parseable as ISO format
        datetime.fromisoformat(data.timestamp)

    def test_to_dict_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = CheckpointData(
            model_state=make_dummy_state_dict(),
            step=42,
            epoch=3,
            best_val_loss=0.5,
            config={"lr": 1e-4},
        )

        # Convert to dict and back
        as_dict = original.to_dict()
        restored = CheckpointData.from_dict(as_dict)

        assert restored.step == original.step
        assert restored.epoch == original.epoch
        assert restored.best_val_loss == original.best_val_loss
        assert restored.config == original.config

    def test_from_dict_with_missing_optional_fields(self):
        """Test from_dict handles missing optional fields."""
        minimal_dict = {"model_state": make_dummy_state_dict()}

        data = CheckpointData.from_dict(minimal_dict)

        assert data.step == 0
        assert data.epoch == 0
        assert data.optimizer_state is None
        assert data.best_val_loss == float("inf")


# =============================================================================
# Test save_checkpoint and load_checkpoint
# =============================================================================

class TestSaveLoadCheckpoint:
    """Tests for save_checkpoint and load_checkpoint functions."""

    def test_save_load_roundtrip(self):
        """Test basic save and load roundtrip."""
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

            # Should raise CheckpointError wrapping the RuntimeError
            with pytest.raises(CheckpointError, match="strict"):
                load_model_weights(model, path, strict=True)

    def test_non_strict_mode_allows_mismatch(self):
        """Test that non-strict mode allows partial loading."""
        model = make_dummy_model()

        # Get partial state dict (just first 5 keys)
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

            # Load with explicit CPU mapping
            model2 = make_dummy_model()
            model2 = load_model_weights(model2, path, device="cpu")

        # Check all parameters are on CPU
        for param in model2.parameters():
            assert param.device.type == "cpu"


# =============================================================================
# Test CheckpointManager
# =============================================================================

class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, keep_best=2, keep_last=3)

            assert manager.checkpoint_dir == Path(tmpdir)
            assert manager.keep_best == 2
            assert manager.keep_last == 3

    def test_save_periodic(self):
        """Test saving periodic checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, keep_best=1, keep_last=2)

            data = CheckpointData(model_state=make_dummy_state_dict(), step=100)
            path = manager.save(data, is_best=False)

            assert path.exists()
            assert "checkpoint" in path.name

    def test_save_best(self):
        """Test saving best checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, keep_best=1, keep_last=2)

            data = CheckpointData(
                model_state=make_dummy_state_dict(),
                step=42,
                epoch=3,
            )
            path = manager.save(data, is_best=True)

            assert path.exists()
            assert "best" in path.name

    def test_cleanup_periodic(self):
        """Test that old periodic checkpoints are cleaned up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, keep_best=1, keep_last=2)

            # Save 4 periodic checkpoints (should keep only 2)
            for step in range(4):
                data = CheckpointData(
                    model_state=make_dummy_state_dict(),
                    step=step,
                )
                manager.save(data, is_best=False)

            # Should have at most keep_last periodic checkpoints
            periodic = list(manager.checkpoint_dir.glob("checkpoint_*.pt"))
            assert len(periodic) <= 2

    def test_cleanup_best(self):
        """Test that old best checkpoints are cleaned up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, keep_best=2, keep_last=1)

            # Save 4 best checkpoints (should keep only 2)
            for step in range(4):
                data = CheckpointData(
                    model_state=make_dummy_state_dict(),
                    step=step,
                    best_val_loss=1.0 / (step + 1),  # Decreasing loss
                )
                manager.save(data, is_best=True)

            # Should have at most keep_best best checkpoints
            best = list(manager.checkpoint_dir.glob("best_*.pt"))
            assert len(best) <= 2

    def test_load_best(self):
        """Test loading best checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            data = CheckpointData(
                model_state=make_dummy_state_dict(),
                step=42,
                epoch=3,
            )
            manager.save(data, is_best=True)

            loaded = manager.load_best()
            assert loaded.step == 42

    def test_load_latest(self):
        """Test loading latest checkpoint."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            # Save multiple checkpoints with small delays to ensure different mtimes
            for step in [10, 20, 30]:
                data = CheckpointData(
                    model_state=make_dummy_state_dict(),
                    step=step,
                )
                manager.save(data, is_best=False)
                time.sleep(0.01)  # Small delay to ensure different mtime

            loaded = manager.load_latest()
            # Should load the most recent one (step=30)
            assert loaded.step == 30

    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            # Save some checkpoints
            for step in [10, 20]:
                data = CheckpointData(
                    model_state=make_dummy_state_dict(),
                    step=step,
                )
                manager.save(data, is_best=(step == 20))

            ckpts = manager.list_checkpoints()
            assert len(ckpts) >= 1

    def test_load_best_raises_when_none(self):
        """Test that load_best raises when no best checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            with pytest.raises(CheckpointError, match="No best checkpoint"):
                manager.load_best()

    def test_load_latest_raises_when_none(self):
        """Test that load_latest raises when no checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            with pytest.raises(CheckpointError, match="No checkpoints"):
                manager.load_latest()


# =============================================================================
# Test compare_checkpoints
# =============================================================================

class TestCompareCheckpoints:
    """Tests for compare_checkpoints utility."""

    def test_identical_checkpoints(self):
        """Test comparing identical checkpoints."""
        state = make_dummy_state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save same state to two files
            data = CheckpointData(model_state=state, step=42)
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
        with tempfile.TemporaryDirectory() as tmpdir:
            data1 = CheckpointData(model_state=make_dummy_state_dict(), step=1)
            data2 = CheckpointData(model_state=make_dummy_state_dict(), step=2)

            path1 = Path(tmpdir) / "ckpt1.pt"
            path2 = Path(tmpdir) / "ckpt2.pt"
            save_checkpoint(data1, path1)
            save_checkpoint(data2, path2)

            comparison = compare_checkpoints(path1, path2)

        # Random states should differ
        assert comparison["differing_weights"] > 0

    def test_different_keys(self):
        """Test comparing checkpoints with different keys."""
        state1 = {"layer1.weight": torch.randn(10, 10)}
        state2 = {"layer2.weight": torch.randn(10, 10)}

        with tempfile.TemporaryDirectory() as tmpdir:
            data1 = CheckpointData(model_state=state1)
            data2 = CheckpointData(model_state=state2)

            path1 = Path(tmpdir) / "ckpt1.pt"
            path2 = Path(tmpdir) / "ckpt2.pt"
            save_checkpoint(data1, path1)
            save_checkpoint(data2, path2)

            comparison = compare_checkpoints(path1, path2)

        assert "layer1.weight" in comparison["keys_only_in_1"]
        assert "layer2.weight" in comparison["keys_only_in_2"]


# =============================================================================
# Test export_for_inference
# =============================================================================

class TestExportForInference:
    """Tests for export_for_inference utility."""

    def test_export_removes_optimizer(self):
        """Test that export removes optimizer state."""
        model = make_dummy_model()
        optimizer = torch.optim.AdamW(model.parameters())

        # Take a step to populate optimizer state
        loss = sum(p.sum() for p in model.parameters())
        loss.backward()
        optimizer.step()

        data = CheckpointData(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            full_path = Path(tmpdir) / "full.pt"
            export_path = Path(tmpdir) / "export.pt"

            save_checkpoint(data, full_path)
            export_for_inference(full_path, export_path)

            exported = load_checkpoint(export_path)

        assert exported.optimizer_state is None

    def test_export_preserves_weights(self):
        """Test that export preserves model weights."""
        state = make_dummy_state_dict()
        data = CheckpointData(model_state=state)

        with tempfile.TemporaryDirectory() as tmpdir:
            full_path = Path(tmpdir) / "full.pt"
            export_path = Path(tmpdir) / "export.pt"

            save_checkpoint(data, full_path)
            export_for_inference(full_path, export_path)

            exported = load_checkpoint(export_path)

        # Weights should match exactly
        for key in state:
            assert torch.equal(exported.model_state[key], state[key])


# =============================================================================
# Test compute_checkpoint_hash
# =============================================================================

class TestComputeCheckpointHash:
    """Tests for compute_checkpoint_hash utility."""

    def test_hash_is_deterministic(self):
        """Test that hash is deterministic for same file."""
        data = CheckpointData(model_state=make_dummy_state_dict())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)

            hash1 = compute_checkpoint_hash(path)
            hash2 = compute_checkpoint_hash(path)

        assert hash1 == hash2

    def test_different_files_different_hash(self):
        """Test that different files have different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data1 = CheckpointData(model_state=make_dummy_state_dict())
            data2 = CheckpointData(model_state=make_dummy_state_dict())

            path1 = Path(tmpdir) / "test1.pt"
            path2 = Path(tmpdir) / "test2.pt"
            save_checkpoint(data1, path1)
            save_checkpoint(data2, path2)

            hash1 = compute_checkpoint_hash(path1)
            hash2 = compute_checkpoint_hash(path2)

        # Different random states should produce different hashes
        assert hash1 != hash2

    def test_hash_format(self):
        """Test that hash is a valid hex string."""
        data = CheckpointData(model_state=make_dummy_state_dict())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)

            hash_value = compute_checkpoint_hash(path)

        # SHA256 produces 64 hex characters
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)


# =============================================================================
# Test Edge Cases
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
            "big_weight": torch.randn(1000, 1000),
            "small_weight": torch.randn(10),
        }
        data = CheckpointData(model_state=large_state)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            loaded = load_checkpoint(path)

        assert loaded.model_state["big_weight"].shape == (1000, 1000)

    def test_special_float_values(self):
        """Test handling special float values (inf, nan)."""
        state = {
            "inf_tensor": torch.tensor([float("inf"), float("-inf")]),
            "nan_tensor": torch.tensor([float("nan")]),
        }
        data = CheckpointData(model_state=state)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            loaded = load_checkpoint(path)

        assert torch.isinf(loaded.model_state["inf_tensor"]).any()
        assert torch.isnan(loaded.model_state["nan_tensor"]).any()

    def test_various_dtypes(self):
        """Test handling various tensor dtypes."""
        state = {
            "float32": torch.randn(10, dtype=torch.float32),
            "float64": torch.randn(10, dtype=torch.float64),
            "int32": torch.randint(0, 100, (10,), dtype=torch.int32),
            "int64": torch.randint(0, 100, (10,), dtype=torch.int64),
            "bool": torch.tensor([True, False, True]),
        }
        data = CheckpointData(model_state=state)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(data, path)
            loaded = load_checkpoint(path)

        for key, original in state.items():
            assert loaded.model_state[key].dtype == original.dtype

    def test_nested_config_preserved(self):
        """Test that nested config dicts are preserved."""
        config = {
            "model": {
                "hidden_dim": 128,
                "num_layers": 4,
            },
            "training": {
                "lr": 1e-4,
                "batch_size": 32,
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

        assert loaded.config == config
        assert loaded.config["model"]["hidden_dim"] == 128