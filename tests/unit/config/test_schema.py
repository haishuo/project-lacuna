"""
Tests for lacuna.config.schema
"""

import pytest
import torch
from lacuna.config.schema import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    GeneratorConfig,
    LacunaConfig,
)


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.max_cols == 32
        assert cfg.max_rows == 256
        assert cfg.n_range == (50, 500)
        assert cfg.d_range == (3, 20)
    
    def test_invalid_n_range_raises(self):
        with pytest.raises(ValueError):
            DataConfig(n_range=(100, 50))  # min > max
    
    def test_invalid_d_range_raises(self):
        with pytest.raises(ValueError):
            DataConfig(d_range=(20, 5))  # min > max


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.hidden_dim == 128
        assert cfg.evidence_dim == 64
        assert cfg.n_layers == 4
        assert cfg.n_heads == 4
    
    def test_invalid_hidden_dim_raises(self):
        with pytest.raises(ValueError):
            ModelConfig(hidden_dim=100, n_heads=8)  # Not divisible
    
    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError):
            ModelConfig(dropout=1.5)


class TestTrainingConfig:
    """Tests for TrainingConfig."""
    
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.batch_size == 32
        assert cfg.lr == 1e-4
        assert cfg.epochs == 50
    
    def test_invalid_lr_raises(self):
        with pytest.raises(ValueError):
            TrainingConfig(lr=-0.001)
    
    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)


class TestGeneratorConfig:
    """Tests for GeneratorConfig."""
    
    def test_defaults(self):
        cfg = GeneratorConfig()
        assert cfg.n_generators == 6


class TestLacunaConfig:
    """Tests for LacunaConfig."""
    
    def test_defaults(self):
        cfg = LacunaConfig()
        assert cfg.seed == 42
        assert cfg.device == "cuda"
    
    def test_loss_matrix_tensor(self):
        cfg = LacunaConfig()
        tensor = cfg.get_loss_matrix_tensor()
        
        assert tensor.shape == (3, 3)
        assert tensor.dtype == torch.float32
        # Check specific value
        assert tensor[0, 2].item() == 1.0
    
    def test_minimal(self):
        cfg = LacunaConfig.minimal()
        assert cfg.device == "cpu"
        assert cfg.data.max_cols == 16
        assert cfg.data.max_rows == 64
        assert cfg.model.hidden_dim == 64
