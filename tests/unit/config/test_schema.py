"""
Tests for lacuna.config.schema

Verify config validation and defaults.
"""

import pytest
import torch
from lacuna.config.schema import (
    LacunaConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    GeneratorConfig,
)


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.max_cols == 32
        assert cfg.normalization == "robust"
    
    def test_invalid_normalization_raises(self):
        with pytest.raises(ValueError, match="Invalid normalization"):
            DataConfig(normalization="invalid")
    
    def test_invalid_n_range_raises(self):
        with pytest.raises(ValueError, match="n_range"):
            DataConfig(n_range=(500, 50))  # Backwards


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.hidden_dim == 128
        assert cfg.n_layers == 4
    
    def test_hidden_dim_divisibility_check(self):
        with pytest.raises(ValueError, match="divisible"):
            ModelConfig(hidden_dim=100, n_heads=3)  # 100 % 3 != 0
    
    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            ModelConfig(dropout=1.5)


class TestTrainingConfig:
    """Tests for TrainingConfig."""
    
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.batch_size == 64
        assert cfg.lr == 1e-4
    
    def test_invalid_lr_raises(self):
        with pytest.raises(ValueError, match="lr"):
            TrainingConfig(lr=-0.001)
    
    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=0)


class TestGeneratorConfig:
    """Tests for GeneratorConfig."""
    
    def test_defaults(self):
        cfg = GeneratorConfig()
        assert cfg.n_generators == 6
    
    def test_invalid_class_balance_raises(self):
        with pytest.raises(ValueError, match="sum to 1"):
            GeneratorConfig(class_balance=(0.5, 0.5, 0.5))


class TestLacunaConfig:
    """Tests for LacunaConfig."""
    
    def test_default_construction(self):
        cfg = LacunaConfig()
        assert cfg.seed == 42
        assert cfg.device == "cuda"
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.model, ModelConfig)
    
    def test_minimal_factory(self):
        cfg = LacunaConfig.minimal()
        assert cfg.model.hidden_dim == 64
        assert cfg.training.epochs == 5
    
    def test_loss_matrix_tensor(self):
        cfg = LacunaConfig()
        L = cfg.get_loss_matrix_tensor()
        
        assert L.shape == (3, 3)
        assert L[0, 0] == 0.0  # Green, MCAR
        assert L[0, 2] == 10.0  # Green, MNAR
        assert L[2, 2] == 0.0  # Red, MNAR