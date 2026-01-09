"""
Tests for lacuna.generators.base
"""

import pytest
import torch
from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams


class ConcreteGenerator(Generator):
    """Concrete implementation for testing."""
    
    def sample(self, rng, n, d):
        X = rng.randn(n, d)
        R = torch.ones(n, d, dtype=torch.bool)
        R[:, 0] = rng.rand(n) > 0.3  # 30% missing in first column
        if R.sum() == 0:
            R[0, 0] = True
        return X, R


class TestGeneratorBase:
    """Tests for Generator base class."""
    
    def test_construction(self):
        params = GeneratorParams(miss_rate=0.3)
        gen = ConcreteGenerator(
            generator_id=0,
            name="test-gen",
            class_id=MCAR,
            params=params,
        )
        
        assert gen.generator_id == 0
        assert gen.name == "test-gen"
        assert gen.class_id == MCAR
        assert gen.params == params
    
    def test_invalid_class_id_raises(self):
        params = GeneratorParams()
        with pytest.raises(ValueError, match="class_id must be"):
            ConcreteGenerator(0, "test", class_id=5, params=params)
    
    def test_negative_generator_id_raises(self):
        params = GeneratorParams()
        with pytest.raises(ValueError, match="generator_id must be"):
            ConcreteGenerator(-1, "test", class_id=MCAR, params=params)
    
    def test_sample_shapes(self):
        gen = ConcreteGenerator(0, "test", MCAR, GeneratorParams())
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=100, d=5)
        
        assert X.shape == (100, 5)
        assert R.shape == (100, 5)
        assert R.dtype == torch.bool
    
    def test_sample_observed(self):
        gen = ConcreteGenerator(0, "test", MCAR, GeneratorParams())
        rng = RNGState(seed=42)
        
        ds = gen.sample_observed(rng, n=50, d=3, dataset_id="test_001")
        
        assert ds.n == 50
        assert ds.d == 3
        assert ds.dataset_id == "test_001"
        assert ds.meta["generator_id"] == 0
        assert ds.meta["generator_name"] == "test"
    
    def test_sample_observed_zeros_missing(self):
        gen = ConcreteGenerator(0, "test", MCAR, GeneratorParams())
        rng = RNGState(seed=42)
        
        ds = gen.sample_observed(rng, n=100, d=5, dataset_id="test")
        
        # Check that missing values are zeroed
        missing_mask = ~ds.r
        if missing_mask.any():
            assert (ds.x[missing_mask] == 0).all()
    
    def test_repr(self):
        gen = ConcreteGenerator(5, "my-generator", MAR, GeneratorParams())
        r = repr(gen)
        assert "ConcreteGenerator" in r
        assert "id=5" in r
        assert "my-generator" in r
