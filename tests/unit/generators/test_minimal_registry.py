"""
Tests for create_minimal_registry factory.
"""

import pytest
from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.generators import create_minimal_registry


class TestMinimalRegistry:
    """Tests for the minimal 6-generator registry."""
    
    def test_has_six_generators(self):
        registry = create_minimal_registry()
        assert registry.K == 6
    
    def test_class_balance(self):
        registry = create_minimal_registry()
        counts = registry.class_counts()
        
        assert counts[MCAR] == 2
        assert counts[MAR] == 2
        assert counts[MNAR] == 2
    
    def test_all_generators_sample_correctly(self):
        registry = create_minimal_registry()
        rng = RNGState(seed=42)
        
        for gen in registry:
            X, R = gen.sample(rng.spawn(), n=50, d=5)
            
            assert X.shape == (50, 5)
            assert R.shape == (50, 5)
            assert R.dtype == torch.bool
            assert R.sum() >= 1
    
    def test_generators_produce_observed_datasets(self):
        registry = create_minimal_registry()
        rng = RNGState(seed=42)
        
        for gen in registry:
            ds = gen.sample_observed(rng.spawn(), n=50, d=5, dataset_id=f"test_{gen.generator_id}")
            
            assert ds.n == 50
            assert ds.d == 5
            assert ds.meta["generator_id"] == gen.generator_id


# Need this import for the test
import torch
