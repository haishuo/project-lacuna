"""
Tests for lacuna.core.rng

Verify RNG reproducibility and independence.
"""

import pytest
import torch
import numpy as np
from lacuna.core.rng import RNGState


class TestRNGReproducibility:
    """Same seed produces same results."""
    
    def test_randn_reproducible(self):
        rng1 = RNGState(seed=42)
        rng2 = RNGState(seed=42)
        
        x1 = rng1.randn(10, 5)
        x2 = rng2.randn(10, 5)
        
        assert torch.allclose(x1, x2)
    
    def test_rand_reproducible(self):
        rng1 = RNGState(seed=123)
        rng2 = RNGState(seed=123)
        
        x1 = rng1.rand(100)
        x2 = rng2.rand(100)
        
        assert torch.allclose(x1, x2)
    
    def test_randint_reproducible(self):
        rng1 = RNGState(seed=456)
        rng2 = RNGState(seed=456)
        
        x1 = rng1.randint(0, 100, (50,))
        x2 = rng2.randint(0, 100, (50,))
        
        assert torch.equal(x1, x2)
    
    def test_choice_reproducible(self):
        rng1 = RNGState(seed=789)
        rng2 = RNGState(seed=789)
        
        x1 = rng1.choice(100, size=20, replace=False)
        x2 = rng2.choice(100, size=20, replace=False)
        
        np.testing.assert_array_equal(x1, x2)


class TestRNGIndependence:
    """Different seeds produce different results."""
    
    def test_different_seeds_different_randn(self):
        rng1 = RNGState(seed=1)
        rng2 = RNGState(seed=2)
        
        x1 = rng1.randn(100)
        x2 = rng2.randn(100)
        
        assert not torch.allclose(x1, x2)


class TestRNGSpawn:
    """Child RNGs are independent but deterministic."""
    
    def test_spawn_deterministic(self):
        rng1 = RNGState(seed=42)
        rng2 = RNGState(seed=42)
        
        child1 = rng1.spawn()
        child2 = rng2.spawn()
        
        x1 = child1.randn(10)
        x2 = child2.randn(10)
        
        assert torch.allclose(x1, x2)
    
    def test_spawn_independent_from_parent(self):
        rng = RNGState(seed=42)
        child = rng.spawn()
        
        x_parent = rng.randn(100)
        x_child = child.randn(100)
        
        assert not torch.allclose(x_parent, x_child)
    
    def test_multiple_spawns_independent(self):
        rng = RNGState(seed=42)
        
        children = rng.spawn_many(5)
        outputs = [c.randn(50) for c in children]
        
        # All pairs should be different
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j])
    
    def test_spawn_many_deterministic(self):
        rng1 = RNGState(seed=42)
        rng2 = RNGState(seed=42)
        
        children1 = rng1.spawn_many(3)
        children2 = rng2.spawn_many(3)
        
        for c1, c2 in zip(children1, children2):
            x1 = c1.randn(10)
            x2 = c2.randn(10)
            assert torch.allclose(x1, x2)


class TestRNGShapes:
    """Output shapes are correct."""
    
    def test_randn_shape(self):
        rng = RNGState(seed=42)
        x = rng.randn(3, 4, 5)
        assert x.shape == (3, 4, 5)
    
    def test_rand_shape(self):
        rng = RNGState(seed=42)
        x = rng.rand(10, 20)
        assert x.shape == (10, 20)
    
    def test_randint_shape(self):
        rng = RNGState(seed=42)
        x = rng.randint(0, 10, (5, 5))
        assert x.shape == (5, 5)


class TestRNGDtypes:
    """Dtype handling is correct."""
    
    def test_randn_float32_default(self):
        rng = RNGState(seed=42)
        x = rng.randn(10)
        assert x.dtype == torch.float32
    
    def test_randn_float64(self):
        rng = RNGState(seed=42)
        x = rng.randn(10, dtype=torch.float64)
        assert x.dtype == torch.float64
