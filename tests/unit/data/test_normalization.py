"""
Tests for lacuna.data.normalization
"""

import pytest
import torch
from lacuna.data.observed import create_observed_dataset
from lacuna.data.normalization import (
    compute_normalization_stats,
    normalize_dataset,
    denormalize_values,
    NormalizationStats,
)


class TestComputeNormalizationStats:
    """Tests for compute_normalization_stats."""
    
    def test_standard_stats(self):
        x = torch.randn(1000, 3) * 2 + 5  # mean=5, std=2
        r = torch.ones(1000, 3, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        stats = compute_normalization_stats(ds, method="standard")
        
        assert stats.method == "standard"
        assert abs(stats.center[0].item() - 5.0) < 0.2
        assert abs(stats.scale[0].item() - 2.0) < 0.2
    
    def test_robust_stats(self):
        x = torch.randn(1000, 3)
        r = torch.ones(1000, 3, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        stats = compute_normalization_stats(ds, method="robust")
        
        assert stats.method == "robust"
        # Median should be close to 0 for standard normal
        assert abs(stats.center[0].item()) < 0.2
    
    def test_uses_only_observed(self):
        x = torch.zeros(100, 2)
        x[:50, 0] = 10.0  # First 50 are 10
        x[50:, 0] = 0.0   # Last 50 are 0 (will be missing)
        
        r = torch.ones(100, 2, dtype=torch.bool)
        r[50:, 0] = False  # Last 50 missing
        
        ds = create_observed_dataset(x=x, r=r)
        
        stats = compute_normalization_stats(ds, method="standard")
        
        # Mean should be 10 (only observed values)
        assert abs(stats.center[0].item() - 10.0) < 0.01


class TestNormalizeDataset:
    """Tests for normalize_dataset."""
    
    def test_normalize_centers_data(self):
        x = torch.randn(100, 3) * 2 + 5
        r = torch.ones(100, 3, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        stats = compute_normalization_stats(ds, method="standard")
        ds_norm = normalize_dataset(ds, stats)
        
        # Normalized data should have mean ~0, std ~1
        assert abs(ds_norm.x.mean().item()) < 0.2
        assert abs(ds_norm.x.std().item() - 1.0) < 0.2
    
    def test_normalize_preserves_missingness(self):
        x = torch.randn(50, 3)
        r = torch.ones(50, 3, dtype=torch.bool)
        r[:10, 0] = False
        
        ds = create_observed_dataset(x=x, r=r)
        stats = compute_normalization_stats(ds)
        ds_norm = normalize_dataset(ds, stats)
        
        # Missing values should still be 0
        assert (ds_norm.x[~ds_norm.r] == 0).all()
    
    def test_normalize_returns_new_dataset(self):
        x = torch.randn(50, 3)
        r = torch.ones(50, 3, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        stats = compute_normalization_stats(ds)
        ds_norm = normalize_dataset(ds, stats)
        
        # Should be different objects
        assert ds is not ds_norm
        assert not torch.equal(ds.x, ds_norm.x)


class TestDenormalizeValues:
    """Tests for denormalize_values."""
    
    def test_roundtrip(self):
        x = torch.randn(100, 3) * 2 + 5
        r = torch.ones(100, 3, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        stats = compute_normalization_stats(ds, method="standard")
        ds_norm = normalize_dataset(ds, stats)
        
        # Denormalize
        x_recovered = denormalize_values(ds_norm.x, stats)
        
        # Should match original (approximately, due to re-zeroing)
        observed_mask = ds.r
        assert torch.allclose(x_recovered[observed_mask], ds.x[observed_mask], atol=1e-5)
