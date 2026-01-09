"""
Tests for lacuna.data.features
"""

import pytest
import torch
from lacuna.data.observed import create_observed_dataset
from lacuna.data.features import extract_column_features, FEATURE_DIM


class TestExtractColumnFeatures:
    """Tests for extract_column_features."""
    
    def test_output_shape(self):
        x = torch.randn(100, 5)
        r = torch.ones(100, 5, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        features = extract_column_features(ds)
        
        assert features.shape == (5, FEATURE_DIM)
    
    def test_output_dtype(self):
        x = torch.randn(50, 3)
        r = torch.ones(50, 3, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        features = extract_column_features(ds)
        
        assert features.dtype == torch.float32
    
    def test_no_nan_in_features(self):
        x = torch.randn(100, 5)
        r = torch.ones(100, 5, dtype=torch.bool)
        r[:20, 0] = False  # 20% missing in first column
        
        ds = create_observed_dataset(x=x, r=r)
        features = extract_column_features(ds)
        
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()
    
    def test_missingness_feature_correct(self):
        x = torch.randn(100, 3)
        r = torch.ones(100, 3, dtype=torch.bool)
        r[:30, 0] = False  # 30% missing in column 0
        
        ds = create_observed_dataset(x=x, r=r)
        features = extract_column_features(ds)
        
        # First feature is missingness rate
        assert abs(features[0, 0].item() - 0.3) < 0.01
        assert abs(features[1, 0].item() - 0.0) < 0.01  # Column 1 fully observed
    
    def test_different_datasets_different_features(self):
        # Dataset 1: low missingness
        x1 = torch.randn(100, 3)
        r1 = torch.ones(100, 3, dtype=torch.bool)
        r1[:5, 0] = False
        ds1 = create_observed_dataset(x=x1, r=r1)
        
        # Dataset 2: high missingness
        x2 = torch.randn(100, 3)
        r2 = torch.ones(100, 3, dtype=torch.bool)
        r2[:50, 0] = False
        ds2 = create_observed_dataset(x=x2, r=r2)
        
        f1 = extract_column_features(ds1)
        f2 = extract_column_features(ds2)
        
        # Features should differ
        assert not torch.allclose(f1, f2)
    
    def test_handles_all_observed(self):
        x = torch.randn(50, 3)
        r = torch.ones(50, 3, dtype=torch.bool)  # All observed
        ds = create_observed_dataset(x=x, r=r)
        
        features = extract_column_features(ds)
        
        # Should work without errors
        assert features.shape == (3, FEATURE_DIM)
        # Missingness rate should be 0
        assert features[0, 0].item() == 0.0
