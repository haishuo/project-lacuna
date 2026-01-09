"""
Tests for lacuna.data.observed
"""

import pytest
import torch
import numpy as np
from lacuna.data.observed import create_observed_dataset, from_numpy, split_dataset


class TestCreateObservedDataset:
    """Tests for create_observed_dataset."""
    
    def test_basic_creation(self):
        x = torch.randn(100, 5)
        r = torch.ones(100, 5, dtype=torch.bool)
        r[:10, 0] = False  # 10% missing in first column
        
        ds = create_observed_dataset(x=x, r=r, dataset_id="test")
        
        assert ds.n == 100
        assert ds.d == 5
        assert ds.dataset_id == "test"
    
    def test_infer_missingness_from_nan(self):
        x = torch.randn(50, 3)
        x[0, 0] = float('nan')
        x[1, 1] = float('nan')
        
        ds = create_observed_dataset(x=x, dataset_id="nan_test")
        
        assert ds.r[0, 0] == False
        assert ds.r[1, 1] == False
        assert ds.r[0, 1] == True
        # NaN should be replaced with 0
        assert ds.x[0, 0] == 0.0
    
    def test_auto_feature_names(self):
        x = torch.randn(10, 4)
        ds = create_observed_dataset(x=x)
        
        assert ds.feature_names == ("col_0", "col_1", "col_2", "col_3")
    
    def test_custom_feature_names(self):
        x = torch.randn(10, 3)
        ds = create_observed_dataset(
            x=x,
            feature_names=("age", "weight", "height"),
        )
        
        assert ds.feature_names == ("age", "weight", "height")
    
    def test_dtype_conversion(self):
        x = torch.randn(10, 3).double()  # float64
        r = torch.ones(10, 3, dtype=torch.int32)  # int32
        
        ds = create_observed_dataset(x=x, r=r)
        
        assert ds.x.dtype == torch.float32
        assert ds.r.dtype == torch.bool
    
    def test_missing_values_zeroed(self):
        x = torch.randn(10, 3)
        r = torch.ones(10, 3, dtype=torch.bool)
        r[0, 0] = False
        
        ds = create_observed_dataset(x=x, r=r)
        
        assert ds.x[0, 0] == 0.0


class TestFromNumpy:
    """Tests for from_numpy."""
    
    def test_basic_conversion(self):
        x = np.random.randn(50, 4).astype(np.float32)
        r = np.ones((50, 4), dtype=bool)
        
        ds = from_numpy(x=x, r=r, dataset_id="numpy_test")
        
        assert ds.n == 50
        assert ds.d == 4
        assert ds.x.dtype == torch.float32
    
    def test_nan_handling(self):
        x = np.random.randn(20, 3)
        x[0, 0] = np.nan
        
        ds = from_numpy(x=x)
        
        assert ds.r[0, 0] == False
        assert ds.x[0, 0] == 0.0


class TestSplitDataset:
    """Tests for split_dataset."""
    
    def test_split_sizes(self):
        x = torch.randn(100, 5)
        r = torch.ones(100, 5, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r, dataset_id="full")
        
        train, val = split_dataset(ds, train_frac=0.8)
        
        assert train.n == 80
        assert val.n == 20
    
    def test_split_reproducible(self):
        x = torch.randn(100, 5)
        r = torch.ones(100, 5, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        train1, val1 = split_dataset(ds, seed=42)
        train2, val2 = split_dataset(ds, seed=42)
        
        assert torch.equal(train1.x, train2.x)
        assert torch.equal(val1.x, val2.x)
    
    def test_split_different_seeds(self):
        x = torch.randn(100, 5)
        r = torch.ones(100, 5, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        train1, _ = split_dataset(ds, seed=1)
        train2, _ = split_dataset(ds, seed=2)
        
        assert not torch.equal(train1.x, train2.x)
    
    def test_split_preserves_feature_names(self):
        x = torch.randn(50, 3)
        r = torch.ones(50, 3, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r, feature_names=("a", "b", "c"))
        
        train, val = split_dataset(ds)
        
        assert train.feature_names == ("a", "b", "c")
        assert val.feature_names == ("a", "b", "c")
