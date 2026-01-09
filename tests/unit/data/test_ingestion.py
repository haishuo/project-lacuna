"""
Tests for lacuna.data.ingestion
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from lacuna.data.ingestion import (
    RawDataset,
    load_csv,
    load_sklearn_dataset,
)


class TestRawDataset:
    """Tests for RawDataset."""
    
    def test_construction(self):
        data = np.random.randn(100, 5)
        raw = RawDataset(
            data=data,
            feature_names=("a", "b", "c", "d", "e"),
            name="test",
        )
        
        assert raw.n == 100
        assert raw.d == 5
        assert raw.feature_names == ("a", "b", "c", "d", "e")
    
    def test_to_observed_dataset(self):
        data = np.random.randn(50, 3)
        raw = RawDataset(
            data=data,
            feature_names=("x", "y", "z"),
            name="test",
        )
        
        ds = raw.to_observed_dataset()
        
        assert ds.n == 50
        assert ds.d == 3
        assert ds.r.all()  # All observed (complete data)
        assert ds.feature_names == ("x", "y", "z")


class TestLoadCsv:
    """Tests for load_csv."""
    
    def test_load_simple_csv(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("a,b,c\n")
            f.write("1.0,2.0,3.0\n")
            f.write("4.0,5.0,6.0\n")
            f.write("7.0,8.0,9.0\n")
            path = f.name
        
        try:
            raw = load_csv(path)
            
            assert raw.n == 3
            assert raw.d == 3
            assert raw.feature_names == ("a", "b", "c")
        finally:
            Path(path).unlink()
    
    def test_load_csv_with_target(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("feature1,feature2,target\n")
            f.write("1.0,2.0,0\n")
            f.write("3.0,4.0,1\n")
            path = f.name
        
        try:
            raw = load_csv(path, target_column="target")
            
            assert raw.d == 2  # Target removed from features
            assert raw.target is not None
            assert raw.target_name == "target"
            assert "target" not in raw.feature_names
        finally:
            Path(path).unlink()
    
    def test_load_csv_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path.csv")


class TestLoadSklearn:
    """Tests for load_sklearn_dataset."""
    
    def test_load_breast_cancer(self):
        raw = load_sklearn_dataset("breast_cancer")
        
        assert raw.n == 569
        assert raw.d == 30
        assert raw.target is not None
        assert raw.source == "sklearn:breast_cancer"
        assert raw.name == "breast_cancer"
    
    def test_load_diabetes(self):
        raw = load_sklearn_dataset("diabetes")
        
        assert raw.n == 442
        assert raw.d == 10
    
    def test_load_wine(self):
        raw = load_sklearn_dataset("wine")
        
        assert raw.n == 178
        assert raw.d == 13
    
    def test_load_iris(self):
        raw = load_sklearn_dataset("iris")
        
        assert raw.n == 150
        assert raw.d == 4
    
    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown sklearn dataset"):
            load_sklearn_dataset("nonexistent")
    
    def test_converts_to_observed(self):
        raw = load_sklearn_dataset("iris")
        ds = raw.to_observed_dataset()
        
        assert ds.n == 150
        assert ds.d == 4
        assert ds.r.all()  # Complete data
