"""
Tests for lacuna.data.catalog
"""

import pytest
import tempfile
from pathlib import Path

from lacuna.data.catalog import (
    DatasetCatalog,
    DatasetInfo,
    create_default_catalog,
)


class TestDatasetCatalog:
    """Tests for DatasetCatalog."""
    
    def test_register_sklearn(self):
        catalog = DatasetCatalog()
        catalog.register_sklearn("breast_cancer")
        
        assert "breast_cancer" in catalog
        assert "breast_cancer" in catalog.list_datasets()
    
    def test_register_csv(self):
        catalog = DatasetCatalog()
        catalog.register_csv("/fake/path.csv", name="my_data")
        
        assert "my_data" in catalog
        info = catalog.get_info("my_data")
        assert info.source_type == "csv"
    
    def test_load_sklearn(self):
        catalog = DatasetCatalog()
        catalog.register_sklearn("iris")
        
        raw = catalog.load("iris")
        
        assert raw.n == 150
        assert raw.d == 4
    
    def test_load_unknown_raises(self):
        catalog = DatasetCatalog()
        
        with pytest.raises(KeyError, match="not in catalog"):
            catalog.load("nonexistent")
    
    def test_scan_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some CSV files
            (Path(tmpdir) / "data1.csv").write_text("a,b\n1,2\n")
            (Path(tmpdir) / "data2.csv").write_text("x,y\n3,4\n")
            (Path(tmpdir) / "not_a_csv.txt").write_text("hello")
            
            catalog = DatasetCatalog(raw_dir=tmpdir)
            count = catalog.scan_directory()
            
            assert count == 2
            assert "data1" in catalog
            assert "data2" in catalog
            assert "not_a_csv" not in catalog


class TestCreateDefaultCatalog:
    """Tests for create_default_catalog."""
    
    def test_has_sklearn_datasets(self):
        catalog = create_default_catalog()
        
        assert "breast_cancer" in catalog
        assert "diabetes" in catalog
        assert "wine" in catalog
        assert "iris" in catalog
    
    def test_can_load_all_sklearn(self):
        catalog = create_default_catalog()
        
        for name in ["breast_cancer", "diabetes", "wine", "iris"]:
            raw = catalog.load(name)
            assert raw.n > 0
            assert raw.d > 0
