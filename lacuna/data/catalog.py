"""
lacuna.data.catalog

Dataset catalog and registry.

Provides a centralized way to discover and load datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
import json

from .ingestion import RawDataset, load_csv, load_parquet, load_sklearn_dataset


# Default data directories
DEFAULT_RAW_DIR = Path("/mnt/data/lacuna/raw")
DEFAULT_PROCESSED_DIR = Path("/mnt/data/lacuna/processed")


@dataclass
class DatasetInfo:
    """Metadata about a dataset in the catalog."""
    name: str
    source_type: str  # "csv", "parquet", "sklearn", "url"
    path_or_name: str  # File path or sklearn dataset name
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    target_column: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)


class DatasetCatalog:
    """Registry of available datasets.
    
    Usage:
        catalog = DatasetCatalog()
        catalog.register_sklearn("breast_cancer")
        catalog.register_csv("/mnt/data/lacuna/raw/my_data.csv", name="my_data")
        
        # List available datasets
        print(catalog.list_datasets())
        
        # Load a dataset
        raw = catalog.load("breast_cancer")
    """
    
    def __init__(self, raw_dir: Path = DEFAULT_RAW_DIR):
        self.raw_dir = Path(raw_dir)
        self._datasets: Dict[str, DatasetInfo] = {}
    
    def register_sklearn(self, name: str, description: str = "") -> None:
        """Register a sklearn built-in dataset."""
        self._datasets[name] = DatasetInfo(
            name=name,
            source_type="sklearn",
            path_or_name=name,
            description=description or f"sklearn.datasets.load_{name}",
            tags=["sklearn", "builtin"],
        )
    
    def register_csv(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        target_column: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Register a CSV file."""
        path = Path(path)
        name = name or path.stem
        
        self._datasets[name] = DatasetInfo(
            name=name,
            source_type="csv",
            path_or_name=str(path),
            target_column=target_column,
            description=description,
            tags=tags or ["csv"],
        )
    
    def register_parquet(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        target_column: Optional[str] = None,
        description: str = "",
    ) -> None:
        """Register a Parquet file."""
        path = Path(path)
        name = name or path.stem
        
        self._datasets[name] = DatasetInfo(
            name=name,
            source_type="parquet",
            path_or_name=str(path),
            target_column=target_column,
            description=description,
            tags=["parquet"],
        )
    
    def scan_directory(self, directory: Optional[Path] = None) -> int:
        """Scan directory for CSV/Parquet files and register them.
        
        Returns number of datasets found.
        """
        directory = directory or self.raw_dir
        if not directory.exists():
            return 0
        
        count = 0
        for f in directory.iterdir():
            if f.suffix == ".csv":
                self.register_csv(f)
                count += 1
            elif f.suffix == ".parquet":
                self.register_parquet(f)
                count += 1
        
        return count
    
    def list_datasets(self) -> List[str]:
        """Return list of registered dataset names."""
        return sorted(self._datasets.keys())
    
    def get_info(self, name: str) -> DatasetInfo:
        """Get metadata for a dataset."""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not in catalog. Available: {self.list_datasets()}")
        return self._datasets[name]
    
    def load(self, name: str) -> RawDataset:
        """Load a dataset by name.
        
        Args:
            name: Registered dataset name.
        
        Returns:
            RawDataset ready for use.
        """
        info = self.get_info(name)
        
        if info.source_type == "sklearn":
            return load_sklearn_dataset(info.path_or_name)
        elif info.source_type == "csv":
            return load_csv(info.path_or_name, target_column=info.target_column, name=name)
        elif info.source_type == "parquet":
            return load_parquet(info.path_or_name, target_column=info.target_column, name=name)
        else:
            raise ValueError(f"Unknown source type: {info.source_type}")
    
    def __len__(self) -> int:
        return len(self._datasets)
    
    def __contains__(self, name: str) -> bool:
        return name in self._datasets


def create_default_catalog() -> DatasetCatalog:
    """Create catalog with sklearn datasets pre-registered."""
    catalog = DatasetCatalog()
    
    # Register sklearn built-ins
    catalog.register_sklearn("breast_cancer", "Wisconsin breast cancer dataset (classification)")
    catalog.register_sklearn("diabetes", "Diabetes dataset (regression)")
    catalog.register_sklearn("wine", "Wine dataset (classification)")
    catalog.register_sklearn("iris", "Iris dataset (classification)")
    
    # Scan raw directory for any user-added files
    catalog.scan_directory()
    
    return catalog
