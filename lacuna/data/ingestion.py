"""
lacuna.data.ingestion

Load external datasets from various sources.

Supported sources:
- CSV files
- Parquet files
- sklearn built-in datasets
- UCI ML Repository (via URL)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

from lacuna.core.types import ObservedDataset
from .observed import create_observed_dataset


@dataclass
class RawDataset:
    """Container for raw loaded data before conversion to ObservedDataset.
    
    This intermediate representation allows inspection and preprocessing
    before creating the final ObservedDataset.
    """
    data: np.ndarray          # [n, d] numeric data
    feature_names: Tuple[str, ...]
    target: Optional[np.ndarray] = None  # Optional target variable
    target_name: Optional[str] = None
    source: str = "unknown"
    name: str = "unnamed"
    
    @property
    def n(self) -> int:
        return self.data.shape[0]
    
    @property
    def d(self) -> int:
        return self.data.shape[1]
    
    def to_observed_dataset(self, dataset_id: Optional[str] = None) -> ObservedDataset:
        """Convert to ObservedDataset (assumes complete data)."""
        x = torch.from_numpy(self.data.astype(np.float32))
        r = torch.ones(self.n, self.d, dtype=torch.bool)
        
        return create_observed_dataset(
            x=x,
            r=r,
            feature_names=self.feature_names,
            dataset_id=dataset_id or self.name,
            meta={
                "source": self.source,
                "has_target": self.target is not None,
                "target_name": self.target_name,
            },
        )


def load_csv(
    path: Union[str, Path],
    target_column: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> RawDataset:
    """Load dataset from CSV file.
    
    Args:
        path: Path to CSV file.
        target_column: Column to use as target (removed from features).
        drop_columns: Columns to drop entirely.
        name: Dataset name (defaults to filename).
    
    Returns:
        RawDataset with numeric columns only.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    
    df = pd.read_csv(path)
    
    # Drop specified columns
    if drop_columns:
        df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    
    # Extract target if specified
    target = None
    target_name = None
    if target_column and target_column in df.columns:
        target = df[target_column].values
        target_name = target_column
        df = df.drop(columns=[target_column])
    
    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {path}")
    
    # Handle missing values - for now, drop rows with NaN
    # (semi-synthetic will add missingness back)
    n_before = len(numeric_df)
    numeric_df = numeric_df.dropna()
    n_after = len(numeric_df)
    
    if n_after < n_before:
        print(f"  Dropped {n_before - n_after} rows with missing values")
    
    if target is not None:
        # Also filter target to match
        valid_idx = numeric_df.index
        target = target[valid_idx]
    
    return RawDataset(
        data=numeric_df.values,
        feature_names=tuple(numeric_df.columns),
        target=target,
        target_name=target_name,
        source=f"csv:{path}",
        name=name or path.stem,
    )


def load_parquet(
    path: Union[str, Path],
    target_column: Optional[str] = None,
    name: Optional[str] = None,
) -> RawDataset:
    """Load dataset from Parquet file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    
    df = pd.read_parquet(path)
    
    target = None
    target_name = None
    if target_column and target_column in df.columns:
        target = df[target_column].values
        target_name = target_column
        df = df.drop(columns=[target_column])
    
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    
    return RawDataset(
        data=numeric_df.values,
        feature_names=tuple(numeric_df.columns),
        target=target,
        target_name=target_name,
        source=f"parquet:{path}",
        name=name or path.stem,
    )


def load_sklearn_dataset(name: str) -> RawDataset:
    """Load a scikit-learn built-in dataset.
    
    Supported datasets:
    - "breast_cancer": Wisconsin breast cancer (569 samples, 30 features)
    - "diabetes": Diabetes regression (442 samples, 10 features)
    - "wine": Wine classification (178 samples, 13 features)
    - "iris": Iris classification (150 samples, 4 features)
    
    Args:
        name: Dataset name.
    
    Returns:
        RawDataset with complete data.
    """
    from sklearn import datasets
    
    loaders = {
        "breast_cancer": datasets.load_breast_cancer,
        "diabetes": datasets.load_diabetes,
        "wine": datasets.load_wine,
        "iris": datasets.load_iris,
    }
    
    if name not in loaders:
        available = ", ".join(loaders.keys())
        raise ValueError(f"Unknown sklearn dataset: {name}. Available: {available}")
    
    bunch = loaders[name]()
    
    return RawDataset(
        data=bunch.data,
        feature_names=tuple(bunch.feature_names),
        target=bunch.target,
        target_name="target",
        source=f"sklearn:{name}",
        name=name,
    )


def load_from_url(
    url: str,
    name: str,
    target_column: Optional[str] = None,
    sep: str = ",",
    header: Union[int, str] = "infer",
) -> RawDataset:
    """Load dataset from URL (CSV format).
    
    Args:
        url: URL to CSV file.
        name: Dataset name.
        target_column: Column to use as target.
        sep: Column separator.
        header: Header row specification.
    
    Returns:
        RawDataset.
    """
    df = pd.read_csv(url, sep=sep, header=header)
    
    target = None
    target_name = None
    if target_column and target_column in df.columns:
        target = df[target_column].values
        target_name = target_column
        df = df.drop(columns=[target_column])
    
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    
    if numeric_df.shape[1] == 0:
        raise ValueError(f"No numeric columns found from {url}")
    
    return RawDataset(
        data=numeric_df.values,
        feature_names=tuple(numeric_df.columns),
        target=target,
        target_name=target_name,
        source=f"url:{url}",
        name=name,
    )
