"""
lacuna.data.observed

ObservedDataset construction utilities.
"""

import torch
from typing import Optional, Tuple, List
import numpy as np

from lacuna.core.types import ObservedDataset
from lacuna.core.validation import validate_no_nan_inf


def create_observed_dataset(
    x: torch.Tensor,
    r: Optional[torch.Tensor] = None,
    feature_names: Optional[Tuple[str, ...]] = None,
    dataset_id: str = "unnamed",
    meta: Optional[dict] = None,
) -> ObservedDataset:
    """Create ObservedDataset with validation.
    
    Args:
        x: [n, d] data tensor. Missing values should be marked by NaN
           if r is not provided, or can be any value if r is provided.
        r: [n, d] bool tensor. True = observed. If None, inferred from NaN in x.
        feature_names: Column names. If None, auto-generated as col_0, col_1, ...
        dataset_id: Unique identifier.
        meta: Optional metadata dict.
    
    Returns:
        Validated ObservedDataset.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D, got {x.dim()}D")
    
    n, d = x.shape
    
    # Infer missingness from NaN if r not provided
    if r is None:
        r = ~torch.isnan(x)
        # Replace NaN with 0 in x
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    else:
        # Validate no NaN/Inf in observed values
        observed_vals = x[r]
        if len(observed_vals) > 0:
            validate_no_nan_inf(observed_vals, "observed values")
    
    # Ensure correct dtype
    if x.dtype != torch.float32:
        x = x.float()
    if r.dtype != torch.bool:
        r = r.bool()
    
    # Generate feature names if not provided
    if feature_names is None:
        feature_names = tuple(f"col_{j}" for j in range(d))
    
    # Zero out missing values (defensive)
    x = x * r.float()
    
    return ObservedDataset(
        x=x,
        r=r,
        n=n,
        d=d,
        feature_names=feature_names,
        dataset_id=dataset_id,
        meta=meta,
    )


def from_numpy(
    x: np.ndarray,
    r: Optional[np.ndarray] = None,
    feature_names: Optional[Tuple[str, ...]] = None,
    dataset_id: str = "unnamed",
) -> ObservedDataset:
    """Create ObservedDataset from numpy arrays.
    
    Args:
        x: [n, d] numpy array.
        r: [n, d] bool numpy array. If None, inferred from NaN.
        feature_names: Column names.
        dataset_id: Unique identifier.
    
    Returns:
        ObservedDataset.
    """
    x_tensor = torch.from_numpy(x.astype(np.float32))
    r_tensor = torch.from_numpy(r.astype(bool)) if r is not None else None
    
    return create_observed_dataset(
        x=x_tensor,
        r=r_tensor,
        feature_names=feature_names,
        dataset_id=dataset_id,
    )


def split_dataset(
    dataset: ObservedDataset,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[ObservedDataset, ObservedDataset]:
    """Split dataset into train and validation sets.
    
    Args:
        dataset: Dataset to split.
        train_frac: Fraction for training.
        seed: Random seed for reproducibility.
    
    Returns:
        (train_dataset, val_dataset)
    """
    n = dataset.n
    n_train = int(n * train_frac)
    
    # Shuffle indices
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    
    train_ds = ObservedDataset(
        x=dataset.x[train_idx],
        r=dataset.r[train_idx],
        n=len(train_idx),
        d=dataset.d,
        feature_names=dataset.feature_names,
        dataset_id=f"{dataset.dataset_id}_train",
        meta=dataset.meta,
    )
    
    val_ds = ObservedDataset(
        x=dataset.x[val_idx],
        r=dataset.r[val_idx],
        n=len(val_idx),
        d=dataset.d,
        feature_names=dataset.feature_names,
        dataset_id=f"{dataset.dataset_id}_val",
        meta=dataset.meta,
    )
    
    return train_ds, val_ds
