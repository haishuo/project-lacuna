"""
lacuna.data.normalization

Value normalization strategies.

Design: Normalize based on observed values only.
"""

import torch
from typing import Tuple, Optional
from dataclasses import dataclass

from lacuna.core.types import ObservedDataset


@dataclass(frozen=True)
class NormalizationStats:
    """Statistics for normalization."""
    center: torch.Tensor  # [d] center values (mean or median)
    scale: torch.Tensor   # [d] scale values (std or IQR)
    method: str           # "robust" or "standard"
    
    def to(self, device: str) -> "NormalizationStats":
        return NormalizationStats(
            center=self.center.to(device),
            scale=self.scale.to(device),
            method=self.method,
        )


def compute_normalization_stats(
    dataset: ObservedDataset,
    method: str = "robust",
) -> NormalizationStats:
    """Compute normalization statistics from observed values.
    
    Args:
        dataset: Dataset to compute stats from.
        method: "robust" (median/IQR) or "standard" (mean/std).
    
    Returns:
        NormalizationStats object.
    """
    d = dataset.d
    center = torch.zeros(d)
    scale = torch.ones(d)
    
    for j in range(d):
        col_mask = dataset.r[:, j]
        if col_mask.sum() == 0:
            continue
        
        observed = dataset.x[col_mask, j]
        
        if method == "robust":
            center[j] = observed.median()
            q75 = torch.quantile(observed, 0.75)
            q25 = torch.quantile(observed, 0.25)
            iqr = q75 - q25
            # Use IQR/1.35 to approximate std for normal data
            scale[j] = iqr / 1.35 if iqr > 1e-8 else 1.0
        elif method == "standard":
            center[j] = observed.mean()
            scale[j] = observed.std() if len(observed) > 1 else 1.0
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Avoid division by zero
        if scale[j] < 1e-8:
            scale[j] = 1.0
    
    return NormalizationStats(center=center, scale=scale, method=method)


def normalize_dataset(
    dataset: ObservedDataset,
    stats: NormalizationStats,
) -> ObservedDataset:
    """Apply normalization to dataset.
    
    Args:
        dataset: Dataset to normalize.
        stats: Precomputed normalization statistics.
    
    Returns:
        Normalized dataset (new object, immutable).
    """
    # Normalize: (x - center) / scale
    x_norm = (dataset.x - stats.center.unsqueeze(0)) / stats.scale.unsqueeze(0)
    
    # Re-zero missing values
    x_norm = x_norm * dataset.r.float()
    
    return ObservedDataset(
        x=x_norm,
        r=dataset.r,
        n=dataset.n,
        d=dataset.d,
        feature_names=dataset.feature_names,
        dataset_id=dataset.dataset_id,
        meta=dataset.meta,
    )


def denormalize_values(
    x: torch.Tensor,
    stats: NormalizationStats,
) -> torch.Tensor:
    """Reverse normalization.
    
    Args:
        x: Normalized values.
        stats: Normalization statistics.
    
    Returns:
        Original-scale values.
    """
    return x * stats.scale + stats.center
