"""
lacuna.data.features

Column-level feature extraction.

Design: Each column becomes a fixed-size feature vector capturing
value statistics and missingness patterns.
"""

import torch
from typing import List

from lacuna.core.types import ObservedDataset


# Feature dimension (number of features per column)
FEATURE_DIM = 12


def extract_column_features(dataset: ObservedDataset) -> torch.Tensor:
    """Extract feature vector for each column.
    
    Features capture:
    - Missingness statistics
    - Value distribution (observed values)
    - Cross-column missingness patterns
    
    Args:
        dataset: Input dataset.
    
    Returns:
        [d, FEATURE_DIM] tensor of column features.
    """
    d = dataset.d
    features = []
    
    # Precompute dataset-level stats
    overall_miss_rate = dataset.missing_rate
    row_miss_counts = (~dataset.r).float().sum(dim=1)
    row_miss_mean = row_miss_counts.mean() / d
    row_miss_std = row_miss_counts.std() / d if dataset.n > 1 else torch.tensor(0.0)
    
    for j in range(d):
        f_j = _extract_single_column_features(
            col_x=dataset.x[:, j],
            col_r=dataset.r[:, j],
            dataset=dataset,
            overall_miss_rate=overall_miss_rate,
            row_miss_mean=row_miss_mean,
            row_miss_std=row_miss_std,
        )
        features.append(f_j)
    
    return torch.stack(features, dim=0)  # [d, FEATURE_DIM]


def _extract_single_column_features(
    col_x: torch.Tensor,
    col_r: torch.Tensor,
    dataset: ObservedDataset,
    overall_miss_rate: float,
    row_miss_mean: torch.Tensor,
    row_miss_std: torch.Tensor,
) -> torch.Tensor:
    """Extract features for one column.
    
    Returns [FEATURE_DIM] tensor.
    """
    n = dataset.n
    n_obs = col_r.sum().float()
    
    features = []
    
    # === Missingness features (3) ===
    miss_rate = 1.0 - n_obs / n
    features.append(miss_rate)
    features.append(torch.tensor(overall_miss_rate))
    features.append(miss_rate - overall_miss_rate)  # Relative miss rate
    
    # === Value features on observed (5) ===
    if n_obs > 0:
        observed_vals = col_x[col_r]
        mean_obs = observed_vals.mean()
        std_obs = observed_vals.std() if n_obs > 1 else torch.tensor(0.0)
        
        if n_obs >= 4:
            q25 = torch.quantile(observed_vals, 0.25)
            q50 = torch.quantile(observed_vals, 0.50)
            q75 = torch.quantile(observed_vals, 0.75)
        else:
            q25 = q50 = q75 = mean_obs
    else:
        mean_obs = torch.tensor(0.0)
        std_obs = torch.tensor(0.0)
        q25 = q50 = q75 = torch.tensor(0.0)
    
    features.extend([mean_obs, std_obs, q25, q50, q75])
    
    # === Row-level missingness features (2) ===
    features.append(row_miss_mean)
    features.append(row_miss_std)
    
    # === Cross-column missingness correlation (2) ===
    # Correlation between this column's missingness and other columns
    col_miss = (~col_r).float()
    other_miss = (~dataset.r).float()
    
    # Mean correlation with other columns
    correlations = []
    for k in range(dataset.d):
        if k != _get_col_index(col_r, dataset):
            other_col_miss = other_miss[:, k]
            if col_miss.std() > 0 and other_col_miss.std() > 0:
                corr = _pearson_corr(col_miss, other_col_miss)
                correlations.append(corr)
    
    if correlations:
        mean_miss_corr = torch.stack(correlations).mean()
        max_miss_corr = torch.stack(correlations).abs().max()
    else:
        mean_miss_corr = torch.tensor(0.0)
        max_miss_corr = torch.tensor(0.0)
    
    features.append(mean_miss_corr)
    features.append(max_miss_corr)
    
    # Stack and ensure float32
    result = torch.stack([f if isinstance(f, torch.Tensor) else torch.tensor(f) 
                          for f in features])
    return result.float()


def _get_col_index(col_r: torch.Tensor, dataset: ObservedDataset) -> int:
    """Find which column index this is (helper for avoiding self-correlation)."""
    for j in range(dataset.d):
        if torch.equal(col_r, dataset.r[:, j]):
            return j
    return -1


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Pearson correlation coefficient."""
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    
    num = (x_centered * y_centered).sum()
    denom = (x_centered.pow(2).sum().sqrt() * y_centered.pow(2).sum().sqrt())
    
    if denom < 1e-8:
        return torch.tensor(0.0)
    
    return num / denom
