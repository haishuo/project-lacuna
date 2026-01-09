"""
lacuna.generators.families.base_data

Base data samplers (complete data without missingness).

Design: These generate X_full; missingness is applied separately.
"""

import torch
from typing import Optional

from lacuna.core.rng import RNGState


def sample_gaussian(
    rng: RNGState,
    n: int,
    d: int,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Sample from multivariate Gaussian (independent columns).
    
    Args:
        rng: RNG state.
        n: Number of rows.
        d: Number of columns.
        mean: Mean for all columns.
        std: Standard deviation for all columns.
    
    Returns:
        [n, d] tensor of Gaussian samples.
    """
    return rng.randn(n, d) * std + mean


def sample_gaussian_correlated(
    rng: RNGState,
    n: int,
    d: int,
    mean: float = 0.0,
    std: float = 1.0,
    rho: float = 0.5,
) -> torch.Tensor:
    """Sample from multivariate Gaussian with AR(1) correlation.
    
    Correlation structure: Corr(X_i, X_j) = rho^|i-j|
    
    Args:
        rng: RNG state.
        n: Number of rows.
        d: Number of columns.
        mean: Mean for all columns.
        std: Standard deviation for all columns.
        rho: AR(1) correlation parameter.
    
    Returns:
        [n, d] tensor of correlated Gaussian samples.
    """
    # Build AR(1) correlation matrix
    indices = torch.arange(d, dtype=torch.float32)
    corr = rho ** torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
    
    # Cholesky decomposition for sampling
    L = torch.linalg.cholesky(corr)
    
    # Sample independent normals and transform
    Z = rng.randn(n, d)
    X = Z @ L.T
    
    return X * std + mean


def sample_uniform(
    rng: RNGState,
    n: int,
    d: int,
    low: float = 0.0,
    high: float = 1.0,
) -> torch.Tensor:
    """Sample from uniform distribution.
    
    Args:
        rng: RNG state.
        n: Number of rows.
        d: Number of columns.
        low: Lower bound.
        high: Upper bound.
    
    Returns:
        [n, d] tensor of uniform samples.
    """
    return rng.rand(n, d) * (high - low) + low


def sample_mixed(
    rng: RNGState,
    n: int,
    d: int,
    gaussian_cols: int = None,
) -> torch.Tensor:
    """Sample mixed Gaussian and uniform columns.
    
    First `gaussian_cols` columns are Gaussian, rest are uniform.
    
    Args:
        rng: RNG state.
        n: Number of rows.
        d: Number of columns.
        gaussian_cols: Number of Gaussian columns (default: d // 2).
    
    Returns:
        [n, d] tensor of mixed samples.
    """
    if gaussian_cols is None:
        gaussian_cols = d // 2
    
    gaussian_cols = min(gaussian_cols, d)
    uniform_cols = d - gaussian_cols
    
    parts = []
    if gaussian_cols > 0:
        parts.append(sample_gaussian(rng.spawn(), n, gaussian_cols))
    if uniform_cols > 0:
        parts.append(sample_uniform(rng.spawn(), n, uniform_cols))
    
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
