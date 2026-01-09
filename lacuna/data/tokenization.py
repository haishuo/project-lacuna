"""
lacuna.data.tokenization

Row-level tokenization: ObservedDataset -> token sequences.

Architecture:
- Each row is a sequence of d tokens
- Each token represents one cell: (value, missingness_indicator)
- Token dim = 2: [normalized_value, is_observed]
- Missing values get value=0, is_observed=0
- Observed values get normalized_value, is_observed=1

The transformer sees the actual data structure, enabling it to learn:
- MAR: missingness predictable from other observed columns
- MNAR: missingness depends on the (unobserved) value itself
- MCAR: missingness is random, no pattern

This is fundamentally different from column-level summary statistics,
which lose the row-level structure that distinguishes mechanisms.
"""

import torch
from typing import Optional, Tuple

from lacuna.core.types import ObservedDataset


# Token dimension: [value, is_observed]
TOKEN_DIM = 2


def tokenize_row(
    row_x: torch.Tensor,  # [d] values
    row_r: torch.Tensor,  # [d] bool, True=observed
) -> torch.Tensor:
    """Tokenize a single row.
    
    Args:
        row_x: [d] feature values (may contain garbage where missing).
        row_r: [d] observation mask (True = observed).
    
    Returns:
        [d, TOKEN_DIM] token tensor.
    """
    d = row_x.shape[0]
    tokens = torch.zeros(d, TOKEN_DIM)
    
    # Channel 0: normalized value (0 if missing)
    tokens[:, 0] = row_x * row_r.float()
    
    # Channel 1: observation indicator
    tokens[:, 1] = row_r.float()
    
    return tokens


def tokenize_dataset(
    dataset: ObservedDataset,
    normalize: bool = True,
) -> torch.Tensor:
    """Tokenize entire dataset row-by-row.
    
    Args:
        dataset: Input ObservedDataset.
        normalize: Whether to normalize values (recommended).
    
    Returns:
        [n, d, TOKEN_DIM] tensor of row token sequences.
    """
    x = dataset.x.clone()
    r = dataset.r
    
    if normalize:
        # Normalize each column using observed values only
        for j in range(dataset.d):
            observed_mask = r[:, j]
            if observed_mask.sum() > 0:
                observed_vals = x[observed_mask, j]
                mean = observed_vals.mean()
                std = observed_vals.std()
                if std > 1e-8:
                    x[:, j] = (x[:, j] - mean) / std
                else:
                    x[:, j] = x[:, j] - mean
    
    # Tokenize each row
    n, d = dataset.n, dataset.d
    tokens = torch.zeros(n, d, TOKEN_DIM)
    
    for i in range(n):
        tokens[i] = tokenize_row(x[i], r[i])
    
    return tokens


def get_token_dim() -> int:
    """Return token feature dimension."""
    return TOKEN_DIM


def aggregate_row_tokens(
    row_tokens: torch.Tensor,  # [n, d, TOKEN_DIM]
    method: str = "mean",
) -> torch.Tensor:
    """Aggregate row tokens to dataset-level representation.
    
    This is used AFTER the transformer encodes each row.
    
    Args:
        row_tokens: [n, d, h] encoded row representations.
        method: Aggregation method ("mean", "max", "cls").
    
    Returns:
        [h] or [n, h] depending on method.
    """
    if method == "mean":
        # Mean over rows
        return row_tokens.mean(dim=0).mean(dim=0)
    elif method == "max":
        return row_tokens.max(dim=0)[0].max(dim=0)[0]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
