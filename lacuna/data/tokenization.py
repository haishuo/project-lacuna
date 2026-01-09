"""
lacuna.data.tokenization

Full tokenization pipeline: ObservedDataset -> token tensor.

Design: Deterministic, reproducible tokenization.
"""

import torch
from typing import Optional

from lacuna.core.types import ObservedDataset
from .features import extract_column_features, FEATURE_DIM
from .normalization import NormalizationStats, compute_normalization_stats, normalize_dataset


def tokenize_dataset(
    dataset: ObservedDataset,
    normalize: bool = True,
    stats: Optional[NormalizationStats] = None,
) -> torch.Tensor:
    """Convert ObservedDataset to column tokens.
    
    Args:
        dataset: Input dataset.
        normalize: Whether to normalize before feature extraction.
        stats: Precomputed normalization stats. If None and normalize=True,
               computed from this dataset.
    
    Returns:
        [d, FEATURE_DIM] tensor of column tokens.
    """
    if normalize:
        if stats is None:
            stats = compute_normalization_stats(dataset, method="robust")
        dataset = normalize_dataset(dataset, stats)
    
    return extract_column_features(dataset)


def get_token_dim() -> int:
    """Return the token feature dimension."""
    return FEATURE_DIM
