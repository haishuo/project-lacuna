"""
lacuna.data

Data processing pipeline for Project Lacuna.

Pipeline:
1. Create/load ObservedDataset
2. Tokenize to row-level representation
3. Batch with padding
4. Feed to model

The key insight: tokenization preserves ROW-LEVEL structure.
Each cell becomes (value, is_observed), enabling the transformer
to learn which missingness patterns are predictable from context.
"""

from .observed import (
    create_observed_dataset,
    from_numpy,
    split_dataset,
)

from .tokenization import (
    tokenize_dataset,
    tokenize_row,
    get_token_dim,
    TOKEN_DIM,
)

from .batching import (
    tokenize_and_batch,
    collate_fn,
    SyntheticDataLoader,
)

__all__ = [
    # Observed
    "create_observed_dataset",
    "from_numpy",
    "split_dataset",
    # Tokenization
    "tokenize_dataset",
    "tokenize_row",
    "get_token_dim",
    "TOKEN_DIM",
    # Batching
    "tokenize_and_batch",
    "collate_fn",
    "SyntheticDataLoader",
]

# Catalog imports (if not already present)
try:
    from .catalog import (
        DatasetInfo,
        DatasetCatalog,
        create_default_catalog,
        DEFAULT_RAW_DIR,
        DEFAULT_PROCESSED_DIR,
    )
except ImportError:
    pass
