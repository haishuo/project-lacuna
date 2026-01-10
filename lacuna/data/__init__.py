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

# Ingestion (loading external data)
from .ingestion import (
    RawDataset,
    load_csv,
    load_parquet,
    load_sklearn_dataset,
    load_from_url,
)

# Semi-synthetic data generation
from .semisynthetic import (
    SemiSyntheticDataset,
    apply_missingness,
    subsample_rows,
    generate_semisynthetic_batch,
    SemiSyntheticDataLoader,
    MixedDataLoader,
)

# Dataset catalog
from .catalog import (
    DatasetInfo,
    DatasetCatalog,
    create_default_catalog,
    DEFAULT_RAW_DIR,
    DEFAULT_PROCESSED_DIR,
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
    # Ingestion
    "RawDataset",
    "load_csv",
    "load_parquet",
    "load_sklearn_dataset",
    "load_from_url",
    # Semi-synthetic
    "SemiSyntheticDataset",
    "apply_missingness",
    "subsample_rows",
    "generate_semisynthetic_batch",
    "SemiSyntheticDataLoader",
    "MixedDataLoader",
    # Catalog
    "DatasetInfo",
    "DatasetCatalog",
    "create_default_catalog",
    "DEFAULT_RAW_DIR",
    "DEFAULT_PROCESSED_DIR",
]