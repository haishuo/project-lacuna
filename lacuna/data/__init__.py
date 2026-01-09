"""
lacuna.data

Data processing pipeline for Project Lacuna.

Pipeline:
1. Create/load ObservedDataset
2. Normalize values
3. Extract column features
4. Tokenize to fixed-size representation
5. Batch for model input

Semi-synthetic:
1. Load real dataset via ingestion
2. Apply synthetic missingness
3. Train on real data structure with known mechanisms
"""

from .observed import (
    create_observed_dataset,
    from_numpy,
    split_dataset,
)

from .normalization import (
    NormalizationStats,
    compute_normalization_stats,
    normalize_dataset,
    denormalize_values,
)

from .features import (
    extract_column_features,
    FEATURE_DIM,
)

from .tokenization import (
    tokenize_dataset,
    get_token_dim,
)

from .batching import (
    tokenize_and_batch,
    collate_fn,
    SyntheticDataLoader,
)

from .ingestion import (
    RawDataset,
    load_csv,
    load_parquet,
    load_sklearn_dataset,
    load_from_url,
)

from .semisynthetic import (
    SemiSyntheticDataset,
    apply_missingness,
    generate_semisynthetic_batch,
    SemiSyntheticDataLoader,
)

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
    # Normalization
    "NormalizationStats",
    "compute_normalization_stats",
    "normalize_dataset",
    "denormalize_values",
    # Features
    "extract_column_features",
    "FEATURE_DIM",
    # Tokenization
    "tokenize_dataset",
    "get_token_dim",
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
    "generate_semisynthetic_batch",
    "SemiSyntheticDataLoader",
    # Catalog
    "DatasetInfo",
    "DatasetCatalog",
    "create_default_catalog",
    "DEFAULT_RAW_DIR",
    "DEFAULT_PROCESSED_DIR",
]
