"""
lacuna.data

Data processing pipeline for missing data mechanism classification.

Pipeline Overview:
    1. Create/load ObservedDataset (values + missingness mask)
    2. Apply artificial masking for self-supervised pretraining (optional)
    3. Tokenize: convert to [value, is_observed, mask_type, feature_id] tokens
    4. Batch: combine multiple datasets into TokenBatch for model input

Token Structure:
    Each cell in a dataset becomes a 4-dimensional token:
    - normalized_value: the observed value (0 if missing)
    - is_observed: 1.0 if observed, 0.0 if missing
    - mask_type: 0.0 = natural missing, 1.0 = artificially masked
    - feature_id_normalized: column position / max_cols

Quick Start:
    >>> from lacuna.data import tokenize_and_batch, apply_artificial_masking
    >>> 
    >>> # Apply artificial masking for pretraining
    >>> X_masked, R_masked, art_mask = apply_artificial_masking(X, R, config)
    >>> 
    >>> # Tokenize and batch
    >>> batch = tokenize_and_batch(datasets, max_rows=100, max_cols=32)
    >>> print(batch.tokens.shape)  # [B, max_rows, max_cols, 4]

Semi-Synthetic Data:
    For training on real data with synthetic missingness:
    >>> from lacuna.data import SemiSyntheticDataLoader
    >>> loader = SemiSyntheticDataLoader(catalog, generators, ...)
"""

# === Core Types ===
from lacuna.core.types import ObservedDataset, TokenBatch

# === Tokenization ===
from lacuna.data.tokenization import (
    # Constants
    TOKEN_DIM,
    IDX_VALUE,
    IDX_OBSERVED,
    IDX_MASK_TYPE,
    IDX_FEATURE_ID,
    MASK_TYPE_NATURAL,
    MASK_TYPE_ARTIFICIAL,
    # Core functions
    tokenize_row,
    tokenize_dataset,
    tokenize_and_batch,
    get_token_dim,
    # Token utilities
    extract_values,
    extract_observed_mask,
    extract_mask_type,
    extract_feature_ids,
    count_observed,
    compute_missing_rate,
    # Artificial masking
    MaskingConfig,
    apply_artificial_masking,
    # Collation
    collate_token_batches,
)

# === Batching ===
from lacuna.data.batching import (
    SyntheticDataLoader,
    collate_fn,
)

# === Ingestion (for real datasets) ===
from lacuna.data.ingestion import (
    RawDataset,
    load_csv,
    load_parquet,
    load_sklearn_dataset,
    load_from_url,
)

# === Semi-Synthetic ===
from lacuna.data.semisynthetic import (
    SemiSyntheticDataset,
    apply_missingness,
    generate_semisynthetic_batch,
    SemiSyntheticDataLoader,
)

# === Catalog ===
from lacuna.data.catalog import (
    DatasetInfo,
    DatasetCatalog,
    create_default_catalog,
    DEFAULT_RAW_DIR,
    DEFAULT_PROCESSED_DIR,
)

from lacuna.data.missingness_features import (
       extract_missingness_features,
       MissingnessFeatureExtractor,
       MissingnessFeatureConfig,
       get_feature_names,
   )

__all__ = [
    # === Core Types ===
    "ObservedDataset",
    "TokenBatch",
    
    # === Tokenization Constants ===
    "TOKEN_DIM",
    "IDX_VALUE",
    "IDX_OBSERVED",
    "IDX_MASK_TYPE",
    "IDX_FEATURE_ID",
    "MASK_TYPE_NATURAL",
    "MASK_TYPE_ARTIFICIAL",
    
    # === Tokenization Functions ===
    "tokenize_row",
    "tokenize_dataset",
    "tokenize_and_batch",
    "get_token_dim",
    
    # === Token Utilities ===
    "extract_values",
    "extract_observed_mask",
    "extract_mask_type",
    "extract_feature_ids",
    "count_observed",
    "compute_missing_rate",
    
    # === Artificial Masking ===
    "MaskingConfig",
    "apply_artificial_masking",
    
    # === Collation ===
    "collate_token_batches",
    
    # === Batching ===
    "SyntheticDataLoader",
    "collate_fn",
    
    # === Ingestion ===
    "RawDataset",
    "load_csv",
    "load_parquet",
    "load_sklearn_dataset",
    "load_from_url",
    
    # === Semi-Synthetic ===
    "SemiSyntheticDataset",
    "apply_missingness",
    "generate_semisynthetic_batch",
    "SemiSyntheticDataLoader",
    
    # === Catalog ===
    "DatasetInfo",
    "DatasetCatalog",
    "create_default_catalog",
    "DEFAULT_RAW_DIR",
    "DEFAULT_PROCESSED_DIR",
]