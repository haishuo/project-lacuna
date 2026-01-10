"""
lacuna.data.tokenization

BERT-inspired tokenization for missing data mechanism classification.

Architecture:
    Each row of a dataset becomes a sequence of "feature tokens".
    Each feature token contains:
        - value_embedding: projection of the observed value (or zero if missing)
        - observed_indicator: binary flag (1.0 if observed, 0.0 if missing)
        - mask_type_indicator: distinguishes natural missingness from artificial masking
        - feature_id: integer identifying which column/feature this token represents
    
    The transformer then attends over these feature tokens within each row,
    learning cross-column dependencies that distinguish MAR from MNAR.

Token Structure (per cell):
    [normalized_value, is_observed, mask_type, feature_id_normalized]
    
    Where:
        - normalized_value: float in roughly [-3, 3] after robust normalization
        - is_observed: 1.0 if value is observed, 0.0 if missing
        - mask_type: 0.0 = naturally missing, 1.0 = artificially masked (for pretraining)
        - feature_id_normalized: feature index / max_features (positional info)

Design Decisions:
    1. Row-level tokenization preserves cross-column structure needed for MAR detection
    2. Feature ID embedding allows the model to learn feature-specific patterns
    3. Mask type separation enables self-supervised pretraining with artificial masks
    4. Fixed token dimension (4) keeps memory manageable for large datasets
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from lacuna.core.types import ObservedDataset, TokenBatch
from lacuna.core.validation import validate_no_nan_inf
from lacuna.core.exceptions import ValidationError


# =============================================================================
# Constants
# =============================================================================

# Token dimension: [value, is_observed, mask_type, feature_id_normalized]
TOKEN_DIM = 4

# Default limits for padding
DEFAULT_MAX_ROWS = 128
DEFAULT_MAX_COLS = 32

# Indices into the token vector
IDX_VALUE = 0
IDX_OBSERVED = 1
IDX_MASK_TYPE = 2
IDX_FEATURE_ID = 3

# Mask type constants
MASK_TYPE_NATURAL = 0.0      # Naturally missing in the data
MASK_TYPE_ARTIFICIAL = 1.0   # Artificially masked for self-supervised learning


# =============================================================================
# Core Tokenization
# =============================================================================

def tokenize_row(
    values: np.ndarray,           # [d] observed values (NaN where missing)
    mask: np.ndarray,             # [d] boolean mask (True = observed)
    max_cols: int,                # Maximum number of columns (for padding)
    artificial_mask: Optional[np.ndarray] = None,  # [d] cells we artificially masked
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tokenize a single row into feature tokens.
    
    Args:
        values: Array of observed values with NaN for missing entries.
        mask: Boolean array where True indicates the value is observed.
        max_cols: Maximum number of columns for padding.
        artificial_mask: Optional boolean array indicating which observed values
                        were artificially masked for self-supervised training.
    
    Returns:
        tokens: [max_cols, TOKEN_DIM] token array
        col_mask: [max_cols] boolean mask (True for real columns, False for padding)
    
    Example:
        >>> values = np.array([1.5, np.nan, 0.3, np.nan])
        >>> mask = np.array([True, False, True, False])
        >>> tokens, col_mask = tokenize_row(values, mask, max_cols=6)
        >>> tokens.shape
        (6, 4)
        >>> col_mask
        array([ True,  True,  True,  True, False, False])
    """
    d = len(values)
    
    if d > max_cols:
        raise ValidationError(
            f"Number of columns ({d}) exceeds max_cols ({max_cols}). "
            "Either increase max_cols or subsample columns."
        )
    
    # Initialize output arrays
    tokens = np.zeros((max_cols, TOKEN_DIM), dtype=np.float32)
    col_mask = np.zeros(max_cols, dtype=bool)
    
    # Mark real columns
    col_mask[:d] = True
    
    # Fill in tokens for each feature
    for j in range(d):
        # Value: use observed value if present, else 0
        if mask[j]:
            tokens[j, IDX_VALUE] = values[j]
        else:
            tokens[j, IDX_VALUE] = 0.0
        
        # Observed indicator
        tokens[j, IDX_OBSERVED] = 1.0 if mask[j] else 0.0
        
        # Mask type: natural vs artificial
        if artificial_mask is not None and artificial_mask[j]:
            # This cell was observed but we artificially masked it
            tokens[j, IDX_MASK_TYPE] = MASK_TYPE_ARTIFICIAL
        else:
            # Either naturally missing or naturally observed
            tokens[j, IDX_MASK_TYPE] = MASK_TYPE_NATURAL
        
        # Feature ID (normalized to [0, 1] range)
        tokens[j, IDX_FEATURE_ID] = j / max(max_cols - 1, 1)
    
    return tokens, col_mask


def tokenize_dataset(
    dataset: ObservedDataset,
    max_rows: int,
    max_cols: int,
    artificial_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tokenize an entire dataset into a 3D token tensor.
    
    Args:
        dataset: ObservedDataset containing x (values) and r (mask).
        max_rows: Maximum number of rows (for padding/subsampling).
        max_cols: Maximum number of columns (for padding).
        artificial_mask: Optional [n, d] boolean array of artificially masked cells.
    
    Returns:
        tokens: [max_rows, max_cols, TOKEN_DIM] token tensor
        row_mask: [max_rows] boolean mask (True for real rows)
        col_mask: [max_cols] boolean mask (True for real columns)
        original_values: [max_rows, max_cols] original values before masking (for reconstruction)
    
    Notes:
        - If dataset has more rows than max_rows, rows are randomly subsampled
        - If dataset has fewer rows than max_rows, output is zero-padded
        - Row and column masks indicate which positions contain real data
    """
    # Extract numpy arrays from dataset (current API uses .x and .r)
    # Handle both tensor and numpy array inputs
    if hasattr(dataset.x, 'numpy'):
        X = dataset.x.numpy()  # Convert torch tensor to numpy
    else:
        X = dataset.x
    
    if hasattr(dataset.r, 'numpy'):
        R = dataset.r.numpy()  # Convert torch tensor to numpy
    else:
        R = dataset.r
    
    n, d = dataset.n, dataset.d
    
    # Handle row subsampling/padding
    if n > max_rows:
        # Random subsample (use dataset_id for reproducibility if needed)
        indices = np.random.choice(n, max_rows, replace=False)
        X = X[indices]
        R = R[indices]
        if artificial_mask is not None:
            artificial_mask = artificial_mask[indices]
        n = max_rows
    
    # Initialize outputs
    tokens = np.zeros((max_rows, max_cols, TOKEN_DIM), dtype=np.float32)
    row_mask = np.zeros(max_rows, dtype=bool)
    col_mask = np.zeros(max_cols, dtype=bool)
    original_values = np.zeros((max_rows, max_cols), dtype=np.float32)
    
    # Mark real rows and columns
    row_mask[:n] = True
    col_mask[:d] = True
    
    # Tokenize each row
    for i in range(n):
        row_artificial_mask = artificial_mask[i] if artificial_mask is not None else None
        
        # For the current API, dataset.x has missing values zeroed out (not NaN)
        # We need to reconstruct which values to pass to tokenize_row
        row_values = X[i].copy().astype(np.float32)
        row_mask_vals = R[i].astype(bool)
        
        # Set missing values to NaN for tokenize_row (it expects NaN for missing)
        row_values[~row_mask_vals] = np.nan
        
        row_tokens, _ = tokenize_row(row_values, row_mask_vals, max_cols, row_artificial_mask)
        tokens[i] = row_tokens
        
        # Store original values for reconstruction targets
        # Use the values directly (already 0 where missing in current API)
        original_values[i, :d] = X[i]
    
    return tokens, row_mask, col_mask, original_values


# =============================================================================
# Artificial Masking for Self-Supervised Learning
# =============================================================================

@dataclass
class MaskingConfig:
    """Configuration for artificial masking during self-supervised pretraining."""
    mask_ratio: float = 0.15        # Fraction of observed values to mask
    mask_observed_only: bool = True # Only mask cells that are observed
    min_masked: int = 1             # Minimum cells to mask per row
    max_masked: Optional[int] = None # Maximum cells to mask per row (None = no limit)


def apply_artificial_masking(
    X: np.ndarray,                  # [n, d] values (NaN for missing)
    R: np.ndarray,                  # [n, d] observation mask
    config: MaskingConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply artificial masking for self-supervised pretraining.
    
    This is the BERT-style "corruption" step: we take some observed values
    and hide them, then train the model to reconstruct them from context.
    
    Args:
        X: Values array with NaN for missing entries.
        R: Boolean observation mask (True = observed).
        config: Masking configuration.
        rng: Random number generator for reproducibility.
    
    Returns:
        X_masked: Values with additional artificial masks (new NaNs)
        R_masked: Updated observation mask
        artificial_mask: Boolean array indicating which cells were artificially masked
    
    Example:
        >>> X = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]])
        >>> R = np.array([[True, True, True], [True, False, True]])
        >>> config = MaskingConfig(mask_ratio=0.5)
        >>> X_masked, R_masked, art_mask = apply_artificial_masking(X, R, config)
        >>> # Some observed values are now masked
        >>> art_mask.sum() > 0
        True
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n, d = X.shape
    
    # Initialize outputs as copies
    X_masked = X.copy()
    R_masked = R.copy()
    artificial_mask = np.zeros((n, d), dtype=bool)
    
    for i in range(n):
        if config.mask_observed_only:
            # Only consider observed cells for masking
            candidates = np.where(R[i])[0]
        else:
            # Consider all cells (though masking already-missing cells is weird)
            candidates = np.arange(d)
        
        if len(candidates) == 0:
            continue
        
        # Determine how many cells to mask
        n_to_mask = int(len(candidates) * config.mask_ratio)
        n_to_mask = max(n_to_mask, config.min_masked)
        
        if config.max_masked is not None:
            n_to_mask = min(n_to_mask, config.max_masked)
        
        n_to_mask = min(n_to_mask, len(candidates))
        
        if n_to_mask == 0:
            continue
        
        # Randomly select cells to mask
        mask_indices = rng.choice(candidates, n_to_mask, replace=False)
        
        # Apply masking
        for j in mask_indices:
            artificial_mask[i, j] = True
            X_masked[i, j] = np.nan
            R_masked[i, j] = False
    
    return X_masked, R_masked, artificial_mask


# =============================================================================
# Batching
# =============================================================================

def tokenize_and_batch(
    datasets: List[ObservedDataset],
    max_rows: int,
    max_cols: int,
    generator_ids: Optional[List[int]] = None,
    class_mapping: Optional[dict] = None,
    variant_ids: Optional[List[int]] = None,
    artificial_masks: Optional[List[np.ndarray]] = None,
) -> TokenBatch:
    """
    Tokenize multiple datasets and combine into a batch.
    
    Args:
        datasets: List of ObservedDataset objects to batch.
        max_rows: Maximum rows per dataset (for padding/subsampling).
        max_cols: Maximum columns per dataset (for padding).
        generator_ids: Optional list of generator IDs for each dataset.
        class_mapping: Optional dict mapping generator_id -> class_id.
        variant_ids: Optional list of MNAR variant IDs for each dataset.
        artificial_masks: Optional list of artificial mask arrays (one per dataset).
    
    Returns:
        TokenBatch ready for model input.
    
    Example:
        >>> datasets = [dataset1, dataset2, dataset3]
        >>> batch = tokenize_and_batch(datasets, max_rows=100, max_cols=32)
        >>> batch.tokens.shape
        torch.Size([3, 100, 32, 4])
    """
    B = len(datasets)
    
    # Initialize batch arrays
    all_tokens = np.zeros((B, max_rows, max_cols, TOKEN_DIM), dtype=np.float32)
    all_row_masks = np.zeros((B, max_rows), dtype=bool)
    all_col_masks = np.zeros((B, max_cols), dtype=bool)
    all_original_values = np.zeros((B, max_rows, max_cols), dtype=np.float32)
    all_reconstruction_masks = np.zeros((B, max_rows, max_cols), dtype=bool)
    
    for i, dataset in enumerate(datasets):
        art_mask = artificial_masks[i] if artificial_masks is not None else None
        
        tokens, row_mask, col_mask, orig_vals = tokenize_dataset(
            dataset, max_rows, max_cols, art_mask
        )
        
        all_tokens[i] = tokens
        all_row_masks[i] = row_mask
        all_col_masks[i] = col_mask
        all_original_values[i] = orig_vals
        
        # Reconstruction mask: which cells were artificially masked
        if art_mask is not None:
            # Pad artificial mask to max dimensions
            n, d = art_mask.shape
            all_reconstruction_masks[i, :n, :d] = art_mask
    
    # Convert to tensors
    tokens_tensor = torch.from_numpy(all_tokens)
    row_mask_tensor = torch.from_numpy(all_row_masks)
    col_mask_tensor = torch.from_numpy(all_col_masks)
    original_values_tensor = torch.from_numpy(all_original_values)
    reconstruction_mask_tensor = torch.from_numpy(all_reconstruction_masks)
    
    # Handle labels
    gen_ids_tensor = None
    class_ids_tensor = None
    variant_ids_tensor = None
    
    if generator_ids is not None:
        gen_ids_tensor = torch.tensor(generator_ids, dtype=torch.long)
        
        if class_mapping is not None:
            class_ids = [class_mapping[gid] for gid in generator_ids]
            class_ids_tensor = torch.tensor(class_ids, dtype=torch.long)
    
    if variant_ids is not None:
        variant_ids_tensor = torch.tensor(variant_ids, dtype=torch.long)
    
    return TokenBatch(
        tokens=tokens_tensor,
        row_mask=row_mask_tensor,
        col_mask=col_mask_tensor,
        generator_ids=gen_ids_tensor,
        class_ids=class_ids_tensor,
        variant_ids=variant_ids_tensor,
        original_values=original_values_tensor,
        reconstruction_mask=reconstruction_mask_tensor,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def get_token_dim() -> int:
    """Return the token dimension."""
    return TOKEN_DIM


def extract_values(tokens: torch.Tensor) -> torch.Tensor:
    """
    Extract value component from tokens.
    
    Args:
        tokens: [..., TOKEN_DIM] token tensor
    
    Returns:
        values: [...] value tensor
    """
    return tokens[..., IDX_VALUE]


def extract_observed_mask(tokens: torch.Tensor) -> torch.Tensor:
    """
    Extract observed indicator from tokens.
    
    Args:
        tokens: [..., TOKEN_DIM] token tensor
    
    Returns:
        observed: [...] float tensor (1.0 = observed)
    """
    return tokens[..., IDX_OBSERVED]


def extract_mask_type(tokens: torch.Tensor) -> torch.Tensor:
    """
    Extract mask type from tokens.
    
    Args:
        tokens: [..., TOKEN_DIM] token tensor
    
    Returns:
        mask_type: [...] float tensor (0.0 = natural, 1.0 = artificial)
    """
    return tokens[..., IDX_MASK_TYPE]


def extract_feature_ids(tokens: torch.Tensor) -> torch.Tensor:
    """
    Extract feature ID component from tokens.
    
    Args:
        tokens: [..., TOKEN_DIM] token tensor
    
    Returns:
        feature_ids: [...] normalized feature ID tensor
    """
    return tokens[..., IDX_FEATURE_ID]


def count_observed(
    tokens: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Count observed values per sample.
    
    Args:
        tokens: [B, max_rows, max_cols, TOKEN_DIM] token tensor
        row_mask: [B, max_rows] row validity mask
        col_mask: [B, max_cols] column validity mask
    
    Returns:
        counts: [B] count per sample
    """
    observed = extract_observed_mask(tokens)
    
    # Create validity mask: [B, max_rows, max_cols]
    validity = row_mask.unsqueeze(-1) & col_mask.unsqueeze(-2)
    
    # Count observed values within valid cells
    counts = (observed * validity.float()).sum(dim=(1, 2))
    
    return counts


def compute_missing_rate(
    tokens: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute missing rate per sample.
    
    Args:
        tokens: [B, max_rows, max_cols, TOKEN_DIM] token tensor
        row_mask: [B, max_rows] row validity mask
        col_mask: [B, max_cols] column validity mask
    
    Returns:
        rates: [B] missing rate per sample (0 to 1)
    """
    observed = extract_observed_mask(tokens)
    validity = row_mask.unsqueeze(-1) & col_mask.unsqueeze(-2)
    
    total_cells = validity.float().sum(dim=(1, 2))
    observed_cells = (observed * validity.float()).sum(dim=(1, 2))
    
    # Avoid division by zero
    missing_rate = 1.0 - observed_cells / total_cells.clamp(min=1.0)
    
    return missing_rate


# =============================================================================
# Collate Function for DataLoader
# =============================================================================

def collate_token_batches(batches: List[TokenBatch]) -> TokenBatch:
    """
    Collate multiple TokenBatch objects into a single batch.
    
    This is useful when using PyTorch DataLoader with pre-tokenized data.
    
    Args:
        batches: List of TokenBatch objects (each typically with batch_size=1)
    
    Returns:
        Combined TokenBatch
    """
    # Stack all tensors along batch dimension
    tokens = torch.cat([b.tokens for b in batches], dim=0)
    row_mask = torch.cat([b.row_mask for b in batches], dim=0)
    col_mask = torch.cat([b.col_mask for b in batches], dim=0)
    
    # Handle optional tensors
    gen_ids = None
    if all(b.generator_ids is not None for b in batches):
        gen_ids = torch.cat([b.generator_ids for b in batches], dim=0)
    
    class_ids = None
    if all(b.class_ids is not None for b in batches):
        class_ids = torch.cat([b.class_ids for b in batches], dim=0)
    
    variant_ids = None
    if all(b.variant_ids is not None for b in batches):
        variant_ids = torch.cat([b.variant_ids for b in batches], dim=0)
    
    original_values = None
    if all(b.original_values is not None for b in batches):
        original_values = torch.cat([b.original_values for b in batches], dim=0)
    
    reconstruction_mask = None
    if all(b.reconstruction_mask is not None for b in batches):
        reconstruction_mask = torch.cat([b.reconstruction_mask for b in batches], dim=0)
    
    return TokenBatch(
        tokens=tokens,
        row_mask=row_mask,
        col_mask=col_mask,
        generator_ids=gen_ids,
        class_ids=class_ids,
        variant_ids=variant_ids,
        original_values=original_values,
        reconstruction_mask=reconstruction_mask,
    )