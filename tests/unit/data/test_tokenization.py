"""
Tests for lacuna.data.tokenization

Tests the BERT-inspired tokenization system:
    - Token structure: [value, is_observed, mask_type, feature_id]
    - Row and dataset tokenization
    - Artificial masking for self-supervised learning
    - Batching multiple datasets
    - Utility functions for extracting token components
"""

import pytest
import torch
import numpy as np
from typing import Tuple

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
    # Utility functions
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
from lacuna.core.types import ObservedDataset, TokenBatch, MCAR, MAR, MNAR


# =============================================================================
# Helper Functions
# =============================================================================

def make_observed_dataset(
    n: int = 100,
    d: int = 10,
    miss_rate: float = 0.2,
    seed: int = 42,
) -> ObservedDataset:
    """Create ObservedDataset for testing with current API."""
    np.random.seed(seed)
    
    # Generate complete data
    X = np.random.randn(n, d).astype(np.float32)
    
    # Generate missingness mask
    R = np.random.rand(n, d) > miss_rate
    
    # Ensure at least one observed value per column
    for j in range(d):
        if not R[:, j].any():
            R[np.random.randint(n), j] = True
    
    # Zero out missing values (current API expects this)
    X_obs = X.copy()
    X_obs[~R] = 0.0
    
    # Convert to tensors
    x_tensor = torch.from_numpy(X_obs)
    r_tensor = torch.from_numpy(R)
    
    return ObservedDataset(
        x=x_tensor,
        r=r_tensor,
        n=n,
        d=d,
        feature_names=tuple(f"col_{j}" for j in range(d)),
        dataset_id=f"test_dataset_{seed}",
        meta=None,
    )


def make_simple_row(
    d: int = 10,
    miss_rate: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a single row with missingness for testing tokenize_row."""
    np.random.seed(seed)
    
    values = np.random.randn(d).astype(np.float32)
    mask = np.random.rand(d) > miss_rate
    
    # Ensure at least one observed
    if not mask.any():
        mask[0] = True
    
    # Set missing values to NaN (tokenize_row expects NaN for missing)
    values[~mask] = np.nan
    
    return values, mask


# =============================================================================
# Test Constants
# =============================================================================

class TestConstants:
    """Tests for tokenization constants."""
    
    def test_token_dim_is_four(self):
        """Token dimension should be 4."""
        assert TOKEN_DIM == 4
        assert get_token_dim() == 4
    
    def test_index_constants_are_valid(self):
        """Index constants should be in valid range."""
        assert 0 <= IDX_VALUE < TOKEN_DIM
        assert 0 <= IDX_OBSERVED < TOKEN_DIM
        assert 0 <= IDX_MASK_TYPE < TOKEN_DIM
        assert 0 <= IDX_FEATURE_ID < TOKEN_DIM
    
    def test_index_constants_are_distinct(self):
        """Index constants should be distinct."""
        indices = [IDX_VALUE, IDX_OBSERVED, IDX_MASK_TYPE, IDX_FEATURE_ID]
        assert len(set(indices)) == len(indices)
    
    def test_mask_type_constants(self):
        """Mask type constants should be distinct."""
        assert MASK_TYPE_NATURAL != MASK_TYPE_ARTIFICIAL
        assert MASK_TYPE_NATURAL == 0.0
        assert MASK_TYPE_ARTIFICIAL == 1.0


# =============================================================================
# Test tokenize_row
# =============================================================================

class TestTokenizeRow:
    """Tests for tokenize_row function."""
    
    def test_basic_tokenization(self):
        """Test basic row tokenization."""
        values, mask = make_simple_row(d=5)
        max_cols = 10
        
        tokens, col_mask = tokenize_row(values, mask, max_cols)
        
        assert tokens.shape == (max_cols, TOKEN_DIM)
        assert col_mask.shape == (max_cols,)
    
    def test_col_mask_correct(self):
        """Test that col_mask correctly indicates valid columns."""
        values, mask = make_simple_row(d=5)
        max_cols = 10
        
        tokens, col_mask = tokenize_row(values, mask, max_cols)
        
        # First 5 columns should be valid
        assert col_mask[:5].all()
        # Remaining should be padding
        assert not col_mask[5:].any()
    
    def test_observed_indicator(self):
        """Test that observed indicator is set correctly."""
        # Create row with known missingness pattern
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
        mask = np.array([True, False, True, False, True])
        
        tokens, _ = tokenize_row(values, mask, max_cols=10)
        
        # Check is_observed indicator (index 1)
        assert tokens[0, IDX_OBSERVED] == 1.0  # Observed
        assert tokens[1, IDX_OBSERVED] == 0.0  # Missing
        assert tokens[2, IDX_OBSERVED] == 1.0  # Observed
        assert tokens[3, IDX_OBSERVED] == 0.0  # Missing
        assert tokens[4, IDX_OBSERVED] == 1.0  # Observed
    
    def test_mask_type_natural(self):
        """Test that natural missingness is marked correctly."""
        values, mask = make_simple_row(d=5, miss_rate=0.5)
        
        tokens, _ = tokenize_row(values, mask, max_cols=10)
        
        # All mask types should be NATURAL (no artificial masking)
        assert (tokens[:5, IDX_MASK_TYPE] == MASK_TYPE_NATURAL).all()
    
    def test_mask_type_artificial(self):
        """Test that artificial masking is marked correctly."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        mask = np.ones(5, dtype=bool)
        artificial_mask = np.array([False, True, False, True, False])  # Mask indices 1, 3
        
        tokens, _ = tokenize_row(values, mask, max_cols=10, artificial_mask=artificial_mask)
        
        # Check mask type at artificially masked positions
        assert tokens[0, IDX_MASK_TYPE] == MASK_TYPE_NATURAL
        assert tokens[1, IDX_MASK_TYPE] == MASK_TYPE_ARTIFICIAL
        assert tokens[2, IDX_MASK_TYPE] == MASK_TYPE_NATURAL
        assert tokens[3, IDX_MASK_TYPE] == MASK_TYPE_ARTIFICIAL
        assert tokens[4, IDX_MASK_TYPE] == MASK_TYPE_NATURAL
    
    def test_feature_id_normalized(self):
        """Test that feature IDs are normalized correctly."""
        values, mask = make_simple_row(d=5)
        max_cols = 10
        
        tokens, _ = tokenize_row(values, mask, max_cols)
        
        # Feature IDs should be normalized: j / (max_cols - 1)
        for j in range(5):
            expected = j / (max_cols - 1)
            assert abs(tokens[j, IDX_FEATURE_ID] - expected) < 1e-5
    
    def test_padding_values(self):
        """Test that padding tokens have correct values."""
        values, mask = make_simple_row(d=3)
        max_cols = 10
        
        tokens, col_mask = tokenize_row(values, mask, max_cols)
        
        # Padding positions (indices 3-9) should be zeros
        assert (tokens[3:, :] == 0.0).all()
    
    def test_no_nan_in_output(self):
        """Test that output contains no NaN values."""
        values, mask = make_simple_row(d=5, miss_rate=0.5)
        
        tokens, _ = tokenize_row(values, mask, max_cols=10)
        
        assert not np.isnan(tokens).any()
    
    def test_no_inf_in_output(self):
        """Test that output contains no infinite values."""
        values, mask = make_simple_row(d=5)
        
        tokens, _ = tokenize_row(values, mask, max_cols=10)
        
        assert not np.isinf(tokens).any()


# =============================================================================
# Test tokenize_dataset
# =============================================================================

class TestTokenizeDataset:
    """Tests for tokenize_dataset function."""
    
    def test_output_shapes(self):
        """Test that output tensors have correct shapes."""
        dataset = make_observed_dataset(n=100, d=10)
        max_rows, max_cols = 128, 16
        
        tokens, row_mask, col_mask, orig_vals = tokenize_dataset(
            dataset, max_rows, max_cols
        )
        
        assert tokens.shape == (max_rows, max_cols, TOKEN_DIM)
        assert row_mask.shape == (max_rows,)
        assert col_mask.shape == (max_cols,)
        assert orig_vals.shape == (max_rows, max_cols)
    
    def test_row_mask_correct(self):
        """Test that row_mask correctly indicates valid rows."""
        dataset = make_observed_dataset(n=50, d=5)
        max_rows, max_cols = 128, 16
        
        tokens, row_mask, col_mask, _ = tokenize_dataset(
            dataset, max_rows, max_cols
        )
        
        # First 50 rows should be valid, rest should be masked
        assert row_mask[:50].all()
        assert not row_mask[50:].any()
    
    def test_col_mask_correct(self):
        """Test that col_mask correctly indicates valid columns."""
        dataset = make_observed_dataset(n=50, d=5)
        max_rows, max_cols = 128, 16
        
        tokens, row_mask, col_mask, _ = tokenize_dataset(
            dataset, max_rows, max_cols
        )
        
        # First 5 columns should be valid, rest should be masked
        assert col_mask[:5].all()
        assert not col_mask[5:].any()
    
    def test_original_values_preserved(self):
        """Test that original values are captured correctly."""
        dataset = make_observed_dataset(n=50, d=5, miss_rate=0.0)  # No missing
        max_rows, max_cols = 128, 16
        
        tokens, row_mask, col_mask, orig_vals = tokenize_dataset(
            dataset, max_rows, max_cols
        )
        
        # Original values should match dataset values for valid cells
        for i in range(50):
            for j in range(5):
                assert abs(orig_vals[i, j] - dataset.x[i, j].item()) < 1e-5
    
    def test_with_artificial_mask(self):
        """Test that artificial missingness mask is applied correctly."""
        dataset = make_observed_dataset(n=100, d=10, miss_rate=0.0)
        
        # Create artificial mask with 50% missingness
        artificial_mask = np.random.rand(100, 10) > 0.5
        
        tokens, row_mask, col_mask, _ = tokenize_dataset(
            dataset, max_rows=128, max_cols=16,
            artificial_mask=artificial_mask,
        )
        
        # Should still produce valid output
        assert tokens.shape == (128, 16, TOKEN_DIM)
        assert not np.isnan(tokens).any()
    
    def test_subsampling_when_exceeding_max_rows(self):
        """Test that rows are subsampled when n > max_rows."""
        dataset = make_observed_dataset(n=200, d=5)
        max_rows = 100
        
        tokens, row_mask, col_mask, _ = tokenize_dataset(
            dataset, max_rows=max_rows, max_cols=16
        )
        
        # All max_rows should be used (subsampled from 200)
        assert row_mask.sum() == max_rows
    
    def test_no_nan_in_output(self):
        """Test that output contains no NaN values."""
        dataset = make_observed_dataset(n=100, d=10, miss_rate=0.3)
        
        tokens, _, _, _ = tokenize_dataset(dataset, max_rows=128, max_cols=16)
        
        assert not np.isnan(tokens).any()
    
    def test_no_inf_in_output(self):
        """Test that output contains no infinite values."""
        dataset = make_observed_dataset(n=100, d=10, miss_rate=0.3)
        
        tokens, _, _, _ = tokenize_dataset(dataset, max_rows=128, max_cols=16)
        
        assert not np.isinf(tokens).any()


# =============================================================================
# Test tokenize_and_batch
# =============================================================================

class TestTokenizeAndBatch:
    """Tests for tokenize_and_batch function."""
    
    def test_output_is_token_batch(self):
        """Test that output is TokenBatch."""
        datasets = [make_observed_dataset(n=50, d=5, seed=i) for i in range(4)]
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=128,
            max_cols=16,
        )
        
        assert isinstance(batch, TokenBatch)
    
    def test_batch_size_matches_input(self):
        """Test that batch size matches number of input datasets."""
        datasets = [make_observed_dataset(n=50, d=5, seed=i) for i in range(8)]
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=128,
            max_cols=16,
        )
        
        assert batch.tokens.shape[0] == 8
    
    def test_token_shape(self):
        """Test that tokens have correct shape."""
        datasets = [make_observed_dataset(n=50, d=5, seed=i) for i in range(4)]
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=128,
            max_cols=16,
        )
        
        # [batch_size, max_rows, max_cols, token_dim]
        assert batch.tokens.shape == (4, 128, 16, TOKEN_DIM)
    
    def test_with_generator_ids(self):
        """Test that generator_ids are properly included."""
        datasets = [make_observed_dataset(n=50, d=5, seed=i) for i in range(4)]
        generator_ids = [0, 1, 2, 3]
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=128,
            max_cols=16,
            generator_ids=generator_ids,
        )
        
        assert batch.generator_ids is not None
        assert batch.generator_ids.tolist() == generator_ids
    
    def test_with_class_mapping(self):
        """Test that class labels are properly mapped."""
        datasets = [make_observed_dataset(n=50, d=5, seed=i) for i in range(4)]
        generator_ids = [0, 1, 2, 3]
        # Map generators 0,1 -> class 0 (MCAR), 2 -> class 1 (MAR), 3 -> class 2 (MNAR)
        class_mapping = {0: MCAR, 1: MCAR, 2: MAR, 3: MNAR}
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=128,
            max_cols=16,
            generator_ids=generator_ids,
            class_mapping=class_mapping,
        )
        
        assert batch.class_ids is not None
        expected = torch.tensor([MCAR, MCAR, MAR, MNAR])
        assert torch.equal(batch.class_ids, expected)
    
    def test_with_variant_ids(self):
        """Test that variant_ids are properly included."""
        datasets = [make_observed_dataset(n=50, d=5, seed=i) for i in range(4)]
        variant_ids = [0, 0, 1, 2]
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=128,
            max_cols=16,
            variant_ids=variant_ids,
        )
        
        assert batch.variant_ids is not None
        assert batch.variant_ids.tolist() == variant_ids
    
    def test_with_artificial_masks(self):
        """Test that artificial masks are applied."""
        datasets = [make_observed_dataset(n=50, d=5, seed=i, miss_rate=0.0) for i in range(4)]
        
        # Create artificial masks for each dataset
        artificial_masks = [np.random.rand(50, 5) > 0.5 for _ in range(4)]
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=128,
            max_cols=16,
            artificial_masks=artificial_masks,
        )
        
        assert batch.tokens.shape == (4, 128, 16, TOKEN_DIM)
    
    def test_no_nan_in_tokens(self):
        """Test that tokens contain no NaN values."""
        datasets = [make_observed_dataset(n=50, d=5, seed=i, miss_rate=0.3) for i in range(4)]
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=128,
            max_cols=16,
        )
        
        assert not torch.isnan(batch.tokens).any()
    
    def test_no_inf_in_tokens(self):
        """Test that tokens contain no infinite values."""
        datasets = [make_observed_dataset(n=50, d=5, seed=i, miss_rate=0.3) for i in range(4)]
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=128,
            max_cols=16,
        )
        
        assert not torch.isinf(batch.tokens).any()


# =============================================================================
# Test Utility Functions
# =============================================================================

class TestExtractValues:
    """Tests for extract_values function."""
    
    def test_extracts_correct_index(self):
        """Test that values are extracted from correct token index."""
        # Create a simple token tensor
        tokens = torch.zeros(2, 3, 4, TOKEN_DIM)
        tokens[:, :, :, IDX_VALUE] = 42.0
        
        values = extract_values(tokens)
        
        assert (values == 42.0).all()
        assert values.shape == (2, 3, 4)


class TestExtractObservedMask:
    """Tests for extract_observed_mask function."""
    
    def test_extracts_correct_index(self):
        """Test that observed mask is extracted from correct token index."""
        tokens = torch.zeros(2, 3, 4, TOKEN_DIM)
        tokens[:, :, :, IDX_OBSERVED] = 1.0
        
        observed = extract_observed_mask(tokens)
        
        assert (observed == 1.0).all()
        assert observed.shape == (2, 3, 4)


class TestExtractMaskType:
    """Tests for extract_mask_type function."""
    
    def test_extracts_correct_index(self):
        """Test that mask type is extracted from correct token index."""
        tokens = torch.zeros(2, 3, 4, TOKEN_DIM)
        tokens[:, :, :, IDX_MASK_TYPE] = MASK_TYPE_ARTIFICIAL
        
        mask_type = extract_mask_type(tokens)
        
        assert (mask_type == MASK_TYPE_ARTIFICIAL).all()


class TestExtractFeatureIds:
    """Tests for extract_feature_ids function."""
    
    def test_extracts_correct_index(self):
        """Test that feature IDs are extracted from correct token index."""
        tokens = torch.zeros(2, 3, 4, TOKEN_DIM)
        tokens[:, :, :, IDX_FEATURE_ID] = 0.5
        
        feature_ids = extract_feature_ids(tokens)
        
        assert (feature_ids == 0.5).all()


class TestCountObserved:
    """Tests for count_observed function."""
    
    def test_counts_correctly(self):
        """Test that observed values are counted correctly."""
        tokens = torch.zeros(2, 3, 4, TOKEN_DIM)
        # Set some cells as observed
        tokens[0, 0, 0, IDX_OBSERVED] = 1.0
        tokens[0, 1, 1, IDX_OBSERVED] = 1.0
        tokens[1, 2, 3, IDX_OBSERVED] = 1.0
        
        row_mask = torch.ones(2, 3, dtype=torch.bool)
        col_mask = torch.ones(2, 4, dtype=torch.bool)
        
        counts = count_observed(tokens, row_mask, col_mask)
        
        assert counts[0] == 2  # First sample has 2 observed
        assert counts[1] == 1  # Second sample has 1 observed


class TestComputeMissingRate:
    """Tests for compute_missing_rate function."""
    
    def test_computes_correctly(self):
        """Test that missing rate is computed correctly."""
        tokens = torch.zeros(1, 4, 4, TOKEN_DIM)
        # Set 8 out of 16 cells as observed
        tokens[0, :2, :, IDX_OBSERVED] = 1.0
        
        row_mask = torch.ones(1, 4, dtype=torch.bool)
        col_mask = torch.ones(1, 4, dtype=torch.bool)
        
        rates = compute_missing_rate(tokens, row_mask, col_mask)
        
        # 8 observed out of 16 -> 50% missing
        assert abs(rates[0].item() - 0.5) < 1e-5


# =============================================================================
# Test MaskingConfig and apply_artificial_masking
# =============================================================================

class TestMaskingConfig:
    """Tests for MaskingConfig dataclass."""
    
    def test_default_values(self):
        """Test that MaskingConfig has sensible defaults."""
        config = MaskingConfig()
        
        assert 0.0 < config.mask_ratio < 1.0
        assert config.min_masked >= 1
    
    def test_custom_values(self):
        """Test that custom values are accepted."""
        config = MaskingConfig(mask_ratio=0.3, min_masked=5)
        
        assert config.mask_ratio == 0.3
        assert config.min_masked == 5


class TestApplyArtificialMasking:
    """Tests for apply_artificial_masking function."""
    
    def test_returns_tuple(self):
        """Test that function returns expected tuple."""
        X = np.random.randn(100, 10).astype(np.float32)
        R = np.ones((100, 10), dtype=bool)
        config = MaskingConfig(mask_ratio=0.2)
        
        X_masked, R_masked, art_mask = apply_artificial_masking(X, R, config)
        
        assert X_masked.shape == X.shape
        assert R_masked.shape == R.shape
        assert art_mask.shape == R.shape
    
    def test_masks_some_values(self):
        """Test that some values are artificially masked."""
        X = np.random.randn(100, 10).astype(np.float32)
        R = np.ones((100, 10), dtype=bool)
        config = MaskingConfig(mask_ratio=0.2)
        
        _, _, art_mask = apply_artificial_masking(X, R, config)
        
        # Should have approximately 20% artificially masked
        mask_ratio = art_mask.mean()
        assert 0.1 < mask_ratio < 0.3
    
    def test_respects_min_masked(self):
        """Test that min_masked constraint is respected."""
        X = np.random.randn(100, 10).astype(np.float32)
        R = np.ones((100, 10), dtype=bool)
        config = MaskingConfig(mask_ratio=0.9, min_masked=2, max_masked=8)
        
        _, R_masked, art_mask = apply_artificial_masking(X, R, config)
        
        # Each row should have at least min_masked cells artificially masked
        for i in range(100):
            assert art_mask[i].sum() >= config.min_masked


# =============================================================================
# Test collate_token_batches
# =============================================================================

class TestCollateTokenBatches:
    """Tests for collate_token_batches function."""
    
    def test_combines_batches(self):
        """Test that multiple batches are combined correctly."""
        datasets1 = [make_observed_dataset(n=50, d=5, seed=i) for i in range(2)]
        datasets2 = [make_observed_dataset(n=50, d=5, seed=i+10) for i in range(3)]
        
        batch1 = tokenize_and_batch(datasets1, max_rows=64, max_cols=8)
        batch2 = tokenize_and_batch(datasets2, max_rows=64, max_cols=8)
        
        combined = collate_token_batches([batch1, batch2])
        
        assert combined.tokens.shape[0] == 5  # 2 + 3
    
    def test_preserves_shape(self):
        """Test that combined batch has correct shape."""
        datasets = [make_observed_dataset(n=50, d=5, seed=i) for i in range(4)]
        batches = [
            tokenize_and_batch([ds], max_rows=64, max_cols=8)
            for ds in datasets
        ]
        
        combined = collate_token_batches(batches)
        
        assert combined.tokens.shape == (4, 64, 8, TOKEN_DIM)


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_dataset_batch(self):
        """Test batch with single dataset."""
        datasets = [make_observed_dataset(n=50, d=5)]
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=128,
            max_cols=16,
        )
        
        assert batch.tokens.shape[0] == 1
    
    def test_single_column_dataset(self):
        """Test dataset with single column."""
        n, d = 100, 1
        X = torch.randn(n, d)
        R = torch.ones(n, d, dtype=torch.bool)
        
        dataset = ObservedDataset(
            x=X,
            r=R,
            n=n,
            d=d,
            feature_names=("col_0",),
            dataset_id="single_col",
            meta=None,
        )
        
        tokens, row_mask, col_mask, _ = tokenize_dataset(
            dataset, max_rows=128, max_cols=16
        )
        
        assert col_mask.sum() == 1
    
    def test_single_row_dataset(self):
        """Test dataset with single row."""
        n, d = 1, 5
        X = torch.randn(n, d)
        R = torch.ones(n, d, dtype=torch.bool)
        
        dataset = ObservedDataset(
            x=X,
            r=R,
            n=n,
            d=d,
            feature_names=tuple(f"col_{j}" for j in range(d)),
            dataset_id="single_row",
            meta=None,
        )
        
        tokens, row_mask, col_mask, _ = tokenize_dataset(
            dataset, max_rows=128, max_cols=16
        )
        
        assert row_mask.sum() == 1
    
    def test_exact_max_rows(self):
        """Test dataset with exactly max_rows."""
        dataset = make_observed_dataset(n=128, d=5)
        
        tokens, row_mask, col_mask, _ = tokenize_dataset(
            dataset, max_rows=128, max_cols=16
        )
        
        assert row_mask.sum() == 128
    
    def test_exact_max_cols(self):
        """Test dataset with exactly max_cols."""
        dataset = make_observed_dataset(n=50, d=16)
        
        tokens, row_mask, col_mask, _ = tokenize_dataset(
            dataset, max_rows=128, max_cols=16
        )
        
        assert col_mask.sum() == 16
    
    def test_high_missingness_single_observed(self):
        """Test column with only single observed value."""
        n, d = 100, 5
        X = torch.randn(n, d)
        R = torch.zeros(n, d, dtype=torch.bool)
        R[0, :] = True  # Only first row observed
        X[~R] = 0.0
        
        dataset = ObservedDataset(
            x=X,
            r=R,
            n=n,
            d=d,
            feature_names=tuple(f"col_{j}" for j in range(d)),
            dataset_id="sparse",
            meta=None,
        )
        
        tokens, row_mask, col_mask, _ = tokenize_dataset(
            dataset, max_rows=128, max_cols=16
        )
        
        # Should still produce valid tokens
        assert not np.isnan(tokens).any()