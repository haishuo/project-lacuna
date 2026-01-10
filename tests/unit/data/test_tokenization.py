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

from lacuna.core.types import ObservedDataset, TokenBatch
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
from lacuna.core.exceptions import ValidationError


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
        values = np.array([1.5, 2.0, -0.5, 0.0], dtype=np.float32)
        mask = np.array([True, True, True, True])
        max_cols = 6
        
        tokens, col_mask = tokenize_row(values, mask, max_cols)
        
        # Check shapes
        assert tokens.shape == (max_cols, TOKEN_DIM)
        assert col_mask.shape == (max_cols,)
        
        # Check col_mask: first 4 are real, last 2 are padding
        assert col_mask[:4].all()
        assert not col_mask[4:].any()
    
    def test_values_are_preserved(self):
        """Test that observed values are preserved in tokens."""
        values = np.array([1.5, -2.0, 0.3], dtype=np.float32)
        mask = np.array([True, True, True])
        max_cols = 4
        
        tokens, _ = tokenize_row(values, mask, max_cols)
        
        # Check values
        assert np.isclose(tokens[0, IDX_VALUE], 1.5)
        assert np.isclose(tokens[1, IDX_VALUE], -2.0)
        assert np.isclose(tokens[2, IDX_VALUE], 0.3)
    
    def test_missing_values_are_zero(self):
        """Test that missing values are set to zero."""
        values = np.array([1.5, np.nan, 0.3, np.nan], dtype=np.float32)
        mask = np.array([True, False, True, False])
        max_cols = 4
        
        tokens, _ = tokenize_row(values, mask, max_cols)
        
        # Observed values preserved
        assert np.isclose(tokens[0, IDX_VALUE], 1.5)
        assert np.isclose(tokens[2, IDX_VALUE], 0.3)
        
        # Missing values are zero
        assert tokens[1, IDX_VALUE] == 0.0
        assert tokens[3, IDX_VALUE] == 0.0
    
    def test_observed_indicator(self):
        """Test that observed indicator is set correctly."""
        values = np.array([1.0, np.nan, 2.0, np.nan], dtype=np.float32)
        mask = np.array([True, False, True, False])
        max_cols = 4
        
        tokens, _ = tokenize_row(values, mask, max_cols)
        
        # Check observed indicator
        assert tokens[0, IDX_OBSERVED] == 1.0
        assert tokens[1, IDX_OBSERVED] == 0.0
        assert tokens[2, IDX_OBSERVED] == 1.0
        assert tokens[3, IDX_OBSERVED] == 0.0
    
    def test_mask_type_natural(self):
        """Test that mask type is NATURAL by default."""
        values = np.array([1.0, np.nan, 2.0], dtype=np.float32)
        mask = np.array([True, False, True])
        max_cols = 4
        
        tokens, _ = tokenize_row(values, mask, max_cols)
        
        # All should be natural (no artificial masking)
        for j in range(3):
            assert tokens[j, IDX_MASK_TYPE] == MASK_TYPE_NATURAL
    
    def test_mask_type_artificial(self):
        """Test that artificial mask type is set correctly."""
        values = np.array([1.0, np.nan, 2.0], dtype=np.float32)
        mask = np.array([True, False, True])
        artificial_mask = np.array([True, False, False])  # First cell artificially masked
        max_cols = 4
        
        tokens, _ = tokenize_row(values, mask, max_cols, artificial_mask)
        
        # First cell should be marked as artificial
        assert tokens[0, IDX_MASK_TYPE] == MASK_TYPE_ARTIFICIAL
        # Others should be natural
        assert tokens[1, IDX_MASK_TYPE] == MASK_TYPE_NATURAL
        assert tokens[2, IDX_MASK_TYPE] == MASK_TYPE_NATURAL
    
    def test_feature_id_normalized(self):
        """Test that feature IDs are normalized to [0, 1]."""
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        mask = np.ones(4, dtype=bool)
        max_cols = 4
        
        tokens, _ = tokenize_row(values, mask, max_cols)
        
        # Feature IDs should be j / (max_cols - 1)
        assert np.isclose(tokens[0, IDX_FEATURE_ID], 0 / 3)
        assert np.isclose(tokens[1, IDX_FEATURE_ID], 1 / 3)
        assert np.isclose(tokens[2, IDX_FEATURE_ID], 2 / 3)
        assert np.isclose(tokens[3, IDX_FEATURE_ID], 3 / 3)
    
    def test_padding_is_zero(self):
        """Test that padding tokens are all zeros."""
        values = np.array([1.0, 2.0], dtype=np.float32)
        mask = np.ones(2, dtype=bool)
        max_cols = 5
        
        tokens, col_mask = tokenize_row(values, mask, max_cols)
        
        # Padding positions (indices 2, 3, 4) should be zero
        assert (tokens[2:, :] == 0).all()
        assert not col_mask[2:].any()
    
    def test_exceeding_max_cols_raises(self):
        """Test that exceeding max_cols raises ValidationError."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        mask = np.ones(5, dtype=bool)
        max_cols = 3  # Fewer than actual columns
        
        with pytest.raises(ValidationError):
            tokenize_row(values, mask, max_cols)


# =============================================================================
# Test tokenize_dataset
# =============================================================================

class TestTokenizeDataset:
    """Tests for tokenize_dataset function."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample ObservedDataset."""
        n, d = 10, 5
        X = np.random.randn(n, d).astype(np.float32)
        R = np.random.rand(n, d) > 0.2
        X[~R] = np.nan
        
        return ObservedDataset(
            X_obs=X,
            R=R,
            dataset_id="test",
            n_original=n,
            d_original=d,
        )
    
    def test_output_shapes(self, sample_dataset):
        """Test output tensor shapes."""
        max_rows, max_cols = 20, 10
        
        tokens, row_mask, col_mask, orig_values = tokenize_dataset(
            sample_dataset, max_rows, max_cols
        )
        
        assert tokens.shape == (max_rows, max_cols, TOKEN_DIM)
        assert row_mask.shape == (max_rows,)
        assert col_mask.shape == (max_cols,)
        assert orig_values.shape == (max_rows, max_cols)
    
    def test_row_mask_correct(self, sample_dataset):
        """Test that row_mask correctly indicates valid rows."""
        max_rows, max_cols = 20, 10
        n = sample_dataset.n  # 10 rows
        
        tokens, row_mask, col_mask, orig_values = tokenize_dataset(
            sample_dataset, max_rows, max_cols
        )
        
        # First n rows should be valid
        assert row_mask[:n].all()
        # Remaining rows should be invalid (padding)
        assert not row_mask[n:].any()
    
    def test_col_mask_correct(self, sample_dataset):
        """Test that col_mask correctly indicates valid columns."""
        max_rows, max_cols = 20, 10
        d = sample_dataset.d  # 5 columns
        
        tokens, row_mask, col_mask, orig_values = tokenize_dataset(
            sample_dataset, max_rows, max_cols
        )
        
        # First d columns should be valid
        assert col_mask[:d].all()
        # Remaining columns should be invalid (padding)
        assert not col_mask[d:].any()
    
    def test_subsampling_when_exceeding_max_rows(self):
        """Test that rows are subsampled when n > max_rows."""
        n, d = 100, 5
        X = np.random.randn(n, d).astype(np.float32)
        R = np.ones((n, d), dtype=bool)
        
        dataset = ObservedDataset(X_obs=X, R=R)
        
        max_rows, max_cols = 20, 10
        tokens, row_mask, col_mask, orig_values = tokenize_dataset(
            dataset, max_rows, max_cols
        )
        
        # Should have exactly max_rows
        assert tokens.shape[0] == max_rows
        # All rows should be valid (after subsampling)
        assert row_mask.all()
    
    def test_original_values_preserved(self, sample_dataset):
        """Test that original values are preserved for reconstruction."""
        max_rows, max_cols = 20, 10
        
        tokens, row_mask, col_mask, orig_values = tokenize_dataset(
            sample_dataset, max_rows, max_cols
        )
        
        # Original values should match the input (for non-NaN entries)
        n, d = sample_dataset.n, sample_dataset.d
        for i in range(n):
            for j in range(d):
                if sample_dataset.R[i, j]:
                    assert np.isclose(
                        orig_values[i, j],
                        sample_dataset.X_obs[i, j],
                        rtol=1e-5
                    )
    
    def test_with_artificial_mask(self, sample_dataset):
        """Test tokenization with artificial masking."""
        n, d = sample_dataset.n, sample_dataset.d
        max_rows, max_cols = 20, 10
        
        # Create artificial mask (mask some observed values)
        artificial_mask = np.zeros((n, d), dtype=bool)
        artificial_mask[0, 0] = True  # Artificially mask first cell
        artificial_mask[1, 1] = True  # And second cell in second row
        
        tokens, row_mask, col_mask, orig_values = tokenize_dataset(
            sample_dataset, max_rows, max_cols, artificial_mask
        )
        
        # Check that artificial mask type is set
        assert tokens[0, 0, IDX_MASK_TYPE] == MASK_TYPE_ARTIFICIAL
        assert tokens[1, 1, IDX_MASK_TYPE] == MASK_TYPE_ARTIFICIAL


# =============================================================================
# Test apply_artificial_masking
# =============================================================================

class TestApplyArtificialMasking:
    """Tests for apply_artificial_masking function."""
    
    def test_basic_masking(self):
        """Test basic artificial masking."""
        n, d = 20, 5
        X = np.random.randn(n, d).astype(np.float32)
        R = np.ones((n, d), dtype=bool)  # All observed
        
        config = MaskingConfig(mask_ratio=0.2, min_masked=1)
        
        X_masked, R_masked, artificial_mask = apply_artificial_masking(X, R, config)
        
        # Some values should now be masked
        assert artificial_mask.any()
        # R_masked should have fewer True values
        assert R_masked.sum() < R.sum()
        # X_masked should have NaN where artificially masked
        assert np.isnan(X_masked[artificial_mask]).all()
    
    def test_mask_ratio_approximate(self):
        """Test that mask ratio is approximately respected."""
        n, d = 100, 10
        X = np.random.randn(n, d).astype(np.float32)
        R = np.ones((n, d), dtype=bool)
        
        config = MaskingConfig(mask_ratio=0.15, min_masked=1)
        rng = np.random.default_rng(42)
        
        X_masked, R_masked, artificial_mask = apply_artificial_masking(X, R, config, rng)
        
        # Count fraction masked
        actual_ratio = artificial_mask.sum() / (n * d)
        # Should be approximately 0.15 (allow some variance)
        assert 0.10 <= actual_ratio <= 0.25
    
    def test_only_masks_observed(self):
        """Test that only observed values are masked."""
        n, d = 20, 5
        X = np.random.randn(n, d).astype(np.float32)
        R = np.random.rand(n, d) > 0.3  # Some already missing
        X[~R] = np.nan
        
        config = MaskingConfig(mask_ratio=0.3, mask_observed_only=True, min_masked=1)
        
        X_masked, R_masked, artificial_mask = apply_artificial_masking(X, R, config)
        
        # Artificial mask should only be True where R was True
        assert not (artificial_mask & ~R).any()
    
    def test_min_masked_respected(self):
        """Test that min_masked is respected."""
        n, d = 10, 5
        X = np.random.randn(n, d).astype(np.float32)
        R = np.ones((n, d), dtype=bool)
        
        config = MaskingConfig(mask_ratio=0.01, min_masked=2)  # Very low ratio but min=2
        
        X_masked, R_masked, artificial_mask = apply_artificial_masking(X, R, config)
        
        # Each row should have at least min_masked cells masked
        for i in range(n):
            assert artificial_mask[i].sum() >= 2
    
    def test_max_masked_respected(self):
        """Test that max_masked is respected."""
        n, d = 10, 10
        X = np.random.randn(n, d).astype(np.float32)
        R = np.ones((n, d), dtype=bool)
        
        config = MaskingConfig(mask_ratio=0.9, min_masked=1, max_masked=3)
        
        X_masked, R_masked, artificial_mask = apply_artificial_masking(X, R, config)
        
        # Each row should have at most max_masked cells masked
        for i in range(n):
            assert artificial_mask[i].sum() <= 3
    
    def test_reproducibility_with_rng(self):
        """Test that results are reproducible with same RNG."""
        n, d = 20, 5
        X = np.random.randn(n, d).astype(np.float32)
        R = np.ones((n, d), dtype=bool)
        
        config = MaskingConfig(mask_ratio=0.2)
        
        rng1 = np.random.default_rng(42)
        _, _, mask1 = apply_artificial_masking(X, R, config, rng1)
        
        rng2 = np.random.default_rng(42)
        _, _, mask2 = apply_artificial_masking(X, R, config, rng2)
        
        assert (mask1 == mask2).all()


# =============================================================================
# Test MaskingConfig
# =============================================================================

class TestMaskingConfig:
    """Tests for MaskingConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MaskingConfig()
        
        assert config.mask_ratio == 0.15
        assert config.mask_observed_only == True
        assert config.min_masked == 1
        assert config.max_masked is None
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = MaskingConfig(
            mask_ratio=0.3,
            mask_observed_only=False,
            min_masked=2,
            max_masked=5,
        )
        
        assert config.mask_ratio == 0.3
        assert config.mask_observed_only == False
        assert config.min_masked == 2
        assert config.max_masked == 5


# =============================================================================
# Test tokenize_and_batch
# =============================================================================

class TestTokenizeAndBatch:
    """Tests for tokenize_and_batch function."""
    
    @pytest.fixture
    def sample_datasets(self):
        """Create sample datasets."""
        datasets = []
        for i in range(4):
            n = np.random.randint(10, 50)
            d = np.random.randint(3, 10)
            X = np.random.randn(n, d).astype(np.float32)
            R = np.random.rand(n, d) > 0.2
            X[~R] = np.nan
            
            datasets.append(ObservedDataset(
                X_obs=X,
                R=R,
                dataset_id=f"test_{i}",
            ))
        
        return datasets
    
    def test_output_is_token_batch(self, sample_datasets):
        """Test that output is a TokenBatch."""
        batch = tokenize_and_batch(
            sample_datasets,
            max_rows=64,
            max_cols=16,
        )
        
        assert isinstance(batch, TokenBatch)
    
    def test_batch_size_matches_input(self, sample_datasets):
        """Test that batch size matches number of input datasets."""
        batch = tokenize_and_batch(
            sample_datasets,
            max_rows=64,
            max_cols=16,
        )
        
        assert batch.batch_size == len(sample_datasets)
    
    def test_token_shape(self, sample_datasets):
        """Test token tensor shape."""
        max_rows, max_cols = 64, 16
        
        batch = tokenize_and_batch(
            sample_datasets,
            max_rows=max_rows,
            max_cols=max_cols,
        )
        
        assert batch.tokens.shape == (len(sample_datasets), max_rows, max_cols, TOKEN_DIM)
    
    def test_with_generator_ids(self, sample_datasets):
        """Test batching with generator IDs."""
        generator_ids = [0, 1, 2, 3]
        
        batch = tokenize_and_batch(
            sample_datasets,
            max_rows=64,
            max_cols=16,
            generator_ids=generator_ids,
        )
        
        assert batch.generator_ids is not None
        assert batch.generator_ids.tolist() == generator_ids
    
    def test_with_class_mapping(self, sample_datasets):
        """Test batching with class mapping."""
        generator_ids = [0, 1, 2, 3]
        class_mapping = {0: 0, 1: 0, 2: 1, 3: 2}  # gen 0,1 -> MCAR, gen 2 -> MAR, gen 3 -> MNAR
        
        batch = tokenize_and_batch(
            sample_datasets,
            max_rows=64,
            max_cols=16,
            generator_ids=generator_ids,
            class_mapping=class_mapping,
        )
        
        assert batch.class_ids is not None
        assert batch.class_ids.tolist() == [0, 0, 1, 2]
    
    def test_with_variant_ids(self, sample_datasets):
        """Test batching with MNAR variant IDs."""
        variant_ids = [-1, -1, -1, 0]  # Only last is MNAR variant 0
        
        batch = tokenize_and_batch(
            sample_datasets,
            max_rows=64,
            max_cols=16,
            variant_ids=variant_ids,
        )
        
        assert batch.variant_ids is not None
        assert batch.variant_ids.tolist() == variant_ids
    
    def test_with_artificial_masks(self, sample_datasets):
        """Test batching with artificial masking."""
        artificial_masks = []
        for ds in sample_datasets:
            mask = np.random.rand(ds.n, ds.d) < 0.1  # 10% artificially masked
            artificial_masks.append(mask)
        
        batch = tokenize_and_batch(
            sample_datasets,
            max_rows=64,
            max_cols=16,
            artificial_masks=artificial_masks,
        )
        
        assert batch.original_values is not None
        assert batch.reconstruction_mask is not None
    
    def test_no_nan_in_tokens(self, sample_datasets):
        """Test that there are no NaN values in tokens."""
        batch = tokenize_and_batch(
            sample_datasets,
            max_rows=64,
            max_cols=16,
        )
        
        assert not torch.isnan(batch.tokens).any()
    
    def test_no_inf_in_tokens(self, sample_datasets):
        """Test that there are no Inf values in tokens."""
        batch = tokenize_and_batch(
            sample_datasets,
            max_rows=64,
            max_cols=16,
        )
        
        assert not torch.isinf(batch.tokens).any()


# =============================================================================
# Test Extraction Utilities
# =============================================================================

class TestExtractionUtilities:
    """Tests for token component extraction functions."""
    
    @pytest.fixture
    def sample_tokens(self):
        """Create sample token tensor."""
        B, max_rows, max_cols = 2, 10, 5
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        
        # Set specific values for testing
        tokens[..., IDX_VALUE] = torch.randn(B, max_rows, max_cols)
        tokens[..., IDX_OBSERVED] = (torch.rand(B, max_rows, max_cols) > 0.2).float()
        tokens[..., IDX_MASK_TYPE] = (torch.rand(B, max_rows, max_cols) > 0.9).float()
        tokens[..., IDX_FEATURE_ID] = torch.linspace(0, 1, max_cols).unsqueeze(0).unsqueeze(0).expand(B, max_rows, -1)
        
        return tokens
    
    def test_extract_values(self, sample_tokens):
        """Test extract_values function."""
        values = extract_values(sample_tokens)
        
        assert values.shape == sample_tokens.shape[:-1]
        assert torch.allclose(values, sample_tokens[..., IDX_VALUE])
    
    def test_extract_observed_mask(self, sample_tokens):
        """Test extract_observed_mask function."""
        observed = extract_observed_mask(sample_tokens)
        
        assert observed.shape == sample_tokens.shape[:-1]
        assert torch.allclose(observed, sample_tokens[..., IDX_OBSERVED])
    
    def test_extract_mask_type(self, sample_tokens):
        """Test extract_mask_type function."""
        mask_type = extract_mask_type(sample_tokens)
        
        assert mask_type.shape == sample_tokens.shape[:-1]
        assert torch.allclose(mask_type, sample_tokens[..., IDX_MASK_TYPE])
    
    def test_extract_feature_ids(self, sample_tokens):
        """Test extract_feature_ids function."""
        feature_ids = extract_feature_ids(sample_tokens)
        
        assert feature_ids.shape == sample_tokens.shape[:-1]
        assert torch.allclose(feature_ids, sample_tokens[..., IDX_FEATURE_ID])


# =============================================================================
# Test count_observed and compute_missing_rate
# =============================================================================

class TestObservedCounting:
    """Tests for count_observed and compute_missing_rate."""
    
    def test_count_observed_all_observed(self):
        """Test counting when all values are observed."""
        B, max_rows, max_cols = 2, 10, 5
        tokens = torch.zeros(B, max_rows, max_cols, TOKEN_DIM)
        tokens[..., IDX_OBSERVED] = 1.0  # All observed
        
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        counts = count_observed(tokens, row_mask, col_mask)
        
        assert counts.shape == (B,)
        assert (counts == max_rows * max_cols).all()
    
    def test_count_observed_with_missing(self):
        """Test counting with some missing values."""
        B, max_rows, max_cols = 2, 10, 5
        tokens = torch.zeros(B, max_rows, max_cols, TOKEN_DIM)
        tokens[..., IDX_OBSERVED] = 1.0
        
        # Make some values missing
        tokens[0, 0, 0, IDX_OBSERVED] = 0.0
        tokens[0, 1, 1, IDX_OBSERVED] = 0.0
        tokens[1, 2, 2, IDX_OBSERVED] = 0.0
        
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        counts = count_observed(tokens, row_mask, col_mask)
        
        assert counts[0] == max_rows * max_cols - 2
        assert counts[1] == max_rows * max_cols - 1
    
    def test_count_observed_respects_masks(self):
        """Test that counting respects row and column masks."""
        B, max_rows, max_cols = 2, 10, 5
        tokens = torch.zeros(B, max_rows, max_cols, TOKEN_DIM)
        tokens[..., IDX_OBSERVED] = 1.0
        
        # Only first 5 rows and 3 cols are valid
        row_mask = torch.zeros(B, max_rows, dtype=torch.bool)
        row_mask[:, :5] = True
        col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
        col_mask[:, :3] = True
        
        counts = count_observed(tokens, row_mask, col_mask)
        
        assert (counts == 5 * 3).all()
    
    def test_compute_missing_rate_all_observed(self):
        """Test missing rate when all values are observed."""
        B, max_rows, max_cols = 2, 10, 5
        tokens = torch.zeros(B, max_rows, max_cols, TOKEN_DIM)
        tokens[..., IDX_OBSERVED] = 1.0
        
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        rates = compute_missing_rate(tokens, row_mask, col_mask)
        
        assert rates.shape == (B,)
        assert torch.allclose(rates, torch.zeros(B))
    
    def test_compute_missing_rate_half_missing(self):
        """Test missing rate when half values are missing."""
        B, max_rows, max_cols = 1, 10, 10
        tokens = torch.zeros(B, max_rows, max_cols, TOKEN_DIM)
        tokens[..., IDX_OBSERVED] = 1.0
        
        # Make half missing
        tokens[0, :5, :, IDX_OBSERVED] = 0.0
        
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        rates = compute_missing_rate(tokens, row_mask, col_mask)
        
        assert torch.isclose(rates[0], torch.tensor(0.5))


# =============================================================================
# Test collate_token_batches
# =============================================================================

class TestCollateTokenBatches:
    """Tests for collate_token_batches function."""
    
    def test_collate_single_batch(self):
        """Test collating a single batch."""
        batch = TokenBatch(
            tokens=torch.randn(2, 10, 5, TOKEN_DIM),
            row_mask=torch.ones(2, 10, dtype=torch.bool),
            col_mask=torch.ones(2, 5, dtype=torch.bool),
        )
        
        collated = collate_token_batches([batch])
        
        assert collated.batch_size == 2
        assert torch.equal(collated.tokens, batch.tokens)
    
    def test_collate_multiple_batches(self):
        """Test collating multiple batches."""
        batches = [
            TokenBatch(
                tokens=torch.randn(2, 10, 5, TOKEN_DIM),
                row_mask=torch.ones(2, 10, dtype=torch.bool),
                col_mask=torch.ones(2, 5, dtype=torch.bool),
            )
            for _ in range(3)
        ]
        
        collated = collate_token_batches(batches)
        
        assert collated.batch_size == 6  # 2 * 3
    
    def test_collate_with_labels(self):
        """Test collating batches with labels."""
        batches = [
            TokenBatch(
                tokens=torch.randn(2, 10, 5, TOKEN_DIM),
                row_mask=torch.ones(2, 10, dtype=torch.bool),
                col_mask=torch.ones(2, 5, dtype=torch.bool),
                generator_ids=torch.tensor([0, 1]),
                class_ids=torch.tensor([0, 1]),
            )
            for _ in range(3)
        ]
        
        collated = collate_token_batches(batches)
        
        assert collated.generator_ids is not None
        assert collated.generator_ids.shape == (6,)
        assert collated.class_ids is not None
        assert collated.class_ids.shape == (6,)
    
    def test_collate_preserves_none_fields(self):
        """Test that None fields remain None after collation."""
        batches = [
            TokenBatch(
                tokens=torch.randn(2, 10, 5, TOKEN_DIM),
                row_mask=torch.ones(2, 10, dtype=torch.bool),
                col_mask=torch.ones(2, 5, dtype=torch.bool),
                # No optional fields
            )
            for _ in range(2)
        ]
        
        collated = collate_token_batches(batches)
        
        assert collated.generator_ids is None
        assert collated.class_ids is None
        assert collated.variant_ids is None
        assert collated.original_values is None
        assert collated.reconstruction_mask is None