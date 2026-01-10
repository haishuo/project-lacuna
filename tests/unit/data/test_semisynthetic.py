"""
Tests for lacuna.data.semisynthetic

Tests semi-synthetic data generation with real data + synthetic missingness.
"""

import pytest
import torch
import numpy as np

from lacuna.data.semisynthetic import (
    SemiSyntheticDataset,
    apply_missingness,
    subsample_rows,
    generate_semisynthetic_batch,
    SemiSyntheticDataLoader,
    MixedDataLoader,
)
from lacuna.data.ingestion import RawDataset, load_sklearn_dataset
from lacuna.generators import create_minimal_registry
from lacuna.generators.priors import GeneratorPrior
from lacuna.core.rng import RNGState


@pytest.fixture
def small_raw():
    """Small raw dataset for testing."""
    data = np.random.randn(50, 5).astype(np.float32)
    return RawDataset(data, ("a", "b", "c", "d", "e"), name="test_small")


@pytest.fixture
def large_raw():
    """Large raw dataset for row subsampling tests."""
    data = np.random.randn(500, 8).astype(np.float32)
    return RawDataset(data, tuple(f"col{i}" for i in range(8)), name="test_large")


@pytest.fixture
def breast_cancer_raw():
    """Real sklearn dataset."""
    return load_sklearn_dataset("breast_cancer")


@pytest.fixture
def minimal_registry():
    """Minimal generator registry."""
    return create_minimal_registry()


class TestApplyMissingness:
    """Tests for apply_missingness function."""
    
    def test_basic_application(self, small_raw, minimal_registry):
        """Should apply missingness to complete data."""
        generator = minimal_registry[0]  # MCAR
        rng = RNGState(seed=42)
        
        ss = apply_missingness(small_raw, generator, rng)
        
        assert isinstance(ss, SemiSyntheticDataset)
        assert ss.observed.n == small_raw.n
        assert ss.observed.d == small_raw.d
        assert ss.generator_id == generator.generator_id
        assert ss.class_id == generator.class_id
        assert ss.source_name == small_raw.name
    
    def test_preserves_observed_values(self, small_raw, minimal_registry):
        """Observed values should match original data."""
        generator = minimal_registry[0]
        rng = RNGState(seed=42)
        
        ss = apply_missingness(small_raw, generator, rng)
        
        # Where observed, values should match
        R = ss.observed.r
        X_orig = torch.from_numpy(small_raw.data.astype('float32'))
        X_obs = ss.observed.x
        
        assert torch.allclose(X_obs[R], X_orig[R])
    
    def test_missing_values_zeroed(self, small_raw, minimal_registry):
        """Missing values should be zeroed."""
        generator = minimal_registry[0]
        rng = RNGState(seed=42)
        
        ss = apply_missingness(small_raw, generator, rng)
        
        R = ss.observed.r
        X_obs = ss.observed.x
        
        # Missing positions should be zero
        assert (X_obs[~R] == 0).all()
    
    def test_reproducible(self, small_raw, minimal_registry):
        """Same seed should produce same result."""
        generator = minimal_registry[1]  # MAR
        
        ss1 = apply_missingness(small_raw, generator, RNGState(seed=123))
        ss2 = apply_missingness(small_raw, generator, RNGState(seed=123))
        
        assert torch.equal(ss1.observed.r, ss2.observed.r)
        assert torch.equal(ss1.observed.x, ss2.observed.x)
    
    def test_ensures_observed_per_column(self, small_raw, minimal_registry):
        """Each column should have at least one observed value."""
        generator = minimal_registry[0]
        rng = RNGState(seed=42)
        
        ss = apply_missingness(small_raw, generator, rng)
        
        # Each column should have at least one True
        for col in range(ss.observed.d):
            assert ss.observed.r[:, col].sum() > 0


class TestSubsampleRows:
    """Tests for subsample_rows function."""
    
    def test_no_subsample_when_small(self, small_raw, minimal_registry):
        """Should not subsample if dataset is smaller than max_rows."""
        generator = minimal_registry[0]
        ss = apply_missingness(small_raw, generator, RNGState(42))
        
        result = subsample_rows(ss.observed, max_rows=100, rng=RNGState(42))
        
        assert result.n == ss.observed.n  # Unchanged
        assert torch.equal(result.x, ss.observed.x)
    
    def test_subsample_when_large(self, large_raw, minimal_registry):
        """Should subsample if dataset exceeds max_rows."""
        generator = minimal_registry[0]
        ss = apply_missingness(large_raw, generator, RNGState(42))
        
        max_rows = 128
        result = subsample_rows(ss.observed, max_rows=max_rows, rng=RNGState(42))
        
        assert result.n == max_rows
        assert result.d == ss.observed.d
    
    def test_subsample_reproducible(self, large_raw, minimal_registry):
        """Same seed should give same subsample."""
        generator = minimal_registry[0]
        ss = apply_missingness(large_raw, generator, RNGState(42))
        
        r1 = subsample_rows(ss.observed, max_rows=100, rng=RNGState(999))
        r2 = subsample_rows(ss.observed, max_rows=100, rng=RNGState(999))
        
        assert torch.equal(r1.x, r2.x)
        assert torch.equal(r1.r, r2.r)


class TestGenerateSemisyntheticBatch:
    """Tests for generate_semisynthetic_batch function."""
    
    def test_generates_correct_count(self, small_raw, minimal_registry):
        """Should generate requested number of samples."""
        prior = GeneratorPrior.uniform(minimal_registry)
        rng = RNGState(seed=42)
        
        results = generate_semisynthetic_batch(
            raw_datasets=[small_raw],
            registry=minimal_registry,
            prior=prior,
            rng=rng,
            samples_per_dataset=5,
        )
        
        assert len(results) == 5
    
    def test_multiple_datasets(self, minimal_registry):
        """Should handle multiple raw datasets."""
        raw1 = RawDataset(np.random.randn(50, 3), ("a", "b", "c"), name="ds1")
        raw2 = RawDataset(np.random.randn(60, 4), ("w", "x", "y", "z"), name="ds2")
        
        prior = GeneratorPrior.uniform(minimal_registry)
        rng = RNGState(seed=42)
        
        results = generate_semisynthetic_batch(
            raw_datasets=[raw1, raw2],
            registry=minimal_registry,
            prior=prior,
            rng=rng,
            samples_per_dataset=3,
        )
        
        assert len(results) == 6  # 2 datasets * 3 samples


class TestSemiSyntheticDataLoader:
    """Tests for SemiSyntheticDataLoader."""
    
    def test_iteration(self, small_raw, minimal_registry):
        """Should iterate and produce batches."""
        prior = GeneratorPrior.uniform(minimal_registry)
        
        loader = SemiSyntheticDataLoader(
            raw_datasets=[small_raw],
            registry=minimal_registry,
            prior=prior,
            max_rows=64,
            max_cols=16,
            batch_size=8,
            batches_per_epoch=3,
            seed=42,
        )
        
        batches = list(loader)
        
        assert len(batches) == 3
        for batch in batches:
            assert batch.batch_size == 8
            assert batch.generator_ids is not None
            assert batch.class_ids is not None
    
    def test_row_subsampling(self, large_raw, minimal_registry):
        """Should subsample rows when dataset is large."""
        prior = GeneratorPrior.uniform(minimal_registry)
        
        max_rows = 64
        loader = SemiSyntheticDataLoader(
            raw_datasets=[large_raw],
            registry=minimal_registry,
            prior=prior,
            max_rows=max_rows,
            max_cols=16,
            batch_size=4,
            batches_per_epoch=2,
            seed=42,
        )
        
        batches = list(loader)
        
        # Tokens should have max_rows dimension
        for batch in batches:
            # Check that row dimension is bounded
            assert batch.tokens.shape[1] <= max_rows
    
    def test_with_real_dataset(self, breast_cancer_raw, minimal_registry):
        """Test with actual sklearn dataset."""
        prior = GeneratorPrior.uniform(minimal_registry)
        
        loader = SemiSyntheticDataLoader(
            raw_datasets=[breast_cancer_raw],
            registry=minimal_registry,
            prior=prior,
            max_rows=128,
            max_cols=32,
            batch_size=4,
            batches_per_epoch=2,
            seed=42,
        )
        
        batches = list(loader)
        
        assert len(batches) == 2
        # Should use all 30 features (padded to fit)
        assert batches[0].col_mask[0, :30].all()
    
    def test_reproducible(self, small_raw, minimal_registry):
        """Same seed should produce same batches."""
        prior = GeneratorPrior.uniform(minimal_registry)
        
        loader1 = SemiSyntheticDataLoader(
            raw_datasets=[small_raw],
            registry=minimal_registry,
            prior=prior,
            max_rows=64,
            max_cols=16,
            batch_size=4,
            batches_per_epoch=2,
            seed=999,
        )
        
        loader2 = SemiSyntheticDataLoader(
            raw_datasets=[small_raw],
            registry=minimal_registry,
            prior=prior,
            max_rows=64,
            max_cols=16,
            batch_size=4,
            batches_per_epoch=2,
            seed=999,
        )
        
        b1 = list(loader1)
        b2 = list(loader2)
        
        assert torch.equal(b1[0].generator_ids, b2[0].generator_ids)
        assert torch.equal(b1[0].tokens, b2[0].tokens)
    
    def test_rejects_empty_datasets(self, minimal_registry):
        """Should raise error with empty dataset list."""
        prior = GeneratorPrior.uniform(minimal_registry)
        
        with pytest.raises(ValueError, match="at least one"):
            SemiSyntheticDataLoader(
                raw_datasets=[],
                registry=minimal_registry,
                prior=prior,
                max_rows=64,
                max_cols=16,
                batch_size=4,
                batches_per_epoch=2,
            )
    
    def test_rejects_wide_datasets(self, minimal_registry):
        """Should raise error if dataset has too many columns."""
        wide_raw = RawDataset(
            np.random.randn(50, 20),
            tuple(f"col{i}" for i in range(20)),
            name="wide"
        )
        prior = GeneratorPrior.uniform(minimal_registry)
        
        with pytest.raises(ValueError, match="columns"):
            SemiSyntheticDataLoader(
                raw_datasets=[wide_raw],
                registry=minimal_registry,
                prior=prior,
                max_rows=64,
                max_cols=10,  # Less than dataset columns
                batch_size=4,
                batches_per_epoch=2,
            )


class TestMixedDataLoader:
    """Tests for MixedDataLoader."""
    
    def test_basic_iteration(self, small_raw, minimal_registry):
        """Should iterate through both loaders."""
        prior = GeneratorPrior.uniform(minimal_registry)
        
        from lacuna.data.batching import SyntheticDataLoader, SyntheticDataLoaderConfig
        
        syn_config = SyntheticDataLoaderConfig(
            batch_size=4,
            n_range=(30, 50),
            d_range=(3, 5),
            max_rows=64,
            max_cols=16,
            batches_per_epoch=3,
            seed=42,
        )
        syn_loader = SyntheticDataLoader(
            generators=list(minimal_registry.generators),
            config=syn_config,
        )
        
        semi_loader = SemiSyntheticDataLoader(
            raw_datasets=[small_raw],
            registry=minimal_registry,
            prior=prior,
            max_rows=64,
            max_cols=16,
            batch_size=4,
            batches_per_epoch=2,
            seed=43,
        )
        
        mixed = MixedDataLoader(
            synthetic_loader=syn_loader,
            semisynthetic_loader=semi_loader,
            mix_ratio=0.5,
            seed=44,
        )
        
        batches = list(mixed)
        
        # Should have all batches from both
        assert len(batches) == 5  # 3 + 2
