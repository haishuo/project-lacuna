"""
Tests for lacuna.data.semisynthetic
"""

import pytest
import torch
import numpy as np

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.data.ingestion import RawDataset, load_sklearn_dataset
from lacuna.data.semisynthetic import (
    apply_missingness,
    generate_semisynthetic_batch,
    SemiSyntheticDataLoader,
)
from lacuna.generators import create_minimal_registry, GeneratorPrior


@pytest.fixture
def breast_cancer_raw():
    """Load breast cancer as test dataset."""
    return load_sklearn_dataset("breast_cancer")


@pytest.fixture
def small_raw():
    """Small synthetic raw dataset for fast tests."""
    return RawDataset(
        data=np.random.randn(100, 5),
        feature_names=("a", "b", "c", "d", "e"),
        name="small_test",
    )


class TestApplyMissingness:
    """Tests for apply_missingness."""
    
    def test_creates_missingness(self, small_raw, minimal_registry):
        rng = RNGState(seed=42)
        generator = minimal_registry[0]  # MCAR-Uniform-10
        
        ss = apply_missingness(small_raw, generator, rng)
        
        # Should have some missing values
        assert ss.observed.n_missing > 0
        # Complete data preserved
        assert ss.complete.shape == (small_raw.n, small_raw.d)
        # Metadata correct
        assert ss.generator_id == 0
        assert ss.class_id == MCAR
    
    def test_preserves_observed_values(self, small_raw, minimal_registry):
        rng = RNGState(seed=42)
        generator = minimal_registry[0]
        
        ss = apply_missingness(small_raw, generator, rng)
        
        # Observed values should match original
        observed_mask = ss.observed.r
        original_tensor = torch.from_numpy(small_raw.data.astype('float32'))
        
        assert torch.allclose(
            ss.observed.x[observed_mask],
            original_tensor[observed_mask],
        )
    
    def test_reproducible(self, small_raw, minimal_registry):
        generator = minimal_registry[0]
        
        rng1 = RNGState(seed=123)
        rng2 = RNGState(seed=123)
        
        ss1 = apply_missingness(small_raw, generator, rng1)
        ss2 = apply_missingness(small_raw, generator, rng2)
        
        assert torch.equal(ss1.observed.r, ss2.observed.r)
    
    def test_different_generators_different_patterns(self, small_raw, minimal_registry):
        rng = RNGState(seed=42)
        
        ss_mcar = apply_missingness(small_raw, minimal_registry[0], rng.spawn())  # MCAR
        ss_mar = apply_missingness(small_raw, minimal_registry[2], rng.spawn())   # MAR
        ss_mnar = apply_missingness(small_raw, minimal_registry[4], rng.spawn())  # MNAR
        
        # All should have missingness but different patterns
        assert not torch.equal(ss_mcar.observed.r, ss_mar.observed.r)
        assert not torch.equal(ss_mar.observed.r, ss_mnar.observed.r)


class TestGenerateSemisyntheticBatch:
    """Tests for generate_semisynthetic_batch."""
    
    def test_generates_correct_count(self, small_raw, minimal_registry):
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
        prior = GeneratorPrior.uniform(minimal_registry)
        
        loader = SemiSyntheticDataLoader(
            raw_datasets=[small_raw],
            registry=minimal_registry,
            prior=prior,
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
    
    def test_with_real_dataset(self, breast_cancer_raw, minimal_registry):
        """Test with actual sklearn dataset."""
        prior = GeneratorPrior.uniform(minimal_registry)
        
        loader = SemiSyntheticDataLoader(
            raw_datasets=[breast_cancer_raw],
            registry=minimal_registry,
            prior=prior,
            max_cols=32,
            batch_size=4,
            batches_per_epoch=2,
            seed=42,
        )
        
        batches = list(loader)
        
        assert len(batches) == 2
        # Should use all 30 features (padded to 32)
        assert batches[0].col_mask[0, :30].all()
    
    def test_reproducible(self, small_raw, minimal_registry):
        prior = GeneratorPrior.uniform(minimal_registry)
        
        loader1 = SemiSyntheticDataLoader(
            raw_datasets=[small_raw],
            registry=minimal_registry,
            prior=prior,
            max_cols=16,
            batch_size=4,
            batches_per_epoch=2,
            seed=999,
        )
        
        loader2 = SemiSyntheticDataLoader(
            raw_datasets=[small_raw],
            registry=minimal_registry,
            prior=prior,
            max_cols=16,
            batch_size=4,
            batches_per_epoch=2,
            seed=999,
        )
        
        b1 = list(loader1)
        b2 = list(loader2)
        
        assert torch.equal(b1[0].generator_ids, b2[0].generator_ids)
