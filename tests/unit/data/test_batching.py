"""
Tests for lacuna.data.batching
"""

import pytest
import torch
from lacuna.core.rng import RNGState
from lacuna.data.observed import create_observed_dataset
from lacuna.data.batching import tokenize_and_batch, SyntheticDataLoader
from lacuna.data.features import FEATURE_DIM


class TestTokenizeAndBatch:
    """Tests for tokenize_and_batch."""
    
    def test_output_shape(self):
        datasets = []
        for i in range(4):
            x = torch.randn(50, 5)
            r = torch.ones(50, 5, dtype=torch.bool)
            datasets.append(create_observed_dataset(x=x, r=r, dataset_id=f"ds_{i}"))
        
        batch = tokenize_and_batch(datasets, max_cols=10)
        
        assert batch.tokens.shape == (4, 10, FEATURE_DIM)
        assert batch.col_mask.shape == (4, 10)
        assert batch.batch_size == 4
    
    def test_padding(self):
        # Dataset with 3 columns
        x = torch.randn(50, 3)
        r = torch.ones(50, 3, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        batch = tokenize_and_batch([ds], max_cols=10)
        
        # First 3 columns should be valid
        assert batch.col_mask[0, :3].all()
        # Rest should be padding
        assert not batch.col_mask[0, 3:].any()
    
    def test_truncation(self):
        # Dataset with 10 columns
        x = torch.randn(50, 10)
        r = torch.ones(50, 10, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        batch = tokenize_and_batch([ds], max_cols=5)
        
        # Only first 5 columns kept
        assert batch.tokens.shape == (1, 5, FEATURE_DIM)
        assert batch.col_mask[0].all()
    
    def test_with_labels(self):
        datasets = []
        for i in range(4):
            x = torch.randn(30, 3)
            r = torch.ones(30, 3, dtype=torch.bool)
            datasets.append(create_observed_dataset(x=x, r=r))
        
        gen_ids = [0, 1, 2, 3]
        class_mapping = torch.tensor([0, 0, 1, 2])  # gen -> class
        
        batch = tokenize_and_batch(
            datasets,
            max_cols=5,
            generator_ids=gen_ids,
            class_mapping=class_mapping,
        )
        
        assert batch.generator_ids is not None
        assert batch.class_ids is not None
        assert torch.equal(batch.generator_ids, torch.tensor([0, 1, 2, 3]))
        assert torch.equal(batch.class_ids, torch.tensor([0, 0, 1, 2]))
    
    def test_variable_column_counts(self):
        # Datasets with different column counts
        ds1 = create_observed_dataset(torch.randn(30, 3), dataset_id="a")
        ds2 = create_observed_dataset(torch.randn(30, 7), dataset_id="b")
        ds3 = create_observed_dataset(torch.randn(30, 5), dataset_id="c")
        
        batch = tokenize_and_batch([ds1, ds2, ds3], max_cols=10)
        
        assert batch.tokens.shape == (3, 10, FEATURE_DIM)
        assert batch.col_mask[0, :3].all() and not batch.col_mask[0, 3:].any()
        assert batch.col_mask[1, :7].all() and not batch.col_mask[1, 7:].any()
        assert batch.col_mask[2, :5].all() and not batch.col_mask[2, 5:].any()


class TestSyntheticDataLoader:
    """Tests for SyntheticDataLoader."""
    
    def test_iteration(self, minimal_registry):
        from lacuna.generators import GeneratorPrior
        
        prior = GeneratorPrior.uniform(minimal_registry)
        
        loader = SyntheticDataLoader(
            registry=minimal_registry,
            prior=prior,
            n_range=(50, 100),
            d_range=(3, 8),
            max_cols=16,
            batch_size=8,
            batches_per_epoch=5,
            seed=42,
        )
        
        batches = list(loader)
        
        assert len(batches) == 5
        for batch in batches:
            assert batch.batch_size == 8
            assert batch.tokens.shape == (8, 16, FEATURE_DIM)
            assert batch.generator_ids is not None
            assert batch.class_ids is not None
    
    def test_reproducible(self, minimal_registry):
        from lacuna.generators import GeneratorPrior
        
        prior = GeneratorPrior.uniform(minimal_registry)
        
        loader1 = SyntheticDataLoader(
            registry=minimal_registry,
            prior=prior,
            n_range=(50, 100),
            d_range=(3, 8),
            max_cols=16,
            batch_size=4,
            batches_per_epoch=3,
            seed=123,
        )
        
        loader2 = SyntheticDataLoader(
            registry=minimal_registry,
            prior=prior,
            n_range=(50, 100),
            d_range=(3, 8),
            max_cols=16,
            batch_size=4,
            batches_per_epoch=3,
            seed=123,
        )
        
        batches1 = list(loader1)
        batches2 = list(loader2)
        
        for b1, b2 in zip(batches1, batches2):
            assert torch.equal(b1.tokens, b2.tokens)
            assert torch.equal(b1.generator_ids, b2.generator_ids)
    
    def test_different_seeds(self, minimal_registry):
        from lacuna.generators import GeneratorPrior
        
        prior = GeneratorPrior.uniform(minimal_registry)
        
        loader1 = SyntheticDataLoader(
            registry=minimal_registry,
            prior=prior,
            n_range=(50, 100),
            d_range=(3, 8),
            max_cols=16,
            batch_size=4,
            batches_per_epoch=2,
            seed=1,
        )
        
        loader2 = SyntheticDataLoader(
            registry=minimal_registry,
            prior=prior,
            n_range=(50, 100),
            d_range=(3, 8),
            max_cols=16,
            batch_size=4,
            batches_per_epoch=2,
            seed=2,
        )
        
        batches1 = list(loader1)
        batches2 = list(loader2)
        
        # Should be different
        assert not torch.equal(batches1[0].tokens, batches2[0].tokens)
