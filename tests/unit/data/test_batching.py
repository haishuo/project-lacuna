"""
Tests for lacuna.data.batching (row-level)
"""

import pytest
import torch
from lacuna.core.rng import RNGState
from lacuna.data.observed import create_observed_dataset
from lacuna.data.batching import tokenize_and_batch, SyntheticDataLoader
from lacuna.data.tokenization import TOKEN_DIM


class TestTokenizeAndBatch:
    """Tests for tokenize_and_batch."""
    
    def test_output_shape(self):
        datasets = []
        for i in range(4):
            x = torch.randn(50, 5)
            r = torch.ones(50, 5, dtype=torch.bool)
            ds = create_observed_dataset(x=x, r=r, dataset_id=f"ds_{i}")
            datasets.append(ds)
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=64,
            max_cols=8,
            generator_ids=[0, 1, 2, 3],
            class_mapping=torch.tensor([0, 0, 1, 1]),
        )
        
        assert batch.tokens.shape == (4, 64, 8, TOKEN_DIM)
        assert batch.row_mask.shape == (4, 64)
        assert batch.col_mask.shape == (4, 8)
        assert batch.generator_ids.shape == (4,)
        assert batch.class_ids.shape == (4,)
    
    def test_row_padding(self):
        # Dataset with fewer rows than max_rows
        x = torch.randn(30, 5)
        r = torch.ones(30, 5, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        batch = tokenize_and_batch(
            datasets=[ds],
            max_rows=64,
            max_cols=8,
        )
        
        # First 30 rows valid, rest padding
        assert batch.row_mask[0, :30].all()
        assert not batch.row_mask[0, 30:].any()
    
    def test_col_padding(self):
        # Dataset with fewer cols than max_cols
        x = torch.randn(50, 3)
        r = torch.ones(50, 3, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        batch = tokenize_and_batch(
            datasets=[ds],
            max_rows=64,
            max_cols=8,
        )
        
        # First 3 cols valid, rest padding
        assert batch.col_mask[0, :3].all()
        assert not batch.col_mask[0, 3:].any()
    
    def test_row_sampling(self):
        # Dataset with more rows than max_rows
        x = torch.randn(200, 5)
        r = torch.ones(200, 5, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        batch = tokenize_and_batch(
            datasets=[ds],
            max_rows=64,
            max_cols=8,
            row_sample_seed=42,
        )
        
        # Should have exactly max_rows valid rows
        assert batch.row_mask[0].sum() == 64
    
    def test_variable_dataset_sizes(self):
        datasets = []
        for n, d in [(30, 3), (50, 5), (100, 8), (20, 2)]:
            x = torch.randn(n, d)
            r = torch.ones(n, d, dtype=torch.bool)
            ds = create_observed_dataset(x=x, r=r, dataset_id=f"ds_{n}_{d}")
            datasets.append(ds)
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=64,
            max_cols=10,
        )
        
        # Check masks are correct for each dataset
        assert batch.row_mask[0, :30].all() and not batch.row_mask[0, 30:].any()
        assert batch.row_mask[1, :50].all() and not batch.row_mask[1, 50:].any()
        assert batch.row_mask[2, :64].all()  # Sampled down
        assert batch.row_mask[3, :20].all() and not batch.row_mask[3, 20:].any()


class TestSyntheticDataLoader:
    """Tests for SyntheticDataLoader."""
    
    @pytest.fixture
    def minimal_registry(self):
        from lacuna.generators import create_minimal_registry
        return create_minimal_registry()
    
    def test_iteration(self, minimal_registry):
        from lacuna.generators import GeneratorPrior
        
        prior = GeneratorPrior.uniform(minimal_registry)
        
        loader = SyntheticDataLoader(
            registry=minimal_registry,
            prior=prior,
            n_range=(50, 100),
            d_range=(3, 8),
            max_rows=64,
            max_cols=16,
            batch_size=8,
            batches_per_epoch=5,
            seed=42,
        )
        
        batches = list(loader)
        
        assert len(batches) == 5
        for batch in batches:
            assert batch.batch_size == 8
            assert batch.tokens.shape == (8, 64, 16, TOKEN_DIM)
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
            max_rows=64,
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
            max_rows=64,
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
