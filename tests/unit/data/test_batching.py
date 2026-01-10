"""
Tests for lacuna.data.batching

Tests the data loading infrastructure:
    - collate_fn: Collate function for PyTorch DataLoader
    - SyntheticDataLoaderConfig: Configuration dataclass
    - SyntheticDataLoader: On-the-fly synthetic data generation
    - ValidationDataLoader: Fixed validation data
    - MixedDataLoader: Mixing synthetic and semi-synthetic data
    - Factory functions: create_synthetic_loader, create_validation_loader
"""

import pytest
import torch
import numpy as np
from typing import List

from lacuna.data.batching import (
    collate_fn,
    SyntheticDataLoaderConfig,
    SyntheticDataLoader,
    ValidationDataLoader,
    create_synthetic_loader,
    create_validation_loader,
)
from lacuna.core.types import TokenBatch, ObservedDataset, MCAR, MAR, MNAR
from lacuna.core.rng import RNGState
from lacuna.data.tokenization import TOKEN_DIM
from lacuna.generators.base import BaseGenerator


# =============================================================================
# Mock Generator for Testing
# =============================================================================

class MockGenerator(BaseGenerator):
    """Simple generator for testing that produces predictable output."""
    
    def __init__(self, generator_id: int = 0, class_id: int = MCAR):
        super().__init__(
            generator_id=generator_id,
            class_id=class_id,
            name=f"mock_{generator_id}",
        )
        self.variant_id = 0 if class_id != MNAR else generator_id - 2
    
    def sample_observed(
        self,
        rng: RNGState,
        n: int,
        d: int,
        dataset_id: str = None,
    ) -> ObservedDataset:
        """Generate simple synthetic data."""
        np_rng = rng.numpy_rng
        
        # Simple multivariate normal
        X = np_rng.standard_normal((n, d)).astype(np.float32)
        
        # Random missingness (~20%)
        R = np_rng.random((n, d)) > 0.2
        
        # Apply missingness
        X_obs = X.copy()
        X_obs[~R] = np.nan
        
        return ObservedDataset(
            X_obs=X_obs,
            R=R,
            dataset_id=dataset_id or f"mock_{self.generator_id}",
            n_original=n,
            d_original=d,
        )


def create_mock_generators(n: int = 3) -> List[MockGenerator]:
    """Create list of mock generators covering all classes."""
    return [
        MockGenerator(generator_id=0, class_id=MCAR),
        MockGenerator(generator_id=1, class_id=MAR),
        MockGenerator(generator_id=2, class_id=MNAR),
    ][:n]


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_generators():
    """Create mock generators for testing."""
    return create_mock_generators(3)


@pytest.fixture
def default_config():
    """Default SyntheticDataLoaderConfig."""
    return SyntheticDataLoaderConfig(
        batch_size=4,
        n_range=(20, 50),
        d_range=(5, 10),
        max_rows=64,
        max_cols=16,
        apply_masking=False,
        seed=42,
    )


@pytest.fixture
def sample_token_batches():
    """Create sample TokenBatch objects for collate testing."""
    batches = []
    for i in range(3):
        batch = TokenBatch(
            tokens=torch.randn(1, 32, 16, TOKEN_DIM),
            row_mask=torch.ones(1, 32, dtype=torch.bool),
            col_mask=torch.ones(1, 16, dtype=torch.bool),
            generator_ids=torch.tensor([i]),
            class_ids=torch.tensor([i % 3]),
            variant_ids=torch.tensor([0]),
            original_values=torch.randn(1, 32, 16),
            reconstruction_mask=torch.rand(1, 32, 16) > 0.7,
        )
        batches.append(batch)
    return batches


# =============================================================================
# Test collate_fn
# =============================================================================

class TestCollateFn:
    """Tests for collate_fn."""
    
    def test_combines_batches(self, sample_token_batches):
        """Test that collate combines multiple batches."""
        combined = collate_fn(sample_token_batches)
        
        assert isinstance(combined, TokenBatch)
        assert combined.tokens.shape[0] == 3  # 3 batches of size 1
    
    def test_preserves_tensor_shapes(self, sample_token_batches):
        """Test that tensor dimensions are preserved."""
        combined = collate_fn(sample_token_batches)
        
        # [B=3, max_rows=32, max_cols=16, TOKEN_DIM]
        assert combined.tokens.shape == (3, 32, 16, TOKEN_DIM)
        assert combined.row_mask.shape == (3, 32)
        assert combined.col_mask.shape == (3, 16)
    
    def test_concatenates_labels(self, sample_token_batches):
        """Test that labels are concatenated."""
        combined = collate_fn(sample_token_batches)
        
        assert combined.generator_ids.shape == (3,)
        assert combined.class_ids.shape == (3,)
        assert combined.variant_ids.shape == (3,)
    
    def test_concatenates_optional_tensors(self, sample_token_batches):
        """Test that optional tensors are concatenated."""
        combined = collate_fn(sample_token_batches)
        
        assert combined.original_values.shape == (3, 32, 16)
        assert combined.reconstruction_mask.shape == (3, 32, 16)
    
    def test_handles_missing_optional_tensors(self):
        """Test handling when optional tensors are None."""
        batches = [
            TokenBatch(
                tokens=torch.randn(1, 16, 8, TOKEN_DIM),
                row_mask=torch.ones(1, 16, dtype=torch.bool),
                col_mask=torch.ones(1, 8, dtype=torch.bool),
            )
            for _ in range(3)
        ]
        
        combined = collate_fn(batches)
        
        assert combined.generator_ids is None
        assert combined.class_ids is None
        assert combined.original_values is None
    
    def test_handles_single_batch(self):
        """Test handling of single batch."""
        batch = TokenBatch(
            tokens=torch.randn(1, 32, 16, TOKEN_DIM),
            row_mask=torch.ones(1, 32, dtype=torch.bool),
            col_mask=torch.ones(1, 16, dtype=torch.bool),
        )
        
        combined = collate_fn([batch])
        
        assert combined.tokens.shape == (1, 32, 16, TOKEN_DIM)
    
    def test_handles_larger_batches(self):
        """Test combining batches that already have B > 1."""
        batches = [
            TokenBatch(
                tokens=torch.randn(4, 32, 16, TOKEN_DIM),
                row_mask=torch.ones(4, 32, dtype=torch.bool),
                col_mask=torch.ones(4, 16, dtype=torch.bool),
                class_ids=torch.randint(0, 3, (4,)),
            )
            for _ in range(3)
        ]
        
        combined = collate_fn(batches)
        
        assert combined.tokens.shape[0] == 12  # 3 * 4


# =============================================================================
# Test SyntheticDataLoaderConfig
# =============================================================================

class TestSyntheticDataLoaderConfig:
    """Tests for SyntheticDataLoaderConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = SyntheticDataLoaderConfig()
        
        assert config.batch_size == 32
        assert config.n_range == (50, 500)
        assert config.d_range == (5, 20)
        assert config.max_rows == 256
        assert config.max_cols == 32
        assert config.apply_masking is True
        assert config.mask_ratio == 0.15
        assert config.batches_per_epoch is None
        assert config.seed is None
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = SyntheticDataLoaderConfig(
            batch_size=16,
            n_range=(100, 200),
            d_range=(10, 30),
            max_rows=128,
            max_cols=64,
            apply_masking=False,
            mask_ratio=0.2,
            batches_per_epoch=100,
            seed=123,
        )
        
        assert config.batch_size == 16
        assert config.n_range == (100, 200)
        assert config.d_range == (10, 30)
        assert config.max_rows == 128
        assert config.max_cols == 64
        assert config.apply_masking is False
        assert config.mask_ratio == 0.2
        assert config.batches_per_epoch == 100
        assert config.seed == 123


# =============================================================================
# Test SyntheticDataLoader
# =============================================================================

class TestSyntheticDataLoader:
    """Tests for SyntheticDataLoader."""
    
    def test_initialization(self, mock_generators, default_config):
        """Test loader initialization."""
        loader = SyntheticDataLoader(
            generators=mock_generators,
            config=default_config,
        )
        
        assert loader.generators == mock_generators
        assert loader.config == default_config
    
    def test_iteration(self, mock_generators, default_config):
        """Test that loader can be iterated."""
        default_config.batches_per_epoch = 3
        loader = SyntheticDataLoader(
            generators=mock_generators,
            config=default_config,
        )
        
        batches = list(loader)
        
        assert len(batches) == 3
        for batch in batches:
            assert isinstance(batch, TokenBatch)
    
    def test_batch_shape(self, mock_generators, default_config):
        """Test that batches have correct shape."""
        loader = SyntheticDataLoader(
            generators=mock_generators,
            config=default_config,
        )
        
        batch = next(iter(loader))
        
        assert batch.tokens.shape[0] == default_config.batch_size
        assert batch.tokens.shape[1] == default_config.max_rows
        assert batch.tokens.shape[2] == default_config.max_cols
        assert batch.tokens.shape[3] == TOKEN_DIM
    
    def test_includes_labels(self, mock_generators, default_config):
        """Test that batches include labels."""
        loader = SyntheticDataLoader(
            generators=mock_generators,
            config=default_config,
        )
        
        batch = next(iter(loader))
        
        assert batch.generator_ids is not None
        assert batch.class_ids is not None
        assert batch.generator_ids.shape == (default_config.batch_size,)
        assert batch.class_ids.shape == (default_config.batch_size,)
    
    def test_class_ids_valid(self, mock_generators, default_config):
        """Test that class IDs are in valid range."""
        loader = SyntheticDataLoader(
            generators=mock_generators,
            config=default_config,
        )
        
        batch = next(iter(loader))
        
        assert (batch.class_ids >= 0).all()
        assert (batch.class_ids < 3).all()
    
    def test_with_masking(self, mock_generators):
        """Test with artificial masking enabled."""
        config = SyntheticDataLoaderConfig(
            batch_size=4,
            n_range=(20, 50),
            d_range=(5, 10),
            max_rows=64,
            max_cols=16,
            apply_masking=True,
            mask_ratio=0.15,
            seed=42,
        )
        loader = SyntheticDataLoader(
            generators=mock_generators,
            config=config,
        )
        
        batch = next(iter(loader))
        
        # Should have reconstruction targets
        assert batch.original_values is not None
        assert batch.reconstruction_mask is not None
    
    def test_without_masking(self, mock_generators, default_config):
        """Test with artificial masking disabled."""
        default_config.apply_masking = False
        loader = SyntheticDataLoader(
            generators=mock_generators,
            config=default_config,
        )
        
        batch = next(iter(loader))
        
        # May or may not have reconstruction targets depending on implementation
        # Main check is that it runs without error
        assert batch.tokens is not None
    
    def test_reproducibility_with_seed(self, mock_generators):
        """Test reproducibility when seed is set."""
        config = SyntheticDataLoaderConfig(
            batch_size=4,
            n_range=(20, 50),
            d_range=(5, 10),
            max_rows=64,
            max_cols=16,
            apply_masking=False,
            seed=42,
        )
        
        loader1 = SyntheticDataLoader(generators=mock_generators, config=config)
        loader2 = SyntheticDataLoader(generators=mock_generators, config=config)
        
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))
        
        # Generator IDs should match
        assert torch.equal(batch1.generator_ids, batch2.generator_ids)
        assert torch.equal(batch1.class_ids, batch2.class_ids)
    
    def test_different_seeds_different_data(self, mock_generators):
        """Test that different seeds produce different data."""
        config1 = SyntheticDataLoaderConfig(
            batch_size=4,
            n_range=(20, 50),
            d_range=(5, 10),
            max_rows=64,
            max_cols=16,
            seed=42,
        )
        config2 = SyntheticDataLoaderConfig(
            batch_size=4,
            n_range=(20, 50),
            d_range=(5, 10),
            max_rows=64,
            max_cols=16,
            seed=123,
        )
        
        loader1 = SyntheticDataLoader(generators=mock_generators, config=config1)
        loader2 = SyntheticDataLoader(generators=mock_generators, config=config2)
        
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))
        
        # Should be different (with high probability)
        assert not torch.equal(batch1.tokens, batch2.tokens)
    
    def test_len_with_batches_per_epoch(self, mock_generators, default_config):
        """Test __len__ when batches_per_epoch is set."""
        default_config.batches_per_epoch = 50
        loader = SyntheticDataLoader(
            generators=mock_generators,
            config=default_config,
        )
        
        assert len(loader) == 50
    
    def test_len_without_batches_per_epoch(self, mock_generators, default_config):
        """Test __len__ when batches_per_epoch is None (infinite)."""
        default_config.batches_per_epoch = None
        loader = SyntheticDataLoader(
            generators=mock_generators,
            config=default_config,
        )
        
        # Should return a large number for "infinite"
        assert len(loader) >= 1000
    
    def test_reset(self, mock_generators, default_config):
        """Test reset method."""
        default_config.batches_per_epoch = 5
        loader = SyntheticDataLoader(
            generators=mock_generators,
            config=default_config,
        )
        
        # Exhaust the loader
        _ = list(loader)
        
        # Reset and iterate again
        loader.reset()
        batches = list(loader)
        
        assert len(batches) == 5
    
    def test_reset_with_new_seed(self, mock_generators, default_config):
        """Test reset with new seed."""
        default_config.seed = 42
        loader = SyntheticDataLoader(
            generators=mock_generators,
            config=default_config,
        )
        
        batch1 = next(iter(loader))
        
        loader.reset(seed=123)
        batch2 = next(iter(loader))
        
        # Different seed should give different data
        assert not torch.equal(batch1.tokens, batch2.tokens)


# =============================================================================
# Test ValidationDataLoader
# =============================================================================

class TestValidationDataLoader:
    """Tests for ValidationDataLoader."""
    
    def test_initialization(self, mock_generators):
        """Test loader initialization."""
        loader = ValidationDataLoader(
            generators=mock_generators,
            n_samples=100,
            batch_size=32,
            seed=42,
        )
        
        assert len(loader.batches) > 0
    
    def test_fixed_batches(self, mock_generators):
        """Test that batches are fixed (pre-generated)."""
        loader = ValidationDataLoader(
            generators=mock_generators,
            n_samples=50,
            batch_size=16,
            seed=42,
        )
        
        batches1 = list(loader)
        batches2 = list(loader)
        
        # Same batches each time
        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2):
            assert torch.equal(b1.tokens, b2.tokens)
    
    def test_correct_number_of_batches(self, mock_generators):
        """Test that correct number of batches is generated."""
        loader = ValidationDataLoader(
            generators=mock_generators,
            n_samples=100,
            batch_size=32,
            seed=42,
        )
        
        # ceil(100 / 32) = 4 batches
        assert len(loader) == 4
    
    def test_reproducibility(self, mock_generators):
        """Test reproducibility with same seed."""
        loader1 = ValidationDataLoader(
            generators=mock_generators,
            n_samples=50,
            batch_size=16,
            seed=42,
        )
        loader2 = ValidationDataLoader(
            generators=mock_generators,
            n_samples=50,
            batch_size=16,
            seed=42,
        )
        
        batch1 = loader1.batches[0]
        batch2 = loader2.batches[0]
        
        assert torch.equal(batch1.tokens, batch2.tokens)


# =============================================================================
# Test Factory Functions
# =============================================================================

class TestCreateSyntheticLoader:
    """Tests for create_synthetic_loader factory."""
    
    def test_creates_loader(self, mock_generators):
        """Test factory creates loader."""
        loader = create_synthetic_loader(
            generators=mock_generators,
            batch_size=16,
            seed=42,
        )
        
        assert isinstance(loader, SyntheticDataLoader)
    
    def test_respects_parameters(self, mock_generators):
        """Test factory respects all parameters."""
        loader = create_synthetic_loader(
            generators=mock_generators,
            batch_size=16,
            n_range=(100, 200),
            d_range=(10, 20),
            max_rows=128,
            max_cols=64,
            apply_masking=True,
            mask_ratio=0.2,
            batches_per_epoch=50,
            seed=123,
        )
        
        assert loader.config.batch_size == 16
        assert loader.config.n_range == (100, 200)
        assert loader.config.d_range == (10, 20)
        assert loader.config.max_rows == 128
        assert loader.config.max_cols == 64
        assert loader.config.apply_masking is True
        assert loader.config.mask_ratio == 0.2
        assert loader.config.batches_per_epoch == 50
        assert loader.config.seed == 123


class TestCreateValidationLoader:
    """Tests for create_validation_loader factory."""
    
    def test_creates_loader(self, mock_generators):
        """Test factory creates loader."""
        loader = create_validation_loader(
            generators=mock_generators,
            n_samples=100,
            batch_size=32,
            seed=42,
        )
        
        assert isinstance(loader, ValidationDataLoader)
    
    def test_respects_parameters(self, mock_generators):
        """Test factory respects parameters."""
        loader = create_validation_loader(
            generators=mock_generators,
            n_samples=200,
            batch_size=64,
            max_rows=128,
            max_cols=32,
            seed=123,
        )
        
        assert loader.batch_size == 64
        # Check batch dimensions
        if len(loader.batches) > 0:
            batch = loader.batches[0]
            assert batch.tokens.shape[1] == 128  # max_rows
            assert batch.tokens.shape[2] == 32   # max_cols


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_generator(self):
        """Test with single generator."""
        generators = [MockGenerator(generator_id=0, class_id=MCAR)]
        config = SyntheticDataLoaderConfig(
            batch_size=4,
            n_range=(20, 50),
            d_range=(5, 10),
            max_rows=64,
            max_cols=16,
            seed=42,
        )
        
        loader = SyntheticDataLoader(generators=generators, config=config)
        batch = next(iter(loader))
        
        # All samples should be from the same generator
        assert (batch.generator_ids == 0).all()
        assert (batch.class_ids == MCAR).all()
    
    def test_small_dataset_dimensions(self):
        """Test with very small dataset dimensions."""
        generators = create_mock_generators(2)
        config = SyntheticDataLoaderConfig(
            batch_size=2,
            n_range=(5, 10),
            d_range=(2, 4),
            max_rows=16,
            max_cols=8,
            seed=42,
        )
        
        loader = SyntheticDataLoader(generators=generators, config=config)
        batch = next(iter(loader))
        
        assert batch.tokens.shape == (2, 16, 8, TOKEN_DIM)
    
    def test_max_equals_min_range(self):
        """Test when n_range and d_range have min==max."""
        generators = create_mock_generators(2)
        config = SyntheticDataLoaderConfig(
            batch_size=2,
            n_range=(50, 50),  # Fixed size
            d_range=(10, 10),  # Fixed size
            max_rows=64,
            max_cols=16,
            seed=42,
        )
        
        loader = SyntheticDataLoader(generators=generators, config=config)
        batch = next(iter(loader))
        
        # Should work without error
        assert batch.tokens is not None
    
    def test_many_generators(self):
        """Test with many generators."""
        generators = [
            MockGenerator(generator_id=i, class_id=i % 3)
            for i in range(10)
        ]
        config = SyntheticDataLoaderConfig(
            batch_size=16,
            n_range=(20, 50),
            d_range=(5, 10),
            max_rows=64,
            max_cols=16,
            seed=42,
        )
        
        loader = SyntheticDataLoader(generators=generators, config=config)
        batch = next(iter(loader))
        
        # Generator IDs should be in valid range
        assert (batch.generator_ids >= 0).all()
        assert (batch.generator_ids < 10).all()