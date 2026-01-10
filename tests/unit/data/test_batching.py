"""
Tests for lacuna.data.batching

Tests the data loading infrastructure:
    - collate_fn: Collate function for PyTorch DataLoader
    - SyntheticDataLoaderConfig: Configuration dataclass
    - SyntheticDataLoader: On-the-fly synthetic data generation
    - ValidationDataLoader: Fixed validation data
    - Factory functions: create_synthetic_loader, create_validation_loader
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple

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
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams


# =============================================================================
# Mock Generator for Testing
# =============================================================================

class MockGenerator(Generator):
    """Simple generator for testing that produces predictable output."""
    
    def __init__(self, generator_id: int = 0, class_id: int = MCAR):
        params = GeneratorParams(miss_rate=0.2)
        super().__init__(
            generator_id=generator_id,
            name=f"mock_{generator_id}",
            class_id=class_id,
            params=params,
        )
        # Store variant_id for MNAR generators
        self._variant_id = 0 if class_id != MNAR else generator_id - 2
    
    @property
    def variant_id(self) -> int:
        """Return variant ID for MNAR generators."""
        return self._variant_id
    
    def sample(
        self,
        rng: RNGState,
        n: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate simple synthetic data.
        
        Args:
            rng: RNG state for reproducibility.
            n: Number of rows.
            d: Number of columns.
            
        Returns:
            X_full: [n, d] complete data tensor.
            R: [n, d] bool tensor indicating observed values.
        """
        # Generate complete data using torch
        X = rng.randn(n, d)
        
        # Generate random missingness (~20%)
        R = rng.rand(n, d) > 0.2
        
        # Ensure at least one observed value per column
        for j in range(d):
            if not R[:, j].any():
                R[0, j] = True
        
        return X, R


def create_mock_generators(n: int = 3) -> List[MockGenerator]:
    """Create list of mock generators covering all classes.
    
    Args:
        n: Number of generators to create (max 3).
        
    Returns:
        List of MockGenerator instances.
    """
    generators = [
        MockGenerator(generator_id=0, class_id=MCAR),
        MockGenerator(generator_id=1, class_id=MAR),
        MockGenerator(generator_id=2, class_id=MNAR),
    ]
    return generators[:n]


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
        n_range=(50, 100),
        d_range=(3, 5),
        max_rows=128,  # Must be >= n_range[1] to avoid shape mismatch
        max_cols=8,
        mask_ratio=0.15,
        seed=42,
        batches_per_epoch=10,
    )


@pytest.fixture
def small_config():
    """Small config for fast tests."""
    return SyntheticDataLoaderConfig(
        batch_size=2,
        n_range=(10, 20),
        d_range=(2, 3),
        max_rows=32,
        max_cols=8,
        mask_ratio=0.15,
        seed=42,
        batches_per_epoch=5,
    )


@pytest.fixture
def sample_token_batches():
    """Create sample TokenBatch objects for collate testing."""
    max_rows, max_cols = 32, 16
    batches = []
    for i in range(3):
        batches.append(TokenBatch(
            tokens=torch.randn(1, max_rows, max_cols, TOKEN_DIM),
            row_mask=torch.ones(1, max_rows, dtype=torch.bool),
            col_mask=torch.ones(1, max_cols, dtype=torch.bool),
            generator_ids=torch.tensor([i]),
            class_ids=torch.tensor([i % 3]),
            variant_ids=torch.tensor([-1]),
            original_values=torch.randn(1, max_rows, max_cols),
            reconstruction_mask=torch.zeros(1, max_rows, max_cols, dtype=torch.bool),
        ))
    return batches


# =============================================================================
# Test collate_fn
# =============================================================================

class TestCollateFn:
    """Tests for the collate function."""
    
    def test_collate_single_batch(self):
        """Collate a single TokenBatch returns it unchanged."""
        max_rows, max_cols = 32, 16
        batch = TokenBatch(
            tokens=torch.randn(1, max_rows, max_cols, TOKEN_DIM),
            row_mask=torch.ones(1, max_rows, dtype=torch.bool),
            col_mask=torch.ones(1, max_cols, dtype=torch.bool),
            generator_ids=torch.zeros(1, dtype=torch.long),
            class_ids=torch.zeros(1, dtype=torch.long),
            variant_ids=torch.full((1,), -1, dtype=torch.long),
        )
        
        result = collate_fn([batch])
        
        assert result.tokens.shape == batch.tokens.shape
        assert result.row_mask.shape == batch.row_mask.shape
        assert result.col_mask.shape == batch.col_mask.shape
        assert torch.equal(result.tokens, batch.tokens)
    
    def test_collate_multiple_batches(self):
        """Collate multiple TokenBatches concatenates them."""
        max_rows, max_cols = 32, 16
        batch1 = TokenBatch(
            tokens=torch.randn(1, max_rows, max_cols, TOKEN_DIM),
            row_mask=torch.ones(1, max_rows, dtype=torch.bool),
            col_mask=torch.ones(1, max_cols, dtype=torch.bool),
            generator_ids=torch.tensor([0]),
            class_ids=torch.tensor([MCAR]),
            variant_ids=torch.tensor([-1]),
        )
        batch2 = TokenBatch(
            tokens=torch.randn(1, max_rows, max_cols, TOKEN_DIM),
            row_mask=torch.zeros(1, max_rows, dtype=torch.bool),
            col_mask=torch.zeros(1, max_cols, dtype=torch.bool),
            generator_ids=torch.tensor([1]),
            class_ids=torch.tensor([MAR]),
            variant_ids=torch.tensor([-1]),
        )
        
        result = collate_fn([batch1, batch2])
        
        # Tokens should be concatenated along batch dimension
        assert result.tokens.shape[0] == 2
        assert result.tokens.shape[1] == max_rows
        assert result.tokens.shape[2] == max_cols
        assert result.tokens.shape[3] == TOKEN_DIM
        
        # Generator IDs concatenated
        assert result.generator_ids.shape[0] == 2
        assert result.generator_ids[0] == 0
        assert result.generator_ids[1] == 1
        
        # Class IDs concatenated
        assert result.class_ids[0] == MCAR
        assert result.class_ids[1] == MAR
    
    def test_collate_preserves_dtype(self):
        """Collate preserves tensor dtypes."""
        max_rows, max_cols = 32, 16
        batch = TokenBatch(
            tokens=torch.randn(1, max_rows, max_cols, TOKEN_DIM, dtype=torch.float32),
            row_mask=torch.ones(1, max_rows, dtype=torch.bool),
            col_mask=torch.ones(1, max_cols, dtype=torch.bool),
            generator_ids=torch.tensor([0], dtype=torch.long),
            class_ids=torch.tensor([0], dtype=torch.long),
            variant_ids=torch.tensor([-1], dtype=torch.long),
        )
        
        result = collate_fn([batch])
        
        assert result.tokens.dtype == torch.float32
        assert result.row_mask.dtype == torch.bool
        assert result.col_mask.dtype == torch.bool
        assert result.generator_ids.dtype == torch.long
    
    def test_concatenates_labels(self, sample_token_batches):
        """Test that labels are properly concatenated."""
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
        max_rows, max_cols = 16, 8
        batches = [
            TokenBatch(
                tokens=torch.randn(1, max_rows, max_cols, TOKEN_DIM),
                row_mask=torch.ones(1, max_rows, dtype=torch.bool),
                col_mask=torch.ones(1, max_cols, dtype=torch.bool),
            )
            for _ in range(3)
        ]
        
        combined = collate_fn(batches)
        
        assert combined.generator_ids is None
        assert combined.class_ids is None
        assert combined.original_values is None
    
    def test_handles_single_batch(self):
        """Test handling of single batch."""
        max_rows, max_cols = 32, 16
        batch = TokenBatch(
            tokens=torch.randn(1, max_rows, max_cols, TOKEN_DIM),
            row_mask=torch.ones(1, max_rows, dtype=torch.bool),
            col_mask=torch.ones(1, max_cols, dtype=torch.bool),
        )
        
        combined = collate_fn([batch])
        
        assert combined.tokens.shape == (1, max_rows, max_cols, TOKEN_DIM)
    
    def test_handles_larger_batches(self):
        """Test combining batches that already have B > 1."""
        max_rows, max_cols = 32, 16
        batches = [
            TokenBatch(
                tokens=torch.randn(4, max_rows, max_cols, TOKEN_DIM),
                row_mask=torch.ones(4, max_rows, dtype=torch.bool),
                col_mask=torch.ones(4, max_cols, dtype=torch.bool),
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
    """Tests for configuration dataclass."""
    
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
    
    def test_single_value_ranges(self):
        """Config can use single-value tuples for fixed sizes."""
        config = SyntheticDataLoaderConfig(
            n_range=(100, 100),  # Fixed at 100
            d_range=(5, 5),      # Fixed at 5
        )
        
        assert config.n_range[0] == config.n_range[1]
        assert config.d_range[0] == config.d_range[1]


# =============================================================================
# Test SyntheticDataLoader
# =============================================================================

class TestSyntheticDataLoader:
    """Tests for the synthetic data loader."""
    
    def test_construction(self, mock_generators, small_config):
        """Test loader construction."""
        loader = SyntheticDataLoader(mock_generators, small_config)
        
        assert len(loader.generators) == 3
        assert loader.config == small_config
    
    def test_iteration_produces_batches(self, mock_generators, small_config):
        """Iterating produces TokenBatch objects."""
        loader = SyntheticDataLoader(mock_generators, small_config)
        
        batch = next(iter(loader))
        
        assert isinstance(batch, TokenBatch)
        assert batch.tokens.shape[3] == TOKEN_DIM
        assert batch.row_mask.dtype == torch.bool
        assert batch.col_mask.dtype == torch.bool
    
    def test_batch_has_correct_structure(self, mock_generators, small_config):
        """Batches have all required fields."""
        loader = SyntheticDataLoader(mock_generators, small_config)
        
        batch = next(iter(loader))
        
        # Check all fields exist
        assert hasattr(batch, 'tokens')
        assert hasattr(batch, 'row_mask')
        assert hasattr(batch, 'col_mask')
        assert hasattr(batch, 'generator_ids')
        assert hasattr(batch, 'class_ids')
        assert hasattr(batch, 'variant_ids')
        
        # Check shape is [B, max_rows, max_cols, TOKEN_DIM]
        B = batch.tokens.shape[0]
        assert batch.tokens.shape == (B, small_config.max_rows, small_config.max_cols, TOKEN_DIM)
        assert batch.row_mask.shape == (B, small_config.max_rows)
        assert batch.col_mask.shape == (B, small_config.max_cols)
    
    def test_respects_batches_per_epoch(self, mock_generators, small_config):
        """Loader stops after batches_per_epoch batches."""
        loader = SyntheticDataLoader(mock_generators, small_config)
        
        batches = list(loader)
        
        assert len(batches) == small_config.batches_per_epoch
    
    def test_len_returns_batches_per_epoch(self, mock_generators, small_config):
        """__len__ returns correct value."""
        loader = SyntheticDataLoader(mock_generators, small_config)
        
        assert len(loader) == small_config.batches_per_epoch
    
    def test_reproducibility_with_seed(self, mock_generators):
        """Same seed produces same data."""
        config = SyntheticDataLoaderConfig(
            batch_size=2,
            n_range=(10, 20),
            d_range=(3, 3),
            max_rows=32,
            max_cols=8,
            seed=12345,
            batches_per_epoch=3,
        )
        
        loader1 = SyntheticDataLoader(mock_generators, config)
        loader2 = SyntheticDataLoader(mock_generators, config)
        
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))
        
        # Same seed should produce same tokens
        assert torch.allclose(batch1.tokens, batch2.tokens)
    
    def test_different_seeds_produce_different_data(self, mock_generators):
        """Different seeds produce different data."""
        config1 = SyntheticDataLoaderConfig(
            batch_size=2,
            n_range=(10, 20),
            d_range=(3, 3),
            max_rows=32,
            max_cols=8,
            seed=111,
            batches_per_epoch=3,
        )
        config2 = SyntheticDataLoaderConfig(
            batch_size=2,
            n_range=(10, 20),
            d_range=(3, 3),
            max_rows=32,
            max_cols=8,
            seed=222,
            batches_per_epoch=3,
        )
        
        loader1 = SyntheticDataLoader(mock_generators, config1)
        loader2 = SyntheticDataLoader(mock_generators, config2)
        
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))
        
        # Different seeds should produce different tokens
        assert not torch.allclose(batch1.tokens, batch2.tokens)
    
    def test_class_ids_match_generators(self, mock_generators, small_config):
        """Class IDs in batches match generator class IDs."""
        loader = SyntheticDataLoader(mock_generators, small_config)
        
        # Build expected class mapping
        class_mapping = {g.generator_id: g.class_id for g in mock_generators}
        
        for batch in loader:
            for i, gen_id in enumerate(batch.generator_ids):
                expected_class = class_mapping[gen_id.item()]
                actual_class = batch.class_ids[i].item()
                assert actual_class == expected_class
    
    def test_batch_dimensions(self, mock_generators, default_config):
        """Test batch tensor dimensions."""
        loader = SyntheticDataLoader(mock_generators, default_config)
        
        batch = next(iter(loader))
        
        B = default_config.batch_size
        max_rows = default_config.max_rows
        max_cols = default_config.max_cols
        
        assert batch.tokens.shape == (B, max_rows, max_cols, TOKEN_DIM)
        assert batch.row_mask.shape == (B, max_rows)
        assert batch.col_mask.shape == (B, max_cols)
    
    def test_includes_labels(self, mock_generators, default_config):
        """Test that batches include labels."""
        loader = SyntheticDataLoader(mock_generators, default_config)
        
        batch = next(iter(loader))
        
        B = default_config.batch_size
        assert batch.generator_ids is not None
        assert batch.generator_ids.shape == (B,)
        assert batch.class_ids is not None
        assert batch.class_ids.shape == (B,)


# =============================================================================
# Test ValidationDataLoader
# =============================================================================

class TestValidationDataLoader:
    """Tests for the validation data loader."""
    
    def test_construction(self, mock_generators):
        """Test validation loader construction."""
        loader = ValidationDataLoader(
            generators=mock_generators,
            n_samples=20,
            batch_size=4,
            max_rows=64,
            max_cols=16,
            seed=42,
        )
        
        # Should have ceil(20/4) = 5 batches
        assert len(loader.batches) == 5
    
    def test_produces_fixed_data(self, mock_generators):
        """Validation loader produces same data each iteration."""
        loader = ValidationDataLoader(
            generators=mock_generators,
            n_samples=10,
            batch_size=4,
            max_rows=32,
            max_cols=8,
            seed=42,
        )
        
        # First iteration
        batches1 = list(loader)
        
        # Second iteration
        batches2 = list(loader)
        
        # Should be identical
        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2):
            assert torch.equal(b1.tokens, b2.tokens)
            assert torch.equal(b1.row_mask, b2.row_mask)
            assert torch.equal(b1.col_mask, b2.col_mask)
    
    def test_len_is_deterministic(self, mock_generators):
        """Validation loader length is fixed."""
        loader = ValidationDataLoader(
            generators=mock_generators,
            n_samples=12,
            batch_size=4,
            max_rows=32,
            max_cols=8,
            seed=42,
        )
        
        # Length should equal ceil(n_samples / batch_size)
        expected_len = (12 + 4 - 1) // 4  # ceil division = 3
        assert len(loader) == expected_len
        
        # Actually iterate and count
        batches = list(loader)
        assert len(batches) == expected_len


# =============================================================================
# Test Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Tests for loader factory functions."""
    
    def test_create_synthetic_loader(self, mock_generators):
        """Test create_synthetic_loader factory."""
        loader = create_synthetic_loader(
            generators=mock_generators,
            batch_size=8,
            seed=42,
            batches_per_epoch=5,
        )
        
        assert isinstance(loader, SyntheticDataLoader)
        assert loader.config.batch_size == 8
        assert loader.config.seed == 42
        assert loader.config.batches_per_epoch == 5
    
    def test_create_validation_loader(self, mock_generators):
        """Test create_validation_loader factory."""
        loader = create_validation_loader(
            generators=mock_generators,
            n_samples=20,
            batch_size=4,
            seed=42,
        )
        
        assert isinstance(loader, ValidationDataLoader)
        assert len(loader.batches) == 5  # ceil(20/4)
    
    def test_create_synthetic_loader_with_all_params(self, mock_generators):
        """Test factory with all parameters."""
        loader = create_synthetic_loader(
            generators=mock_generators,
            batch_size=16,
            n_range=(100, 200),
            d_range=(5, 15),
            max_rows=128,
            max_cols=32,
            apply_masking=True,
            mask_ratio=0.2,
            batches_per_epoch=50,
            seed=999,
        )
        
        assert loader.config.batch_size == 16
        assert loader.config.n_range == (100, 200)
        assert loader.config.d_range == (5, 15)
        assert loader.config.max_rows == 128
        assert loader.config.max_cols == 32
        assert loader.config.apply_masking is True
        assert loader.config.mask_ratio == 0.2
        assert loader.config.batches_per_epoch == 50
        assert loader.config.seed == 999
    
    def test_create_validation_loader_respects_parameters(self, mock_generators):
        """Test validation factory respects parameters."""
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
        """Loader works with single generator."""
        generators = [MockGenerator(generator_id=0, class_id=MCAR)]
        config = SyntheticDataLoaderConfig(
            batch_size=4,
            n_range=(20, 50),
            d_range=(5, 10),
            max_rows=64,
            max_cols=16,
            seed=42,
            batches_per_epoch=3,
        )
        
        loader = SyntheticDataLoader(generators, config)
        batch = next(iter(loader))
        
        assert isinstance(batch, TokenBatch)
        assert all(cid == MCAR for cid in batch.class_ids.tolist())
    
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
            batches_per_epoch=2,
        )
        
        loader = SyntheticDataLoader(generators, config)
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
            batches_per_epoch=2,
        )
        
        loader = SyntheticDataLoader(generators, config)
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
            batches_per_epoch=3,
        )
        
        loader = SyntheticDataLoader(generators, config)
        batch = next(iter(loader))
        
        # Generator IDs should be in valid range
        assert (batch.generator_ids >= 0).all()
        assert (batch.generator_ids < 10).all()
    
    def test_empty_iteration_after_epoch(self, mock_generators, small_config):
        """Loader raises StopIteration after epoch completion."""
        loader = SyntheticDataLoader(mock_generators, small_config)
        
        # Exhaust the loader
        _ = list(loader)
        
        # Next iteration should restart
        batch = next(iter(loader))
        assert isinstance(batch, TokenBatch)
    
    def test_all_class_ids_represented(self, mock_generators):
        """Over many batches, all class IDs appear."""
        config = SyntheticDataLoaderConfig(
            batch_size=30,
            n_range=(10, 20),
            d_range=(3, 5),
            max_rows=32,
            max_cols=8,
            seed=42,
            batches_per_epoch=10,
        )
        
        loader = SyntheticDataLoader(mock_generators, config)
        
        seen_classes = set()
        for batch in loader:
            seen_classes.update(batch.class_ids.tolist())
        
        # Should see all three classes
        assert MCAR in seen_classes
        assert MAR in seen_classes
        assert MNAR in seen_classes
    
    def test_large_batch_size(self, mock_generators):
        """Loader handles large batch sizes."""
        config = SyntheticDataLoaderConfig(
            batch_size=100,
            n_range=(10, 20),
            d_range=(3, 5),
            max_rows=32,
            max_cols=8,
            seed=42,
            batches_per_epoch=2,
        )
        
        loader = SyntheticDataLoader(mock_generators, config)
        batch = next(iter(loader))
        
        # Should have 100 datasets in batch
        assert len(batch.generator_ids) == 100
        assert batch.tokens.shape[0] == 100