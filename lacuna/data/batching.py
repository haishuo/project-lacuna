"""
lacuna.data.batching

Data loaders for synthetic and semi-synthetic training.

Provides:
    - SyntheticDataLoader: Generates infinite stream of synthetic data
    - collate_fn: Collate function for PyTorch DataLoader
    - Utilities for batch generation with artificial masking

Design:
    - Generators produce ObservedDataset with known mechanism
    - Tokenization converts to TokenBatch for model input
    - Artificial masking enables self-supervised pretraining
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Iterator, Tuple, Union
from dataclasses import dataclass

from lacuna.core.types import ObservedDataset, TokenBatch
from lacuna.core.rng import RNGState
from lacuna.data.tokenization import (
    tokenize_and_batch,
    apply_artificial_masking,
    MaskingConfig,
    TOKEN_DIM,
)


# =============================================================================
# Collate Function
# =============================================================================

def collate_fn(batches: List[TokenBatch]) -> TokenBatch:
    """
    Collate multiple TokenBatch objects into a single batch.
    
    For use with PyTorch DataLoader when datasets are pre-tokenized.
    
    Args:
        batches: List of TokenBatch objects (typically batch_size=1 each).
    
    Returns:
        Combined TokenBatch with all samples.
    
    Example:
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
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


# =============================================================================
# Synthetic Data Loader
# =============================================================================

@dataclass
class SyntheticDataLoaderConfig:
    """Configuration for SyntheticDataLoader."""
    
    # Batch settings
    batch_size: int = 32
    
    # Dataset dimensions
    n_range: Tuple[int, int] = (50, 500)   # Row range (min, max)
    d_range: Tuple[int, int] = (5, 20)     # Column range (min, max)
    max_rows: int = 256                     # Max rows for tokenization
    max_cols: int = 32                      # Max cols for tokenization
    
    # Artificial masking for self-supervised
    apply_masking: bool = True
    mask_ratio: float = 0.15
    
    # Iteration
    batches_per_epoch: Optional[int] = None  # None = infinite
    
    # Reproducibility
    seed: Optional[int] = None


class SyntheticDataLoader:
    """
    Data loader that generates synthetic data on-the-fly.
    
    Samples from registered generators to produce infinite streams
    of labeled (X, R, mechanism) training data.
    
    Attributes:
        generators: List of generators to sample from.
        config: SyntheticDataLoaderConfig with settings.
        rng: RNG state for reproducibility.
    
    Example:
        >>> from lacuna.generators import create_default_registry
        >>> registry = create_default_registry()
        >>> loader = SyntheticDataLoader(registry.generators, config)
        >>> for batch in loader:
        ...     output = model(batch)
    """
    
    def __init__(
        self,
        generators: List,  # List of Generator
        config: SyntheticDataLoaderConfig,
        class_mapping: Optional[Dict[int, int]] = None,
    ):
        """
        Initialize synthetic data loader.
        
        Args:
            generators: List of generator instances.
            config: Loader configuration.
            class_mapping: Optional mapping from generator_id to class_id.
        """
        self.generators = generators
        self.config = config
        
        # Build class mapping if not provided
        if class_mapping is None:
            self.class_mapping = {g.generator_id: g.class_id for g in generators}
        else:
            self.class_mapping = class_mapping
        
        # Build variant mapping (for MNAR generators)
        self.variant_mapping = {}
        for g in generators:
            if hasattr(g, 'variant_id'):
                self.variant_mapping[g.generator_id] = g.variant_id
            else:
                self.variant_mapping[g.generator_id] = -1  # No variant
        
        # Initialize RNG
        seed = config.seed if config.seed is not None else 42
        self.rng = RNGState(seed=seed)
        
        # Masking config
        self.masking_config = MaskingConfig(
            mask_ratio=config.mask_ratio,
            mask_observed_only=True,
            min_masked=1,
        )
        
        # Track iteration
        self._batch_count = 0
    
    def __iter__(self) -> Iterator[TokenBatch]:
        """Iterate over batches."""
        self._batch_count = 0
        return self
    
    def __next__(self) -> TokenBatch:
        """Generate next batch."""
        # Check epoch limit
        if (
            self.config.batches_per_epoch is not None
            and self._batch_count >= self.config.batches_per_epoch
        ):
            raise StopIteration
        
        batch = self._generate_batch()
        self._batch_count += 1
        return batch
    
    def __len__(self) -> int:
        """Return batches per epoch (or large number if infinite)."""
        if self.config.batches_per_epoch is not None:
            return self.config.batches_per_epoch
        return 10000  # Arbitrary large number for infinite loader
    
    def _generate_batch(self) -> TokenBatch:
        """Generate a single batch of synthetic data."""
        datasets = []
        generator_ids = []
        variant_ids = []
        artificial_masks = []
        
        for _ in range(self.config.batch_size):
            # Randomly select generator
            gen_idx = self.rng.numpy_rng.integers(0, len(self.generators))
            generator = self.generators[gen_idx]
            
            # Random dataset dimensions
            n = self.rng.numpy_rng.integers(
                self.config.n_range[0],
                self.config.n_range[1] + 1
            )
            d = self.rng.numpy_rng.integers(
                self.config.d_range[0],
                min(self.config.d_range[1] + 1, self.config.max_cols + 1)
            )
            
            # Generate dataset
            dataset = generator.sample_observed(
                rng=self.rng.spawn(),
                n=n,
                d=d,
                dataset_id=f"syn_{self._batch_count}_{len(datasets)}",
            )
            
            # Apply artificial masking for self-supervised learning
            if self.config.apply_masking:
                # Convert tensors to numpy for apply_artificial_masking
                # The function expects numpy arrays with NaN for missing values
                x_np = dataset.x.numpy() if hasattr(dataset.x, 'numpy') else dataset.x
                r_np = dataset.r.numpy() if hasattr(dataset.r, 'numpy') else dataset.r
                
                # apply_artificial_masking expects NaN for missing values
                # but current ObservedDataset uses 0 for missing, so convert
                x_with_nan = x_np.copy()
                x_with_nan[~r_np] = np.nan
                
                x_masked_np, r_masked_np, art_mask = apply_artificial_masking(
                    x_with_nan,
                    r_np,
                    self.masking_config,
                    rng=self.rng.numpy_rng,
                )
                
                # Convert back: replace NaN with 0 for ObservedDataset convention
                x_masked_np = np.nan_to_num(x_masked_np, nan=0.0)
                
                # Convert back to tensors for new ObservedDataset
                x_masked = torch.from_numpy(x_masked_np.astype(np.float32))
                r_masked = torch.from_numpy(r_masked_np.astype(bool))
                
                # Create new dataset with masked values using current API
                dataset = ObservedDataset(
                    x=x_masked,
                    r=r_masked,
                    n=dataset.n,
                    d=dataset.d,
                    feature_names=dataset.feature_names,
                    dataset_id=dataset.dataset_id,
                    meta=dataset.meta,
                )
                artificial_masks.append(art_mask)
            else:
                artificial_masks.append(None)
            
            datasets.append(dataset)
            generator_ids.append(generator.generator_id)
            variant_ids.append(self.variant_mapping.get(generator.generator_id, -1))
        
        # Tokenize and batch
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=self.config.max_rows,
            max_cols=self.config.max_cols,
            generator_ids=generator_ids,
            class_mapping=self.class_mapping,
            variant_ids=variant_ids,
            artificial_masks=artificial_masks if self.config.apply_masking else None,
        )
        
        return batch
    
    def reset(self, seed: Optional[int] = None):
        """
        Reset the data loader.
        
        Args:
            seed: Optional new seed for RNG.
        """
        if seed is not None:
            self.rng = RNGState(seed=seed)
        self._batch_count = 0


# =============================================================================
# Validation Data Loader
# =============================================================================

class ValidationDataLoader:
    """
    Data loader for validation with fixed data.
    
    Unlike SyntheticDataLoader, this generates a fixed set of validation
    samples at initialization for consistent evaluation.
    
    Attributes:
        batches: Pre-generated list of TokenBatch objects.
    """
    
    def __init__(
        self,
        generators: List,
        n_samples: int = 1000,
        batch_size: int = 32,
        n_range: Tuple[int, int] = (50, 500),
        d_range: Tuple[int, int] = (5, 20),
        max_rows: int = 256,
        max_cols: int = 32,
        apply_masking: bool = False,
        seed: int = 12345,
    ):
        """
        Initialize validation data loader.
        
        Args:
            generators: List of generator instances.
            n_samples: Total number of validation samples.
            batch_size: Samples per batch.
            n_range: Row range for datasets.
            d_range: Column range for datasets.
            max_rows: Max rows for tokenization.
            max_cols: Max cols for tokenization.
            apply_masking: Whether to apply artificial masking.
            seed: Random seed for reproducible validation set.
        """
        self.batch_size = batch_size
        
        # Create temporary config
        config = SyntheticDataLoaderConfig(
            batch_size=batch_size,
            n_range=n_range,
            d_range=d_range,
            max_rows=max_rows,
            max_cols=max_cols,
            apply_masking=apply_masking,
            seed=seed,
        )
        
        # Generate all batches
        temp_loader = SyntheticDataLoader(
            generators=generators,
            config=config,
        )
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        self.batches = [temp_loader._generate_batch() for _ in range(n_batches)]
    
    def __iter__(self) -> Iterator[TokenBatch]:
        """Iterate over pre-generated batches."""
        return iter(self.batches)
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.batches)


# =============================================================================
# Mixed Data Loader (Synthetic + Semi-Synthetic)
# =============================================================================

class MixedDataLoader:
    """
    Data loader that mixes synthetic and semi-synthetic data.
    
    Useful for curriculum learning or robustness testing.
    
    Attributes:
        synthetic_loader: SyntheticDataLoader for fully synthetic data.
        semisynthetic_loader: DataLoader for semi-synthetic data.
        synthetic_ratio: Fraction of batches from synthetic source.
    """
    
    def __init__(
        self,
        synthetic_loader: SyntheticDataLoader,
        semisynthetic_loader,  # SemiSyntheticDataLoader or similar
        synthetic_ratio: float = 0.5,
        seed: int = 42,
    ):
        """
        Initialize mixed data loader.
        
        Args:
            synthetic_loader: Loader for synthetic data.
            semisynthetic_loader: Loader for semi-synthetic data.
            synthetic_ratio: Probability of using synthetic loader (0 to 1).
            seed: Random seed for mixing.
        """
        self.synthetic_loader = synthetic_loader
        self.semisynthetic_loader = semisynthetic_loader
        self.synthetic_ratio = synthetic_ratio
        self.rng = np.random.default_rng(seed)
        
        self._synthetic_iter = None
        self._semisynthetic_iter = None
    
    def __iter__(self) -> Iterator[TokenBatch]:
        """Initialize iterators."""
        self._synthetic_iter = iter(self.synthetic_loader)
        self._semisynthetic_iter = iter(self.semisynthetic_loader)
        return self
    
    def __next__(self) -> TokenBatch:
        """Get next batch from either source."""
        use_synthetic = self.rng.random() < self.synthetic_ratio
        
        if use_synthetic:
            try:
                return next(self._synthetic_iter)
            except StopIteration:
                # Fall back to semi-synthetic
                return next(self._semisynthetic_iter)
        else:
            try:
                return next(self._semisynthetic_iter)
            except StopIteration:
                # Fall back to synthetic
                return next(self._synthetic_iter)
    
    def __len__(self) -> int:
        """Approximate length."""
        return len(self.synthetic_loader) + len(self.semisynthetic_loader)


# =============================================================================
# Factory Functions
# =============================================================================

def create_synthetic_loader(
    generators: List,
    batch_size: int = 32,
    n_range: Tuple[int, int] = (50, 500),
    d_range: Tuple[int, int] = (5, 20),
    max_rows: int = 256,
    max_cols: int = 32,
    apply_masking: bool = True,
    mask_ratio: float = 0.15,
    batches_per_epoch: Optional[int] = None,
    seed: Optional[int] = None,
) -> SyntheticDataLoader:
    """
    Factory function to create SyntheticDataLoader.
    
    Args:
        generators: List of generator instances.
        batch_size: Samples per batch.
        n_range: Row range for datasets.
        d_range: Column range for datasets.
        max_rows: Max rows for tokenization.
        max_cols: Max cols for tokenization.
        apply_masking: Whether to apply artificial masking.
        mask_ratio: Fraction of observed values to mask.
        batches_per_epoch: Batches per epoch (None = infinite).
        seed: Random seed.
    
    Returns:
        Configured SyntheticDataLoader.
    """
    config = SyntheticDataLoaderConfig(
        batch_size=batch_size,
        n_range=n_range,
        d_range=d_range,
        max_rows=max_rows,
        max_cols=max_cols,
        apply_masking=apply_masking,
        mask_ratio=mask_ratio,
        batches_per_epoch=batches_per_epoch,
        seed=seed,
    )
    
    return SyntheticDataLoader(generators=generators, config=config)


def create_validation_loader(
    generators: List,
    n_samples: int = 1000,
    batch_size: int = 32,
    max_rows: int = 256,
    max_cols: int = 32,
    seed: int = 12345,
) -> ValidationDataLoader:
    """
    Factory function to create ValidationDataLoader.
    
    Args:
        generators: List of generator instances.
        n_samples: Total validation samples.
        batch_size: Samples per batch.
        max_rows: Max rows for tokenization.
        max_cols: Max cols for tokenization.
        seed: Random seed for reproducibility.
    
    Returns:
        Configured ValidationDataLoader.
    """
    return ValidationDataLoader(
        generators=generators,
        n_samples=n_samples,
        batch_size=batch_size,
        max_rows=max_rows,
        max_cols=max_cols,
        apply_masking=False,  # No masking for validation
        seed=seed,
    )