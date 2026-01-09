"""
lacuna.data.batching

Batch construction and collation.

Design: Pad variable-width datasets to fixed max_cols.
"""

import torch
from typing import List, Optional, Tuple

from lacuna.core.types import ObservedDataset, TokenBatch
from lacuna.core.rng import RNGState
from .tokenization import tokenize_dataset, get_token_dim
from .normalization import NormalizationStats


def tokenize_and_batch(
    datasets: List[ObservedDataset],
    max_cols: int,
    generator_ids: Optional[List[int]] = None,
    class_mapping: Optional[torch.Tensor] = None,
    normalize: bool = True,
    stats: Optional[NormalizationStats] = None,
) -> TokenBatch:
    """Tokenize and batch multiple datasets.
    
    Args:
        datasets: List of ObservedDataset objects.
        max_cols: Pad/truncate to this many columns.
        generator_ids: Optional generator labels for each dataset.
        class_mapping: [K] tensor mapping generator_id -> class_id.
        normalize: Whether to normalize data.
        stats: Shared normalization stats (optional).
    
    Returns:
        TokenBatch ready for model input.
    """
    B = len(datasets)
    q = get_token_dim()
    
    # Initialize padded tensors
    tokens = torch.zeros(B, max_cols, q)
    col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
    
    for i, ds in enumerate(datasets):
        ds_tokens = tokenize_dataset(ds, normalize=normalize, stats=stats)
        d = min(ds.d, max_cols)
        
        tokens[i, :d, :] = ds_tokens[:d, :]
        col_mask[i, :d] = True
    
    # Process labels
    gen_ids = None
    cls_ids = None
    
    if generator_ids is not None:
        gen_ids = torch.tensor(generator_ids, dtype=torch.long)
        
        if class_mapping is not None:
            cls_ids = class_mapping[gen_ids]
    
    return TokenBatch(
        tokens=tokens,
        col_mask=col_mask,
        generator_ids=gen_ids,
        class_ids=cls_ids,
    )


def collate_fn(
    batch: List[Tuple[ObservedDataset, int]],
    max_cols: int,
    class_mapping: Optional[torch.Tensor] = None,
) -> TokenBatch:
    """Collate function for DataLoader.
    
    Args:
        batch: List of (dataset, generator_id) tuples.
        max_cols: Maximum columns.
        class_mapping: Generator to class mapping.
    
    Returns:
        TokenBatch.
    """
    datasets = [item[0] for item in batch]
    generator_ids = [item[1] for item in batch]
    
    return tokenize_and_batch(
        datasets=datasets,
        max_cols=max_cols,
        generator_ids=generator_ids,
        class_mapping=class_mapping,
    )


class SyntheticDataLoader:
    """Data loader that generates synthetic data on-the-fly.
    
    More efficient than pre-generating and storing datasets.
    """
    
    def __init__(
        self,
        registry,  # GeneratorRegistry
        prior,     # GeneratorPrior
        n_range: Tuple[int, int],
        d_range: Tuple[int, int],
        max_cols: int,
        batch_size: int,
        batches_per_epoch: int,
        seed: int = 42,
    ):
        self.registry = registry
        self.prior = prior
        self.n_range = n_range
        self.d_range = d_range
        self.max_cols = max_cols
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        
        self._class_mapping = registry.get_class_mapping()
    
    def __len__(self) -> int:
        return self.batches_per_epoch
    
    def __iter__(self):
        rng = RNGState(seed=self.seed)
        
        for batch_idx in range(self.batches_per_epoch):
            batch_rng = rng.spawn()
            
            # Sample generator IDs
            gen_ids = self.prior.sample_batch(batch_rng.spawn(), self.batch_size)
            
            # Generate datasets
            datasets = []
            for i in range(self.batch_size):
                gen_id = gen_ids[i].item()
                generator = self.registry[gen_id]
                
                # Random n and d
                n = batch_rng.randint(self.n_range[0], self.n_range[1] + 1, (1,)).item()
                d = batch_rng.randint(self.d_range[0], self.d_range[1] + 1, (1,)).item()
                
                ds = generator.sample_observed(
                    rng=batch_rng.spawn(),
                    n=n,
                    d=d,
                    dataset_id=f"batch{batch_idx}_item{i}",
                )
                datasets.append(ds)
            
            # Tokenize and batch
            batch = tokenize_and_batch(
                datasets=datasets,
                max_cols=self.max_cols,
                generator_ids=gen_ids.tolist(),
                class_mapping=self._class_mapping,
            )
            
            yield batch
    
    def get_epoch_iterator(self, epoch: int):
        """Get iterator with epoch-specific seed."""
        original_seed = self.seed
        self.seed = original_seed + epoch * 1000000
        iterator = iter(self)
        self.seed = original_seed
        return iterator
