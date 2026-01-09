"""
lacuna.data.batching

Batch construction for row-level tokenization.

Design:
- Each dataset becomes [n, d, TOKEN_DIM] tokens
- Batching pads both rows (n) and columns (d)
- Row sampling for large datasets
"""

import torch
from typing import List, Optional, Tuple

from lacuna.core.types import ObservedDataset, TokenBatch
from lacuna.core.rng import RNGState
from .tokenization import tokenize_dataset, TOKEN_DIM


def tokenize_and_batch(
    datasets: List[ObservedDataset],
    max_rows: int,
    max_cols: int,
    generator_ids: Optional[List[int]] = None,
    class_mapping: Optional[torch.Tensor] = None,
    row_sample_seed: Optional[int] = None,
) -> TokenBatch:
    """Tokenize and batch multiple datasets.
    
    Args:
        datasets: List of ObservedDataset objects.
        max_rows: Maximum rows per dataset (sample if larger).
        max_cols: Maximum columns (pad/truncate).
        generator_ids: Optional generator labels.
        class_mapping: [K] tensor mapping generator_id -> class_id.
        row_sample_seed: Seed for row sampling (for reproducibility).
    
    Returns:
        TokenBatch with shape [B, max_rows, max_cols, TOKEN_DIM].
    """
    B = len(datasets)
    
    # Initialize padded tensors
    tokens = torch.zeros(B, max_rows, max_cols, TOKEN_DIM)
    row_mask = torch.zeros(B, max_rows, dtype=torch.bool)
    col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
    
    rng = RNGState(seed=row_sample_seed) if row_sample_seed else None
    
    for i, ds in enumerate(datasets):
        # Tokenize dataset
        ds_tokens = tokenize_dataset(ds, normalize=True)  # [n, d, TOKEN_DIM]
        n, d = ds.n, ds.d
        
        # Sample rows if too many
        if n > max_rows:
            if rng:
                indices = rng.choice(n, size=max_rows, replace=False)
                indices = torch.from_numpy(indices).sort()[0]
            else:
                indices = torch.arange(max_rows)
            ds_tokens = ds_tokens[indices]
            n = max_rows
        
        # Truncate columns if needed
        d_use = min(d, max_cols)
        
        # Fill in batch tensors
        tokens[i, :n, :d_use, :] = ds_tokens[:n, :d_use, :]
        row_mask[i, :n] = True
        col_mask[i, :d_use] = True
    
    # Process labels
    gen_ids = None
    cls_ids = None
    
    if generator_ids is not None:
        gen_ids = torch.tensor(generator_ids, dtype=torch.long)
        
        if class_mapping is not None:
            cls_ids = class_mapping[gen_ids]
    
    return TokenBatch(
        tokens=tokens,
        row_mask=row_mask,
        col_mask=col_mask,
        generator_ids=gen_ids,
        class_ids=cls_ids,
    )


class SyntheticDataLoader:
    """Data loader that generates synthetic data on-the-fly."""
    
    def __init__(
        self,
        registry,  # GeneratorRegistry
        prior,     # GeneratorPrior
        n_range: Tuple[int, int],
        d_range: Tuple[int, int],
        max_rows: int,
        max_cols: int,
        batch_size: int,
        batches_per_epoch: int,
        seed: int = 42,
    ):
        self.registry = registry
        self.prior = prior
        self.n_range = n_range
        self.d_range = d_range
        self.max_rows = max_rows
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
                max_rows=self.max_rows,
                max_cols=self.max_cols,
                generator_ids=gen_ids.tolist(),
                class_mapping=self._class_mapping,
                row_sample_seed=batch_rng.seed,
            )
            
            yield batch


def collate_fn(
    batch: List[Tuple[ObservedDataset, int]],
    max_rows: int,
    max_cols: int,
    class_mapping: Optional[torch.Tensor] = None,
) -> TokenBatch:
    """Collate function for DataLoader."""
    datasets = [item[0] for item in batch]
    generator_ids = [item[1] for item in batch]
    
    return tokenize_and_batch(
        datasets=datasets,
        max_rows=max_rows,
        max_cols=max_cols,
        generator_ids=generator_ids,
        class_mapping=class_mapping,
    )
