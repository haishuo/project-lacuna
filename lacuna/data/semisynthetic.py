"""
lacuna.data.semisynthetic

Apply synthetic missingness mechanisms to complete (real) data.

This creates semi-synthetic datasets where:
- X (data values) come from real datasets
- R (missingness pattern) is synthetically generated with known mechanism
"""

import torch
from typing import Tuple, Optional, List
from dataclasses import dataclass

from lacuna.core.types import ObservedDataset, MCAR, MAR, MNAR
from lacuna.core.rng import RNGState
from lacuna.generators.base import Generator
from lacuna.generators.registry import GeneratorRegistry
from lacuna.generators.priors import GeneratorPrior
from .ingestion import RawDataset
from .observed import create_observed_dataset


@dataclass
class SemiSyntheticDataset:
    """A real dataset with synthetic missingness applied.
    
    Tracks the source data and the mechanism used.
    """
    observed: ObservedDataset      # The dataset with missingness
    complete: torch.Tensor         # Original complete data [n, d]
    generator_id: int              # Which generator was used
    generator_name: str
    class_id: int                  # MCAR/MAR/MNAR
    source_name: str               # Original dataset name


def apply_missingness(
    raw: RawDataset,
    generator: Generator,
    rng: RNGState,
    dataset_id: Optional[str] = None,
) -> SemiSyntheticDataset:
    """Apply a missingness mechanism to complete data.
    
    Args:
        raw: Complete dataset (no missing values).
        generator: Missingness generator to apply.
        rng: RNG state for reproducibility.
        dataset_id: ID for the resulting dataset.
    
    Returns:
        SemiSyntheticDataset with known mechanism.
    """
    n, d = raw.n, raw.d
    
    # Get the complete data as tensor
    X_complete = torch.from_numpy(raw.data.astype('float32'))
    
    # Sample missingness pattern using generator
    # Generator.sample() returns (X_generated, R)
    # We only use R and apply it to our real X
    _, R = generator.sample(rng, n, d)
    
    # Ensure at least one observed value
    if R.sum() == 0:
        R[0, 0] = True
    
    # Zero out missing values
    X_observed = X_complete * R.float()
    
    observed_ds = ObservedDataset(
        x=X_observed,
        r=R,
        n=n,
        d=d,
        feature_names=raw.feature_names,
        dataset_id=dataset_id or f"{raw.name}_{generator.name}",
        meta={
            "source": raw.source,
            "generator_id": generator.generator_id,
            "generator_name": generator.name,
            "class_id": generator.class_id,
            "is_semisynthetic": True,
        },
    )
    
    return SemiSyntheticDataset(
        observed=observed_ds,
        complete=X_complete,
        generator_id=generator.generator_id,
        generator_name=generator.name,
        class_id=generator.class_id,
        source_name=raw.name,
    )


def generate_semisynthetic_batch(
    raw_datasets: List[RawDataset],
    registry: GeneratorRegistry,
    prior: GeneratorPrior,
    rng: RNGState,
    samples_per_dataset: int = 1,
) -> List[SemiSyntheticDataset]:
    """Generate batch of semi-synthetic datasets.
    
    For each raw dataset, applies random missingness mechanisms.
    
    Args:
        raw_datasets: List of complete datasets.
        registry: Generator registry.
        prior: Prior over generators.
        rng: RNG state.
        samples_per_dataset: How many different missingness patterns per dataset.
    
    Returns:
        List of SemiSyntheticDataset objects.
    """
    results = []
    
    for raw in raw_datasets:
        for i in range(samples_per_dataset):
            # Sample a generator
            gen_id = prior.sample(rng.spawn())
            generator = registry[gen_id]
            
            # Apply missingness
            ss_dataset = apply_missingness(
                raw=raw,
                generator=generator,
                rng=rng.spawn(),
                dataset_id=f"{raw.name}_gen{gen_id}_sample{i}",
            )
            results.append(ss_dataset)
    
    return results


class SemiSyntheticDataLoader:
    """Data loader for semi-synthetic data.
    
    Takes a pool of real datasets and generates training batches
    by applying random missingness mechanisms.
    """
    
    def __init__(
        self,
        raw_datasets: List[RawDataset],
        registry: GeneratorRegistry,
        prior: GeneratorPrior,
        max_cols: int,
        batch_size: int,
        batches_per_epoch: int,
        seed: int = 42,
    ):
        self.raw_datasets = raw_datasets
        self.registry = registry
        self.prior = prior
        self.max_cols = max_cols
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        
        self._class_mapping = registry.get_class_mapping()
        
        if len(raw_datasets) == 0:
            raise ValueError("Need at least one raw dataset")
    
    def __len__(self) -> int:
        return self.batches_per_epoch
    
    def __iter__(self):
        from .batching import tokenize_and_batch
        
        rng = RNGState(seed=self.seed)
        n_datasets = len(self.raw_datasets)
        
        for batch_idx in range(self.batches_per_epoch):
            batch_rng = rng.spawn()
            
            datasets = []
            generator_ids = []
            
            for i in range(self.batch_size):
                # Pick a random raw dataset
                ds_idx = batch_rng.randint(0, n_datasets, (1,)).item()
                raw = self.raw_datasets[ds_idx]
                
                # Sample a generator
                gen_id = self.prior.sample(batch_rng.spawn())
                generator = self.registry[gen_id]
                
                # Apply missingness
                ss = apply_missingness(
                    raw=raw,
                    generator=generator,
                    rng=batch_rng.spawn(),
                    dataset_id=f"batch{batch_idx}_item{i}",
                )
                
                datasets.append(ss.observed)
                generator_ids.append(gen_id)
            
            # Tokenize and batch
            batch = tokenize_and_batch(
                datasets=datasets,
                max_cols=self.max_cols,
                generator_ids=generator_ids,
                class_mapping=self._class_mapping,
            )
            
            yield batch
