"""
lacuna.generators.priors

Prior distribution over generators.

Design: Priors are probability distributions over generator IDs.
"""

import torch
from typing import Optional

from lacuna.core.rng import RNGState
from lacuna.core.validation import validate_probabilities_sum_to_one
from .registry import GeneratorRegistry


class GeneratorPrior:
    """Prior distribution over generators.
    
    Supports:
    - Uniform prior (default)
    - Class-balanced prior (equal weight per class)
    - Custom weights
    
    Usage:
        prior = GeneratorPrior.uniform(registry)
        gen_id = prior.sample(rng)
        
        prior = GeneratorPrior.class_balanced(registry)
        gen_ids = prior.sample_batch(rng, n=64)
    """
    
    def __init__(self, registry: GeneratorRegistry, weights: torch.Tensor):
        """Create prior with explicit weights.
        
        Args:
            registry: Generator registry.
            weights: [K] tensor of probabilities (must sum to 1).
        """
        if weights.shape != (registry.K,):
            raise ValueError(f"weights shape {weights.shape} != ({registry.K},)")
        
        validate_probabilities_sum_to_one(weights.unsqueeze(0))
        
        self._registry = registry
        self._weights = weights
        self._cumsum = weights.cumsum(dim=0)
    
    @classmethod
    def uniform(cls, registry: GeneratorRegistry) -> "GeneratorPrior":
        """Create uniform prior over all generators."""
        weights = torch.ones(registry.K) / registry.K
        return cls(registry, weights)
    
    @classmethod
    def class_balanced(cls, registry: GeneratorRegistry) -> "GeneratorPrior":
        """Create class-balanced prior.
        
        Each class gets equal total weight, distributed uniformly
        among generators in that class.
        """
        K = registry.K
        weights = torch.zeros(K)
        
        class_counts = registry.class_counts()
        n_classes = sum(1 for c in class_counts.values() if c > 0)
        
        for class_id, count in class_counts.items():
            if count > 0:
                class_weight = 1.0 / n_classes
                per_gen_weight = class_weight / count
                for gen_id in registry.generator_ids_for_class(class_id):
                    weights[gen_id] = per_gen_weight
        
        return cls(registry, weights)
    
    @classmethod
    def custom(
        cls,
        registry: GeneratorRegistry,
        weights: torch.Tensor,
    ) -> "GeneratorPrior":
        """Create prior with custom weights."""
        return cls(registry, weights)
    
    @property
    def weights(self) -> torch.Tensor:
        return self._weights
    
    @property
    def registry(self) -> GeneratorRegistry:
        return self._registry
    
    def sample(self, rng: RNGState) -> int:
        """Sample single generator ID."""
        u = rng.rand(1).item()
        idx = torch.searchsorted(self._cumsum, u).item()
        return min(idx, self._registry.K - 1)
    
    def sample_batch(self, rng: RNGState, n: int) -> torch.Tensor:
        """Sample batch of generator IDs.
        
        Args:
            rng: RNG state.
            n: Batch size.
        
        Returns:
            [n] long tensor of generator IDs.
        """
        u = rng.rand(n)
        indices = torch.searchsorted(self._cumsum, u)
        return indices.clamp(max=self._registry.K - 1).long()
    
    def __repr__(self) -> str:
        return f"GeneratorPrior(K={self._registry.K})"
