"""
lacuna.core.rng

RNG management - no global state.

Design Principles:
- Explicit RNG passing (no global seeds)
- Reproducible by construction
- Thread-safe
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import numpy as np


@dataclass
class RNGState:
    """Encapsulates RNG state for reproducibility.
    
    All randomness in Lacuna flows through RNGState instances.
    No global torch/numpy seeds should be set.
    
    Usage:
        rng = RNGState(seed=42)
        x = rng.randn(100, 10)
        child_rng = rng.spawn()  # Independent stream
    """
    
    seed: int
    _generator: torch.Generator = None
    _np_rng: np.random.Generator = None
    _spawn_counter: int = 0
    
    def __post_init__(self):
        if self._generator is None:
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)
        if self._np_rng is None:
            self._np_rng = np.random.default_rng(self.seed)
    
    def randn(self, *shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Sample from standard normal distribution."""
        return torch.randn(*shape, generator=self._generator, dtype=dtype)
    
    def rand(self, *shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Sample from uniform [0, 1) distribution."""
        return torch.rand(*shape, generator=self._generator, dtype=dtype)
    
    def randint(self, low: int, high: int, shape: Tuple[int, ...]) -> torch.Tensor:
        """Sample integers from [low, high)."""
        return torch.randint(low, high, shape, generator=self._generator)
    
    def choice(self, n: int, size: int, replace: bool = False) -> np.ndarray:
        """Choose indices (numpy-based for compatibility)."""
        return self._np_rng.choice(n, size=size, replace=replace)
    
    def shuffle_indices(self, n: int) -> np.ndarray:
        """Return shuffled indices [0, n)."""
        indices = np.arange(n)
        self._np_rng.shuffle(indices)
        return indices
    
    def spawn(self) -> "RNGState":
        """Create independent child RNG.
        
        Each spawn gets a deterministic but different seed,
        enabling parallel reproducible streams.
        """
        self._spawn_counter += 1
        child_seed = self.seed + self._spawn_counter * 1000003  # Large prime
        return RNGState(seed=child_seed)
    
    def spawn_many(self, n: int) -> list["RNGState"]:
        """Create n independent child RNGs."""
        return [self.spawn() for _ in range(n)]
    
    @property
    def torch_generator(self) -> torch.Generator:
        """Access underlying torch Generator (for library calls)."""
        return self._generator
    
    @property 
    def numpy_rng(self) -> np.random.Generator:
        """Access underlying numpy Generator (for library calls)."""
        return self._np_rng
