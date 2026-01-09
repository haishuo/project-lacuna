"""
lacuna.generators.base

Abstract Generator class and class ID constants.

Design: Generators are immutable factories that produce (X_full, R) pairs.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import ObservedDataset, MCAR, MAR, MNAR
from .params import GeneratorParams


class Generator(ABC):
    """Abstract base for missingness mechanism generators.
    
    A Generator is a frozen specification that can sample datasets
    with a specific missingness mechanism.
    
    Attributes:
        generator_id: Unique integer ID (0 to K-1).
        name: Human-readable name.
        class_id: Mechanism class (MCAR=0, MAR=1, MNAR=2).
        params: Frozen parameter container.
    """
    
    def __init__(
        self,
        generator_id: int,
        name: str,
        class_id: int,
        params: GeneratorParams,
    ):
        if class_id not in (MCAR, MAR, MNAR):
            raise ValueError(f"class_id must be 0, 1, or 2, got {class_id}")
        if generator_id < 0:
            raise ValueError(f"generator_id must be non-negative, got {generator_id}")
        
        self._generator_id = generator_id
        self._name = name
        self._class_id = class_id
        self._params = params
    
    @property
    def generator_id(self) -> int:
        return self._generator_id
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def class_id(self) -> int:
        return self._class_id
    
    @property
    def params(self) -> GeneratorParams:
        return self._params
    
    @abstractmethod
    def sample(
        self,
        rng: RNGState,
        n: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample complete data and missingness indicator.
        
        Args:
            rng: RNG state for reproducibility.
            n: Number of rows (observations).
            d: Number of columns (features).
        
        Returns:
            X_full: [n, d] complete data (no missingness applied).
            R: [n, d] bool tensor, True = observed.
        """
        pass
    
    def sample_observed(
        self,
        rng: RNGState,
        n: int,
        d: int,
        dataset_id: str,
    ) -> ObservedDataset:
        """Sample and construct ObservedDataset.
        
        Convenience method that calls sample() and wraps result.
        
        Args:
            rng: RNG state for reproducibility.
            n: Number of rows.
            d: Number of columns.
            dataset_id: Unique identifier for this dataset.
        
        Returns:
            ObservedDataset with missing values zeroed out.
        """
        X_full, R = self.sample(rng, n, d)
        
        # Zero out missing values
        X_obs = X_full * R.float()
        
        return ObservedDataset(
            x=X_obs,
            r=R,
            n=n,
            d=d,
            feature_names=tuple(f"col_{j}" for j in range(d)),
            dataset_id=dataset_id,
            meta={"generator_id": self.generator_id, "generator_name": self.name},
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.generator_id}, name={self.name!r})"
