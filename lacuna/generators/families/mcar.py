"""
lacuna.generators.families.mcar

MCAR (Missing Completely At Random) mechanism implementations.

MCAR: P(R | X) = P(R) - missingness is independent of all data.
"""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR
from ..base import Generator
from ..params import GeneratorParams
from .base_data import sample_gaussian, sample_gaussian_correlated


class MCARUniform(Generator):
    """MCAR with uniform missingness probability.
    
    Each cell is independently missing with probability `miss_rate`.
    
    Required params:
        miss_rate: Probability of missingness per cell.
        base_mean: Mean for Gaussian base data (default: 0.0).
        base_std: Std for Gaussian base data (default: 1.0).
    """
    
    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        super().__init__(generator_id, name, MCAR, params)
        
        # Validate required params
        if "miss_rate" not in params:
            raise ValueError("MCARUniform requires 'miss_rate' parameter")
        
        miss_rate = params["miss_rate"]
        if not (0 < miss_rate < 1):
            raise ValueError(f"miss_rate must be in (0, 1), got {miss_rate}")
    
    def sample(
        self,
        rng: RNGState,
        n: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample base data
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        
        # Sample missingness independently
        miss_rate = self.params["miss_rate"]
        R = rng.rand(n, d) >= miss_rate  # True = observed
        
        # Ensure at least one observed value
        if R.sum() == 0:
            R[0, 0] = True
        
        return X, R


class MCARColumnwise(Generator):
    """MCAR with column-specific missingness rates.
    
    Different columns can have different missingness rates,
    but within each column, missingness is uniform.
    
    Required params:
        miss_rate_range: (low, high) tuple for column miss rates.
        base_mean: Mean for Gaussian base data (default: 0.0).
        base_std: Std for Gaussian base data (default: 1.0).
    """
    
    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        super().__init__(generator_id, name, MCAR, params)
        
        if "miss_rate_range" not in params:
            raise ValueError("MCARColumnwise requires 'miss_rate_range' parameter")
        
        low, high = params["miss_rate_range"]
        if not (0 <= low < high <= 1):
            raise ValueError(f"miss_rate_range must satisfy 0 <= low < high <= 1")
    
    def sample(
        self,
        rng: RNGState,
        n: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        
        # Sample per-column missingness rates
        low, high = self.params["miss_rate_range"]
        col_rates = rng.rand(d) * (high - low) + low
        
        # Apply column-specific rates
        R = torch.ones(n, d, dtype=torch.bool)
        for j in range(d):
            R[:, j] = rng.rand(n) >= col_rates[j]
        
        # Ensure at least one observed
        if R.sum() == 0:
            R[0, 0] = True
        
        return X, R
