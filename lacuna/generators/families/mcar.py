"""
lacuna.generators.families.mcar

MCAR (Missing Completely At Random) mechanism implementations.

MCAR: P(R | X) = P(R) - missingness is independent of data values.

For MCAR, apply_to() is straightforward since missingness doesn't depend
on data values. We just generate random missingness patterns based on
the data dimensions.
"""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR
from ..base import Generator
from ..params import GeneratorParams
from .base_data import sample_gaussian


class MCARUniform(Generator):
    """MCAR mechanism with uniform missingness rate across all cells.
    
    Each cell is independently missing with probability miss_rate.
    
    Required params:
        miss_rate: Probability of missingness per cell (0 to 1).
        
    Optional params:
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
        
        if "miss_rate" not in params:
            raise ValueError("MCARUniform requires 'miss_rate' parameter")
        
        miss_rate = params["miss_rate"]
        if not 0 <= miss_rate <= 1:
            raise ValueError(f"miss_rate must be in [0, 1], got {miss_rate}")
    
    def _compute_missingness(
        self,
        n: int,
        d: int,
        rng: RNGState,
    ) -> torch.Tensor:
        """Compute MCAR missingness mask.
        
        For MCAR, missingness doesn't depend on data values,
        so we just need dimensions n and d.
        
        Args:
            n: Number of rows
            d: Number of columns
            rng: RNG state
            
        Returns:
            R: Boolean mask [n, d], True = observed
        """
        miss_rate = self.params["miss_rate"]
        
        # Each cell is independently missing with probability miss_rate
        R = rng.rand(n, d) >= miss_rate
        
        # Ensure at least one observed value
        if R.sum() == 0:
            R[0, 0] = True
        
        return R
    
    def sample(
        self,
        rng: RNGState,
        n: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample synthetic data AND missingness.
        
        Args:
            rng: RNG state for reproducibility
            n: Number of rows
            d: Number of columns
            
        Returns:
            X: Complete synthetic data [n, d]
            R: Missingness mask [n, d], True = observed
        """
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(n, d, rng.spawn())
        
        return X, R
    
    def apply_to(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Apply MCAR missingness to existing data.
        
        For MCAR, missingness is independent of data values,
        so we just generate random missingness based on dimensions.
        
        Args:
            X: Complete data tensor [n, d]
            rng: RNG state for reproducibility
            
        Returns:
            R: Missingness mask [n, d], True = observed
        """
        n, d = X.shape
        return self._compute_missingness(n, d, rng)


class MCARColumnwise(Generator):
    """MCAR mechanism with different missingness rates per column.
    
    Each column gets a random missingness rate drawn from a range.
    Within each column, cells are independently missing.
    
    Required params:
        miss_rate_range: Tuple (min_rate, max_rate) for column missingness.
        
    Optional params:
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
        
        rate_range = params["miss_rate_range"]
        if len(rate_range) != 2:
            raise ValueError("miss_rate_range must be (min, max) tuple")
        if not (0 <= rate_range[0] <= rate_range[1] <= 1):
            raise ValueError("miss_rate_range must satisfy 0 <= min <= max <= 1")
    
    def _compute_missingness(
        self,
        n: int,
        d: int,
        rng: RNGState,
    ) -> torch.Tensor:
        """Compute column-wise MCAR missingness mask.
        
        Args:
            n: Number of rows
            d: Number of columns
            rng: RNG state
            
        Returns:
            R: Boolean mask [n, d], True = observed
        """
        min_rate, max_rate = self.params["miss_rate_range"]
        
        # Sample a missingness rate for each column
        col_rates = rng.rand(d) * (max_rate - min_rate) + min_rate
        
        # Generate missingness per column
        R = torch.ones(n, d, dtype=torch.bool)
        for j in range(d):
            R[:, j] = rng.rand(n) >= col_rates[j]
        
        # Ensure at least one observed value
        if R.sum() == 0:
            R[0, 0] = True
        
        return R
    
    def sample(
        self,
        rng: RNGState,
        n: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample synthetic data AND missingness.
        
        Args:
            rng: RNG state for reproducibility
            n: Number of rows
            d: Number of columns
            
        Returns:
            X: Complete synthetic data [n, d]
            R: Missingness mask [n, d], True = observed
        """
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(n, d, rng.spawn())
        
        return X, R
    
    def apply_to(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Apply column-wise MCAR missingness to existing data.
        
        Args:
            X: Complete data tensor [n, d]
            rng: RNG state for reproducibility
            
        Returns:
            R: Missingness mask [n, d], True = observed
        """
        n, d = X.shape
        return self._compute_missingness(n, d, rng)