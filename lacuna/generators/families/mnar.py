"""
lacuna.generators.families.mnar

MNAR (Missing Not At Random) generators.

MNAR mechanisms have missingness that depends on the unobserved value itself.
This is the most problematic case for inference since the mechanism is 
fundamentally unidentifiable from observed data alone.

Generators:
- MNARLogistic: Missingness depends on target column value via logistic model
- MNARSelfCensoring: Missingness depends on value in same column (self-censoring)

CRITICAL: All generators implement apply_to(X, rng) for semi-synthetic data.
"""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MNAR
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from .base_data import sample_gaussian


class MNARLogistic(Generator):
    """MNAR generator using logistic model with target dependence.
    
    Missingness in target column depends on:
    - The target column's own value (beta2 term - the MNAR signature)
    - Optionally, a predictor column value (beta1 term)
    
    P(R_target=0 | X) = sigmoid(beta0 + beta1*X_predictor + beta2*X_target)
    
    The key distinguishing feature from MAR is that beta2 != 0, meaning
    missingness depends on the value that would be missing.
    
    Required params:
        beta0: Intercept (controls baseline missingness rate)
        beta2: Coefficient for target column (must be non-zero for MNAR)
    
    Optional params:
        beta1: Coefficient for predictor column (default 0.0)
        target_col_idx: Column index for missingness target (default -1, last column)
        predictor_col_idx: Column index for predictor (default 0, first column)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """
    
    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        if "beta0" not in params:
            raise ValueError("MNARLogistic requires 'beta0' parameter")
        if "beta2" not in params:
            raise ValueError("MNARLogistic requires 'beta2' parameter")
        
        beta2 = params["beta2"]
        if beta2 == 0.0:
            raise ValueError(
                "beta2 must be non-zero for MNAR (otherwise mechanism is MAR). "
                f"Got beta2={beta2}"
            )
        
        super().__init__(
            generator_id=generator_id,
            name=name,
            class_id=MNAR,
            params=params,
        )
    
    def _compute_missingness(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Compute MNAR missingness mask based on data X.
        
        The key MNAR relationship: missingness depends on X_target (the value
        that will be missing), not just on observed values.
        
        Args:
            X: Complete data tensor [n, d]
            rng: RNG state
            
        Returns:
            R: Boolean mask [n, d], True = observed
        """
        n, d = X.shape
        
        if d < 2:
            raise ValueError("MNARLogistic requires d >= 2")
        
        # Get column indices
        target = self.params.get("target_col_idx", -1)
        predictor = self.params.get("predictor_col_idx", 0)
        
        # Handle negative indices
        if target < 0:
            target = d + target
        if predictor < 0:
            predictor = d + predictor
        
        # Ensure valid and different
        target = target % d
        predictor = predictor % d
        if predictor == target:
            predictor = (target + 1) % d
        
        # Initialize all observed
        R = torch.ones(n, d, dtype=torch.bool)
        
        # Get coefficients
        beta0 = self.params["beta0"]
        beta1 = self.params.get("beta1", 0.0)
        beta2 = self.params["beta2"]
        
        # Compute missingness probability
        # KEY MNAR: includes beta2 * X_target (the value that will be missing)
        logits = beta0 + beta1 * X[:, predictor] + beta2 * X[:, target]
        p_missing = torch.sigmoid(logits)
        
        missing_mask = rng.rand(n) < p_missing
        R[:, target] = ~missing_mask
        
        # Ensure at least one observed
        if R.sum() == 0:
            R[0, 0] = True
        
        return R
    
    def sample(
        self,
        rng: RNGState,
        n: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample synthetic data AND missingness."""
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())
        
        return X, R
    
    def apply_to(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Apply MNAR missingness to existing data.
        
        For MNAR, missingness depends on the actual data values (including
        the value that will be missing), so apply_to() uses the provided X.
        """
        return self._compute_missingness(X, rng)


class MNARSelfCensoring(Generator):
    """MNAR Self-Censoring generator.
    
    Each column's missingness depends on its own value. This is a 
    "pure MNAR" pattern where missingness is entirely driven by
    the unobserved values themselves.
    
    For each column j:
        P(R_j=0 | X_j) = sigmoid(beta0 + beta1 * X_j)
    
    This models scenarios like:
    - High income individuals refusing to report income
    - Extreme health values causing patient dropout
    - Outliers being flagged and removed
    
    Required params:
        beta0: Intercept (controls baseline missingness rate)
        beta1: Self-censoring strength (must be non-zero)
    
    Optional params:
        affected_frac: Fraction of columns with self-censoring (default 0.5)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """
    
    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        if "beta0" not in params:
            raise ValueError("MNARSelfCensoring requires 'beta0' parameter")
        if "beta1" not in params:
            raise ValueError("MNARSelfCensoring requires 'beta1' parameter")
        
        beta1 = params["beta1"]
        if beta1 == 0.0:
            raise ValueError(
                "beta1 must be non-zero for self-censoring MNAR. "
                f"Got beta1={beta1}"
            )
        
        super().__init__(
            generator_id=generator_id,
            name=name,
            class_id=MNAR,
            params=params,
        )
    
    def _compute_missingness(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Compute self-censoring MNAR missingness mask.
        
        For each affected column, missingness probability depends on
        that column's own value.
        
        Args:
            X: Complete data tensor [n, d]
            rng: RNG state
            
        Returns:
            R: Boolean mask [n, d], True = observed
        """
        n, d = X.shape
        
        beta0 = self.params["beta0"]
        beta1 = self.params["beta1"]
        affected_frac = self.params.get("affected_frac", 0.5)
        
        # Determine which columns are affected
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)
        
        # Initialize all observed
        R = torch.ones(n, d, dtype=torch.bool)
        
        # Apply self-censoring to affected columns
        for col in affected_cols:
            # Missingness depends on this column's own value
            logits = beta0 + beta1 * X[:, col]
            p_missing = torch.sigmoid(logits)
            
            missing_mask = rng.rand(n) < p_missing
            R[:, col] = ~missing_mask
        
        # Ensure at least one observed
        if R.sum() == 0:
            R[0, 0] = True
        
        return R
    
    def sample(
        self,
        rng: RNGState,
        n: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample synthetic data AND missingness."""
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())
        
        return X, R
    
    def apply_to(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Apply self-censoring MNAR missingness to existing data."""
        return self._compute_missingness(X, rng)