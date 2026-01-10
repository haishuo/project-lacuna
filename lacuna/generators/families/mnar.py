"""
lacuna.generators.families.mnar

MNAR (Missing Not At Random) generators.

MNAR mechanisms have missingness that depends on the unobserved value itself.
This is the most problematic case for inference since the mechanism is 
fundamentally unidentifiable from observed data alone.

Generators:
- MNARLogistic: Missingness depends on target column value via logistic model
- MNARSelfCensoring: Missingness depends on value in same column (self-censoring)
"""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MNAR
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from .base_data import sample_gaussian


# =============================================================================
# MNARLogistic
# =============================================================================

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
    """
    
    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        # Validate required parameters
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
            X_full: [n, d] complete data tensor.
            R: [n, d] bool tensor, True = observed.
        """
        if d < 2:
            raise ValueError(f"MNARLogistic requires d >= 2, got d={d}")
        
        # Generate complete data
        X = sample_gaussian(rng, n, d)
        
        # Get parameters
        beta0 = self.params["beta0"]
        beta1 = self.params.get("beta1", 0.0)
        beta2 = self.params["beta2"]
        target_idx = self.params.get("target_col_idx", -1)
        predictor_idx = self.params.get("predictor_col_idx", 0)
        
        # Handle negative indices
        if target_idx < 0:
            target_idx = d + target_idx
        if predictor_idx < 0:
            predictor_idx = d + predictor_idx
        
        # Compute logit for missingness
        # MNAR signature: depends on X[:, target_idx] (the value that will be missing)
        logit = beta0 + beta1 * X[:, predictor_idx] + beta2 * X[:, target_idx]
        
        # Convert to probability via sigmoid
        p_missing = torch.sigmoid(logit)
        
        # Sample missingness
        R = torch.ones(n, d, dtype=torch.bool)
        missing = rng.rand(n) < p_missing
        R[:, target_idx] = ~missing
        
        # Ensure at least one observation per column
        if R[:, target_idx].sum() == 0:
            R[0, target_idx] = True
        
        return X, R


# =============================================================================
# MNARSelfCensoring
# =============================================================================

class MNARSelfCensoring(Generator):
    """MNAR generator with self-censoring mechanism.
    
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
    """
    
    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        # Validate required parameters
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
            X_full: [n, d] complete data tensor.
            R: [n, d] bool tensor, True = observed.
        """
        # Generate complete data
        X = sample_gaussian(rng, n, d)
        
        # Get parameters
        beta0 = self.params["beta0"]
        beta1 = self.params["beta1"]
        affected_frac = self.params.get("affected_frac", 0.5)
        
        # Determine which columns are affected
        n_affected = max(1, int(d * affected_frac))
        
        # Use RNG to select affected columns (reproducibly)
        col_probs = rng.rand(d)
        affected_cols = torch.argsort(col_probs)[:n_affected]
        
        # Initialize R (all observed)
        R = torch.ones(n, d, dtype=torch.bool)
        
        # Apply self-censoring to affected columns
        for j in affected_cols:
            # Self-censoring: missingness depends on own value
            logit = beta0 + beta1 * X[:, j]
            p_missing = torch.sigmoid(logit)
            
            # Sample missingness
            missing = rng.rand(n) < p_missing
            R[:, j] = ~missing
            
            # Ensure at least one observation
            if R[:, j].sum() == 0:
                R[0, j] = True
        
        return X, R