"""
lacuna.generators.families.mar

MAR (Missing At Random) mechanism implementations.

MAR: P(R | X_obs, X_mis) = P(R | X_obs) - missingness depends only on observed data.

CRITICAL FIX (2026-01-10):
--------------------------
1. Added apply_to(X, rng) method for semi-synthetic data generation.
2. Added MARMultiColumn generator that affects MULTIPLE target columns,
   creating a stronger MAR signal for the model to learn from.

The original MARLogistic only made ONE column missing based on another column.
This creates a very weak signal when averaged across all columns in the dataset.
MARMultiColumn applies the MAR pattern to multiple columns, making the
cross-column dependency pattern much more detectable.
"""

from typing import Tuple, List, Optional
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR
from ..base import Generator
from ..params import GeneratorParams
from .base_data import sample_gaussian, sample_gaussian_correlated


class MARLogistic(Generator):
    """MAR mechanism via logistic model (single target column).
    
    Missingness in target column depends on predictor column:
    P(R_target = 0 | X) = sigmoid(alpha0 + alpha1 * X_predictor)
    
    Required params:
        alpha0: Intercept (controls baseline missingness).
        alpha1: Slope (controls dependence strength).
        
    Optional params:
        target_col_idx: Index of target column (default: -1, last column)
        predictor_col_idx: Index of predictor column (default: 0)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
        correlation: Correlation between columns for synthetic data (default: 0.0)
    """
    
    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        super().__init__(generator_id, name, MAR, params)
        
        required = ["alpha0", "alpha1"]
        for key in required:
            if key not in params:
                raise ValueError(f"MARLogistic requires '{key}' parameter")
    
    def _compute_missingness(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Compute missingness mask R based on data X."""
        n, d = X.shape
        
        if d < 2:
            raise ValueError("MARLogistic requires d >= 2")
        
        # Determine target and predictor columns
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
        
        # Apply MAR mechanism
        alpha0 = self.params["alpha0"]
        alpha1 = self.params["alpha1"]
        
        logits = alpha0 + alpha1 * X[:, predictor]
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
        rho = self.params.get("correlation", 0.0)
        
        if rho > 0:
            X = sample_gaussian_correlated(rng.spawn(), n, d, mean=mean, std=std, rho=rho)
        else:
            X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        
        R = self._compute_missingness(X, rng.spawn())
        
        return X, R
    
    def apply_to(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Apply missingness mechanism to existing data."""
        return self._compute_missingness(X, rng)


class MARMultiColumn(Generator):
    """MAR mechanism affecting MULTIPLE target columns.
    
    This creates a much stronger MAR signal than single-column MAR.
    Multiple columns have missingness that depends on a predictor column,
    making the cross-column dependency pattern more detectable.
    
    For each target column j in targets:
        P(R_j = 0 | X) = sigmoid(alpha0 + alpha1 * X_predictor)
    
    The key insight: when multiple columns share the same predictor-based
    missingness pattern, the MARHead's cross-attention can learn this
    dependency more effectively.
    
    Required params:
        alpha0: Intercept (controls baseline missingness)
        alpha1: Slope (controls dependence strength)
        
    Optional params:
        n_targets: Number of target columns to affect (default: 3)
        target_frac: Alternative: fraction of columns to affect (default: None)
        predictor_col_idx: Index of predictor column (default: 0)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """
    
    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        super().__init__(generator_id, name, MAR, params)
        
        required = ["alpha0", "alpha1"]
        for key in required:
            if key not in params:
                raise ValueError(f"MARMultiColumn requires '{key}' parameter")
    
    def _compute_missingness(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Compute missingness mask R based on data X."""
        n, d = X.shape
        
        if d < 2:
            raise ValueError("MARMultiColumn requires d >= 2")
        
        # Determine predictor column (this one stays fully observed)
        predictor = self.params.get("predictor_col_idx", 0)
        if predictor < 0:
            predictor = d + predictor
        predictor = predictor % d
        
        # Determine number of target columns
        n_targets = self.params.get("n_targets", None)
        target_frac = self.params.get("target_frac", None)
        
        if n_targets is None and target_frac is None:
            # Default: affect ~30% of columns or at least 2
            n_targets = max(2, int(d * 0.3))
        elif target_frac is not None:
            n_targets = max(1, int(d * target_frac))
        
        # Cap at d-1 (need at least predictor to stay observed)
        n_targets = min(n_targets, d - 1)
        
        # Select target columns (exclude predictor)
        available_cols = [j for j in range(d) if j != predictor]
        target_indices = rng.choice(len(available_cols), size=n_targets, replace=False)
        targets = [available_cols[i] for i in target_indices]
        
        # Initialize all observed
        R = torch.ones(n, d, dtype=torch.bool)
        
        # Get MAR parameters
        alpha0 = self.params["alpha0"]
        alpha1 = self.params["alpha1"]
        
        # Compute missingness probability based on predictor
        # Same predictor drives missingness in ALL target columns
        logits = alpha0 + alpha1 * X[:, predictor]
        p_missing = torch.sigmoid(logits)
        
        # Apply to each target column with independent randomness
        # (same probability, but independent coin flips)
        for target in targets:
            missing_mask = rng.rand(n) < p_missing
            R[:, target] = ~missing_mask
        
        # Ensure at least one observed value per column
        for col in range(d):
            if R[:, col].sum() == 0:
                rand_row = rng.randint(0, n, (1,)).item()
                R[rand_row, col] = True
        
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
        """Apply missingness mechanism to existing data."""
        return self._compute_missingness(X, rng)


class MARMultiPredictor(Generator):
    """MAR mechanism with multiple predictors for a single target.
    
    Missingness depends on multiple observed columns:
    P(R_target = 0 | X) = sigmoid(alpha0 + sum_k alpha_k * X_k)
    
    Required params:
        alpha0: Intercept
        alphas: List of coefficients for predictor columns
        
    Optional params:
        target_col_idx: Index of target column (default: -1)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """
    
    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        super().__init__(generator_id, name, MAR, params)
        
        required = ["alpha0", "alphas"]
        for key in required:
            if key not in params:
                raise ValueError(f"MARMultiPredictor requires '{key}' parameter")
    
    def _compute_missingness(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Compute missingness mask R based on data X."""
        n, d = X.shape
        alphas = self.params["alphas"]
        n_predictors = len(alphas)
        
        if d < n_predictors + 1:
            raise ValueError(
                f"MARMultiPredictor with {n_predictors} predictors "
                f"requires d >= {n_predictors + 1}"
            )
        
        # Target column (default: last)
        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d
        
        # Predictor columns: all columns except target, take first n_predictors
        predictor_cols = [j for j in range(d) if j != target][:n_predictors]
        
        # Initialize all observed
        R = torch.ones(n, d, dtype=torch.bool)
        
        # Compute logits as linear combination of predictor columns
        alpha0 = self.params["alpha0"]
        logits = torch.full((n,), alpha0)
        
        for k, alpha_k in enumerate(alphas):
            if k < len(predictor_cols):
                logits = logits + alpha_k * X[:, predictor_cols[k]]
        
        p_missing = torch.sigmoid(logits)
        missing_mask = rng.rand(n) < p_missing
        R[:, target] = ~missing_mask
        
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
        """Apply missingness mechanism to existing data."""
        return self._compute_missingness(X, rng)