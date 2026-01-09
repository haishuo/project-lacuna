"""
lacuna.generators.families.mar

MAR (Missing At Random) mechanism implementations.

MAR: P(R | X_obs, X_mis) = P(R | X_obs) - missingness depends only on observed data.
"""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR
from ..base import Generator
from ..params import GeneratorParams
from .base_data import sample_gaussian, sample_gaussian_correlated


class MARLogistic(Generator):
    """MAR mechanism via logistic model.
    
    Missingness in target column depends on predictor column:
    P(R_ij = 0 | X) = sigmoid(alpha0 + alpha1 * X_ik)
    
    where k is the predictor column (k != j).
    
    Required params:
        alpha0: Intercept (controls baseline missingness).
        alpha1: Slope (controls dependence strength).
        target_col_idx: Index of target column (0-indexed, or -1 for last).
        predictor_col_idx: Index of predictor column (0-indexed, or -1 for last).
        base_mean: Mean for Gaussian base data (default: 0.0).
        base_std: Std for Gaussian base data (default: 1.0).
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
    
    def sample(
        self,
        rng: RNGState,
        n: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if d < 2:
            raise ValueError("MARLogistic requires d >= 2")
        
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        rho = self.params.get("correlation", 0.0)
        
        if rho > 0:
            X = sample_gaussian_correlated(rng.spawn(), n, d, mean=mean, std=std, rho=rho)
        else:
            X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        
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
        
        return X, R


class MARMultiPredictor(Generator):
    """MAR mechanism with multiple predictors.
    
    Missingness depends on multiple observed columns:
    P(R_ij = 0 | X) = sigmoid(alpha0 + sum_k alpha_k * X_ik)
    
    Required params:
        alpha0: Intercept.
        alphas: List of coefficients for predictor columns.
        target_col_idx: Index of target column.
        base_mean: Mean for Gaussian base data (default: 0.0).
        base_std: Std for Gaussian base data (default: 1.0).
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
    
    def sample(
        self,
        rng: RNGState,
        n: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alphas = self.params["alphas"]
        n_predictors = len(alphas)
        
        if d < n_predictors + 1:
            raise ValueError(f"MARMultiPredictor with {n_predictors} predictors requires d >= {n_predictors + 1}")
        
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        
        # Target column (default: last)
        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d
        
        # Predictor columns: all columns except target, take first n_predictors
        predictor_cols = [j for j in range(d) if j != target][:n_predictors]
        
        # Initialize all observed
        R = torch.ones(n, d, dtype=torch.bool)
        
        # Compute logits
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
        
        return X, R
