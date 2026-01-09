"""
lacuna.generators.families.mnar

MNAR (Missing Not At Random) mechanism implementations.

MNAR: P(R | X_obs, X_mis) depends on X_mis - missingness depends on unobserved values.
"""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MNAR
from ..base import Generator
from ..params import GeneratorParams
from .base_data import sample_gaussian, sample_gaussian_correlated


class MNARLogistic(Generator):
    """MNAR mechanism via logistic model.
    
    Missingness depends on the value itself (plus optionally a predictor):
    P(R_ij = 0 | X) = sigmoid(beta0 + beta1 * X_ik + beta2 * X_ij)
    
    where beta2 != 0 makes this MNAR.
    
    Required params:
        beta0: Intercept.
        beta1: Coefficient for predictor column (can be 0).
        beta2: Coefficient for target column (MUST be != 0).
        target_col_idx: Index of target column.
        predictor_col_idx: Index of predictor column.
        base_mean: Mean for Gaussian base data (default: 0.0).
        base_std: Std for Gaussian base data (default: 1.0).
    """
    
    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        super().__init__(generator_id, name, MNAR, params)
        
        required = ["beta0", "beta2"]
        for key in required:
            if key not in params:
                raise ValueError(f"MNARLogistic requires '{key}' parameter")
        
        if params["beta2"] == 0:
            raise ValueError("beta2 must be non-zero for MNAR (beta2=0 would be MAR)")
    
    def sample(
        self,
        rng: RNGState,
        n: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if d < 2:
            raise ValueError("MNARLogistic requires d >= 2")
        
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        rho = self.params.get("correlation", 0.0)
        
        if rho > 0:
            X = sample_gaussian_correlated(rng.spawn(), n, d, mean=mean, std=std, rho=rho)
        else:
            X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        
        # Determine columns
        target = self.params.get("target_col_idx", -1)
        predictor = self.params.get("predictor_col_idx", 0)
        
        if target < 0:
            target = d + target
        if predictor < 0:
            predictor = d + predictor
        
        target = target % d
        predictor = predictor % d
        if predictor == target:
            predictor = (target + 1) % d
        
        R = torch.ones(n, d, dtype=torch.bool)
        
        beta0 = self.params["beta0"]
        beta1 = self.params.get("beta1", 0.0)
        beta2 = self.params["beta2"]
        
        # MNAR: depends on target value itself
        logits = beta0 + beta1 * X[:, predictor] + beta2 * X[:, target]
        p_missing = torch.sigmoid(logits)
        
        missing_mask = rng.rand(n) < p_missing
        R[:, target] = ~missing_mask
        
        if R.sum() == 0:
            R[0, 0] = True
        
        return X, R


class MNARSelfCensoring(Generator):
    """MNAR via self-censoring (value-dependent missingness).
    
    Values above/below threshold are more likely to be missing.
    P(R_ij = 0 | X_ij) = sigmoid(beta0 + beta1 * X_ij)
    
    Pure MNAR: only depends on unobserved value, no predictor.
    
    Required params:
        beta0: Intercept.
        beta1: Coefficient (positive = high values missing, negative = low values missing).
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
        super().__init__(generator_id, name, MNAR, params)
        
        required = ["beta0", "beta1"]
        for key in required:
            if key not in params:
                raise ValueError(f"MNARSelfCensoring requires '{key}' parameter")
        
        if params["beta1"] == 0:
            raise ValueError("beta1 must be non-zero for MNAR self-censoring")
    
    def sample(
        self,
        rng: RNGState,
        n: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        
        target = self.params.get("target_col_idx", 0)
        if target < 0:
            target = d + target
        target = target % d
        
        R = torch.ones(n, d, dtype=torch.bool)
        
        beta0 = self.params["beta0"]
        beta1 = self.params["beta1"]
        
        logits = beta0 + beta1 * X[:, target]
        p_missing = torch.sigmoid(logits)
        
        missing_mask = rng.rand(n) < p_missing
        R[:, target] = ~missing_mask
        
        if R.sum() == 0:
            R[0, 0] = True
        
        return X, R
