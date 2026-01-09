"""
Tests for lacuna.generators.families (MCAR, MAR, MNAR)
"""

import pytest
import torch
from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.generators.params import GeneratorParams
from lacuna.generators.families.mcar import MCARUniform, MCARColumnwise
from lacuna.generators.families.mar import MARLogistic, MARMultiPredictor
from lacuna.generators.families.mnar import MNARLogistic, MNARSelfCensoring


class TestMCARUniform:
    """Tests for MCARUniform generator."""
    
    def test_construction(self):
        gen = MCARUniform(0, "test", GeneratorParams(miss_rate=0.2))
        assert gen.class_id == MCAR
    
    def test_missing_param_raises(self):
        with pytest.raises(ValueError, match="miss_rate"):
            MCARUniform(0, "test", GeneratorParams())
    
    def test_invalid_miss_rate_raises(self):
        with pytest.raises(ValueError, match="miss_rate"):
            MCARUniform(0, "test", GeneratorParams(miss_rate=1.5))
        with pytest.raises(ValueError, match="miss_rate"):
            MCARUniform(0, "test", GeneratorParams(miss_rate=0.0))
    
    def test_sample_shapes(self):
        gen = MCARUniform(0, "test", GeneratorParams(miss_rate=0.3))
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=100, d=5)
        
        assert X.shape == (100, 5)
        assert R.shape == (100, 5)
        assert R.dtype == torch.bool
    
    def test_approximate_miss_rate(self):
        gen = MCARUniform(0, "test", GeneratorParams(miss_rate=0.3))
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=10000, d=10)
        
        empirical_rate = 1.0 - R.float().mean().item()
        assert abs(empirical_rate - 0.3) < 0.02
    
    def test_at_least_one_observed(self):
        gen = MCARUniform(0, "test", GeneratorParams(miss_rate=0.99))
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=10, d=2)
        
        assert R.sum() >= 1


class TestMCARColumnwise:
    """Tests for MCARColumnwise generator."""
    
    def test_sample_different_column_rates(self):
        gen = MCARColumnwise(0, "test", GeneratorParams(miss_rate_range=(0.1, 0.5)))
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=5000, d=5)
        
        col_rates = 1.0 - R.float().mean(dim=0)
        
        # All rates should be in range
        assert (col_rates >= 0.05).all()  # Allow some margin
        assert (col_rates <= 0.55).all()
        
        # Rates should vary between columns
        assert col_rates.std() > 0.01


class TestMARLogistic:
    """Tests for MARLogistic generator."""
    
    def test_construction(self):
        gen = MARLogistic(0, "test", GeneratorParams(alpha0=0.0, alpha1=1.0))
        assert gen.class_id == MAR
    
    def test_missing_param_raises(self):
        with pytest.raises(ValueError, match="alpha0"):
            MARLogistic(0, "test", GeneratorParams(alpha1=1.0))
    
    def test_requires_d_ge_2(self):
        gen = MARLogistic(0, "test", GeneratorParams(alpha0=0.0, alpha1=1.0))
        rng = RNGState(seed=42)
        
        with pytest.raises(ValueError, match="d >= 2"):
            gen.sample(rng, n=100, d=1)
    
    def test_sample_shapes(self):
        gen = MARLogistic(0, "test", GeneratorParams(alpha0=0.0, alpha1=1.0))
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=100, d=5)
        
        assert X.shape == (100, 5)
        assert R.shape == (100, 5)
    
    def test_missingness_depends_on_predictor(self):
        """Verify MAR: missingness correlates with predictor column."""
        gen = MARLogistic(
            0, "test",
            GeneratorParams(
                alpha0=0.0,
                alpha1=2.0,  # Strong positive relationship
                target_col_idx=-1,  # Last column is target
                predictor_col_idx=0,  # First column is predictor
            )
        )
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=5000, d=3)
        
        # Split by predictor value
        predictor = X[:, 0]
        target_missing = ~R[:, -1]
        
        high_pred = predictor > predictor.median()
        
        miss_rate_high = target_missing[high_pred].float().mean()
        miss_rate_low = target_missing[~high_pred].float().mean()
        
        # Higher predictor -> higher missingness (positive alpha1)
        assert miss_rate_high > miss_rate_low + 0.1


class TestMARMultiPredictor:
    """Tests for MARMultiPredictor generator."""
    
    def test_sample_shapes(self):
        gen = MARMultiPredictor(
            0, "test",
            GeneratorParams(alpha0=0.0, alphas=[1.0, 0.5])
        )
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=100, d=5)
        
        assert X.shape == (100, 5)
        assert R.shape == (100, 5)


class TestMNARLogistic:
    """Tests for MNARLogistic generator."""
    
    def test_construction(self):
        gen = MNARLogistic(0, "test", GeneratorParams(beta0=0.0, beta2=1.0))
        assert gen.class_id == MNAR
    
    def test_beta2_zero_raises(self):
        with pytest.raises(ValueError, match="beta2 must be non-zero"):
            MNARLogistic(0, "test", GeneratorParams(beta0=0.0, beta2=0.0))
    
    def test_sample_shapes(self):
        gen = MNARLogistic(0, "test", GeneratorParams(beta0=0.0, beta2=1.0))
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=100, d=5)
        
        assert X.shape == (100, 5)
        assert R.shape == (100, 5)
    
    def test_missingness_depends_on_target(self):
        """Verify MNAR: missingness correlates with target column value."""
        gen = MNARLogistic(
            0, "test",
            GeneratorParams(
                beta0=0.0,
                beta1=0.0,  # No predictor effect
                beta2=2.0,  # Strong target effect
                target_col_idx=-1,
                predictor_col_idx=0,
            )
        )
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=5000, d=3)
        
        # For MNAR, we need to look at the COMPLETE data X
        # (before missingness is applied)
        target_vals = X[:, -1]
        target_missing = ~R[:, -1]
        
        high_vals = target_vals > target_vals.median()
        
        miss_rate_high = target_missing[high_vals].float().mean()
        miss_rate_low = target_missing[~high_vals].float().mean()
        
        # Higher target value -> higher missingness (positive beta2)
        assert miss_rate_high > miss_rate_low + 0.1


class TestMNARSelfCensoring:
    """Tests for MNARSelfCensoring generator."""
    
    def test_construction(self):
        gen = MNARSelfCensoring(0, "test", GeneratorParams(beta0=0.0, beta1=1.0))
        assert gen.class_id == MNAR
    
    def test_beta1_zero_raises(self):
        with pytest.raises(ValueError, match="beta1 must be non-zero"):
            MNARSelfCensoring(0, "test", GeneratorParams(beta0=0.0, beta1=0.0))
    
    def test_sample_shapes(self):
        gen = MNARSelfCensoring(0, "test", GeneratorParams(beta0=0.0, beta1=1.0))
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=100, d=5)
        
        assert X.shape == (100, 5)
        assert R.shape == (100, 5)


class TestBaseDataSamplers:
    """Tests for base data sampling functions."""
    
    def test_sample_gaussian(self):
        from lacuna.generators.families.base_data import sample_gaussian
        
        rng = RNGState(seed=42)
        X = sample_gaussian(rng, n=1000, d=5, mean=10.0, std=2.0)
        
        assert X.shape == (1000, 5)
        assert abs(X.mean().item() - 10.0) < 0.2
        assert abs(X.std().item() - 2.0) < 0.2
    
    def test_sample_gaussian_correlated(self):
        from lacuna.generators.families.base_data import sample_gaussian_correlated
        
        rng = RNGState(seed=42)
        X = sample_gaussian_correlated(rng, n=5000, d=5, rho=0.8)
        
        # Check adjacent columns are correlated
        corr = torch.corrcoef(X.T)
        assert corr[0, 1].item() > 0.7  # Should be close to 0.8
    
    def test_sample_uniform(self):
        from lacuna.generators.families.base_data import sample_uniform
        
        rng = RNGState(seed=42)
        X = sample_uniform(rng, n=1000, d=5, low=2.0, high=5.0)
        
        assert X.shape == (1000, 5)
        assert (X >= 2.0).all()
        assert (X <= 5.0).all()
    
    def test_sample_mixed(self):
        from lacuna.generators.families.base_data import sample_mixed
        
        rng = RNGState(seed=42)
        X = sample_mixed(rng, n=100, d=6, gaussian_cols=4)
        
        assert X.shape == (100, 6)
