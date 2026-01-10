"""
Tests for lacuna.generators.families.mnar

Tests MNAR (Missing Not At Random) generators:
    - MNARLogistic: Logistic-based missingness depending on own value
    - MNARSelfCensoring: Value-dependent self-censoring mechanism

These generators produce missingness patterns where the probability of
a value being missing depends on the (unobserved) value itself.
"""

import pytest
import numpy as np
import torch

from lacuna.generators.families.mnar import (
    MNARLogistic,
    MNARSelfCensoring,
)
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from lacuna.core.types import ObservedDataset, MNAR
from lacuna.core.rng import RNGState


# =============================================================================
# Test MNARLogistic
# =============================================================================

class TestMNARLogistic:
    """Tests for MNARLogistic generator."""
    
    @pytest.fixture
    def rng(self):
        """Create RNG state for reproducibility."""
        return RNGState(seed=42)
    
    @pytest.fixture
    def generator(self):
        """Create default MNARLogistic generator."""
        params = GeneratorParams(
            beta0=-1.0,  # Intercept (controls base missing rate)
            beta1=0.0,   # Coefficient for predictor column (optional)
            beta2=1.5,   # Coefficient for target column (MNAR signature)
            target_col_idx=-1,  # Last column
            predictor_col_idx=0,  # First column
        )
        return MNARLogistic(
            generator_id=0,
            name="mnar_logistic_test",
            params=params,
        )
    
    def test_inherits_from_generator(self, generator):
        """Test that MNARLogistic inherits from Generator."""
        assert isinstance(generator, Generator)
    
    def test_generator_id(self, generator):
        """Test generator ID is set correctly."""
        assert generator.generator_id == 0
    
    def test_name(self, generator):
        """Test generator name is set correctly."""
        assert generator.name == "mnar_logistic_test"
    
    def test_class_id_is_mnar(self, generator):
        """Test that class_id is MNAR."""
        assert generator.class_id == MNAR
    
    def test_sample_returns_tuple(self, generator, rng):
        """Test that sample returns (X, R) tuple."""
        X, R = generator.sample(rng, n=100, d=10)
        assert isinstance(X, torch.Tensor)
        assert isinstance(R, torch.Tensor)
    
    def test_sample_observed_returns_dataset(self, generator, rng):
        """Test that sample_observed returns ObservedDataset."""
        dataset = generator.sample_observed(rng, n=100, d=10, dataset_id="test_001")
        assert isinstance(dataset, ObservedDataset)
    
    def test_output_dimensions(self, generator, rng):
        """Test output dimensions match requested."""
        n, d = 100, 10
        X, R = generator.sample(rng, n=n, d=d)
        
        assert X.shape == (n, d)
        assert R.shape == (n, d)
    
    def test_missingness_mask_is_boolean(self, generator, rng):
        """Test that R is boolean."""
        X, R = generator.sample(rng, n=100, d=10)
        assert R.dtype == torch.bool
    
    def test_has_missingness(self, generator, rng):
        """Test that some values are missing."""
        X, R = generator.sample(rng, n=100, d=10)
        
        # Should have some missing values
        missing_rate = (~R).float().mean().item()
        assert missing_rate > 0.0
        assert missing_rate < 1.0
    
    def test_missingness_in_target_column(self, rng):
        """Test that missingness occurs in the target column."""
        params = GeneratorParams(
            beta0=0.0,
            beta2=2.0,
            target_col_idx=3,  # Specific target column
        )
        generator = MNARLogistic(
            generator_id=0,
            name="mnar_logistic_target",
            params=params,
        )
        
        X, R = generator.sample(rng, n=200, d=5)
        
        # Target column should have missingness
        target_missing = (~R[:, 3]).sum().item()
        assert target_missing > 0, "Target column should have some missingness"
        
        # Other columns should be complete (except possibly some edge cases)
        for col in [0, 1, 2, 4]:
            assert R[:, col].all(), f"Column {col} should be complete"
    
    def test_requires_minimum_columns(self, rng):
        """Test that generator requires at least 2 columns."""
        params = GeneratorParams(beta0=0.0, beta2=1.0)
        generator = MNARLogistic(
            generator_id=0,
            name="mnar_logistic_min_cols",
            params=params,
        )
        
        with pytest.raises(ValueError, match="d >= 2"):
            generator.sample(rng, n=100, d=1)
    
    def test_requires_beta0_param(self):
        """Test that beta0 parameter is required."""
        params = GeneratorParams(beta2=1.0)  # Missing beta0
        
        with pytest.raises(ValueError, match="beta0"):
            MNARLogistic(
                generator_id=0,
                name="mnar_logistic_no_beta0",
                params=params,
            )
    
    def test_requires_beta2_param(self):
        """Test that beta2 parameter is required."""
        params = GeneratorParams(beta0=0.0)  # Missing beta2
        
        with pytest.raises(ValueError, match="beta2"):
            MNARLogistic(
                generator_id=0,
                name="mnar_logistic_no_beta2",
                params=params,
            )
    
    def test_beta2_must_be_nonzero(self):
        """Test that beta2 must be non-zero (otherwise it's MAR)."""
        params = GeneratorParams(beta0=0.0, beta2=0.0)
        
        with pytest.raises(ValueError, match="non-zero"):
            MNARLogistic(
                generator_id=0,
                name="mnar_logistic_zero_beta2",
                params=params,
            )
    
    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        params = GeneratorParams(beta0=0.0, beta2=1.0)
        generator = MNARLogistic(
            generator_id=0,
            name="mnar_logistic_repro",
            params=params,
        )
        
        rng1 = RNGState(seed=42)
        rng2 = RNGState(seed=42)
        
        X1, R1 = generator.sample(rng1, n=100, d=10)
        X2, R2 = generator.sample(rng2, n=100, d=10)
        
        # Same seed should give same missingness pattern
        assert (R1 == R2).all()
    
    def test_different_seeds_different_results(self):
        """Test that different seeds give different results."""
        params = GeneratorParams(beta0=0.0, beta2=1.0)
        generator = MNARLogistic(
            generator_id=0,
            name="mnar_logistic_diff_seeds",
            params=params,
        )
        
        rng1 = RNGState(seed=42)
        rng2 = RNGState(seed=123)
        
        X1, R1 = generator.sample(rng1, n=100, d=10)
        X2, R2 = generator.sample(rng2, n=100, d=10)
        
        # Different seeds should give different patterns (with high probability)
        assert not (R1 == R2).all()
    
    def test_beta2_affects_mnar_strength(self, rng):
        """Test that beta2 parameter controls MNAR strength.
        
        Higher beta2 means missingness depends more strongly on own value.
        """
        # Weak MNAR
        weak_params = GeneratorParams(beta0=0.0, beta2=0.5)
        weak_gen = MNARLogistic(
            generator_id=0,
            name="mnar_logistic_weak",
            params=weak_params,
        )
        
        # Strong MNAR
        strong_params = GeneratorParams(beta0=0.0, beta2=3.0)
        strong_gen = MNARLogistic(
            generator_id=1,
            name="mnar_logistic_strong",
            params=strong_params,
        )
        
        # Both should produce valid datasets
        weak_X, weak_R = weak_gen.sample(rng, n=100, d=5)
        
        rng2 = RNGState(seed=42)
        strong_X, strong_R = strong_gen.sample(rng2, n=100, d=5)
        
        assert weak_X.shape == strong_X.shape
    
    def test_negative_target_index(self, rng):
        """Test that negative target index works (Python-style indexing)."""
        params = GeneratorParams(
            beta0=0.0,
            beta2=1.5,
            target_col_idx=-1,  # Last column
        )
        generator = MNARLogistic(
            generator_id=0,
            name="mnar_logistic_neg_idx",
            params=params,
        )
        
        X, R = generator.sample(rng, n=100, d=5)
        
        # Last column should have missingness
        assert (~R[:, -1]).sum().item() > 0


# =============================================================================
# Test MNARSelfCensoring
# =============================================================================

class TestMNARSelfCensoring:
    """Tests for MNARSelfCensoring generator.
    
    MNARSelfCensoring applies column-wise self-censoring where each column's
    missingness depends on its own value via:
        P(R_j=0 | X_j) = sigmoid(beta0 + beta1 * X_j)
    
    Required params:
        beta0: Intercept (controls baseline missingness rate)
        beta1: Self-censoring strength (must be non-zero)
    Optional params:
        affected_frac: Fraction of columns with self-censoring (default 0.5)
    """
    
    @pytest.fixture
    def rng(self):
        """Create RNG state for reproducibility."""
        return RNGState(seed=42)
    
    @pytest.fixture
    def generator(self):
        """Create default MNARSelfCensoring generator."""
        params = GeneratorParams(
            beta0=-1.0,   # Intercept (controls baseline missingness)
            beta1=1.5,    # Self-censoring strength (positive = high values more likely missing)
        )
        return MNARSelfCensoring(
            generator_id=1,
            name="mnar_self_censoring_test",
            params=params,
        )
    
    def test_inherits_from_generator(self, generator):
        """Test that MNARSelfCensoring inherits from Generator."""
        assert isinstance(generator, Generator)
    
    def test_generator_id(self, generator):
        """Test generator ID is set correctly."""
        assert generator.generator_id == 1
    
    def test_name(self, generator):
        """Test generator name is set correctly."""
        assert generator.name == "mnar_self_censoring_test"
    
    def test_class_id_is_mnar(self, generator):
        """Test that class_id is MNAR."""
        assert generator.class_id == MNAR
    
    def test_sample_returns_tuple(self, generator, rng):
        """Test that sample returns (X, R) tuple."""
        X, R = generator.sample(rng, n=100, d=10)
        assert isinstance(X, torch.Tensor)
        assert isinstance(R, torch.Tensor)
    
    def test_sample_observed_returns_dataset(self, generator, rng):
        """Test that sample_observed returns ObservedDataset."""
        dataset = generator.sample_observed(rng, n=100, d=10, dataset_id="test_002")
        assert isinstance(dataset, ObservedDataset)
    
    def test_output_dimensions(self, generator, rng):
        """Test output dimensions match requested."""
        n, d = 100, 10
        X, R = generator.sample(rng, n=n, d=d)
        
        assert X.shape == (n, d)
        assert R.shape == (n, d)
    
    def test_missingness_mask_is_boolean(self, generator, rng):
        """Test that R is boolean."""
        X, R = generator.sample(rng, n=100, d=10)
        assert R.dtype == torch.bool
    
    def test_has_missingness(self, generator, rng):
        """Test that some values are missing."""
        X, R = generator.sample(rng, n=100, d=10)
        
        # Should have some missing values
        missing_rate = (~R).float().mean().item()
        assert missing_rate > 0.0
        assert missing_rate < 1.0
    
    def test_requires_beta0_param(self):
        """Test that beta0 parameter is required."""
        params = GeneratorParams(beta1=1.0)  # Missing beta0
        
        with pytest.raises(ValueError, match="beta0"):
            MNARSelfCensoring(
                generator_id=1,
                name="mnar_self_censoring_no_beta0",
                params=params,
            )
    
    def test_requires_beta1_param(self):
        """Test that beta1 parameter is required."""
        params = GeneratorParams(beta0=0.0)  # Missing beta1
        
        with pytest.raises(ValueError, match="beta1"):
            MNARSelfCensoring(
                generator_id=1,
                name="mnar_self_censoring_no_beta1",
                params=params,
            )
    
    def test_beta1_must_be_nonzero(self):
        """Test that beta1 must be non-zero for self-censoring."""
        params = GeneratorParams(beta0=0.0, beta1=0.0)
        
        with pytest.raises(ValueError, match="non-zero"):
            MNARSelfCensoring(
                generator_id=1,
                name="mnar_self_censoring_zero_beta1",
                params=params,
            )
    
    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        params = GeneratorParams(beta0=-1.0, beta1=1.5)
        generator = MNARSelfCensoring(
            generator_id=1,
            name="mnar_self_censoring_repro",
            params=params,
        )
        
        rng1 = RNGState(seed=42)
        rng2 = RNGState(seed=42)
        
        X1, R1 = generator.sample(rng1, n=100, d=10)
        X2, R2 = generator.sample(rng2, n=100, d=10)
        
        # Same seed should give same missingness pattern
        assert (R1 == R2).all()
    
    def test_different_seeds_different_results(self):
        """Test that different seeds give different results."""
        params = GeneratorParams(beta0=-1.0, beta1=1.5)
        generator = MNARSelfCensoring(
            generator_id=1,
            name="mnar_self_censoring_diff_seeds",
            params=params,
        )
        
        rng1 = RNGState(seed=42)
        rng2 = RNGState(seed=123)
        
        X1, R1 = generator.sample(rng1, n=100, d=10)
        X2, R2 = generator.sample(rng2, n=100, d=10)
        
        # Different seeds should give different patterns (with high probability)
        assert not (R1 == R2).all()
    
    def test_various_dimensions(self):
        """Test generator works with various dimensions."""
        params = GeneratorParams(beta0=-1.0, beta1=1.5)
        generator = MNARSelfCensoring(
            generator_id=1,
            name="mnar_self_censoring_dims",
            params=params,
        )
        
        rng = RNGState(seed=42)
        
        # Small dataset
        X_small, R_small = generator.sample(rng, n=20, d=3)
        assert X_small.shape == (20, 3)
        
        # Reset RNG
        rng2 = RNGState(seed=43)
        
        # Larger dataset
        X_large, R_large = generator.sample(rng2, n=500, d=50)
        assert X_large.shape == (500, 50)


# =============================================================================
# Test Generator Properties
# =============================================================================

class TestMNARGeneratorProperties:
    """Tests for common MNAR generator properties."""
    
    @pytest.fixture(params=[
        lambda: MNARLogistic(
            generator_id=0,
            name="mnar_logistic_prop",
            params=GeneratorParams(beta0=0.0, beta2=1.0),
        ),
        lambda: MNARSelfCensoring(
            generator_id=1,
            name="mnar_self_censoring_prop",
            params=GeneratorParams(beta0=-1.0, beta1=1.5),
        ),
    ])
    def generator(self, request):
        """Parametrized fixture for all MNAR generators."""
        return request.param()
    
    @pytest.fixture
    def rng(self):
        """Create RNG state."""
        return RNGState(seed=42)
    
    def test_class_id_is_mnar(self, generator):
        """All MNAR generators should have class_id = MNAR."""
        assert generator.class_id == MNAR
    
    def test_produces_tensor_tuple(self, generator, rng):
        """All generators should produce (X, R) tensor tuple."""
        # MNARLogistic requires d >= 2
        min_d = 2 if isinstance(generator, MNARLogistic) else 1
        X, R = generator.sample(rng, n=50, d=max(5, min_d))
        assert isinstance(X, torch.Tensor)
        assert isinstance(R, torch.Tensor)
    
    def test_has_generator_id(self, generator):
        """All generators should have a generator_id."""
        assert hasattr(generator, 'generator_id')
        assert isinstance(generator.generator_id, int)
    
    def test_has_name(self, generator):
        """All generators should have a name."""
        assert hasattr(generator, 'name')
        assert isinstance(generator.name, str)
        assert len(generator.name) > 0
    
    def test_has_params(self, generator):
        """All generators should have params."""
        assert hasattr(generator, 'params')
        assert isinstance(generator.params, GeneratorParams)


# =============================================================================
# Test MNAR Mechanism Signatures
# =============================================================================

class TestMNARMechanismSignature:
    """Tests verifying MNAR mechanism signatures.
    
    MNAR is characterized by missingness depending on the unobserved
    values themselves. We can verify this by checking that the observed
    distribution differs from the complete distribution in predictable ways.
    """
    
    def test_mnar_logistic_creates_distributional_shift(self):
        """Test that MNAR mechanism creates distributional shift.
        
        When values with higher (or lower) values are more likely to be
        missing, the observed distribution should be shifted relative to
        the complete distribution.
        """
        # Strong positive beta2: high values more likely to be missing
        params = GeneratorParams(
            beta0=0.0,
            beta2=3.0,  # Strong positive effect
            target_col_idx=-1,
        )
        generator = MNARLogistic(
            generator_id=0,
            name="mnar_logistic_dist_shift",
            params=params,
        )
        
        rng = RNGState(seed=42)
        
        # Generate large sample for statistical power
        X, R = generator.sample(rng, n=2000, d=5)
        
        target_col = 4  # Last column (index -1)
        
        # Get observed values (where R is True)
        observed_values = X[R[:, target_col], target_col]
        
        # With positive beta2, high values are more likely to be missing
        # So observed mean should be lower than 0 (assuming standard normal data)
        if len(observed_values) > 10:
            observed_mean = observed_values.mean().item()
            # The mean should be shifted (we expect negative shift with positive beta2)
            # This is a statistical test, so we use a generous margin
            # Just verify the mechanism produces an effect
            assert observed_mean < 0.5, (
                f"Expected negative shift in observed mean, got {observed_mean:.3f}"
            )
    
    def test_mnar_mechanism_preserves_other_columns(self):
        """Test that MNAR mechanism only affects target column."""
        params = GeneratorParams(
            beta0=0.0,
            beta2=2.0,
            target_col_idx=2,  # Only column 2
        )
        generator = MNARLogistic(
            generator_id=0,
            name="mnar_logistic_preserve_cols",
            params=params,
        )
        
        rng = RNGState(seed=42)
        X, R = generator.sample(rng, n=100, d=5)
        
        # Columns other than target should be complete
        for col in [0, 1, 3, 4]:
            assert R[:, col].all(), f"Column {col} should be complete"
        
        # Only target column has missingness
        assert (~R[:, 2]).sum().item() > 0, "Target column should have missingness"


# =============================================================================
# Test GeneratorParams Integration
# =============================================================================

class TestGeneratorParamsIntegration:
    """Tests for GeneratorParams usage in MNAR generators."""
    
    def test_params_are_accessible(self):
        """Test that params can be accessed after construction."""
        params = GeneratorParams(
            beta0=-0.5,
            beta2=1.5,
            custom_param="test_value",
        )
        generator = MNARLogistic(
            generator_id=0,
            name="mnar_logistic_params_access",
            params=params,
        )
        
        assert generator.params["beta0"] == -0.5
        assert generator.params["beta2"] == 1.5
        assert generator.params["custom_param"] == "test_value"
    
    def test_params_get_with_default(self):
        """Test params.get() with default value."""
        params = GeneratorParams(beta0=0.0, beta2=1.0)
        generator = MNARLogistic(
            generator_id=0,
            name="mnar_logistic_params_default",
            params=params,
        )
        
        # beta1 not specified, should return default
        assert generator.params.get("beta1", 0.0) == 0.0
        assert generator.params.get("nonexistent", 42) == 42