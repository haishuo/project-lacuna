"""
Tests for lacuna.core.validation

Verify validation functions catch errors appropriately.
"""

import pytest
import torch
from lacuna.core.validation import (
    validate_tensor_shape,
    validate_tensor_dtype,
    validate_no_nan_inf,
    validate_positive,
    validate_non_negative,
    validate_probability,
    validate_in_range,
    validate_positive_int,
    validate_non_empty_sequence,
    validate_unique_elements,
    validate_class_id,
    validate_probabilities_sum_to_one,
)
from lacuna.core.exceptions import ValidationError, NumericalError


class TestValidateTensorShape:
    """Tests for validate_tensor_shape."""
    
    def test_correct_shape_passes(self):
        t = torch.randn(3, 4, 5)
        validate_tensor_shape(t, (3, 4, 5))  # Should not raise
    
    def test_wrong_shape_raises(self):
        t = torch.randn(3, 4, 5)
        with pytest.raises(ValidationError):
            validate_tensor_shape(t, (3, 4, 6))
    
    def test_wrong_ndim_raises(self):
        t = torch.randn(3, 4)
        with pytest.raises(ValidationError):
            validate_tensor_shape(t, (3, 4, 5))
    
    def test_wildcard_dim_passes(self):
        t = torch.randn(3, 100, 5)
        validate_tensor_shape(t, (3, -1, 5))  # -1 = any


class TestValidateTensorDtype:
    """Tests for validate_tensor_dtype."""
    
    def test_correct_dtype_passes(self):
        t = torch.randn(10, dtype=torch.float32)
        validate_tensor_dtype(t, torch.float32)
    
    def test_wrong_dtype_raises(self):
        t = torch.randn(10, dtype=torch.float64)
        with pytest.raises(ValidationError):
            validate_tensor_dtype(t, torch.float32)


class TestValidateNoNanInf:
    """Tests for validate_no_nan_inf."""
    
    def test_clean_tensor_passes(self):
        t = torch.randn(100)
        validate_no_nan_inf(t)
    
    def test_nan_raises(self):
        t = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(NumericalError, match="NaN"):
            validate_no_nan_inf(t)
    
    def test_inf_raises(self):
        t = torch.tensor([1.0, float("inf"), 3.0])
        with pytest.raises(NumericalError, match="Inf"):
            validate_no_nan_inf(t)
    
    def test_neg_inf_raises(self):
        t = torch.tensor([1.0, float("-inf"), 3.0])
        with pytest.raises(NumericalError, match="Inf"):
            validate_no_nan_inf(t)


class TestValidatePositive:
    """Tests for validate_positive."""
    
    def test_positive_passes(self):
        validate_positive(1.0)
        validate_positive(0.001)
    
    def test_zero_raises(self):
        with pytest.raises(ValidationError):
            validate_positive(0.0)
    
    def test_negative_raises(self):
        with pytest.raises(ValidationError):
            validate_positive(-1.0)


class TestValidateNonNegative:
    """Tests for validate_non_negative."""
    
    def test_positive_passes(self):
        validate_non_negative(1.0)
    
    def test_zero_passes(self):
        validate_non_negative(0.0)
    
    def test_negative_raises(self):
        with pytest.raises(ValidationError):
            validate_non_negative(-0.001)


class TestValidateProbability:
    """Tests for validate_probability."""
    
    def test_valid_probabilities_pass(self):
        validate_probability(0.0)
        validate_probability(0.5)
        validate_probability(1.0)
    
    def test_negative_raises(self):
        with pytest.raises(ValidationError):
            validate_probability(-0.1)
    
    def test_greater_than_one_raises(self):
        with pytest.raises(ValidationError):
            validate_probability(1.1)


class TestValidateInRange:
    """Tests for validate_in_range."""
    
    def test_in_range_passes(self):
        validate_in_range(5, 0, 10)
        validate_in_range(0, 0, 10)  # Boundary
        validate_in_range(10, 0, 10)  # Boundary
    
    def test_below_range_raises(self):
        with pytest.raises(ValidationError):
            validate_in_range(-1, 0, 10)
    
    def test_above_range_raises(self):
        with pytest.raises(ValidationError):
            validate_in_range(11, 0, 10)


class TestValidatePositiveInt:
    """Tests for validate_positive_int."""
    
    def test_positive_int_passes(self):
        validate_positive_int(1)
        validate_positive_int(100)
    
    def test_zero_raises(self):
        with pytest.raises(ValidationError):
            validate_positive_int(0)
    
    def test_negative_raises(self):
        with pytest.raises(ValidationError):
            validate_positive_int(-1)
    
    def test_float_raises(self):
        with pytest.raises(ValidationError):
            validate_positive_int(1.5)


class TestValidateNonEmptySequence:
    """Tests for validate_non_empty_sequence."""
    
    def test_non_empty_passes(self):
        validate_non_empty_sequence([1, 2, 3])
        validate_non_empty_sequence((1,))
        validate_non_empty_sequence("abc")
    
    def test_empty_raises(self):
        with pytest.raises(ValidationError):
            validate_non_empty_sequence([])


class TestValidateUniqueElements:
    """Tests for validate_unique_elements."""
    
    def test_unique_passes(self):
        validate_unique_elements([1, 2, 3])
        validate_unique_elements(("a", "b", "c"))
    
    def test_duplicates_raises(self):
        with pytest.raises(ValidationError):
            validate_unique_elements([1, 2, 2, 3])


class TestValidateClassId:
    """Tests for validate_class_id."""
    
    def test_valid_class_ids_pass(self):
        validate_class_id(0)
        validate_class_id(1)
        validate_class_id(2)
    
    def test_invalid_class_id_raises(self):
        with pytest.raises(ValidationError):
            validate_class_id(3)
        with pytest.raises(ValidationError):
            validate_class_id(-1)


class TestValidateProbabilitiesSumToOne:
    """Tests for validate_probabilities_sum_to_one."""
    
    def test_valid_probs_pass(self):
        p = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.25, 0.25]])
        validate_probabilities_sum_to_one(p)
    
    def test_invalid_probs_raise(self):
        p = torch.tensor([[0.3, 0.3, 0.3]])  # Sums to 0.9
        with pytest.raises(ValidationError, match="do not sum to 1"):
            validate_probabilities_sum_to_one(p)
