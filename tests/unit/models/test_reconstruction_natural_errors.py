"""
Tests for natural error computation in reconstruction heads.

Run with: pytest tests/unit/models/test_reconstruction_natural_errors.py -v
"""

import pytest
import torch

from lacuna.models.reconstruction import (
    ReconstructionHeads,
    ReconstructionConfig,
    ExtendedReconstructionResult,
    create_reconstruction_heads,
)
from lacuna.data.tokenization import TOKEN_DIM, IDX_VALUE, IDX_OBSERVED, IDX_MASK_TYPE, IDX_FEATURE_ID


@pytest.fixture
def config():
    return ReconstructionConfig(
        hidden_dim=64,
        head_hidden_dim=32,
        n_head_layers=2,
        dropout=0.0,
        mnar_variants=["self_censoring", "threshold"],
    )


@pytest.fixture
def heads(config):
    return ReconstructionHeads(config)


@pytest.fixture
def sample_inputs():
    B, max_rows, max_cols, hidden_dim = 4, 32, 8, 64
    
    token_repr = torch.randn(B, max_rows, max_cols, hidden_dim)
    tokens = torch.zeros(B, max_rows, max_cols, TOKEN_DIM)
    tokens[..., IDX_VALUE] = torch.randn(B, max_rows, max_cols)
    
    is_observed = torch.rand(B, max_rows, max_cols) > 0.3
    tokens[..., IDX_OBSERVED] = is_observed.float()
    tokens[..., IDX_VALUE] = tokens[..., IDX_VALUE] * is_observed.float()
    tokens[..., IDX_MASK_TYPE] = 0.0
    
    for i in range(max_cols):
        tokens[..., i, IDX_FEATURE_ID] = i / max_cols
    
    row_mask = torch.ones(B, max_rows, dtype=torch.bool)
    col_mask = torch.ones(B, max_cols, dtype=torch.bool)
    original_values = torch.randn(B, max_rows, max_cols)
    artificial_mask = is_observed & (torch.rand(B, max_rows, max_cols) < 0.2)
    
    return {
        "token_repr": token_repr,
        "tokens": tokens,
        "row_mask": row_mask,
        "col_mask": col_mask,
        "original_values": original_values,
        "reconstruction_mask": artificial_mask,
        "is_observed": is_observed,
    }


class TestNaturalErrorComputation:
    
    def test_natural_errors_computed(self, heads, sample_inputs):
        results = heads(
            token_repr=sample_inputs["token_repr"],
            tokens=sample_inputs["tokens"],
            row_mask=sample_inputs["row_mask"],
            col_mask=sample_inputs["col_mask"],
            original_values=sample_inputs["original_values"],
            reconstruction_mask=sample_inputs["reconstruction_mask"],
            compute_natural_errors=True,
        )
        
        for name in heads.head_names:
            assert results[name].natural_errors is not None
            assert results[name].n_natural_missing is not None
    
    def test_natural_errors_tensor_shape(self, heads, sample_inputs):
        results = heads(
            token_repr=sample_inputs["token_repr"],
            tokens=sample_inputs["tokens"],
            row_mask=sample_inputs["row_mask"],
            col_mask=sample_inputs["col_mask"],
            original_values=sample_inputs["original_values"],
            reconstruction_mask=sample_inputs["reconstruction_mask"],
            compute_natural_errors=True,
        )
        
        natural_errors = heads.get_natural_error_tensor(results)
        B = sample_inputs["token_repr"].shape[0]
        
        assert natural_errors is not None
        assert natural_errors.shape == (B, heads.n_heads)
    
    def test_natural_missing_mask_correctness(self, heads, sample_inputs):
        natural_missing = heads._compute_natural_missing_mask(
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        expected = ~sample_inputs["is_observed"]
        assert torch.equal(natural_missing, expected)


class TestDiscriminativeFeatures:
    
    def test_feature_extraction_shape(self, heads, sample_inputs):
        results = heads(
            token_repr=sample_inputs["token_repr"],
            tokens=sample_inputs["tokens"],
            row_mask=sample_inputs["row_mask"],
            col_mask=sample_inputs["col_mask"],
            original_values=sample_inputs["original_values"],
            reconstruction_mask=sample_inputs["reconstruction_mask"],
            compute_natural_errors=True,
        )
        
        features = heads.get_natural_error_features(results)
        B = sample_inputs["token_repr"].shape[0]
        n_heads = heads.n_heads
        
        # Features: log_ratios (n_heads-1) + differences (n_heads-1) + mar_vs_mnar (1)
        expected_dim = (n_heads - 1) * 2 + 1
        
        assert features is not None
        assert features.shape == (B, expected_dim)
    
    def test_features_finite(self, heads, sample_inputs):
        results = heads(
            token_repr=sample_inputs["token_repr"],
            tokens=sample_inputs["tokens"],
            row_mask=sample_inputs["row_mask"],
            col_mask=sample_inputs["col_mask"],
            original_values=sample_inputs["original_values"],
            reconstruction_mask=sample_inputs["reconstruction_mask"],
            compute_natural_errors=True,
        )
        
        features = heads.get_natural_error_features(results)
        
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()