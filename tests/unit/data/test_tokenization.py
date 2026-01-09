"""
Tests for lacuna.data.tokenization
"""

import pytest
import torch
from lacuna.data.observed import create_observed_dataset
from lacuna.data.tokenization import tokenize_dataset, get_token_dim
from lacuna.data.features import FEATURE_DIM


class TestTokenizeDataset:
    """Tests for tokenize_dataset."""
    
    def test_output_shape(self):
        x = torch.randn(100, 5)
        r = torch.ones(100, 5, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        tokens = tokenize_dataset(ds)
        
        assert tokens.shape == (5, FEATURE_DIM)
    
    def test_deterministic(self):
        x = torch.randn(50, 3)
        r = torch.ones(50, 3, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        tokens1 = tokenize_dataset(ds)
        tokens2 = tokenize_dataset(ds)
        
        assert torch.equal(tokens1, tokens2)
    
    def test_normalize_option(self):
        x = torch.randn(100, 3) * 10 + 50  # Non-standard scale
        r = torch.ones(100, 3, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        tokens_norm = tokenize_dataset(ds, normalize=True)
        tokens_raw = tokenize_dataset(ds, normalize=False)
        
        # Should produce different results
        assert not torch.allclose(tokens_norm, tokens_raw)
    
    def test_get_token_dim(self):
        assert get_token_dim() == FEATURE_DIM
        assert get_token_dim() > 0
