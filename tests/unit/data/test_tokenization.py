"""
Tests for lacuna.data.tokenization (row-level)
"""

import pytest
import torch
from lacuna.data.observed import create_observed_dataset
from lacuna.data.tokenization import (
    tokenize_dataset,
    tokenize_row,
    get_token_dim,
    TOKEN_DIM,
)


class TestTokenizeRow:
    """Tests for tokenize_row."""
    
    def test_output_shape(self):
        row_x = torch.randn(5)
        row_r = torch.ones(5, dtype=torch.bool)
        
        tokens = tokenize_row(row_x, row_r)
        
        assert tokens.shape == (5, TOKEN_DIM)
    
    def test_observed_values_preserved(self):
        row_x = torch.tensor([1.0, 2.0, 3.0])
        row_r = torch.ones(3, dtype=torch.bool)
        
        tokens = tokenize_row(row_x, row_r)
        
        # Channel 0 should have values
        assert torch.allclose(tokens[:, 0], row_x)
        # Channel 1 should be all 1s (observed)
        assert torch.allclose(tokens[:, 1], torch.ones(3))
    
    def test_missing_values_zeroed(self):
        row_x = torch.tensor([1.0, 2.0, 3.0])
        row_r = torch.tensor([True, False, True])
        
        tokens = tokenize_row(row_x, row_r)
        
        # Channel 0: observed values kept, missing zeroed
        assert tokens[0, 0] == 1.0
        assert tokens[1, 0] == 0.0  # Missing -> 0
        assert tokens[2, 0] == 3.0
        
        # Channel 1: observation indicator
        assert tokens[0, 1] == 1.0
        assert tokens[1, 1] == 0.0  # Missing
        assert tokens[2, 1] == 1.0


class TestTokenizeDataset:
    """Tests for tokenize_dataset."""
    
    def test_output_shape(self):
        x = torch.randn(100, 5)
        r = torch.ones(100, 5, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        tokens = tokenize_dataset(ds)
        
        assert tokens.shape == (100, 5, TOKEN_DIM)
    
    def test_normalization(self):
        # Create data with known mean/std
        x = torch.randn(100, 3) * 10 + 50
        r = torch.ones(100, 3, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        tokens = tokenize_dataset(ds, normalize=True)
        
        # After normalization, values should be roughly standard normal
        values = tokens[:, :, 0]
        assert abs(values.mean().item()) < 0.5
        assert abs(values.std().item() - 1.0) < 0.5
    
    def test_no_normalization(self):
        x = torch.randn(100, 3) * 10 + 50
        r = torch.ones(100, 3, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        tokens = tokenize_dataset(ds, normalize=False)
        
        # Values should be close to original
        values = tokens[:, :, 0]
        assert abs(values.mean().item() - 50) < 5
    
    def test_missing_values_handled(self):
        x = torch.randn(50, 3)
        r = torch.ones(50, 3, dtype=torch.bool)
        r[:10, 0] = False  # 20% missing in column 0
        
        ds = create_observed_dataset(x=x, r=r)
        tokens = tokenize_dataset(ds)
        
        # Check missing indicators
        assert tokens[:10, 0, 1].sum() == 0  # Missing
        assert tokens[10:, 0, 1].sum() == 40  # Observed
    
    def test_deterministic(self):
        x = torch.randn(50, 3)
        r = torch.ones(50, 3, dtype=torch.bool)
        ds = create_observed_dataset(x=x, r=r)
        
        tokens1 = tokenize_dataset(ds)
        tokens2 = tokenize_dataset(ds)
        
        assert torch.equal(tokens1, tokens2)


class TestTokenDim:
    """Tests for token dimension."""
    
    def test_get_token_dim(self):
        assert get_token_dim() == TOKEN_DIM
        assert TOKEN_DIM == 2  # [value, is_observed]
