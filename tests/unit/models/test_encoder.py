"""
Tests for lacuna.models.encoder (row-level)
"""

import pytest
import torch
from lacuna.models.encoder import EvidenceEncoder, RowEncoder
from lacuna.data.tokenization import TOKEN_DIM


class TestRowEncoder:
    """Tests for RowEncoder."""
    
    @pytest.fixture
    def encoder(self):
        return RowEncoder(
            token_dim=TOKEN_DIM,
            hidden_dim=64,
            n_layers=2,
            n_heads=4,
            max_cols=16,
        )
    
    def test_output_shape(self, encoder):
        B, d = 4, 8
        tokens = torch.randn(B, d, TOKEN_DIM)
        col_mask = torch.ones(B, d, dtype=torch.bool)
        
        out = encoder(tokens, col_mask)
        
        assert out.shape == (B, 64)  # hidden_dim
    
    def test_handles_padding(self, encoder):
        B, max_d = 4, 16
        tokens = torch.randn(B, max_d, TOKEN_DIM)
        
        col_mask = torch.zeros(B, max_d, dtype=torch.bool)
        col_mask[0, :5] = True
        col_mask[1, :10] = True
        col_mask[2, :3] = True
        col_mask[3, :8] = True
        
        out = encoder(tokens, col_mask)
        
        assert out.shape == (B, 64)
        assert not torch.isnan(out).any()


class TestEvidenceEncoder:
    """Tests for EvidenceEncoder."""
    
    @pytest.fixture
    def encoder(self):
        return EvidenceEncoder(
            token_dim=TOKEN_DIM,
            hidden_dim=64,
            evidence_dim=32,
            n_layers=4,
            n_heads=4,
            max_cols=16,
            max_rows=64,
            row_agg="attention",
        )
    
    def test_output_shape(self, encoder):
        B, max_rows, max_cols = 4, 64, 16
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)  # evidence_dim
    
    def test_handles_variable_sizes(self, encoder):
        B, max_rows, max_cols = 4, 64, 16
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        
        # Variable row counts
        row_mask = torch.zeros(B, max_rows, dtype=torch.bool)
        row_mask[0, :30] = True
        row_mask[1, :50] = True
        row_mask[2, :20] = True
        row_mask[3, :64] = True
        
        # Variable col counts
        col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
        col_mask[0, :5] = True
        col_mask[1, :10] = True
        col_mask[2, :3] = True
        col_mask[3, :16] = True
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)
        assert not torch.isnan(evidence).any()
    
    def test_gradients_flow(self, encoder):
        B, max_rows, max_cols = 2, 32, 8
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM, requires_grad=True)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        evidence = encoder(tokens, row_mask, col_mask)
        loss = evidence.sum()
        loss.backward()
        
        assert tokens.grad is not None
        assert not torch.isnan(tokens.grad).any()
    
    def test_deterministic(self, encoder):
        encoder.eval()
        
        B, max_rows, max_cols = 2, 32, 8
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        with torch.no_grad():
            e1 = encoder(tokens, row_mask, col_mask)
            e2 = encoder(tokens, row_mask, col_mask)
        
        assert torch.equal(e1, e2)


class TestRowAggregationMethods:
    """Test different row aggregation methods."""
    
    def test_attention_aggregation(self):
        encoder = EvidenceEncoder(
            token_dim=TOKEN_DIM,
            hidden_dim=64,
            evidence_dim=32,
            n_layers=4,
            n_heads=4,
            max_cols=16,
            max_rows=64,
            row_agg="attention",
        )
        
        tokens = torch.randn(2, 64, 16, TOKEN_DIM)
        row_mask = torch.ones(2, 64, dtype=torch.bool)
        col_mask = torch.ones(2, 16, dtype=torch.bool)
        
        evidence = encoder(tokens, row_mask, col_mask)
        assert evidence.shape == (2, 32)
    
    def test_mean_aggregation(self):
        encoder = EvidenceEncoder(
            token_dim=TOKEN_DIM,
            hidden_dim=64,
            evidence_dim=32,
            n_layers=4,
            n_heads=4,
            max_cols=16,
            max_rows=64,
            row_agg="mean",
        )
        
        tokens = torch.randn(2, 64, 16, TOKEN_DIM)
        row_mask = torch.ones(2, 64, dtype=torch.bool)
        col_mask = torch.ones(2, 16, dtype=torch.bool)
        
        evidence = encoder(tokens, row_mask, col_mask)
        assert evidence.shape == (2, 32)
