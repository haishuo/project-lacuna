"""
Tests for lacuna.models.encoder
"""

import pytest
import torch
from lacuna.models.encoder import EvidenceEncoder
from lacuna.data.features import FEATURE_DIM


class TestEvidenceEncoder:
    """Tests for EvidenceEncoder."""
    
    @pytest.fixture
    def encoder(self):
        return EvidenceEncoder(
            token_dim=FEATURE_DIM,
            hidden_dim=64,
            evidence_dim=32,
            n_layers=2,
            n_heads=4,
            max_cols=16,
        )
    
    def test_output_shape(self, encoder):
        B, d, q = 4, 8, FEATURE_DIM
        tokens = torch.randn(B, d, q)
        col_mask = torch.ones(B, d, dtype=torch.bool)
        
        evidence = encoder(tokens, col_mask)
        
        assert evidence.shape == (B, 32)  # evidence_dim=32
    
    def test_handles_padding(self, encoder):
        B, max_d, q = 4, 16, FEATURE_DIM
        tokens = torch.randn(B, max_d, q)
        
        # Variable number of real columns
        col_mask = torch.zeros(B, max_d, dtype=torch.bool)
        col_mask[0, :5] = True   # 5 columns
        col_mask[1, :10] = True  # 10 columns
        col_mask[2, :3] = True   # 3 columns
        col_mask[3, :8] = True   # 8 columns
        
        evidence = encoder(tokens, col_mask)
        
        assert evidence.shape == (B, 32)
        assert not torch.isnan(evidence).any()
    
    def test_all_padding_handled(self, encoder):
        """Edge case: what if all columns are padding?"""
        B, d, q = 2, 8, FEATURE_DIM
        tokens = torch.randn(B, d, q)
        col_mask = torch.zeros(B, d, dtype=torch.bool)  # All padding
        
        # Should still work (CLS token provides representation)
        evidence = encoder(tokens, col_mask)
        
        assert evidence.shape == (B, 32)
        # Note: output might be degenerate but shouldn't crash
    
    def test_gradients_flow(self, encoder):
        B, d, q = 2, 6, FEATURE_DIM
        tokens = torch.randn(B, d, q, requires_grad=True)
        col_mask = torch.ones(B, d, dtype=torch.bool)
        
        evidence = encoder(tokens, col_mask)
        loss = evidence.sum()
        loss.backward()
        
        assert tokens.grad is not None
        assert not torch.isnan(tokens.grad).any()
    
    def test_deterministic(self, encoder):
        encoder.eval()
        
        B, d, q = 2, 6, FEATURE_DIM
        tokens = torch.randn(B, d, q)
        col_mask = torch.ones(B, d, dtype=torch.bool)
        
        with torch.no_grad():
            e1 = encoder(tokens, col_mask)
            e2 = encoder(tokens, col_mask)
        
        assert torch.equal(e1, e2)
    
    def test_different_batch_sizes(self, encoder):
        encoder.eval()
        
        for B in [1, 2, 8, 16]:
            tokens = torch.randn(B, 8, FEATURE_DIM)
            col_mask = torch.ones(B, 8, dtype=torch.bool)
            
            evidence = encoder(tokens, col_mask)
            assert evidence.shape == (B, 32)
