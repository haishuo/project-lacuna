"""
Tests for lacuna.models.heads
"""

import pytest
import torch
from lacuna.models.heads import GeneratorHead, ClassHead


class TestGeneratorHead:
    """Tests for GeneratorHead."""
    
    def test_linear_head(self):
        head = GeneratorHead(evidence_dim=64, n_generators=6)
        
        evidence = torch.randn(4, 64)
        logits = head(evidence)
        
        assert logits.shape == (4, 6)
    
    def test_mlp_head(self):
        head = GeneratorHead(
            evidence_dim=64,
            n_generators=6,
            hidden_dim=32,
        )
        
        evidence = torch.randn(4, 64)
        logits = head(evidence)
        
        assert logits.shape == (4, 6)
    
    def test_gradients_flow(self):
        head = GeneratorHead(evidence_dim=32, n_generators=6, hidden_dim=16)
        
        evidence = torch.randn(2, 32, requires_grad=True)
        logits = head(evidence)
        loss = logits.sum()
        loss.backward()
        
        assert evidence.grad is not None


class TestClassHead:
    """Tests for ClassHead."""
    
    def test_output_shape(self):
        head = ClassHead(evidence_dim=64, n_classes=3)
        
        evidence = torch.randn(4, 64)
        logits = head(evidence)
        
        assert logits.shape == (4, 3)
    
    def test_with_hidden(self):
        head = ClassHead(evidence_dim=64, n_classes=3, hidden_dim=32)
        
        evidence = torch.randn(4, 64)
        logits = head(evidence)
        
        assert logits.shape == (4, 3)
