"""
Tests for lacuna.models.assembly (row-level)
"""

import pytest
import torch
from lacuna.core.types import TokenBatch
from lacuna.data.tokenization import TOKEN_DIM
from lacuna.models.assembly import LacunaModel


@pytest.fixture
def class_mapping():
    """6 generators: 2 per class."""
    return torch.tensor([0, 0, 1, 1, 2, 2])


@pytest.fixture
def model(class_mapping):
    """Create test model."""
    return LacunaModel(
        n_generators=6,
        class_mapping=class_mapping,
        hidden_dim=64,
        evidence_dim=32,
        n_layers=4,
        n_heads=4,
        max_cols=16,
        max_rows=64,
    )


@pytest.fixture
def sample_batch():
    """Create sample TokenBatch."""
    B, max_rows, max_cols = 4, 64, 16
    return TokenBatch(
        tokens=torch.randn(B, max_rows, max_cols, TOKEN_DIM),
        row_mask=torch.ones(B, max_rows, dtype=torch.bool),
        col_mask=torch.ones(B, max_cols, dtype=torch.bool),
        generator_ids=torch.tensor([0, 2, 4, 1]),
        class_ids=torch.tensor([0, 1, 2, 0]),
    )


class TestLacunaModel:
    """Tests for LacunaModel."""
    
    def test_forward_shapes(self, model, sample_batch):
        posterior = model(sample_batch)
        
        assert posterior.p_generator.shape == (4, 6)
        assert posterior.p_class.shape == (4, 3)
        assert posterior.entropy_generator.shape == (4,)
        assert posterior.entropy_class.shape == (4,)
        assert posterior.logits_generator.shape == (4, 6)
    
    def test_posteriors_valid(self, model, sample_batch):
        posterior = model(sample_batch)
        
        # Generator posterior sums to 1
        gen_sums = posterior.p_generator.sum(dim=-1)
        assert torch.allclose(gen_sums, torch.ones(4))
        
        # Class posterior sums to 1
        class_sums = posterior.p_class.sum(dim=-1)
        assert torch.allclose(class_sums, torch.ones(4))
        
        # All probabilities non-negative
        assert (posterior.p_generator >= 0).all()
        assert (posterior.p_class >= 0).all()
    
    def test_decide(self, model, sample_batch):
        posterior = model(sample_batch)
        decision = model.decide(posterior)
        
        assert decision.batch_size == 4
        assert (decision.action_ids >= 0).all()
        assert (decision.action_ids <= 2).all()
    
    def test_forward_with_decision(self, model, sample_batch):
        posterior, decision = model.forward_with_decision(sample_batch)
        
        assert posterior.p_class.shape == (4, 3)
        assert decision.action_ids.shape == (4,)
    
    def test_gradients_flow(self, model, sample_batch):
        # Make tokens require grad
        tokens = sample_batch.tokens.clone().requires_grad_(True)
        batch = TokenBatch(
            tokens=tokens,
            row_mask=sample_batch.row_mask,
            col_mask=sample_batch.col_mask,
            generator_ids=sample_batch.generator_ids,
            class_ids=sample_batch.class_ids,
        )
        
        posterior = model(batch)
        loss = posterior.logits_generator.sum()
        loss.backward()
        
        assert tokens.grad is not None
    
    def test_variable_batch_sizes(self, model):
        for B in [1, 2, 8]:
            batch = TokenBatch(
                tokens=torch.randn(B, 64, 16, TOKEN_DIM),
                row_mask=torch.ones(B, 64, dtype=torch.bool),
                col_mask=torch.ones(B, 16, dtype=torch.bool),
            )
            
            posterior = model(batch)
            assert posterior.p_class.shape == (B, 3)
