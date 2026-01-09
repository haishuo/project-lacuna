"""
Tests for lacuna.models.assembly (LacunaModel)
"""

import pytest
import torch
from lacuna.core.types import TokenBatch
from lacuna.config.schema import LacunaConfig
from lacuna.data.features import FEATURE_DIM
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
        n_layers=2,
        n_heads=4,
        max_cols=16,
    )


@pytest.fixture
def sample_batch():
    """Create sample TokenBatch."""
    B, max_cols, q = 4, 16, FEATURE_DIM
    return TokenBatch(
        tokens=torch.randn(B, max_cols, q),
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
        assert decision.batch_size == 4
    
    def test_get_evidence(self, model, sample_batch):
        evidence = model.get_evidence(sample_batch)
        
        assert evidence.shape == (4, 32)  # evidence_dim=32
    
    def test_gradients_flow(self, model, sample_batch):
        model.train()
        
        # Make tokens require grad
        tokens = sample_batch.tokens.clone().requires_grad_(True)
        batch = TokenBatch(
            tokens=tokens,
            col_mask=sample_batch.col_mask,
            generator_ids=sample_batch.generator_ids,
        )
        
        posterior = model(batch)
        loss = posterior.logits_generator.sum()
        loss.backward()
        
        assert tokens.grad is not None
        assert not torch.isnan(tokens.grad).any()
    
    def test_no_nan_in_output(self, model, sample_batch):
        model.eval()
        
        with torch.no_grad():
            posterior = model(sample_batch)
        
        assert not torch.isnan(posterior.p_generator).any()
        assert not torch.isnan(posterior.p_class).any()
        assert not torch.isnan(posterior.entropy_generator).any()
        assert not torch.isnan(posterior.entropy_class).any()
    
    def test_from_config(self, class_mapping):
        config = LacunaConfig.minimal()
        model = LacunaModel.from_config(config, class_mapping)
        
        assert model.n_generators == 6
        assert model.hidden_dim == 64  # minimal config
    
    def test_variable_batch_sizes(self, model):
        for B in [1, 2, 8]:
            batch = TokenBatch(
                tokens=torch.randn(B, 16, FEATURE_DIM),
                col_mask=torch.ones(B, 16, dtype=torch.bool),
            )
            
            posterior = model(batch)
            assert posterior.p_class.shape == (B, 3)
    
    def test_handles_partial_padding(self, model):
        B = 4
        batch = TokenBatch(
            tokens=torch.randn(B, 16, FEATURE_DIM),
            col_mask=torch.tensor([
                [True]*5 + [False]*11,
                [True]*10 + [False]*6,
                [True]*3 + [False]*13,
                [True]*16,
            ]),
        )
        
        posterior = model(batch)
        
        assert posterior.p_class.shape == (B, 3)
        assert not torch.isnan(posterior.p_class).any()


class TestLacunaModelTraining:
    """Tests for training-related behavior."""
    
    def test_train_eval_modes(self, model, sample_batch):
        # Train mode
        model.train()
        assert model.training
        
        p1 = model(sample_batch)
        
        # Eval mode
        model.eval()
        assert not model.training
        
        with torch.no_grad():
            p2 = model(sample_batch)
        
        # Both should produce valid output
        assert p1.p_class.shape == p2.p_class.shape
    
    def test_parameter_count(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        
        # Should have reasonable number of parameters
        assert n_params > 1000  # Not trivial
        assert n_params < 10_000_000  # Not huge for this config
        
        print(f"Model has {n_params:,} parameters")
