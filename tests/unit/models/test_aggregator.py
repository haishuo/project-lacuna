"""
Tests for lacuna.models.aggregator
"""

import pytest
import torch
from lacuna.models.aggregator import (
    aggregate_to_class_posterior,
    aggregate_to_class_posterior_efficient,
    compute_entropy,
    compute_confidence,
    get_predicted_class,
)


class TestAggregateToClassPosterior:
    """Tests for generator->class aggregation."""
    
    @pytest.fixture
    def class_mapping(self):
        # 6 generators: 2 MCAR, 2 MAR, 2 MNAR
        return torch.tensor([0, 0, 1, 1, 2, 2])
    
    def test_basic_aggregation(self, class_mapping):
        B, K = 2, 6
        # Uniform generator posterior
        p_gen = torch.ones(B, K) / K
        
        p_class = aggregate_to_class_posterior(p_gen, class_mapping)
        
        assert p_class.shape == (B, 3)
        # Each class should get 2/6 = 1/3
        assert torch.allclose(p_class, torch.ones(B, 3) / 3)
    
    def test_sums_to_one(self, class_mapping):
        B, K = 4, 6
        p_gen = torch.softmax(torch.randn(B, K), dim=-1)
        
        p_class = aggregate_to_class_posterior(p_gen, class_mapping)
        
        sums = p_class.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B))
    
    def test_concentrated_posterior(self, class_mapping):
        B, K = 1, 6
        # All probability on generator 0 (MCAR)
        p_gen = torch.zeros(B, K)
        p_gen[0, 0] = 1.0
        
        p_class = aggregate_to_class_posterior(p_gen, class_mapping)
        
        assert p_class[0, 0].item() == 1.0  # MCAR
        assert p_class[0, 1].item() == 0.0  # MAR
        assert p_class[0, 2].item() == 0.0  # MNAR
    
    def test_efficient_matches_basic(self, class_mapping):
        B, K = 8, 6
        p_gen = torch.softmax(torch.randn(B, K), dim=-1)
        
        p_class_basic = aggregate_to_class_posterior(p_gen, class_mapping)
        p_class_efficient = aggregate_to_class_posterior_efficient(p_gen, class_mapping)
        
        assert torch.allclose(p_class_basic, p_class_efficient)


class TestComputeEntropy:
    """Tests for entropy computation."""
    
    def test_uniform_max_entropy(self):
        # Uniform distribution has max entropy
        p = torch.ones(4, 3) / 3
        
        entropy = compute_entropy(p, dim=-1)
        max_entropy = torch.log(torch.tensor(3.0))
        
        assert torch.allclose(entropy, torch.full((4,), max_entropy.item()))
    
    def test_concentrated_zero_entropy(self):
        # Concentrated distribution has zero entropy
        p = torch.zeros(2, 3)
        p[:, 0] = 1.0
        
        entropy = compute_entropy(p, dim=-1)
        
        assert torch.allclose(entropy, torch.zeros(2), atol=1e-6)
    
    def test_non_negative(self):
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        entropy = compute_entropy(p, dim=-1)
        
        assert (entropy >= 0).all()


class TestComputeConfidence:
    """Tests for confidence computation."""
    
    def test_uniform_zero_confidence(self):
        p = torch.ones(4, 3) / 3
        conf = compute_confidence(p)
        
        assert torch.allclose(conf, torch.zeros(4), atol=1e-6)
    
    def test_concentrated_full_confidence(self):
        p = torch.zeros(2, 3)
        p[:, 1] = 1.0
        
        conf = compute_confidence(p)
        
        assert torch.allclose(conf, torch.ones(2), atol=1e-6)
    
    def test_in_range(self):
        p = torch.softmax(torch.randn(10, 3), dim=-1)
        conf = compute_confidence(p)
        
        assert (conf >= 0).all()
        assert (conf <= 1).all()


class TestGetPredictedClass:
    """Tests for get_predicted_class."""
    
    def test_basic(self):
        p_class = torch.tensor([
            [0.8, 0.1, 0.1],  # MCAR
            [0.1, 0.7, 0.2],  # MAR
            [0.1, 0.2, 0.7],  # MNAR
        ])
        
        preds = get_predicted_class(p_class)
        
        assert preds.tolist() == [0, 1, 2]
