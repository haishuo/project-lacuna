"""
Tests for lacuna.generators.priors
"""

import pytest
import torch
from lacuna.core.rng import RNGState
from lacuna.generators.priors import GeneratorPrior
from lacuna.generators.registry import GeneratorRegistry
from lacuna.generators.params import GeneratorParams
from lacuna.generators.families.mcar import MCARUniform
from lacuna.generators.families.mar import MARLogistic
from lacuna.generators.families.mnar import MNARLogistic


@pytest.fixture
def registry():
    """Create test registry."""
    gens = (
        MCARUniform(0, "mcar-0", GeneratorParams(miss_rate=0.1)),
        MCARUniform(1, "mcar-1", GeneratorParams(miss_rate=0.2)),
        MARLogistic(2, "mar-0", GeneratorParams(alpha0=0, alpha1=1.0)),
        MNARLogistic(3, "mnar-0", GeneratorParams(beta0=0, beta2=1.0)),
    )
    return GeneratorRegistry(gens)


class TestGeneratorPrior:
    """Tests for GeneratorPrior."""
    
    def test_uniform_prior(self, registry):
        prior = GeneratorPrior.uniform(registry)
        
        expected = torch.tensor([0.25, 0.25, 0.25, 0.25])
        assert torch.allclose(prior.weights, expected)
    
    def test_class_balanced_prior(self, registry):
        prior = GeneratorPrior.class_balanced(registry)
        
        # 2 MCAR, 1 MAR, 1 MNAR -> each class gets 1/3
        # MCAR generators each get 1/6, MAR gets 1/3, MNAR gets 1/3
        weights = prior.weights
        
        assert torch.isclose(weights[0], torch.tensor(1/6), atol=1e-6)
        assert torch.isclose(weights[1], torch.tensor(1/6), atol=1e-6)
        assert torch.isclose(weights[2], torch.tensor(1/3), atol=1e-6)
        assert torch.isclose(weights[3], torch.tensor(1/3), atol=1e-6)
    
    def test_custom_prior(self, registry):
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        prior = GeneratorPrior.custom(registry, weights)
        
        assert torch.allclose(prior.weights, weights)
    
    def test_invalid_weights_shape_raises(self, registry):
        weights = torch.tensor([0.5, 0.5])  # Wrong size
        
        with pytest.raises(ValueError, match="shape"):
            GeneratorPrior(registry, weights)
    
    def test_invalid_weights_sum_raises(self, registry):
        weights = torch.tensor([0.1, 0.1, 0.1, 0.1])  # Sums to 0.4
        
        with pytest.raises(Exception):  # ValidationError
            GeneratorPrior(registry, weights)
    
    def test_sample_single(self, registry):
        prior = GeneratorPrior.uniform(registry)
        rng = RNGState(seed=42)
        
        gen_id = prior.sample(rng)
        
        assert 0 <= gen_id < registry.K
    
    def test_sample_batch(self, registry):
        prior = GeneratorPrior.uniform(registry)
        rng = RNGState(seed=42)
        
        gen_ids = prior.sample_batch(rng, n=100)
        
        assert gen_ids.shape == (100,)
        assert gen_ids.dtype == torch.long
        assert (gen_ids >= 0).all()
        assert (gen_ids < registry.K).all()
    
    def test_sample_reproducible(self, registry):
        prior = GeneratorPrior.uniform(registry)
        
        rng1 = RNGState(seed=123)
        rng2 = RNGState(seed=123)
        
        ids1 = prior.sample_batch(rng1, n=50)
        ids2 = prior.sample_batch(rng2, n=50)
        
        assert torch.equal(ids1, ids2)
    
    def test_sample_distribution(self, registry):
        """Verify samples approximately match prior weights."""
        prior = GeneratorPrior.uniform(registry)
        rng = RNGState(seed=42)
        
        ids = prior.sample_batch(rng, n=10000)
        
        for k in range(registry.K):
            empirical_prob = (ids == k).float().mean()
            expected_prob = prior.weights[k]
            assert torch.isclose(empirical_prob, expected_prob, atol=0.02)
