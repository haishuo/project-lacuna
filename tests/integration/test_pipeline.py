"""
Integration test: Full pipeline from data generation to posterior output.

Tests the complete flow:
    Generator -> ObservedDataset -> Tokenization -> Model -> PosteriorResult -> Decision
"""

import pytest
import torch

from lacuna.core.types import ObservedDataset, TokenBatch, PosteriorResult, Decision, MCAR, MAR, MNAR
from lacuna.core.rng import RNGState
from lacuna.config.schema import LacunaConfig
from lacuna.generators import create_minimal_registry
from lacuna.data.features import FEATURE_DIM
from lacuna.data.batching import tokenize_and_batch
from lacuna.models.assembly import LacunaModel


@pytest.fixture
def registry():
    """Create a minimal generator registry."""
    return create_minimal_registry()


@pytest.fixture
def config():
    """Minimal config for testing."""
    return LacunaConfig.minimal()


@pytest.fixture
def model(registry):
    """Create model matching the registry."""
    cfg = LacunaConfig.minimal()
    class_mapping = registry.get_class_mapping()
    return LacunaModel.from_config(cfg, class_mapping)


class TestFullPipeline:
    """Test complete data-to-decision pipeline."""
    
    def test_generator_produces_valid_data(self, registry):
        """Generators produce valid data."""
        rng = RNGState(seed=42)
        
        for gen in registry:
            X, R = gen.sample(rng.spawn(), n=100, d=5)
            
            assert X.shape == (100, 5)
            assert R.shape == (100, 5)
            assert R.dtype == torch.bool
            
            # Some values should be missing (R=False means missing)
            assert (~R).any(), f"Generator {gen.generator_id} produced no missing values"
    
    def test_generator_produces_observed_dataset(self, registry):
        """Generators produce valid ObservedDataset."""
        rng = RNGState(seed=42)
        
        for gen in registry:
            dataset = gen.sample_observed(rng.spawn(), n=100, d=5, dataset_id=f"test_{gen.generator_id}")
            
            assert isinstance(dataset, ObservedDataset)
            assert dataset.n == 100
            assert dataset.d == 5
            
            # Observed values where R=False should be zeroed
            missing_mask = ~dataset.r
            if missing_mask.any():
                assert (dataset.x[missing_mask] == 0).all()
    
    def test_tokenization_produces_valid_batch(self, registry):
        """Tokenization produces valid TokenBatch."""
        rng = RNGState(seed=42)
        max_cols = 16
        
        datasets = []
        gen_ids = []
        
        for gen in registry:
            dataset = gen.sample_observed(rng.spawn(), n=50, d=8, dataset_id=f"test_{gen.generator_id}")
            datasets.append(dataset)
            gen_ids.append(gen.generator_id)
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_cols=max_cols,
            generator_ids=gen_ids,
            class_mapping=registry.get_class_mapping(),
        )
        
        assert isinstance(batch, TokenBatch)
        assert batch.tokens.shape == (6, max_cols, FEATURE_DIM)
        assert batch.col_mask.shape == (6, max_cols)
        assert batch.generator_ids.shape == (6,)
        assert batch.class_ids.shape == (6,)
        
        # No NaN/Inf in tokens
        assert not torch.isnan(batch.tokens).any()
        assert not torch.isinf(batch.tokens).any()
    
    def test_model_forward_produces_valid_posterior(self, registry, model):
        """Model produces valid PosteriorResult."""
        rng = RNGState(seed=42)
        max_cols = 16
        
        # Generate batch
        datasets = []
        gen_ids = []
        
        for gen in registry:
            dataset = gen.sample_observed(rng.spawn(), n=50, d=8, dataset_id=f"test_{gen.generator_id}")
            datasets.append(dataset)
            gen_ids.append(gen.generator_id)
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_cols=max_cols,
            generator_ids=gen_ids,
            class_mapping=registry.get_class_mapping(),
        )
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            posterior = model(batch)
        
        assert isinstance(posterior, PosteriorResult)
        
        # Check shapes
        K = registry.K
        assert posterior.p_generator.shape == (6, K)
        assert posterior.p_class.shape == (6, 3)
        assert posterior.entropy_generator.shape == (6,)
        assert posterior.entropy_class.shape == (6,)
        assert posterior.logits_generator.shape == (6, K)
        
        # Probabilities sum to 1
        assert torch.allclose(posterior.p_generator.sum(dim=-1), torch.ones(6), atol=1e-5)
        assert torch.allclose(posterior.p_class.sum(dim=-1), torch.ones(6), atol=1e-5)
        
        # No NaN/Inf
        assert not torch.isnan(posterior.p_generator).any()
        assert not torch.isnan(posterior.p_class).any()
    
    def test_decision_produces_valid_actions(self, registry, model):
        """Decision rule produces valid actions."""
        rng = RNGState(seed=42)
        max_cols = 16
        
        # Generate batch
        datasets = []
        gen_ids = []
        
        for gen in registry:
            dataset = gen.sample_observed(rng.spawn(), n=50, d=8, dataset_id=f"test_{gen.generator_id}")
            datasets.append(dataset)
            gen_ids.append(gen.generator_id)
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_cols=max_cols,
            generator_ids=gen_ids,
            class_mapping=registry.get_class_mapping(),
        )
        
        # Forward with decision
        model.eval()
        with torch.no_grad():
            posterior, decision = model.forward_with_decision(batch)
        
        assert isinstance(decision, Decision)
        assert decision.action_ids.shape == (6,)
        assert decision.expected_risks.shape == (6,)
        
        # Actions in valid range [0, 1, 2]
        assert (decision.action_ids >= 0).all()
        assert (decision.action_ids <= 2).all()
        
        # Risks are non-negative
        assert (decision.expected_risks >= 0).all()
    
    def test_full_pipeline_end_to_end(self, registry, model):
        """Complete pipeline from generation to decision."""
        rng = RNGState(seed=12345)
        
        # 1. Generate data from each mechanism class
        results = []
        class_names = ["MCAR", "MAR", "MNAR"]
        
        for class_id in [MCAR, MAR, MNAR]:
            # Get a generator for this class
            gen_ids_for_class = registry.generator_ids_for_class(class_id)
            gen = registry[gen_ids_for_class[0]]
            
            # Generate dataset
            dataset = gen.sample_observed(rng.spawn(), n=100, d=10, dataset_id=f"test_{class_id}")
            
            # Create single-item batch
            batch = tokenize_and_batch(
                datasets=[dataset],
                max_cols=16,
                generator_ids=[gen.generator_id],
                class_mapping=registry.get_class_mapping(),
            )
            
            # Get prediction
            model.eval()
            with torch.no_grad():
                posterior, decision = model.forward_with_decision(batch)
            
            results.append({
                "true_class": class_names[class_id],
                "true_class_id": class_id,
                "p_class": posterior.p_class[0].tolist(),
                "predicted_class": posterior.p_class[0].argmax().item(),
                "entropy": posterior.entropy_class[0].item(),
                "action": decision.action_names[decision.action_ids[0].item()],
                "risk": decision.expected_risks[0].item(),
            })
        
        # Verify we got results for all three classes
        assert len(results) == 3
        
        # Each result should have valid structure
        for r in results:
            assert len(r["p_class"]) == 3
            assert sum(r["p_class"]) == pytest.approx(1.0, abs=1e-5)
            assert r["predicted_class"] in [0, 1, 2]
            assert r["action"] in ["Green", "Yellow", "Red"]
