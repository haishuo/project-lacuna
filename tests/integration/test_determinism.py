"""
Integration test: Reproducibility and determinism.

Tests:
- Same seed produces same data
- Same seed produces same model outputs
- Same seed produces same training trajectory
"""

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.config.schema import LacunaConfig
from lacuna.generators import create_minimal_registry
from lacuna.data.batching import tokenize_and_batch
from lacuna.models.assembly import LacunaModel
from lacuna.training.trainer import Trainer, TrainerConfig


@pytest.fixture
def registry():
    """Create a minimal generator registry."""
    return create_minimal_registry()


def generate_batch(registry, batch_size: int, seed: int):
    """Generate batch with specific seed."""
    rng = RNGState(seed=seed)
    
    datasets = []
    gen_ids = []
    
    for i in range(batch_size):
        gen_id = i % registry.K
        gen = registry[gen_id]
        dataset = gen.sample_observed(rng.spawn(), n=50, d=8, dataset_id=f"batch_{i}")
        
        datasets.append(dataset)
        gen_ids.append(gen.generator_id)
    
    return tokenize_and_batch(
        datasets=datasets,
        max_cols=16,
        generator_ids=gen_ids,
        class_mapping=registry.get_class_mapping(),
    )


def seed_torch(seed: int):
    """Seed PyTorch for reproducible model init."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TestDataDeterminism:
    """Test data generation determinism."""
    
    def test_same_seed_same_data(self, registry):
        """Same seed should produce identical data."""
        batch1 = generate_batch(registry, batch_size=4, seed=42)
        batch2 = generate_batch(registry, batch_size=4, seed=42)
        
        assert torch.allclose(batch1.tokens, batch2.tokens)
        assert torch.equal(batch1.col_mask, batch2.col_mask)
        assert torch.equal(batch1.generator_ids, batch2.generator_ids)
    
    def test_different_seed_different_data(self, registry):
        """Different seeds should produce different data."""
        batch1 = generate_batch(registry, batch_size=4, seed=42)
        batch2 = generate_batch(registry, batch_size=4, seed=43)
        
        # Tokens should differ
        assert not torch.allclose(batch1.tokens, batch2.tokens)


class TestModelDeterminism:
    """Test model initialization and forward pass determinism."""
    
    def test_same_seed_same_init(self, registry):
        """Same seed should produce identical model initialization."""
        cfg = LacunaConfig.minimal()
        class_mapping = registry.get_class_mapping()
        
        seed_torch(42)
        model1 = LacunaModel.from_config(cfg, class_mapping)
        
        seed_torch(42)
        model2 = LacunaModel.from_config(cfg, class_mapping)
        
        # All parameters should match
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Parameter {n1} differs"
    
    def test_same_input_same_output(self, registry):
        """Same model + same input should produce same output."""
        cfg = LacunaConfig.minimal()
        class_mapping = registry.get_class_mapping()
        
        seed_torch(42)
        model = LacunaModel.from_config(cfg, class_mapping)
        model.eval()
        
        batch = generate_batch(registry, batch_size=4, seed=99)
        
        with torch.no_grad():
            out1 = model(batch)
            out2 = model(batch)
        
        assert torch.equal(out1.logits_generator, out2.logits_generator)
        assert torch.equal(out1.p_class, out2.p_class)


class TestTrainingDeterminism:
    """Test training determinism."""
    
    def test_same_seed_same_trajectory(self, registry):
        """Same seed should produce identical training trajectory."""
        cfg = LacunaConfig.minimal()
        class_mapping = registry.get_class_mapping()
        
        def train_model(seed):
            seed_torch(seed)
            model = LacunaModel.from_config(cfg, class_mapping)
            
            trainer_config = TrainerConfig(
                lr=1e-3,
                epochs=2,
                warmup_steps=5,
            )
            trainer = Trainer(model, trainer_config, device="cpu")
            
            # Create deterministic data loader
            class DeterministicLoader:
                def __init__(self, data_seed):
                    self.data_seed = data_seed
                
                def __iter__(self):
                    for i in range(5):
                        yield generate_batch(registry, batch_size=4, seed=self.data_seed + i)
                
                def __len__(self):
                    return 5
            
            loader = DeterministicLoader(data_seed=100)
            result = trainer.fit(loader)
            
            return model, result
        
        model1, result1 = train_model(42)
        model2, result2 = train_model(42)
        
        # Final parameters should match
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2, atol=1e-6), f"Parameter {n1} differs after training"
        
        # Results should match
        assert result1["total_steps"] == result2["total_steps"]
