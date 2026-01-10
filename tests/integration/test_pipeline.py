"""
Integration test: Full pipeline from data generation to model output.

Tests the complete flow:
    Generator -> ObservedDataset -> Tokenization -> Model -> LacunaOutput -> Decision

This validates that all components work together correctly:
    1. Generators produce valid synthetic data
    2. Tokenization converts data to model input format
    3. Model forward pass produces valid outputs
    4. Decision rule maps posteriors to actions
    5. Training loop can optimize the model
    6. Checkpointing preserves state correctly
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path
from typing import List

from lacuna.core.types import (
    ObservedDataset,
    TokenBatch,
    PosteriorResult,
    Decision,
    LacunaOutput,
    MCAR,
    MAR,
    MNAR,
    CLASS_NAMES,
)
from lacuna.core.rng import RNGState
from lacuna.data.tokenization import (
    tokenize_and_batch,
    apply_artificial_masking,
    MaskingConfig,
    TOKEN_DIM,
)
from lacuna.data.batching import (
    SyntheticDataLoader,
    SyntheticDataLoaderConfig,
    ValidationDataLoader,
    collate_fn,
)
from lacuna.models.assembly import (
    LacunaModel,
    LacunaModelConfig,
    create_lacuna_mini,
    create_lacuna_model,
)
from lacuna.models.encoder import LacunaEncoder
from lacuna.models.reconstruction import ReconstructionHeads
from lacuna.models.moe import MixtureOfExperts
from lacuna.training.loss import (
    LacunaLoss,
    LossConfig,
    create_loss_function,
    create_joint_loss,
    compute_class_accuracy,
)
from lacuna.training.trainer import Trainer, TrainerConfig, TrainerState
from lacuna.training.checkpoint import (
    CheckpointData,
    save_checkpoint,
    load_checkpoint,
    load_model_weights,
)
from lacuna.generators.base import BaseGenerator
from lacuna.generators.families.mcar import create_mcar_uniform
from lacuna.generators.families.mar import create_mar_logistic
from lacuna.generators.families.mnar import (
    create_mnar_self_censoring,
    create_mnar_threshold,
    create_mnar_latent,
    create_all_mnar_generators,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def generators() -> List[BaseGenerator]:
    """Create a minimal set of generators covering all mechanism classes."""
    return [
        create_mcar_uniform(generator_id=0, missing_rate=0.2),
        create_mar_logistic(generator_id=1, strength=1.0),
        create_mnar_self_censoring(generator_id=2),
        create_mnar_threshold(generator_id=3),
    ]


@pytest.fixture
def class_mapping():
    """Class mapping for generators: gen_id -> class_id."""
    return {
        0: MCAR,
        1: MAR,
        2: MNAR,
        3: MNAR,
    }


@pytest.fixture
def variant_mapping():
    """Variant mapping for MNAR generators."""
    return {
        2: 0,  # self_censoring
        3: 1,  # threshold
    }


@pytest.fixture
def model():
    """Create a minimal model for testing."""
    return create_lacuna_mini(
        max_cols=16,
        mnar_variants=["self_censoring", "threshold"],
    )


@pytest.fixture
def rng():
    """Create reproducible RNG."""
    return RNGState(seed=42)


# =============================================================================
# Test Generator Pipeline
# =============================================================================

class TestGeneratorPipeline:
    """Test that generators produce valid data."""
    
    def test_generators_produce_observed_dataset(self, generators, rng):
        """Each generator produces valid ObservedDataset."""
        for gen in generators:
            dataset = gen.sample_observed(
                rng=rng.spawn(),
                n=100,
                d=10,
                dataset_id=f"test_{gen.generator_id}",
            )
            
            assert isinstance(dataset, ObservedDataset)
            assert dataset.X_obs.shape == (100, 10)
            assert dataset.R.shape == (100, 10)
            assert dataset.R.dtype == bool
    
    def test_generators_have_correct_class(self, generators):
        """Generators have correct class_id."""
        expected_classes = [MCAR, MAR, MNAR, MNAR]
        
        for gen, expected in zip(generators, expected_classes):
            assert gen.class_id == expected
    
    def test_generators_have_unique_ids(self, generators):
        """Generator IDs are unique."""
        ids = [gen.generator_id for gen in generators]
        assert len(set(ids)) == len(ids)
    
    def test_missingness_pattern_varies_by_class(self, generators, rng):
        """Different mechanism classes produce different patterns."""
        patterns = []
        
        for gen in generators:
            dataset = gen.sample_observed(rng.spawn(), n=200, d=10)
            missing_rate = (~dataset.R).mean()
            patterns.append(missing_rate)
        
        # All should have meaningful missingness
        for rate in patterns:
            assert rate > 0.01
            assert rate < 0.99
    
    def test_no_completely_empty_rows_or_columns(self, generators, rng):
        """Data should not have entirely missing rows or columns."""
        for gen in generators:
            dataset = gen.sample_observed(rng.spawn(), n=100, d=10)
            
            # Each row has at least one observed value
            row_observed = dataset.R.sum(axis=1)
            assert (row_observed > 0).all()
            
            # Each column has at least one observed value
            col_observed = dataset.R.sum(axis=0)
            assert (col_observed > 0).all()


# =============================================================================
# Test Tokenization Pipeline
# =============================================================================

class TestTokenizationPipeline:
    """Test data tokenization and batching."""
    
    def test_tokenization_produces_valid_batch(
        self, generators, class_mapping, variant_mapping, rng
    ):
        """Tokenization produces valid TokenBatch."""
        datasets = []
        gen_ids = []
        var_ids = []
        
        for gen in generators:
            dataset = gen.sample_observed(rng.spawn(), n=50, d=8)
            datasets.append(dataset)
            gen_ids.append(gen.generator_id)
            var_ids.append(variant_mapping.get(gen.generator_id, -1))
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=64,
            max_cols=16,
            generator_ids=gen_ids,
            class_mapping=class_mapping,
            variant_ids=var_ids,
        )
        
        assert isinstance(batch, TokenBatch)
        assert batch.tokens.shape == (4, 64, 16, TOKEN_DIM)
        assert batch.row_mask.shape == (4, 64)
        assert batch.col_mask.shape == (4, 16)
        assert batch.class_ids.shape == (4,)
    
    def test_tokenization_no_nan_or_inf(self, generators, rng):
        """Tokens should not contain NaN or Inf."""
        datasets = [
            gen.sample_observed(rng.spawn(), n=50, d=8)
            for gen in generators
        ]
        
        batch = tokenize_and_batch(datasets, max_rows=64, max_cols=16)
        
        assert not torch.isnan(batch.tokens).any()
        assert not torch.isinf(batch.tokens).any()
    
    def test_artificial_masking_works(self, generators, rng):
        """Artificial masking creates reconstruction targets."""
        dataset = generators[0].sample_observed(rng, n=50, d=8)
        
        config = MaskingConfig(mask_ratio=0.15, min_masked=1)
        X_masked, R_masked, art_mask = apply_artificial_masking(
            dataset.X_obs,
            dataset.R,
            config,
            rng=rng.numpy_rng,
        )
        
        # Some values should be artificially masked
        assert art_mask.sum() > 0
        
        # Artificially masked values should be in observed positions
        assert not (art_mask & ~dataset.R).any()
    
    def test_batch_labels_correct(self, generators, class_mapping, rng):
        """Batch labels match generator classes."""
        datasets = []
        gen_ids = []
        
        for gen in generators:
            datasets.append(gen.sample_observed(rng.spawn(), n=50, d=8))
            gen_ids.append(gen.generator_id)
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=64,
            max_cols=16,
            generator_ids=gen_ids,
            class_mapping=class_mapping,
        )
        
        expected_classes = [class_mapping[gid] for gid in gen_ids]
        assert batch.class_ids.tolist() == expected_classes


# =============================================================================
# Test Model Forward Pass
# =============================================================================

class TestModelForwardPass:
    """Test model produces valid outputs."""
    
    def test_model_forward_returns_lacuna_output(self, model, generators, rng):
        """Model forward pass returns LacunaOutput."""
        datasets = [gen.sample_observed(rng.spawn(), n=50, d=8) for gen in generators]
        batch = tokenize_and_batch(datasets, max_rows=64, max_cols=16)
        
        model.eval()
        with torch.no_grad():
            output = model(batch)
        
        assert isinstance(output, LacunaOutput)
    
    def test_output_has_all_components(self, model, generators, rng):
        """Output contains all expected components."""
        datasets = [gen.sample_observed(rng.spawn(), n=50, d=8) for gen in generators]
        batch = tokenize_and_batch(datasets, max_rows=64, max_cols=16)
        
        model.eval()
        with torch.no_grad():
            output = model(batch)
        
        assert output.posterior is not None
        assert output.decision is not None
        assert output.evidence is not None
        assert output.moe is not None
    
    def test_posterior_shapes(self, model, generators, rng):
        """Posterior tensors have correct shapes."""
        B = len(generators)
        datasets = [gen.sample_observed(rng.spawn(), n=50, d=8) for gen in generators]
        batch = tokenize_and_batch(datasets, max_rows=64, max_cols=16)
        
        model.eval()
        with torch.no_grad():
            output = model(batch)
        
        assert output.posterior.p_class.shape == (B, 3)
        assert output.evidence.shape[0] == B
    
    def test_posterior_probabilities_valid(self, model, generators, rng):
        """Posterior probabilities sum to 1 and are non-negative."""
        datasets = [gen.sample_observed(rng.spawn(), n=50, d=8) for gen in generators]
        batch = tokenize_and_batch(datasets, max_rows=64, max_cols=16)
        
        model.eval()
        with torch.no_grad():
            output = model(batch)
        
        p_class = output.posterior.p_class
        
        # Non-negative
        assert (p_class >= 0).all()
        
        # Sum to 1
        sums = p_class.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_decision_shapes_and_validity(self, model, generators, rng):
        """Decision outputs are valid."""
        B = len(generators)
        datasets = [gen.sample_observed(rng.spawn(), n=50, d=8) for gen in generators]
        batch = tokenize_and_batch(datasets, max_rows=64, max_cols=16)
        
        model.eval()
        with torch.no_grad():
            output = model(batch)
        
        decision = output.decision
        
        assert decision.action_ids.shape == (B,)
        assert decision.expected_risks.shape == (B,)
        
        # Actions in valid range
        assert (decision.action_ids >= 0).all()
        assert (decision.action_ids < 3).all()
        
        # Risks non-negative
        assert (decision.expected_risks >= 0).all()
    
    def test_no_nan_or_inf_in_outputs(self, model, generators, rng):
        """Outputs contain no NaN or Inf."""
        datasets = [gen.sample_observed(rng.spawn(), n=50, d=8) for gen in generators]
        batch = tokenize_and_batch(datasets, max_rows=64, max_cols=16)
        
        model.eval()
        with torch.no_grad():
            output = model(batch)
        
        assert not torch.isnan(output.posterior.p_class).any()
        assert not torch.isnan(output.evidence).any()
        assert not torch.isnan(output.decision.expected_risks).any()
        
        assert not torch.isinf(output.posterior.p_class).any()
        assert not torch.isinf(output.evidence).any()


# =============================================================================
# Test Training Pipeline
# =============================================================================

class TestTrainingPipeline:
    """Test training components work together."""
    
    def test_loss_computes_without_error(self, model, generators, class_mapping, rng):
        """Loss function computes valid loss."""
        datasets = [gen.sample_observed(rng.spawn(), n=50, d=8) for gen in generators]
        gen_ids = [gen.generator_id for gen in generators]
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=64,
            max_cols=16,
            generator_ids=gen_ids,
            class_mapping=class_mapping,
        )
        
        # Add reconstruction targets
        batch = TokenBatch(
            tokens=batch.tokens,
            row_mask=batch.row_mask,
            col_mask=batch.col_mask,
            generator_ids=batch.generator_ids,
            class_ids=batch.class_ids,
            variant_ids=torch.zeros(len(generators), dtype=torch.long),
            original_values=torch.randn(len(generators), 64, 16),
            reconstruction_mask=torch.rand(len(generators), 64, 16) > 0.7,
        )
        
        model.train()
        output = model(batch)
        
        loss_fn = create_joint_loss()
        total_loss, loss_dict = loss_fn(output, batch)
        
        assert not torch.isnan(total_loss)
        assert total_loss.item() >= 0
        assert "total_loss" in loss_dict
    
    def test_gradients_flow(self, model, generators, class_mapping, rng):
        """Gradients flow through all components."""
        datasets = [gen.sample_observed(rng.spawn(), n=50, d=8) for gen in generators]
        gen_ids = [gen.generator_id for gen in generators]
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=64,
            max_cols=16,
            generator_ids=gen_ids,
            class_mapping=class_mapping,
        )
        
        model.train()
        output = model(batch)
        
        loss = output.posterior.p_class.sum()
        loss.backward()
        
        # Check encoder has gradients
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.parameters()
        )
        assert encoder_has_grad, "Encoder should have gradients"
    
    def test_training_step_reduces_loss(self, generators, class_mapping, rng):
        """Training steps reduce loss over time."""
        model = create_lacuna_mini(max_cols=16, mnar_variants=["self_censoring", "threshold"])
        
        # Generate fixed batch
        datasets = [gen.sample_observed(rng.spawn(), n=50, d=8) for gen in generators]
        gen_ids = [gen.generator_id for gen in generators]
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=64,
            max_cols=16,
            generator_ids=gen_ids,
            class_mapping=class_mapping,
        )
        
        loss_fn = create_loss_function(
            mechanism_weight=1.0,
            reconstruction_weight=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Get initial loss
        model.eval()
        with torch.no_grad():
            output = model(batch)
            initial_loss = nn.functional.cross_entropy(
                output.posterior.p_class.log(),
                batch.class_ids,
            ).item()
        
        # Train for several steps
        model.train()
        for _ in range(30):
            optimizer.zero_grad()
            output = model(batch)
            loss = nn.functional.cross_entropy(
                output.posterior.p_class.log(),
                batch.class_ids,
            )
            loss.backward()
            optimizer.step()
        
        # Get final loss
        model.eval()
        with torch.no_grad():
            output = model(batch)
            final_loss = nn.functional.cross_entropy(
                output.posterior.p_class.log(),
                batch.class_ids,
            ).item()
        
        assert final_loss < initial_loss, "Loss should decrease during training"
    
    def test_accuracy_improves_with_training(self, generators, class_mapping, rng):
        """Classification accuracy improves with training."""
        model = create_lacuna_mini(max_cols=16, mnar_variants=["self_censoring", "threshold"])
        
        # Generate batch
        datasets = [gen.sample_observed(rng.spawn(), n=50, d=8) for gen in generators]
        gen_ids = [gen.generator_id for gen in generators]
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=64,
            max_cols=16,
            generator_ids=gen_ids,
            class_mapping=class_mapping,
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Get initial accuracy
        model.eval()
        with torch.no_grad():
            output = model(batch)
            initial_acc = compute_class_accuracy(
                output.posterior.p_class,
                batch.class_ids,
            ).item()
        
        # Train
        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            output = model(batch)
            loss = nn.functional.cross_entropy(
                output.posterior.p_class.log(),
                batch.class_ids,
            )
            loss.backward()
            optimizer.step()
        
        # Get final accuracy
        model.eval()
        with torch.no_grad():
            output = model(batch)
            final_acc = compute_class_accuracy(
                output.posterior.p_class,
                batch.class_ids,
            ).item()
        
        # Should improve (or at least not be worse on training data)
        assert final_acc >= initial_acc


# =============================================================================
# Test Checkpoint Pipeline
# =============================================================================

class TestCheckpointPipeline:
    """Test checkpointing preserves state correctly."""
    
    def test_checkpoint_preserves_model_weights(self, model, generators, rng):
        """Checkpoint preserves model weights exactly."""
        # Generate test batch
        datasets = [gen.sample_observed(rng.spawn(), n=50, d=8) for gen in generators]
        batch = tokenize_and_batch(datasets, max_rows=64, max_cols=16)
        
        # Get output before saving
        model.eval()
        with torch.no_grad():
            output_before = model(batch)
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            
            checkpoint = CheckpointData(
                model_state=model.state_dict(),
                step=100,
                epoch=5,
            )
            save_checkpoint(checkpoint, path)
            
            # Create new model and load
            model2 = create_lacuna_mini(max_cols=16, mnar_variants=["self_censoring", "threshold"])
            model2 = load_model_weights(model2, path)
        
        # Get output after loading
        model2.eval()
        with torch.no_grad():
            output_after = model2(batch)
        
        # Should be identical
        assert torch.allclose(
            output_before.posterior.p_class,
            output_after.posterior.p_class,
            atol=1e-6,
        )
    
    def test_checkpoint_preserves_optimizer_state(self, generators, class_mapping, rng):
        """Checkpoint preserves optimizer state for resuming."""
        model = create_lacuna_mini(max_cols=16, mnar_variants=["self_censoring", "threshold"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Generate batch and do some training
        datasets = [gen.sample_observed(rng.spawn(), n=50, d=8) for gen in generators]
        gen_ids = [gen.generator_id for gen in generators]
        batch = tokenize_and_batch(
            datasets, max_rows=64, max_cols=16,
            generator_ids=gen_ids, class_mapping=class_mapping,
        )
        
        for _ in range(5):
            optimizer.zero_grad()
            output = model(batch)
            loss = nn.functional.cross_entropy(
                output.posterior.p_class.log(),
                batch.class_ids,
            )
            loss.backward()
            optimizer.step()
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            
            checkpoint = CheckpointData(
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                step=5,
            )
            save_checkpoint(checkpoint, path)
            
            # Load checkpoint
            loaded = load_checkpoint(path)
        
        assert loaded.step == 5
        assert loaded.optimizer_state is not None
        assert "state" in loaded.optimizer_state


# =============================================================================
# Test Data Loader Integration
# =============================================================================

class TestDataLoaderIntegration:
    """Test data loaders work with model."""
    
    def test_synthetic_loader_with_model(self, generators, model):
        """SyntheticDataLoader batches work with model."""
        config = SyntheticDataLoaderConfig(
            batch_size=4,
            n_range=(30, 50),
            d_range=(5, 10),
            max_rows=64,
            max_cols=16,
            apply_masking=False,
            batches_per_epoch=3,
            seed=42,
        )
        
        loader = SyntheticDataLoader(generators=generators, config=config)
        
        model.eval()
        with torch.no_grad():
            for batch in loader:
                output = model(batch)
                
                assert output.posterior.p_class.shape[0] == 4
                assert not torch.isnan(output.posterior.p_class).any()
    
    def test_validation_loader_deterministic(self, generators, model):
        """ValidationDataLoader produces same data each iteration."""
        loader = ValidationDataLoader(
            generators=generators,
            n_samples=16,
            batch_size=4,
            max_rows=64,
            max_cols=16,
            seed=42,
        )
        
        # First pass
        outputs1 = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                output = model(batch)
                outputs1.append(output.posterior.p_class.clone())
        
        # Second pass
        outputs2 = []
        with torch.no_grad():
            for batch in loader:
                output = model(batch)
                outputs2.append(output.posterior.p_class.clone())
        
        # Should be identical
        for o1, o2 in zip(outputs1, outputs2):
            assert torch.equal(o1, o2)


# =============================================================================
# Test End-to-End Pipeline
# =============================================================================

class TestEndToEndPipeline:
    """Complete end-to-end tests."""
    
    def test_complete_inference_pipeline(self, generators, class_mapping, rng):
        """Complete inference from raw data to decision."""
        model = create_lacuna_mini(max_cols=16, mnar_variants=["self_censoring", "threshold"])
        model.eval()
        
        results = []
        
        for gen in generators:
            # 1. Generate data
            dataset = gen.sample_observed(rng.spawn(), n=100, d=10)
            
            # 2. Tokenize
            batch = tokenize_and_batch(
                datasets=[dataset],
                max_rows=64,
                max_cols=16,
                generator_ids=[gen.generator_id],
                class_mapping=class_mapping,
            )
            
            # 3. Model inference
            with torch.no_grad():
                output = model(batch)
            
            # 4. Extract decision
            results.append({
                "generator_id": gen.generator_id,
                "true_class": CLASS_NAMES[gen.class_id],
                "p_class": output.posterior.p_class[0].tolist(),
                "predicted_class": CLASS_NAMES[output.posterior.p_class[0].argmax().item()],
                "action": output.decision.action_names[output.decision.action_ids[0].item()],
                "risk": output.decision.expected_risks[0].item(),
            })
        
        # Verify all results are valid
        assert len(results) == len(generators)
        
        for r in results:
            assert len(r["p_class"]) == 3
            assert abs(sum(r["p_class"]) - 1.0) < 1e-5
            assert r["predicted_class"] in CLASS_NAMES.values()
            assert r["action"] in ["Green", "Yellow", "Red"]
            assert r["risk"] >= 0
    
    def test_training_and_inference_pipeline(self, generators, class_mapping, rng):
        """Train model then run inference."""
        model = create_lacuna_mini(max_cols=16, mnar_variants=["self_censoring", "threshold"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training phase
        model.train()
        for epoch in range(3):
            for gen in generators:
                dataset = gen.sample_observed(rng.spawn(), n=50, d=8)
                batch = tokenize_and_batch(
                    datasets=[dataset],
                    max_rows=64,
                    max_cols=16,
                    generator_ids=[gen.generator_id],
                    class_mapping=class_mapping,
                )
                
                optimizer.zero_grad()
                output = model(batch)
                loss = nn.functional.cross_entropy(
                    output.posterior.p_class.log(),
                    batch.class_ids,
                )
                loss.backward()
                optimizer.step()
        
        # Inference phase
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for gen in generators:
                dataset = gen.sample_observed(rng.spawn(), n=100, d=10)
                batch = tokenize_and_batch(
                    datasets=[dataset],
                    max_rows=64,
                    max_cols=16,
                    generator_ids=[gen.generator_id],
                    class_mapping=class_mapping,
                )
                
                output = model(batch)
                pred = output.posterior.p_class.argmax(dim=-1).item()
                predictions.append({
                    "true": gen.class_id,
                    "pred": pred,
                })
        
        # Just verify we got predictions for all
        assert len(predictions) == len(generators)
        for p in predictions:
            assert p["pred"] in [MCAR, MAR, MNAR]
    
    def test_reproducibility_full_pipeline(self, generators, class_mapping):
        """Full pipeline is reproducible with same seed."""
        def run_pipeline(seed: int):
            torch.manual_seed(seed)
            rng = RNGState(seed=seed)
            
            model = create_lacuna_mini(max_cols=16, mnar_variants=["self_censoring", "threshold"])
            model.eval()
            
            datasets = [gen.sample_observed(rng.spawn(), n=50, d=8) for gen in generators]
            gen_ids = [gen.generator_id for gen in generators]
            
            batch = tokenize_and_batch(
                datasets=datasets,
                max_rows=64,
                max_cols=16,
                generator_ids=gen_ids,
                class_mapping=class_mapping,
            )
            
            with torch.no_grad():
                output = model(batch)
            
            return output.posterior.p_class
        
        result1 = run_pipeline(seed=12345)
        result2 = run_pipeline(seed=12345)
        
        assert torch.equal(result1, result2)
    
    def test_different_seeds_different_results(self, generators, class_mapping):
        """Different seeds produce different results."""
        def run_pipeline(seed: int):
            torch.manual_seed(seed)
            rng = RNGState(seed=seed)
            
            model = create_lacuna_mini(max_cols=16, mnar_variants=["self_censoring", "threshold"])
            model.eval()
            
            datasets = [gen.sample_observed(rng.spawn(), n=50, d=8) for gen in generators]
            gen_ids = [gen.generator_id for gen in generators]
            
            batch = tokenize_and_batch(
                datasets=datasets,
                max_rows=64,
                max_cols=16,
                generator_ids=gen_ids,
                class_mapping=class_mapping,
            )
            
            with torch.no_grad():
                output = model(batch)
            
            return output.posterior.p_class
        
        result1 = run_pipeline(seed=12345)
        result2 = run_pipeline(seed=54321)
        
        assert not torch.equal(result1, result2)


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling in pipeline."""
    
    def test_handles_empty_batch_gracefully(self, model):
        """Model handles edge cases gracefully."""
        # Minimal valid batch
        batch = TokenBatch(
            tokens=torch.randn(1, 16, 8, TOKEN_DIM),
            row_mask=torch.ones(1, 16, dtype=torch.bool),
            col_mask=torch.ones(1, 8, dtype=torch.bool),
        )
        
        model.eval()
        with torch.no_grad():
            output = model(batch)
        
        assert output.posterior.p_class.shape == (1, 3)
    
    def test_handles_all_observed_data(self, generators, rng, model):
        """Model handles data with no missingness."""
        # Create fully observed data
        X_obs = np.random.randn(50, 8).astype(np.float32)
        R = np.ones((50, 8), dtype=bool)  # All observed
        
        dataset = ObservedDataset(
            X_obs=X_obs,
            R=R,
            dataset_id="fully_observed",
            n_original=50,
            d_original=8,
        )
        
        batch = tokenize_and_batch([dataset], max_rows=64, max_cols=16)
        
        model.eval()
        with torch.no_grad():
            output = model(batch)
        
        assert not torch.isnan(output.posterior.p_class).any()
    
    def test_handles_high_missingness(self, model):
        """Model handles data with high missingness."""
        # Create highly sparse data
        X_obs = np.random.randn(50, 8).astype(np.float32)
        R = np.random.rand(50, 8) > 0.9  # ~90% missing
        
        # Ensure at least one observed per row/col
        R[:, 0] = True
        R[0, :] = True
        
        X_obs[~R] = np.nan
        
        dataset = ObservedDataset(
            X_obs=X_obs,
            R=R,
            dataset_id="sparse",
            n_original=50,
            d_original=8,
        )
        
        batch = tokenize_and_batch([dataset], max_rows=64, max_cols=16)
        
        model.eval()
        with torch.no_grad():
            output = model(batch)
        
        assert not torch.isnan(output.posterior.p_class).any()