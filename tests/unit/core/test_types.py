"""
Tests for lacuna.core.types

Tests the core data structures:
    - ObservedDataset: Wrapper for observed data with missingness mask
    - TokenBatch: Batched tokenized datasets for model input
    - ReconstructionResult: Output from reconstruction heads
    - MoEOutput: Output from Mixture of Experts layer
    - PosteriorResult: Model output posteriors
    - Decision: Bayes-optimal decision output
    - LacunaOutput: Complete model output container
"""

import pytest
import torch
import numpy as np

from lacuna.core.types import (
    # Constants
    MCAR,
    MAR,
    MNAR,
    CLASS_NAMES,
    MNAR_SELF_CENSORING,
    MNAR_THRESHOLD,
    MNAR_LATENT,
    MNAR_MIXTURE,
    # Data structures
    ObservedDataset,
    TokenBatch,
    ReconstructionResult,
    MoEOutput,
    PosteriorResult,
    Decision,
    LacunaOutput,
)


# =============================================================================
# Test Constants
# =============================================================================

class TestConstants:
    """Tests for mechanism class constants."""
    
    def test_class_ids_are_distinct(self):
        """MCAR, MAR, MNAR should have distinct IDs."""
        assert MCAR != MAR
        assert MAR != MNAR
        assert MCAR != MNAR
    
    def test_class_ids_are_sequential(self):
        """Class IDs should be 0, 1, 2."""
        assert MCAR == 0
        assert MAR == 1
        assert MNAR == 2
    
    def test_class_names_match_ids(self):
        """CLASS_NAMES should map IDs to names."""
        assert CLASS_NAMES[MCAR] == "MCAR"
        assert CLASS_NAMES[MAR] == "MAR"
        assert CLASS_NAMES[MNAR] == "MNAR"
    
    def test_mnar_variant_ids_are_distinct(self):
        """MNAR variant IDs should be distinct."""
        variants = [MNAR_SELF_CENSORING, MNAR_THRESHOLD, MNAR_LATENT, MNAR_MIXTURE]
        assert len(set(variants)) == len(variants)
    
    def test_mnar_variant_ids_are_sequential(self):
        """MNAR variant IDs should be 0, 1, 2, 3."""
        assert MNAR_SELF_CENSORING == 0
        assert MNAR_THRESHOLD == 1
        assert MNAR_LATENT == 2
        assert MNAR_MIXTURE == 3


# =============================================================================
# Test ObservedDataset
# =============================================================================

class TestObservedDataset:
    """Tests for ObservedDataset."""
    
    def test_valid_construction(self):
        """Test basic construction with valid inputs."""
        n, d = 100, 10
        X_obs = np.random.randn(n, d).astype(np.float32)
        R = np.random.rand(n, d) > 0.2  # ~20% missing
        
        # Set missing values to NaN
        X_obs[~R] = np.nan
        
        dataset = ObservedDataset(
            X_obs=X_obs,
            R=R,
            dataset_id="test_dataset",
            n_original=n,
            d_original=d,
        )
        
        assert dataset.n == n
        assert dataset.d == d
        assert dataset.dataset_id == "test_dataset"
        assert dataset.n_original == n
        assert dataset.d_original == d
    
    def test_shape_properties(self):
        """Test n and d property accessors."""
        X_obs = np.random.randn(50, 8).astype(np.float32)
        R = np.ones((50, 8), dtype=bool)
        
        dataset = ObservedDataset(X_obs=X_obs, R=R)
        
        assert dataset.n == 50
        assert dataset.d == 8
    
    def test_missing_values_are_nan(self):
        """Test that missing values are properly represented as NaN."""
        X_obs = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
        R = np.array([[True, False, True], [True, True, False]])
        
        dataset = ObservedDataset(X_obs=X_obs, R=R)
        
        # Check that NaN positions match R=False
        assert np.isnan(dataset.X_obs[0, 1])
        assert np.isnan(dataset.X_obs[1, 2])
        assert not np.isnan(dataset.X_obs[0, 0])
    
    def test_optional_fields_default_to_none(self):
        """Test that optional fields have sensible defaults."""
        X_obs = np.random.randn(10, 5).astype(np.float32)
        R = np.ones((10, 5), dtype=bool)
        
        dataset = ObservedDataset(X_obs=X_obs, R=R)
        
        # dataset_id should default to None or empty string
        # n_original and d_original should match shape if not provided
        assert dataset.n_original is None or dataset.n_original == 10
        assert dataset.d_original is None or dataset.d_original == 5
    
    def test_immutability(self):
        """Test that ObservedDataset is frozen (immutable)."""
        X_obs = np.random.randn(10, 5).astype(np.float32)
        R = np.ones((10, 5), dtype=bool)
        
        dataset = ObservedDataset(X_obs=X_obs, R=R, dataset_id="test")
        
        # Attempting to modify should raise an error
        with pytest.raises((AttributeError, TypeError)):
            dataset.dataset_id = "modified"


# =============================================================================
# Test TokenBatch
# =============================================================================

class TestTokenBatch:
    """Tests for TokenBatch."""
    
    def test_valid_construction(self):
        """Test basic construction with valid inputs."""
        B, max_rows, max_cols, token_dim = 4, 64, 16, 4
        
        batch = TokenBatch(
            tokens=torch.randn(B, max_rows, max_cols, token_dim),
            row_mask=torch.ones(B, max_rows, dtype=torch.bool),
            col_mask=torch.ones(B, max_cols, dtype=torch.bool),
        )
        
        assert batch.batch_size == B
        assert batch.max_rows == max_rows
        assert batch.max_cols == max_cols
        assert batch.token_dim == token_dim
    
    def test_with_generator_and_class_labels(self):
        """Test construction with generator and class IDs."""
        B, max_rows, max_cols, token_dim = 4, 64, 16, 4
        
        batch = TokenBatch(
            tokens=torch.randn(B, max_rows, max_cols, token_dim),
            row_mask=torch.ones(B, max_rows, dtype=torch.bool),
            col_mask=torch.ones(B, max_cols, dtype=torch.bool),
            generator_ids=torch.tensor([0, 1, 2, 3]),
            class_ids=torch.tensor([0, 0, 1, 2]),
        )
        
        assert batch.generator_ids is not None
        assert batch.generator_ids.shape == (B,)
        assert batch.class_ids is not None
        assert batch.class_ids.shape == (B,)
    
    def test_with_variant_ids(self):
        """Test construction with MNAR variant IDs."""
        B, max_rows, max_cols, token_dim = 4, 64, 16, 4
        
        batch = TokenBatch(
            tokens=torch.randn(B, max_rows, max_cols, token_dim),
            row_mask=torch.ones(B, max_rows, dtype=torch.bool),
            col_mask=torch.ones(B, max_cols, dtype=torch.bool),
            variant_ids=torch.tensor([-1, -1, 0, 1]),  # -1 for non-MNAR
        )
        
        assert batch.variant_ids is not None
        assert batch.variant_ids.shape == (B,)
    
    def test_with_reconstruction_fields(self):
        """Test construction with reconstruction-related fields."""
        B, max_rows, max_cols, token_dim = 4, 64, 16, 4
        
        batch = TokenBatch(
            tokens=torch.randn(B, max_rows, max_cols, token_dim),
            row_mask=torch.ones(B, max_rows, dtype=torch.bool),
            col_mask=torch.ones(B, max_cols, dtype=torch.bool),
            original_values=torch.randn(B, max_rows, max_cols),
            reconstruction_mask=torch.zeros(B, max_rows, max_cols, dtype=torch.bool),
        )
        
        assert batch.original_values is not None
        assert batch.original_values.shape == (B, max_rows, max_cols)
        assert batch.reconstruction_mask is not None
        assert batch.reconstruction_mask.shape == (B, max_rows, max_cols)
    
    def test_wrong_row_mask_shape_raises(self):
        """Test that mismatched row_mask shape raises ValueError."""
        B, max_rows, max_cols, token_dim = 4, 64, 16, 4
        
        with pytest.raises(ValueError):
            TokenBatch(
                tokens=torch.randn(B, max_rows, max_cols, token_dim),
                row_mask=torch.ones(B, 32, dtype=torch.bool),  # Wrong shape
                col_mask=torch.ones(B, max_cols, dtype=torch.bool),
            )
    
    def test_wrong_col_mask_shape_raises(self):
        """Test that mismatched col_mask shape raises ValueError."""
        B, max_rows, max_cols, token_dim = 4, 64, 16, 4
        
        with pytest.raises(ValueError):
            TokenBatch(
                tokens=torch.randn(B, max_rows, max_cols, token_dim),
                row_mask=torch.ones(B, max_rows, dtype=torch.bool),
                col_mask=torch.ones(B, 8, dtype=torch.bool),  # Wrong shape
            )
    
    def test_wrong_generator_ids_shape_raises(self):
        """Test that mismatched generator_ids shape raises ValueError."""
        B, max_rows, max_cols, token_dim = 4, 64, 16, 4
        
        with pytest.raises(ValueError):
            TokenBatch(
                tokens=torch.randn(B, max_rows, max_cols, token_dim),
                row_mask=torch.ones(B, max_rows, dtype=torch.bool),
                col_mask=torch.ones(B, max_cols, dtype=torch.bool),
                generator_ids=torch.tensor([0, 1]),  # Wrong size
            )
    
    def test_wrong_original_values_shape_raises(self):
        """Test that mismatched original_values shape raises ValueError."""
        B, max_rows, max_cols, token_dim = 4, 64, 16, 4
        
        with pytest.raises(ValueError):
            TokenBatch(
                tokens=torch.randn(B, max_rows, max_cols, token_dim),
                row_mask=torch.ones(B, max_rows, dtype=torch.bool),
                col_mask=torch.ones(B, max_cols, dtype=torch.bool),
                original_values=torch.randn(B, 32, max_cols),  # Wrong shape
            )
    
    def test_to_device(self):
        """Test moving batch to different device."""
        B, max_rows, max_cols, token_dim = 2, 32, 8, 4
        
        batch = TokenBatch(
            tokens=torch.randn(B, max_rows, max_cols, token_dim),
            row_mask=torch.ones(B, max_rows, dtype=torch.bool),
            col_mask=torch.ones(B, max_cols, dtype=torch.bool),
            generator_ids=torch.tensor([0, 1]),
            class_ids=torch.tensor([0, 1]),
            variant_ids=torch.tensor([-1, 0]),
            original_values=torch.randn(B, max_rows, max_cols),
            reconstruction_mask=torch.zeros(B, max_rows, max_cols, dtype=torch.bool),
        )
        
        batch_cpu = batch.to("cpu")
        
        assert batch_cpu.tokens.device.type == "cpu"
        assert batch_cpu.row_mask.device.type == "cpu"
        assert batch_cpu.col_mask.device.type == "cpu"
        assert batch_cpu.generator_ids.device.type == "cpu"
        assert batch_cpu.class_ids.device.type == "cpu"
        assert batch_cpu.variant_ids.device.type == "cpu"
        assert batch_cpu.original_values.device.type == "cpu"
        assert batch_cpu.reconstruction_mask.device.type == "cpu"
    
    def test_to_device_with_none_fields(self):
        """Test moving batch with None optional fields."""
        B, max_rows, max_cols, token_dim = 2, 32, 8, 4
        
        batch = TokenBatch(
            tokens=torch.randn(B, max_rows, max_cols, token_dim),
            row_mask=torch.ones(B, max_rows, dtype=torch.bool),
            col_mask=torch.ones(B, max_cols, dtype=torch.bool),
            # All optional fields are None
        )
        
        batch_cpu = batch.to("cpu")
        
        assert batch_cpu.tokens.device.type == "cpu"
        assert batch_cpu.generator_ids is None
        assert batch_cpu.class_ids is None
        assert batch_cpu.variant_ids is None
        assert batch_cpu.original_values is None
        assert batch_cpu.reconstruction_mask is None


# =============================================================================
# Test ReconstructionResult
# =============================================================================

class TestReconstructionResult:
    """Tests for ReconstructionResult."""
    
    def test_valid_construction(self):
        """Test basic construction."""
        B, max_rows, max_cols = 4, 64, 16
        
        result = ReconstructionResult(
            predictions=torch.randn(B, max_rows, max_cols),
            errors=torch.rand(B),
        )
        
        assert result.predictions.shape == (B, max_rows, max_cols)
        assert result.errors.shape == (B,)
        assert result.per_cell_errors is None
    
    def test_with_per_cell_errors(self):
        """Test construction with per-cell errors."""
        B, max_rows, max_cols = 4, 64, 16
        
        result = ReconstructionResult(
            predictions=torch.randn(B, max_rows, max_cols),
            errors=torch.rand(B),
            per_cell_errors=torch.rand(B, max_rows, max_cols),
        )
        
        assert result.per_cell_errors is not None
        assert result.per_cell_errors.shape == (B, max_rows, max_cols)
    
    def test_immutability(self):
        """Test that ReconstructionResult is frozen."""
        result = ReconstructionResult(
            predictions=torch.randn(4, 64, 16),
            errors=torch.rand(4),
        )
        
        with pytest.raises((AttributeError, TypeError)):
            result.errors = torch.zeros(4)


# =============================================================================
# Test MoEOutput
# =============================================================================

class TestMoEOutput:
    """Tests for MoEOutput."""
    
    def test_valid_construction(self):
        """Test basic construction."""
        B, n_experts = 4, 5
        
        output = MoEOutput(
            gate_logits=torch.randn(B, n_experts),
            gate_probs=torch.softmax(torch.randn(B, n_experts), dim=-1),
        )
        
        assert output.gate_logits.shape == (B, n_experts)
        assert output.gate_probs.shape == (B, n_experts)
        assert output.n_experts == n_experts
    
    def test_with_expert_outputs(self):
        """Test construction with expert outputs."""
        B, n_experts = 4, 5
        
        expert_outputs = [torch.randn(B, 32) for _ in range(n_experts)]
        
        output = MoEOutput(
            gate_logits=torch.randn(B, n_experts),
            gate_probs=torch.softmax(torch.randn(B, n_experts), dim=-1),
            expert_outputs=expert_outputs,
        )
        
        assert output.expert_outputs is not None
        assert len(output.expert_outputs) == n_experts
    
    def test_with_combined_output(self):
        """Test construction with combined output."""
        B, n_experts = 4, 5
        
        output = MoEOutput(
            gate_logits=torch.randn(B, n_experts),
            gate_probs=torch.softmax(torch.randn(B, n_experts), dim=-1),
            combined_output=torch.randn(B, n_experts),
        )
        
        assert output.combined_output is not None
        assert output.combined_output.shape == (B, n_experts)
    
    def test_n_experts_property(self):
        """Test n_experts property accessor."""
        for n_experts in [3, 5, 7, 10]:
            output = MoEOutput(
                gate_logits=torch.randn(4, n_experts),
                gate_probs=torch.softmax(torch.randn(4, n_experts), dim=-1),
            )
            assert output.n_experts == n_experts


# =============================================================================
# Test PosteriorResult
# =============================================================================

class TestPosteriorResult:
    """Tests for PosteriorResult."""
    
    def test_valid_construction_minimal(self):
        """Test construction with minimal required fields."""
        B = 4
        
        result = PosteriorResult(
            p_class=torch.softmax(torch.randn(B, 3), dim=-1),
            entropy_class=torch.rand(B),
        )
        
        assert result.p_class.shape == (B, 3)
        assert result.entropy_class.shape == (B,)
    
    def test_with_mnar_variant_posterior(self):
        """Test construction with MNAR variant posterior."""
        B, n_variants = 4, 3
        
        result = PosteriorResult(
            p_class=torch.softmax(torch.randn(B, 3), dim=-1),
            p_mnar_variant=torch.softmax(torch.randn(B, n_variants), dim=-1),
            entropy_class=torch.rand(B),
        )
        
        assert result.p_mnar_variant is not None
        assert result.p_mnar_variant.shape == (B, n_variants)
    
    def test_with_full_mechanism_posterior(self):
        """Test construction with full mechanism posterior."""
        B = 4
        n_mechanisms = 5  # MCAR, MAR, + 3 MNAR variants
        
        result = PosteriorResult(
            p_class=torch.softmax(torch.randn(B, 3), dim=-1),
            p_mechanism=torch.softmax(torch.randn(B, n_mechanisms), dim=-1),
            entropy_class=torch.rand(B),
            entropy_mechanism=torch.rand(B),
        )
        
        assert result.p_mechanism is not None
        assert result.p_mechanism.shape == (B, n_mechanisms)
        assert result.entropy_mechanism is not None
    
    def test_with_reconstruction_errors(self):
        """Test construction with reconstruction errors dict."""
        B = 4
        
        recon_errors = {
            "mcar": torch.rand(B),
            "mar": torch.rand(B),
            "self_censoring": torch.rand(B),
        }
        
        result = PosteriorResult(
            p_class=torch.softmax(torch.randn(B, 3), dim=-1),
            entropy_class=torch.rand(B),
            reconstruction_errors=recon_errors,
        )
        
        assert result.reconstruction_errors is not None
        assert "mcar" in result.reconstruction_errors
        assert "mar" in result.reconstruction_errors
    
    def test_with_gate_probs(self):
        """Test construction with MoE gate probabilities."""
        B, n_experts = 4, 5
        
        result = PosteriorResult(
            p_class=torch.softmax(torch.randn(B, 3), dim=-1),
            entropy_class=torch.rand(B),
            gate_probs=torch.softmax(torch.randn(B, n_experts), dim=-1),
        )
        
        assert result.gate_probs is not None
        assert result.gate_probs.shape == (B, n_experts)
    
    def test_probabilities_sum_to_one(self):
        """Test that probability tensors sum to 1."""
        B = 4
        
        p_class = torch.softmax(torch.randn(B, 3), dim=-1)
        p_mechanism = torch.softmax(torch.randn(B, 5), dim=-1)
        
        result = PosteriorResult(
            p_class=p_class,
            p_mechanism=p_mechanism,
            entropy_class=torch.rand(B),
        )
        
        # Check that probabilities sum to 1
        assert torch.allclose(result.p_class.sum(dim=-1), torch.ones(B), atol=1e-5)
        assert torch.allclose(result.p_mechanism.sum(dim=-1), torch.ones(B), atol=1e-5)


# =============================================================================
# Test Decision
# =============================================================================

class TestDecision:
    """Tests for Decision."""
    
    def test_valid_construction(self):
        """Test basic construction."""
        B = 4
        
        decision = Decision(
            action_ids=torch.tensor([0, 1, 2, 0]),
            action_names=("Green", "Yellow", "Red"),
            expected_risks=torch.rand(B),
        )
        
        assert decision.action_ids.shape == (B,)
        assert len(decision.action_names) == 3
        assert decision.expected_risks.shape == (B,)
    
    def test_with_confidence(self):
        """Test construction with confidence scores."""
        B = 4
        
        decision = Decision(
            action_ids=torch.tensor([0, 1, 2, 0]),
            action_names=("Green", "Yellow", "Red"),
            expected_risks=torch.rand(B),
            confidence=torch.rand(B),
        )
        
        assert decision.confidence is not None
        assert decision.confidence.shape == (B,)
    
    def test_batch_size_property(self):
        """Test batch_size property accessor."""
        for B in [1, 4, 8, 16]:
            decision = Decision(
                action_ids=torch.randint(0, 3, (B,)),
                action_names=("Green", "Yellow", "Red"),
                expected_risks=torch.rand(B),
            )
            assert decision.batch_size == B
    
    def test_action_ids_in_valid_range(self):
        """Test that action IDs are within valid range."""
        B = 4
        action_ids = torch.tensor([0, 1, 2, 1])
        
        decision = Decision(
            action_ids=action_ids,
            action_names=("Green", "Yellow", "Red"),
            expected_risks=torch.rand(B),
        )
        
        # All action IDs should be in [0, len(action_names))
        assert (decision.action_ids >= 0).all()
        assert (decision.action_ids < len(decision.action_names)).all()


# =============================================================================
# Test LacunaOutput
# =============================================================================

class TestLacunaOutput:
    """Tests for LacunaOutput."""
    
    @pytest.fixture
    def sample_posterior(self):
        """Create sample PosteriorResult."""
        B = 4
        return PosteriorResult(
            p_class=torch.softmax(torch.randn(B, 3), dim=-1),
            entropy_class=torch.rand(B),
        )
    
    @pytest.fixture
    def sample_decision(self):
        """Create sample Decision."""
        B = 4
        return Decision(
            action_ids=torch.randint(0, 3, (B,)),
            action_names=("Green", "Yellow", "Red"),
            expected_risks=torch.rand(B),
        )
    
    @pytest.fixture
    def sample_moe_output(self):
        """Create sample MoEOutput."""
        B, n_experts = 4, 5
        return MoEOutput(
            gate_logits=torch.randn(B, n_experts),
            gate_probs=torch.softmax(torch.randn(B, n_experts), dim=-1),
        )
    
    def test_valid_construction_minimal(self, sample_posterior):
        """Test construction with minimal fields."""
        B = 4
        
        output = LacunaOutput(
            posterior=sample_posterior,
            evidence=torch.randn(B, 64),
        )
        
        assert output.posterior is not None
        assert output.evidence.shape == (B, 64)
        assert output.decision is None
        assert output.reconstruction is None
        assert output.moe_output is None
    
    def test_valid_construction_full(
        self,
        sample_posterior,
        sample_decision,
        sample_moe_output,
    ):
        """Test construction with all fields."""
        B, max_rows, max_cols = 4, 64, 16
        
        reconstruction = {
            "mcar": ReconstructionResult(
                predictions=torch.randn(B, max_rows, max_cols),
                errors=torch.rand(B),
            ),
            "mar": ReconstructionResult(
                predictions=torch.randn(B, max_rows, max_cols),
                errors=torch.rand(B),
            ),
        }
        
        output = LacunaOutput(
            posterior=sample_posterior,
            decision=sample_decision,
            reconstruction=reconstruction,
            moe_output=sample_moe_output,
            evidence=torch.randn(B, 64),
        )
        
        assert output.posterior is not None
        assert output.decision is not None
        assert output.reconstruction is not None
        assert len(output.reconstruction) == 2
        assert output.moe_output is not None
        assert output.evidence is not None
    
    def test_accessing_nested_fields(self, sample_posterior, sample_decision):
        """Test accessing fields of nested structures."""
        B = 4
        
        output = LacunaOutput(
            posterior=sample_posterior,
            decision=sample_decision,
            evidence=torch.randn(B, 64),
        )
        
        # Access posterior fields
        assert output.posterior.p_class.shape == (B, 3)
        
        # Access decision fields
        assert output.decision.action_ids.shape == (B,)
        assert output.decision.action_names == ("Green", "Yellow", "Red")