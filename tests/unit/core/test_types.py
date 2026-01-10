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
    MNAR_VARIANT_NAMES,
    DEFAULT_N_MNAR_VARIANTS,
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
    
    def test_mnar_variant_names(self):
        """MNAR_VARIANT_NAMES should have correct entries."""
        assert len(MNAR_VARIANT_NAMES) == DEFAULT_N_MNAR_VARIANTS
        assert "SelfCensoring" in MNAR_VARIANT_NAMES[MNAR_SELF_CENSORING]
        assert "Threshold" in MNAR_VARIANT_NAMES[MNAR_THRESHOLD]
        assert "Latent" in MNAR_VARIANT_NAMES[MNAR_LATENT]
        assert "Mixture" in MNAR_VARIANT_NAMES[MNAR_MIXTURE]
    
    def test_default_n_mnar_variants(self):
        """DEFAULT_N_MNAR_VARIANTS should be 4."""
        assert DEFAULT_N_MNAR_VARIANTS == 4


# =============================================================================
# Test ObservedDataset
# =============================================================================

class TestObservedDataset:
    """Tests for ObservedDataset."""
    
    def test_valid_construction(self):
        """Test basic construction with valid inputs."""
        n, d = 100, 10
        x = torch.randn(n, d, dtype=torch.float32)
        r = torch.rand(n, d) > 0.2  # ~20% missing
        
        # Zero out missing values (convention for ObservedDataset)
        x = x * r.float()
        
        dataset = ObservedDataset(
            x=x,
            r=r,
            n=n,
            d=d,
            dataset_id="test_dataset",
        )
        
        assert dataset.n == n
        assert dataset.d == d
        assert dataset.dataset_id == "test_dataset"
    
    def test_shape_properties(self):
        """Test n and d property accessors."""
        n, d = 50, 8
        x = torch.randn(n, d, dtype=torch.float32)
        r = torch.ones(n, d, dtype=torch.bool)
        
        dataset = ObservedDataset(x=x, r=r, n=n, d=d)
        
        assert dataset.n == 50
        assert dataset.d == 8
    
    def test_missing_values_zeroed(self):
        """Test that missing values should be zeros (by convention)."""
        # Create data with zeros where r=False
        x = torch.tensor([[1.0, 0.0, 3.0], [4.0, 5.0, 0.0]], dtype=torch.float32)
        r = torch.tensor([[True, False, True], [True, True, False]])
        
        dataset = ObservedDataset(x=x, r=r, n=2, d=3)
        
        # Check that missing positions have zeros
        assert dataset.x[0, 1] == 0.0
        assert dataset.x[1, 2] == 0.0
        # Check that observed values are preserved
        assert dataset.x[0, 0] == 1.0
        assert dataset.x[0, 2] == 3.0
    
    def test_optional_fields_default_to_none(self):
        """Test that optional fields have sensible defaults."""
        n, d = 10, 5
        x = torch.randn(n, d, dtype=torch.float32)
        r = torch.ones(n, d, dtype=torch.bool)
        
        dataset = ObservedDataset(x=x, r=r, n=n, d=d)
        
        # feature_names defaults to None
        assert dataset.feature_names is None
        # meta defaults to None
        assert dataset.meta is None
        # dataset_id has a default
        assert dataset.dataset_id == "unnamed"
    
    def test_with_feature_names(self):
        """Test construction with feature_names."""
        n, d = 10, 3
        x = torch.randn(n, d, dtype=torch.float32)
        r = torch.ones(n, d, dtype=torch.bool)
        feature_names = ("age", "income", "score")
        
        dataset = ObservedDataset(
            x=x,
            r=r,
            n=n,
            d=d,
            feature_names=feature_names,
        )
        
        assert dataset.feature_names == ("age", "income", "score")
    
    def test_with_meta(self):
        """Test construction with meta dictionary."""
        n, d = 10, 5
        x = torch.randn(n, d, dtype=torch.float32)
        r = torch.ones(n, d, dtype=torch.bool)
        meta = {"source": "synthetic", "seed": 42}
        
        dataset = ObservedDataset(x=x, r=r, n=n, d=d, meta=meta)
        
        assert dataset.meta == {"source": "synthetic", "seed": 42}
    
    def test_immutability(self):
        """Test that ObservedDataset is frozen (immutable)."""
        n, d = 10, 5
        x = torch.randn(n, d, dtype=torch.float32)
        r = torch.ones(n, d, dtype=torch.bool)
        
        dataset = ObservedDataset(x=x, r=r, n=n, d=d)
        
        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(Exception):  # FrozenInstanceError is a subclass
            dataset.n = 999
    
    def test_shape_validation_x(self):
        """Test that x shape must match (n, d)."""
        n, d = 10, 5
        x = torch.randn(20, 5, dtype=torch.float32)  # Wrong n
        r = torch.ones(n, d, dtype=torch.bool)
        
        with pytest.raises(ValueError):
            ObservedDataset(x=x, r=r, n=n, d=d)
    
    def test_shape_validation_r(self):
        """Test that r shape must match (n, d)."""
        n, d = 10, 5
        x = torch.randn(n, d, dtype=torch.float32)
        r = torch.ones(n, 10, dtype=torch.bool)  # Wrong d
        
        with pytest.raises(ValueError):
            ObservedDataset(x=x, r=r, n=n, d=d)
    
    def test_r_dtype_must_be_bool(self):
        """Test that r must have dtype bool."""
        n, d = 10, 5
        x = torch.randn(n, d, dtype=torch.float32)
        r = torch.ones(n, d, dtype=torch.float32)  # Wrong dtype
        
        with pytest.raises(TypeError):
            ObservedDataset(x=x, r=r, n=n, d=d)
    
    def test_missing_rate_property(self):
        """Test the missing_rate property."""
        n, d = 100, 10
        x = torch.randn(n, d, dtype=torch.float32)
        # Create 30% missing
        r = torch.rand(n, d) > 0.3
        x = x * r.float()
        
        dataset = ObservedDataset(x=x, r=r, n=n, d=d)
        
        expected_rate = 1.0 - r.float().mean().item()
        assert abs(dataset.missing_rate - expected_rate) < 1e-6
    
    def test_n_observed_property(self):
        """Test the n_observed property."""
        n, d = 10, 5
        x = torch.randn(n, d, dtype=torch.float32)
        r = torch.ones(n, d, dtype=torch.bool)
        r[0, 0] = False
        r[1, 1] = False
        x = x * r.float()
        
        dataset = ObservedDataset(x=x, r=r, n=n, d=d)
        
        assert dataset.n_observed == 48  # 50 - 2


# =============================================================================
# Test TokenBatch
# =============================================================================

class TestTokenBatch:
    """Tests for TokenBatch."""
    
    def test_valid_construction(self):
        """Test basic construction with valid inputs."""
        B, max_rows, max_cols, token_dim = 4, 64, 16, 3
        tokens = torch.randn(B, max_rows, max_cols, token_dim)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        generator_ids = torch.tensor([0, 1, 2, 3])
        class_ids = torch.tensor([0, 0, 1, 2])
        
        batch = TokenBatch(
            tokens=tokens,
            row_mask=row_mask,
            col_mask=col_mask,
            generator_ids=generator_ids,
            class_ids=class_ids,
        )
        
        assert batch.tokens.shape == (B, max_rows, max_cols, token_dim)
        assert batch.row_mask.shape == (B, max_rows)
        assert batch.col_mask.shape == (B, max_cols)
        assert batch.generator_ids.shape == (B,)
        assert batch.class_ids.shape == (B,)
    
    def test_batch_size_property(self):
        """Test that batch_size property works correctly."""
        B, max_rows, max_cols, token_dim = 8, 32, 10, 3
        tokens = torch.randn(B, max_rows, max_cols, token_dim)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        generator_ids = torch.zeros(B, dtype=torch.long)
        class_ids = torch.zeros(B, dtype=torch.long)
        
        batch = TokenBatch(
            tokens=tokens,
            row_mask=row_mask,
            col_mask=col_mask,
            generator_ids=generator_ids,
            class_ids=class_ids,
        )
        
        assert batch.batch_size == B
    
    def test_optional_original_values(self):
        """Test that original_values is optional."""
        B, max_rows, max_cols, token_dim = 4, 64, 16, 3
        tokens = torch.randn(B, max_rows, max_cols, token_dim)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        batch = TokenBatch(
            tokens=tokens,
            row_mask=row_mask,
            col_mask=col_mask,
        )
        
        assert batch.original_values is None
    
    def test_with_original_values(self):
        """Test construction with original_values for reconstruction."""
        B, max_rows, max_cols, token_dim = 4, 64, 16, 3
        tokens = torch.randn(B, max_rows, max_cols, token_dim)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        original_values = torch.randn(B, max_rows, max_cols)
        
        batch = TokenBatch(
            tokens=tokens,
            row_mask=row_mask,
            col_mask=col_mask,
            original_values=original_values,
        )
        
        assert batch.original_values is not None
        assert batch.original_values.shape == (B, max_rows, max_cols)
    
    def test_optional_variant_ids(self):
        """Test that variant_ids is optional."""
        B, max_rows, max_cols, token_dim = 4, 64, 16, 3
        tokens = torch.randn(B, max_rows, max_cols, token_dim)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        generator_ids = torch.zeros(B, dtype=torch.long)
        class_ids = torch.zeros(B, dtype=torch.long)
        
        batch = TokenBatch(
            tokens=tokens,
            row_mask=row_mask,
            col_mask=col_mask,
            generator_ids=generator_ids,
            class_ids=class_ids,
        )
        
        assert batch.variant_ids is None
    
    def test_with_variant_ids(self):
        """Test construction with variant_ids for MNAR variants."""
        B, max_rows, max_cols, token_dim = 4, 64, 16, 3
        tokens = torch.randn(B, max_rows, max_cols, token_dim)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        generator_ids = torch.zeros(B, dtype=torch.long)
        class_ids = torch.tensor([MNAR, MNAR, MNAR, MNAR])
        variant_ids = torch.tensor([0, 1, 2, 3])  # Different MNAR variants
        
        batch = TokenBatch(
            tokens=tokens,
            row_mask=row_mask,
            col_mask=col_mask,
            generator_ids=generator_ids,
            class_ids=class_ids,
            variant_ids=variant_ids,
        )
        
        assert batch.variant_ids is not None
        assert batch.variant_ids.shape == (B,)


# =============================================================================
# Test ReconstructionResult
# =============================================================================

class TestReconstructionResult:
    """Tests for ReconstructionResult."""
    
    def test_valid_construction(self):
        """Test basic construction."""
        B, max_rows, max_cols = 4, 64, 16
        predictions = torch.randn(B, max_rows, max_cols)
        errors = torch.rand(B)
        
        result = ReconstructionResult(
            predictions=predictions,
            errors=errors,
        )
        
        assert result.predictions.shape == (B, max_rows, max_cols)
        assert result.errors.shape == (B,)
        assert result.per_cell_errors is None
    
    def test_with_per_cell_errors(self):
        """Test construction with per-cell errors."""
        B, max_rows, max_cols = 4, 64, 16
        predictions = torch.randn(B, max_rows, max_cols)
        errors = torch.rand(B)
        per_cell_errors = torch.rand(B, max_rows, max_cols)
        
        result = ReconstructionResult(
            predictions=predictions,
            errors=errors,
            per_cell_errors=per_cell_errors,
        )
        
        assert result.per_cell_errors is not None
        assert result.per_cell_errors.shape == (B, max_rows, max_cols)
    
    def test_errors_non_negative(self):
        """Test that errors should typically be non-negative."""
        B, max_rows, max_cols = 4, 64, 16
        predictions = torch.randn(B, max_rows, max_cols)
        errors = torch.abs(torch.randn(B))  # Ensure non-negative
        
        result = ReconstructionResult(
            predictions=predictions,
            errors=errors,
        )
        
        assert (result.errors >= 0).all()


# =============================================================================
# Test MoEOutput
# =============================================================================

class TestMoEOutput:
    """Tests for MoEOutput."""
    
    def test_valid_construction(self):
        """Test basic construction."""
        B, n_experts = 4, 5
        gate_logits = torch.randn(B, n_experts)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        
        output = MoEOutput(
            gate_logits=gate_logits,
            gate_probs=gate_probs,
        )
        
        assert output.gate_logits.shape == (B, n_experts)
        assert output.gate_probs.shape == (B, n_experts)
    
    def test_n_experts_property(self):
        """Test n_experts property."""
        B, n_experts = 4, 5
        gate_logits = torch.randn(B, n_experts)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        
        output = MoEOutput(
            gate_logits=gate_logits,
            gate_probs=gate_probs,
        )
        
        assert output.n_experts == n_experts
    
    def test_gate_probs_sum_to_one(self):
        """Test that gate_probs sum to 1."""
        B, n_experts = 4, 5
        gate_logits = torch.randn(B, n_experts)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        
        output = MoEOutput(
            gate_logits=gate_logits,
            gate_probs=gate_probs,
        )
        
        sums = output.gate_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5)
    
    def test_optional_expert_outputs(self):
        """Test that expert_outputs is optional."""
        B, n_experts = 4, 5
        gate_logits = torch.randn(B, n_experts)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        
        output = MoEOutput(
            gate_logits=gate_logits,
            gate_probs=gate_probs,
        )
        
        assert output.expert_outputs is None
    
    def test_with_expert_outputs(self):
        """Test construction with expert_outputs."""
        B, hidden_dim, n_experts = 4, 64, 5
        gate_logits = torch.randn(B, n_experts)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        expert_outputs = [torch.randn(B, hidden_dim) for _ in range(n_experts)]
        
        output = MoEOutput(
            gate_logits=gate_logits,
            gate_probs=gate_probs,
            expert_outputs=expert_outputs,
        )
        
        assert output.expert_outputs is not None
        assert len(output.expert_outputs) == n_experts
    
    def test_optional_combined_output(self):
        """Test that combined_output is optional."""
        B, n_experts = 4, 5
        gate_logits = torch.randn(B, n_experts)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        
        output = MoEOutput(
            gate_logits=gate_logits,
            gate_probs=gate_probs,
        )
        
        assert output.combined_output is None
    
    def test_with_combined_output(self):
        """Test construction with combined_output."""
        B, hidden_dim, n_experts = 4, 64, 5
        gate_logits = torch.randn(B, n_experts)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        combined_output = torch.randn(B, hidden_dim)
        
        output = MoEOutput(
            gate_logits=gate_logits,
            gate_probs=gate_probs,
            combined_output=combined_output,
        )
        
        assert output.combined_output is not None
        assert output.combined_output.shape == (B, hidden_dim)


# =============================================================================
# Test PosteriorResult
# =============================================================================

class TestPosteriorResult:
    """Tests for PosteriorResult."""
    
    def test_valid_construction_minimal(self):
        """Test minimal construction with only p_class."""
        B = 4
        n_classes = 3
        p_class = torch.softmax(torch.randn(B, n_classes), dim=-1)
        
        posterior = PosteriorResult(p_class=p_class)
        
        assert posterior.p_class.shape == (B, n_classes)
    
    def test_probabilities_sum_to_one(self):
        """Test that p_class sums to 1 along class dimension."""
        B = 4
        n_classes = 3
        p_class = torch.softmax(torch.randn(B, n_classes), dim=-1)
        
        posterior = PosteriorResult(p_class=p_class)
        
        sums = posterior.p_class.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5)
    
    def test_optional_entropy_class(self):
        """Test that entropy_class is optional."""
        B = 4
        p_class = torch.softmax(torch.randn(B, 3), dim=-1)
        
        posterior = PosteriorResult(p_class=p_class)
        
        assert posterior.entropy_class is None
    
    def test_with_entropy_class(self):
        """Test construction with entropy_class."""
        B = 4
        p_class = torch.softmax(torch.randn(B, 3), dim=-1)
        entropy_class = torch.rand(B)
        
        posterior = PosteriorResult(
            p_class=p_class,
            entropy_class=entropy_class,
        )
        
        assert posterior.entropy_class is not None
        assert posterior.entropy_class.shape == (B,)
    
    def test_with_mnar_variant_posteriors(self):
        """Test construction with MNAR variant posteriors."""
        B = 4
        n_classes = 3
        n_variants = 4
        
        p_class = torch.softmax(torch.randn(B, n_classes), dim=-1)
        p_mnar_variant = torch.softmax(torch.randn(B, n_variants), dim=-1)
        
        posterior = PosteriorResult(
            p_class=p_class,
            p_mnar_variant=p_mnar_variant,
        )
        
        assert posterior.p_mnar_variant is not None
        assert posterior.p_mnar_variant.shape == (B, n_variants)
    
    def test_with_full_mechanism_posterior(self):
        """Test construction with full mechanism posterior (class + variants)."""
        B = 4
        n_classes = 3
        n_variants = 4
        # Full mechanism: MCAR, MAR, MNAR_v0, MNAR_v1, MNAR_v2, MNAR_v3
        n_mechanisms = 2 + n_variants
        
        p_class = torch.softmax(torch.randn(B, n_classes), dim=-1)
        p_mechanism = torch.softmax(torch.randn(B, n_mechanisms), dim=-1)
        
        posterior = PosteriorResult(
            p_class=p_class,
            p_mechanism=p_mechanism,
        )
        
        assert posterior.p_mechanism is not None
        assert posterior.p_mechanism.shape == (B, n_mechanisms)
    
    def test_optional_logits(self):
        """Test that logits fields are optional."""
        B = 4
        p_class = torch.softmax(torch.randn(B, 3), dim=-1)
        
        posterior = PosteriorResult(p_class=p_class)
        
        assert posterior.logits_class is None
        assert posterior.logits_mnar_variant is None
    
    def test_with_logits(self):
        """Test construction with raw logits."""
        B = 4
        n_classes = 3
        n_variants = 4
        
        logits_class = torch.randn(B, n_classes)
        logits_mnar_variant = torch.randn(B, n_variants)
        p_class = torch.softmax(logits_class, dim=-1)
        
        posterior = PosteriorResult(
            p_class=p_class,
            logits_class=logits_class,
            logits_mnar_variant=logits_mnar_variant,
        )
        
        assert posterior.logits_class is not None
        assert posterior.logits_mnar_variant is not None
    
    def test_gate_probs_optional(self):
        """Test that gate_probs (from MoE) is optional."""
        B = 4
        p_class = torch.softmax(torch.randn(B, 3), dim=-1)
        
        posterior = PosteriorResult(p_class=p_class)
        
        assert posterior.gate_probs is None
    
    def test_reconstruction_errors_optional(self):
        """Test that reconstruction_errors is optional."""
        B = 4
        p_class = torch.softmax(torch.randn(B, 3), dim=-1)
        
        posterior = PosteriorResult(p_class=p_class)
        
        assert posterior.reconstruction_errors is None


# =============================================================================
# Test Decision
# =============================================================================

class TestDecision:
    """Tests for Decision."""
    
    def test_valid_construction(self):
        """Test basic construction."""
        B = 4
        action_ids = torch.tensor([0, 1, 2, 0])
        expected_risks = torch.rand(B)
        
        decision = Decision(
            action_ids=action_ids,
            expected_risks=expected_risks,
        )
        
        assert decision.action_ids.shape == (B,)
        assert decision.expected_risks.shape == (B,)
    
    def test_default_action_names(self):
        """Test that default action_names is Green, Yellow, Red."""
        B = 4
        action_ids = torch.tensor([0, 1, 2, 0])
        expected_risks = torch.rand(B)
        
        decision = Decision(
            action_ids=action_ids,
            expected_risks=expected_risks,
        )
        
        assert decision.action_names == ("Green", "Yellow", "Red")
    
    def test_custom_action_names(self):
        """Test construction with custom action_names."""
        B = 4
        action_ids = torch.tensor([0, 1, 2, 0])
        expected_risks = torch.rand(B)
        
        decision = Decision(
            action_ids=action_ids,
            expected_risks=expected_risks,
            action_names=("Safe", "Caution", "Danger"),
        )
        
        assert decision.action_names == ("Safe", "Caution", "Danger")
    
    def test_action_ids_in_range(self):
        """Test that action_ids are valid indices."""
        B = 4
        action_ids = torch.tensor([0, 1, 2, 0])
        expected_risks = torch.rand(B)
        
        decision = Decision(
            action_ids=action_ids,
            expected_risks=expected_risks,
        )
        
        num_actions = len(decision.action_names)
        assert (decision.action_ids >= 0).all()
        assert (decision.action_ids < num_actions).all()
    
    def test_confidence_optional(self):
        """Test that confidence is optional."""
        B = 4
        action_ids = torch.tensor([0, 1, 2, 0])
        expected_risks = torch.rand(B)
        
        decision = Decision(
            action_ids=action_ids,
            expected_risks=expected_risks,
        )
        
        assert decision.confidence is None
    
    def test_with_confidence(self):
        """Test construction with confidence."""
        B = 4
        action_ids = torch.tensor([0, 1, 2, 0])
        expected_risks = torch.rand(B)
        confidence = torch.rand(B)
        
        decision = Decision(
            action_ids=action_ids,
            expected_risks=expected_risks,
            confidence=confidence,
        )
        
        assert decision.confidence is not None
        assert decision.confidence.shape == (B,)
    
    def test_batch_size_property(self):
        """Test batch_size property."""
        B = 8
        action_ids = torch.zeros(B, dtype=torch.long)
        expected_risks = torch.rand(B)
        
        decision = Decision(
            action_ids=action_ids,
            expected_risks=expected_risks,
        )
        
        assert decision.batch_size == B
    
    def test_get_actions_method(self):
        """Test get_actions method returns action names."""
        B = 4
        action_ids = torch.tensor([0, 1, 2, 1])
        expected_risks = torch.rand(B)
        
        decision = Decision(
            action_ids=action_ids,
            expected_risks=expected_risks,
        )
        
        actions = decision.get_actions()
        assert actions == ["Green", "Yellow", "Red", "Yellow"]


# =============================================================================
# Test LacunaOutput
# =============================================================================

class TestLacunaOutput:
    """Tests for LacunaOutput."""
    
    @pytest.fixture
    def sample_posterior(self):
        """Create a sample PosteriorResult."""
        B = 4
        return PosteriorResult(
            p_class=torch.softmax(torch.randn(B, 3), dim=-1),
        )
    
    @pytest.fixture
    def sample_decision(self):
        """Create a sample Decision."""
        B = 4
        return Decision(
            action_ids=torch.tensor([0, 1, 2, 0]),
            expected_risks=torch.rand(B),
        )
    
    @pytest.fixture
    def sample_moe_output(self):
        """Create a sample MoEOutput."""
        B, n_experts = 4, 4
        gate_logits = torch.randn(B, n_experts)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        return MoEOutput(
            gate_logits=gate_logits,
            gate_probs=gate_probs,
        )
    
    def test_valid_construction_minimal(self, sample_posterior):
        """Test minimal construction with only required fields."""
        output = LacunaOutput(posterior=sample_posterior)
        
        assert output.posterior is not None
        assert output.decision is None
        assert output.reconstruction is None
        assert output.moe is None
        assert output.evidence is None
    
    def test_valid_construction_full(self, sample_posterior, sample_decision, sample_moe_output):
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
            moe=sample_moe_output,
            evidence=torch.randn(B, 64),
        )
        
        assert output.posterior is not None
        assert output.decision is not None
        assert output.reconstruction is not None
        assert len(output.reconstruction) == 2
        assert output.moe is not None
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


# =============================================================================
# Integration Tests
# =============================================================================

class TestTypeInteractions:
    """Tests for interactions between types."""
    
    def test_token_batch_to_output_flow(self):
        """Test that types work together in a typical flow."""
        B, max_rows, max_cols, token_dim = 4, 64, 16, 3
        
        # Create input batch
        batch = TokenBatch(
            tokens=torch.randn(B, max_rows, max_cols, token_dim),
            row_mask=torch.ones(B, max_rows, dtype=torch.bool),
            col_mask=torch.ones(B, max_cols, dtype=torch.bool),
            generator_ids=torch.tensor([0, 1, 2, 3]),
            class_ids=torch.tensor([0, 0, 1, 2]),
        )
        
        # Simulate model output
        posterior = PosteriorResult(
            p_class=torch.softmax(torch.randn(B, 3), dim=-1),
        )
        
        decision = Decision(
            action_ids=torch.argmax(posterior.p_class, dim=-1),
            expected_risks=torch.rand(B),
        )
        
        output = LacunaOutput(
            posterior=posterior,
            decision=decision,
            evidence=torch.randn(B, 64),
        )
        
        # Verify shapes are consistent
        assert batch.batch_size == B
        assert output.posterior.p_class.shape[0] == B
        assert output.decision.action_ids.shape[0] == B
    
    def test_observed_dataset_consistent_shapes(self):
        """Test that ObservedDataset maintains shape consistency."""
        n, d = 100, 20
        x = torch.randn(n, d, dtype=torch.float32)
        r = torch.rand(n, d) > 0.3
        x = x * r.float()  # Zero out missing
        
        dataset = ObservedDataset(
            x=x,
            r=r,
            n=n,
            d=d,
            dataset_id="test",
        )
        
        # All shapes should be consistent
        assert dataset.x.shape == (n, d)
        assert dataset.r.shape == (n, d)
        assert dataset.n == n
        assert dataset.d == d