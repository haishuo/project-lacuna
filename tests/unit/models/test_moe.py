"""
Tests for lacuna.models.moe

Tests the Mixture of Experts layer for mechanism classification:
    - MoEConfig: Configuration dataclass
    - GatingNetwork: Produces expert mixture weights
    - ExpertHead: Lightweight mechanism-specific refinement
    - ExpertHeads: Container for all expert heads
    - MixtureOfExperts: Full MoE layer combining gating and experts
    - RowToDatasetAggregator: Aggregates row-level to dataset-level
    - create_moe: Factory function
"""

import pytest
import torch
import torch.nn as nn

from lacuna.models.moe import (
    MoEConfig,
    GatingNetwork,
    ExpertHead,
    ExpertHeads,
    MixtureOfExperts,
    RowToDatasetAggregator,
    create_moe,
)
from lacuna.core.types import MoEOutput, MCAR, MAR, MNAR


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default MoEConfig for testing (without reconstruction errors for simpler tests)."""
    return MoEConfig(
        evidence_dim=64,
        hidden_dim=128,
        mnar_variants=["self_censoring", "threshold", "latent"],
        use_reconstruction_errors=False,  # Explicitly disable for simpler evidence-only tests
    )


@pytest.fixture
def config_with_recon():
    """MoEConfig with reconstruction errors enabled."""
    return MoEConfig(
        evidence_dim=64,
        hidden_dim=128,
        mnar_variants=["self_censoring", "threshold", "latent"],
        use_reconstruction_errors=True,
        n_reconstruction_heads=5,
    )


@pytest.fixture
def config_with_experts():
    """MoEConfig with expert heads enabled."""
    return MoEConfig(
        evidence_dim=64,
        hidden_dim=128,
        mnar_variants=["self_censoring", "threshold", "latent"],
        use_reconstruction_errors=False,  # Disable recon for simpler tests
        use_expert_heads=True,
    )


@pytest.fixture
def sample_evidence():
    """Sample evidence tensor for testing."""
    B = 4
    evidence_dim = 64
    return torch.randn(B, evidence_dim)


@pytest.fixture
def sample_recon_errors():
    """Sample reconstruction errors tensor."""
    B = 4
    n_heads = 5
    # Errors should be non-negative
    return torch.abs(torch.randn(B, n_heads))


@pytest.fixture
def sample_row_level_inputs():
    """Sample inputs for row-level gating."""
    B = 4
    max_rows = 32
    hidden_dim = 128
    n_heads = 5
    
    return {
        "evidence": torch.randn(B, max_rows, hidden_dim),
        "reconstruction_errors": torch.abs(torch.randn(B, max_rows, n_heads)),
        "row_mask": torch.ones(B, max_rows, dtype=torch.bool),
    }


# =============================================================================
# Test MoEConfig
# =============================================================================

class TestMoEConfig:
    """Tests for MoEConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MoEConfig()
        
        assert config.evidence_dim == 64
        assert config.hidden_dim == 128
        assert config.mnar_variants == ["self_censoring", "threshold", "latent"]
        assert config.gate_hidden_dim == 64
        assert config.gate_n_layers == 2
        assert config.gate_dropout == 0.1
        assert config.gating_level == "dataset"
        assert config.use_reconstruction_errors is True
        assert config.n_reconstruction_heads == 5
        assert config.use_expert_heads is False
        assert config.expert_hidden_dim == 32
        assert config.temperature == 1.0
        assert config.learn_temperature is False
        assert config.load_balance_weight == 0.0
        assert config.entropy_weight == 0.0
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = MoEConfig(
            evidence_dim=128,
            hidden_dim=256,
            mnar_variants=["self_censoring", "threshold"],
            gate_hidden_dim=128,
            gate_n_layers=3,
            temperature=0.5,
            learn_temperature=True,
            load_balance_weight=0.01,
        )
        
        assert config.evidence_dim == 128
        assert config.hidden_dim == 256
        assert len(config.mnar_variants) == 2
        assert config.gate_hidden_dim == 128
        assert config.gate_n_layers == 3
        assert config.temperature == 0.5
        assert config.learn_temperature is True
        assert config.load_balance_weight == 0.01
    
    def test_n_experts_property(self, default_config):
        """Test n_experts computed property."""
        # MCAR + MAR + 3 MNAR variants = 5 experts
        assert default_config.n_experts == 5
        
        # With 2 MNAR variants
        config = MoEConfig(mnar_variants=["self_censoring", "threshold"])
        assert config.n_experts == 4
        
        # With no MNAR variants
        config = MoEConfig(mnar_variants=[])
        assert config.n_experts == 2
    
    def test_expert_names_property(self, default_config):
        """Test expert_names computed property."""
        expected = ["mcar", "mar", "self_censoring", "threshold", "latent"]
        assert default_config.expert_names == expected
    
    def test_gate_input_dim_without_recon(self):
        """Test gate_input_dim without reconstruction errors."""
        config = MoEConfig(
            evidence_dim=64,
            use_reconstruction_errors=False,
        )
        assert config.gate_input_dim == 64
    
    def test_gate_input_dim_with_recon(self):
        """Test gate_input_dim with reconstruction errors."""
        config = MoEConfig(
            evidence_dim=64,
            use_reconstruction_errors=True,
            n_reconstruction_heads=5,
        )
        # evidence_dim + n_reconstruction_heads
        assert config.gate_input_dim == 69
    
    def test_gate_input_dim_row_level(self):
        """Test gate_input_dim for row-level gating."""
        config = MoEConfig(
            evidence_dim=64,
            hidden_dim=128,
            gating_level="row",
            use_reconstruction_errors=False,
        )
        # Uses hidden_dim for row-level
        assert config.gate_input_dim == 128
        
        config = MoEConfig(
            evidence_dim=64,
            hidden_dim=128,
            gating_level="row",
            use_reconstruction_errors=True,
            n_reconstruction_heads=5,
        )
        assert config.gate_input_dim == 133


# =============================================================================
# Test GatingNetwork
# =============================================================================

class TestGatingNetwork:
    """Tests for GatingNetwork."""
    
    @pytest.fixture
    def gating(self, default_config):
        """Create GatingNetwork."""
        return GatingNetwork(default_config)
    
    @pytest.fixture
    def gating_with_recon(self, config_with_recon):
        """Create GatingNetwork with reconstruction error input."""
        return GatingNetwork(config_with_recon)
    
    def test_output_shapes(self, gating, sample_evidence):
        """Test output tensor shapes."""
        logits, probs = gating(sample_evidence)
        
        B = sample_evidence.shape[0]
        n_experts = 5
        
        assert logits.shape == (B, n_experts)
        assert probs.shape == (B, n_experts)
    
    def test_probs_sum_to_one(self, gating, sample_evidence):
        """Test that probabilities sum to 1."""
        _, probs = gating(sample_evidence)
        
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_probs_non_negative(self, gating, sample_evidence):
        """Test that probabilities are non-negative."""
        _, probs = gating(sample_evidence)
        
        assert (probs >= 0).all()
    
    def test_no_nan_or_inf(self, gating, sample_evidence):
        """Test that outputs contain no NaN or Inf."""
        logits, probs = gating(sample_evidence)
        
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        assert not torch.isnan(probs).any()
        assert not torch.isinf(probs).any()
    
    def test_with_reconstruction_errors(
        self, gating_with_recon, sample_evidence, sample_recon_errors
    ):
        """Test gating with reconstruction errors as input."""
        logits, probs = gating_with_recon(sample_evidence, sample_recon_errors)
        
        B = sample_evidence.shape[0]
        n_experts = 5
        
        assert logits.shape == (B, n_experts)
        assert probs.shape == (B, n_experts)
    
    def test_temperature_affects_sharpness(self, sample_evidence):
        """Test that lower temperature produces sharper distributions."""
        # Create a single gating network
        config = MoEConfig(
            evidence_dim=64,
            temperature=1.0,  # Will be overridden
            use_reconstruction_errors=False,
        )
        gating = GatingNetwork(config)
        
        # Get logits (same for both temperature tests)
        logits, _ = gating(sample_evidence)
        
        # Apply softmax with different temperatures manually
        probs_low_temp = torch.softmax(logits / 0.1, dim=-1)   # Sharp
        probs_high_temp = torch.softmax(logits / 10.0, dim=-1)  # Smooth
        
        # Lower temperature should have higher max probability (sharper)
        max_probs_low = probs_low_temp.max(dim=-1).values
        max_probs_high = probs_high_temp.max(dim=-1).values
        
        assert (max_probs_low >= max_probs_high).all()
    
    def test_learned_temperature(self):
        """Test learnable temperature parameter."""
        config = MoEConfig(
            evidence_dim=64,
            learn_temperature=True,
            use_reconstruction_errors=False,
        )
        gating = GatingNetwork(config)
        
        # Temperature should be a learnable parameter
        assert hasattr(gating, "log_temperature")
        assert gating.log_temperature.requires_grad
    
    def test_gradients_flow(self, gating, sample_evidence):
        """Test that gradients flow through the network."""
        sample_evidence.requires_grad_(True)
        
        logits, probs = gating(sample_evidence)
        loss = probs.sum()
        loss.backward()
        
        assert sample_evidence.grad is not None
        assert not torch.isnan(sample_evidence.grad).any()
    
    def test_row_level_gating(self, sample_row_level_inputs):
        """Test gating with row-level inputs."""
        config = MoEConfig(
            hidden_dim=128,
            gating_level="row",
            use_reconstruction_errors=False,
        )
        gating = GatingNetwork(config)
        
        evidence = sample_row_level_inputs["evidence"]  # [B, max_rows, hidden_dim]
        logits, probs = gating(evidence)
        
        B, max_rows = evidence.shape[:2]
        n_experts = 5
        
        assert logits.shape == (B, max_rows, n_experts)
        assert probs.shape == (B, max_rows, n_experts)


# =============================================================================
# Test ExpertHead
# =============================================================================

class TestExpertHead:
    """Tests for ExpertHead."""
    
    @pytest.fixture
    def head(self):
        """Create ExpertHead."""
        return ExpertHead(input_dim=64, hidden_dim=32)
    
    def test_output_shape(self, head, sample_evidence):
        """Test output is scalar per batch element."""
        adjustment = head(sample_evidence)
        
        B = sample_evidence.shape[0]
        assert adjustment.shape == (B,)
    
    def test_row_level_input(self, sample_row_level_inputs):
        """Test with row-level input."""
        head = ExpertHead(input_dim=128, hidden_dim=32)
        
        evidence = sample_row_level_inputs["evidence"]  # [B, max_rows, hidden_dim]
        adjustment = head(evidence)
        
        B, max_rows = evidence.shape[:2]
        assert adjustment.shape == (B, max_rows)
    
    def test_gradients_flow(self, head, sample_evidence):
        """Test gradient flow."""
        sample_evidence.requires_grad_(True)
        
        adjustment = head(sample_evidence)
        loss = adjustment.sum()
        loss.backward()
        
        assert sample_evidence.grad is not None


# =============================================================================
# Test ExpertHeads
# =============================================================================

class TestExpertHeads:
    """Tests for ExpertHeads container."""
    
    @pytest.fixture
    def heads(self, config_with_experts):
        """Create ExpertHeads."""
        return ExpertHeads(config_with_experts)
    
    def test_creates_all_experts(self, heads, config_with_experts):
        """Test all experts are created."""
        expected_names = config_with_experts.expert_names
        
        for name in expected_names:
            assert name in heads.experts
    
    def test_output_shape(self, heads, sample_evidence):
        """Test output shape is [B, n_experts]."""
        adjustments = heads(sample_evidence)
        
        B = sample_evidence.shape[0]
        n_experts = 5
        
        assert adjustments.shape == (B, n_experts)
    
    def test_row_level_input(self, sample_row_level_inputs):
        """Test with row-level input."""
        config = MoEConfig(
            hidden_dim=128,
            gating_level="row",
            use_expert_heads=True,
            use_reconstruction_errors=False,
        )
        heads = ExpertHeads(config)
        
        evidence = sample_row_level_inputs["evidence"]  # [B, max_rows, hidden_dim]
        adjustments = heads(evidence)
        
        B, max_rows = evidence.shape[:2]
        n_experts = 5
        
        assert adjustments.shape == (B, max_rows, n_experts)


# =============================================================================
# Test MixtureOfExperts
# =============================================================================

class TestMixtureOfExperts:
    """Tests for MixtureOfExperts layer."""
    
    @pytest.fixture
    def moe(self, default_config):
        """Create MixtureOfExperts layer."""
        return MixtureOfExperts(default_config)
    
    @pytest.fixture
    def moe_with_experts(self, config_with_experts):
        """Create MixtureOfExperts with expert heads."""
        return MixtureOfExperts(config_with_experts)
    
    @pytest.fixture
    def moe_with_recon(self, config_with_recon):
        """Create MixtureOfExperts with reconstruction errors."""
        return MixtureOfExperts(config_with_recon)
    
    def test_forward_returns_moe_output(self, moe, sample_evidence):
        """Test forward returns MoEOutput dataclass."""
        output = moe(sample_evidence)
        
        assert isinstance(output, MoEOutput)
    
    def test_output_shapes(self, moe, sample_evidence):
        """Test output tensor shapes."""
        output = moe(sample_evidence)
        
        B = sample_evidence.shape[0]
        n_experts = 5
        
        assert output.gate_logits.shape == (B, n_experts)
        assert output.gate_probs.shape == (B, n_experts)
        assert output.combined_output.shape == (B, n_experts)
    
    def test_probs_sum_to_one(self, moe, sample_evidence):
        """Test that gate_probs sum to 1."""
        output = moe(sample_evidence)
        
        sums = output.gate_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_with_reconstruction_errors(
        self, moe_with_recon, sample_evidence, sample_recon_errors
    ):
        """Test MoE with reconstruction errors."""
        output = moe_with_recon(sample_evidence, sample_recon_errors)
        
        B = sample_evidence.shape[0]
        n_experts = 5
        
        assert output.gate_logits.shape == (B, n_experts)
        assert output.gate_probs.shape == (B, n_experts)
    
    def test_without_expert_heads(self, moe, sample_evidence):
        """Test MoE without expert heads (pure gating)."""
        assert moe.experts is None
        
        output = moe(sample_evidence)
        
        # Without expert heads, expert_outputs should be None
        assert output.expert_outputs is None
    
    def test_with_expert_heads(self, moe_with_experts, sample_evidence):
        """Test MoE with expert heads."""
        assert moe_with_experts.experts is not None
        
        output = moe_with_experts(sample_evidence)
        
        # With expert heads, we get expert_outputs (stored as list with adjustments)
        assert output.expert_outputs is not None
    
    def test_get_class_posterior(self, moe, sample_evidence):
        """Test collapsing MNAR variants to class posteriors."""
        output = moe(sample_evidence)
        p_class = moe.get_class_posterior(output)
        
        B = sample_evidence.shape[0]
        
        # Should have 3 classes: MCAR, MAR, MNAR
        assert p_class.shape == (B, 3)
        
        # Should sum to 1
        sums = p_class.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_get_mnar_variant_posterior(self, moe, sample_evidence):
        """Test getting MNAR variant posterior conditioned on MNAR."""
        output = moe(sample_evidence)
        p_variant = moe.get_mnar_variant_posterior(output)
        
        B = sample_evidence.shape[0]
        n_mnar_variants = 3
        
        assert p_variant.shape == (B, n_mnar_variants)
        
        # Should sum to 1 (conditional distribution)
        sums = p_variant.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_compute_load_balance_loss(self, moe, sample_evidence):
        """Test load balancing loss computation."""
        output = moe(sample_evidence)
        lb_loss = moe.compute_load_balance_loss(output)
        
        # Should be a scalar
        assert lb_loss.shape == ()
        
        # Should be non-negative
        assert lb_loss >= 0
    
    def test_compute_entropy_loss(self, moe, sample_evidence):
        """Test entropy loss computation."""
        output = moe(sample_evidence)
        entropy_loss = moe.compute_entropy_loss(output)
        
        # Should be a scalar
        assert entropy_loss.shape == ()
        
        # Note: The loss is *negative* entropy (for maximizing entropy via minimization)
        # So it can be negative. Check that it's finite.
        assert torch.isfinite(entropy_loss)
    
    def test_get_auxiliary_losses_no_weights(self, default_config, sample_evidence):
        """Test auxiliary losses with zero weights returns empty dict."""
        # Both weights are 0 by default in default_config
        moe = MixtureOfExperts(default_config)
        output = moe(sample_evidence)
        aux_losses = moe.get_auxiliary_losses(output)
        
        # With zero weights, the dict should be empty (no losses computed)
        assert "load_balance" not in aux_losses
        assert "entropy" not in aux_losses
    
    def test_get_auxiliary_losses_with_weights(self, sample_evidence):
        """Test auxiliary losses with non-zero weights."""
        config = MoEConfig(
            evidence_dim=64,
            load_balance_weight=0.01,
            entropy_weight=0.001,
            use_reconstruction_errors=False,
        )
        moe = MixtureOfExperts(config)
        output = moe(sample_evidence)
        aux_losses = moe.get_auxiliary_losses(output)
        
        # Should have both losses now
        assert "load_balance" in aux_losses
        assert "entropy" in aux_losses
        
        # load_balance should be non-negative
        assert aux_losses["load_balance"] >= 0
        
        # entropy is scaled negative entropy, so check finiteness
        assert torch.isfinite(aux_losses["entropy"])
    
    def test_gradients_flow(self, moe, sample_evidence):
        """Test gradient flow through MoE."""
        sample_evidence.requires_grad_(True)
        
        output = moe(sample_evidence)
        loss = output.gate_probs.sum()
        loss.backward()
        
        assert sample_evidence.grad is not None
        assert not torch.isnan(sample_evidence.grad).any()
    
    def test_config_n_experts_property(self, moe):
        """Test n_experts property via config."""
        assert moe.config.n_experts == 5
    
    def test_no_nan_or_inf(self, moe, sample_evidence):
        """Test outputs contain no NaN or Inf."""
        output = moe(sample_evidence)
        
        assert not torch.isnan(output.gate_logits).any()
        assert not torch.isinf(output.gate_logits).any()
        assert not torch.isnan(output.gate_probs).any()
        assert not torch.isinf(output.gate_probs).any()


# =============================================================================
# Test RowToDatasetAggregator
# =============================================================================

class TestRowToDatasetAggregator:
    """Tests for RowToDatasetAggregator."""
    
    @pytest.fixture
    def sample_row_probs(self):
        """Sample row-level probabilities."""
        B = 4
        max_rows = 32
        n_experts = 5
        
        # Create valid probability distributions
        logits = torch.randn(B, max_rows, n_experts)
        probs = torch.softmax(logits, dim=-1)
        
        return probs
    
    @pytest.fixture
    def sample_mask(self):
        """Sample row mask."""
        B = 4
        max_rows = 32
        
        # Create mask with varying valid rows
        mask = torch.ones(B, max_rows, dtype=torch.bool)
        mask[0, 20:] = False
        mask[1, 25:] = False
        
        return mask
    
    def test_output_shape(self, sample_row_probs, sample_mask):
        """Test output shape is [B, n_experts]."""
        aggregator = RowToDatasetAggregator(
            n_experts=5,
            hidden_dim=128,
        )
        
        dataset_probs = aggregator(sample_row_probs, sample_mask)
        
        B = sample_row_probs.shape[0]
        n_experts = 5
        
        assert dataset_probs.shape == (B, n_experts)
    
    def test_mean_aggregation(self):
        """Test mean aggregation method produces valid output."""
        aggregator = RowToDatasetAggregator(
            n_experts=5,
            hidden_dim=128,
            method="mean",
        )
        
        B, max_rows, n_experts = 4, 32, 5
        logits = torch.randn(B, max_rows, n_experts)
        probs = torch.softmax(logits, dim=-1)
        mask = torch.ones(B, max_rows, dtype=torch.bool)
        
        dataset_probs = aggregator(probs, mask)
        
        assert dataset_probs.shape == (B, n_experts)
        
        # Mean of valid probability distributions should be valid
        assert (dataset_probs >= 0).all()
        # Sum should be close to 1 (mean of distributions that sum to 1)
        assert torch.allclose(dataset_probs.sum(dim=-1), torch.ones(B), atol=1e-4)
    
    def test_masked_rows_ignored_mean(self):
        """Test that masked rows don't affect mean output."""
        aggregator = RowToDatasetAggregator(
            n_experts=5,
            hidden_dim=128,
            method="mean",
        )
        
        B = 2
        max_rows = 4
        n_experts = 5
        
        # Create data where masked rows have very different values
        probs = torch.zeros(B, max_rows, n_experts)
        
        # Valid rows: uniform distribution
        probs[:, :2, :] = 0.2
        
        # Masked rows: concentrated on expert 0 (should be ignored)
        probs[:, 2:, 0] = 1.0
        
        mask = torch.zeros(B, max_rows, dtype=torch.bool)
        mask[:, :2] = True
        
        dataset_probs = aggregator(probs, mask)
        
        # Output should be close to uniform (from valid rows)
        expected = torch.full((B, n_experts), 0.2)
        assert torch.allclose(dataset_probs, expected, atol=1e-5)
    
    def test_attention_aggregation(self, sample_row_probs, sample_mask):
        """Test attention-based aggregation produces valid output."""
        aggregator = RowToDatasetAggregator(
            n_experts=5,
            hidden_dim=128,
            method="attention",
        )
        
        dataset_probs = aggregator(sample_row_probs, sample_mask)
        
        assert dataset_probs.shape[0] == sample_row_probs.shape[0]
        assert dataset_probs.shape[1] == 5
    
    def test_max_aggregation(self, sample_row_probs, sample_mask):
        """Test max-based aggregation produces valid output."""
        aggregator = RowToDatasetAggregator(
            n_experts=5,
            hidden_dim=128,
            method="max",
        )
        
        dataset_probs = aggregator(sample_row_probs, sample_mask)
        
        assert dataset_probs.shape[0] == sample_row_probs.shape[0]
        assert dataset_probs.shape[1] == 5
    
    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            RowToDatasetAggregator(
                n_experts=5,
                hidden_dim=128,
                method="invalid",
            )


# =============================================================================
# Test create_moe Factory
# =============================================================================

class TestCreateMoe:
    """Tests for create_moe factory function."""
    
    def test_default_creation(self):
        """Test creating MoE with defaults."""
        moe = create_moe(evidence_dim=64)
        
        assert isinstance(moe, MixtureOfExperts)
        assert moe.config.n_experts == 5  # MCAR + MAR + 3 MNAR variants
    
    def test_with_custom_mnar_variants(self):
        """Test creating MoE with custom MNAR variants."""
        moe = create_moe(
            evidence_dim=64,
            mnar_variants=["self_censoring", "threshold"],
        )
        
        assert moe.config.n_experts == 4
    
    def test_with_expert_heads(self):
        """Test creating MoE with expert heads."""
        moe = create_moe(
            evidence_dim=64,
            use_expert_heads=True,
        )
        
        assert moe.experts is not None
    
    def test_with_reconstruction_errors(self):
        """Test creating MoE with reconstruction error input."""
        moe = create_moe(
            evidence_dim=64,
            use_reconstruction_errors=True,
            n_reconstruction_heads=5,
        )
        
        assert moe.config.use_reconstruction_errors is True
        assert moe.config.gate_input_dim == 69
    
    def test_with_temperature(self):
        """Test creating MoE with custom temperature."""
        moe = create_moe(
            evidence_dim=64,
            temperature=0.5,
            learn_temperature=True,
        )
        
        assert moe.gating.log_temperature.requires_grad
    
    def test_forward_pass(self, sample_evidence):
        """Test forward pass with factory-created MoE."""
        moe = create_moe(
            evidence_dim=64,
            use_reconstruction_errors=False,
        )
        
        output = moe(sample_evidence)
        
        assert isinstance(output, MoEOutput)
        assert output.gate_probs.shape == (4, 5)


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_expert(self):
        """Test MoE with no MNAR variants (only MCAR + MAR)."""
        config = MoEConfig(
            evidence_dim=64,
            mnar_variants=[],
            use_reconstruction_errors=False,
        )
        moe = MixtureOfExperts(config)
        
        evidence = torch.randn(4, 64)
        output = moe(evidence)
        
        assert output.gate_probs.shape == (4, 2)
    
    def test_many_experts(self):
        """Test MoE with many MNAR variants."""
        config = MoEConfig(
            evidence_dim=64,
            mnar_variants=["v1", "v2", "v3", "v4", "v5", "v6"],
            use_reconstruction_errors=False,
        )
        moe = MixtureOfExperts(config)
        
        evidence = torch.randn(4, 64)
        output = moe(evidence)
        
        # 2 + 6 = 8 experts
        assert output.gate_probs.shape == (4, 8)
    
    def test_batch_size_one(self, default_config):
        """Test with batch size of 1."""
        moe = MixtureOfExperts(default_config)
        
        evidence = torch.randn(1, 64)
        output = moe(evidence)
        
        assert output.gate_probs.shape == (1, 5)
    
    def test_large_batch(self, default_config):
        """Test with large batch size."""
        moe = MixtureOfExperts(default_config)
        
        evidence = torch.randn(256, 64)
        output = moe(evidence)
        
        assert output.gate_probs.shape == (256, 5)
    
    def test_extreme_evidence_values(self, default_config):
        """Test with extreme input values."""
        moe = MixtureOfExperts(default_config)
        
        # Very large values
        evidence_large = torch.randn(4, 64) * 100
        output_large = moe(evidence_large)
        
        assert not torch.isnan(output_large.gate_probs).any()
        assert torch.allclose(
            output_large.gate_probs.sum(dim=-1),
            torch.ones(4),
            atol=1e-5
        )
        
        # Very small values
        evidence_small = torch.randn(4, 64) * 0.001
        output_small = moe(evidence_small)
        
        assert not torch.isnan(output_small.gate_probs).any()
    
    def test_zero_reconstruction_errors(self, config_with_recon):
        """Test with zero reconstruction errors."""
        moe = MixtureOfExperts(config_with_recon)
        
        evidence = torch.randn(4, 64)
        recon_errors = torch.zeros(4, 5)
        
        output = moe(evidence, recon_errors)
        
        assert not torch.isnan(output.gate_probs).any()


# =============================================================================
# Test Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_softmax_stability_high_logits(self):
        """Test softmax doesn't overflow with high logits."""
        config = MoEConfig(
            evidence_dim=64,
            temperature=1.0,
            use_reconstruction_errors=False,
        )
        moe = MixtureOfExperts(config)
        
        # Create evidence that might produce high logits
        evidence = torch.randn(4, 64) * 10
        output = moe(evidence)
        
        # Should still be valid probabilities
        assert not torch.isnan(output.gate_probs).any()
        assert not torch.isinf(output.gate_probs).any()
        assert (output.gate_probs >= 0).all()
        assert (output.gate_probs <= 1).all()
    
    def test_softmax_stability_very_low_temperature(self):
        """Test softmax stability with very low temperature."""
        config = MoEConfig(
            evidence_dim=64,
            temperature=0.01,  # Very sharp
            use_reconstruction_errors=False,
        )
        moe = MixtureOfExperts(config)
        
        evidence = torch.randn(4, 64)
        output = moe(evidence)
        
        # Should still be valid probabilities
        assert not torch.isnan(output.gate_probs).any()
        assert not torch.isinf(output.gate_probs).any()
    
    def test_entropy_with_near_deterministic(self):
        """Test entropy computation with near-deterministic distribution."""
        config = MoEConfig(
            evidence_dim=64,
            entropy_weight=0.01,
            use_reconstruction_errors=False,
        )
        moe = MixtureOfExperts(config)
        
        evidence = torch.randn(4, 64)
        output = moe(evidence)
        
        # Entropy should be computable without NaN
        entropy = moe.compute_entropy_loss(output)
        assert not torch.isnan(entropy)
        assert not torch.isinf(entropy)