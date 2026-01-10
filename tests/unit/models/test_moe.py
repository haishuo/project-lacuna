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
    """Default MoEConfig for testing."""
    return MoEConfig(
        evidence_dim=64,
        hidden_dim=128,
        mnar_variants=["self_censoring", "threshold", "latent"],
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
        assert not torch.isnan(probs).any()
        assert not torch.isinf(logits).any()
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
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)
    
    def test_temperature_property(self, default_config):
        """Test temperature property."""
        gating = GatingNetwork(default_config)
        
        assert gating.temperature.item() == pytest.approx(1.0)
    
    def test_learnable_temperature(self):
        """Test learnable temperature parameter."""
        config = MoEConfig(
            learn_temperature=True,
            temperature=2.0,
        )
        gating = GatingNetwork(config)
        
        # Temperature should be close to 2.0
        assert gating.temperature.item() == pytest.approx(2.0, rel=1e-3)
        
        # log_temperature should be a parameter
        assert isinstance(gating.log_temperature, nn.Parameter)
    
    def test_fixed_temperature(self):
        """Test fixed temperature (not learnable)."""
        config = MoEConfig(
            learn_temperature=False,
            temperature=0.5,
        )
        gating = GatingNetwork(config)
        
        # Temperature should be 0.5
        assert gating.temperature.item() == pytest.approx(0.5, rel=1e-3)
        
        # log_temperature should be a buffer, not parameter
        assert not isinstance(gating.log_temperature, nn.Parameter)
    
    def test_temperature_affects_sharpness(self, sample_evidence):
        """Test that lower temperature produces sharper distributions."""
        config_high_temp = MoEConfig(temperature=2.0)
        config_low_temp = MoEConfig(temperature=0.5)
        
        gating_high = GatingNetwork(config_high_temp)
        gating_low = GatingNetwork(config_low_temp)
        
        # Use same weights for fair comparison
        gating_low.load_state_dict(gating_high.state_dict(), strict=False)
        
        _, probs_high = gating_high(sample_evidence)
        _, probs_low = gating_low(sample_evidence)
        
        # Lower temperature → higher max probability (sharper)
        max_prob_high = probs_high.max(dim=-1).values.mean()
        max_prob_low = probs_low.max(dim=-1).values.mean()
        
        assert max_prob_low > max_prob_high
    
    def test_gradients_flow(self, gating, sample_evidence):
        """Test that gradients flow through the network."""
        sample_evidence = sample_evidence.clone().requires_grad_(True)
        
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
        return ExpertHead(input_dim=64, hidden_dim=32, dropout=0.1)
    
    def test_output_shape(self, head):
        """Test output shape is scalar per sample."""
        B = 4
        evidence = torch.randn(B, 64)
        
        adjustment = head(evidence)
        
        assert adjustment.shape == (B,)
    
    def test_row_level_input(self, head):
        """Test with row-level input shape."""
        B, max_rows = 4, 32
        evidence = torch.randn(B, max_rows, 64)
        
        adjustment = head(evidence)
        
        assert adjustment.shape == (B, max_rows)
    
    def test_no_nan_or_inf(self, head):
        """Test outputs contain no NaN or Inf."""
        evidence = torch.randn(4, 64)
        adjustment = head(evidence)
        
        assert not torch.isnan(adjustment).any()
        assert not torch.isinf(adjustment).any()
    
    def test_gradients_flow(self, head):
        """Test gradients flow through the head."""
        evidence = torch.randn(4, 64, requires_grad=True)
        
        adjustment = head(evidence)
        loss = adjustment.sum()
        loss.backward()
        
        assert evidence.grad is not None


# =============================================================================
# Test ExpertHeads Container
# =============================================================================

class TestExpertHeads:
    """Tests for ExpertHeads container."""
    
    @pytest.fixture
    def heads(self, config_with_experts):
        """Create ExpertHeads container."""
        return ExpertHeads(config_with_experts)
    
    def test_has_all_experts(self, heads, config_with_experts):
        """Test container has all expected experts."""
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
        
        assert isinstance(output, MoEOutput)
        assert output.gate_probs.shape == (sample_evidence.shape[0], 5)
    
    def test_without_expert_heads(self, moe, sample_evidence):
        """Test MoE without expert heads (pure gating)."""
        output = moe(sample_evidence)
        
        # expert_outputs should be None when no expert heads
        assert output.expert_outputs is None
        
        # combined_output should equal gate_logits
        assert torch.allclose(output.combined_output, output.gate_logits)
    
    def test_with_expert_heads(self, moe_with_experts, sample_evidence):
        """Test MoE with expert heads."""
        output = moe_with_experts(sample_evidence)
        
        # expert_outputs should contain adjustments
        assert output.expert_outputs is not None
        assert len(output.expert_outputs) == 1  # Single tensor in list
        
        # combined_output should differ from gate_logits
        # (combined = gate_logits + expert_adjustments)
        assert not torch.allclose(output.combined_output, output.gate_logits)
    
    def test_expert_to_class_buffer(self, moe):
        """Test expert_to_class mapping buffer."""
        # Experts 0, 1 are MCAR, MAR; rest are MNAR
        expected = torch.tensor([MCAR, MAR, MNAR, MNAR, MNAR])
        
        assert torch.equal(moe.expert_to_class, expected)
    
    def test_get_class_posterior(self, moe, sample_evidence):
        """Test aggregating experts to class posterior."""
        output = moe(sample_evidence)
        p_class = moe.get_class_posterior(output)
        
        B = sample_evidence.shape[0]
        
        # Should be [B, 3] for MCAR, MAR, MNAR
        assert p_class.shape == (B, 3)
        
        # Should sum to 1
        assert torch.allclose(p_class.sum(dim=-1), torch.ones(B), atol=1e-5)
        
        # MNAR probability should be sum of MNAR variant probabilities
        mnar_expert_probs = output.gate_probs[:, 2:]  # Experts 2, 3, 4 are MNAR
        assert torch.allclose(p_class[:, MNAR], mnar_expert_probs.sum(dim=-1), atol=1e-5)
    
    def test_get_mnar_variant_posterior(self, moe, sample_evidence):
        """Test extracting MNAR variant posterior."""
        output = moe(sample_evidence)
        p_variant = moe.get_mnar_variant_posterior(output)
        
        B = sample_evidence.shape[0]
        n_mnar_variants = 3
        
        # Should be [B, n_mnar_variants]
        assert p_variant.shape == (B, n_mnar_variants)
        
        # Should sum to 1 (conditional on MNAR)
        assert torch.allclose(p_variant.sum(dim=-1), torch.ones(B), atol=1e-5)
    
    def test_compute_load_balance_loss(self, moe, sample_evidence):
        """Test load balance loss computation."""
        output = moe(sample_evidence)
        loss = moe.compute_load_balance_loss(output)
        
        # Should be non-negative scalar
        assert loss.shape == ()
        assert loss >= 0
    
    def test_compute_entropy_loss(self, moe, sample_evidence):
        """Test entropy loss computation."""
        output = moe(sample_evidence)
        loss = moe.compute_entropy_loss(output)
        
        # Should be scalar
        assert loss.shape == ()
    
    def test_get_auxiliary_losses_no_weights(self, moe, sample_evidence):
        """Test auxiliary losses with zero weights."""
        output = moe(sample_evidence)
        losses = moe.get_auxiliary_losses(output)
        
        # Should be empty dict when weights are 0
        assert losses == {}
    
    def test_get_auxiliary_losses_with_weights(self, sample_evidence):
        """Test auxiliary losses with non-zero weights."""
        config = MoEConfig(
            load_balance_weight=0.01,
            entropy_weight=0.001,
        )
        moe = MixtureOfExperts(config)
        
        output = moe(sample_evidence)
        losses = moe.get_auxiliary_losses(output)
        
        assert "load_balance" in losses
        assert "entropy" in losses
        assert losses["load_balance"].shape == ()
        assert losses["entropy"].shape == ()
    
    def test_gradients_flow(self, moe, sample_evidence):
        """Test gradients flow through MoE."""
        sample_evidence = sample_evidence.clone().requires_grad_(True)
        
        output = moe(sample_evidence)
        loss = output.gate_probs.sum()
        loss.backward()
        
        assert sample_evidence.grad is not None
        assert not torch.isnan(sample_evidence.grad).any()
    
    def test_n_experts_property(self, moe, sample_evidence):
        """Test n_experts accessible via output."""
        output = moe(sample_evidence)
        
        assert output.n_experts == 5
    
    def test_no_nan_or_inf(self, moe, sample_evidence):
        """Test outputs contain no NaN or Inf."""
        output = moe(sample_evidence)
        
        assert not torch.isnan(output.gate_logits).any()
        assert not torch.isnan(output.gate_probs).any()
        assert not torch.isinf(output.gate_logits).any()
        assert not torch.isinf(output.gate_probs).any()


# =============================================================================
# Test RowToDatasetAggregator
# =============================================================================

class TestRowToDatasetAggregator:
    """Tests for RowToDatasetAggregator."""
    
    @pytest.fixture
    def sample_row_probs(self):
        """Sample row-level probabilities."""
        B, max_rows, n_experts = 4, 32, 5
        # Generate valid probabilities (sum to 1 per row)
        logits = torch.randn(B, max_rows, n_experts)
        probs = torch.softmax(logits, dim=-1)
        return probs
    
    @pytest.fixture
    def sample_row_mask(self):
        """Sample row mask."""
        B, max_rows = 4, 32
        mask = torch.ones(B, max_rows, dtype=torch.bool)
        # Mask out some rows per sample
        mask[:, 20:] = False
        return mask
    
    def test_mean_aggregation(self, sample_row_probs, sample_row_mask):
        """Test mean aggregation method."""
        aggregator = RowToDatasetAggregator(
            n_experts=5,
            hidden_dim=128,
            method="mean",
        )
        
        dataset_probs = aggregator(sample_row_probs, sample_row_mask)
        
        B, n_experts = 4, 5
        assert dataset_probs.shape == (B, n_experts)
        
        # Should sum to approximately 1
        assert torch.allclose(dataset_probs.sum(dim=-1), torch.ones(B), atol=1e-4)
    
    def test_max_aggregation(self, sample_row_probs, sample_row_mask):
        """Test max aggregation method."""
        aggregator = RowToDatasetAggregator(
            n_experts=5,
            hidden_dim=128,
            method="max",
        )
        
        dataset_probs = aggregator(sample_row_probs, sample_row_mask)
        
        B, n_experts = 4, 5
        assert dataset_probs.shape == (B, n_experts)
    
    def test_attention_aggregation(self, sample_row_probs, sample_row_mask):
        """Test attention aggregation method."""
        aggregator = RowToDatasetAggregator(
            n_experts=5,
            hidden_dim=128,
            method="attention",
        )
        
        dataset_probs = aggregator(sample_row_probs, sample_row_mask)
        
        B, n_experts = 4, 5
        assert dataset_probs.shape == (B, n_experts)
    
    def test_respects_row_mask(self, sample_row_probs, sample_row_mask):
        """Test that aggregation respects row mask."""
        aggregator = RowToDatasetAggregator(
            n_experts=5,
            hidden_dim=128,
            method="mean",
        )
        
        # Create specific mask: only first row valid
        mask = torch.zeros(4, 32, dtype=torch.bool)
        mask[:, 0] = True
        
        dataset_probs = aggregator(sample_row_probs, mask)
        
        # Result should equal first row's probabilities
        expected = sample_row_probs[:, 0, :]
        assert torch.allclose(dataset_probs, expected, atol=1e-5)
    
    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            RowToDatasetAggregator(
                n_experts=5,
                hidden_dim=128,
                method="invalid",
            )
    
    def test_no_nan_with_sparse_mask(self, sample_row_probs):
        """Test no NaN when very few rows are valid."""
        aggregator = RowToDatasetAggregator(
            n_experts=5,
            hidden_dim=128,
            method="mean",
        )
        
        # Only 1 row valid per sample
        mask = torch.zeros(4, 32, dtype=torch.bool)
        mask[:, 0] = True
        
        dataset_probs = aggregator(sample_row_probs, mask)
        
        assert not torch.isnan(dataset_probs).any()


# =============================================================================
# Test create_moe Factory
# =============================================================================

class TestCreateMoe:
    """Tests for create_moe factory function."""
    
    def test_default_creation(self):
        """Test creating MoE with defaults."""
        moe = create_moe()
        
        assert isinstance(moe, MixtureOfExperts)
        assert moe.config.evidence_dim == 64
        assert moe.config.n_experts == 5
    
    def test_custom_evidence_dim(self):
        """Test creating MoE with custom evidence dimension."""
        moe = create_moe(evidence_dim=128)
        
        assert moe.config.evidence_dim == 128
    
    def test_custom_mnar_variants(self):
        """Test creating MoE with custom MNAR variants."""
        variants = ["self_censoring", "threshold"]
        moe = create_moe(mnar_variants=variants)
        
        assert moe.config.mnar_variants == variants
        assert moe.config.n_experts == 4  # MCAR + MAR + 2 variants
    
    def test_with_reconstruction_errors(self):
        """Test creating MoE with reconstruction error input."""
        moe = create_moe(
            use_reconstruction_errors=True,
            n_reconstruction_heads=7,
        )
        
        assert moe.config.use_reconstruction_errors is True
        assert moe.config.n_reconstruction_heads == 7
    
    def test_with_expert_heads(self):
        """Test creating MoE with expert heads."""
        moe = create_moe(use_expert_heads=True)
        
        assert moe.config.use_expert_heads is True
        assert moe.experts is not None
    
    def test_without_expert_heads(self):
        """Test creating MoE without expert heads."""
        moe = create_moe(use_expert_heads=False)
        
        assert moe.config.use_expert_heads is False
        assert moe.experts is None
    
    def test_row_level_gating(self):
        """Test creating MoE for row-level gating."""
        moe = create_moe(gating_level="row", hidden_dim=256)
        
        assert moe.config.gating_level == "row"
        assert moe.config.hidden_dim == 256
    
    def test_learnable_temperature(self):
        """Test creating MoE with learnable temperature."""
        moe = create_moe(
            temperature=2.0,
            learn_temperature=True,
        )
        
        assert moe.config.learn_temperature is True
        assert moe.gating.temperature.item() == pytest.approx(2.0, rel=1e-3)
    
    def test_auxiliary_loss_weights(self):
        """Test creating MoE with auxiliary loss weights."""
        moe = create_moe(
            load_balance_weight=0.01,
            entropy_weight=0.001,
        )
        
        assert moe.config.load_balance_weight == 0.01
        assert moe.config.entropy_weight == 0.001
    
    def test_functional_forward_pass(self):
        """Test that created MoE works in forward pass."""
        moe = create_moe(
            evidence_dim=64,
            use_reconstruction_errors=True,
            n_reconstruction_heads=5,
        )
        
        B = 4
        evidence = torch.randn(B, 64)
        recon_errors = torch.abs(torch.randn(B, 5))
        
        output = moe(evidence, recon_errors)
        
        assert output.gate_probs.shape == (B, 5)
        assert torch.allclose(output.gate_probs.sum(dim=-1), torch.ones(B), atol=1e-5)


# =============================================================================
# Test Integration: MoE with Class/Variant Posteriors
# =============================================================================

class TestMoEIntegration:
    """Integration tests for MoE with posterior computations."""
    
    def test_class_posterior_sums_correctly(self):
        """Test that class posterior correctly aggregates expert probs."""
        moe = create_moe(mnar_variants=["self_censoring", "threshold", "latent"])
        
        # Create deterministic gate probs for verification
        B = 2
        # Expert probs: [MCAR, MAR, SC, Thresh, Latent]
        gate_probs = torch.tensor([
            [0.1, 0.2, 0.3, 0.25, 0.15],  # Sample 0
            [0.3, 0.3, 0.1, 0.2, 0.1],    # Sample 1
        ])
        
        # Manually create MoEOutput
        output = MoEOutput(
            gate_logits=torch.zeros_like(gate_probs),
            gate_probs=gate_probs,
            combined_output=torch.zeros_like(gate_probs),
        )
        
        p_class = moe.get_class_posterior(output)
        
        # Expected: [p_MCAR, p_MAR, p_MNAR]
        # p_MNAR = sum of MNAR variant probs
        expected = torch.tensor([
            [0.1, 0.2, 0.7],   # 0.3 + 0.25 + 0.15 = 0.7
            [0.3, 0.3, 0.4],   # 0.1 + 0.2 + 0.1 = 0.4
        ])
        
        assert torch.allclose(p_class, expected, atol=1e-5)
    
    def test_variant_posterior_normalized(self):
        """Test that MNAR variant posterior is properly normalized."""
        moe = create_moe(mnar_variants=["self_censoring", "threshold", "latent"])
        
        B = 2
        gate_probs = torch.tensor([
            [0.1, 0.2, 0.3, 0.25, 0.15],
            [0.3, 0.3, 0.1, 0.2, 0.1],
        ])
        
        output = MoEOutput(
            gate_logits=torch.zeros_like(gate_probs),
            gate_probs=gate_probs,
            combined_output=torch.zeros_like(gate_probs),
        )
        
        p_variant = moe.get_mnar_variant_posterior(output)
        
        # Should sum to 1 (conditional on MNAR)
        assert torch.allclose(p_variant.sum(dim=-1), torch.ones(B), atol=1e-5)
        
        # Verify normalization
        # Sample 0: MNAR probs = [0.3, 0.25, 0.15], sum = 0.7
        # Normalized: [0.3/0.7, 0.25/0.7, 0.15/0.7]
        expected_0 = torch.tensor([0.3/0.7, 0.25/0.7, 0.15/0.7])
        assert torch.allclose(p_variant[0], expected_0, atol=1e-5)
    
    def test_end_to_end_inference(self):
        """Test complete inference pipeline."""
        moe = create_moe(
            evidence_dim=64,
            use_reconstruction_errors=True,
            n_reconstruction_heads=5,
            use_expert_heads=True,
        )
        
        B = 8
        evidence = torch.randn(B, 64)
        recon_errors = torch.abs(torch.randn(B, 5))
        
        # Forward pass
        output = moe(evidence, recon_errors)
        
        # Get posteriors
        p_class = moe.get_class_posterior(output)
        p_variant = moe.get_mnar_variant_posterior(output)
        
        # Get auxiliary losses
        aux_losses = moe.get_auxiliary_losses(output)
        
        # All should be valid
        assert p_class.shape == (B, 3)
        assert p_variant.shape == (B, 3)
        assert torch.allclose(p_class.sum(dim=-1), torch.ones(B), atol=1e-5)
        assert torch.allclose(p_variant.sum(dim=-1), torch.ones(B), atol=1e-5)
        
        # No NaN anywhere
        assert not torch.isnan(p_class).any()
        assert not torch.isnan(p_variant).any()
        assert not torch.isnan(output.gate_probs).any()