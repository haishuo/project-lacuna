"""
Tests for lacuna.models.assembly

Tests the complete model assembly:
    - LacunaModelConfig: Configuration with derived configs
    - BayesOptimalDecision: Decision rule from posteriors
    - compute_entropy: Entropy computation utility
    - LacunaModel: Full model combining all components
    - Factory functions: create_lacuna_model, create_lacuna_mini, create_lacuna_base, create_lacuna_large
"""

import pytest
import torch
import torch.nn as nn

from lacuna.models.assembly import (
    LacunaModelConfig,
    BayesOptimalDecision,
    compute_entropy,
    LacunaModel,
    create_lacuna_model,
    create_lacuna_mini,
    create_lacuna_base,
    create_lacuna_large,
)
from lacuna.models.encoder import EncoderConfig
from lacuna.models.reconstruction import ReconstructionConfig
from lacuna.models.moe import MoEConfig
from lacuna.core.types import (
    TokenBatch,
    PosteriorResult,
    Decision,
    ReconstructionResult,
    MoEOutput,
    LacunaOutput,
    MCAR,
    MAR,
    MNAR,
)
from lacuna.data.tokenization import TOKEN_DIM


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default LacunaModelConfig for testing."""
    return LacunaModelConfig(
        hidden_dim=64,
        evidence_dim=32,
        n_layers=2,
        n_heads=2,
        max_cols=16,
        mnar_variants=["self_censoring", "threshold"],
    )


@pytest.fixture
def mini_config():
    """Minimal config for fast tests."""
    return LacunaModelConfig(
        hidden_dim=32,
        evidence_dim=16,
        n_layers=1,
        n_heads=2,
        max_cols=8,
        row_pooling="mean",
        dataset_pooling="mean",
        recon_head_hidden_dim=16,
        recon_n_head_layers=1,
        gate_hidden_dim=16,
        gate_n_layers=1,
        mnar_variants=["self_censoring"],
        use_reconstruction_errors=False,
        use_expert_heads=False,
    )


@pytest.fixture
def sample_batch():
    """Create sample TokenBatch for testing."""
    B, max_rows, max_cols = 4, 32, 16
    
    tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
    row_mask = torch.ones(B, max_rows, dtype=torch.bool)
    col_mask = torch.ones(B, max_cols, dtype=torch.bool)
    
    # Mask out some rows/cols to test variable-size handling
    row_mask[:, 20:] = False
    col_mask[:, 10:] = False
    
    # Add reconstruction targets
    original_values = torch.randn(B, max_rows, max_cols)
    reconstruction_mask = torch.rand(B, max_rows, max_cols) > 0.7
    
    return TokenBatch(
        tokens=tokens,
        row_mask=row_mask,
        col_mask=col_mask,
        class_ids=torch.randint(0, 3, (B,)),
        variant_ids=torch.zeros(B, dtype=torch.long),
        original_values=original_values,
        reconstruction_mask=reconstruction_mask,
    )


@pytest.fixture
def sample_batch_mini():
    """Create smaller sample TokenBatch for mini model tests."""
    B, max_rows, max_cols = 2, 16, 8
    
    tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
    row_mask = torch.ones(B, max_rows, dtype=torch.bool)
    col_mask = torch.ones(B, max_cols, dtype=torch.bool)
    
    return TokenBatch(
        tokens=tokens,
        row_mask=row_mask,
        col_mask=col_mask,
        class_ids=torch.randint(0, 3, (B,)),
    )


# =============================================================================
# Test LacunaModelConfig
# =============================================================================

class TestLacunaModelConfig:
    """Tests for LacunaModelConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = LacunaModelConfig()
        
        assert config.hidden_dim == 128
        assert config.evidence_dim == 64
        assert config.n_layers == 4
        assert config.n_heads == 4
        assert config.max_cols == 32
        assert config.row_pooling == "attention"
        assert config.dataset_pooling == "attention"
        assert config.dropout == 0.1
    
    def test_custom_values(self, default_config):
        """Test custom configuration values."""
        assert default_config.hidden_dim == 64
        assert default_config.evidence_dim == 32
        assert default_config.n_layers == 2
        assert default_config.n_heads == 2
        assert default_config.max_cols == 16
        assert default_config.mnar_variants == ["self_censoring", "threshold"]
    
    def test_default_mnar_variants(self):
        """Test default MNAR variants are set in __post_init__."""
        config = LacunaModelConfig()
        
        assert config.mnar_variants == ["self_censoring", "threshold", "latent"]
    
    def test_default_loss_matrix(self):
        """Test default loss matrix is set in __post_init__."""
        config = LacunaModelConfig()
        
        assert config.loss_matrix is not None
        assert len(config.loss_matrix) == 9  # 3x3 flattened
    
    def test_n_experts_property(self, default_config):
        """Test n_experts computed property."""
        # MCAR + MAR + 2 MNAR variants = 4 experts
        assert default_config.n_experts == 4
        
        # With 3 MNAR variants
        config = LacunaModelConfig(mnar_variants=["a", "b", "c"])
        assert config.n_experts == 5
    
    def test_n_reconstruction_heads_property(self, default_config):
        """Test n_reconstruction_heads equals n_experts."""
        assert default_config.n_reconstruction_heads == default_config.n_experts
    
    def test_get_encoder_config(self, default_config):
        """Test get_encoder_config creates valid EncoderConfig."""
        encoder_config = default_config.get_encoder_config()
        
        assert isinstance(encoder_config, EncoderConfig)
        assert encoder_config.hidden_dim == default_config.hidden_dim
        assert encoder_config.evidence_dim == default_config.evidence_dim
        assert encoder_config.n_layers == default_config.n_layers
        assert encoder_config.n_heads == default_config.n_heads
        assert encoder_config.max_cols == default_config.max_cols
        assert encoder_config.dropout == default_config.dropout
        assert encoder_config.row_pooling == default_config.row_pooling
        assert encoder_config.dataset_pooling == default_config.dataset_pooling
    
    def test_get_reconstruction_config(self, default_config):
        """Test get_reconstruction_config creates valid ReconstructionConfig."""
        recon_config = default_config.get_reconstruction_config()
        
        assert isinstance(recon_config, ReconstructionConfig)
        assert recon_config.hidden_dim == default_config.hidden_dim
        assert recon_config.head_hidden_dim == default_config.recon_head_hidden_dim
        assert recon_config.n_head_layers == default_config.recon_n_head_layers
        assert recon_config.dropout == default_config.dropout
        assert recon_config.mnar_variants == default_config.mnar_variants
    
    def test_get_moe_config(self, default_config):
        """Test get_moe_config creates valid MoEConfig."""
        moe_config = default_config.get_moe_config()
        
        assert isinstance(moe_config, MoEConfig)
        assert moe_config.evidence_dim == default_config.evidence_dim
        assert moe_config.hidden_dim == default_config.hidden_dim
        assert moe_config.mnar_variants == default_config.mnar_variants
        assert moe_config.gate_hidden_dim == default_config.gate_hidden_dim
        assert moe_config.gate_n_layers == default_config.gate_n_layers
        assert moe_config.gating_level == default_config.gating_level
        assert moe_config.use_reconstruction_errors == default_config.use_reconstruction_errors
        assert moe_config.n_reconstruction_heads == default_config.n_reconstruction_heads
        assert moe_config.use_expert_heads == default_config.use_expert_heads
        assert moe_config.temperature == default_config.temperature
        assert moe_config.learn_temperature == default_config.learn_temperature


# =============================================================================
# Test BayesOptimalDecision
# =============================================================================

class TestBayesOptimalDecision:
    """Tests for BayesOptimalDecision."""
    
    @pytest.fixture
    def decision_rule(self):
        """Create BayesOptimalDecision with default loss matrix."""
        # Default loss matrix:
        #          MCAR  MAR  MNAR
        # Green:    0     0    10
        # Yellow:   1     1     2
        # Red:      3     2     0
        loss_matrix = torch.tensor([
            [0.0, 0.0, 10.0],  # Green
            [1.0, 1.0,  2.0],  # Yellow
            [3.0, 2.0,  0.0],  # Red
        ])
        return BayesOptimalDecision(loss_matrix)
    
    def test_output_type(self, decision_rule):
        """Test that forward returns Decision."""
        B = 4
        p_class = torch.softmax(torch.randn(B, 3), dim=-1)
        
        decision = decision_rule(p_class)
        
        assert isinstance(decision, Decision)
    
    def test_output_shapes(self, decision_rule):
        """Test output tensor shapes."""
        B = 4
        p_class = torch.softmax(torch.randn(B, 3), dim=-1)
        
        decision = decision_rule(p_class)
        
        assert decision.action_ids.shape == (B,)
        assert decision.expected_risks.shape == (B,)
        assert decision.confidence.shape == (B,)
    
    def test_action_ids_in_valid_range(self, decision_rule):
        """Test that action IDs are in {0, 1, 2}."""
        B = 100
        p_class = torch.softmax(torch.randn(B, 3), dim=-1)
        
        decision = decision_rule(p_class)
        
        assert (decision.action_ids >= 0).all()
        assert (decision.action_ids < 3).all()
    
    def test_certain_mcar_gives_green(self, decision_rule):
        """Test that certain MCAR posterior gives Green action."""
        B = 4
        # 100% probability on MCAR
        p_class = torch.zeros(B, 3)
        p_class[:, MCAR] = 1.0
        
        decision = decision_rule(p_class)
        
        # Green has 0 loss for MCAR, should be selected
        assert (decision.action_ids == 0).all()  # Green
        assert torch.allclose(decision.expected_risks, torch.zeros(B))
    
    def test_certain_mnar_gives_red(self, decision_rule):
        """Test that certain MNAR posterior gives Red action."""
        B = 4
        # 100% probability on MNAR
        p_class = torch.zeros(B, 3)
        p_class[:, MNAR] = 1.0
        
        decision = decision_rule(p_class)
        
        # Red has 0 loss for MNAR, should be selected
        assert (decision.action_ids == 2).all()  # Red
        assert torch.allclose(decision.expected_risks, torch.zeros(B))
    
    def test_uncertain_may_give_yellow(self, decision_rule):
        """Test that uncertain posteriors may give Yellow."""
        B = 4
        # Equal probability across all classes
        p_class = torch.ones(B, 3) / 3
        
        decision = decision_rule(p_class)
        
        # Yellow has moderate loss across all classes
        # Expected losses:
        # Green: (0 + 0 + 10) / 3 = 3.33
        # Yellow: (1 + 1 + 2) / 3 = 1.33
        # Red: (3 + 2 + 0) / 3 = 1.67
        # Yellow should be selected
        assert (decision.action_ids == 1).all()  # Yellow
    
    def test_expected_risk_computation(self, decision_rule):
        """Test that expected risks are computed correctly."""
        B = 2
        # Known probabilities
        p_class = torch.tensor([
            [1.0, 0.0, 0.0],  # 100% MCAR
            [0.0, 0.0, 1.0],  # 100% MNAR
        ])
        
        decision = decision_rule(p_class)
        
        # MCAR: Green selected, risk = 0
        # MNAR: Red selected, risk = 0
        expected_risks = torch.tensor([0.0, 0.0])
        assert torch.allclose(decision.expected_risks, expected_risks)
    
    def test_confidence_in_valid_range(self, decision_rule):
        """Test that confidence is in [0, 1]."""
        B = 100
        p_class = torch.softmax(torch.randn(B, 3), dim=-1)
        
        decision = decision_rule(p_class)
        
        assert (decision.confidence >= 0).all()
        assert (decision.confidence <= 1).all()
    
    def test_forward_with_all_risks(self, decision_rule):
        """Test forward_with_all_risks returns all risks."""
        B = 4
        p_class = torch.softmax(torch.randn(B, 3), dim=-1)
        
        decision, all_risks = decision_rule.forward_with_all_risks(p_class)
        
        assert isinstance(decision, Decision)
        assert all_risks.shape == (B, 3)  # [B, n_actions]
    
    def test_action_names_accessible(self, decision_rule):
        """Test that action names are accessible."""
        assert decision_rule.action_names == ("Green", "Yellow", "Red")
    
    def test_gradients_do_not_flow(self, decision_rule):
        """Test that decision rule doesn't break gradient flow."""
        B = 4
        p_class = torch.softmax(torch.randn(B, 3, requires_grad=True), dim=-1)
        
        decision = decision_rule(p_class)
        
        # expected_risks should have gradients (needed for training)
        loss = decision.expected_risks.sum()
        loss.backward()
        
        # p_class should have gradients
        # (Note: action_ids are discrete, so no gradient there)


# =============================================================================
# Test compute_entropy
# =============================================================================

class TestComputeEntropy:
    """Tests for compute_entropy utility."""
    
    def test_uniform_distribution_max_entropy(self):
        """Test that uniform distribution has maximum entropy."""
        B = 4
        n_classes = 3
        
        # Uniform distribution
        probs = torch.ones(B, n_classes) / n_classes
        entropy = compute_entropy(probs)
        
        # Max entropy for 3 classes is log(3)
        max_entropy = torch.log(torch.tensor(float(n_classes)))
        assert torch.allclose(entropy, max_entropy.expand(B), atol=1e-5)
    
    def test_certain_distribution_zero_entropy(self):
        """Test that certain distribution has zero entropy."""
        B = 4
        n_classes = 3
        
        # Certain distribution (all mass on one class)
        probs = torch.zeros(B, n_classes)
        probs[:, 0] = 1.0
        
        entropy = compute_entropy(probs)
        
        assert torch.allclose(entropy, torch.zeros(B), atol=1e-5)
    
    def test_entropy_non_negative(self):
        """Test that entropy is always non-negative."""
        B = 100
        n_classes = 5
        
        probs = torch.softmax(torch.randn(B, n_classes), dim=-1)
        entropy = compute_entropy(probs)
        
        assert (entropy >= 0).all()
    
    def test_custom_dim(self):
        """Test entropy computation along custom dimension."""
        B, T, n_classes = 4, 10, 3
        
        probs = torch.softmax(torch.randn(B, T, n_classes), dim=-1)
        entropy = compute_entropy(probs, dim=-1)
        
        assert entropy.shape == (B, T)
    
    def test_handles_near_zero_probs(self):
        """Test that near-zero probabilities don't cause NaN."""
        B = 4
        n_classes = 3
        
        # Very peaked distribution
        probs = torch.zeros(B, n_classes)
        probs[:, 0] = 0.999999
        probs[:, 1] = 0.0000005
        probs[:, 2] = 0.0000005
        
        entropy = compute_entropy(probs)
        
        assert not torch.isnan(entropy).any()
        assert not torch.isinf(entropy).any()


# =============================================================================
# Test LacunaModel
# =============================================================================

class TestLacunaModel:
    """Tests for LacunaModel."""
    
    @pytest.fixture
    def model(self, default_config):
        """Create LacunaModel with default config."""
        return LacunaModel(default_config)
    
    @pytest.fixture
    def model_mini(self, mini_config):
        """Create minimal LacunaModel for fast tests."""
        return LacunaModel(mini_config)
    
    def test_has_all_components(self, model):
        """Test that model has all required components."""
        assert hasattr(model, "encoder")
        assert hasattr(model, "reconstruction")
        assert hasattr(model, "moe")
        assert hasattr(model, "decision_rule")
    
    def test_forward_returns_lacuna_output(self, model_mini, sample_batch_mini):
        """Test that forward returns LacunaOutput."""
        output = model_mini(sample_batch_mini)
        
        assert isinstance(output, LacunaOutput)
    
    def test_output_has_posterior(self, model_mini, sample_batch_mini):
        """Test that output contains posterior."""
        output = model_mini(sample_batch_mini)
        
        assert output.posterior is not None
        assert isinstance(output.posterior, PosteriorResult)
        assert output.posterior.p_class.shape == (2, 3)  # [B, 3]
    
    def test_output_has_decision(self, model_mini, sample_batch_mini):
        """Test that output contains decision."""
        output = model_mini(sample_batch_mini)
        
        assert output.decision is not None
        assert isinstance(output.decision, Decision)
        assert output.decision.action_ids.shape == (2,)  # [B]
    
    def test_output_has_evidence(self, model_mini, sample_batch_mini):
        """Test that output contains evidence vector."""
        output = model_mini(sample_batch_mini)
        
        assert output.evidence is not None
        assert output.evidence.shape == (2, 16)  # [B, evidence_dim]
    
    def test_output_has_moe_output(self, model_mini, sample_batch_mini):
        """Test that output contains MoE details."""
        output = model_mini(sample_batch_mini)
        
        assert output.moe is not None
        assert isinstance(output.moe, MoEOutput)
    
    def test_output_has_reconstruction(self, model, sample_batch):
        """Test that output contains reconstruction results."""
        output = model(sample_batch, compute_reconstruction=True)
        
        assert output.reconstruction is not None
        assert isinstance(output.reconstruction, dict)
        
        # Should have one result per head
        expected_heads = ["mcar", "mar", "self_censoring", "threshold"]
        for head_name in expected_heads:
            assert head_name in output.reconstruction
            assert isinstance(output.reconstruction[head_name], ReconstructionResult)
    
    def test_skip_reconstruction(self, model_mini, sample_batch_mini):
        """Test that reconstruction can be skipped."""
        output = model_mini(sample_batch_mini, compute_reconstruction=False)
        
        # Reconstruction should still be computed for MoE (if use_reconstruction_errors=True)
        # but with mini_config use_reconstruction_errors=False, so dict may be empty or minimal
        # The key point is the model runs without error
        assert output.posterior is not None
    
    def test_skip_decision(self, model_mini, sample_batch_mini):
        """Test that decision computation can be skipped."""
        output = model_mini(sample_batch_mini, compute_decision=False)
        
        assert output.decision is None
        assert output.posterior is not None
    
    def test_posterior_p_class_sums_to_one(self, model_mini, sample_batch_mini):
        """Test that class posterior sums to 1."""
        output = model_mini(sample_batch_mini)
        
        p_class = output.posterior.p_class
        sums = p_class.sum(dim=-1)
        
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_posterior_p_class_non_negative(self, model_mini, sample_batch_mini):
        """Test that class posterior is non-negative."""
        output = model_mini(sample_batch_mini)
        
        assert (output.posterior.p_class >= 0).all()
    
    def test_no_nan_or_inf(self, model_mini, sample_batch_mini):
        """Test that outputs contain no NaN or Inf."""
        output = model_mini(sample_batch_mini)
        
        assert not torch.isnan(output.posterior.p_class).any()
        assert not torch.isnan(output.evidence).any()
        assert not torch.isnan(output.decision.expected_risks).any()
        assert not torch.isinf(output.posterior.p_class).any()
        assert not torch.isinf(output.evidence).any()
    
    def test_gradients_flow(self, model_mini, sample_batch_mini):
        """Test that gradients flow through the model."""
        model_mini.train()
        
        output = model_mini(sample_batch_mini)
        loss = output.posterior.p_class.sum()
        loss.backward()
        
        # Check that encoder parameters have gradients
        for param in model_mini.encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break
    
    def test_get_evidence(self, model_mini, sample_batch_mini):
        """Test get_evidence helper method."""
        evidence = model_mini.get_evidence(sample_batch_mini)
        
        assert evidence.shape == (2, 16)  # [B, evidence_dim]
        assert not torch.isnan(evidence).any()
    
    def test_get_token_representations(self, model_mini, sample_batch_mini):
        """Test get_token_representations helper method."""
        token_repr = model_mini.get_token_representations(sample_batch_mini)
        
        # [B, max_rows, max_cols, hidden_dim]
        assert token_repr.shape == (2, 16, 8, 32)
    
    def test_get_row_representations(self, model_mini, sample_batch_mini):
        """Test get_row_representations helper method."""
        row_repr = model_mini.get_row_representations(sample_batch_mini)
        
        # [B, max_rows, hidden_dim]
        assert row_repr.shape == (2, 16, 32)
    
    def test_expert_to_class_buffer(self, model):
        """Test expert_to_class mapping buffer."""
        # With 2 MNAR variants: experts are MCAR, MAR, SC, Thresh
        expected = torch.tensor([MCAR, MAR, MNAR, MNAR])
        
        assert torch.equal(model.expert_to_class, expected)
    
    def test_handles_variable_batch_sizes(self, model_mini):
        """Test model handles different batch sizes."""
        for B in [1, 2, 4, 8]:
            batch = TokenBatch(
                tokens=torch.randn(B, 16, 8, TOKEN_DIM),
                row_mask=torch.ones(B, 16, dtype=torch.bool),
                col_mask=torch.ones(B, 8, dtype=torch.bool),
            )
            
            output = model_mini(batch)
            
            assert output.posterior.p_class.shape == (B, 3)
            assert output.evidence.shape == (B, 16)


# =============================================================================
# Test Factory Functions
# =============================================================================

class TestCreateLacunaModel:
    """Tests for create_lacuna_model factory."""
    
    def test_creates_model(self):
        """Test factory creates a model."""
        model = create_lacuna_model(
            hidden_dim=64,
            evidence_dim=32,
            n_layers=2,
            n_heads=2,
            max_cols=16,
        )
        
        assert isinstance(model, LacunaModel)
    
    def test_respects_parameters(self):
        """Test factory respects all parameters."""
        model = create_lacuna_model(
            hidden_dim=256,
            evidence_dim=128,
            n_layers=6,
            n_heads=8,
            max_cols=64,
            mnar_variants=["self_censoring", "threshold"],
            use_reconstruction_errors=True,
            use_expert_heads=True,
            temperature=0.5,
            learn_temperature=True,
        )
        
        assert model.config.hidden_dim == 256
        assert model.config.evidence_dim == 128
        assert model.config.n_layers == 6
        assert model.config.n_heads == 8
        assert model.config.max_cols == 64
        assert model.config.mnar_variants == ["self_censoring", "threshold"]
        assert model.config.use_reconstruction_errors is True
        assert model.config.use_expert_heads is True
        assert model.config.temperature == 0.5
        assert model.config.learn_temperature is True
    
    def test_default_mnar_variants(self):
        """Test default MNAR variants when None."""
        model = create_lacuna_model()
        
        assert model.config.mnar_variants == ["self_censoring", "threshold", "latent"]
    
    def test_custom_loss_matrix(self):
        """Test custom loss matrix."""
        custom_matrix = [
            1.0, 1.0, 5.0,   # Green
            0.5, 0.5, 1.0,   # Yellow
            2.0, 1.5, 0.0,   # Red
        ]
        
        model = create_lacuna_model(loss_matrix=custom_matrix)
        
        expected = torch.tensor(custom_matrix).reshape(3, 3)
        assert torch.allclose(model.decision_rule.loss_matrix, expected)


class TestCreateLacunaMini:
    """Tests for create_lacuna_mini factory."""
    
    def test_creates_small_model(self):
        """Test factory creates a small model."""
        model = create_lacuna_mini()
        
        assert isinstance(model, LacunaModel)
        assert model.config.hidden_dim == 64
        assert model.config.evidence_dim == 32
        assert model.config.n_layers == 2
    
    def test_forward_pass_works(self):
        """Test that mini model can do forward pass."""
        model = create_lacuna_mini(max_cols=8)
        
        batch = TokenBatch(
            tokens=torch.randn(2, 16, 8, TOKEN_DIM),
            row_mask=torch.ones(2, 16, dtype=torch.bool),
            col_mask=torch.ones(2, 8, dtype=torch.bool),
        )
        
        output = model(batch)
        
        assert output.posterior.p_class.shape == (2, 3)
    
    def test_respects_max_cols(self):
        """Test max_cols parameter is respected."""
        model = create_lacuna_mini(max_cols=16)
        
        assert model.config.max_cols == 16
    
    def test_respects_mnar_variants(self):
        """Test mnar_variants parameter is respected."""
        model = create_lacuna_mini(mnar_variants=["self_censoring"])
        
        assert model.config.mnar_variants == ["self_censoring"]
        assert model.config.n_experts == 3  # MCAR + MAR + 1


class TestCreateLacunaBase:
    """Tests for create_lacuna_base factory."""
    
    def test_creates_base_model(self):
        """Test factory creates base model."""
        model = create_lacuna_base()
        
        assert isinstance(model, LacunaModel)
        assert model.config.hidden_dim == 128
        assert model.config.evidence_dim == 64
        assert model.config.n_layers == 4
        assert model.config.row_pooling == "attention"
        assert model.config.dataset_pooling == "attention"
    
    def test_respects_parameters(self):
        """Test factory respects parameters."""
        model = create_lacuna_base(
            max_cols=64,
            mnar_variants=["self_censoring", "threshold"],
        )
        
        assert model.config.max_cols == 64
        assert model.config.mnar_variants == ["self_censoring", "threshold"]


class TestCreateLacunaLarge:
    """Tests for create_lacuna_large factory."""
    
    def test_creates_large_model(self):
        """Test factory creates large model."""
        model = create_lacuna_large()
        
        assert isinstance(model, LacunaModel)
        assert model.config.hidden_dim == 256
        assert model.config.evidence_dim == 128
        assert model.config.n_layers == 6
        assert model.config.n_heads == 8
    
    def test_uses_expert_heads(self):
        """Test large model uses expert heads."""
        model = create_lacuna_large()
        
        assert model.config.use_expert_heads is True
    
    def test_uses_learnable_temperature(self):
        """Test large model uses learnable temperature."""
        model = create_lacuna_large()
        
        assert model.config.learn_temperature is True
    
    def test_uses_load_balancing(self):
        """Test large model uses load balancing."""
        model = create_lacuna_large()
        
        assert model.config.load_balance_weight > 0


# =============================================================================
# Test Integration: Full Pipeline
# =============================================================================

class TestFullPipeline:
    """Integration tests for complete model pipeline."""
    
    def test_training_step(self):
        """Test a complete training step."""
        model = create_lacuna_mini(max_cols=8)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create batch with labels
        batch = TokenBatch(
            tokens=torch.randn(4, 16, 8, TOKEN_DIM),
            row_mask=torch.ones(4, 16, dtype=torch.bool),
            col_mask=torch.ones(4, 8, dtype=torch.bool),
            class_ids=torch.randint(0, 3, (4,)),
            original_values=torch.randn(4, 16, 8),
            reconstruction_mask=torch.rand(4, 16, 8) > 0.5,
        )
        
        # Forward pass
        output = model(batch)
        
        # Compute classification loss
        targets = batch.class_ids
        log_probs = torch.log(output.posterior.p_class.clamp(min=1e-8))
        loss = nn.functional.nll_loss(log_probs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify loss is valid
        assert not torch.isnan(loss)
        assert loss.item() >= 0
    
    def test_inference_mode(self):
        """Test model in inference mode."""
        model = create_lacuna_mini(max_cols=8)
        model.eval()
        
        batch = TokenBatch(
            tokens=torch.randn(4, 16, 8, TOKEN_DIM),
            row_mask=torch.ones(4, 16, dtype=torch.bool),
            col_mask=torch.ones(4, 8, dtype=torch.bool),
        )
        
        with torch.no_grad():
            output = model(batch)
        
        # Check outputs
        assert output.posterior.p_class.shape == (4, 3)
        assert output.decision.action_ids.shape == (4,)
        
        # Get action names
        actions = output.decision.get_actions()
        assert len(actions) == 4
        assert all(a in ["Green", "Yellow", "Red"] for a in actions)
    
    def test_deterministic_inference(self):
        """Test that inference is deterministic in eval mode."""
        model = create_lacuna_mini(max_cols=8)
        model.eval()
        
        batch = TokenBatch(
            tokens=torch.randn(2, 16, 8, TOKEN_DIM),
            row_mask=torch.ones(2, 16, dtype=torch.bool),
            col_mask=torch.ones(2, 8, dtype=torch.bool),
        )
        
        with torch.no_grad():
            output1 = model(batch)
            output2 = model(batch)
        
        assert torch.allclose(output1.posterior.p_class, output2.posterior.p_class)
        assert torch.equal(output1.decision.action_ids, output2.decision.action_ids)
    
    def test_batch_independence(self):
        """Test that samples in batch are processed independently."""
        model = create_lacuna_mini(max_cols=8)
        model.eval()
        
        # Create two separate samples
        sample1 = torch.randn(1, 16, 8, TOKEN_DIM)
        sample2 = torch.randn(1, 16, 8, TOKEN_DIM)
        
        # Process separately
        batch1 = TokenBatch(
            tokens=sample1,
            row_mask=torch.ones(1, 16, dtype=torch.bool),
            col_mask=torch.ones(1, 8, dtype=torch.bool),
        )
        batch2 = TokenBatch(
            tokens=sample2,
            row_mask=torch.ones(1, 16, dtype=torch.bool),
            col_mask=torch.ones(1, 8, dtype=torch.bool),
        )
        
        with torch.no_grad():
            out1 = model(batch1)
            out2 = model(batch2)
        
        # Process together
        batch_combined = TokenBatch(
            tokens=torch.cat([sample1, sample2], dim=0),
            row_mask=torch.ones(2, 16, dtype=torch.bool),
            col_mask=torch.ones(2, 8, dtype=torch.bool),
        )
        
        with torch.no_grad():
            out_combined = model(batch_combined)
        
        # Results should match
        assert torch.allclose(
            out1.posterior.p_class, 
            out_combined.posterior.p_class[0:1],
            atol=1e-5
        )
        assert torch.allclose(
            out2.posterior.p_class, 
            out_combined.posterior.p_class[1:2],
            atol=1e-5
        )