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
from lacuna.data.tokenization import (
    TOKEN_DIM,
    IDX_VALUE,
    IDX_OBSERVED,
    IDX_MASK_TYPE,
    IDX_FEATURE_ID,
)


# =============================================================================
# Helper Functions for Creating Valid Tokens
# =============================================================================

def create_valid_tokens(B: int, max_rows: int, max_cols: int) -> torch.Tensor:
    """
    Create properly structured token tensors for testing.
    
    Tokens have structure: [value, is_observed, mask_type, feature_id_normalized]
    - value: continuous float (can be any value, normalized roughly to [-3, 3])
    - is_observed: binary 0.0 or 1.0
    - mask_type: binary 0.0 or 1.0  
    - feature_id_normalized: float in [0, 1] representing j / (max_cols - 1)
    
    The TokenEmbedding layer expects these specific ranges:
    - is_observed and mask_type are converted to .long() for embedding lookup (0 or 1)
    - feature_id_normalized is multiplied by (max_cols - 1) and converted to .long()
      for position embedding lookup, so it MUST be in [0, 1]
    
    Args:
        B: Batch size
        max_rows: Maximum number of rows
        max_cols: Maximum number of columns
    
    Returns:
        tokens: [B, max_rows, max_cols, TOKEN_DIM] properly structured tensor
    """
    tokens = torch.zeros(B, max_rows, max_cols, TOKEN_DIM)
    
    # Value: random continuous values (normalized roughly to [-3, 3])
    tokens[..., IDX_VALUE] = torch.randn(B, max_rows, max_cols)
    
    # is_observed: binary (randomly set ~80% as observed)
    tokens[..., IDX_OBSERVED] = (torch.rand(B, max_rows, max_cols) > 0.2).float()
    
    # mask_type: binary (mostly natural=0, some artificial=1)
    tokens[..., IDX_MASK_TYPE] = (torch.rand(B, max_rows, max_cols) > 0.9).float()
    
    # feature_id_normalized: float in [0, 1] representing column position
    # For each column j, feature_id = j / (max_cols - 1) if max_cols > 1, else 0
    # This is CRITICAL: the encoder uses this to look up position embeddings
    for j in range(max_cols):
        tokens[..., j, IDX_FEATURE_ID] = j / max(max_cols - 1, 1)
    
    return tokens


def create_sample_batch(
    B: int,
    max_rows: int,
    max_cols: int,
    include_reconstruction: bool = False,
    include_labels: bool = True,
) -> TokenBatch:
    """
    Create a properly structured TokenBatch for testing.
    
    Args:
        B: Batch size
        max_rows: Maximum number of rows
        max_cols: Maximum number of columns
        include_reconstruction: Whether to include reconstruction targets
        include_labels: Whether to include class labels
    
    Returns:
        TokenBatch with properly structured tokens
    """
    tokens = create_valid_tokens(B, max_rows, max_cols)
    row_mask = torch.ones(B, max_rows, dtype=torch.bool)
    col_mask = torch.ones(B, max_cols, dtype=torch.bool)
    
    kwargs = {
        "tokens": tokens,
        "row_mask": row_mask,
        "col_mask": col_mask,
    }
    
    if include_labels:
        kwargs["class_ids"] = torch.randint(0, 3, (B,))
        kwargs["variant_ids"] = torch.zeros(B, dtype=torch.long)
    
    if include_reconstruction:
        kwargs["original_values"] = torch.randn(B, max_rows, max_cols)
        kwargs["reconstruction_mask"] = torch.rand(B, max_rows, max_cols) > 0.7
    
    return TokenBatch(**kwargs)


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
    """Create sample TokenBatch for testing with default config (max_cols=16)."""
    B, max_rows, max_cols = 4, 32, 16
    
    # Create properly structured tokens
    tokens = create_valid_tokens(B, max_rows, max_cols)
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
    """Create smaller sample TokenBatch for mini model tests (max_cols=8)."""
    B, max_rows, max_cols = 2, 16, 8
    
    # Create properly structured tokens - this was the bug!
    # Previously used torch.randn() which gave invalid feature_id values
    tokens = create_valid_tokens(B, max_rows, max_cols)
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
        assert default_config.max_cols == 16
    
    def test_mnar_variants(self, default_config):
        """Test MNAR variants configuration."""
        assert default_config.mnar_variants == ["self_censoring", "threshold"]
        assert default_config.n_experts == 4  # MCAR + MAR + 2 MNAR
    
    def test_n_experts_property(self):
        """Test n_experts is computed correctly."""
        config = LacunaModelConfig(mnar_variants=["sc", "th", "lat"])
        assert config.n_experts == 5  # MCAR + MAR + 3 MNAR
    
    def test_n_reconstruction_heads_property(self, default_config):
        """Test n_reconstruction_heads equals n_experts."""
        assert default_config.n_reconstruction_heads == default_config.n_experts
    
    def test_get_encoder_config(self, default_config):
        """Test encoder config generation."""
        encoder_config = default_config.get_encoder_config()
        
        assert isinstance(encoder_config, EncoderConfig)
        assert encoder_config.hidden_dim == default_config.hidden_dim
        assert encoder_config.evidence_dim == default_config.evidence_dim
        assert encoder_config.n_layers == default_config.n_layers
        assert encoder_config.n_heads == default_config.n_heads
        assert encoder_config.max_cols == default_config.max_cols
    
    def test_get_reconstruction_config(self, default_config):
        """Test reconstruction config generation."""
        recon_config = default_config.get_reconstruction_config()
        
        assert isinstance(recon_config, ReconstructionConfig)
        assert recon_config.hidden_dim == default_config.hidden_dim
        # ReconstructionConfig doesn't have n_experts directly
        # It uses mnar_variants to determine number of heads
        assert recon_config.mnar_variants == default_config.mnar_variants
    
    def test_get_moe_config(self, default_config):
        """Test MoE config generation."""
        moe_config = default_config.get_moe_config()
        
        assert isinstance(moe_config, MoEConfig)
        assert moe_config.evidence_dim == default_config.evidence_dim
        assert moe_config.n_experts == default_config.n_experts
    
    def test_loss_matrix_default(self):
        """Test default loss matrix structure."""
        config = LacunaModelConfig()
        
        # Should be a flat list of 9 values (3x3 matrix)
        assert len(config.loss_matrix) == 9
        
        # Reshape to verify structure
        matrix = torch.tensor(config.loss_matrix).reshape(3, 3)
        
        # Default loss matrix from assembly.py:
        # [0.0, 0.0, 10.0,  # Green
        #  1.0, 1.0,  2.0,  # Yellow
        #  3.0, 2.0,  0.0]  # Red
        # Note: diagonal is NOT all zeros in the default - only Green/MCAR and Red/MNAR are 0
        assert matrix[0, 0] == 0.0  # Green action, MCAR state
        assert matrix[2, 2] == 0.0  # Red action, MNAR state
    
    def test_loss_matrix_custom(self):
        """Test custom loss matrix."""
        custom_matrix = [
            1.0, 1.0, 5.0,   # Green
            0.5, 0.5, 1.0,   # Yellow
            2.0, 1.5, 0.0,   # Red
        ]
        config = LacunaModelConfig(loss_matrix=custom_matrix)
        
        assert config.loss_matrix == custom_matrix


# =============================================================================
# Test BayesOptimalDecision
# =============================================================================

class TestBayesOptimalDecision:
    """Tests for BayesOptimalDecision."""
    
    @pytest.fixture
    def loss_matrix(self):
        """Default loss matrix for testing."""
        return torch.tensor([
            [0.0, 1.0, 3.0],   # Action 0 (Green): Cost for MCAR, MAR, MNAR
            [0.5, 0.0, 1.0],   # Action 1 (Yellow): Cost for MCAR, MAR, MNAR
            [1.0, 0.5, 0.0],   # Action 2 (Red): Cost for MCAR, MAR, MNAR
        ])
    
    @pytest.fixture
    def decision_rule(self, loss_matrix):
        """Create decision rule with default loss matrix."""
        return BayesOptimalDecision(loss_matrix)
    
    def test_construction(self, decision_rule, loss_matrix):
        """Test construction."""
        assert torch.allclose(decision_rule.loss_matrix, loss_matrix)
    
    def test_action_names(self, decision_rule):
        """Test action names."""
        # action_names is a tuple, not a list
        assert decision_rule.action_names == ("Green", "Yellow", "Red")
    
    def test_forward_certain_mcar(self, decision_rule):
        """Test decision with certain MCAR posterior."""
        B = 4
        p_class = torch.zeros(B, 3)
        p_class[:, MCAR] = 1.0  # 100% MCAR
        
        decision = decision_rule(p_class)
        
        # Should choose Green (action 0) for MCAR
        assert (decision.action_ids == 0).all()
    
    def test_forward_certain_mar(self, decision_rule):
        """Test decision with certain MAR posterior."""
        B = 4
        p_class = torch.zeros(B, 3)
        p_class[:, MAR] = 1.0  # 100% MAR
        
        decision = decision_rule(p_class)
        
        # Should choose Yellow (action 1) for MAR
        assert (decision.action_ids == 1).all()
    
    def test_forward_certain_mnar(self, decision_rule):
        """Test decision with certain MNAR posterior."""
        B = 4
        p_class = torch.zeros(B, 3)
        p_class[:, MNAR] = 1.0  # 100% MNAR
        
        decision = decision_rule(p_class)
        
        # Should choose Red (action 2) for MNAR
        assert (decision.action_ids == 2).all()
    
    def test_forward_uncertain(self, decision_rule):
        """Test decision with uncertain posterior."""
        B = 2
        p_class = torch.tensor([
            [0.4, 0.4, 0.2],  # Leaning MCAR/MAR
            [0.1, 0.1, 0.8],  # Leaning MNAR
        ])
        
        decision = decision_rule(p_class)
        
        # Each should minimize expected loss
        assert decision.action_ids.shape == (B,)
        # Second sample should choose Red (MNAR heavy)
        assert decision.action_ids[1] == 2
    
    def test_expected_risks_computed(self, decision_rule):
        """Test that expected risks are computed."""
        p_class = torch.rand(4, 3)
        p_class = p_class / p_class.sum(dim=-1, keepdim=True)  # Normalize
        
        decision = decision_rule(p_class)
        
        # Decision uses expected_risks, not expected_loss
        assert decision.expected_risks.shape == (4,)
        assert (decision.expected_risks >= 0).all()
    
    def test_get_actions_helper(self, decision_rule):
        """Test get_actions helper method."""
        p_class = torch.zeros(3, 3)
        p_class[0, MCAR] = 1.0
        p_class[1, MAR] = 1.0
        p_class[2, MNAR] = 1.0
        
        decision = decision_rule(p_class)
        actions = decision.get_actions()
        
        assert actions == ["Green", "Yellow", "Red"]
    
    def test_decision_output_type(self, decision_rule):
        """Test that output is Decision type."""
        p_class = torch.rand(4, 3)
        p_class = p_class / p_class.sum(dim=-1, keepdim=True)
        
        decision = decision_rule(p_class)
        
        assert isinstance(decision, Decision)


# =============================================================================
# Test compute_entropy
# =============================================================================

class TestComputeEntropy:
    """Tests for compute_entropy utility function."""
    
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
        
        # LacunaOutput uses 'moe' not 'moe_output' as field name
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
        
        assert output.reconstruction is None
    
    def test_skip_decision(self, model_mini, sample_batch_mini):
        """Test that decision can be skipped."""
        output = model_mini(sample_batch_mini, compute_decision=False)
        
        assert output.decision is None
    
    def test_posterior_p_class_sums_to_one(self, model_mini, sample_batch_mini):
        """Test that posterior probabilities sum to 1."""
        output = model_mini(sample_batch_mini)
        
        sums = output.posterior.p_class.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)
    
    def test_posterior_p_class_non_negative(self, model_mini, sample_batch_mini):
        """Test that posterior probabilities are non-negative."""
        output = model_mini(sample_batch_mini)
        
        assert (output.posterior.p_class >= 0).all()
    
    def test_no_nan_or_inf(self, model_mini, sample_batch_mini):
        """Test no NaN or Inf in outputs."""
        output = model_mini(sample_batch_mini)
        
        assert not torch.isnan(output.posterior.p_class).any()
        assert not torch.isinf(output.posterior.p_class).any()
        assert not torch.isnan(output.evidence).any()
        assert not torch.isinf(output.evidence).any()
    
    def test_gradients_flow(self, model_mini, sample_batch_mini):
        """Test that gradients flow through the model."""
        model_mini.train()
        model_mini.zero_grad()
        
        output = model_mini(sample_batch_mini)
        
        # Use cross-entropy loss with class_ids to ensure proper gradient flow
        # Simple .sum() on softmax output doesn't propagate gradients well
        targets = sample_batch_mini.class_ids
        log_probs = torch.log(output.posterior.p_class.clamp(min=1e-8))
        loss = nn.functional.nll_loss(log_probs, targets)
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = False
        for param in model_mini.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients flowed through model"
    
    def test_get_token_representations(self, model_mini, sample_batch_mini):
        """Test get_token_representations helper method."""
        # Check if model has get_token_representations method
        if not hasattr(model_mini, 'get_token_representations'):
            pytest.skip("LacunaModel does not have get_token_representations method")
        
        token_repr = model_mini.get_token_representations(sample_batch_mini)
        
        # [B, max_rows, max_cols, hidden_dim]
        assert token_repr.shape == (2, 16, 8, 32)
    
    def test_get_row_representations(self, model_mini, sample_batch_mini):
        """Test get_row_representations helper method."""
        # Check if model has get_row_representations method
        if not hasattr(model_mini, 'get_row_representations'):
            pytest.skip("LacunaModel does not have get_row_representations method")
        
        row_repr = model_mini.get_row_representations(sample_batch_mini)
        
        # [B, max_rows, hidden_dim]
        assert row_repr.shape == (2, 16, 32)
    
    def test_encode_method(self, model_mini, sample_batch_mini):
        """Test encode helper method if it exists."""
        if not hasattr(model_mini, 'encode'):
            pytest.skip("LacunaModel does not have encode method")
        
        evidence = model_mini.encode(sample_batch_mini)
        
        assert evidence.shape == (2, 16)  # [B, evidence_dim]
        assert not torch.isnan(evidence).any()
    
    def test_expert_to_class_buffer(self, model):
        """Test expert_to_class mapping buffer."""
        # With 2 MNAR variants: experts are MCAR, MAR, SC, Thresh
        expected = torch.tensor([MCAR, MAR, MNAR, MNAR])
        
        assert torch.equal(model.expert_to_class, expected)
    
    def test_handles_variable_batch_sizes(self, model_mini):
        """Test model handles different batch sizes."""
        for B in [1, 2, 4, 8]:
            batch = create_sample_batch(
                B=B,
                max_rows=16,
                max_cols=8,  # Must match model's max_cols
                include_reconstruction=False,
                include_labels=False,
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
        
        # Use properly structured tokens
        batch = create_sample_batch(
            B=2,
            max_rows=16,
            max_cols=8,
            include_reconstruction=False,
            include_labels=False,
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
        assert model.config.n_experts == 3  # MCAR + MAR + 1 MNAR


class TestCreateLacunaBase:
    """Tests for create_lacuna_base factory."""
    
    def test_creates_standard_model(self):
        """Test factory creates a standard model."""
        model = create_lacuna_base()
        
        assert isinstance(model, LacunaModel)
        assert model.config.hidden_dim == 128
        assert model.config.evidence_dim == 64
        assert model.config.n_layers == 4
    
    def test_uses_attention_pooling(self):
        """Test base model uses attention pooling."""
        model = create_lacuna_base()
        
        assert model.config.row_pooling == "attention"
        assert model.config.dataset_pooling == "attention"


class TestCreateLacunaLarge:
    """Tests for create_lacuna_large factory."""
    
    def test_creates_large_model(self):
        """Test factory creates a large model."""
        model = create_lacuna_large()
        
        assert isinstance(model, LacunaModel)
        assert model.config.hidden_dim == 256
        assert model.config.evidence_dim == 128
        assert model.config.n_layers == 6
    
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
        
        # Create batch with labels using properly structured tokens
        batch = create_sample_batch(
            B=4,
            max_rows=16,
            max_cols=8,
            include_reconstruction=True,
            include_labels=True,
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
        
        batch = create_sample_batch(
            B=4,
            max_rows=16,
            max_cols=8,
            include_reconstruction=False,
            include_labels=False,
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
        
        batch = create_sample_batch(
            B=2,
            max_rows=16,
            max_cols=8,
            include_reconstruction=False,
            include_labels=False,
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
        
        # Create two separate samples with properly structured tokens
        sample1 = create_valid_tokens(1, 16, 8)
        sample2 = create_valid_tokens(1, 16, 8)
        
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