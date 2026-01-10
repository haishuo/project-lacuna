"""
Tests for lacuna.training.loss

Tests the multi-task loss functions:
    - LossConfig: Configuration dataclass
    - mechanism_cross_entropy, mechanism_cross_entropy_from_probs: Classification losses
    - brier_score: Proper scoring rule
    - class_cross_entropy, mechanism_full_cross_entropy: Class/mechanism losses
    - reconstruction_mse, reconstruction_huber: Reconstruction losses
    - multi_head_reconstruction_loss: Per-head reconstruction
    - kl_divergence_loss, entropy_loss, load_balance_loss: Auxiliary losses
    - LacunaLoss: Combined multi-task loss
    - Factory functions: create_loss_function, create_pretraining_loss, etc.
    - Accuracy metrics: compute_class_accuracy, compute_mechanism_accuracy
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lacuna.training.loss import (
    # Config
    LossConfig,
    # Mechanism losses
    mechanism_cross_entropy,
    mechanism_cross_entropy_from_probs,
    brier_score,
    class_cross_entropy,
    mechanism_full_cross_entropy,
    # Reconstruction losses
    reconstruction_mse,
    reconstruction_huber,
    multi_head_reconstruction_loss,
    # Auxiliary losses
    kl_divergence_loss,
    entropy_loss,
    load_balance_loss,
    # Combined loss
    LacunaLoss,
    # Factory functions
    create_loss_function,
    create_pretraining_loss,
    create_classification_loss,
    create_joint_loss,
    # Metrics
    compute_class_accuracy,
    compute_mechanism_accuracy,
    compute_per_class_accuracy,
)
from lacuna.core.types import (
    TokenBatch,
    PosteriorResult,
    LacunaOutput,
    ReconstructionResult,
    MoEOutput,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default LossConfig for testing."""
    return LossConfig()


@pytest.fixture
def sample_logits():
    """Sample logits for classification tests."""
    B, n_classes = 8, 3
    return torch.randn(B, n_classes)


@pytest.fixture
def sample_probs():
    """Sample probabilities (from softmax)."""
    B, n_classes = 8, 3
    return torch.softmax(torch.randn(B, n_classes), dim=-1)


@pytest.fixture
def sample_targets():
    """Sample integer targets for classification."""
    B = 8
    return torch.randint(0, 3, (B,))


@pytest.fixture
def sample_reconstruction_tensors():
    """Sample tensors for reconstruction loss tests."""
    B, max_rows, max_cols = 4, 32, 8
    
    return {
        "predictions": torch.randn(B, max_rows, max_cols),
        "targets": torch.randn(B, max_rows, max_cols),
        "reconstruction_mask": torch.rand(B, max_rows, max_cols) > 0.5,
        "row_mask": torch.ones(B, max_rows, dtype=torch.bool),
        "col_mask": torch.ones(B, max_cols, dtype=torch.bool),
    }


@pytest.fixture
def sample_reconstruction_result():
    """Sample ReconstructionResult for multi-head tests."""
    B, max_rows, max_cols = 4, 32, 8
    
    return ReconstructionResult(
        predictions=torch.randn(B, max_rows, max_cols),
        errors=torch.rand(B),
        per_cell_errors=torch.rand(B, max_rows, max_cols),
    )


@pytest.fixture
def sample_posterior_result():
    """Sample PosteriorResult for combined loss tests."""
    B = 4
    n_mechanisms = 5  # MCAR, MAR, 3 MNAR variants
    n_variants = 3
    
    p_mechanism = torch.softmax(torch.randn(B, n_mechanisms), dim=-1)
    # Aggregate to class: sum MNAR variants
    p_class = torch.zeros(B, 3)
    p_class[:, 0] = p_mechanism[:, 0]  # MCAR
    p_class[:, 1] = p_mechanism[:, 1]  # MAR
    p_class[:, 2] = p_mechanism[:, 2:].sum(dim=-1)  # MNAR
    
    # MNAR variant posterior (conditional on MNAR)
    p_mnar_variant = torch.softmax(torch.randn(B, n_variants), dim=-1)
    
    return PosteriorResult(
        p_class=p_class,
        p_mnar_variant=p_mnar_variant,
        p_mechanism=p_mechanism,
        entropy_class=torch.rand(B),
        entropy_mechanism=torch.rand(B),
    )


@pytest.fixture
def sample_moe_output():
    """Sample MoEOutput for auxiliary loss tests."""
    B = 4
    n_experts = 4
    
    gate_logits = torch.randn(B, n_experts)
    gate_probs = torch.softmax(gate_logits, dim=-1)
    
    return MoEOutput(
        gate_logits=gate_logits,
        gate_probs=gate_probs,
    )


@pytest.fixture
def sample_token_batch():
    """Sample TokenBatch for combined loss tests."""
    B, max_rows, max_cols = 4, 32, 8
    d_token = 4  # TOKEN_DIM from tokenization
    
    return TokenBatch(
        tokens=torch.randn(B, max_rows, max_cols, d_token),
        col_mask=torch.ones(B, max_cols, dtype=torch.bool),
        row_mask=torch.ones(B, max_rows, dtype=torch.bool),
        class_ids=torch.randint(0, 3, (B,)),
        generator_ids=torch.randint(0, 5, (B,)),
        variant_ids=torch.where(
            torch.randint(0, 3, (B,)) == 2,
            torch.randint(0, 3, (B,)),
            torch.tensor(-1),
        ),
        reconstruction_mask=torch.rand(B, max_rows, max_cols) > 0.9,
        original_values=torch.randn(B, max_rows, max_cols),
    )


@pytest.fixture
def sample_lacuna_output(sample_posterior_result, sample_moe_output, sample_reconstruction_result):
    """Sample LacunaOutput for combined loss tests."""
    B = 4
    
    return LacunaOutput(
        posterior=sample_posterior_result,
        moe=sample_moe_output,
        reconstruction={"mean": sample_reconstruction_result},
        evidence=torch.randn(B, 64),
    )


# =============================================================================
# Test LossConfig
# =============================================================================

class TestLossConfig:
    """Tests for LossConfig dataclass."""
    
    def test_default_values(self, default_config):
        """Test default configuration values."""
        assert default_config.mechanism_weight == 1.0
        assert default_config.reconstruction_weight == 0.5
        assert default_config.class_weight == 0.5
        assert default_config.mechanism_loss_type == "cross_entropy"
        assert default_config.reconstruction_loss_type == "mse"
        assert default_config.label_smoothing == 0.0
        assert default_config.load_balance_weight == 0.01
        assert default_config.entropy_weight == 0.0
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = LossConfig(
            mechanism_weight=2.0,
            reconstruction_weight=0.8,
            mechanism_loss_type="brier",
            label_smoothing=0.1,
        )
        
        assert config.mechanism_weight == 2.0
        assert config.reconstruction_weight == 0.8
        assert config.mechanism_loss_type == "brier"
        assert config.label_smoothing == 0.1
    
    def test_invalid_mechanism_loss_type(self):
        """Test that invalid mechanism_loss_type raises error."""
        with pytest.raises(ValueError, match="Unknown mechanism_loss_type"):
            LossConfig(mechanism_loss_type="invalid")
    
    def test_invalid_reconstruction_loss_type(self):
        """Test that invalid reconstruction_loss_type raises error."""
        with pytest.raises(ValueError, match="Unknown reconstruction_loss_type"):
            LossConfig(reconstruction_loss_type="invalid")


# =============================================================================
# Test Mechanism Classification Losses
# =============================================================================

class TestMechanismCrossEntropy:
    """Tests for mechanism_cross_entropy."""
    
    def test_output_shape(self, sample_logits, sample_targets):
        """Test that output is a scalar."""
        loss = mechanism_cross_entropy(sample_logits, sample_targets)
        
        assert loss.shape == ()
    
    def test_non_negative(self, sample_logits, sample_targets):
        """Test that cross-entropy is non-negative."""
        loss = mechanism_cross_entropy(sample_logits, sample_targets)
        
        assert loss >= 0
    
    def test_perfect_prediction_low_loss(self):
        """Test that perfect predictions give low loss."""
        B = 8
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        
        # Logits strongly favoring correct class
        logits = torch.zeros(B, 3)
        logits[torch.arange(B), targets] = 10.0
        
        loss = mechanism_cross_entropy(logits, targets)
        
        assert loss < 0.01
    
    def test_label_smoothing(self, sample_logits, sample_targets):
        """Test that label smoothing changes the loss."""
        loss_no_smooth = mechanism_cross_entropy(
            sample_logits, sample_targets, label_smoothing=0.0
        )
        loss_smooth = mechanism_cross_entropy(
            sample_logits, sample_targets, label_smoothing=0.1
        )
        
        # Label smoothing typically increases loss for confident predictions
        assert loss_no_smooth != loss_smooth
    
    def test_reduction_none(self, sample_logits, sample_targets):
        """Test reduction='none' returns per-sample losses."""
        loss = mechanism_cross_entropy(
            sample_logits, sample_targets, reduction="none"
        )
        
        assert loss.shape == (8,)
    
    def test_reduction_sum(self, sample_logits, sample_targets):
        """Test reduction='sum' returns summed loss."""
        loss_mean = mechanism_cross_entropy(
            sample_logits, sample_targets, reduction="mean"
        )
        loss_sum = mechanism_cross_entropy(
            sample_logits, sample_targets, reduction="sum"
        )
        
        assert torch.allclose(loss_sum, loss_mean * 8)
    
    def test_matches_pytorch(self, sample_logits, sample_targets):
        """Test that it matches PyTorch's cross_entropy."""
        loss = mechanism_cross_entropy(sample_logits, sample_targets)
        expected = F.cross_entropy(sample_logits, sample_targets)
        
        assert torch.allclose(loss, expected)


class TestMechanismCrossEntropyFromProbs:
    """Tests for mechanism_cross_entropy_from_probs."""
    
    def test_output_shape(self, sample_probs, sample_targets):
        """Test that output is a scalar."""
        loss = mechanism_cross_entropy_from_probs(sample_probs, sample_targets)
        
        assert loss.shape == ()
    
    def test_non_negative(self, sample_probs, sample_targets):
        """Test that cross-entropy is non-negative."""
        loss = mechanism_cross_entropy_from_probs(sample_probs, sample_targets)
        
        assert loss >= 0
    
    def test_perfect_prediction_low_loss(self):
        """Test that perfect predictions give near-zero loss."""
        B = 8
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        
        # One-hot probabilities
        probs = torch.zeros(B, 3)
        probs[torch.arange(B), targets] = 1.0
        
        loss = mechanism_cross_entropy_from_probs(probs, targets)
        
        # Should be very close to 0 (clamping prevents exact 0)
        assert loss < 1e-6
    
    def test_uniform_prediction_known_loss(self):
        """Test that uniform predictions give known loss."""
        B = 8
        targets = torch.zeros(B, dtype=torch.long)
        
        # Uniform distribution
        probs = torch.ones(B, 3) / 3
        
        loss = mechanism_cross_entropy_from_probs(probs, targets)
        
        # -log(1/3) ≈ 1.0986
        expected = torch.tensor(-torch.log(torch.tensor(1/3)))
        assert torch.allclose(loss, expected, atol=1e-4)
    
    def test_label_smoothing(self, sample_probs, sample_targets):
        """Test that label smoothing changes the loss."""
        loss_no_smooth = mechanism_cross_entropy_from_probs(
            sample_probs, sample_targets, label_smoothing=0.0
        )
        loss_smooth = mechanism_cross_entropy_from_probs(
            sample_probs, sample_targets, label_smoothing=0.1
        )
        
        assert loss_no_smooth != loss_smooth


class TestBrierScore:
    """Tests for brier_score."""
    
    def test_output_shape(self, sample_probs, sample_targets):
        """Test that output is a scalar."""
        loss = brier_score(sample_probs, sample_targets)
        
        assert loss.shape == ()
    
    def test_non_negative(self, sample_probs, sample_targets):
        """Test that Brier score is non-negative."""
        loss = brier_score(sample_probs, sample_targets)
        
        assert loss >= 0
    
    def test_perfect_prediction_zero_loss(self):
        """Test that perfect predictions give zero loss."""
        B = 4
        targets = torch.tensor([0, 1, 2, 0])
        
        # One-hot probabilities
        probs = torch.zeros(B, 3)
        probs[torch.arange(B), targets] = 1.0
        
        loss = brier_score(probs, targets)
        
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_worst_prediction_known_loss(self):
        """Test worst predictions give known loss."""
        B = 4
        targets = torch.tensor([0, 0, 0, 0])
        
        # All probability on wrong class
        probs = torch.zeros(B, 3)
        probs[:, 1] = 1.0  # Wrong class
        
        loss = brier_score(probs, targets)
        
        # BS = (1/3) * (1^2 + 1^2 + 0^2) = 2/3 ≈ 0.667
        expected = torch.tensor(2/3)
        assert torch.allclose(loss, expected, atol=1e-5)
    
    def test_uniform_prediction_known_loss(self):
        """Test uniform predictions give known loss."""
        B = 4
        targets = torch.zeros(B, dtype=torch.long)
        
        # Uniform distribution
        probs = torch.ones(B, 3) / 3
        
        loss = brier_score(probs, targets)
        
        # BS = (1/3) * ((1/3-1)^2 + (1/3-0)^2 + (1/3-0)^2)
        # = (1/3) * (4/9 + 1/9 + 1/9) = (1/3) * (6/9) = 2/9 ≈ 0.222
        expected = torch.tensor(2/9)
        assert torch.allclose(loss, expected, atol=1e-5)
    
    def test_in_valid_range(self):
        """Test that Brier score is in [0, 2]."""
        for _ in range(10):
            B = 8
            probs = torch.softmax(torch.randn(B, 3), dim=-1)
            targets = torch.randint(0, 3, (B,))
            
            loss = brier_score(probs, targets)
            
            assert loss >= 0
            assert loss <= 2  # Maximum is 2 for 3-class


class TestClassCrossEntropy:
    """Tests for class_cross_entropy."""
    
    def test_uses_probs_not_logits(self):
        """Test that it uses probabilities (not logits)."""
        B = 4
        probs = torch.softmax(torch.randn(B, 3), dim=-1)
        targets = torch.randint(0, 3, (B,))
        
        loss = class_cross_entropy(probs, targets)
        
        # Should equal mechanism_cross_entropy_from_probs
        expected = mechanism_cross_entropy_from_probs(probs, targets)
        assert torch.allclose(loss, expected)


class TestMechanismFullCrossEntropy:
    """Tests for mechanism_full_cross_entropy."""
    
    def test_handles_more_classes(self):
        """Test with more mechanism classes (MNAR variants)."""
        B = 4
        n_mechanisms = 5  # MCAR, MAR, 3 MNAR variants
        
        probs = torch.softmax(torch.randn(B, n_mechanisms), dim=-1)
        targets = torch.randint(0, n_mechanisms, (B,))
        
        loss = mechanism_full_cross_entropy(probs, targets)
        
        assert loss.shape == ()
        assert loss >= 0


# =============================================================================
# Test Reconstruction Losses
# =============================================================================

class TestReconstructionMSE:
    """Tests for reconstruction_mse."""
    
    def test_output_shape(self, sample_reconstruction_tensors):
        """Test scalar output."""
        t = sample_reconstruction_tensors
        
        loss = reconstruction_mse(
            t["predictions"],
            t["targets"],
            t["reconstruction_mask"],
            t["row_mask"],
            t["col_mask"],
        )
        
        assert loss.shape == ()
    
    def test_non_negative(self, sample_reconstruction_tensors):
        """Test that MSE is non-negative."""
        t = sample_reconstruction_tensors
        
        loss = reconstruction_mse(
            t["predictions"],
            t["targets"],
            t["reconstruction_mask"],
            t["row_mask"],
            t["col_mask"],
        )
        
        assert loss >= 0
    
    def test_perfect_prediction_zero_loss(self, sample_reconstruction_tensors):
        """Test zero loss for perfect predictions."""
        t = sample_reconstruction_tensors
        
        # Use targets as predictions
        loss = reconstruction_mse(
            t["targets"],  # Perfect prediction
            t["targets"],
            t["reconstruction_mask"],
            t["row_mask"],
            t["col_mask"],
        )
        
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_respects_masks(self, sample_reconstruction_tensors):
        """Test that masks are respected."""
        t = sample_reconstruction_tensors
        
        # Create mask that excludes all cells
        empty_mask = torch.zeros_like(t["reconstruction_mask"])
        
        # With empty mask, loss should be 0 (no cells to evaluate)
        loss = reconstruction_mse(
            t["predictions"],
            t["targets"],
            empty_mask,
            t["row_mask"],
            t["col_mask"],
        )
        
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_reduction_none(self, sample_reconstruction_tensors):
        """Test reduction='none' returns per-cell losses."""
        t = sample_reconstruction_tensors
        
        loss = reconstruction_mse(
            t["predictions"],
            t["targets"],
            t["reconstruction_mask"],
            t["row_mask"],
            t["col_mask"],
            reduction="none",
        )
        
        assert loss.shape == t["predictions"].shape


class TestReconstructionHuber:
    """Tests for reconstruction_huber."""
    
    def test_output_shape(self, sample_reconstruction_tensors):
        """Test scalar output."""
        t = sample_reconstruction_tensors
        
        loss = reconstruction_huber(
            t["predictions"],
            t["targets"],
            t["reconstruction_mask"],
            t["row_mask"],
            t["col_mask"],
        )
        
        assert loss.shape == ()
    
    def test_non_negative(self, sample_reconstruction_tensors):
        """Test that Huber loss is non-negative."""
        t = sample_reconstruction_tensors
        
        loss = reconstruction_huber(
            t["predictions"],
            t["targets"],
            t["reconstruction_mask"],
            t["row_mask"],
            t["col_mask"],
        )
        
        assert loss >= 0
    
    def test_perfect_prediction_zero_loss(self, sample_reconstruction_tensors):
        """Test zero loss for perfect predictions."""
        t = sample_reconstruction_tensors
        
        loss = reconstruction_huber(
            t["targets"],
            t["targets"],
            t["reconstruction_mask"],
            t["row_mask"],
            t["col_mask"],
        )
        
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_equals_mse_for_small_errors(self):
        """Test that Huber equals MSE/2 for small errors."""
        B, max_rows, max_cols = 4, 8, 4
        
        targets = torch.zeros(B, max_rows, max_cols)
        predictions = torch.full((B, max_rows, max_cols), 0.1)  # Small error
        mask = torch.ones(B, max_rows, max_cols, dtype=torch.bool)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        mse_loss = reconstruction_mse(
            predictions, targets, mask, row_mask, col_mask
        )
        huber_loss = reconstruction_huber(
            predictions, targets, mask, row_mask, col_mask, delta=1.0
        )
        
        # For |x| <= delta, Huber = 0.5 * x^2
        assert torch.allclose(huber_loss, 0.5 * mse_loss, atol=1e-5)
    
    def test_linear_for_large_errors(self):
        """Test that Huber is linear for large errors."""
        B, max_rows, max_cols = 4, 8, 4
        delta = 1.0
        
        targets = torch.zeros(B, max_rows, max_cols)
        predictions = torch.full((B, max_rows, max_cols), 5.0)  # Large error
        mask = torch.ones(B, max_rows, max_cols, dtype=torch.bool)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        huber_loss = reconstruction_huber(
            predictions, targets, mask, row_mask, col_mask, delta=delta
        )
        
        # For |x| > delta, Huber = delta * (|x| - 0.5 * delta)
        expected = delta * (5.0 - 0.5 * delta)
        assert torch.allclose(huber_loss, torch.tensor(expected), atol=1e-5)


class TestMultiHeadReconstructionLoss:
    """Tests for multi_head_reconstruction_loss."""
    
    def test_single_head(self, sample_reconstruction_tensors, sample_reconstruction_result):
        """Test with single reconstruction head."""
        t = sample_reconstruction_tensors
        
        total_loss, per_head = multi_head_reconstruction_loss(
            reconstruction_results={"mean": sample_reconstruction_result},
            original_values=t["targets"],
            reconstruction_mask=t["reconstruction_mask"],
            row_mask=t["row_mask"],
            col_mask=t["col_mask"],
        )
        
        assert total_loss.shape == ()
        assert "mean" in per_head
        assert torch.allclose(total_loss, per_head["mean"])
    
    def test_multiple_heads(self, sample_reconstruction_tensors):
        """Test with multiple reconstruction heads."""
        t = sample_reconstruction_tensors
        B, max_rows, max_cols = t["predictions"].shape
        
        results = {
            "mean": ReconstructionResult(
                predictions=torch.randn(B, max_rows, max_cols),
                errors=torch.rand(B),
                per_cell_errors=torch.rand(B, max_rows, max_cols),
            ),
            "median": ReconstructionResult(
                predictions=torch.randn(B, max_rows, max_cols),
                errors=torch.rand(B),
                per_cell_errors=torch.rand(B, max_rows, max_cols),
            ),
        }
        
        total_loss, per_head = multi_head_reconstruction_loss(
            reconstruction_results=results,
            original_values=t["targets"],
            reconstruction_mask=t["reconstruction_mask"],
            row_mask=t["row_mask"],
            col_mask=t["col_mask"],
        )
        
        assert total_loss.shape == ()
        assert "mean" in per_head
        assert "median" in per_head
        
        # Default: uniform weights, so average
        expected = (per_head["mean"] + per_head["median"]) / 2
        assert torch.allclose(total_loss, expected)
    
    def test_custom_weights(self, sample_reconstruction_tensors):
        """Test custom head weights."""
        t = sample_reconstruction_tensors
        B, max_rows, max_cols = t["predictions"].shape
        
        results = {
            "mean": ReconstructionResult(
                predictions=torch.randn(B, max_rows, max_cols),
                errors=torch.rand(B),
                per_cell_errors=torch.rand(B, max_rows, max_cols),
            ),
            "median": ReconstructionResult(
                predictions=torch.randn(B, max_rows, max_cols),
                errors=torch.rand(B),
                per_cell_errors=torch.rand(B, max_rows, max_cols),
            ),
        }
        
        head_weights = {"mean": 2.0, "median": 1.0}
        
        total_loss, per_head = multi_head_reconstruction_loss(
            reconstruction_results=results,
            original_values=t["targets"],
            reconstruction_mask=t["reconstruction_mask"],
            row_mask=t["row_mask"],
            col_mask=t["col_mask"],
            head_weights=head_weights,
        )
        
        # Weighted average: (2*mean + 1*median) / 3
        expected = (2 * per_head["mean"] + 1 * per_head["median"]) / 3
        assert torch.allclose(total_loss, expected)
    
    def test_huber_loss_type(self, sample_reconstruction_tensors, sample_reconstruction_result):
        """Test Huber loss type."""
        t = sample_reconstruction_tensors
        
        total_mse, _ = multi_head_reconstruction_loss(
            reconstruction_results={"mean": sample_reconstruction_result},
            original_values=t["targets"],
            reconstruction_mask=t["reconstruction_mask"],
            row_mask=t["row_mask"],
            col_mask=t["col_mask"],
            loss_type="mse",
        )
        
        total_huber, _ = multi_head_reconstruction_loss(
            reconstruction_results={"mean": sample_reconstruction_result},
            original_values=t["targets"],
            reconstruction_mask=t["reconstruction_mask"],
            row_mask=t["row_mask"],
            col_mask=t["col_mask"],
            loss_type="huber",
        )
        
        # MSE and Huber should give different results for large errors
        # (but might be similar for small errors)
        assert total_mse.shape == total_huber.shape


# =============================================================================
# Test Auxiliary Losses
# =============================================================================

class TestKLDivergenceLoss:
    """Tests for kl_divergence_loss."""
    
    def test_output_shape(self):
        """Test scalar output."""
        B, d_latent = 8, 32
        mean = torch.randn(B, d_latent)
        logvar = torch.randn(B, d_latent)
        
        loss = kl_divergence_loss(mean, logvar)
        
        assert loss.shape == ()
    
    def test_non_negative(self):
        """Test that KL divergence is non-negative."""
        B, d_latent = 8, 32
        mean = torch.randn(B, d_latent)
        logvar = torch.randn(B, d_latent)
        
        loss = kl_divergence_loss(mean, logvar)
        
        assert loss >= 0
    
    def test_zero_for_standard_normal(self):
        """Test that KL is zero when q equals prior (N(0,1))."""
        B, d_latent = 8, 32
        mean = torch.zeros(B, d_latent)
        logvar = torch.zeros(B, d_latent)  # variance = 1
        
        loss = kl_divergence_loss(mean, logvar)
        
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_increases_with_mean_shift(self):
        """Test that KL increases as mean shifts from 0."""
        B, d_latent = 8, 32
        logvar = torch.zeros(B, d_latent)
        
        loss_zero = kl_divergence_loss(torch.zeros(B, d_latent), logvar)
        loss_shifted = kl_divergence_loss(torch.ones(B, d_latent), logvar)
        
        assert loss_shifted > loss_zero
    
    def test_increases_with_variance_change(self):
        """Test that KL increases as variance deviates from 1."""
        B, d_latent = 8, 32
        mean = torch.zeros(B, d_latent)
        
        loss_unit = kl_divergence_loss(mean, torch.zeros(B, d_latent))
        loss_high = kl_divergence_loss(mean, torch.ones(B, d_latent))  # var = e
        
        assert loss_high > loss_unit


class TestEntropyLoss:
    """Tests for entropy_loss."""
    
    def test_output_shape(self):
        """Test scalar output."""
        B, K = 8, 3
        probs = torch.softmax(torch.randn(B, K), dim=-1)
        
        loss = entropy_loss(probs)
        
        assert loss.shape == ()
    
    def test_non_negative(self):
        """Test that entropy is non-negative."""
        B, K = 8, 3
        probs = torch.softmax(torch.randn(B, K), dim=-1)
        
        loss = entropy_loss(probs)
        
        assert loss >= 0
    
    def test_zero_for_one_hot(self):
        """Test that entropy is zero for one-hot distributions."""
        B, K = 8, 3
        probs = torch.zeros(B, K)
        probs[:, 0] = 1.0
        
        loss = entropy_loss(probs)
        
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_max_for_uniform(self):
        """Test that entropy is maximized for uniform distribution."""
        B, K = 8, 3
        probs = torch.ones(B, K) / K
        
        loss = entropy_loss(probs)
        
        # Max entropy = log(K)
        expected = torch.tensor(K).float().log()
        assert torch.allclose(loss, expected, atol=1e-5)


class TestLoadBalanceLoss:
    """Tests for load_balance_loss."""
    
    def test_output_shape(self):
        """Test scalar output."""
        B, n_experts = 16, 4
        gate_probs = torch.softmax(torch.randn(B, n_experts), dim=-1)
        
        loss = load_balance_loss(gate_probs)
        
        assert loss.shape == ()
    
    def test_non_negative(self):
        """Test that load balance loss is non-negative."""
        B, n_experts = 16, 4
        gate_probs = torch.softmax(torch.randn(B, n_experts), dim=-1)
        
        loss = load_balance_loss(gate_probs)
        
        assert loss >= 0
    
    def test_low_for_uniform_usage(self):
        """Test that loss is low when experts are used uniformly."""
        B, n_experts = 16, 4
        
        # Create probs that route uniformly
        # Each expert gets exactly B/n_experts samples
        probs = torch.zeros(B, n_experts)
        for i in range(B):
            probs[i, i % n_experts] = 1.0
        
        loss = load_balance_loss(probs)
        
        # Should equal n_experts * (1/n_experts)^2 * n_experts = 1
        assert torch.allclose(loss, torch.tensor(1.0), atol=1e-4)


# =============================================================================
# Test LacunaLoss
# =============================================================================

class TestLacunaLoss:
    """Tests for LacunaLoss combined loss."""
    
    @pytest.fixture
    def loss_fn(self):
        """Default loss function."""
        return LacunaLoss(LossConfig())
    
    def test_output_types(
        self, loss_fn, sample_lacuna_output, sample_token_batch
    ):
        """Test that output is (tensor, dict)."""
        total_loss, loss_dict = loss_fn(sample_lacuna_output, sample_token_batch)
        
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(loss_dict, dict)
        assert total_loss.shape == ()
    
    def test_loss_dict_keys(
        self, loss_fn, sample_lacuna_output, sample_token_batch
    ):
        """Test that loss_dict contains expected keys."""
        _, loss_dict = loss_fn(sample_lacuna_output, sample_token_batch)
        
        assert "total_loss" in loss_dict
        # At least one of class_loss or mechanism_loss should be present
        assert "class_loss" in loss_dict or loss_dict.get("mechanism_loss") is not None
    
    def test_mechanism_weight_zero(
        self, sample_lacuna_output, sample_token_batch
    ):
        """Test that mechanism_weight=0 excludes classification loss."""
        loss_fn = LacunaLoss(LossConfig(mechanism_weight=0.0))
        
        total_loss, loss_dict = loss_fn(sample_lacuna_output, sample_token_batch)
        
        # Classification loss should not contribute
        assert "class_loss" not in loss_dict or loss_dict.get("class_loss", 0) == 0
    
    def test_reconstruction_weight_zero(
        self, sample_lacuna_output, sample_token_batch
    ):
        """Test that reconstruction_weight=0 excludes reconstruction loss."""
        loss_fn = LacunaLoss(LossConfig(reconstruction_weight=0.0))
        
        total_loss, loss_dict = loss_fn(sample_lacuna_output, sample_token_batch)
        
        assert "reconstruction_loss" not in loss_dict
    
    def test_brier_score_mode(
        self, sample_lacuna_output, sample_token_batch
    ):
        """Test Brier score mode."""
        loss_fn = LacunaLoss(LossConfig(mechanism_loss_type="brier"))
        
        total_loss, loss_dict = loss_fn(sample_lacuna_output, sample_token_batch)
        
        assert total_loss.shape == ()
        assert "class_loss" in loss_dict
    
    def test_gradient_flow(
        self, loss_fn, sample_moe_output, sample_reconstruction_result, sample_token_batch
    ):
        """Test that gradients flow through the loss."""
        B = 4
        n_mechanisms = 5
        n_variants = 3
        
        # Create p_class that requires grad (simulating model output)
        p_class_logits = torch.randn(B, 3, requires_grad=True)
        p_class = torch.softmax(p_class_logits, dim=-1)
        
        p_mechanism = torch.softmax(torch.randn(B, n_mechanisms), dim=-1)
        p_mnar_variant = torch.softmax(torch.randn(B, n_variants), dim=-1)
        
        posterior = PosteriorResult(
            p_class=p_class,
            p_mnar_variant=p_mnar_variant,
            p_mechanism=p_mechanism,
            entropy_class=torch.rand(B),
            entropy_mechanism=torch.rand(B),
        )
        
        output = LacunaOutput(
            posterior=posterior,
            moe=sample_moe_output,
            reconstruction={"mean": sample_reconstruction_result},
            evidence=torch.randn(B, 64),
        )
        
        total_loss, _ = loss_fn(output, sample_token_batch)
        total_loss.backward()
        
        # p_class_logits should have gradient since loss depends on p_class
        assert p_class_logits.grad is not None
        assert not torch.all(p_class_logits.grad == 0)
    
    def test_pretraining_loss_method(
        self, loss_fn, sample_lacuna_output, sample_token_batch
    ):
        """Test pretraining_loss helper method."""
        total_loss, loss_dict = loss_fn.pretraining_loss(
            sample_lacuna_output, sample_token_batch
        )
        
        # Should not include mechanism loss
        assert "class_loss" not in loss_dict or loss_dict.get("class_loss", 0) == 0
    
    def test_classification_loss_method(
        self, loss_fn, sample_lacuna_output, sample_token_batch
    ):
        """Test classification_loss helper method."""
        total_loss, loss_dict = loss_fn.classification_loss(
            sample_lacuna_output, sample_token_batch
        )
        
        # Should not include reconstruction loss
        assert "reconstruction_loss" not in loss_dict


# =============================================================================
# Test Factory Functions
# =============================================================================

class TestCreateLossFunction:
    """Tests for create_loss_function factory."""
    
    def test_creates_loss(self):
        """Test factory creates LacunaLoss."""
        loss_fn = create_loss_function()
        
        assert isinstance(loss_fn, LacunaLoss)
    
    def test_respects_parameters(self):
        """Test factory respects all parameters."""
        loss_fn = create_loss_function(
            mechanism_weight=2.0,
            reconstruction_weight=0.5,
            mechanism_loss_type="brier",
            label_smoothing=0.1,
        )
        
        assert loss_fn.config.mechanism_weight == 2.0
        assert loss_fn.config.reconstruction_weight == 0.5
        assert loss_fn.config.mechanism_loss_type == "brier"
        assert loss_fn.config.label_smoothing == 0.1


class TestCreatePretrainingLoss:
    """Tests for create_pretraining_loss factory."""
    
    def test_no_mechanism_weight(self):
        """Test pretraining loss has no mechanism weight."""
        loss_fn = create_pretraining_loss()
        
        assert loss_fn.config.mechanism_weight == 0.0
    
    def test_has_reconstruction_weight(self):
        """Test pretraining loss has reconstruction weight."""
        loss_fn = create_pretraining_loss()
        
        assert loss_fn.config.reconstruction_weight > 0


class TestCreateClassificationLoss:
    """Tests for create_classification_loss factory."""
    
    def test_has_mechanism_weight(self):
        """Test classification loss has mechanism weight."""
        loss_fn = create_classification_loss()
        
        assert loss_fn.config.mechanism_weight > 0
    
    def test_no_reconstruction_weight(self):
        """Test classification loss has no reconstruction weight."""
        loss_fn = create_classification_loss()
        
        assert loss_fn.config.reconstruction_weight == 0.0


class TestCreateJointLoss:
    """Tests for create_joint_loss factory."""
    
    def test_has_both_weights(self):
        """Test joint loss has both weights."""
        loss_fn = create_joint_loss()
        
        assert loss_fn.config.mechanism_weight > 0
        assert loss_fn.config.reconstruction_weight > 0
    
    def test_respects_parameters(self):
        """Test factory respects parameters."""
        loss_fn = create_joint_loss(
            mechanism_weight=2.0,
            reconstruction_weight=0.3,
        )
        
        assert loss_fn.config.mechanism_weight == 2.0
        assert loss_fn.config.reconstruction_weight == 0.3


# =============================================================================
# Test Accuracy Metrics
# =============================================================================

class TestComputeClassAccuracy:
    """Tests for compute_class_accuracy."""
    
    def test_perfect_accuracy(self):
        """Test 100% accuracy when predictions match targets."""
        B = 8
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        
        # Perfect predictions
        p_class = torch.zeros(B, 3)
        p_class[torch.arange(B), targets] = 1.0
        
        acc = compute_class_accuracy(p_class, targets)
        
        assert acc == 1.0
    
    def test_zero_accuracy(self):
        """Test 0% accuracy when all predictions wrong."""
        B = 8
        targets = torch.zeros(B, dtype=torch.long)
        
        # All predict class 1 (wrong)
        p_class = torch.zeros(B, 3)
        p_class[:, 1] = 1.0
        
        acc = compute_class_accuracy(p_class, targets)
        
        assert acc == 0.0
    
    def test_partial_accuracy(self):
        """Test partial accuracy."""
        B = 4
        targets = torch.tensor([0, 0, 0, 0])
        
        # 2 correct, 2 wrong
        p_class = torch.zeros(B, 3)
        p_class[0, 0] = 1.0  # Correct
        p_class[1, 0] = 1.0  # Correct
        p_class[2, 1] = 1.0  # Wrong
        p_class[3, 2] = 1.0  # Wrong
        
        acc = compute_class_accuracy(p_class, targets)
        
        assert acc == 0.5


class TestComputeMechanismAccuracy:
    """Tests for compute_mechanism_accuracy."""
    
    def test_with_mnar_variants(self):
        """Test accuracy with MNAR variants."""
        B = 6
        n_mechanisms = 5  # MCAR, MAR, 3 MNAR variants
        
        targets = torch.tensor([0, 1, 2, 3, 4, 0])  # Various mechanisms
        
        # Perfect predictions
        p_mechanism = torch.zeros(B, n_mechanisms)
        p_mechanism[torch.arange(B), targets] = 1.0
        
        acc = compute_mechanism_accuracy(p_mechanism, targets)
        
        assert acc == 1.0


class TestComputePerClassAccuracy:
    """Tests for compute_per_class_accuracy."""
    
    def test_returns_all_classes(self):
        """Test that all three classes are returned."""
        B = 9
        targets = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        p_class = torch.zeros(B, 3)
        p_class[torch.arange(B), targets] = 1.0
        
        per_class = compute_per_class_accuracy(p_class, targets)
        
        assert "mcar_acc" in per_class
        assert "mar_acc" in per_class
        assert "mnar_acc" in per_class
    
    def test_perfect_per_class(self):
        """Test perfect per-class accuracy."""
        B = 9
        targets = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        p_class = torch.zeros(B, 3)
        p_class[torch.arange(B), targets] = 1.0
        
        per_class = compute_per_class_accuracy(p_class, targets)
        
        assert per_class["mcar_acc"] == 1.0
        assert per_class["mar_acc"] == 1.0
        assert per_class["mnar_acc"] == 1.0
    
    def test_handles_missing_class(self):
        """Test handling when a class has no samples."""
        B = 4
        targets = torch.tensor([0, 0, 1, 1])  # No class 2
        p_class = torch.zeros(B, 3)
        p_class[torch.arange(B), targets] = 1.0
        
        per_class = compute_per_class_accuracy(p_class, targets)
        
        assert per_class["mcar_acc"] == 1.0
        assert per_class["mar_acc"] == 1.0
        assert torch.isnan(per_class["mnar_acc"])  # No MNAR samples
    
    def test_varying_accuracy(self):
        """Test varying accuracy per class."""
        # MCAR: 2/2 correct, MAR: 1/2 correct, MNAR: 0/2 correct
        targets = torch.tensor([0, 0, 1, 1, 2, 2])
        p_class = torch.zeros(6, 3)
        p_class[0, 0] = 1.0  # MCAR correct
        p_class[1, 0] = 1.0  # MCAR correct
        p_class[2, 1] = 1.0  # MAR correct
        p_class[3, 0] = 1.0  # MAR wrong
        p_class[4, 0] = 1.0  # MNAR wrong
        p_class[5, 1] = 1.0  # MNAR wrong
        
        per_class = compute_per_class_accuracy(p_class, targets)
        
        assert per_class["mcar_acc"] == 1.0
        assert per_class["mar_acc"] == 0.5
        assert per_class["mnar_acc"] == 0.0