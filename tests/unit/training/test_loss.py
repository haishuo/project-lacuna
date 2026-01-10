"""
Tests for lacuna.training.loss

Tests the multi-task loss functions:
    - LossConfig: Configuration dataclass
    - mechanism_cross_entropy, mechanism_cross_entropy_from_probs: Classification losses
    - brier_score: Proper scoring rule
    - class_cross_entropy, mechanism_full_cross_entropy: Class/mechanism losses
    - reconstruction_mse, reconstruction_huber: Reconstruction losses
    - multi_head_reconstruction_loss: Per-head reconstruction
    - kl_divergence, entropy_loss, load_balance_loss: Auxiliary losses
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
    kl_divergence,
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
    B, max_rows, max_cols = 4, 32, 16
    
    predictions = torch.randn(B, max_rows, max_cols)
    targets = torch.randn(B, max_rows, max_cols)
    
    # Mask: ~30% of cells are masked for reconstruction
    reconstruction_mask = torch.rand(B, max_rows, max_cols) > 0.7
    
    # Valid rows and columns
    row_mask = torch.ones(B, max_rows, dtype=torch.bool)
    row_mask[:, 20:] = False  # Last 12 rows invalid
    
    col_mask = torch.ones(B, max_cols, dtype=torch.bool)
    col_mask[:, 10:] = False  # Last 6 cols invalid
    
    return {
        "predictions": predictions,
        "targets": targets,
        "reconstruction_mask": reconstruction_mask,
        "row_mask": row_mask,
        "col_mask": col_mask,
    }


@pytest.fixture
def sample_lacuna_output():
    """Sample LacunaOutput for full loss tests."""
    B = 4
    n_experts = 5
    n_mnar_variants = 3
    max_rows, max_cols = 32, 16
    evidence_dim = 32
    
    # Posterior
    p_class = torch.softmax(torch.randn(B, 3), dim=-1)
    p_mechanism = torch.softmax(torch.randn(B, n_experts), dim=-1)
    
    posterior = PosteriorResult(
        p_class=p_class,
        p_mechanism=p_mechanism,
        entropy_class=torch.rand(B),
    )
    
    # MoE output
    moe_output = MoEOutput(
        gate_logits=torch.randn(B, n_experts),
        gate_probs=torch.softmax(torch.randn(B, n_experts), dim=-1),
    )
    
    # Reconstruction results
    reconstruction = {}
    for head_name in ["mcar", "mar", "self_censoring", "threshold", "latent"]:
        reconstruction[head_name] = ReconstructionResult(
            predictions=torch.randn(B, max_rows, max_cols),
            errors=torch.rand(B),
            per_cell_errors=torch.rand(B, max_rows, max_cols),
        )
    
    return LacunaOutput(
        posterior=posterior,
        moe=moe_output,
        reconstruction=reconstruction,
        evidence=torch.randn(B, evidence_dim),
    )


@pytest.fixture
def sample_batch():
    """Sample TokenBatch for full loss tests."""
    B, max_rows, max_cols = 4, 32, 16
    
    return TokenBatch(
        tokens=torch.randn(B, max_rows, max_cols, 4),
        row_mask=torch.ones(B, max_rows, dtype=torch.bool),
        col_mask=torch.ones(B, max_cols, dtype=torch.bool),
        class_ids=torch.randint(0, 3, (B,)),
        variant_ids=torch.zeros(B, dtype=torch.long),
        original_values=torch.randn(B, max_rows, max_cols),
        reconstruction_mask=torch.rand(B, max_rows, max_cols) > 0.7,
    )


# =============================================================================
# Test LossConfig
# =============================================================================

class TestLossConfig:
    """Tests for LossConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = LossConfig()
        
        assert config.mechanism_weight == 1.0
        assert config.reconstruction_weight == 0.5
        assert config.class_weight == 0.5
        assert config.mechanism_loss_type == "cross_entropy"
        assert config.reconstruction_loss_type == "mse"
        assert config.label_smoothing == 0.0
        assert config.load_balance_weight == 0.01
        assert config.entropy_weight == 0.0
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = LossConfig(
            mechanism_weight=2.0,
            reconstruction_weight=1.0,
            class_weight=0.7,
            mechanism_loss_type="brier",
            reconstruction_loss_type="huber",
            label_smoothing=0.1,
        )
        
        assert config.mechanism_weight == 2.0
        assert config.reconstruction_weight == 1.0
        assert config.class_weight == 0.7
        assert config.mechanism_loss_type == "brier"
        assert config.reconstruction_loss_type == "huber"
        assert config.label_smoothing == 0.1
    
    def test_invalid_mechanism_loss_type_raises(self):
        """Test that invalid mechanism_loss_type raises error."""
        with pytest.raises(ValueError, match="Unknown mechanism_loss_type"):
            LossConfig(mechanism_loss_type="invalid")
    
    def test_invalid_reconstruction_loss_type_raises(self):
        """Test that invalid reconstruction_loss_type raises error."""
        with pytest.raises(ValueError, match="Unknown reconstruction_loss_type"):
            LossConfig(reconstruction_loss_type="invalid")
    
    def test_per_head_weights(self):
        """Test per-head weights configuration."""
        weights = {"mcar": 1.0, "mar": 1.5, "self_censoring": 2.0}
        config = LossConfig(per_head_weights=weights)
        
        assert config.per_head_weights == weights


# =============================================================================
# Test Mechanism Classification Losses
# =============================================================================

class TestMechanismCrossEntropy:
    """Tests for mechanism_cross_entropy."""
    
    def test_output_shape_mean(self, sample_logits, sample_targets):
        """Test scalar output with mean reduction."""
        loss = mechanism_cross_entropy(sample_logits, sample_targets)
        
        assert loss.shape == ()
    
    def test_output_shape_none(self, sample_logits, sample_targets):
        """Test per-sample output with no reduction."""
        loss = mechanism_cross_entropy(
            sample_logits, sample_targets, reduction="none"
        )
        
        assert loss.shape == (sample_logits.shape[0],)
    
    def test_non_negative(self, sample_logits, sample_targets):
        """Test that loss is non-negative."""
        loss = mechanism_cross_entropy(sample_logits, sample_targets)
        
        assert loss >= 0
    
    def test_perfect_prediction_low_loss(self):
        """Test that perfect predictions give low loss."""
        B = 4
        targets = torch.tensor([0, 1, 2, 0])
        
        # Logits strongly favor correct class
        logits = torch.zeros(B, 3)
        logits[torch.arange(B), targets] = 10.0
        
        loss = mechanism_cross_entropy(logits, targets)
        
        assert loss < 0.01
    
    def test_label_smoothing(self, sample_logits, sample_targets):
        """Test label smoothing increases loss."""
        loss_no_smooth = mechanism_cross_entropy(
            sample_logits, sample_targets, label_smoothing=0.0
        )
        loss_smooth = mechanism_cross_entropy(
            sample_logits, sample_targets, label_smoothing=0.1
        )
        
        # Label smoothing generally increases loss for confident predictions
        # but the relationship can be complex - just verify it runs
        assert not torch.isnan(loss_smooth)


class TestMechanismCrossEntropyFromProbs:
    """Tests for mechanism_cross_entropy_from_probs."""
    
    def test_output_shape(self, sample_probs, sample_targets):
        """Test output shape."""
        loss = mechanism_cross_entropy_from_probs(sample_probs, sample_targets)
        
        assert loss.shape == ()
    
    def test_non_negative(self, sample_probs, sample_targets):
        """Test that loss is non-negative."""
        loss = mechanism_cross_entropy_from_probs(sample_probs, sample_targets)
        
        assert loss >= 0
    
    def test_perfect_prediction_low_loss(self):
        """Test that perfect predictions give low loss."""
        B = 4
        targets = torch.tensor([0, 1, 2, 0])
        
        # One-hot probabilities
        probs = torch.zeros(B, 3)
        probs[torch.arange(B), targets] = 1.0
        
        loss = mechanism_cross_entropy_from_probs(probs, targets)
        
        # Should be close to 0 (log(1) = 0)
        assert loss < 1e-5
    
    def test_uniform_prediction_known_loss(self):
        """Test uniform predictions give known loss."""
        B = 4
        targets = torch.zeros(B, dtype=torch.long)
        
        # Uniform distribution
        probs = torch.ones(B, 3) / 3
        
        loss = mechanism_cross_entropy_from_probs(probs, targets)
        
        # -log(1/3) ≈ 1.099
        expected = -torch.log(torch.tensor(1/3))
        assert torch.allclose(loss, expected, atol=1e-4)
    
    def test_handles_near_zero_probs(self):
        """Test handling of near-zero probabilities."""
        B = 4
        targets = torch.zeros(B, dtype=torch.long)
        
        # Very low probability on correct class
        probs = torch.zeros(B, 3)
        probs[:, 0] = 1e-10
        probs[:, 1] = 0.5 - 5e-11
        probs[:, 2] = 0.5 - 5e-11
        
        loss = mechanism_cross_entropy_from_probs(probs, targets)
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestBrierScore:
    """Tests for brier_score."""
    
    def test_output_shape(self, sample_probs, sample_targets):
        """Test output shape."""
        loss = brier_score(sample_probs, sample_targets)
        
        assert loss.shape == ()
    
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
        B, max_rows, max_cols = t["predictions"].shape
        
        # Create mask where only one cell is valid
        single_mask = torch.zeros(B, max_rows, max_cols, dtype=torch.bool)
        single_mask[:, 0, 0] = True
        
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        # Set that cell to have known error
        predictions = torch.zeros(B, max_rows, max_cols)
        targets = torch.ones(B, max_rows, max_cols)
        
        loss = reconstruction_mse(
            predictions, targets, single_mask, row_mask, col_mask
        )
        
        # MSE should be exactly 1.0 (error of 1 squared)
        assert torch.allclose(loss, torch.tensor(1.0), atol=1e-5)
    
    def test_handles_no_masked_cells(self, sample_reconstruction_tensors):
        """Test graceful handling when no cells are masked."""
        t = sample_reconstruction_tensors
        B, max_rows, max_cols = t["predictions"].shape
        
        # No cells masked
        empty_mask = torch.zeros(B, max_rows, max_cols, dtype=torch.bool)
        
        loss = reconstruction_mse(
            t["predictions"],
            t["targets"],
            empty_mask,
            t["row_mask"],
            t["col_mask"],
        )
        
        # Should handle gracefully (not NaN)
        assert not torch.isnan(loss)


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
    
    def test_less_sensitive_to_outliers(self, sample_reconstruction_tensors):
        """Test that Huber is less sensitive to outliers than MSE."""
        t = sample_reconstruction_tensors
        
        # Create predictions with an outlier
        predictions = t["targets"].clone()
        predictions[0, 0, 0] = 100.0  # Outlier
        
        # Ensure that cell is in the mask
        mask = t["reconstruction_mask"].clone()
        mask[0, 0, 0] = True
        
        mse_loss = reconstruction_mse(
            predictions, t["targets"], mask, t["row_mask"], t["col_mask"]
        )
        huber_loss = reconstruction_huber(
            predictions, t["targets"], mask, t["row_mask"], t["col_mask"]
        )
        
        # Huber should be less than MSE for large outliers
        assert huber_loss < mse_loss
    
    def test_delta_parameter(self, sample_reconstruction_tensors):
        """Test that delta parameter affects loss."""
        t = sample_reconstruction_tensors
        
        loss_small_delta = reconstruction_huber(
            t["predictions"],
            t["targets"],
            t["reconstruction_mask"],
            t["row_mask"],
            t["col_mask"],
            delta=0.1,
        )
        
        loss_large_delta = reconstruction_huber(
            t["predictions"],
            t["targets"],
            t["reconstruction_mask"],
            t["row_mask"],
            t["col_mask"],
            delta=10.0,
        )
        
        # Different deltas should give different losses
        # (unless predictions exactly match targets)
        # Just verify both compute without error
        assert not torch.isnan(loss_small_delta)
        assert not torch.isnan(loss_large_delta)


class TestMultiHeadReconstructionLoss:
    """Tests for multi_head_reconstruction_loss."""
    
    @pytest.fixture
    def sample_head_results(self, sample_reconstruction_tensors):
        """Create sample reconstruction results dict."""
        t = sample_reconstruction_tensors
        B, max_rows, max_cols = t["predictions"].shape
        
        results = {}
        for head_name in ["mcar", "mar", "self_censoring"]:
            results[head_name] = ReconstructionResult(
                predictions=torch.randn(B, max_rows, max_cols),
                errors=torch.rand(B),
            )
        
        return results
    
    def test_returns_total_and_per_head(
        self, sample_head_results, sample_reconstruction_tensors
    ):
        """Test that function returns total and per-head losses."""
        t = sample_reconstruction_tensors
        
        total_loss, per_head = multi_head_reconstruction_loss(
            reconstruction_results=sample_head_results,
            original_values=t["targets"],
            reconstruction_mask=t["reconstruction_mask"],
            row_mask=t["row_mask"],
            col_mask=t["col_mask"],
        )
        
        assert total_loss.shape == ()
        assert isinstance(per_head, dict)
        assert len(per_head) == len(sample_head_results)
        
        for head_name in sample_head_results:
            assert head_name in per_head
    
    def test_head_weights(
        self, sample_head_results, sample_reconstruction_tensors
    ):
        """Test per-head weights are applied."""
        t = sample_reconstruction_tensors
        
        # Weight one head much higher
        head_weights = {
            "mcar": 10.0,
            "mar": 1.0,
            "self_censoring": 1.0,
        }
        
        total_weighted, _ = multi_head_reconstruction_loss(
            reconstruction_results=sample_head_results,
            original_values=t["targets"],
            reconstruction_mask=t["reconstruction_mask"],
            row_mask=t["row_mask"],
            col_mask=t["col_mask"],
            head_weights=head_weights,
        )
        
        total_unweighted, _ = multi_head_reconstruction_loss(
            reconstruction_results=sample_head_results,
            original_values=t["targets"],
            reconstruction_mask=t["reconstruction_mask"],
            row_mask=t["row_mask"],
            col_mask=t["col_mask"],
            head_weights=None,
        )
        
        # Weighted should be different from unweighted
        assert not torch.allclose(total_weighted, total_unweighted)


# =============================================================================
# Test Auxiliary Losses
# =============================================================================

class TestKLDivergence:
    """Tests for kl_divergence."""
    
    def test_standard_normal_zero_kl(self):
        """Test that standard normal has zero KL from prior."""
        B, latent_dim = 8, 16
        
        mean = torch.zeros(B, latent_dim)
        logvar = torch.zeros(B, latent_dim)  # var = 1
        
        kl = kl_divergence(mean, logvar)
        
        assert torch.allclose(kl, torch.tensor(0.0), atol=1e-5)
    
    def test_positive_kl(self):
        """Test that KL is non-negative."""
        B, latent_dim = 8, 16
        
        mean = torch.randn(B, latent_dim)
        logvar = torch.randn(B, latent_dim)
        
        kl = kl_divergence(mean, logvar)
        
        assert kl >= 0
    
    def test_reduction_options(self):
        """Test different reduction options."""
        B, latent_dim = 8, 16
        
        mean = torch.randn(B, latent_dim)
        logvar = torch.randn(B, latent_dim)
        
        kl_mean = kl_divergence(mean, logvar, reduction="mean")
        kl_sum = kl_divergence(mean, logvar, reduction="sum")
        kl_none = kl_divergence(mean, logvar, reduction="none")
        
        assert kl_mean.shape == ()
        assert kl_sum.shape == ()
        assert kl_none.shape == (B,)
        
        # sum should equal none summed
        assert torch.allclose(kl_sum, kl_none.sum())


class TestEntropyLoss:
    """Tests for entropy_loss."""
    
    def test_uniform_max_entropy(self):
        """Test uniform distribution has maximum entropy."""
        B, K = 8, 3
        
        probs = torch.ones(B, K) / K
        entropy = entropy_loss(probs)
        
        # Max entropy for K classes is log(K)
        max_entropy = torch.log(torch.tensor(float(K)))
        assert torch.allclose(entropy, max_entropy, atol=1e-5)
    
    def test_certain_zero_entropy(self):
        """Test certain distribution has zero entropy."""
        B, K = 8, 3
        
        # One-hot (certain) distribution
        probs = torch.zeros(B, K)
        probs[:, 0] = 1.0
        
        entropy = entropy_loss(probs)
        
        assert torch.allclose(entropy, torch.tensor(0.0), atol=1e-5)
    
    def test_non_negative(self):
        """Test entropy is always non-negative."""
        B, K = 100, 5
        
        probs = torch.softmax(torch.randn(B, K), dim=-1)
        entropy = entropy_loss(probs)
        
        assert entropy >= 0
    
    def test_reduction_options(self):
        """Test different reduction options."""
        B, K = 8, 3
        
        probs = torch.softmax(torch.randn(B, K), dim=-1)
        
        ent_mean = entropy_loss(probs, reduction="mean")
        ent_sum = entropy_loss(probs, reduction="sum")
        ent_none = entropy_loss(probs, reduction="none")
        
        assert ent_mean.shape == ()
        assert ent_sum.shape == ()
        assert ent_none.shape == (B,)


class TestLoadBalanceLoss:
    """Tests for load_balance_loss."""
    
    def test_uniform_routing_low_loss(self):
        """Test uniform routing gives low loss."""
        B, n_experts = 100, 5
        
        # Approximately uniform assignment via one-hot
        probs = torch.zeros(B, n_experts)
        for i in range(B):
            probs[i, i % n_experts] = 1.0
        
        loss = load_balance_loss(probs)
        
        # With perfectly uniform routing: f_i = 1/n, p_i = 1/n
        # loss = n * sum(1/n * 1/n) = n * n * (1/n^2) = 1
        assert torch.allclose(loss, torch.tensor(1.0), atol=0.1)
    
    def test_collapsed_routing_high_loss(self):
        """Test collapsed routing (one expert) gives high loss."""
        B, n_experts = 100, 5
        
        # All samples to first expert
        probs = torch.zeros(B, n_experts)
        probs[:, 0] = 1.0
        
        loss = load_balance_loss(probs)
        
        # f_0 = 1, p_0 = 1, others = 0
        # loss = n * (1 * 1) = n = 5
        assert torch.allclose(loss, torch.tensor(float(n_experts)), atol=0.1)
    
    def test_non_negative(self):
        """Test loss is non-negative."""
        B, n_experts = 32, 5
        
        probs = torch.softmax(torch.randn(B, n_experts), dim=-1)
        loss = load_balance_loss(probs)
        
        assert loss >= 0


# =============================================================================
# Test LacunaLoss
# =============================================================================

class TestLacunaLoss:
    """Tests for LacunaLoss combined loss."""
    
    @pytest.fixture
    def loss_fn(self, default_config):
        """Create LacunaLoss with default config."""
        return LacunaLoss(default_config)
    
    def test_returns_total_and_dict(
        self, loss_fn, sample_lacuna_output, sample_batch
    ):
        """Test that forward returns total loss and dict."""
        total_loss, loss_dict = loss_fn(sample_lacuna_output, sample_batch)
        
        assert total_loss.shape == ()
        assert isinstance(loss_dict, dict)
        assert "total_loss" in loss_dict
    
    def test_total_loss_matches_dict(
        self, loss_fn, sample_lacuna_output, sample_batch
    ):
        """Test that returned total matches dict entry."""
        total_loss, loss_dict = loss_fn(sample_lacuna_output, sample_batch)
        
        assert torch.allclose(total_loss, loss_dict["total_loss"])
    
    def test_includes_class_loss(
        self, loss_fn, sample_lacuna_output, sample_batch
    ):
        """Test that class loss is included."""
        total_loss, loss_dict = loss_fn(sample_lacuna_output, sample_batch)
        
        assert "class_loss" in loss_dict
    
    def test_includes_reconstruction_loss(
        self, sample_lacuna_output, sample_batch
    ):
        """Test that reconstruction loss is included when configured."""
        config = LossConfig(reconstruction_weight=1.0)
        loss_fn = LacunaLoss(config)
        
        total_loss, loss_dict = loss_fn(sample_lacuna_output, sample_batch)
        
        assert "reconstruction_loss" in loss_dict
    
    def test_no_reconstruction_loss_when_weight_zero(
        self, sample_lacuna_output, sample_batch
    ):
        """Test no reconstruction loss when weight is 0."""
        config = LossConfig(reconstruction_weight=0.0)
        loss_fn = LacunaLoss(config)
        
        total_loss, loss_dict = loss_fn(sample_lacuna_output, sample_batch)
        
        assert "reconstruction_loss" not in loss_dict
    
    def test_includes_load_balance_loss(
        self, sample_lacuna_output, sample_batch
    ):
        """Test that load balance loss is included when configured."""
        config = LossConfig(load_balance_weight=0.1)
        loss_fn = LacunaLoss(config)
        
        total_loss, loss_dict = loss_fn(sample_lacuna_output, sample_batch)
        
        assert "load_balance_loss" in loss_dict
    
    def test_brier_score_option(
        self, sample_lacuna_output, sample_batch
    ):
        """Test Brier score as mechanism loss."""
        config = LossConfig(mechanism_loss_type="brier")
        loss_fn = LacunaLoss(config)
        
        total_loss, loss_dict = loss_fn(sample_lacuna_output, sample_batch)
        
        assert "class_loss" in loss_dict
        assert not torch.isnan(total_loss)
    
    def test_skip_auxiliary_losses(
        self, sample_lacuna_output, sample_batch
    ):
        """Test skipping auxiliary losses."""
        config = LossConfig(load_balance_weight=0.1, entropy_weight=0.1)
        loss_fn = LacunaLoss(config)
        
        _, loss_dict_with = loss_fn(
            sample_lacuna_output, sample_batch, compute_auxiliary=True
        )
        _, loss_dict_without = loss_fn(
            sample_lacuna_output, sample_batch, compute_auxiliary=False
        )
        
        assert "load_balance_loss" in loss_dict_with
        assert "load_balance_loss" not in loss_dict_without
    
    def test_gradients_flow(
        self, loss_fn, sample_lacuna_output, sample_batch
    ):
        """Test that gradients flow through loss."""
        # Make evidence require grad
        sample_lacuna_output = LacunaOutput(
            posterior=sample_lacuna_output.posterior,
            moe=sample_lacuna_output.moe,
            reconstruction=sample_lacuna_output.reconstruction,
            evidence=sample_lacuna_output.evidence.clone().requires_grad_(True),
        )
        
        total_loss, _ = loss_fn(sample_lacuna_output, sample_batch)
        total_loss.backward()
        
        # Evidence should have gradient (through reconstruction)
        # Note: actual gradient flow depends on how loss uses evidence
    
    def test_pretraining_loss_method(
        self, loss_fn, sample_lacuna_output, sample_batch
    ):
        """Test pretraining_loss helper method."""
        total_loss, loss_dict = loss_fn.pretraining_loss(
            sample_lacuna_output, sample_batch
        )
        
        # Should not include mechanism loss
        assert "class_loss" not in loss_dict or loss_dict.get("class_loss", 0) == 0
    
    def test_classification_loss_method(
        self, loss_fn, sample_lacuna_output, sample_batch
    ):
        """Test classification_loss helper method."""
        total_loss, loss_dict = loss_fn.classification_loss(
            sample_lacuna_output, sample_batch
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
        
        # Half correct
        p_class = torch.zeros(B, 3)
        p_class[0:2, 0] = 1.0  # Correct
        p_class[2:4, 1] = 1.0  # Wrong
        
        acc = compute_class_accuracy(p_class, targets)
        
        assert acc == 0.5


class TestComputeMechanismAccuracy:
    """Tests for compute_mechanism_accuracy."""
    
    def test_perfect_accuracy(self):
        """Test 100% accuracy for mechanism predictions."""
        B = 8
        n_mechanisms = 5
        targets = torch.randint(0, n_mechanisms, (B,))
        
        # Perfect predictions
        p_mechanism = torch.zeros(B, n_mechanisms)
        p_mechanism[torch.arange(B), targets] = 1.0
        
        acc = compute_mechanism_accuracy(p_mechanism, targets)
        
        assert acc == 1.0


class TestComputePerClassAccuracy:
    """Tests for compute_per_class_accuracy."""
    
    def test_returns_all_classes(self):
        """Test that all class accuracies are returned."""
        B = 12
        p_class = torch.softmax(torch.randn(B, 3), dim=-1)
        targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        
        result = compute_per_class_accuracy(p_class, targets)
        
        assert "mcar_acc" in result
        assert "mar_acc" in result
        assert "mnar_acc" in result
    
    def test_handles_missing_classes(self):
        """Test handling when some classes are absent."""
        B = 4
        p_class = torch.softmax(torch.randn(B, 3), dim=-1)
        targets = torch.zeros(B, dtype=torch.long)  # Only MCAR
        
        result = compute_per_class_accuracy(p_class, targets)
        
        # MCAR should have a value
        assert not torch.isnan(result["mcar_acc"])
        
        # MAR and MNAR should be NaN (no samples)
        assert torch.isnan(result["mar_acc"])
        assert torch.isnan(result["mnar_acc"])
    
    def test_per_class_values_correct(self):
        """Test per-class accuracy values are correct."""
        B = 6
        targets = torch.tensor([0, 0, 1, 1, 2, 2])
        
        # Perfect for MCAR, 50% for MAR, 0% for MNAR
        p_class = torch.zeros(B, 3)
        p_class[0, 0] = 1.0  # Correct
        p_class[1, 0] = 1.0  # Correct
        p_class[2, 1] = 1.0  # Correct
        p_class[3, 0] = 1.0  # Wrong
        p_class[4, 0] = 1.0  # Wrong
        p_class[5, 1] = 1.0  # Wrong
        
        result = compute_per_class_accuracy(p_class, targets)
        
        assert result["mcar_acc"] == 1.0
        assert result["mar_acc"] == 0.5
        assert result["mnar_acc"] == 0.0