"""
Tests for lacuna.models.reconstruction

Tests the reconstruction heads for self-supervised pretraining:
    - BaseReconstructionHead: Abstract base class
    - MCARHead: Simple MLP reconstruction
    - MARHead: Cross-attention to observed cells
    - MNARSelfCensoringHead: Censoring-aware reconstruction
    - MNARThresholdHead: Threshold-aware reconstruction
    - MNARLatentHead: Latent-conditioned reconstruction
    - ReconstructionHeads: Container for all heads
"""

import pytest
import torch
import torch.nn as nn

from lacuna.models.reconstruction import (
    # Config
    ReconstructionConfig,
    # Base class
    BaseReconstructionHead,
    # Head implementations
    MCARHead,
    MARHead,
    MNARSelfCensoringHead,
    MNARThresholdHead,
    MNARLatentHead,
    # Registry
    HEAD_REGISTRY,
    create_head,
    # Container
    ReconstructionHeads,
    create_reconstruction_heads,
)
from lacuna.core.types import ReconstructionResult
from lacuna.data.tokenization import TOKEN_DIM, IDX_OBSERVED


# =============================================================================
# Test ReconstructionConfig
# =============================================================================

class TestReconstructionConfig:
    """Tests for ReconstructionConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ReconstructionConfig()
        
        assert config.hidden_dim == 128
        assert config.head_hidden_dim == 64
        assert config.n_head_layers == 2
        assert config.dropout == 0.1
        assert config.mnar_variants == ["self_censoring", "threshold", "latent"]
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ReconstructionConfig(
            hidden_dim=256,
            head_hidden_dim=128,
            n_head_layers=3,
            dropout=0.2,
            mnar_variants=["self_censoring", "threshold"],
        )
        
        assert config.hidden_dim == 256
        assert config.head_hidden_dim == 128
        assert config.n_head_layers == 3
        assert config.dropout == 0.2
        assert config.mnar_variants == ["self_censoring", "threshold"]
    
    def test_mnar_variants_default_initialization(self):
        """Test that mnar_variants defaults to full list if None."""
        config = ReconstructionConfig(mnar_variants=None)
        
        assert config.mnar_variants == ["self_censoring", "threshold", "latent"]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return ReconstructionConfig(
        hidden_dim=64,
        head_hidden_dim=32,
        n_head_layers=2,
        dropout=0.0,
        mnar_variants=["self_censoring", "threshold", "latent"],
    )


@pytest.fixture
def sample_inputs():
    """Create sample input tensors."""
    B, max_rows, max_cols, hidden_dim = 4, 32, 16, 64
    
    token_repr = torch.randn(B, max_rows, max_cols, hidden_dim)
    
    # Create tokens with observed indicator
    tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
    tokens[..., IDX_OBSERVED] = (torch.rand(B, max_rows, max_cols) > 0.2).float()
    
    row_mask = torch.ones(B, max_rows, dtype=torch.bool)
    col_mask = torch.ones(B, max_cols, dtype=torch.bool)
    
    original_values = torch.randn(B, max_rows, max_cols)
    reconstruction_mask = torch.rand(B, max_rows, max_cols) < 0.15  # 15% masked
    
    return {
        "token_repr": token_repr,
        "tokens": tokens,
        "row_mask": row_mask,
        "col_mask": col_mask,
        "original_values": original_values,
        "reconstruction_mask": reconstruction_mask,
    }


# =============================================================================
# Test MCARHead
# =============================================================================

class TestMCARHead:
    """Tests for MCARHead."""
    
    @pytest.fixture
    def head(self, config):
        """Create MCARHead."""
        return MCARHead(config)
    
    def test_output_shape(self, head, sample_inputs):
        """Test output tensor shape."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        B, max_rows, max_cols = 4, 32, 16
        assert predictions.shape == (B, max_rows, max_cols)
    
    def test_no_nan_or_inf(self, head, sample_inputs):
        """Test that output contains no NaN or Inf values."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
    
    def test_compute_error(self, head, sample_inputs):
        """Test error computation."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        per_sample_error, per_cell_error = head.compute_error(
            predictions,
            sample_inputs["original_values"],
            sample_inputs["reconstruction_mask"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        assert per_sample_error.shape == (4,)
        assert per_cell_error.shape == (4, 32, 16)
        assert (per_sample_error >= 0).all()
    
    def test_gradients_flow(self, head, sample_inputs):
        """Test that gradients flow through the head."""
        sample_inputs["token_repr"].requires_grad_(True)
        
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        loss = predictions.sum()
        loss.backward()
        
        assert sample_inputs["token_repr"].grad is not None


# =============================================================================
# Test MARHead
# =============================================================================

class TestMARHead:
    """Tests for MARHead."""
    
    @pytest.fixture
    def head(self, config):
        """Create MARHead."""
        return MARHead(config)
    
    def test_output_shape(self, head, sample_inputs):
        """Test output tensor shape."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        B, max_rows, max_cols = 4, 32, 16
        assert predictions.shape == (B, max_rows, max_cols)
    
    def test_no_nan_or_inf(self, head, sample_inputs):
        """Test that output contains no NaN or Inf values."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
    
    def test_uses_cross_attention(self, head, sample_inputs):
        """Test that cross-attention layers exist."""
        assert hasattr(head, 'query_proj')
        assert hasattr(head, 'key_proj')
        assert hasattr(head, 'value_proj')
    
    def test_attends_to_observed_only(self, head, sample_inputs):
        """Test that attention only considers observed cells."""
        # Create input where only first column is observed
        tokens = sample_inputs["tokens"].clone()
        tokens[..., IDX_OBSERVED] = 0.0
        tokens[:, :, 0, IDX_OBSERVED] = 1.0  # Only first column observed
        
        predictions = head(
            sample_inputs["token_repr"],
            tokens,
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        # Should still produce valid output
        assert predictions.shape == (4, 32, 16)
        assert not torch.isnan(predictions).any()
    
    def test_gradients_flow(self, head, sample_inputs):
        """Test that gradients flow through the head."""
        sample_inputs["token_repr"].requires_grad_(True)
        
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        loss = predictions.sum()
        loss.backward()
        
        assert sample_inputs["token_repr"].grad is not None


# =============================================================================
# Test MNARSelfCensoringHead
# =============================================================================

class TestMNARSelfCensoringHead:
    """Tests for MNARSelfCensoringHead."""
    
    @pytest.fixture
    def head(self, config):
        """Create MNARSelfCensoringHead."""
        return MNARSelfCensoringHead(config)
    
    def test_output_shape(self, head, sample_inputs):
        """Test output tensor shape."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        B, max_rows, max_cols = 4, 32, 16
        assert predictions.shape == (B, max_rows, max_cols)
    
    def test_no_nan_or_inf(self, head, sample_inputs):
        """Test that output contains no NaN or Inf values."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
    
    def test_has_censoring_predictor(self, head):
        """Test that censoring predictor exists."""
        assert hasattr(head, 'censoring_predictor')
        assert hasattr(head, 'value_predictor')
        assert hasattr(head, 'bias_adjustment')
    
    def test_get_censoring_scores(self, head, sample_inputs):
        """Test getting censoring scores."""
        scores = head.get_censoring_scores(sample_inputs["token_repr"])
        
        assert scores.shape == (4, 32, 16)
        # Scores should be in [0, 1] after sigmoid
        assert (scores >= 0).all()
        assert (scores <= 1).all()
    
    def test_gradients_flow(self, head, sample_inputs):
        """Test that gradients flow through the head."""
        sample_inputs["token_repr"].requires_grad_(True)
        
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        loss = predictions.sum()
        loss.backward()
        
        assert sample_inputs["token_repr"].grad is not None


# =============================================================================
# Test MNARThresholdHead
# =============================================================================

class TestMNARThresholdHead:
    """Tests for MNARThresholdHead."""
    
    @pytest.fixture
    def head(self, config):
        """Create MNARThresholdHead."""
        return MNARThresholdHead(config)
    
    def test_output_shape(self, head, sample_inputs):
        """Test output tensor shape."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        B, max_rows, max_cols = 4, 32, 16
        assert predictions.shape == (B, max_rows, max_cols)
    
    def test_no_nan_or_inf(self, head, sample_inputs):
        """Test that output contains no NaN or Inf values."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
    
    def test_has_threshold_components(self, head):
        """Test that threshold-related components exist."""
        assert hasattr(head, 'threshold_estimator')
        assert hasattr(head, 'normal_predictor')
        assert hasattr(head, 'truncated_predictor')
        assert hasattr(head, 'selection_gate')
    
    def test_get_thresholds(self, head, sample_inputs):
        """Test getting estimated thresholds."""
        thresholds = head.get_thresholds(sample_inputs["token_repr"])
        
        # Should output [lower, upper] thresholds
        assert thresholds.shape == (4, 32, 16, 2)
    
    def test_gradients_flow(self, head, sample_inputs):
        """Test that gradients flow through the head."""
        sample_inputs["token_repr"].requires_grad_(True)
        
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        loss = predictions.sum()
        loss.backward()
        
        assert sample_inputs["token_repr"].grad is not None


# =============================================================================
# Test MNARLatentHead
# =============================================================================

class TestMNARLatentHead:
    """Tests for MNARLatentHead."""
    
    @pytest.fixture
    def head(self, config):
        """Create MNARLatentHead."""
        return MNARLatentHead(config, latent_dim=16)
    
    def test_output_shape(self, head, sample_inputs):
        """Test output tensor shape."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        B, max_rows, max_cols = 4, 32, 16
        assert predictions.shape == (B, max_rows, max_cols)
    
    def test_no_nan_or_inf(self, head, sample_inputs):
        """Test that output contains no NaN or Inf values."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
    
    def test_has_latent_components(self, head):
        """Test that latent-related components exist."""
        assert hasattr(head, 'latent_encoder')
        assert hasattr(head, 'predictor')
        assert head.latent_dim == 16
    
    def test_get_latent(self, head, sample_inputs):
        """Test getting latent mean and logvar."""
        latent_mean, latent_logvar = head.get_latent(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["col_mask"],
        )
        
        B, max_rows = 4, 32
        assert latent_mean.shape == (B, max_rows, 16)  # latent_dim=16
        assert latent_logvar.shape == (B, max_rows, 16)
    
    def test_training_uses_reparameterization(self, head, sample_inputs):
        """Test that training mode uses reparameterization."""
        head.train()
        
        # Run twice - should get different outputs due to sampling
        pred1 = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        pred2 = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        # Predictions should differ due to stochastic sampling
        assert not torch.allclose(pred1, pred2)
    
    def test_eval_uses_mean(self, head, sample_inputs):
        """Test that eval mode uses mean (deterministic)."""
        head.eval()
        
        with torch.no_grad():
            pred1 = head(
                sample_inputs["token_repr"],
                sample_inputs["tokens"],
                sample_inputs["row_mask"],
                sample_inputs["col_mask"],
            )
            
            pred2 = head(
                sample_inputs["token_repr"],
                sample_inputs["tokens"],
                sample_inputs["row_mask"],
                sample_inputs["col_mask"],
            )
        
        # Predictions should be identical in eval mode
        assert torch.allclose(pred1, pred2)
    
    def test_gradients_flow(self, head, sample_inputs):
        """Test that gradients flow through the head."""
        head.train()
        sample_inputs["token_repr"].requires_grad_(True)
        
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        loss = predictions.sum()
        loss.backward()
        
        assert sample_inputs["token_repr"].grad is not None


# =============================================================================
# Test HEAD_REGISTRY and create_head
# =============================================================================

class TestHeadRegistry:
    """Tests for HEAD_REGISTRY and create_head factory."""
    
    def test_registry_contains_all_heads(self):
        """Test that registry contains all head types."""
        expected_heads = ["mcar", "mar", "self_censoring", "threshold", "latent"]
        
        for name in expected_heads:
            assert name in HEAD_REGISTRY
    
    def test_create_head_mcar(self, config):
        """Test creating MCAR head via factory."""
        head = create_head("mcar", config)
        
        assert isinstance(head, MCARHead)
    
    def test_create_head_mar(self, config):
        """Test creating MAR head via factory."""
        head = create_head("mar", config)
        
        assert isinstance(head, MARHead)
    
    def test_create_head_self_censoring(self, config):
        """Test creating self-censoring head via factory."""
        head = create_head("self_censoring", config)
        
        assert isinstance(head, MNARSelfCensoringHead)
    
    def test_create_head_threshold(self, config):
        """Test creating threshold head via factory."""
        head = create_head("threshold", config)
        
        assert isinstance(head, MNARThresholdHead)
    
    def test_create_head_latent(self, config):
        """Test creating latent head via factory."""
        head = create_head("latent", config)
        
        assert isinstance(head, MNARLatentHead)
    
    def test_create_head_invalid_raises(self, config):
        """Test that creating invalid head raises error."""
        with pytest.raises(ValueError, match="Unknown head type"):
            create_head("invalid_head", config)


# =============================================================================
# Test ReconstructionHeads Container
# =============================================================================

class TestReconstructionHeads:
    """Tests for ReconstructionHeads container."""
    
    @pytest.fixture
    def heads(self, config):
        """Create ReconstructionHeads container."""
        return ReconstructionHeads(config)
    
    def test_n_heads_property(self, heads):
        """Test n_heads property."""
        # MCAR + MAR + 3 MNAR variants = 5
        assert heads.n_heads == 5
    
    def test_head_names_property(self, heads):
        """Test head_names ordering."""
        expected = ["mcar", "mar", "self_censoring", "threshold", "latent"]
        assert heads.head_names == expected
    
    def test_forward_returns_dict(self, heads, sample_inputs):
        """Test that forward returns dict of ReconstructionResult."""
        results = heads(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        assert isinstance(results, dict)
        assert len(results) == 5
        
        for name in heads.head_names:
            assert name in results
            assert isinstance(results[name], ReconstructionResult)
    
    def test_forward_with_targets(self, heads, sample_inputs):
        """Test forward with original values and reconstruction mask."""
        results = heads(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
            original_values=sample_inputs["original_values"],
            reconstruction_mask=sample_inputs["reconstruction_mask"],
        )
        
        for name in heads.head_names:
            result = results[name]
            assert result.predictions.shape == (4, 32, 16)
            assert result.errors.shape == (4,)
            assert result.per_cell_errors is not None
    
    def test_forward_without_targets(self, heads, sample_inputs):
        """Test forward without targets returns zero errors."""
        results = heads(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
            # No original_values or reconstruction_mask
        )
        
        for name in heads.head_names:
            result = results[name]
            assert result.predictions.shape == (4, 32, 16)
            assert (result.errors == 0).all()
            assert result.per_cell_errors is None
    
    def test_get_error_tensor(self, heads, sample_inputs):
        """Test get_error_tensor method."""
        results = heads(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
            original_values=sample_inputs["original_values"],
            reconstruction_mask=sample_inputs["reconstruction_mask"],
        )
        
        error_tensor = heads.get_error_tensor(results)
        
        assert error_tensor.shape == (4, 5)  # B, n_heads
    
    def test_get_predictions_dict(self, heads, sample_inputs):
        """Test get_predictions_dict method."""
        results = heads(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        predictions = heads.get_predictions_dict(results)
        
        assert isinstance(predictions, dict)
        assert len(predictions) == 5
        
        for name in heads.head_names:
            assert predictions[name].shape == (4, 32, 16)
    
    def test_no_nan_or_inf_in_results(self, heads, sample_inputs):
        """Test that all results contain no NaN or Inf values."""
        results = heads(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
            original_values=sample_inputs["original_values"],
            reconstruction_mask=sample_inputs["reconstruction_mask"],
        )
        
        for name in heads.head_names:
            result = results[name]
            assert not torch.isnan(result.predictions).any()
            assert not torch.isinf(result.predictions).any()
            assert not torch.isnan(result.errors).any()
    
    def test_gradients_flow_through_all_heads(self, heads, sample_inputs):
        """Test that gradients flow through all heads."""
        sample_inputs["token_repr"].requires_grad_(True)
        
        results = heads(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        # Sum all predictions for loss
        loss = sum(results[name].predictions.sum() for name in heads.head_names)
        loss.backward()
        
        assert sample_inputs["token_repr"].grad is not None


# =============================================================================
# Test create_reconstruction_heads Factory
# =============================================================================

class TestCreateReconstructionHeads:
    """Tests for create_reconstruction_heads factory function."""
    
    def test_default_configuration(self):
        """Test creating heads with default configuration."""
        heads = create_reconstruction_heads()
        
        assert isinstance(heads, ReconstructionHeads)
        assert heads.n_heads == 5
    
    def test_custom_configuration(self):
        """Test creating heads with custom configuration."""
        heads = create_reconstruction_heads(
            hidden_dim=256,
            head_hidden_dim=128,
            n_head_layers=3,
            dropout=0.2,
            mnar_variants=["self_censoring", "threshold"],
        )
        
        assert heads.n_heads == 4  # MCAR + MAR + 2 variants
        assert heads.config.hidden_dim == 256
        assert heads.config.head_hidden_dim == 128
    
    def test_forward_pass(self):
        """Test forward pass with factory-created heads."""
        heads = create_reconstruction_heads(
            hidden_dim=64,
            head_hidden_dim=32,
            mnar_variants=["self_censoring"],
        )
        
        B, max_rows, max_cols, hidden_dim = 4, 32, 16, 64
        token_repr = torch.randn(B, max_rows, max_cols, hidden_dim)
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        results = heads(token_repr, tokens, row_mask, col_mask)
        
        assert len(results) == 3  # MCAR + MAR + self_censoring


# =============================================================================
# Test BaseReconstructionHead.compute_error
# =============================================================================

class TestComputeError:
    """Tests for error computation in reconstruction heads."""
    
    @pytest.fixture
    def head(self, config):
        """Create a head for testing."""
        return MCARHead(config)
    
    def test_error_only_on_masked_cells(self, head):
        """Test that error is only computed on masked cells."""
        B, max_rows, max_cols = 2, 8, 4
        
        predictions = torch.randn(B, max_rows, max_cols)
        targets = torch.randn(B, max_rows, max_cols)
        
        # Only mask first cell
        reconstruction_mask = torch.zeros(B, max_rows, max_cols, dtype=torch.bool)
        reconstruction_mask[:, 0, 0] = True
        
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        per_sample_error, per_cell_error = head.compute_error(
            predictions, targets, reconstruction_mask, row_mask, col_mask
        )
        
        # Per-sample error should be error at [0,0] only
        expected = (predictions[:, 0, 0] - targets[:, 0, 0]) ** 2
        assert torch.allclose(per_sample_error, expected)
    
    def test_respects_row_mask(self, head):
        """Test that row mask is respected."""
        B, max_rows, max_cols = 2, 8, 4
        
        predictions = torch.randn(B, max_rows, max_cols)
        targets = torch.randn(B, max_rows, max_cols)
        
        # Mask many cells
        reconstruction_mask = torch.ones(B, max_rows, max_cols, dtype=torch.bool)
        
        # But only first 2 rows are valid
        row_mask = torch.zeros(B, max_rows, dtype=torch.bool)
        row_mask[:, :2] = True
        
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        per_sample_error, _ = head.compute_error(
            predictions, targets, reconstruction_mask, row_mask, col_mask
        )
        
        # Error should only consider first 2 rows
        valid_errors = (predictions[:, :2, :] - targets[:, :2, :]) ** 2
        expected = valid_errors.mean(dim=(1, 2))
        assert torch.allclose(per_sample_error, expected)
    
    def test_respects_col_mask(self, head):
        """Test that column mask is respected."""
        B, max_rows, max_cols = 2, 8, 4
        
        predictions = torch.randn(B, max_rows, max_cols)
        targets = torch.randn(B, max_rows, max_cols)
        
        reconstruction_mask = torch.ones(B, max_rows, max_cols, dtype=torch.bool)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        
        # Only first 2 columns valid
        col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
        col_mask[:, :2] = True
        
        per_sample_error, _ = head.compute_error(
            predictions, targets, reconstruction_mask, row_mask, col_mask
        )
        
        # Error should only consider first 2 columns
        valid_errors = (predictions[:, :, :2] - targets[:, :, :2]) ** 2
        expected = valid_errors.mean(dim=(1, 2))
        assert torch.allclose(per_sample_error, expected)
    
    def test_handles_no_masked_cells(self, head):
        """Test handling of no masked cells."""
        B, max_rows, max_cols = 2, 8, 4
        
        predictions = torch.randn(B, max_rows, max_cols)
        targets = torch.randn(B, max_rows, max_cols)
        
        # No cells masked
        reconstruction_mask = torch.zeros(B, max_rows, max_cols, dtype=torch.bool)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        per_sample_error, _ = head.compute_error(
            predictions, targets, reconstruction_mask, row_mask, col_mask
        )
        
        # Should handle gracefully (clamp min=1.0 prevents div by zero)
        assert not torch.isnan(per_sample_error).any()