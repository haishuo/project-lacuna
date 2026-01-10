"""
Tests for lacuna.models.encoder

Tests the BERT-inspired transformer encoder:
    - TokenEmbedding: Projects raw tokens to hidden dimension
    - TransformerLayer: Self-attention over feature tokens within rows
    - RowPooling: Aggregates features into row representations
    - DatasetPooling: Aggregates rows into dataset representation
    - LacunaEncoder: Complete encoder pipeline
"""

import pytest
import torch
import torch.nn as nn

from lacuna.models.encoder import (
    EncoderConfig,
    TokenEmbedding,
    TransformerLayer,
    AttentionPooling,
    RowPooling,
    DatasetPooling,
    LacunaEncoder,
    create_encoder,
)
from lacuna.data.tokenization import TOKEN_DIM, IDX_VALUE, IDX_OBSERVED, IDX_MASK_TYPE, IDX_FEATURE_ID


# =============================================================================
# Helper Functions
# =============================================================================

def create_valid_tokens(B: int, max_rows: int, max_cols: int, n_valid_cols: int = None) -> torch.Tensor:
    """
    Create properly structured token tensors for testing.
    
    Tokens have structure: [value, is_observed, mask_type, feature_id_normalized]
    - value: continuous float (can be any value)
    - is_observed: binary 0.0 or 1.0
    - mask_type: binary 0.0 or 1.0  
    - feature_id_normalized: float in [0, 1]
    
    Args:
        B: Batch size
        max_rows: Maximum number of rows
        max_cols: Maximum number of columns
        n_valid_cols: Number of valid columns (defaults to max_cols)
    
    Returns:
        tokens: [B, max_rows, max_cols, TOKEN_DIM] properly structured tensor
    """
    if n_valid_cols is None:
        n_valid_cols = max_cols
    
    tokens = torch.zeros(B, max_rows, max_cols, TOKEN_DIM)
    
    # Value: random continuous values (normalized roughly to [-3, 3])
    tokens[..., IDX_VALUE] = torch.randn(B, max_rows, max_cols)
    
    # is_observed: binary (randomly set ~80% as observed)
    tokens[..., IDX_OBSERVED] = (torch.rand(B, max_rows, max_cols) > 0.2).float()
    
    # mask_type: binary (mostly natural=0, some artificial=1)
    tokens[..., IDX_MASK_TYPE] = (torch.rand(B, max_rows, max_cols) > 0.9).float()
    
    # feature_id_normalized: float in [0, 1] representing column position
    # For each column j, feature_id = j / (max_cols - 1) if max_cols > 1, else 0
    for j in range(max_cols):
        tokens[..., j, IDX_FEATURE_ID] = j / max(max_cols - 1, 1)
    
    return tokens


def create_masks(B: int, max_rows: int, max_cols: int, 
                 n_valid_rows: int = None, n_valid_cols: int = None) -> tuple:
    """
    Create row and column masks for testing.
    
    Args:
        B: Batch size
        max_rows: Maximum number of rows
        max_cols: Maximum number of columns
        n_valid_rows: Number of valid rows per sample (defaults to max_rows)
        n_valid_cols: Number of valid columns per sample (defaults to max_cols)
    
    Returns:
        row_mask: [B, max_rows] boolean mask
        col_mask: [B, max_cols] boolean mask
    """
    if n_valid_rows is None:
        n_valid_rows = max_rows
    if n_valid_cols is None:
        n_valid_cols = max_cols
    
    row_mask = torch.zeros(B, max_rows, dtype=torch.bool)
    row_mask[:, :n_valid_rows] = True
    
    col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
    col_mask[:, :n_valid_cols] = True
    
    return row_mask, col_mask


# =============================================================================
# Test EncoderConfig
# =============================================================================

class TestEncoderConfig:
    """Tests for EncoderConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = EncoderConfig()
        
        assert config.hidden_dim == 128
        assert config.evidence_dim == 64
        assert config.n_layers == 4
        assert config.n_heads == 4
        assert config.dropout == 0.1
        assert config.row_pooling == "attention"
        assert config.dataset_pooling == "attention"
        assert config.max_cols == 32
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = EncoderConfig(
            hidden_dim=256,
            evidence_dim=128,
            n_layers=6,
            n_heads=8,
            dropout=0.2,
            row_pooling="mean",
            dataset_pooling="max",
            max_cols=64,
        )
        
        assert config.hidden_dim == 256
        assert config.evidence_dim == 128
        assert config.n_layers == 6
        assert config.n_heads == 8
        assert config.dropout == 0.2
        assert config.row_pooling == "mean"
        assert config.dataset_pooling == "max"
        assert config.max_cols == 64
    
    def test_ff_dim_default(self):
        """Test that ff_dim defaults to 4 * hidden_dim."""
        config = EncoderConfig(hidden_dim=128)
        assert config.ff_dim == 512
        
        config = EncoderConfig(hidden_dim=256)
        assert config.ff_dim == 1024
    
    def test_ff_dim_custom(self):
        """Test custom ff_dim."""
        config = EncoderConfig(hidden_dim=128, ff_dim=256)
        assert config.ff_dim == 256
    
    def test_hidden_dim_divisible_by_n_heads(self):
        """Test that hidden_dim must be divisible by n_heads."""
        # Valid: 128 / 4 = 32
        config = EncoderConfig(hidden_dim=128, n_heads=4)
        assert config.hidden_dim % config.n_heads == 0
        
        # Invalid: 128 / 3 is not integer
        with pytest.raises(ValueError):
            EncoderConfig(hidden_dim=128, n_heads=3)


# =============================================================================
# Test TokenEmbedding
# =============================================================================

class TestTokenEmbedding:
    """Tests for TokenEmbedding layer."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return EncoderConfig(
            hidden_dim=64,
            max_cols=16,
            dropout=0.0,  # Disable dropout for deterministic tests
        )
    
    @pytest.fixture
    def embedding(self, config):
        """Create TokenEmbedding layer."""
        return TokenEmbedding(config)
    
    def test_output_shape(self, embedding):
        """Test output tensor shape."""
        B, max_rows, max_cols = 4, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        _, col_mask = create_masks(B, max_rows, max_cols)
        
        output = embedding(tokens, col_mask)
        
        assert output.shape == (B, max_rows, max_cols, 64)  # hidden_dim=64
    
    def test_padding_is_zeroed(self, embedding):
        """Test that padding positions are zeroed out."""
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        
        # Only first 8 columns are valid
        _, col_mask = create_masks(B, max_rows, max_cols, n_valid_cols=8)
        
        output = embedding(tokens, col_mask)
        
        # Padding positions should be zero
        assert (output[:, :, 8:, :] == 0).all()
    
    def test_handles_variable_valid_cols(self, embedding):
        """Test handling of variable number of valid columns per sample."""
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        
        # Different valid columns per sample
        col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
        col_mask[0, :6] = True   # Sample 0: 6 valid columns
        col_mask[1, :12] = True  # Sample 1: 12 valid columns
        
        output = embedding(tokens, col_mask)
        
        # Check padding is zeroed correctly for each sample
        assert (output[0, :, 6:, :] == 0).all()
        assert (output[1, :, 12:, :] == 0).all()
        
        # But valid positions should have non-zero values (at least some)
        # Note: We check sum != 0 because some valid positions could coincidentally be near zero
        assert output[0, :, :6, :].abs().sum() > 0
        assert output[1, :, :12, :].abs().sum() > 0
    
    def test_no_nan_or_inf(self, embedding):
        """Test that output contains no NaN or Inf values."""
        B, max_rows, max_cols = 4, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        _, col_mask = create_masks(B, max_rows, max_cols)
        
        output = embedding(tokens, col_mask)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# =============================================================================
# Test TransformerLayer
# =============================================================================

class TestTransformerLayer:
    """Tests for TransformerLayer."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return EncoderConfig(
            hidden_dim=64,
            n_heads=4,
            ff_dim=256,
            dropout=0.0,  # Disable dropout for deterministic tests
        )
    
    @pytest.fixture
    def layer(self, config):
        """Create TransformerLayer."""
        return TransformerLayer(config)
    
    def test_output_shape(self, layer):
        """Test that output shape matches input shape."""
        B, max_rows, max_cols, hidden_dim = 4, 32, 16, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        output = layer(x, attention_mask=col_mask)
        
        assert output.shape == x.shape
    
    def test_masking_affects_output(self, layer):
        """Test that different masks produce different outputs."""
        B, max_rows, max_cols, hidden_dim = 2, 32, 16, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        
        # Full mask
        col_mask_full = torch.ones(B, max_cols, dtype=torch.bool)
        output_full = layer(x, attention_mask=col_mask_full)
        
        # Partial mask
        col_mask_partial = torch.ones(B, max_cols, dtype=torch.bool)
        col_mask_partial[:, 8:] = False
        output_partial = layer(x, attention_mask=col_mask_partial)
        
        # Outputs should differ (at least in valid positions)
        assert not torch.allclose(output_full[:, :, :8, :], output_partial[:, :, :8, :])
    
    def test_no_nan_or_inf(self, layer):
        """Test that output contains no NaN or Inf values."""
        B, max_rows, max_cols, hidden_dim = 4, 32, 16, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        output = layer(x, attention_mask=col_mask)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_residual_connection(self, layer):
        """Test that residual connections are working (output not too different from input)."""
        B, max_rows, max_cols, hidden_dim = 2, 16, 8, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        output = layer(x, attention_mask=col_mask)
        
        # With residual connections, output shouldn't be wildly different from input
        # The relative change should be bounded
        relative_change = (output - x).abs().mean() / x.abs().mean()
        assert relative_change < 10.0  # Generous bound, just checking residual exists


# =============================================================================
# Test AttentionPooling
# =============================================================================

class TestAttentionPooling:
    """Tests for AttentionPooling layer."""
    
    @pytest.fixture
    def pooling(self):
        """Create AttentionPooling layer."""
        return AttentionPooling(hidden_dim=64, dropout=0.0)
    
    def test_output_shape(self, pooling):
        """Test output tensor shape."""
        B, seq_len, hidden_dim = 4, 16, 64
        x = torch.randn(B, seq_len, hidden_dim)
        mask = torch.ones(B, seq_len, dtype=torch.bool)
        
        output = pooling(x, mask)
        
        assert output.shape == (B, hidden_dim)
    
    def test_respects_mask(self, pooling):
        """Test that masked positions don't contribute to output."""
        B, seq_len, hidden_dim = 2, 16, 64
        x = torch.randn(B, seq_len, hidden_dim)
        
        # Mask out second half
        mask = torch.ones(B, seq_len, dtype=torch.bool)
        mask[:, seq_len//2:] = False
        
        # Modify masked positions
        x_modified = x.clone()
        x_modified[:, seq_len//2:, :] = 1000.0  # Large values
        
        output_original = pooling(x, mask)
        output_modified = pooling(x_modified, mask)
        
        # Outputs should be identical since masked positions are ignored
        assert torch.allclose(output_original, output_modified, atol=1e-5)
    
    def test_no_nan_or_inf(self, pooling):
        """Test that output contains no NaN or Inf values."""
        B, seq_len, hidden_dim = 4, 16, 64
        x = torch.randn(B, seq_len, hidden_dim)
        mask = torch.ones(B, seq_len, dtype=torch.bool)
        
        output = pooling(x, mask)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# =============================================================================
# Test RowPooling
# =============================================================================

class TestRowPooling:
    """Tests for RowPooling layer."""
    
    def test_mean_pooling(self):
        """Test mean pooling over features."""
        pooling = RowPooling(hidden_dim=64, method="mean", dropout=0.0)
        
        B, max_rows, max_cols, hidden_dim = 2, 16, 8, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        output = pooling(x, col_mask)
        
        assert output.shape == (B, max_rows, hidden_dim)
    
    def test_max_pooling(self):
        """Test max pooling over features."""
        pooling = RowPooling(hidden_dim=64, method="max", dropout=0.0)
        
        B, max_rows, max_cols, hidden_dim = 2, 16, 8, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        output = pooling(x, col_mask)
        
        assert output.shape == (B, max_rows, hidden_dim)
    
    def test_attention_pooling(self):
        """Test attention pooling over features."""
        pooling = RowPooling(hidden_dim=64, method="attention", dropout=0.0)
        
        B, max_rows, max_cols, hidden_dim = 2, 16, 8, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        output = pooling(x, col_mask)
        
        assert output.shape == (B, max_rows, hidden_dim)
    
    def test_respects_col_mask(self):
        """Test that column mask is respected in pooling."""
        pooling = RowPooling(hidden_dim=64, method="mean", dropout=0.0)
        
        B, max_rows, max_cols, hidden_dim = 2, 16, 8, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        
        # Only first 4 columns valid
        col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
        col_mask[:, :4] = True
        
        # Modify invalid columns
        x_modified = x.clone()
        x_modified[:, :, 4:, :] = 1000.0
        
        output_original = pooling(x, col_mask)
        output_modified = pooling(x_modified, col_mask)
        
        # Should be identical since invalid columns are masked
        assert torch.allclose(output_original, output_modified, atol=1e-5)


# =============================================================================
# Test DatasetPooling
# =============================================================================

class TestDatasetPooling:
    """Tests for DatasetPooling layer."""
    
    def test_mean_pooling(self):
        """Test mean pooling over rows."""
        pooling = DatasetPooling(hidden_dim=64, evidence_dim=32, method="mean", dropout=0.0)
        
        B, max_rows, hidden_dim = 4, 32, 64
        x = torch.randn(B, max_rows, hidden_dim)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        
        output = pooling(x, row_mask)
        
        assert output.shape == (B, 32)  # evidence_dim
    
    def test_max_pooling(self):
        """Test max pooling over rows."""
        pooling = DatasetPooling(hidden_dim=64, evidence_dim=32, method="max", dropout=0.0)
        
        B, max_rows, hidden_dim = 4, 32, 64
        x = torch.randn(B, max_rows, hidden_dim)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        
        output = pooling(x, row_mask)
        
        assert output.shape == (B, 32)
    
    def test_attention_pooling(self):
        """Test attention pooling over rows."""
        pooling = DatasetPooling(hidden_dim=64, evidence_dim=32, method="attention", dropout=0.0)
        
        B, max_rows, hidden_dim = 4, 32, 64
        x = torch.randn(B, max_rows, hidden_dim)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        
        output = pooling(x, row_mask)
        
        assert output.shape == (B, 32)
    
    def test_respects_row_mask(self):
        """Test that row mask is respected in pooling."""
        pooling = DatasetPooling(hidden_dim=64, evidence_dim=32, method="mean", dropout=0.0)
        
        B, max_rows, hidden_dim = 2, 32, 64
        x = torch.randn(B, max_rows, hidden_dim)
        
        # Only first 16 rows valid
        row_mask = torch.zeros(B, max_rows, dtype=torch.bool)
        row_mask[:, :16] = True
        
        # Modify invalid rows
        x_modified = x.clone()
        x_modified[:, 16:, :] = 1000.0
        
        output_original = pooling(x, row_mask)
        output_modified = pooling(x_modified, row_mask)
        
        # Should be identical since invalid rows are masked
        assert torch.allclose(output_original, output_modified, atol=1e-5)


# =============================================================================
# Test LacunaEncoder
# =============================================================================

class TestLacunaEncoder:
    """Tests for complete LacunaEncoder."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return EncoderConfig(
            hidden_dim=64,
            evidence_dim=32,
            n_layers=2,
            n_heads=4,
            max_cols=16,
            dropout=0.0,
        )
    
    @pytest.fixture
    def encoder(self, config):
        """Create LacunaEncoder."""
        return LacunaEncoder(config)
    
    def test_output_shape(self, encoder):
        """Test output evidence vector shape."""
        B, max_rows, max_cols = 4, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        row_mask, col_mask = create_masks(B, max_rows, max_cols)
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)  # evidence_dim
    
    def test_with_return_intermediates(self, encoder):
        """Test returning intermediate representations."""
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        row_mask, col_mask = create_masks(B, max_rows, max_cols)
        
        result = encoder(tokens, row_mask, col_mask, return_intermediates=True)
        
        assert isinstance(result, dict)
        assert "evidence" in result
        assert "row_representations" in result
        assert "token_representations" in result
        
        assert result["evidence"].shape == (B, 32)
        assert result["row_representations"].shape == (B, max_rows, 64)  # hidden_dim
        assert result["token_representations"].shape == (B, max_rows, max_cols, 64)
    
    def test_get_row_representations(self, encoder):
        """Test getting row-level representations."""
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        row_mask, col_mask = create_masks(B, max_rows, max_cols)
        
        row_repr = encoder.get_row_representations(tokens, row_mask, col_mask)
        
        assert row_repr.shape == (B, max_rows, 64)  # hidden_dim
    
    def test_get_token_representations(self, encoder):
        """Test getting token-level representations."""
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        row_mask, col_mask = create_masks(B, max_rows, max_cols)
        
        token_repr = encoder.get_token_representations(tokens, row_mask, col_mask)
        
        assert token_repr.shape == (B, max_rows, max_cols, 64)  # hidden_dim
    
    def test_no_nan_or_inf(self, encoder):
        """Test that output contains no NaN or Inf values."""
        B, max_rows, max_cols = 4, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        row_mask, col_mask = create_masks(B, max_rows, max_cols)
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert not torch.isnan(evidence).any()
        assert not torch.isinf(evidence).any()
    
    def test_handles_variable_masks(self, encoder):
        """Test handling datasets with different sizes."""
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        
        # Different sizes per sample
        row_mask = torch.zeros(B, max_rows, dtype=torch.bool)
        row_mask[0, :20] = True  # Sample 0: 20 rows
        row_mask[1, :28] = True  # Sample 1: 28 rows
        
        col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
        col_mask[0, :10] = True  # Sample 0: 10 cols
        col_mask[1, :14] = True  # Sample 1: 14 cols
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)
        assert not torch.isnan(evidence).any()
    
    def test_gradients_flow(self, encoder):
        """Test that gradients flow through the encoder parameters.
        
        Note: Gradients don't flow back to input tokens because:
        1. is_observed, mask_type, feature_id are converted to .long() for embedding lookup
        2. Only the value channel could receive gradients, but it's multiplied by 
           is_observed which breaks the gradient flow
        
        What matters is that model PARAMETERS receive gradients for training.
        """
        B, max_rows, max_cols = 2, 16, 8
        tokens = create_valid_tokens(B, max_rows, max_cols)
        row_mask, col_mask = create_masks(B, max_rows, max_cols)
        
        # Zero all parameter gradients
        encoder.zero_grad()
        
        evidence = encoder(tokens, row_mask, col_mask)
        loss = evidence.sum()
        loss.backward()
        
        # Check that gradients flow to model parameters
        total_grad = 0.0
        params_with_grad = 0
        for name, param in encoder.named_parameters():
            if param.grad is not None:
                grad_sum = param.grad.abs().sum().item()
                total_grad += grad_sum
                if grad_sum > 0:
                    params_with_grad += 1
        
        assert total_grad > 0, "No gradients flowed to any parameters"
        assert params_with_grad > 0, "No parameters received non-zero gradients"
    
    def test_different_batch_sizes(self, encoder):
        """Test encoder works with different batch sizes."""
        max_rows, max_cols = 32, 16
        
        for B in [1, 2, 4, 8]:
            tokens = create_valid_tokens(B, max_rows, max_cols)
            row_mask, col_mask = create_masks(B, max_rows, max_cols)
            
            evidence = encoder(tokens, row_mask, col_mask)
            assert evidence.shape == (B, 32)
    
    def test_eval_mode(self, encoder):
        """Test encoder behavior in eval mode (dropout disabled)."""
        encoder_with_dropout = LacunaEncoder(EncoderConfig(
            hidden_dim=64,
            evidence_dim=32,
            n_layers=2,
            max_cols=16,
            dropout=0.5,  # High dropout
        ))
        
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        row_mask, col_mask = create_masks(B, max_rows, max_cols)
        
        # In eval mode, output should be deterministic
        encoder_with_dropout.eval()
        output1 = encoder_with_dropout(tokens, row_mask, col_mask)
        output2 = encoder_with_dropout(tokens, row_mask, col_mask)
        
        assert torch.allclose(output1, output2)


# =============================================================================
# Test Factory Function
# =============================================================================

class TestCreateEncoder:
    """Tests for create_encoder factory function."""
    
    def test_default_parameters(self):
        """Test encoder creation with default parameters."""
        encoder = create_encoder()
        
        assert isinstance(encoder, LacunaEncoder)
        assert encoder.config.hidden_dim == 128
        assert encoder.config.evidence_dim == 64
        assert encoder.config.n_layers == 4
    
    def test_custom_parameters(self):
        """Test encoder creation with custom parameters."""
        encoder = create_encoder(
            hidden_dim=256,
            evidence_dim=128,
            n_layers=6,
            n_heads=8,
            max_cols=64,
        )
        
        assert encoder.config.hidden_dim == 256
        assert encoder.config.evidence_dim == 128
        assert encoder.config.n_layers == 6
        assert encoder.config.n_heads == 8
        assert encoder.config.max_cols == 64
    
    def test_forward_pass(self):
        """Test forward pass with created encoder."""
        encoder = create_encoder(
            hidden_dim=64,
            evidence_dim=32,
            n_layers=2,
            max_cols=16,
        )
        
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        row_mask, col_mask = create_masks(B, max_rows, max_cols)
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)


# =============================================================================
# Test Attention Pooling in Encoder
# =============================================================================

class TestEncoderAttentionPooling:
    """Tests for attention-based pooling in encoder."""
    
    @pytest.fixture
    def encoder_attention(self):
        """Create encoder with attention pooling."""
        return create_encoder(
            hidden_dim=64,
            evidence_dim=32,
            n_layers=2,
            max_cols=16,
            row_pooling="attention",
            dataset_pooling="attention",
            dropout=0.0,
        )
    
    @pytest.fixture
    def encoder_mean(self):
        """Create encoder with mean pooling."""
        return create_encoder(
            hidden_dim=64,
            evidence_dim=32,
            n_layers=2,
            max_cols=16,
            row_pooling="mean",
            dataset_pooling="mean",
            dropout=0.0,
        )
    
    def test_attention_row_pooling(self, encoder_attention):
        """Test that attention row pooling produces valid output."""
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        row_mask, col_mask = create_masks(B, max_rows, max_cols)
        
        evidence = encoder_attention(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)
        assert not torch.isnan(evidence).any()
    
    def test_attention_dataset_pooling(self, encoder_attention):
        """Test that attention dataset pooling produces valid output."""
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        row_mask, col_mask = create_masks(B, max_rows, max_cols)
        
        result = encoder_attention(tokens, row_mask, col_mask, return_intermediates=True)
        
        assert result["evidence"].shape == (B, 32)
        assert not torch.isnan(result["evidence"]).any()
    
    def test_full_attention_pooling(self, encoder_attention):
        """Test complete forward pass with attention pooling."""
        B, max_rows, max_cols = 4, 64, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        row_mask, col_mask = create_masks(B, max_rows, max_cols)
        
        evidence = encoder_attention(tokens, row_mask, col_mask)
        
        # Should produce valid evidence vectors
        assert evidence.shape == (B, 32)
        assert not torch.isnan(evidence).any()
        assert not torch.isinf(evidence).any()
        
        # Evidence vectors should have reasonable magnitude
        assert evidence.abs().max() < 100.0


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEncoderEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_row(self):
        """Test encoder with single-row datasets."""
        encoder = create_encoder(
            hidden_dim=64, evidence_dim=32, n_layers=2, max_cols=16, dropout=0.0
        )
        
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        
        # Only 1 valid row
        row_mask = torch.zeros(B, max_rows, dtype=torch.bool)
        row_mask[:, 0] = True
        _, col_mask = create_masks(B, max_rows, max_cols)
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)
        assert not torch.isnan(evidence).any()
    
    def test_single_column(self):
        """Test encoder with single-column datasets."""
        encoder = create_encoder(
            hidden_dim=64, evidence_dim=32, n_layers=2, max_cols=16, dropout=0.0
        )
        
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        
        # Only 1 valid column
        row_mask, _ = create_masks(B, max_rows, max_cols)
        col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
        col_mask[:, 0] = True
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)
        assert not torch.isnan(evidence).any()
    
    def test_all_missing_values(self):
        """Test encoder when all values in a dataset are missing."""
        encoder = create_encoder(
            hidden_dim=64, evidence_dim=32, n_layers=2, max_cols=16, dropout=0.0
        )
        
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        
        # Set all values as missing (is_observed = 0)
        tokens[..., IDX_OBSERVED] = 0.0
        
        row_mask, col_mask = create_masks(B, max_rows, max_cols)
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)
        assert not torch.isnan(evidence).any()
    
    def test_all_observed_values(self):
        """Test encoder when all values are observed."""
        encoder = create_encoder(
            hidden_dim=64, evidence_dim=32, n_layers=2, max_cols=16, dropout=0.0
        )
        
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        
        # Set all values as observed (is_observed = 1)
        tokens[..., IDX_OBSERVED] = 1.0
        
        row_mask, col_mask = create_masks(B, max_rows, max_cols)
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)
        assert not torch.isnan(evidence).any()
    
    def test_extreme_values(self):
        """Test encoder with extreme input values."""
        encoder = create_encoder(
            hidden_dim=64, evidence_dim=32, n_layers=2, max_cols=16, dropout=0.0
        )
        
        B, max_rows, max_cols = 2, 32, 16
        tokens = create_valid_tokens(B, max_rows, max_cols)
        
        # Set extreme but finite values
        tokens[..., IDX_VALUE] = tokens[..., IDX_VALUE] * 100
        
        row_mask, col_mask = create_masks(B, max_rows, max_cols)
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        # Should handle extreme values without producing NaN/Inf
        assert not torch.isnan(evidence).any()
        assert not torch.isinf(evidence).any()