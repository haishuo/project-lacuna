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
from lacuna.data.tokenization import TOKEN_DIM


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
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        output = embedding(tokens, col_mask)
        
        assert output.shape == (B, max_rows, max_cols, 64)  # hidden_dim=64
    
    def test_padding_is_zeroed(self, embedding):
        """Test that padding positions are zeroed out."""
        B, max_rows, max_cols = 2, 32, 16
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        
        # Only first 8 columns are valid
        col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
        col_mask[:, :8] = True
        
        output = embedding(tokens, col_mask)
        
        # Padding positions should be zero
        assert (output[:, :, 8:, :] == 0).all()
    
    def test_handles_variable_valid_cols(self, embedding):
        """Test handling of variable number of valid columns per sample."""
        B, max_rows, max_cols = 2, 32, 16
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        
        # Different number of valid columns per sample
        col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
        col_mask[0, :5] = True   # Sample 0: 5 valid columns
        col_mask[1, :10] = True  # Sample 1: 10 valid columns
        
        output = embedding(tokens, col_mask)
        
        # Check padding is zeroed correctly for each sample
        assert (output[0, :, 5:, :] == 0).all()
        assert (output[1, :, 10:, :] == 0).all()
        
        # Valid positions should have non-zero values (likely)
        assert output[0, :, :5, :].abs().sum() > 0
        assert output[1, :, :10, :].abs().sum() > 0
    
    def test_no_nan_or_inf(self, embedding):
        """Test that output contains no NaN or Inf values."""
        B, max_rows, max_cols = 4, 32, 16
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
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
            dropout=0.0,
            attention_dropout=0.0,
        )
    
    @pytest.fixture
    def layer(self, config):
        """Create TransformerLayer."""
        return TransformerLayer(config)
    
    def test_output_shape(self, layer):
        """Test output tensor shape."""
        B, max_rows, max_cols, hidden_dim = 4, 32, 16, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        
        output = layer(x)
        
        assert output.shape == x.shape
    
    def test_with_attention_mask(self, layer):
        """Test with attention mask."""
        B, max_rows, max_cols, hidden_dim = 4, 32, 16, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        
        # Create attention mask (only first 8 columns valid)
        attention_mask = torch.zeros(B, max_cols, dtype=torch.bool)
        attention_mask[:, :8] = True
        
        output = layer(x, attention_mask=attention_mask)
        
        assert output.shape == x.shape
    
    def test_residual_connection(self, layer):
        """Test that residual connections preserve input information."""
        B, max_rows, max_cols, hidden_dim = 2, 8, 8, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        
        # With residual connections, output should be correlated with input
        output = layer(x)
        
        # Compute correlation (not exact equality due to transformations)
        correlation = torch.corrcoef(
            torch.stack([x.flatten(), output.flatten()])
        )[0, 1]
        
        # Should have some positive correlation due to residual
        assert correlation > 0
    
    def test_no_nan_or_inf(self, layer):
        """Test that output contains no NaN or Inf values."""
        B, max_rows, max_cols, hidden_dim = 4, 32, 16, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        attention_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        output = layer(x, attention_mask=attention_mask)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_masked_positions_dont_affect_valid(self, layer):
        """Test that masked positions don't affect valid positions."""
        B, max_rows, max_cols, hidden_dim = 2, 8, 8, 64
        x1 = torch.randn(B, max_rows, max_cols, hidden_dim)
        x2 = x1.clone()
        
        # Modify padding positions in x2
        x2[:, :, 4:, :] = torch.randn(B, max_rows, 4, hidden_dim) * 100
        
        # Mask out those positions
        attention_mask = torch.zeros(B, max_cols, dtype=torch.bool)
        attention_mask[:, :4] = True
        
        output1 = layer(x1, attention_mask=attention_mask)
        output2 = layer(x2, attention_mask=attention_mask)
        
        # Valid positions should produce same output
        assert torch.allclose(output1[:, :, :4, :], output2[:, :, :4, :], atol=1e-5)


# =============================================================================
# Test AttentionPooling
# =============================================================================

class TestAttentionPooling:
    """Tests for AttentionPooling layer."""
    
    @pytest.fixture
    def pooler(self):
        """Create AttentionPooling layer."""
        return AttentionPooling(hidden_dim=64, dropout=0.0)
    
    def test_output_shape(self, pooler):
        """Test output tensor shape."""
        B, seq_len, hidden_dim = 4, 16, 64
        x = torch.randn(B, seq_len, hidden_dim)
        
        output = pooler(x)
        
        assert output.shape == (B, hidden_dim)
    
    def test_with_mask(self, pooler):
        """Test with validity mask."""
        B, seq_len, hidden_dim = 4, 16, 64
        x = torch.randn(B, seq_len, hidden_dim)
        
        # Only first 8 positions valid
        mask = torch.zeros(B, seq_len, dtype=torch.bool)
        mask[:, :8] = True
        
        output = pooler(x, mask=mask)
        
        assert output.shape == (B, hidden_dim)
    
    def test_masked_positions_ignored(self, pooler):
        """Test that masked positions are ignored."""
        B, seq_len, hidden_dim = 2, 16, 64
        x1 = torch.randn(B, seq_len, hidden_dim)
        x2 = x1.clone()
        
        # Modify masked positions in x2
        x2[:, 8:, :] = torch.randn(B, 8, hidden_dim) * 100
        
        # Mask out those positions
        mask = torch.zeros(B, seq_len, dtype=torch.bool)
        mask[:, :8] = True
        
        output1 = pooler(x1, mask=mask)
        output2 = pooler(x2, mask=mask)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-5)


# =============================================================================
# Test RowPooling
# =============================================================================

class TestRowPooling:
    """Tests for RowPooling layer."""
    
    @pytest.mark.parametrize("method", ["mean", "max", "attention"])
    def test_output_shape(self, method):
        """Test output tensor shape for different methods."""
        pooler = RowPooling(hidden_dim=64, method=method, dropout=0.0)
        
        B, max_rows, max_cols, hidden_dim = 4, 32, 16, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        output = pooler(x, col_mask)
        
        assert output.shape == (B, max_rows, hidden_dim)
    
    def test_mean_pooling(self):
        """Test mean pooling produces correct values."""
        pooler = RowPooling(hidden_dim=64, method="mean", dropout=0.0)
        
        B, max_rows, max_cols, hidden_dim = 1, 2, 4, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        output = pooler(x, col_mask)
        
        # Mean pooling should equal manual mean
        expected = x.mean(dim=2)
        assert torch.allclose(output, expected, atol=1e-5)
    
    def test_max_pooling(self):
        """Test max pooling produces correct values."""
        pooler = RowPooling(hidden_dim=64, method="max", dropout=0.0)
        
        B, max_rows, max_cols, hidden_dim = 1, 2, 4, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        output = pooler(x, col_mask)
        
        # Max pooling should equal manual max
        expected, _ = x.max(dim=2)
        assert torch.allclose(output, expected, atol=1e-5)
    
    def test_respects_col_mask(self):
        """Test that column mask is respected."""
        pooler = RowPooling(hidden_dim=64, method="mean", dropout=0.0)
        
        B, max_rows, max_cols, hidden_dim = 1, 2, 8, 64
        x = torch.randn(B, max_rows, max_cols, hidden_dim)
        
        # Only first 4 columns valid
        col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
        col_mask[:, :4] = True
        
        output = pooler(x, col_mask)
        
        # Should only average over first 4 columns
        expected = x[:, :, :4, :].mean(dim=2)
        assert torch.allclose(output, expected, atol=1e-5)
    
    def test_invalid_method_raises(self):
        """Test that invalid pooling method raises error."""
        with pytest.raises(ValueError):
            RowPooling(hidden_dim=64, method="invalid")


# =============================================================================
# Test DatasetPooling
# =============================================================================

class TestDatasetPooling:
    """Tests for DatasetPooling layer."""
    
    @pytest.mark.parametrize("method", ["mean", "max", "attention"])
    def test_output_shape(self, method):
        """Test output tensor shape for different methods."""
        pooler = DatasetPooling(
            hidden_dim=64,
            evidence_dim=32,
            method=method,
            dropout=0.0,
        )
        
        B, max_rows, hidden_dim = 4, 32, 64
        x = torch.randn(B, max_rows, hidden_dim)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        
        output = pooler(x, row_mask)
        
        assert output.shape == (B, 32)  # evidence_dim
    
    def test_respects_row_mask(self):
        """Test that row mask is respected."""
        pooler = DatasetPooling(
            hidden_dim=64,
            evidence_dim=32,
            method="mean",
            dropout=0.0,
        )
        
        B, max_rows, hidden_dim = 2, 32, 64
        x1 = torch.randn(B, max_rows, hidden_dim)
        x2 = x1.clone()
        
        # Modify masked rows in x2
        x2[:, 16:, :] = torch.randn(B, 16, hidden_dim) * 100
        
        # Mask out those rows
        row_mask = torch.zeros(B, max_rows, dtype=torch.bool)
        row_mask[:, :16] = True
        
        output1 = pooler(x1, row_mask)
        output2 = pooler(x2, row_mask)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-5)
    
    def test_projects_to_evidence_dim(self):
        """Test that output is projected to evidence_dim."""
        for evidence_dim in [16, 32, 64, 128]:
            pooler = DatasetPooling(
                hidden_dim=64,
                evidence_dim=evidence_dim,
                method="mean",
                dropout=0.0,
            )
            
            B, max_rows, hidden_dim = 4, 32, 64
            x = torch.randn(B, max_rows, hidden_dim)
            row_mask = torch.ones(B, max_rows, dtype=torch.bool)
            
            output = pooler(x, row_mask)
            
            assert output.shape == (B, evidence_dim)


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
            row_pooling="mean",
            dataset_pooling="mean",
        )
    
    @pytest.fixture
    def encoder(self, config):
        """Create LacunaEncoder."""
        return LacunaEncoder(config)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensors."""
        B, max_rows, max_cols = 4, 32, 16
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        return tokens, row_mask, col_mask
    
    def test_output_shape(self, encoder, sample_input):
        """Test output tensor shape."""
        tokens, row_mask, col_mask = sample_input
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (4, 32)  # B, evidence_dim
    
    def test_with_return_intermediates(self, encoder, sample_input):
        """Test returning intermediate representations."""
        tokens, row_mask, col_mask = sample_input
        
        output = encoder(tokens, row_mask, col_mask, return_intermediates=True)
        
        assert isinstance(output, dict)
        assert "evidence" in output
        assert "row_representations" in output
        assert "token_representations" in output
        
        assert output["evidence"].shape == (4, 32)
        assert output["row_representations"].shape == (4, 32, 64)  # B, max_rows, hidden_dim
        assert output["token_representations"].shape == (4, 32, 16, 64)
    
    def test_get_row_representations(self, encoder, sample_input):
        """Test get_row_representations method."""
        tokens, row_mask, col_mask = sample_input
        
        row_repr = encoder.get_row_representations(tokens, row_mask, col_mask)
        
        assert row_repr.shape == (4, 32, 64)  # B, max_rows, hidden_dim
    
    def test_get_token_representations(self, encoder, sample_input):
        """Test get_token_representations method."""
        tokens, row_mask, col_mask = sample_input
        
        token_repr = encoder.get_token_representations(tokens, row_mask, col_mask)
        
        assert token_repr.shape == (4, 32, 16, 64)  # B, max_rows, max_cols, hidden_dim
    
    def test_no_nan_or_inf(self, encoder, sample_input):
        """Test that output contains no NaN or Inf values."""
        tokens, row_mask, col_mask = sample_input
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert not torch.isnan(evidence).any()
        assert not torch.isinf(evidence).any()
    
    def test_handles_variable_masks(self, encoder):
        """Test handling of variable row and column masks."""
        B, max_rows, max_cols = 4, 32, 16
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        
        # Variable number of valid rows and columns
        row_mask = torch.zeros(B, max_rows, dtype=torch.bool)
        row_mask[0, :10] = True
        row_mask[1, :20] = True
        row_mask[2, :15] = True
        row_mask[3, :25] = True
        
        col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
        col_mask[0, :5] = True
        col_mask[1, :10] = True
        col_mask[2, :8] = True
        col_mask[3, :12] = True
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)
        assert not torch.isnan(evidence).any()
    
    def test_gradients_flow(self, encoder, sample_input):
        """Test that gradients flow through the encoder."""
        tokens, row_mask, col_mask = sample_input
        tokens.requires_grad_(True)
        
        evidence = encoder(tokens, row_mask, col_mask)
        loss = evidence.sum()
        loss.backward()
        
        assert tokens.grad is not None
        assert tokens.grad.shape == tokens.shape
    
    def test_different_batch_sizes(self, encoder):
        """Test with different batch sizes."""
        max_rows, max_cols = 32, 16
        
        for B in [1, 2, 4, 8]:
            tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
            row_mask = torch.ones(B, max_rows, dtype=torch.bool)
            col_mask = torch.ones(B, max_cols, dtype=torch.bool)
            
            evidence = encoder(tokens, row_mask, col_mask)
            
            assert evidence.shape == (B, 32)
    
    def test_eval_mode(self, encoder, sample_input):
        """Test encoder in eval mode."""
        tokens, row_mask, col_mask = sample_input
        
        encoder.eval()
        with torch.no_grad():
            evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (4, 32)
        assert not torch.isnan(evidence).any()


# =============================================================================
# Test create_encoder Factory
# =============================================================================

class TestCreateEncoder:
    """Tests for create_encoder factory function."""
    
    def test_default_configuration(self):
        """Test creating encoder with default configuration."""
        encoder = create_encoder()
        
        assert isinstance(encoder, LacunaEncoder)
        assert encoder.config.hidden_dim == 128
        assert encoder.config.evidence_dim == 64
    
    def test_custom_configuration(self):
        """Test creating encoder with custom configuration."""
        encoder = create_encoder(
            hidden_dim=256,
            evidence_dim=128,
            n_layers=6,
            n_heads=8,
            max_cols=64,
            dropout=0.2,
            row_pooling="max",
            dataset_pooling="attention",
        )
        
        assert encoder.config.hidden_dim == 256
        assert encoder.config.evidence_dim == 128
        assert encoder.config.n_layers == 6
        assert encoder.config.n_heads == 8
        assert encoder.config.max_cols == 64
        assert encoder.config.dropout == 0.2
        assert encoder.config.row_pooling == "max"
        assert encoder.config.dataset_pooling == "attention"
    
    def test_forward_pass(self):
        """Test forward pass with factory-created encoder."""
        encoder = create_encoder(
            hidden_dim=64,
            evidence_dim=32,
            n_layers=2,
            n_heads=4,
            max_cols=16,
        )
        
        B, max_rows, max_cols = 4, 32, 16
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)


# =============================================================================
# Test Encoder with Attention Pooling
# =============================================================================

class TestEncoderAttentionPooling:
    """Tests for encoder with attention-based pooling."""
    
    def test_attention_row_pooling(self):
        """Test encoder with attention row pooling."""
        encoder = create_encoder(
            hidden_dim=64,
            evidence_dim=32,
            n_layers=2,
            n_heads=4,
            max_cols=16,
            row_pooling="attention",
            dataset_pooling="mean",
        )
        
        B, max_rows, max_cols = 4, 32, 16
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)
    
    def test_attention_dataset_pooling(self):
        """Test encoder with attention dataset pooling."""
        encoder = create_encoder(
            hidden_dim=64,
            evidence_dim=32,
            n_layers=2,
            n_heads=4,
            max_cols=16,
            row_pooling="mean",
            dataset_pooling="attention",
        )
        
        B, max_rows, max_cols = 4, 32, 16
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)
    
    def test_full_attention_pooling(self):
        """Test encoder with attention pooling at both levels."""
        encoder = create_encoder(
            hidden_dim=64,
            evidence_dim=32,
            n_layers=2,
            n_heads=4,
            max_cols=16,
            row_pooling="attention",
            dataset_pooling="attention",
        )
        
        B, max_rows, max_cols = 4, 32, 16
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        evidence = encoder(tokens, row_mask, col_mask)
        
        assert evidence.shape == (B, 32)
        assert not torch.isnan(evidence).any()