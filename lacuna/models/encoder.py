"""
lacuna.models.encoder

Transformer evidence encoder: column tokens -> evidence embedding.

Architecture:
- Input projection: token_dim -> hidden_dim
- Learnable CLS token prepended
- Positional encoding (learnable)
- Transformer encoder layers
- Output projection: hidden_dim -> evidence_dim
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class EvidenceEncoder(nn.Module):
    """Transformer encoder for evidence extraction.
    
    Takes column tokens [B, d, q] and produces evidence embedding [B, p].
    Uses CLS token pooling (like BERT).
    
    Args:
        token_dim: Input feature dimension per column (q).
        hidden_dim: Transformer hidden dimension (h).
        evidence_dim: Output evidence embedding dimension (p).
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        max_cols: Maximum number of columns (for positional encoding).
    """
    
    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        evidence_dim: int,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_cols: int = 64,
    ):
        super().__init__()
        
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.evidence_dim = evidence_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_cols = max_cols
        
        # Input projection
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Positional encoding (learnable)
        # +1 for CLS token
        self.pos_embed = nn.Parameter(torch.randn(1, max_cols + 1, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,  # For compatibility
        )
        
        # Layer norm before output
        self.pre_output_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, evidence_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # CLS and positional embeddings
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Linear layers
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        tokens: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            tokens: [B, d, q] column token features.
            col_mask: [B, d] bool tensor. True = valid column, False = padding.
        
        Returns:
            [B, p] evidence embedding.
        """
        B, d, q = tokens.shape
        
        # Project tokens to hidden dimension
        x = self.input_proj(tokens)  # [B, d, h]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, h]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, d+1, h]
        
        # Add positional encoding
        x = x + self.pos_embed[:, :d+1, :]
        
        # Create attention mask
        # True in src_key_padding_mask means IGNORE that position
        # CLS token is always attended (False = don't ignore)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
        # Padding columns should be ignored (True = ignore)
        pad_mask = ~col_mask  # [B, d]
        attn_mask = torch.cat([cls_mask, pad_mask], dim=1)  # [B, d+1]
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attn_mask)  # [B, d+1, h]
        
        # Extract CLS token representation
        cls_output = x[:, 0, :]  # [B, h]
        
        # Normalize and project to evidence dimension
        cls_output = self.pre_output_norm(cls_output)
        evidence = self.output_proj(cls_output)  # [B, p]
        
        return evidence
    
    def get_attention_weights(
        self,
        tokens: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get attention weights for interpretability.
        
        Note: This is a simplified version that only gets the last layer's
        attention. For full interpretability, would need hooks.
        
        Returns:
            [B, n_heads, d+1, d+1] attention weights from last layer.
        """
        # This would require registering hooks on the transformer
        # For now, just return placeholder
        raise NotImplementedError("Attention extraction not yet implemented")
