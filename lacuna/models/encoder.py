"""
lacuna.models.encoder

Transformer evidence encoder for row-level token sequences.

Architecture:
- Input: [B, max_rows, max_cols, token_dim] row-level tokens
- Process each row as a sequence of column tokens
- Aggregate row representations to dataset representation
- Output: [B, evidence_dim] evidence embedding

Two-level attention:
1. Column-level: within each row, attend across columns
2. Row-level: aggregate row representations

This captures:
- Within-row dependencies (which columns predict which missingness)
- Across-row patterns (consistency of missingness mechanism)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RowEncoder(nn.Module):
    """Encode a single row (sequence of column tokens).
    
    Uses transformer to capture cross-column dependencies within a row.
    """
    
    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_cols: int = 64,
    ):
        super().__init__()
        
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        
        # CLS token for row representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Positional encoding for columns
        self.pos_embed = nn.Parameter(torch.randn(1, max_cols + 1, hidden_dim))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(
        self,
        row_tokens: torch.Tensor,  # [B, d, token_dim]
        col_mask: torch.Tensor,    # [B, d] bool, True=valid
    ) -> torch.Tensor:
        """Encode rows to fixed-size representations.
        
        Returns: [B, hidden_dim] row representations.
        """
        B, d, _ = row_tokens.shape
        
        # Project tokens
        x = self.input_proj(row_tokens)  # [B, d, h]
        
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, d+1, h]
        
        # Add positional encoding
        x = x + self.pos_embed[:, :d+1, :]
        
        # Attention mask (True = ignore)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        attn_mask = torch.cat([cls_mask, ~col_mask], dim=1)
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Extract CLS representation
        return x[:, 0, :]  # [B, h]


class EvidenceEncoder(nn.Module):
    """Full evidence encoder: dataset tokens -> evidence embedding.
    
    Two-stage architecture:
    1. RowEncoder: encode each row independently
    2. Row aggregation: combine row representations
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
        max_rows: int = 256,
        row_agg: str = "attention",
    ):
        super().__init__()
        
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.evidence_dim = evidence_dim
        self.row_agg = row_agg
        
        # Row encoder (processes each row)
        self.row_encoder = RowEncoder(
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers // 2,  # Split layers between row and aggregation
            n_heads=n_heads,
            dropout=dropout,
            max_cols=max_cols,
        )
        
        # Row aggregation
        if row_agg == "attention":
            # Attention-based aggregation over rows
            self.row_agg_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.row_agg_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
        elif row_agg == "transformer":
            # Full transformer over row representations
            agg_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.row_transformer = nn.TransformerEncoder(
                agg_layer,
                num_layers=n_layers // 2,
            )
            self.row_cls = nn.Parameter(torch.randn(1, 1, hidden_dim))
        else:
            # Simple mean pooling
            pass
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, evidence_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        if self.row_agg == "attention":
            nn.init.trunc_normal_(self.row_agg_query, std=0.02)
        elif self.row_agg == "transformer":
            nn.init.trunc_normal_(self.row_cls, std=0.02)
    
    def forward(
        self,
        tokens: torch.Tensor,     # [B, max_rows, max_cols, token_dim]
        row_mask: torch.Tensor,   # [B, max_rows] bool
        col_mask: torch.Tensor,   # [B, max_cols] bool
    ) -> torch.Tensor:
        """Encode dataset to evidence embedding.
        
        Returns: [B, evidence_dim]
        """
        B, max_rows, max_cols, _ = tokens.shape
        
        # Encode each row
        # Reshape to process all rows at once: [B * max_rows, max_cols, token_dim]
        tokens_flat = tokens.view(B * max_rows, max_cols, -1)
        col_mask_flat = col_mask.unsqueeze(1).expand(-1, max_rows, -1).reshape(B * max_rows, max_cols)
        
        row_reps = self.row_encoder(tokens_flat, col_mask_flat)  # [B * max_rows, h]
        row_reps = row_reps.view(B, max_rows, -1)  # [B, max_rows, h]
        
        # Aggregate rows
        if self.row_agg == "attention":
            # Query-based attention pooling
            query = self.row_agg_query.expand(B, -1, -1)  # [B, 1, h]
            
            # Create key padding mask
            key_padding_mask = ~row_mask  # [B, max_rows]
            
            agg_out, _ = self.row_agg_attn(
                query, row_reps, row_reps,
                key_padding_mask=key_padding_mask,
            )
            dataset_rep = agg_out.squeeze(1)  # [B, h]
            
        elif self.row_agg == "transformer":
            # Prepend CLS token
            cls = self.row_cls.expand(B, -1, -1)
            row_reps = torch.cat([cls, row_reps], dim=1)  # [B, max_rows+1, h]
            
            # Attention mask
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
            attn_mask = torch.cat([cls_mask, ~row_mask], dim=1)
            
            out = self.row_transformer(row_reps, src_key_padding_mask=attn_mask)
            dataset_rep = out[:, 0, :]  # [B, h]
            
        else:
            # Mean pooling over valid rows
            row_mask_expanded = row_mask.unsqueeze(-1).float()  # [B, max_rows, 1]
            dataset_rep = (row_reps * row_mask_expanded).sum(dim=1)
            dataset_rep = dataset_rep / row_mask_expanded.sum(dim=1).clamp(min=1)
        
        # Output projection
        evidence = self.output_proj(self.output_norm(dataset_rep))
        
        return evidence
