"""
lacuna.models.encoder

BERT-inspired transformer encoder for missing data mechanism classification.

Architecture Overview:
    The encoder processes each row of a dataset as a sequence of "feature tokens".
    Self-attention allows each feature to attend to all other features in the same row,
    learning cross-column dependencies that distinguish missingness mechanisms:
    
    - MCAR: Missingness is random scatter, no learnable cross-column patterns
    - MAR: Missingness in column j depends on observed values in other columns
           → Attention should capture these dependencies
    - MNAR: Missingness depends on the (unobserved) value itself
           → Systematic within-column patterns, less cross-column signal

Token Flow:
    Input: [B, max_rows, max_cols, token_dim=4]
           token = [value, is_observed, mask_type, feature_id_normalized]
    
    1. Token Embedding: Linear projection to hidden_dim
       [B, max_rows, max_cols, 4] -> [B, max_rows, max_cols, hidden_dim]
    
    2. Positional Encoding: Add learnable feature position embeddings
       (The normalized feature_id in tokens is supplemented with learned embeddings)
    
    3. Transformer Layers: Self-attention over features within each row
       Each row is treated as a separate sequence
       Attention mask excludes padding columns
    
    4. Row Pooling: Aggregate feature representations into row representation
       [B, max_rows, max_cols, hidden_dim] -> [B, max_rows, hidden_dim]
    
    5. Dataset Pooling: Aggregate row representations into dataset representation
       [B, max_rows, hidden_dim] -> [B, evidence_dim]
    
    Output: evidence vector z ∈ ℝ^{evidence_dim} summarizing the dataset

Design Decisions:
    1. Row-wise attention (not full dataset attention) for memory efficiency
    2. Learnable position embeddings supplement the normalized feature IDs
    3. Two-stage pooling (features→rows→dataset) preserves hierarchical structure
    4. LayerNorm and dropout for stability and regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

from lacuna.data.tokenization import TOKEN_DIM, IDX_VALUE, IDX_OBSERVED, IDX_MASK_TYPE, IDX_FEATURE_ID


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EncoderConfig:
    """Configuration for the transformer encoder."""
    
    # Architecture
    hidden_dim: int = 128          # Transformer hidden dimension
    evidence_dim: int = 64         # Final evidence vector dimension
    n_layers: int = 4              # Number of transformer layers
    n_heads: int = 4               # Number of attention heads
    ff_dim: Optional[int] = None   # Feedforward dimension (default: 4 * hidden_dim)
    
    # Regularization
    dropout: float = 0.1           # Dropout probability
    attention_dropout: float = 0.1 # Dropout in attention weights
    
    # Pooling
    row_pooling: str = "attention" # "mean", "max", or "attention"
    dataset_pooling: str = "attention"  # "mean", "max", or "attention"
    
    # Input
    max_cols: int = 32             # Maximum number of columns (for position embeddings)
    
    def __post_init__(self):
        if self.ff_dim is None:
            self.ff_dim = 4 * self.hidden_dim
        
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})"
            )


# =============================================================================
# Token Embedding Layer
# =============================================================================

class TokenEmbedding(nn.Module):
    """
    Embeds raw tokens into hidden dimension.
    
    Input tokens have structure: [value, is_observed, mask_type, feature_id_normalized]
    
    We process these components separately and combine:
    - value: Linear projection (scaled by observation indicator)
    - is_observed: Embedding lookup (2 classes: observed/missing)
    - mask_type: Embedding lookup (2 classes: natural/artificial)
    - feature_id: Learnable position embedding + linear projection of normalized ID
    """
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        
        self.hidden_dim = config.hidden_dim
        self.max_cols = config.max_cols
        
        # Value projection: scalar -> hidden_dim
        # We'll combine value with observation status
        self.value_proj = nn.Linear(1, config.hidden_dim // 4)
        
        # Observation status embedding: 2 states (missing=0, observed=1)
        self.obs_embedding = nn.Embedding(2, config.hidden_dim // 4)
        
        # Mask type embedding: 2 states (natural=0, artificial=1)
        self.mask_embedding = nn.Embedding(2, config.hidden_dim // 4)
        
        # Feature position embedding: learnable per-position embedding
        self.position_embedding = nn.Embedding(config.max_cols, config.hidden_dim // 4)
        
        # Final projection to combine all components
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        tokens: torch.Tensor,    # [B, max_rows, max_cols, TOKEN_DIM]
        col_mask: torch.Tensor,  # [B, max_cols]
    ) -> torch.Tensor:
        """
        Embed tokens into hidden dimension.
        
        Args:
            tokens: Raw token tensor with [value, is_observed, mask_type, feature_id_norm]
            col_mask: Boolean mask indicating valid columns (True = valid)
        
        Returns:
            embedded: [B, max_rows, max_cols, hidden_dim] embedded tokens
        """
        B, max_rows, max_cols, _ = tokens.shape
        device = tokens.device
        
        # Extract token components
        values = tokens[..., IDX_VALUE:IDX_VALUE+1]  # [B, max_rows, max_cols, 1]
        is_observed = tokens[..., IDX_OBSERVED]       # [B, max_rows, max_cols]
        mask_type = tokens[..., IDX_MASK_TYPE]        # [B, max_rows, max_cols]
        feature_id_norm = tokens[..., IDX_FEATURE_ID] # [B, max_rows, max_cols]
        
        # 1. Value embedding
        # Scale value by observation indicator (missing values contribute less)
        value_scaled = values * is_observed.unsqueeze(-1)
        value_emb = self.value_proj(value_scaled)  # [B, max_rows, max_cols, hidden_dim//4]
        
        # 2. Observation status embedding
        obs_idx = is_observed.long()  # Convert to indices: 0 or 1
        obs_emb = self.obs_embedding(obs_idx)  # [B, max_rows, max_cols, hidden_dim//4]
        
        # 3. Mask type embedding
        mask_idx = mask_type.long()  # Convert to indices: 0 or 1
        mask_emb = self.mask_embedding(mask_idx)  # [B, max_rows, max_cols, hidden_dim//4]
        
        # 4. Position embedding
        # Use integer feature indices for embedding lookup
        feature_indices = (feature_id_norm * (self.max_cols - 1)).long()
        feature_indices = feature_indices.clamp(0, self.max_cols - 1)
        pos_emb = self.position_embedding(feature_indices)  # [B, max_rows, max_cols, hidden_dim//4]
        
        # 5. Concatenate all components
        combined = torch.cat([value_emb, obs_emb, mask_emb, pos_emb], dim=-1)
        # combined: [B, max_rows, max_cols, hidden_dim]
        
        # 6. Project and normalize
        embedded = self.output_proj(combined)
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)
        
        # 7. Zero out padding columns
        # col_mask: [B, max_cols] -> [B, 1, max_cols, 1]
        col_mask_expanded = col_mask.unsqueeze(1).unsqueeze(-1)
        embedded = embedded * col_mask_expanded.float()
        
        return embedded


# =============================================================================
# Transformer Layer
# =============================================================================

class TransformerLayer(nn.Module):
    """
    Single transformer layer with multi-head self-attention and feedforward.
    
    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> Dropout -> + x
                                                          |
        x -> LayerNorm -> FeedForward -> Dropout -------> + x
    
    Uses pre-norm architecture (LayerNorm before attention/FFN) for stability.
    """
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_dim // config.n_heads
        
        # Multi-head self-attention
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.hidden_dim),
        )
        
        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.dropout = nn.Dropout(config.dropout)
        
        # Scaling factor for attention
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,                    # [B, max_rows, max_cols, hidden_dim]
        attention_mask: Optional[torch.Tensor] = None,  # [B, max_cols] or [B, max_rows, max_cols]
    ) -> torch.Tensor:
        """
        Apply transformer layer with row-wise attention.
        
        Each row is treated as a separate sequence. Attention is computed
        independently for each row, allowing parallelization.
        
        Args:
            x: Input tensor [B, max_rows, max_cols, hidden_dim]
            attention_mask: Boolean mask for valid columns [B, max_cols]
                           True = valid (attend to), False = padding (ignore)
        
        Returns:
            Output tensor [B, max_rows, max_cols, hidden_dim]
        """
        B, max_rows, max_cols, hidden_dim = x.shape
        
        # Reshape for row-wise attention: treat each row as a separate batch item
        # [B, max_rows, max_cols, hidden_dim] -> [B * max_rows, max_cols, hidden_dim]
        x_flat = x.view(B * max_rows, max_cols, hidden_dim)
        
        # Expand attention mask to match flattened batch
        if attention_mask is not None:
            # [B, max_cols] -> [B * max_rows, max_cols]
            attn_mask_flat = attention_mask.unsqueeze(1).expand(B, max_rows, max_cols)
            attn_mask_flat = attn_mask_flat.reshape(B * max_rows, max_cols)
        else:
            attn_mask_flat = None
        
        # === Self-Attention ===
        residual = x_flat
        x_norm = self.norm1(x_flat)
        
        # Project to Q, K, V
        Q = self.q_proj(x_norm)  # [B*max_rows, max_cols, hidden_dim]
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)
        
        # Reshape for multi-head attention
        # [B*max_rows, max_cols, hidden_dim] -> [B*max_rows, n_heads, max_cols, head_dim]
        Q = Q.view(B * max_rows, max_cols, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B * max_rows, max_cols, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B * max_rows, max_cols, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # [B*max_rows, n_heads, max_cols, head_dim] @ [B*max_rows, n_heads, head_dim, max_cols]
        # -> [B*max_rows, n_heads, max_cols, max_cols]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask
        if attn_mask_flat is not None:
            # Create causal-style mask: [B*max_rows, 1, 1, max_cols]
            # Positions where mask is False get -inf
            mask_value = torch.finfo(attn_scores.dtype).min
            attn_mask_expanded = attn_mask_flat.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~attn_mask_expanded, mask_value)
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # [B*max_rows, n_heads, max_cols, max_cols] @ [B*max_rows, n_heads, max_cols, head_dim]
        # -> [B*max_rows, n_heads, max_cols, head_dim]
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        # [B*max_rows, n_heads, max_cols, head_dim] -> [B*max_rows, max_cols, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B * max_rows, max_cols, hidden_dim)
        
        # Project and residual
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        x_flat = residual + attn_output
        
        # === Feedforward ===
        residual = x_flat
        x_norm = self.norm2(x_flat)
        ff_output = self.ff(x_norm)
        ff_output = self.dropout(ff_output)
        x_flat = residual + ff_output
        
        # Reshape back to original shape
        # [B * max_rows, max_cols, hidden_dim] -> [B, max_rows, max_cols, hidden_dim]
        output = x_flat.view(B, max_rows, max_cols, hidden_dim)
        
        return output


# =============================================================================
# Pooling Layers
# =============================================================================

class AttentionPooling(nn.Module):
    """
    Attention-based pooling over a sequence dimension.
    
    Learns to weight different positions based on their content,
    producing a weighted average as the pooled representation.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Attention scorer: hidden_dim -> 1
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,                    # [B, seq_len, hidden_dim]
        mask: Optional[torch.Tensor] = None, # [B, seq_len] True = valid
    ) -> torch.Tensor:
        """
        Pool sequence into single vector using learned attention.
        
        Args:
            x: Input tensor [B, seq_len, hidden_dim]
            mask: Boolean mask [B, seq_len] where True = valid position
        
        Returns:
            pooled: [B, hidden_dim] pooled representation
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # [B, seq_len]
        
        # Apply mask
        if mask is not None:
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, mask_value)
        
        # Softmax to get weights
        weights = F.softmax(scores, dim=-1)  # [B, seq_len]
        weights = self.dropout(weights)
        
        # Weighted average
        pooled = torch.einsum("bs,bsd->bd", weights, x)  # [B, hidden_dim]
        
        return pooled


class RowPooling(nn.Module):
    """
    Pool feature representations within each row.
    
    Takes [B, max_rows, max_cols, hidden_dim] and produces [B, max_rows, hidden_dim].
    """
    
    def __init__(self, hidden_dim: int, method: str = "attention", dropout: float = 0.1):
        super().__init__()
        
        self.method = method
        
        if method == "attention":
            self.pooler = AttentionPooling(hidden_dim, dropout)
        elif method not in ("mean", "max"):
            raise ValueError(f"Unknown pooling method: {method}")
    
    def forward(
        self,
        x: torch.Tensor,      # [B, max_rows, max_cols, hidden_dim]
        col_mask: torch.Tensor,  # [B, max_cols]
    ) -> torch.Tensor:
        """
        Pool features within each row.
        
        Args:
            x: Input tensor [B, max_rows, max_cols, hidden_dim]
            col_mask: Boolean mask [B, max_cols] for valid columns
        
        Returns:
            pooled: [B, max_rows, hidden_dim] row representations
        """
        B, max_rows, max_cols, hidden_dim = x.shape
        
        if self.method == "mean":
            # Masked mean pooling
            col_mask_expanded = col_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, max_cols, 1]
            x_masked = x * col_mask_expanded.float()
            sum_x = x_masked.sum(dim=2)  # [B, max_rows, hidden_dim]
            count = col_mask.float().sum(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
            count = count.clamp(min=1.0)
            pooled = sum_x / count
            
        elif self.method == "max":
            # Masked max pooling
            col_mask_expanded = col_mask.unsqueeze(1).unsqueeze(-1)
            x_masked = x.masked_fill(~col_mask_expanded, float('-inf'))
            pooled, _ = x_masked.max(dim=2)  # [B, max_rows, hidden_dim]
            
        elif self.method == "attention":
            # Attention pooling for each row
            # Reshape: [B, max_rows, max_cols, hidden_dim] -> [B*max_rows, max_cols, hidden_dim]
            x_flat = x.view(B * max_rows, max_cols, hidden_dim)
            
            # Expand mask: [B, max_cols] -> [B*max_rows, max_cols]
            col_mask_flat = col_mask.unsqueeze(1).expand(B, max_rows, max_cols)
            col_mask_flat = col_mask_flat.reshape(B * max_rows, max_cols)
            
            # Pool
            pooled_flat = self.pooler(x_flat, col_mask_flat)  # [B*max_rows, hidden_dim]
            
            # Reshape back: [B*max_rows, hidden_dim] -> [B, max_rows, hidden_dim]
            pooled = pooled_flat.view(B, max_rows, hidden_dim)
        
        return pooled


class DatasetPooling(nn.Module):
    """
    Pool row representations into dataset representation.
    
    Takes [B, max_rows, hidden_dim] and produces [B, evidence_dim].
    """
    
    def __init__(
        self,
        hidden_dim: int,
        evidence_dim: int,
        method: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.method = method
        
        if method == "attention":
            self.pooler = AttentionPooling(hidden_dim, dropout)
        elif method not in ("mean", "max"):
            raise ValueError(f"Unknown pooling method: {method}")
        
        # Project to evidence dimension
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, evidence_dim),
            nn.LayerNorm(evidence_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,       # [B, max_rows, hidden_dim]
        row_mask: torch.Tensor,  # [B, max_rows]
    ) -> torch.Tensor:
        """
        Pool rows into dataset representation.
        
        Args:
            x: Input tensor [B, max_rows, hidden_dim]
            row_mask: Boolean mask [B, max_rows] for valid rows
        
        Returns:
            evidence: [B, evidence_dim] dataset representation
        """
        B, max_rows, hidden_dim = x.shape
        
        if self.method == "mean":
            # Masked mean pooling
            row_mask_expanded = row_mask.unsqueeze(-1)  # [B, max_rows, 1]
            x_masked = x * row_mask_expanded.float()
            sum_x = x_masked.sum(dim=1)  # [B, hidden_dim]
            count = row_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
            pooled = sum_x / count
            
        elif self.method == "max":
            # Masked max pooling
            row_mask_expanded = row_mask.unsqueeze(-1)
            x_masked = x.masked_fill(~row_mask_expanded, float('-inf'))
            pooled, _ = x_masked.max(dim=1)  # [B, hidden_dim]
            
        elif self.method == "attention":
            pooled = self.pooler(x, row_mask)  # [B, hidden_dim]
        
        # Project to evidence dimension
        evidence = self.project(pooled)  # [B, evidence_dim]
        
        return evidence


# =============================================================================
# Main Encoder
# =============================================================================

class LacunaEncoder(nn.Module):
    """
    Complete transformer encoder for missing data mechanism classification.
    
    Architecture:
        tokens [B, max_rows, max_cols, 4]
            ↓
        TokenEmbedding
            ↓
        [B, max_rows, max_cols, hidden_dim]
            ↓
        TransformerLayer × n_layers (row-wise attention)
            ↓
        [B, max_rows, max_cols, hidden_dim]
            ↓
        RowPooling (features → row representation)
            ↓
        [B, max_rows, hidden_dim]
            ↓
        DatasetPooling (rows → dataset representation)
            ↓
        [B, evidence_dim]
    
    The output "evidence" vector summarizes the entire dataset's
    structure and missingness patterns for downstream classification.
    """
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        
        self.config = config
        
        # Token embedding layer
        self.token_embedding = TokenEmbedding(config)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.n_layers)
        ])
        
        # Pooling layers
        self.row_pooling = RowPooling(
            config.hidden_dim,
            method=config.row_pooling,
            dropout=config.dropout,
        )
        
        self.dataset_pooling = DatasetPooling(
            config.hidden_dim,
            config.evidence_dim,
            method=config.dataset_pooling,
            dropout=config.dropout,
        )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.evidence_dim)
    
    def forward(
        self,
        tokens: torch.Tensor,     # [B, max_rows, max_cols, TOKEN_DIM]
        row_mask: torch.Tensor,   # [B, max_rows]
        col_mask: torch.Tensor,   # [B, max_cols]
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Encode a batch of datasets into evidence vectors.
        
        Args:
            tokens: Token tensor [B, max_rows, max_cols, TOKEN_DIM]
            row_mask: Boolean mask for valid rows [B, max_rows]
            col_mask: Boolean mask for valid columns [B, max_cols]
            return_intermediates: If True, return dict with intermediate representations
        
        Returns:
            evidence: [B, evidence_dim] evidence vectors
            
            Or if return_intermediates=True:
            dict with keys:
                - "evidence": [B, evidence_dim]
                - "row_representations": [B, max_rows, hidden_dim]
                - "token_representations": [B, max_rows, max_cols, hidden_dim]
        """
        # 1. Embed tokens
        x = self.token_embedding(tokens, col_mask)  # [B, max_rows, max_cols, hidden_dim]
        
        # 2. Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=col_mask)
        
        # Store for intermediate output if needed
        token_representations = x
        
        # 3. Pool features within rows
        row_repr = self.row_pooling(x, col_mask)  # [B, max_rows, hidden_dim]
        
        # Zero out invalid rows
        row_mask_expanded = row_mask.unsqueeze(-1)  # [B, max_rows, 1]
        row_repr = row_repr * row_mask_expanded.float()
        
        # 4. Pool rows into dataset representation
        evidence = self.dataset_pooling(row_repr, row_mask)  # [B, evidence_dim]
        
        # 5. Final normalization
        evidence = self.final_norm(evidence)
        
        if return_intermediates:
            return {
                "evidence": evidence,
                "row_representations": row_repr,
                "token_representations": token_representations,
            }
        
        return evidence
    
    def get_row_representations(
        self,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get row-level representations (before dataset pooling).
        
        Useful for row-level gating in the MoE layer.
        
        Args:
            tokens: Token tensor [B, max_rows, max_cols, TOKEN_DIM]
            row_mask: Boolean mask for valid rows [B, max_rows]
            col_mask: Boolean mask for valid columns [B, max_cols]
        
        Returns:
            row_repr: [B, max_rows, hidden_dim] row representations
        """
        # Embed tokens
        x = self.token_embedding(tokens, col_mask)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=col_mask)
        
        # Pool features within rows
        row_repr = self.row_pooling(x, col_mask)
        
        # Zero out invalid rows
        row_mask_expanded = row_mask.unsqueeze(-1)
        row_repr = row_repr * row_mask_expanded.float()
        
        return row_repr
    
    def get_token_representations(
        self,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get token-level representations (after transformer, before any pooling).
        
        Useful for reconstruction heads that predict individual cell values.
        
        Args:
            tokens: Token tensor [B, max_rows, max_cols, TOKEN_DIM]
            row_mask: Boolean mask for valid rows [B, max_rows]
            col_mask: Boolean mask for valid columns [B, max_cols]
        
        Returns:
            token_repr: [B, max_rows, max_cols, hidden_dim] token representations
        """
        # Embed tokens
        x = self.token_embedding(tokens, col_mask)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=col_mask)
        
        return x


# =============================================================================
# Factory Function
# =============================================================================

def create_encoder(
    hidden_dim: int = 128,
    evidence_dim: int = 64,
    n_layers: int = 4,
    n_heads: int = 4,
    max_cols: int = 32,
    dropout: float = 0.1,
    row_pooling: str = "attention",
    dataset_pooling: str = "attention",
) -> LacunaEncoder:
    """
    Factory function to create a LacunaEncoder with specified parameters.
    
    Args:
        hidden_dim: Transformer hidden dimension
        evidence_dim: Output evidence vector dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        max_cols: Maximum number of columns (for position embeddings)
        dropout: Dropout probability
        row_pooling: Method for pooling features into rows ("mean", "max", "attention")
        dataset_pooling: Method for pooling rows into dataset ("mean", "max", "attention")
    
    Returns:
        Configured LacunaEncoder instance
    """
    config = EncoderConfig(
        hidden_dim=hidden_dim,
        evidence_dim=evidence_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_cols=max_cols,
        dropout=dropout,
        row_pooling=row_pooling,
        dataset_pooling=dataset_pooling,
    )
    
    return LacunaEncoder(config)