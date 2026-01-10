"""
lacuna.models.reconstruction.base

Base class and configuration for reconstruction heads.

Reconstruction heads predict missing/masked values from transformer token
representations. Each head embodies a different "world model" - an assumption
about how missingness relates to values.

The pattern of reconstruction errors across heads becomes a discriminative
signal for mechanism classification:
    - MCAR: Similar error across heads (no special structure)
    - MAR: Low error for heads that use cross-column info
    - MNAR: High error when predicting self-censored values
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from dataclasses import dataclass

from lacuna.core.types import ReconstructionResult


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ReconstructionConfig:
    """Configuration for reconstruction heads."""
    
    # Input dimensions (from encoder)
    hidden_dim: int = 128          # Token representation dimension
    
    # Head architecture
    head_hidden_dim: int = 64      # Hidden dimension within each head
    n_head_layers: int = 2         # Depth of each prediction head
    
    # Regularization
    dropout: float = 0.1
    
    # MNAR variants to include
    mnar_variants: List[str] = None  # e.g., ["self_censoring", "threshold", "latent"]
    
    def __post_init__(self):
        if self.mnar_variants is None:
            self.mnar_variants = ["self_censoring", "threshold", "latent"]


# =============================================================================
# Base Reconstruction Head
# =============================================================================

class BaseReconstructionHead(nn.Module):
    """
    Base class for reconstruction heads.
    
    All heads take token representations and output per-cell value predictions.
    Subclasses implement different inductive biases about how values relate.
    
    Attributes:
        config: ReconstructionConfig with architecture parameters.
    
    Methods:
        forward: Predict values for all cells.
        compute_error: Compute reconstruction error on masked cells.
    """
    
    def __init__(self, config: ReconstructionConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self,
        token_repr: torch.Tensor,   # [B, max_rows, max_cols, hidden_dim]
        tokens: torch.Tensor,        # [B, max_rows, max_cols, TOKEN_DIM] original tokens
        row_mask: torch.Tensor,      # [B, max_rows]
        col_mask: torch.Tensor,      # [B, max_cols]
    ) -> torch.Tensor:
        """
        Predict values for all cells.
        
        Args:
            token_repr: Transformer output representations.
                       Shape: [B, max_rows, max_cols, hidden_dim]
            tokens: Original input tokens (for observed values/masks).
                   Shape: [B, max_rows, max_cols, TOKEN_DIM]
            row_mask: Valid row mask. Shape: [B, max_rows]
            col_mask: Valid column mask. Shape: [B, max_cols]
        
        Returns:
            predictions: Predicted values. Shape: [B, max_rows, max_cols]
        
        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def compute_error(
        self,
        predictions: torch.Tensor,       # [B, max_rows, max_cols]
        targets: torch.Tensor,           # [B, max_rows, max_cols]
        reconstruction_mask: torch.Tensor,  # [B, max_rows, max_cols] cells to evaluate
        row_mask: torch.Tensor,          # [B, max_rows]
        col_mask: torch.Tensor,          # [B, max_cols]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction error on masked cells.
        
        Only computes error on cells that are:
        1. In the reconstruction_mask (artificially masked during pretraining)
        2. In a valid row (row_mask is True)
        3. In a valid column (col_mask is True)
        
        Args:
            predictions: Predicted values. Shape: [B, max_rows, max_cols]
            targets: Ground truth values (original_values from TokenBatch).
                    Shape: [B, max_rows, max_cols]
            reconstruction_mask: Boolean mask of artificially-masked cells.
                                Shape: [B, max_rows, max_cols]
            row_mask: Valid row mask. Shape: [B, max_rows]
            col_mask: Valid column mask. Shape: [B, max_cols]
        
        Returns:
            per_sample_error: MSE per sample (averaged over masked cells).
                             Shape: [B]
            per_cell_error: Squared error per cell.
                           Shape: [B, max_rows, max_cols]
        """
        # Compute squared error everywhere
        per_cell_error = (predictions - targets) ** 2
        
        # Create validity mask: reconstruction_mask AND valid row AND valid column
        # reconstruction_mask: [B, max_rows, max_cols]
        # row_mask: [B, max_rows] -> [B, max_rows, 1]
        # col_mask: [B, max_cols] -> [B, 1, max_cols]
        validity = (
            reconstruction_mask 
            & row_mask.unsqueeze(-1) 
            & col_mask.unsqueeze(-2)
        )
        
        # Masked mean per sample
        masked_error = per_cell_error * validity.float()
        sum_error = masked_error.sum(dim=(1, 2))  # [B]
        count = validity.float().sum(dim=(1, 2)).clamp(min=1.0)  # [B]
        per_sample_error = sum_error / count
        
        return per_sample_error, per_cell_error
    
    def forward_with_error(
        self,
        token_repr: torch.Tensor,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
        original_values: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> ReconstructionResult:
        """
        Convenience method: forward + compute_error in one call.
        
        Args:
            token_repr: Transformer output representations.
            tokens: Original input tokens.
            row_mask: Valid row mask.
            col_mask: Valid column mask.
            original_values: Ground truth values.
            reconstruction_mask: Cells to evaluate.
        
        Returns:
            ReconstructionResult with predictions, errors, and per_cell_errors.
        """
        predictions = self.forward(token_repr, tokens, row_mask, col_mask)
        errors, per_cell_errors = self.compute_error(
            predictions, original_values, reconstruction_mask, row_mask, col_mask
        )
        
        return ReconstructionResult(
            predictions=predictions,
            errors=errors,
            per_cell_errors=per_cell_errors,
        )