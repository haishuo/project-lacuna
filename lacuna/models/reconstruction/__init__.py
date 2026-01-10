"""
lacuna.models.reconstruction

Reconstruction heads for self-supervised pretraining.

This module provides reconstruction heads that predict missing/masked values
from transformer token representations. Each head embodies a different
assumption about how missingness relates to values:

    - MCARHead: Simple MLP, assumes random missingness
    - MARHead: Cross-attention to observed cells, assumes predictable missingness
    - MNARSelfCensoringHead: Adjusts for extreme-value censoring
    - MNARThresholdHead: Learns truncation thresholds
    - MNARLatentHead: Infers latent confounders

The ReconstructionHeads container manages all heads and computes reconstruction
errors that feed into the MoE gating mechanism.

Usage:
    from lacuna.models.reconstruction import (
        ReconstructionHeads,
        create_reconstruction_heads,
    )
    
    heads = create_reconstruction_heads(hidden_dim=128)
    results = heads(token_repr, tokens, row_mask, col_mask, original_values, recon_mask)
    errors = heads.get_error_tensor(results)  # [B, n_heads]
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from lacuna.core.types import ReconstructionResult
from lacuna.models.reconstruction.base import (
    BaseReconstructionHead,
    ReconstructionConfig,
)
from lacuna.models.reconstruction.heads import (
    MCARHead,
    MARHead,
    MNARSelfCensoringHead,
    MNARThresholdHead,
    MNARLatentHead,
    HEAD_REGISTRY,
    create_head,
)


# =============================================================================
# Combined Reconstruction Module
# =============================================================================

class ReconstructionHeads(nn.Module):
    """
    Container for all reconstruction heads.
    
    Manages MCAR, MAR, and MNAR variant heads. Computes predictions and
    reconstruction errors for each head, which feed into the MoE gating.
    
    Attributes:
        config: ReconstructionConfig with architecture parameters.
        mcar_head: MCAR reconstruction head.
        mar_head: MAR reconstruction head.
        mnar_heads: ModuleDict of MNAR variant heads.
        head_names: List of all head names in consistent order.
    
    Example:
        >>> config = ReconstructionConfig(hidden_dim=128)
        >>> heads = ReconstructionHeads(config)
        >>> results = heads(token_repr, tokens, row_mask, col_mask)
        >>> errors = heads.get_error_tensor(results)  # [B, 5] for 5 heads
    """
    
    def __init__(self, config: ReconstructionConfig):
        super().__init__()
        
        self.config = config
        
        # Core heads
        self.mcar_head = MCARHead(config)
        self.mar_head = MARHead(config)
        
        # MNAR variant heads
        self.mnar_heads = nn.ModuleDict()
        
        for variant in config.mnar_variants:
            self.mnar_heads[variant] = create_head(variant, config)
        
        # List of all head names for consistent ordering
        self.head_names = ["mcar", "mar"] + list(config.mnar_variants)
    
    @property
    def n_heads(self) -> int:
        """Total number of reconstruction heads."""
        return 2 + len(self.mnar_heads)  # MCAR + MAR + MNAR variants
    
    def forward(
        self,
        token_repr: torch.Tensor,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
        original_values: Optional[torch.Tensor] = None,
        reconstruction_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, ReconstructionResult]:
        """
        Run all reconstruction heads and compute errors.
        
        Args:
            token_repr: Transformer output representations.
                       Shape: [B, max_rows, max_cols, hidden_dim]
            tokens: Original input tokens.
                   Shape: [B, max_rows, max_cols, TOKEN_DIM]
            row_mask: Valid row mask. Shape: [B, max_rows]
            col_mask: Valid column mask. Shape: [B, max_cols]
            original_values: Ground truth values (for error computation).
                            Shape: [B, max_rows, max_cols]. Optional.
            reconstruction_mask: Cells to evaluate (artificially masked).
                                Shape: [B, max_rows, max_cols]. Optional.
        
        Returns:
            Dict mapping head_name -> ReconstructionResult.
            Each ReconstructionResult contains predictions, errors, and per_cell_errors.
        """
        results = {}
        B = token_repr.shape[0]
        device = token_repr.device
        
        # Helper to compute error or return zeros
        def compute_or_zeros(head, predictions):
            if original_values is not None and reconstruction_mask is not None:
                error, cell_error = head.compute_error(
                    predictions, original_values, reconstruction_mask, row_mask, col_mask
                )
                return error, cell_error
            else:
                return torch.zeros(B, device=device), None
        
        # MCAR head
        mcar_pred = self.mcar_head(token_repr, tokens, row_mask, col_mask)
        mcar_error, mcar_cell_error = compute_or_zeros(self.mcar_head, mcar_pred)
        results["mcar"] = ReconstructionResult(
            predictions=mcar_pred,
            errors=mcar_error,
            per_cell_errors=mcar_cell_error,
        )
        
        # MAR head
        mar_pred = self.mar_head(token_repr, tokens, row_mask, col_mask)
        mar_error, mar_cell_error = compute_or_zeros(self.mar_head, mar_pred)
        results["mar"] = ReconstructionResult(
            predictions=mar_pred,
            errors=mar_error,
            per_cell_errors=mar_cell_error,
        )
        
        # MNAR variant heads
        for variant_name, head in self.mnar_heads.items():
            pred = head(token_repr, tokens, row_mask, col_mask)
            error, cell_error = compute_or_zeros(head, pred)
            results[variant_name] = ReconstructionResult(
                predictions=pred,
                errors=error,
                per_cell_errors=cell_error,
            )
        
        return results
    
    def get_error_tensor(
        self,
        results: Dict[str, ReconstructionResult],
    ) -> torch.Tensor:
        """
        Stack reconstruction errors from all heads into a tensor.
        
        Args:
            results: Output from forward().
        
        Returns:
            errors: Reconstruction error per head. Shape: [B, n_heads]
        """
        errors = []
        for name in self.head_names:
            errors.append(results[name].errors)
        
        return torch.stack(errors, dim=-1)  # [B, n_heads]
    
    def get_predictions_dict(
        self,
        results: Dict[str, ReconstructionResult],
    ) -> Dict[str, torch.Tensor]:
        """
        Extract just the predictions from results.
        
        Args:
            results: Output from forward().
        
        Returns:
            Dict mapping head_name -> predictions tensor [B, max_rows, max_cols].
        """
        return {name: results[name].predictions for name in self.head_names}


# =============================================================================
# Factory Function
# =============================================================================

def create_reconstruction_heads(
    hidden_dim: int = 128,
    head_hidden_dim: int = 64,
    n_head_layers: int = 2,
    dropout: float = 0.1,
    mnar_variants: Optional[List[str]] = None,
) -> ReconstructionHeads:
    """
    Factory function to create ReconstructionHeads.
    
    Args:
        hidden_dim: Input dimension from encoder.
        head_hidden_dim: Hidden dimension within each head.
        n_head_layers: Depth of prediction networks.
        dropout: Dropout probability.
        mnar_variants: List of MNAR variants to include.
                      Default: ["self_censoring", "threshold", "latent"]
    
    Returns:
        Configured ReconstructionHeads instance.
    
    Example:
        >>> heads = create_reconstruction_heads(hidden_dim=128)
        >>> heads.n_heads
        5  # mcar, mar, self_censoring, threshold, latent
    """
    if mnar_variants is None:
        mnar_variants = ["self_censoring", "threshold", "latent"]
    
    config = ReconstructionConfig(
        hidden_dim=hidden_dim,
        head_hidden_dim=head_hidden_dim,
        n_head_layers=n_head_layers,
        dropout=dropout,
        mnar_variants=mnar_variants,
    )
    
    return ReconstructionHeads(config)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config
    "ReconstructionConfig",
    # Base class
    "BaseReconstructionHead",
    # Head implementations
    "MCARHead",
    "MARHead",
    "MNARSelfCensoringHead",
    "MNARThresholdHead",
    "MNARLatentHead",
    # Registry
    "HEAD_REGISTRY",
    "create_head",
    # Container
    "ReconstructionHeads",
    "create_reconstruction_heads",
]