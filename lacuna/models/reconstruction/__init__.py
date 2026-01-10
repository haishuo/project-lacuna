"""
lacuna.models.reconstruction

Reconstruction heads for mechanism classification.

CRITICAL FIX (2026-01-10):
--------------------------
The original implementation computed reconstruction errors on ARTIFICIALLY
masked cells (for self-supervised pretraining). This does NOT distinguish
MAR from MNAR because both mechanisms look similar under artificial masking.

The discriminative signal comes from errors on NATURALLY MISSING cells:
    - MAR: MARHead predicts well (cross-attention to observed values helps)
    - MNAR: MARHead predicts poorly (the missing value itself is informative)

This refactored module computes BOTH:
    1. `artificial_errors`: For self-supervised learning (training signal)
    2. `natural_errors`: For mechanism discrimination (classification signal)

The natural_errors are fed to the MoE gating network as the discriminative
feature that separates MAR from MNAR.

Usage:
    from lacuna.models.reconstruction import (
        ReconstructionHeads,
        create_reconstruction_heads,
    )
    
    heads = create_reconstruction_heads(hidden_dim=128)
    results = heads(token_repr, tokens, row_mask, col_mask, original_values, 
                   reconstruction_mask, compute_natural_errors=True)
    
    # For self-supervised loss
    artificial_errors = heads.get_error_tensor(results)  # [B, n_heads]
    
    # For mechanism discrimination (feed to MoE)
    natural_errors = heads.get_natural_error_tensor(results)  # [B, n_heads]
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
from lacuna.data.tokenization import IDX_OBSERVED


# =============================================================================
# Extended ReconstructionResult with Natural Errors
# =============================================================================

class ExtendedReconstructionResult:
    """
    Extended reconstruction result with both artificial and natural errors.
    
    Attributes:
        predictions: Predicted values for all cells [B, max_rows, max_cols]
        errors: Per-sample error on ARTIFICIALLY masked cells [B]
        per_cell_errors: Per-cell squared errors [B, max_rows, max_cols]
        natural_errors: Per-sample error on NATURALLY missing cells [B]
        n_natural_missing: Count of naturally missing cells per sample [B]
    """
    
    def __init__(
        self,
        predictions: torch.Tensor,
        errors: torch.Tensor,
        per_cell_errors: torch.Tensor,
        natural_errors: Optional[torch.Tensor] = None,
        n_natural_missing: Optional[torch.Tensor] = None,
    ):
        self.predictions = predictions
        self.errors = errors
        self.per_cell_errors = per_cell_errors
        self.natural_errors = natural_errors
        self.n_natural_missing = n_natural_missing


# =============================================================================
# Refactored ReconstructionHeads
# =============================================================================

class ReconstructionHeads(nn.Module):
    """
    Container for all reconstruction heads with natural error computation.
    
    This refactored version computes reconstruction errors on BOTH:
    1. Artificially masked cells (for self-supervised training)
    2. Naturally missing cells (for MAR/MNAR discrimination)
    
    The natural errors are the KEY discriminative signal:
    - Under MAR: MARHead has LOWER error than MCARHead (cross-attention helps)
    - Under MNAR: MARHead has SIMILAR/HIGHER error (cross-attention doesn't help)
    
    Attributes:
        config: ReconstructionConfig with architecture parameters.
        mcar_head: MCAR reconstruction head (simple MLP).
        mar_head: MAR reconstruction head (cross-attention to observed).
        mnar_heads: ModuleDict of MNAR variant heads.
        head_names: List of all head names in consistent order.
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
        return 2 + len(self.mnar_heads)
    
    def _get_head(self, name: str) -> BaseReconstructionHead:
        """Get head by name."""
        if name == "mcar":
            return self.mcar_head
        elif name == "mar":
            return self.mar_head
        else:
            return self.mnar_heads[name]
    
    def _compute_natural_missing_mask(
        self,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mask for naturally missing cells.
        
        Natural missing = valid cell AND not observed (is_observed=0 in token)
        
        Args:
            tokens: Input tokens [B, max_rows, max_cols, TOKEN_DIM]
            row_mask: Valid rows [B, max_rows]
            col_mask: Valid columns [B, max_cols]
        
        Returns:
            natural_missing: Boolean mask [B, max_rows, max_cols]
        """
        # Extract observation status from tokens
        is_observed = tokens[..., IDX_OBSERVED] > 0.5  # [B, max_rows, max_cols]
        
        # Valid cells: row is valid AND column is valid
        valid_cells = row_mask.unsqueeze(-1) & col_mask.unsqueeze(1)
        
        # Naturally missing: valid cell AND not observed
        natural_missing = valid_cells & ~is_observed
        
        return natural_missing
    
    def _compute_error_on_mask(
        self,
        predictions: torch.Tensor,
        original_values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction error on masked cells.
        
        Args:
            predictions: Predicted values [B, max_rows, max_cols]
            original_values: Ground truth values [B, max_rows, max_cols]
            mask: Boolean mask of cells to evaluate [B, max_rows, max_cols]
        
        Returns:
            per_sample_error: Mean squared error per sample [B]
            n_cells: Number of cells in mask per sample [B]
        """
        # Squared error everywhere
        squared_error = (predictions - original_values) ** 2
        
        # Masked sum
        masked_error = squared_error * mask.float()
        sum_error = masked_error.sum(dim=(1, 2))  # [B]
        
        # Count cells
        n_cells = mask.float().sum(dim=(1, 2))  # [B]
        
        # Mean (avoid division by zero)
        per_sample_error = sum_error / n_cells.clamp(min=1.0)
        
        return per_sample_error, n_cells
    
    def forward(
        self,
        token_repr: torch.Tensor,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
        original_values: Optional[torch.Tensor] = None,
        reconstruction_mask: Optional[torch.Tensor] = None,
        compute_natural_errors: bool = True,
    ) -> Dict[str, ExtendedReconstructionResult]:
        """
        Run all reconstruction heads and compute errors.
        
        Args:
            token_repr: Transformer output [B, max_rows, max_cols, hidden_dim]
            tokens: Original input tokens [B, max_rows, max_cols, TOKEN_DIM]
            row_mask: Valid row mask [B, max_rows]
            col_mask: Valid column mask [B, max_cols]
            original_values: Ground truth values [B, max_rows, max_cols]
            reconstruction_mask: Artificially masked cells [B, max_rows, max_cols]
            compute_natural_errors: Whether to compute errors on naturally missing cells.
                                   Set True for training with discrimination signal.
        
        Returns:
            Dict mapping head_name -> ExtendedReconstructionResult
        """
        B = token_repr.shape[0]
        device = token_repr.device
        
        # Compute natural missing mask if needed
        natural_missing_mask = None
        if compute_natural_errors and original_values is not None:
            natural_missing_mask = self._compute_natural_missing_mask(
                tokens, row_mask, col_mask
            )
        
        # Compute combined mask for artificial errors
        # reconstruction_mask AND valid row AND valid column
        artificial_mask = None
        if reconstruction_mask is not None:
            artificial_mask = (
                reconstruction_mask
                & row_mask.unsqueeze(-1)
                & col_mask.unsqueeze(-2)
            )
        
        results = {}
        
        for name in self.head_names:
            head = self._get_head(name)
            
            # Get predictions
            predictions = head(token_repr, tokens, row_mask, col_mask)
            
            # Compute artificial errors (for self-supervised loss)
            if artificial_mask is not None and original_values is not None:
                artificial_errors, _ = self._compute_error_on_mask(
                    predictions, original_values, artificial_mask
                )
                per_cell_errors = (predictions - original_values) ** 2
            else:
                artificial_errors = torch.zeros(B, device=device)
                per_cell_errors = torch.zeros_like(predictions)
            
            # Compute natural errors (for MAR/MNAR discrimination)
            if natural_missing_mask is not None and original_values is not None:
                natural_errors, n_natural = self._compute_error_on_mask(
                    predictions, original_values, natural_missing_mask
                )
            else:
                natural_errors = None
                n_natural = None
            
            results[name] = ExtendedReconstructionResult(
                predictions=predictions,
                errors=artificial_errors,
                per_cell_errors=per_cell_errors,
                natural_errors=natural_errors,
                n_natural_missing=n_natural,
            )
        
        return results
    
    def get_error_tensor(
        self,
        results: Dict[str, ExtendedReconstructionResult],
    ) -> torch.Tensor:
        """
        Get artificial errors as tensor (for self-supervised loss).
        
        Args:
            results: Output from forward().
        
        Returns:
            errors: Artificial reconstruction error per head [B, n_heads]
        """
        errors = []
        for name in self.head_names:
            errors.append(results[name].errors)
        
        return torch.stack(errors, dim=-1)
    
    def get_natural_error_tensor(
        self,
        results: Dict[str, ExtendedReconstructionResult],
    ) -> Optional[torch.Tensor]:
        """
        Get natural errors as tensor (for MAR/MNAR discrimination).
        
        THIS IS THE KEY DISCRIMINATIVE SIGNAL.
        
        The ratio of errors across heads reveals the mechanism:
        - MAR: MARHead error < MCARHead error (cross-attention helps)
        - MNAR: MARHead error >= MCARHead error (cross-attention doesn't help)
        
        Args:
            results: Output from forward().
        
        Returns:
            errors: Natural reconstruction error per head [B, n_heads]
                   Returns None if natural errors weren't computed.
        """
        # Check if natural errors were computed
        first_result = results[self.head_names[0]]
        if first_result.natural_errors is None:
            return None
        
        errors = []
        for name in self.head_names:
            errors.append(results[name].natural_errors)
        
        return torch.stack(errors, dim=-1)
    
    def get_natural_error_features(
        self,
        results: Dict[str, ExtendedReconstructionResult],
    ) -> Optional[torch.Tensor]:
        """
        Compute discriminative features from natural error patterns.
        
        Features:
        1. Error ratios: MAR_error / MCAR_error, MNAR_i / MCAR for each variant
        2. Error differences: MAR_error - MCAR_error, etc.
        
        These features directly encode the MAR/MNAR discrimination signal.
        
        Args:
            results: Output from forward().
        
        Returns:
            features: Discriminative features [B, n_features]
                     Returns None if natural errors weren't computed.
        """
        natural_errors = self.get_natural_error_tensor(results)
        if natural_errors is None:
            return None
        
        B, n_heads = natural_errors.shape
        device = natural_errors.device
        
        # MCAR is always index 0, MAR is always index 1
        mcar_errors = natural_errors[:, 0:1].clamp(min=1e-6)  # [B, 1]
        mar_errors = natural_errors[:, 1:2]  # [B, 1]
        other_errors = natural_errors[:, 1:]  # [B, n_heads-1] (MAR + MNAR variants)
        
        # Feature 1: Log error ratios relative to MCAR
        # log(MAR/MCAR), log(MNAR_i/MCAR)
        log_ratios = torch.log(other_errors.clamp(min=1e-6) / mcar_errors)
        
        # Feature 2: Error differences relative to MCAR
        # MAR - MCAR, MNAR_i - MCAR
        differences = other_errors - mcar_errors
        
        # Feature 3: MAR vs mean MNAR ratio (key discrimination)
        if n_heads > 2:
            mnar_mean = natural_errors[:, 2:].mean(dim=1, keepdim=True)  # [B, 1]
            mar_vs_mnar = mar_errors / mnar_mean.clamp(min=1e-6)
        else:
            mar_vs_mnar = torch.ones(B, 1, device=device)
        
        # Concatenate all features
        features = torch.cat([
            log_ratios,      # [B, n_heads-1]
            differences,     # [B, n_heads-1]
            mar_vs_mnar,     # [B, 1]
        ], dim=-1)
        
        return features
    
    def get_predictions_dict(
        self,
        results: Dict[str, ExtendedReconstructionResult],
    ) -> Dict[str, torch.Tensor]:
        """Extract just the predictions from results."""
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
    "ExtendedReconstructionResult",
    "create_reconstruction_heads",
]