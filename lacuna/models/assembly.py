"""
lacuna.models.assembly

Full model composition: LacunaModel.

Combines:
- EvidenceEncoder: tokens -> evidence
- GeneratorHead: evidence -> generator logits
- Aggregator: generator posterior -> class posterior
- Decision: class posterior -> action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from lacuna.core.types import TokenBatch, PosteriorResult, Decision
from lacuna.data.features import FEATURE_DIM
from .encoder import EvidenceEncoder
from .heads import GeneratorHead
from .aggregator import (
    aggregate_to_class_posterior_efficient,
    compute_entropy,
    compute_confidence,
)
from .decision import bayes_optimal_decision, DEFAULT_LOSS_MATRIX


class LacunaModel(nn.Module):
    """Complete Lacuna model for missingness mechanism classification.
    
    Architecture:
        TokenBatch -> EvidenceEncoder -> GeneratorHead -> Aggregator -> Decision
    
    Args:
        n_generators: Number of generators (K).
        class_mapping: [K] tensor mapping generator_id -> class_id.
        hidden_dim: Transformer hidden dimension.
        evidence_dim: Evidence embedding dimension.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        max_cols: Maximum number of columns.
        head_hidden_dim: Hidden dimension for classifier head (None = linear).
        loss_matrix: [3, 3] loss matrix for decision rule.
    """
    
    def __init__(
        self,
        n_generators: int,
        class_mapping: torch.Tensor,
        hidden_dim: int = 128,
        evidence_dim: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_cols: int = 32,
        head_hidden_dim: Optional[int] = None,
        loss_matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.n_generators = n_generators
        self.hidden_dim = hidden_dim
        self.evidence_dim = evidence_dim
        
        # Evidence encoder
        self.encoder = EvidenceEncoder(
            token_dim=FEATURE_DIM,
            hidden_dim=hidden_dim,
            evidence_dim=evidence_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_cols=max_cols,
        )
        
        # Generator classifier head
        self.head = GeneratorHead(
            evidence_dim=evidence_dim,
            n_generators=n_generators,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )
        
        # Register class mapping as buffer (not a parameter)
        self.register_buffer('class_mapping', class_mapping)
        
        # Register loss matrix
        if loss_matrix is None:
            loss_matrix = DEFAULT_LOSS_MATRIX
        self.register_buffer('loss_matrix', loss_matrix)
    
    def forward(self, batch: TokenBatch) -> PosteriorResult:
        """Forward pass producing posteriors.
        
        Args:
            batch: TokenBatch with tokens and col_mask.
        
        Returns:
            PosteriorResult with generator and class posteriors.
        """
        # Encode to evidence
        evidence = self.encoder(batch.tokens, batch.col_mask)  # [B, p]
        
        # Get generator logits
        logits = self.head(evidence)  # [B, K]
        
        # Generator posterior (softmax)
        p_generator = F.softmax(logits, dim=-1)  # [B, K]
        
        # Aggregate to class posterior
        p_class = aggregate_to_class_posterior_efficient(
            p_generator, self.class_mapping
        )  # [B, 3]
        
        # Compute entropies
        entropy_generator = compute_entropy(p_generator, dim=-1)
        entropy_class = compute_entropy(p_class, dim=-1)
        
        return PosteriorResult(
            p_generator=p_generator,
            p_class=p_class,
            entropy_generator=entropy_generator,
            entropy_class=entropy_class,
            logits_generator=logits,
        )
    
    def decide(self, posterior: PosteriorResult) -> Decision:
        """Make Bayes-optimal decision from posterior.
        
        Args:
            posterior: PosteriorResult from forward().
        
        Returns:
            Decision with action IDs and expected risks.
        """
        action_ids, expected_risks = bayes_optimal_decision(
            posterior.p_class,
            self.loss_matrix,
        )
        
        return Decision(
            action_ids=action_ids,
            action_names=("Green", "Yellow", "Red"),
            expected_risks=expected_risks,
        )
    
    def forward_with_decision(
        self,
        batch: TokenBatch,
    ) -> Tuple[PosteriorResult, Decision]:
        """Forward pass with decision in one call.
        
        Args:
            batch: TokenBatch input.
        
        Returns:
            (PosteriorResult, Decision) tuple.
        """
        posterior = self.forward(batch)
        decision = self.decide(posterior)
        return posterior, decision
    
    def get_evidence(self, batch: TokenBatch) -> torch.Tensor:
        """Get evidence embedding (for analysis/visualization).
        
        Args:
            batch: TokenBatch input.
        
        Returns:
            [B, evidence_dim] evidence embedding.
        """
        return self.encoder(batch.tokens, batch.col_mask)
    
    @classmethod
    def from_config(cls, config, class_mapping: torch.Tensor) -> "LacunaModel":
        """Create model from LacunaConfig.
        
        Args:
            config: LacunaConfig object.
            class_mapping: [K] generator to class mapping.
        
        Returns:
            Initialized LacunaModel.
        """
        return cls(
            n_generators=config.generator.n_generators,
            class_mapping=class_mapping,
            hidden_dim=config.model.hidden_dim,
            evidence_dim=config.model.evidence_dim,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            dropout=config.model.dropout,
            max_cols=config.data.max_cols,
            loss_matrix=config.get_loss_matrix_tensor(),
        )
