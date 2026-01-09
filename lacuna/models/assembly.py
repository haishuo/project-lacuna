"""
lacuna.models.assembly

Full model composition: LacunaModel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from lacuna.core.types import TokenBatch, PosteriorResult, Decision
from lacuna.data.tokenization import TOKEN_DIM
from .encoder import EvidenceEncoder
from .heads import GeneratorHead
from .aggregator import (
    aggregate_to_class_posterior_efficient,
    compute_entropy,
)
from .decision import bayes_optimal_decision, DEFAULT_LOSS_MATRIX


class LacunaModel(nn.Module):
    """Complete Lacuna model for missingness mechanism classification."""
    
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
        max_rows: int = 256,
        head_hidden_dim: Optional[int] = None,
        loss_matrix: Optional[torch.Tensor] = None,
        row_agg: str = "attention",
    ):
        super().__init__()
        
        self.n_generators = n_generators
        self.hidden_dim = hidden_dim
        self.evidence_dim = evidence_dim
        
        # Evidence encoder
        self.encoder = EvidenceEncoder(
            token_dim=TOKEN_DIM,
            hidden_dim=hidden_dim,
            evidence_dim=evidence_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_cols=max_cols,
            max_rows=max_rows,
            row_agg=row_agg,
        )
        
        # Generator classifier head
        self.head = GeneratorHead(
            evidence_dim=evidence_dim,
            n_generators=n_generators,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )
        
        # Register class mapping as buffer
        self.register_buffer('class_mapping', class_mapping)
        
        # Register loss matrix
        if loss_matrix is None:
            loss_matrix = DEFAULT_LOSS_MATRIX
        self.register_buffer('loss_matrix', loss_matrix)
    
    def forward(self, batch: TokenBatch) -> PosteriorResult:
        """Forward pass producing posteriors."""
        evidence = self.encoder(
            batch.tokens,
            batch.row_mask,
            batch.col_mask,
        )
        
        logits = self.head(evidence)
        p_generator = F.softmax(logits, dim=-1)
        
        p_class = aggregate_to_class_posterior_efficient(
            p_generator, self.class_mapping
        )
        
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
        """Make Bayes-optimal decision from posterior."""
        action_ids, expected_risks = bayes_optimal_decision(
            posterior.p_class,
            self.loss_matrix,
        )
        
        return Decision(
            action_ids=action_ids,
            expected_risks=expected_risks,
        )
    
    def forward_with_decision(
        self,
        batch: TokenBatch,
    ) -> Tuple[PosteriorResult, Decision]:
        """Forward pass with decision in one call."""
        posterior = self.forward(batch)
        decision = self.decide(posterior)
        return posterior, decision
    
    def get_evidence(self, batch: TokenBatch) -> torch.Tensor:
        """Get evidence embedding."""
        return self.encoder(batch.tokens, batch.row_mask, batch.col_mask)
    
    @classmethod
    def from_config(cls, config, class_mapping: torch.Tensor) -> "LacunaModel":
        """Create model from LacunaConfig."""
        return cls(
            n_generators=config.generator.n_generators,
            class_mapping=class_mapping,
            hidden_dim=config.model.hidden_dim,
            evidence_dim=config.model.evidence_dim,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            dropout=config.model.dropout,
            max_cols=config.data.max_cols,
            max_rows=config.data.max_rows,
            loss_matrix=config.get_loss_matrix_tensor(),
        )
