"""
lacuna.models.heads

Classifier head: evidence embedding -> generator logits.
"""

import torch
import torch.nn as nn
from typing import Optional


class GeneratorHead(nn.Module):
    """Classifier head for generator prediction.
    
    Maps evidence embedding to logits over generators.
    
    Args:
        evidence_dim: Input dimension (p).
        n_generators: Number of generators (K).
        hidden_dim: Optional hidden layer dimension. If None, direct linear.
        dropout: Dropout rate (only used if hidden_dim is set).
    """
    
    def __init__(
        self,
        evidence_dim: int,
        n_generators: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.evidence_dim = evidence_dim
        self.n_generators = n_generators
        
        if hidden_dim is None:
            # Direct linear projection
            self.net = nn.Linear(evidence_dim, n_generators)
        else:
            # MLP with one hidden layer
            self.net = nn.Sequential(
                nn.Linear(evidence_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_generators),
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            evidence: [B, p] evidence embedding.
        
        Returns:
            [B, K] generator logits (before softmax).
        """
        return self.net(evidence)


class ClassHead(nn.Module):
    """Direct classifier head for mechanism class.
    
    Alternative to aggregating generator posteriors.
    Maps evidence directly to class logits.
    
    Args:
        evidence_dim: Input dimension (p).
        n_classes: Number of classes (3 for MCAR/MAR/MNAR).
        hidden_dim: Optional hidden layer dimension.
    """
    
    def __init__(
        self,
        evidence_dim: int,
        n_classes: int = 3,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        
        if hidden_dim is None:
            self.net = nn.Linear(evidence_dim, n_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(evidence_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_classes),
            )
    
    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            evidence: [B, p] evidence embedding.
        
        Returns:
            [B, 3] class logits.
        """
        return self.net(evidence)
