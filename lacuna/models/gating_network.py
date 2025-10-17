"""
lacuna.models.gating_network

Purpose: Route inputs to domain experts

Design Principles:
- UNIX Philosophy: Do ONE thing well
- No defaults (except top-level config)
- Trust neighbors (no redundant validation)
- Fail fast and loud
- Target: <200 lines

Spec Reference: Section 4.7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """Routes inputs to domain experts
    
    Design: Trust encoder produced valid representations
    """
    
    def __init__(self, config):
        """
        Args:
            config: GatingConfig (required fields)
        """
        if config.input_dim is None:
            raise ValueError("config.input_dim required")
        
        super().__init__()
        # TODO: Build routing network
        # See spec section 4.7
    
    def forward(self, pooled_repr: torch.Tensor, domain_hint: str) -> torch.Tensor:
        """Compute expert routing weights
        
        Args:
            pooled_repr: Encoder output (trust it's valid)
            domain_hint: Domain string (trust it's valid)
        
        Returns:
            Expert weights (batch, num_experts)
        """
        # TODO: Implement routing
        raise NotImplementedError("See spec section 4.7")

