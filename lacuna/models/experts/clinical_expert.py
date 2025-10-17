"""
lacuna.models.experts.clinical_expert

Purpose: Expert for clinical trial data

Design Principles:
- UNIX Philosophy: Do ONE thing well
- No defaults (except top-level config)
- Trust neighbors (no redundant validation)
- Fail fast and loud
- Target: <300 lines

Spec Reference: Section 4.6
"""

import torch
import torch.nn as nn
from .base_expert import BaseExpert


class ClinicalExpert(BaseExpert):
    """Expert specialized for clinical trial missingness patterns"""
    
    def __init__(self, config):
        """
        Args:
            config: ExpertConfig (required fields)
        """
        if config.input_dim is None:
            raise ValueError("config.input_dim required")
        
        super().__init__(config)
        # TODO: Build clinical-specific architecture
        # See spec section 4.6
    
    def forward(self, pooled_repr):
        """Process clinical trial patterns"""
        # TODO: Implement clinical-specific processing
        raise NotImplementedError("See spec section 4.6")
    
    @classmethod
    def load(cls, path):
        """Load from checkpoint"""
        raise NotImplementedError()

