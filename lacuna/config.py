"""
lacuna.config

Purpose: Configuration system - ONLY place with defaults

Design Principles:
- UNIX Philosophy: Do ONE thing well
- No defaults (except top-level config)
- Trust neighbors (no redundant validation)
- Fail fast and loud
- Target: <300 lines

Spec Reference: Section 5.1
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValidatorConfig:
    """Configuration for input validation"""
    valid_domains: List[str]  # Required
    min_rows: int = 10  # DEFAULT OK here
    min_cols: int = 2  # DEFAULT OK here


@dataclass
class MCARConfig:
    """Configuration for MCAR testing"""
    alpha: float = 0.05  # DEFAULT OK here


@dataclass
class TokenizerConfig:
    """Configuration for tokenization"""
    n_bins: int = 50  # DEFAULT OK here
    max_seq_len: int = 512  # DEFAULT OK here


@dataclass
class LacunaBERTConfig:
    """Configuration for BERT encoder"""
    vocab_size: int  # Required - no default
    hidden_size: int = 768  # DEFAULT OK here
    num_layers: int = 12  # DEFAULT OK here
    num_heads: int = 12  # DEFAULT OK here


@dataclass
class ExpertConfig:
    """Configuration for domain experts"""
    input_dim: int  # Required
    hidden_dim: int = 256  # DEFAULT OK here
    output_dim: int = 256  # DEFAULT OK here


@dataclass
class GatingConfig:
    """Configuration for gating network"""
    input_dim: int  # Required
    num_experts: int = 5  # DEFAULT OK here


@dataclass
class FusionConfig:
    """Configuration for fusion classifier"""
    expert_output_dim: int  # Required
    num_experts: int = 5  # DEFAULT OK here
    num_classes: int = 3  # MCAR/MAR/MNAR


@dataclass
class LacunaConfig:
    """Top-level configuration - ONLY place with defaults"""
    validator: ValidatorConfig
    mcar: MCARConfig
    tokenizer: TokenizerConfig
    bert: LacunaBERTConfig
    expert: ExpertConfig
    gating: GatingConfig
    fusion: FusionConfig
    
    @classmethod
    def default(cls):
        """Factory for default configuration"""
        return cls(
            validator=ValidatorConfig(
                valid_domains=['clinical_trials', 'surveys', 'longitudinal',
                             'observational', 'experimental']
            ),
            mcar=MCARConfig(),
            tokenizer=TokenizerConfig(),
            bert=LacunaBERTConfig(vocab_size=10000),  # Placeholder
            expert=ExpertConfig(input_dim=768),
            gating=GatingConfig(input_dim=768),
            fusion=FusionConfig(expert_output_dim=256)
        )

