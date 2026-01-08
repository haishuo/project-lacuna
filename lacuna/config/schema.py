"""
lacuna.config.schema

Configuration schemas using dataclasses.

Design: All config fields have explicit types. Defaults only at top level.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


@dataclass
class DataConfig:
    """Data processing configuration."""
    max_cols: int = 32
    n_range: Tuple[int, int] = (50, 500)
    d_range: Tuple[int, int] = (5, 20)
    normalization: str = "robust"  # "robust" | "standard" | "none"
    
    def __post_init__(self):
        if self.normalization not in ("robust", "standard", "none"):
            raise ValueError(f"Invalid normalization: {self.normalization}")
        if self.n_range[0] > self.n_range[1]:
            raise ValueError("n_range[0] must be <= n_range[1]")
        if self.d_range[0] > self.d_range[1]:
            raise ValueError("d_range[0] must be <= d_range[1]")


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_dim: int = 128
    evidence_dim: int = 64
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 20
    warmup_steps: int = 100
    patience: int = 5
    grad_clip: float = 1.0
    
    def __post_init__(self):
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass  
class GeneratorConfig:
    """Generator system configuration."""
    n_generators: int = 6  # Start with minimal set
    class_balance: Tuple[float, float, float] = (0.33, 0.33, 0.34)  # MCAR, MAR, MNAR
    
    def __post_init__(self):
        if abs(sum(self.class_balance) - 1.0) > 1e-6:
            raise ValueError("class_balance must sum to 1")


@dataclass
class LacunaConfig:
    """Top-level configuration.
    
    This is the ONLY place defaults are specified.
    All sub-configs receive explicit values.
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "/mnt/artifacts/project_lacuna/runs"
    
    # Loss matrix for decision rule [action, true_class]
    # Rows: Green, Yellow, Red
    # Cols: MCAR, MAR, MNAR
    loss_matrix: List[float] = field(default_factory=lambda: [
        0.0, 0.0, 10.0,   # Green: safe for MCAR/MAR, costly for MNAR
        1.0, 1.0, 2.0,    # Yellow: small cost always
        3.0, 2.0, 0.0,    # Red: costly for MCAR/MAR, free for MNAR
    ])
    
    def get_loss_matrix_tensor(self) -> torch.Tensor:
        """Return loss matrix as [3, 3] tensor."""
        return torch.tensor(self.loss_matrix).reshape(3, 3)
    
    @classmethod
    def minimal(cls) -> "LacunaConfig":
        """Factory for minimal testing configuration."""
        return cls(
            data=DataConfig(max_cols=16, n_range=(50, 100), d_range=(3, 8)),
            model=ModelConfig(hidden_dim=64, evidence_dim=32, n_layers=2, n_heads=2),
            training=TrainingConfig(batch_size=16, epochs=5),
            generator=GeneratorConfig(n_generators=6),
        )
