"""
lacuna.config.schema

Configuration dataclasses with validation.

Simplified for row-level tokenization architecture.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import torch


@dataclass
class DataConfig:
    """Data processing configuration."""
    n_range: Tuple[int, int] = (50, 500)
    d_range: Tuple[int, int] = (3, 20)
    max_cols: int = 32
    max_rows: int = 256
    
    def __post_init__(self):
        if self.n_range[0] > self.n_range[1]:
            raise ValueError("n_range[0] must be <= n_range[1]")
        if self.d_range[0] > self.d_range[1]:
            raise ValueError("d_range[0] must be <= d_range[1]")
        if self.max_cols <= 0:
            raise ValueError("max_cols must be positive")
        if self.max_rows <= 0:
            raise ValueError("max_rows must be positive")


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
    """Training configuration."""
    epochs: int = 50
    batch_size: int = 32
    batches_per_epoch: int = 100
    val_batches: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 200
    patience: int = 10
    min_delta: float = 1e-4
    
    def __post_init__(self):
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass
class GeneratorConfig:
    """Generator configuration."""
    n_generators: int = 6


@dataclass
class LacunaConfig:
    """Complete Lacuna configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "/mnt/artifacts/project_lacuna/runs"
    loss_matrix: List[List[float]] = field(default_factory=lambda: [
        [0.0, 0.3, 1.0],
        [0.2, 0.0, 0.2],
        [1.0, 0.3, 0.0],
    ])
    
    def get_loss_matrix_tensor(self) -> torch.Tensor:
        """Convert loss matrix to tensor."""
        return torch.tensor(self.loss_matrix, dtype=torch.float32)
    
    @classmethod
    def minimal(cls) -> "LacunaConfig":
        """Create minimal config for fast testing."""
        return cls(
            data=DataConfig(
                n_range=(50, 100),
                d_range=(3, 8),
                max_cols=16,
                max_rows=64,
            ),
            model=ModelConfig(
                hidden_dim=64,
                evidence_dim=32,
                n_layers=4,
                n_heads=4,
                dropout=0.1,
            ),
            training=TrainingConfig(
                epochs=5,
                batch_size=8,
                batches_per_epoch=20,
                val_batches=5,
                lr=1e-3,
                warmup_steps=10,
                patience=3,
            ),
            generator=GeneratorConfig(n_generators=6),
            device="cpu",
        )
