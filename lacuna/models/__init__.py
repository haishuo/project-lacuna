"""
lacuna.models

Neural network components for missing data mechanism classification.

Architecture Overview:
    LacunaModel combines:
    1. LacunaEncoder: Transformer over feature tokens → evidence vector
    2. ReconstructionHeads: Predict masked values (self-supervised)
    3. MixtureOfExperts: Mechanism classification via gating
    4. BayesOptimalDecision: Convert posteriors to actions

Quick Start:
    >>> from lacuna.models import create_lacuna_model, create_lacuna_mini
    >>> 
    >>> # Create model
    >>> model = create_lacuna_model(hidden_dim=128, n_layers=4)
    >>> 
    >>> # Forward pass
    >>> output = model(batch)
    >>> print(output.posterior.p_class)  # [B, 3] mechanism posteriors
    >>> print(output.decision.action_ids)  # [B] recommended actions

Model Variants:
    - create_lacuna_mini(): Small model for testing/debugging
    - create_lacuna_base(): Standard production model
    - create_lacuna_large(): Maximum accuracy model
"""

# === Encoder ===
from lacuna.models.encoder import (
    LacunaEncoder,
    EncoderConfig,
    create_encoder,
    TokenEmbedding,
    TransformerLayer,
    AttentionPooling,
    RowPooling,
    DatasetPooling,
)

# === Reconstruction Heads ===
from lacuna.models.reconstruction import (
    # Config
    ReconstructionConfig,
    # Base class
    BaseReconstructionHead,
    # Head implementations
    MCARHead,
    MARHead,
    MNARSelfCensoringHead,
    MNARThresholdHead,
    MNARLatentHead,
    # Registry
    HEAD_REGISTRY,
    create_head,
    # Container
    ReconstructionHeads,
    create_reconstruction_heads,
)

# === Mixture of Experts ===
from lacuna.models.moe import (
    MoEConfig,
    GatingNetwork,
    ExpertHead,
    ExpertHeads,
    MixtureOfExperts,
    RowToDatasetAggregator,
    create_moe,
)

# === Complete Model ===
from lacuna.models.assembly import (
    LacunaModelConfig,
    BayesOptimalDecision,
    compute_entropy,
    LacunaModel,
    create_lacuna_model,
    create_lacuna_mini,
    create_lacuna_base,
    create_lacuna_large,
)


__all__ = [
    # === Encoder ===
    "LacunaEncoder",
    "EncoderConfig",
    "create_encoder",
    "TokenEmbedding",
    "TransformerLayer",
    "AttentionPooling",
    "RowPooling",
    "DatasetPooling",
    
    # === Reconstruction ===
    "ReconstructionConfig",
    "BaseReconstructionHead",
    "MCARHead",
    "MARHead",
    "MNARSelfCensoringHead",
    "MNARThresholdHead",
    "MNARLatentHead",
    "HEAD_REGISTRY",
    "create_head",
    "ReconstructionHeads",
    "create_reconstruction_heads",
    
    # === MoE ===
    "MoEConfig",
    "GatingNetwork",
    "ExpertHead",
    "ExpertHeads",
    "MixtureOfExperts",
    "RowToDatasetAggregator",
    "create_moe",
    
    # === Complete Model ===
    "LacunaModelConfig",
    "BayesOptimalDecision",
    "compute_entropy",
    "LacunaModel",
    "create_lacuna_model",
    "create_lacuna_mini",
    "create_lacuna_base",
    "create_lacuna_large",
]