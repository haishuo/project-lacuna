"""
lacuna.config.hashing

Deterministic config hashing for reproducibility tracking.
"""

import hashlib
import json
from typing import Any, Dict

from .schema import LacunaConfig
from .load import config_to_dict


def hash_config(config: LacunaConfig) -> str:
    """Compute deterministic hash of configuration.
    
    Returns:
        16-character hex string.
    """
    d = config_to_dict(config)
    return hash_dict(d)


def hash_dict(d: Dict[str, Any]) -> str:
    """Compute deterministic hash of dictionary.
    
    Keys are sorted for determinism.
    """
    # Convert to JSON with sorted keys for determinism
    json_str = json.dumps(d, sort_keys=True, separators=(",", ":"))
    
    # SHA256 hash, truncated to 16 chars
    h = hashlib.sha256(json_str.encode()).hexdigest()[:16]
    return h


def config_signature(config: LacunaConfig) -> str:
    """Generate human-readable signature for config.
    
    Format: {model_dim}d_{n_layers}L_{n_generators}G_{hash}
    
    Example: "128d_4L_6G_a1b2c3d4e5f6g7h8"
    """
    h = hash_config(config)
    return (
        f"{config.model.hidden_dim}d_"
        f"{config.model.n_layers}L_"
        f"{config.generator.n_generators}G_"
        f"{h}"
    )
