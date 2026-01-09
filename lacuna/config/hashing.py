"""
lacuna.config.hashing

Config hashing for reproducibility tracking.
"""

import hashlib
import json
from typing import Any, Dict

from .schema import LacunaConfig
from .load import config_to_dict


def hash_dict(d: Dict[str, Any]) -> str:
    """Hash a dictionary deterministically."""
    # Sort keys and convert to JSON for deterministic string
    json_str = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


def hash_config(config: LacunaConfig) -> str:
    """Hash a LacunaConfig for reproducibility tracking."""
    d = config_to_dict(config)
    return hash_dict(d)


def config_signature(config: LacunaConfig) -> str:
    """Generate short signature for config (first 8 chars of hash)."""
    return hash_config(config)[:8]
