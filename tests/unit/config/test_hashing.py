"""
Tests for lacuna.config.hashing

Verify deterministic hashing.
"""

import pytest
from lacuna.config.hashing import hash_config, hash_dict, config_signature
from lacuna.config.schema import LacunaConfig


class TestHashConfig:
    """Tests for hash_config."""
    
    def test_deterministic(self):
        cfg = LacunaConfig()
        h1 = hash_config(cfg)
        h2 = hash_config(cfg)
        assert h1 == h2
    
    def test_different_configs_different_hash(self):
        cfg1 = LacunaConfig(seed=1)
        cfg2 = LacunaConfig(seed=2)
        
        h1 = hash_config(cfg1)
        h2 = hash_config(cfg2)
        
        assert h1 != h2
    
    def test_hash_length(self):
        cfg = LacunaConfig()
        h = hash_config(cfg)
        assert len(h) == 16


class TestHashDict:
    """Tests for hash_dict."""
    
    def test_order_independent(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        
        assert hash_dict(d1) == hash_dict(d2)


class TestConfigSignature:
    """Tests for config_signature."""
    
    def test_format(self):
        cfg = LacunaConfig()
        sig = config_signature(cfg)
        
        # Should contain model info and hash
        assert "128d" in sig  # hidden_dim
        assert "4L" in sig    # n_layers
        assert "6G" in sig    # n_generators
    
    def test_different_configs_different_signatures(self):
        cfg1 = LacunaConfig.minimal()
        cfg2 = LacunaConfig()
        
        sig1 = config_signature(cfg1)
        sig2 = config_signature(cfg2)
        
        assert sig1 != sig2
