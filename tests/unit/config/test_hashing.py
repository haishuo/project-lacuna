"""
Tests for lacuna.config.hashing
"""

import pytest
from lacuna.config.schema import LacunaConfig
from lacuna.config.hashing import hash_config, hash_dict, config_signature


class TestHashDict:
    """Tests for hash_dict."""
    
    def test_deterministic(self):
        d = {"a": 1, "b": 2}
        h1 = hash_dict(d)
        h2 = hash_dict(d)
        assert h1 == h2
    
    def test_different_dicts_different_hash(self):
        d1 = {"a": 1}
        d2 = {"a": 2}
        assert hash_dict(d1) != hash_dict(d2)


class TestHashConfig:
    """Tests for hash_config."""
    
    def test_deterministic(self):
        cfg = LacunaConfig.minimal()
        h1 = hash_config(cfg)
        h2 = hash_config(cfg)
        assert h1 == h2
    
    def test_different_configs_different_hash(self):
        cfg1 = LacunaConfig(seed=42)
        cfg2 = LacunaConfig(seed=43)
        assert hash_config(cfg1) != hash_config(cfg2)
    
    def test_hash_length(self):
        cfg = LacunaConfig()
        h = hash_config(cfg)
        assert len(h) == 64  # SHA256


class TestConfigSignature:
    """Tests for config_signature."""
    
    def test_format(self):
        cfg = LacunaConfig()
        sig = config_signature(cfg)
        assert len(sig) == 8
    
    def test_different_configs_different_signatures(self):
        cfg1 = LacunaConfig(seed=1)
        cfg2 = LacunaConfig(seed=2)
        assert config_signature(cfg1) != config_signature(cfg2)
