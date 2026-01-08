"""
Tests for lacuna.config.load

Verify config loading/saving.
"""

import pytest
import tempfile
from pathlib import Path
from lacuna.config.load import (
    load_config,
    save_config,
    config_from_dict,
    config_to_dict,
)
from lacuna.config.schema import LacunaConfig
from lacuna.core.exceptions import ConfigError


class TestConfigFromDict:
    """Tests for config_from_dict."""
    
    def test_empty_dict_uses_defaults(self):
        cfg = config_from_dict({})
        assert cfg.seed == 42
        assert cfg.model.hidden_dim == 128
    
    def test_partial_override(self):
        cfg = config_from_dict({
            "seed": 123,
            "model": {"hidden_dim": 256},
        })
        assert cfg.seed == 123
        assert cfg.model.hidden_dim == 256
        assert cfg.model.n_layers == 4  # Default preserved


class TestConfigToDict:
    """Tests for config_to_dict."""
    
    def test_roundtrip(self):
        cfg1 = LacunaConfig(seed=999)
        d = config_to_dict(cfg1)
        cfg2 = config_from_dict(d)
        
        assert cfg2.seed == 999
        assert cfg2.model.hidden_dim == cfg1.model.hidden_dim


class TestLoadSaveConfig:
    """Tests for load_config and save_config."""
    
    def test_save_and_load(self):
        cfg1 = LacunaConfig.minimal()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            save_config(cfg1, path)
            cfg2 = load_config(path)
        
        assert cfg2.seed == cfg1.seed
        assert cfg2.model.hidden_dim == cfg1.model.hidden_dim
    
    def test_load_nonexistent_raises(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config("/nonexistent/path/config.yaml")
