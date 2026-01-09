"""
Tests for lacuna.config.load
"""

import pytest
import tempfile
from pathlib import Path

from lacuna.config.schema import LacunaConfig
from lacuna.config.load import (
    load_config,
    save_config,
    config_from_dict,
    config_to_dict,
)
from lacuna.core.exceptions import ConfigError


class TestConfigFromDict:
    """Tests for config_from_dict."""
    
    def test_empty_dict_uses_defaults(self):
        cfg = config_from_dict({})
        assert cfg.seed == 42
    
    def test_partial_override(self):
        cfg = config_from_dict({"seed": 123})
        assert cfg.seed == 123
        assert cfg.device == "cuda"  # Default


class TestConfigToDict:
    """Tests for config_to_dict."""
    
    def test_roundtrip(self):
        cfg1 = LacunaConfig.minimal()
        d = config_to_dict(cfg1)
        cfg2 = config_from_dict(d)
        
        assert cfg1.seed == cfg2.seed
        assert cfg1.data.max_cols == cfg2.data.max_cols
        assert cfg1.model.hidden_dim == cfg2.model.hidden_dim


class TestLoadSaveConfig:
    """Tests for load_config and save_config."""
    
    def test_save_and_load(self):
        cfg1 = LacunaConfig.minimal()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            save_config(cfg1, path)
            cfg2 = load_config(path)
        
        assert cfg1.seed == cfg2.seed
        assert cfg1.data.max_rows == cfg2.data.max_rows
    
    def test_load_nonexistent_raises(self):
        with pytest.raises(ConfigError):
            load_config("/nonexistent/path.yaml")
