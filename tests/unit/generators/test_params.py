"""
Tests for lacuna.generators.params
"""

import pytest
from lacuna.generators.params import GeneratorParams


class TestGeneratorParams:
    """Tests for GeneratorParams."""
    
    def test_construction(self):
        params = GeneratorParams(alpha=1.0, beta=0.5, name="test")
        assert params["alpha"] == 1.0
        assert params["beta"] == 0.5
        assert params["name"] == "test"
    
    def test_get_with_default(self):
        params = GeneratorParams(alpha=1.0)
        assert params.get("alpha") == 1.0
        assert params.get("missing") is None
        assert params.get("missing", 42) == 42
    
    def test_contains(self):
        params = GeneratorParams(alpha=1.0)
        assert "alpha" in params
        assert "beta" not in params
    
    def test_missing_key_raises(self):
        params = GeneratorParams(alpha=1.0)
        with pytest.raises(KeyError, match="beta"):
            _ = params["beta"]
    
    def test_keys_property(self):
        params = GeneratorParams(a=1, b=2, c=3)
        assert params.keys == frozenset(["a", "b", "c"])
    
    def test_to_dict(self):
        params = GeneratorParams(alpha=1.0, beta=0.5)
        d = params.to_dict()
        assert d == {"alpha": 1.0, "beta": 0.5}
        
        # Verify it's a copy
        d["alpha"] = 999
        assert params["alpha"] == 1.0
    
    def test_equality(self):
        p1 = GeneratorParams(a=1, b=2)
        p2 = GeneratorParams(a=1, b=2)
        p3 = GeneratorParams(a=1, b=3)
        
        assert p1 == p2
        assert p1 != p3
    
    def test_hashable(self):
        p1 = GeneratorParams(a=1, b=2)
        p2 = GeneratorParams(a=1, b=2)
        
        # Should be usable as dict key
        d = {p1: "value"}
        assert d[p2] == "value"
    
    def test_repr(self):
        params = GeneratorParams(alpha=1.0)
        r = repr(params)
        assert "GeneratorParams" in r
        assert "alpha" in r
