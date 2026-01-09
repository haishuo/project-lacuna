"""
lacuna.generators.params

Parameter validation and frozen container.

Design: Parameters are immutable once created.
"""

from typing import Any, Dict, FrozenSet


class GeneratorParams:
    """Immutable parameter container for generators.
    
    Parameters are accessed via [] or get().
    Once created, parameters cannot be modified.
    
    Usage:
        params = GeneratorParams(alpha=1.0, beta=0.5)
        print(params["alpha"])  # 1.0
        print(params.get("gamma", 0.0))  # 0.0 (default)
    """
    
    def __init__(self, **kwargs):
        self._values: Dict[str, Any] = dict(kwargs)
        self._keys: FrozenSet[str] = frozenset(kwargs.keys())
    
    def __getitem__(self, key: str) -> Any:
        if key not in self._values:
            raise KeyError(f"Parameter '{key}' not found. Available: {sorted(self._keys)}")
        return self._values[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter with optional default."""
        return self._values.get(key, default)
    
    def __contains__(self, key: str) -> bool:
        return key in self._values
    
    @property
    def keys(self) -> FrozenSet[str]:
        """Return frozenset of parameter names."""
        return self._keys
    
    def to_dict(self) -> Dict[str, Any]:
        """Return copy of parameters as dict."""
        return dict(self._values)
    
    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self._values.items()))
        return f"GeneratorParams({items})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, GeneratorParams):
            return False
        return self._values == other._values
    
    def __hash__(self) -> int:
        return hash(tuple(sorted(self._values.items())))
