"""
Pytest configuration and shared fixtures for Lacuna tests.
"""

import pytest
import torch
from lacuna.core.rng import RNGState
from lacuna.config.schema import LacunaConfig


@pytest.fixture
def rng():
    """Provide seeded RNG for reproducible tests."""
    return RNGState(seed=42)


@pytest.fixture
def default_config():
    """Provide default LacunaConfig."""
    return LacunaConfig()


@pytest.fixture
def minimal_config():
    """Provide minimal LacunaConfig for fast tests."""
    return LacunaConfig.minimal()


@pytest.fixture
def sample_observed_dataset(rng):
    """Provide sample ObservedDataset."""
    from lacuna.core.types import ObservedDataset
    
    n, d = 100, 5
    x = rng.randn(n, d)
    r = rng.rand(n, d) > 0.2  # ~20% missing
    r[0, 0] = True  # Ensure at least one observed
    
    return ObservedDataset(
        x=x * r.float(),  # Zero out missing
        r=r,
        n=n,
        d=d,
        feature_names=tuple(f"col_{i}" for i in range(d)),
        dataset_id="test_dataset",
    )


@pytest.fixture
def minimal_registry():
    """Provide minimal 6-generator registry."""
    from lacuna.generators import create_minimal_registry
    return create_minimal_registry()
