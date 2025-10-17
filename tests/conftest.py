"""Pytest configuration and shared fixtures"""

import pytest
from lacuna.config import LacunaConfig


@pytest.fixture
def default_config():
    """Provide default Lacuna configuration"""
    return LacunaConfig.default()


@pytest.fixture
def sample_data():
    """Provide sample data for testing"""
    import pandas as pd
    import numpy as np
    
    # Create simple test data
    data = pd.DataFrame({
        'age': [25, 30, np.nan, 40, 45],
        'income': [50000, np.nan, 60000, 70000, np.nan],
        'education': ['HS', 'BS', 'MS', np.nan, 'PhD']
    })
    return data
