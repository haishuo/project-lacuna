"""
lacuna.core.mcar_detector

Purpose: MCAR detection using PyMVNMLE (ONE thing)

Design Principles:
- UNIX Philosophy: Do ONE thing well
- No defaults (except top-level config)
- Trust neighbors (no redundant validation)
- Fail fast and loud
- Target: <200 lines

Spec Reference: Section 4.3
"""

import numpy as np
import pandas as pd
from pymvnmle.testing import mcar_test  # Reuse our other project
from typing import Dict, Any


class MCARDetector:
    """Tests for MCAR using PyMVNMLE's implementation
    
    Design: Trust validator already checked inputs
    """
    
    def __init__(self, config):
        """
        Args:
            config: MCARConfig with alpha and method (required)
        
        Raises:
            ValueError: If required config missing
        """
        if config.alpha is None:
            raise ValueError("config.alpha is required")
        if config.method is None:
            raise ValueError("config.method is required")
        self.config = config
    
    def test(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test for MCAR
        
        Args:
            data: DataFrame (trust validator checked it)
        
        Returns:
            Dict with test results
        """
        # Trust data is valid - just call PyMVNMLE
        result = mcar_test(
            data=data,
            method=self.config.method,
            alpha=self.config.alpha
        )
        
        return {
            'test_statistic': result.statistic,
            'p_value': result.pvalue,
            'is_plausible': result.pvalue > self.config.alpha,
            'confidence': self._compute_confidence(result.pvalue, data.shape[0]),
            'method': self.config.method
        }
    
    def _compute_confidence(self, pval: float, n: int) -> float:
        """Convert p-value to confidence score"""
        # TODO: Implement confidence calculation
        # Higher p-value → higher confidence in MCAR
        # Account for sample size
        confidence = pval * min(1.0, np.sqrt(n / 100))
        return float(np.clip(confidence, 0, 1))

