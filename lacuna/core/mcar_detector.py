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
import pymvnmle as pmle
from typing import Dict, Any


class MCARDetector:
    """Tests for MCAR using PyMVNMLE's Little's test
    
    Design: Trust validator already checked inputs
    """
    
    def __init__(self, config):
        """
        Args:
            config: MCARConfig with alpha (significance level, required)
        
        Raises:
            ValueError: If required config missing
        """
        if config.alpha is None:
            raise ValueError("config.alpha is required")
        self.config = config
    
    def test(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test for MCAR using PyMVNMLE's little_mcar_test
        
        Args:
            data: DataFrame (trust validator checked it)
        
        Returns:
            Dict with test results and ML estimates
        """
        # Trust data is valid - just call PyMVNMLE
        result = pmle.little_mcar_test(
            data,
            alpha=self.config.alpha,
            verbose=False  # Lacuna handles its own logging
        )
        
        return {
            'test_statistic': result.statistic,
            'p_value': result.p_value,
            'degrees_of_freedom': result.df,
            'is_plausible': not result.rejected,  # MCAR plausible if NOT rejected
            'confidence': self._compute_confidence(result.p_value, data.shape[0]),
            'method': 'little_mcar_test',
            'n_patterns': result.n_patterns,
            'n_patterns_used': result.n_patterns_used,
            'ml_mean': result.ml_mean,  # May be useful downstream
            'ml_cov': result.ml_cov,    # May be useful downstream
            'warnings': result.convergence_warnings
        }
    
    def _compute_confidence(self, pval: float, n: int) -> float:
        """Convert p-value to confidence score
        
        Args:
            pval: P-value from Little's test
            n: Sample size
        
        Returns:
            Confidence score in [0, 1]
        """
        # Higher p-value → higher confidence in MCAR
        # Account for sample size (larger n → more reliable test)
        confidence = pval * min(1.0, np.sqrt(n / 100))
        return float(np.clip(confidence, 0, 1))
