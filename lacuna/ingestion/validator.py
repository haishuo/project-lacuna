"""
lacuna.ingestion.validator

Purpose: Validate inputs - fail fast and loud

Design Principles:
- UNIX Philosophy: Do ONE thing well
- No defaults (except top-level config)
- Trust neighbors (no redundant validation)
- Fail fast and loud
- Target: <200 lines

Spec Reference: Section 4.2
"""

import pandas as pd
from typing import Dict, Any


class Validator:
    """Validates input data - entry point validation only
    
    Design: Validate ONCE, downstream trusts
    """
    
    def __init__(self, config):
        """
        Args:
            config: ValidatorConfig (required fields)
        """
        if config.valid_domains is None:
            raise ValueError("config.valid_domains required")
        self.config = config
    
    def validate(self, data: pd.DataFrame, metadata: Dict) -> Dict[str, Any]:
        """Validate inputs
        
        Args:
            data: Input DataFrame
            metadata: Must contain 'domain' key
        
        Returns:
            Dict with validated data and metadata
        
        Raises:
            ValueError: If validation fails (fail fast!)
        """
        # Check data
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pd.DataFrame")
        
        if data.empty:
            raise ValueError("data cannot be empty")
        
        if data.shape[0] < self.config.min_rows:
            raise ValueError(f"Need {self.config.min_rows} rows, got {data.shape[0]}")
        
        if data.shape[1] < self.config.min_cols:
            raise ValueError(f"Need {self.config.min_cols} cols, got {data.shape[1]}")
        
        if not data.isnull().any().any():
            raise ValueError("No missing data found")
        
        # Check metadata
        if 'domain' not in metadata:
            raise ValueError("metadata must contain 'domain' key")
        
        if metadata['domain'] not in self.config.valid_domains:
            raise ValueError(f"Invalid domain: {metadata['domain']}")
        
        # TODO: Infer feature types if not provided
        # See spec section 4.2
        
        return {
            'data': data,
            'missing_mask': data.isnull(),
            'feature_types': {},  # TODO: Implement
            'n_rows': data.shape[0],
            'n_cols': data.shape[1],
            'missing_pct': data.isnull().sum().sum() / data.size
        }

