"""
lacuna.core.exceptions

All custom exceptions for Lacuna.

Design: Fail fast and loud with informative errors.
"""


class LacunaError(Exception):
    """Base exception for all Lacuna errors."""
    pass


class ValidationError(LacunaError):
    """Input validation failed.
    
    Raised when data or parameters fail boundary checks.
    """
    pass


class ConfigError(LacunaError):
    """Configuration invalid or missing.
    
    Raised when config files are malformed or required fields are absent.
    """
    pass


class RegistryError(LacunaError):
    """Generator registry error.
    
    Raised when generator IDs are invalid, duplicated, or not found.
    """
    pass


class CheckpointError(LacunaError):
    """Checkpoint loading/saving error.
    
    Raised when model checkpoints are corrupted or incompatible.
    """
    pass


class NumericalError(LacunaError):
    """NaN/Inf or other numerical issue.
    
    Raised when computations produce invalid numerical results.
    """
    pass
