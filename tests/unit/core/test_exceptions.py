"""
Tests for lacuna.core.exceptions

Verify exception hierarchy and basic behavior.
"""

import pytest
from lacuna.core.exceptions import (
    LacunaError,
    ValidationError,
    ConfigError,
    RegistryError,
    CheckpointError,
    NumericalError,
)


class TestExceptionHierarchy:
    """All exceptions inherit from LacunaError."""
    
    def test_validation_error_is_lacuna_error(self):
        assert issubclass(ValidationError, LacunaError)
    
    def test_config_error_is_lacuna_error(self):
        assert issubclass(ConfigError, LacunaError)
    
    def test_registry_error_is_lacuna_error(self):
        assert issubclass(RegistryError, LacunaError)
    
    def test_checkpoint_error_is_lacuna_error(self):
        assert issubclass(CheckpointError, LacunaError)
    
    def test_numerical_error_is_lacuna_error(self):
        assert issubclass(NumericalError, LacunaError)


class TestExceptionRaising:
    """Exceptions can be raised and caught."""
    
    def test_raise_validation_error(self):
        with pytest.raises(ValidationError, match="test message"):
            raise ValidationError("test message")
    
    def test_catch_as_lacuna_error(self):
        with pytest.raises(LacunaError):
            raise ValidationError("caught as base")
    
    def test_catch_as_exception(self):
        with pytest.raises(Exception):
            raise NumericalError("caught as Exception")
