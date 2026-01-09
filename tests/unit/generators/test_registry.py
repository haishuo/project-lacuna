"""
Tests for lacuna.generators.registry
"""

import pytest
import torch
from lacuna.core.exceptions import RegistryError
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.generators.registry import GeneratorRegistry
from lacuna.generators.params import GeneratorParams
from lacuna.generators.families.mcar import MCARUniform
from lacuna.generators.families.mar import MARLogistic
from lacuna.generators.families.mnar import MNARLogistic


def make_test_generators():
    """Create minimal test generator set."""
    return (
        MCARUniform(0, "mcar-0", GeneratorParams(miss_rate=0.1)),
        MCARUniform(1, "mcar-1", GeneratorParams(miss_rate=0.2)),
        MARLogistic(2, "mar-0", GeneratorParams(alpha0=0, alpha1=1.0)),
        MNARLogistic(3, "mnar-0", GeneratorParams(beta0=0, beta2=1.0)),
    )


class TestGeneratorRegistry:
    """Tests for GeneratorRegistry."""
    
    def test_construction(self):
        gens = make_test_generators()
        registry = GeneratorRegistry(gens)
        
        assert registry.K == 4
        assert len(registry) == 4
    
    def test_empty_raises(self):
        with pytest.raises(RegistryError, match="at least one"):
            GeneratorRegistry(())
    
    def test_duplicate_ids_raises(self):
        gens = (
            MCARUniform(0, "a", GeneratorParams(miss_rate=0.1)),
            MCARUniform(0, "b", GeneratorParams(miss_rate=0.2)),  # Duplicate ID
        )
        with pytest.raises(RegistryError, match="Duplicate"):
            GeneratorRegistry(gens)
    
    def test_non_sequential_ids_raises(self):
        gens = (
            MCARUniform(0, "a", GeneratorParams(miss_rate=0.1)),
            MCARUniform(2, "b", GeneratorParams(miss_rate=0.2)),  # Skipped 1
        )
        with pytest.raises(RegistryError, match="must be 0"):
            GeneratorRegistry(gens)
    
    def test_getitem(self):
        gens = make_test_generators()
        registry = GeneratorRegistry(gens)
        
        assert registry[0].name == "mcar-0"
        assert registry[2].name == "mar-0"
    
    def test_getitem_invalid_raises(self):
        registry = GeneratorRegistry(make_test_generators())
        
        with pytest.raises(RegistryError, match="not found"):
            _ = registry[99]
    
    def test_get_by_name(self):
        registry = GeneratorRegistry(make_test_generators())
        
        gen = registry.get_by_name("mar-0")
        assert gen.generator_id == 2
    
    def test_get_by_name_invalid_raises(self):
        registry = GeneratorRegistry(make_test_generators())
        
        with pytest.raises(RegistryError, match="not found"):
            registry.get_by_name("nonexistent")
    
    def test_get_class_mapping(self):
        registry = GeneratorRegistry(make_test_generators())
        
        mapping = registry.get_class_mapping()
        
        assert mapping.shape == (4,)
        assert mapping[0] == MCAR
        assert mapping[1] == MCAR
        assert mapping[2] == MAR
        assert mapping[3] == MNAR
    
    def test_generator_ids_for_class(self):
        registry = GeneratorRegistry(make_test_generators())
        
        mcar_ids = registry.generator_ids_for_class(MCAR)
        mar_ids = registry.generator_ids_for_class(MAR)
        mnar_ids = registry.generator_ids_for_class(MNAR)
        
        assert mcar_ids == [0, 1]
        assert mar_ids == [2]
        assert mnar_ids == [3]
    
    def test_class_counts(self):
        registry = GeneratorRegistry(make_test_generators())
        
        counts = registry.class_counts()
        
        assert counts[MCAR] == 2
        assert counts[MAR] == 1
        assert counts[MNAR] == 1
    
    def test_iteration(self):
        registry = GeneratorRegistry(make_test_generators())
        
        names = [g.name for g in registry]
        assert names == ["mcar-0", "mcar-1", "mar-0", "mnar-0"]
    
    def test_repr(self):
        registry = GeneratorRegistry(make_test_generators())
        r = repr(registry)
        
        assert "GeneratorRegistry" in r
        assert "K=4" in r
