"""
lacuna.generators.registry

Generator registry management.

Design: Registry holds a finite, validated set of generators.
"""

from typing import Dict, List, Tuple
import torch

from lacuna.core.exceptions import RegistryError
from lacuna.core.types import MCAR, MAR, MNAR
from .base import Generator


class GeneratorRegistry:
    """Manages finite set of generators.
    
    Validates:
    - Unique generator IDs
    - Sequential IDs from 0 to K-1
    - At least one generator
    
    Attributes:
        generators: Tuple of Generator objects.
        K: Number of generators.
    """
    
    def __init__(self, generators: Tuple[Generator, ...]):
        if len(generators) == 0:
            raise RegistryError("Registry must have at least one generator")
        
        # Validate unique IDs
        ids = [g.generator_id for g in generators]
        if len(ids) != len(set(ids)):
            duplicates = [i for i in ids if ids.count(i) > 1]
            raise RegistryError(f"Duplicate generator IDs: {set(duplicates)}")
        
        # Validate sequential IDs starting from 0
        if sorted(ids) != list(range(len(ids))):
            raise RegistryError(
                f"Generator IDs must be 0..{len(ids)-1}, got {sorted(ids)}"
            )
        
        # Store sorted by ID
        self._generators = tuple(sorted(generators, key=lambda g: g.generator_id))
        self._id_to_gen: Dict[int, Generator] = {g.generator_id: g for g in self._generators}
        self._name_to_gen: Dict[str, Generator] = {g.name: g for g in self._generators}
    
    @property
    def generators(self) -> Tuple[Generator, ...]:
        return self._generators
    
    @property
    def K(self) -> int:
        """Number of generators."""
        return len(self._generators)
    
    def __len__(self) -> int:
        return self.K
    
    def __getitem__(self, generator_id: int) -> Generator:
        """Get generator by ID."""
        if generator_id not in self._id_to_gen:
            raise RegistryError(f"Generator ID {generator_id} not found (K={self.K})")
        return self._id_to_gen[generator_id]
    
    def get_by_name(self, name: str) -> Generator:
        """Get generator by name."""
        if name not in self._name_to_gen:
            raise RegistryError(f"Generator '{name}' not found")
        return self._name_to_gen[name]
    
    def get_class_mapping(self) -> torch.Tensor:
        """Return tensor mapping generator_id -> class_id.
        
        Returns:
            [K] long tensor where entry i is class_id of generator i.
        """
        return torch.tensor([g.class_id for g in self._generators], dtype=torch.long)
    
    def generator_ids_for_class(self, class_id: int) -> List[int]:
        """Get all generator IDs belonging to a class.
        
        Args:
            class_id: MCAR (0), MAR (1), or MNAR (2).
        
        Returns:
            List of generator IDs with that class.
        """
        return [g.generator_id for g in self._generators if g.class_id == class_id]
    
    def class_counts(self) -> Dict[int, int]:
        """Return count of generators per class."""
        counts = {MCAR: 0, MAR: 0, MNAR: 0}
        for g in self._generators:
            counts[g.class_id] += 1
        return counts
    
    def __iter__(self):
        return iter(self._generators)
    
    def __repr__(self) -> str:
        counts = self.class_counts()
        return (
            f"GeneratorRegistry(K={self.K}, "
            f"MCAR={counts[MCAR]}, MAR={counts[MAR]}, MNAR={counts[MNAR]})"
        )
