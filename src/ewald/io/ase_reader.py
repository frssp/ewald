"""Fallback reader for every format ASE understands (POSCAR, OUTCAR, vasprun.xml, extxyz, CIF, ...)."""
from __future__ import annotations

import numpy as np

from ..data import DataCollection
from .base import FrameReader


class AseReader(FrameReader):
    def __init__(self, path, format: str | None = None, **kwargs):
        import ase.io

        self.path = str(path)
        # ASE re-parses the whole file for random access in most formats, so read once.
        images = ase.io.read(self.path, index=":", format=format, **kwargs)
        self._images = images if isinstance(images, list) else [images]
        if not self._images:
            raise ValueError(f"{path}: no structures found")

    def __len__(self):
        return len(self._images)

    def read(self, index: int) -> DataCollection:
        atoms = self._images[index]
        data = DataCollection.from_ase(atoms)
        data.attributes.setdefault("Timestep", index)
        _ovito_names(data, atoms)
        return data


def _ovito_names(data: DataCollection, atoms) -> None:
    """Map ASE conventions onto OVITO-style properties."""
    from ase import units

    if "Momentum" in data.particles:  # ASE stores momenta in its own units; expose velocity in A/fs
        data.particles["Velocity"] = atoms.get_velocities() * units.fs
        del data.particles["Momentum"]
    if atoms.constraints:  # VASP 'Selective dynamics': True = coordinate may move (VASP 'T')
        from ase.constraints import FixAtoms, FixScaled

        free = np.ones((len(atoms), 3), dtype=bool)
        for c in atoms.constraints:
            if isinstance(c, FixAtoms):
                free[c.index] = False
            elif isinstance(c, FixScaled):
                free[c.index] &= ~np.asarray(c.mask, dtype=bool)
        data.particles["Selective Dynamics"] = free
