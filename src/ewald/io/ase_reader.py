"""Fallback reader for every format ASE understands (POSCAR, OUTCAR, vasprun.xml, extxyz, CIF, lammps-data, ...)."""
from __future__ import annotations

from ..data import DataCollection
from .base import FrameReader


class AseReader(FrameReader):
    def __init__(self, path, format: str | None = None, **kwargs):
        import ase.io

        self.path = str(path)
        # ASE re-parses the whole file for random access in most formats, so read once.
        images = ase.io.read(self.path, index=":", format=format, **kwargs)
        self._images = images if isinstance(images, list) else [images]

    def __len__(self):
        return len(self._images)

    def read(self, index: int) -> DataCollection:
        data = DataCollection.from_ase(self._images[index])
        data.attributes.setdefault("Timestep", index)
        return data
