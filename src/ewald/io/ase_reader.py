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


def is_ase_db(path) -> bool:
    """SQLite file with ASE's ``systems`` table (``*.db``), or an ASE JSON database."""
    import sqlite3

    path = str(path)
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError:
        return False
    if head.startswith(b"SQLite format 3"):
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:  # closed explicitly: an open handle keeps the file locked on Windows
                return con.execute("SELECT 1 FROM sqlite_master WHERE name='systems'").fetchone() is not None
            finally:
                con.close()
        except sqlite3.Error:
            return False
    return path.lower().endswith(".json") and b'"ids"' in _head(path, 4096)


def _head(path, n):
    with open(path, "rb") as fh:
        return fh.read(n)


class AseDbReader(FrameReader):
    """ASE database (``ase db``: SQLite ``.db`` or ``.json``), one row per frame, lazy.

    ``selection`` is an ``ase db`` query (``"natoms<100,energy<-3"``, ``"Cu"``, ``"relaxed=1"``).
    Each row's key-value pairs, energy, ``id`` and ``unique_id`` become attributes.
    """

    def __init__(self, path, selection=None, **query):
        import ase.db

        self.path = str(path)
        self.db = ase.db.connect(self.path)
        self.ids = [row.id for row in self.db.select(selection, include_data=False, **query)]
        if not self.ids:
            raise ValueError(f"{path}: no rows match {selection or query or 'the query'}")

    def __len__(self):
        return len(self.ids)

    def read(self, index: int) -> DataCollection:
        row = self.db.get(id=self.ids[index])
        atoms = row.toatoms()
        data = DataCollection.from_ase(atoms)
        _ovito_names(data, atoms)
        data.attributes.update({k: v for k, v in row.key_value_pairs.items()})
        data.attributes["Row Id"] = row.id
        if "unique_id" in row:
            data.attributes["Unique Id"] = row.unique_id
        data.attributes.setdefault("Timestep", index)
        return data


class AseTrajectoryReader(FrameReader):
    """ASE binary trajectory (``*.traj``, written by ASE optimizers and MD), lazy."""

    def __init__(self, path):
        from ase.io.trajectory import Trajectory

        self.path = str(path)
        self.traj = Trajectory(self.path, "r")
        if len(self.traj) == 0:
            raise ValueError(f"{path}: empty trajectory")

    def __len__(self):
        return len(self.traj)

    def read(self, index: int) -> DataCollection:
        atoms = self.traj[index]
        data = DataCollection.from_ase(atoms)
        _ovito_names(data, atoms)
        data.attributes.setdefault("Timestep", index)
        return data


def is_ase_traj(path) -> bool:
    return str(path).lower().endswith(".traj") and _head(path, 24).startswith(b"- of Ulm")
