"""MDTraj HDF5 trajectories (``*.h5``), the format of OpenMM's ``HDF5Reporter`` and ``mdtraj.save_hdf5``.

Needs ``h5py``. The layout (MDTraj's "Pande" convention 1.1) is ``/coordinates``
(frames x atoms x 3, nm), ``/cell_lengths`` (nm) and ``/cell_angles`` (degrees),
optional ``/velocities`` (nm/ps), ``/time`` (ps), ``/kineticEnergy``,
``/potentialEnergy`` (kJ/mol), ``/temperature`` (K) and ``/topology``, a JSON string of
chains -> residues -> atoms with elements, plus bonds. Frames are read on demand.
Positions become Angstrom, velocities A/fs and energies eV, as in the rest of ewald;
"Particle Type" is the element (the atom name when it has none), "Molecule Identifier"
the residue index + 1, and topology bonds go into ``data.bonds``.
"""
from __future__ import annotations

import json

import numpy as np

from ..data import Cell, DataCollection
from .base import FrameReader
from .h5md import open_h5

_NM = 10.0
_KJMOL_TO_EV = 0.010364269656262175
_UNITS = {"nanometers": 10.0, "angstroms": 1.0, "nanometers/picosecond": 0.01, "angstroms/picosecond": 0.001,
          "kilojoules_per_mole": _KJMOL_TO_EV, "kilocalories_per_mole": 4.184 * _KJMOL_TO_EV}
_ATTRS = {"time": "Time", "kineticEnergy": "Kinetic Energy", "potentialEnergy": "Potential Energy",
          "temperature": "Temperature", "lambda": "Lambda"}


def is_mdtraj_h5(path) -> bool:
    try:
        with open(path, "rb") as fh:
            if fh.read(8) != b"\x89HDF\r\n\x1a\n":
                return False
        with open_h5(path) as f:
            conv = f.attrs.get("conventions", b"")
            conv = conv.decode() if isinstance(conv, bytes) else str(conv)
            return "coordinates" in f and (conv == "Pande" or "topology" in f)
    except (OSError, ValueError, ImportError):
        return False


def _factor(ds, default: float) -> float:
    unit = ds.attrs.get("units")
    if unit is None:
        return default
    unit = unit.decode() if isinstance(unit, bytes) else str(unit)
    return _UNITS.get(unit, default if unit not in ("degrees", "picoseconds", "kelvin", "dimensionless") else 1.0)


def cell_from_parameters(lengths, angles) -> np.ndarray:
    """Lattice vectors (rows, a along x, b in the xy plane) from a, b, c and alpha, beta, gamma in degrees."""
    a, b, c = lengths
    al, be, ga = np.radians(angles)
    bx, by = b * np.cos(ga), b * np.sin(ga)
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    m = np.array([[a, 0, 0], [bx, by, 0], [cx, cy, cz]])
    m[np.abs(m) < 1e-10 * max(a, b, c)] = 0.0
    return m


class MdtrajH5Reader(FrameReader):
    """MDTraj / OpenMM HDF5 trajectory, lazy."""

    def __init__(self, path):
        self.path = str(path)
        with open_h5(self.path) as f:
            if "coordinates" not in f:
                raise ValueError(f"{path}: no /coordinates; not an MDTraj HDF5 file?")
            self._n, self.natoms = f["coordinates"].shape[:2]
            self.types = np.array([f"X{i}" for i in range(self.natoms)])
            self.molecules = None
            self.bond_topology = None
            if "topology" in f:
                self._topology(f["topology"][()])

    def _topology(self, raw):
        raw = raw[0] if np.ndim(raw) else raw
        top = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
        types, mols = [""] * self.natoms, np.zeros(self.natoms, np.int64)
        for chain in top.get("chains", []):
            for res in chain.get("residues", []):
                for atom in res.get("atoms", []):
                    i = int(atom["index"])
                    if i < self.natoms:
                        types[i] = atom.get("element") or atom.get("name") or "X"
                        mols[i] = int(res.get("index", 0)) + 1
        self.types = np.array([t or "X" for t in types])
        self.molecules = mols
        bonds = np.asarray([b[:2] for b in top.get("bonds", [])], np.int64).reshape(-1, 2)
        self.bond_topology = bonds if len(bonds) else None

    def __len__(self):
        return self._n

    def read(self, index: int) -> DataCollection:
        index = range(self._n)[index]
        with open_h5(self.path) as f:
            ds = f["coordinates"]
            pos = np.asarray(ds[index], float) * _factor(ds, _NM)
            cell = Cell()
            if "cell_lengths" in f and "cell_angles" in f:
                lengths = np.asarray(f["cell_lengths"][index], float) * _factor(f["cell_lengths"], _NM)
                if np.all(lengths > 0):
                    cell = Cell(cell_from_parameters(lengths, np.asarray(f["cell_angles"][index], float)),
                                np.zeros(3), np.ones(3, bool))
            data = DataCollection(pos, self.types, cell)
            data.particles["Particle Identifier"] = np.arange(1, self.natoms + 1)
            if self.molecules is not None:
                data.particles["Molecule Identifier"] = self.molecules
            if "velocities" in f:
                v = f["velocities"]
                data.particles["Velocity"] = np.asarray(v[index], float) * _factor(v, 0.01)  # nm/ps -> A/fs
            for key, name in _ATTRS.items():
                if key in f and f[key].shape and f[key].shape[0] == self._n:
                    default = _KJMOL_TO_EV if key.endswith("Energy") else 1.0
                    data.attributes[name] = float(f[key][index]) * _factor(f[key], default)
        if self.bond_topology is not None:
            from ..bonds import Bonds, minimum_images

            t = self.bond_topology
            data.bonds = Bonds(t, minimum_images(data, t[:, 0], t[:, 1]))
        data.attributes["Timestep"] = index
        return data
