"""Trajectories stored as NumPy ``.npz`` archives, the layout many MLIP and MD scripts dump.

Arrays read (``np.savez(path, positions=..., types=..., ...)``):

- ``positions``  (frames, atoms, 3) or (atoms, 3) in Angstrom (``coords`` and ``pos`` are read too)
- ``types``      (atoms,) element symbols or integers; or ``symbols``, or ``numbers`` (atomic numbers)
- ``cell``       (3, 3) or (frames, 3, 3), optional; lattice vectors as rows (``box``, ``lattice`` too)
- ``pbc``        (3,), optional; default periodic when there is a cell; ``origin`` (3,), optional
- any other (frames, atoms) or (frames, atoms, k) array becomes a particle property and any
  (frames,) array an attribute, e.g. ``forces`` -> "Force", ``energies`` -> attribute "Energy".

The writer stores the same layout, so ``export_file(pipe, "run.npz")`` reads back as it was.
"""
from __future__ import annotations

import os
import zipfile

import numpy as np

from ..data import Cell, DataCollection
from .base import FrameReader

_NAMES = {"forces": "Force", "velocities": "Velocity", "charges": "Charge", "energies": "Energy",
          "energy": "Energy", "stress": "Stress", "virial": "Virial", "masses": "Mass"}
_POSITIONS = ("positions", "coords", "pos")
_CELLS = ("cell", "box", "lattice")
_TYPES = ("types", "symbols", "numbers")
#: property names the writer stores under their usual .npz key
_KEYS = {v: k for k, v in _NAMES.items() if k not in ("energy",)}


def is_npz_trajectory(path) -> bool:
    """A zip archive (.npz) with a positions array."""
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
    except (OSError, zipfile.BadZipFile, ValueError):
        return False
    return any(f"{k}.npy" in names for k in _POSITIONS)


def _pick(arrays: dict, keys):
    for k in keys:
        if k in arrays:
            return k, arrays.pop(k)
    return None, None


def _name(key: str) -> str:
    return _NAMES.get(key, key[:1].upper() + key[1:])


class NpzTrajectoryReader(FrameReader):
    """A ``.npz`` archive with ``positions`` (frames, atoms, 3), ``types`` and optionally ``cell``;
    other per-atom arrays become properties, per-frame ones attributes. ``type_map`` renames types
    (``{1: "Cu"}`` for integer types). The archive is decompressed once, when it is opened."""

    def __init__(self, path, type_map=None):
        with np.load(os.fspath(path), allow_pickle=False) as archive:   # never unpickle (untrusted files)
            arrays = {key: archive[key] for key in archive.files}
        key, positions = _pick(arrays, _POSITIONS)
        if key is None:
            raise ValueError(f"{path}: no 'positions' array (found {sorted(arrays)})")
        positions = np.asarray(positions, float)
        if positions.ndim == 2:
            positions = positions[None]
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError(f"{path}: '{key}' must be (frames, atoms, 3), not {positions.shape}")
        self.positions = positions
        n_frames, n_atoms = positions.shape[:2]
        key, types = _pick(arrays, _TYPES)
        if types is None:
            types = np.full(n_atoms, "X")
        elif key == "numbers":
            from ase.data import chemical_symbols

            types = np.array([chemical_symbols[int(z)] for z in types])
        types = np.asarray(types)
        if types.ndim == 2:           # (frames, atoms): the types of the first frame
            types = types[0]
        type_map = {str(k): v for k, v in (type_map or {}).items()}
        self.types = np.array([type_map.get(str(t), t) for t in types.tolist()]).astype(str)
        _, cell = _pick(arrays, _CELLS)
        self.cell = None if cell is None else np.asarray(cell, float)
        if self.cell is not None and self.cell.shape not in ((3, 3), (n_frames, 3, 3)):
            raise ValueError(f"{path}: cell must be (3, 3) or (frames, 3, 3), not {self.cell.shape}")
        _, origin = _pick(arrays, ("origin",))
        self.origin = None if origin is None else np.asarray(origin, float).reshape(3)
        _, pbc = _pick(arrays, ("pbc",))
        self.pbc = None if pbc is None else np.asarray(pbc, bool).reshape(-1)[:3]
        self.per_atom = {_name(k): v for k, v in arrays.items() if v.ndim >= 2 and v.shape[:2] == (n_frames, n_atoms)}
        self.per_frame = {_name(k): v for k, v in arrays.items() if v.shape == (n_frames,)
                          and np.issubdtype(v.dtype, np.number)}

    def __len__(self):
        return len(self.positions)

    def read(self, index):
        data = DataCollection(self.positions[index], self.types)
        if self.cell is not None:
            matrix = self.cell[index] if self.cell.ndim == 3 else self.cell
            data.cell = Cell(matrix, origin=self.origin if self.origin is not None else np.zeros(3),
                             pbc=self.pbc if self.pbc is not None else (True, True, True))
        for name, values in self.per_atom.items():
            data.particles[name] = np.array(values[index])
        for name, values in self.per_frame.items():
            data.attributes[name] = values[index].item()
        return data


def write_npz_trajectory(path, frames, compressed: bool = True):
    """All frames as one ``.npz`` archive in the layout :class:`NpzTrajectoryReader` reads:
    positions, types, cell (one per frame when it changes), pbc, every numeric particle
    property found in every frame and every number attribute. The frames must share their
    atom count."""
    frames = list(frames)
    if not frames:
        raise ValueError("nothing to write")
    n = frames[0].count
    if any(d.count != n for d in frames):
        raise ValueError("npz trajectories need the same number of atoms in every frame")
    out = {"positions": np.stack([np.asarray(d.positions, float) for d in frames]),
           "types": np.asarray(frames[0].types).astype(str)}
    first = frames[0]
    if first.cell is not None and first.cell.is_valid:
        cells = np.stack([np.asarray(d.cell.matrix, float) for d in frames])
        out["cell"] = cells[0] if np.allclose(cells, cells[0]) else cells
        out["pbc"] = np.asarray(first.cell.pbc, bool)
        if np.any(first.cell.origin):
            out["origin"] = np.asarray(first.cell.origin, float)
    skip = {"Position", "Particle Type"}
    for name in first.particles.keys():
        if name in skip:
            continue
        try:
            values = [np.asarray(d.particles[name]) for d in frames]
        except KeyError:
            continue
        if all(np.issubdtype(v.dtype, np.number) or v.dtype == bool for v in values) and \
                len({v.shape for v in values}) == 1:
            out[_KEYS.get(name, name)] = np.stack(values)
    for name, value in first.attributes.items():
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            try:
                out[_KEYS.get(name, name)] = np.array([float(d.attributes[name]) for d in frames])
            except (KeyError, TypeError, ValueError):
                continue
    (np.savez_compressed if compressed else np.savez)(os.fspath(path), **out)
