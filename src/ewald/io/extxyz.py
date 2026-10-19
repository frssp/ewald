"""Native lazy reader and streaming writer for XYZ and extended XYZ (extxyz) trajectories.

Indexing reads only the atom-count line of each frame and skips the rest, so opening a
multi-gigabyte extxyz from an MLIP training set or an MD run is fast; a frame is parsed
on demand by ASE's extxyz parser (``Lattice=``, ``Properties=``, ``pbc=``, per-frame
info, forces, energies). Plain XYZ is the same format without the comment-line keys.
"""
from __future__ import annotations

import io
import os
import warnings

import numpy as np

from ..data import DataCollection
from .ase_reader import _ovito_names
from .base import FrameReader, head_lines, open_binary, strip_compression


def is_extxyz(path) -> bool:
    if not strip_compression(path).lower().endswith((".xyz", ".extxyz")):
        return False
    head = head_lines(path, 1)
    return bool(head) and head[0].strip().isdigit()


class ExtxyzReader(FrameReader):
    """XYZ / extended XYZ, lazy (one pass over atom-count lines to index frames)."""

    def __init__(self, path):
        self.path = str(path)
        self._frames: list[tuple[int, int]] = []  # (byte offset, byte length)
        with open_binary(self.path) as fh:
            pos = 0
            while True:
                line = fh.readline()
                if not line:
                    break
                if not line.strip():  # blank lines between frames or at the end
                    pos += len(line)
                    continue
                try:
                    n = int(line.split()[0])
                except ValueError:
                    raise ValueError(f"{path}: expected an atom count at byte {pos}, got "
                                     f"{line[:60].decode(errors='replace')!r}") from None
                size = len(line)
                lines = 0
                for _ in range(n + 1):
                    body = fh.readline()
                    if not body:
                        break
                    size += len(body)
                    lines += 1
                if lines < n + 1:
                    warnings.warn(f"{path}: last frame is incomplete and was skipped", stacklevel=2)
                    break
                self._frames.append((pos, size))
                pos += size
        if not self._frames:
            raise ValueError(f"{path}: no frames found")

    def __len__(self):
        return len(self._frames)

    def read(self, index: int) -> DataCollection:
        from ase.io.extxyz import read_xyz

        offset, size = self._frames[index]
        with open_binary(self.path) as fh:
            fh.seek(offset)
            text = fh.read(size).decode()
        atoms = next(read_xyz(io.StringIO(text), 0))
        data = DataCollection.from_ase(atoms)
        _ovito_names(data, atoms)
        data.attributes.setdefault("Timestep", index)
        return data


def write_extxyz(path, frames, columns=None, **kwargs):
    """Stream ``frames`` (an iterable of DataCollection) to an extxyz file, one frame at a time.

    Every numeric particle property is written (``Force`` as ASE's ``forces``) unless
    ``columns`` names the ones to keep; scalar attributes go into the comment line.
    """
    from ase.io.extxyz import write_xyz

    with open(os.fspath(path), "w") as fh:
        for data in frames:
            atoms = _to_atoms(data, columns)
            write_xyz(fh, [atoms], **kwargs)


def _to_atoms(data: DataCollection, columns=None):
    from ase.calculators.singlepoint import SinglePointCalculator

    if columns is not None:
        keep = {"Position", "Particle Type", *columns}
        data = data.copy()
        data.particles = {k: v for k, v in data.particles.items() if k in keep}
    atoms = data.to_ase()
    results = {}
    if "Force" in atoms.arrays:
        results["forces"] = atoms.arrays.pop("Force")
    for key, value in data.attributes.items():
        if key == "Energy":
            results["energy"] = float(value)
        elif key == "Stress":
            results["stress"] = np.asarray(value)
        elif np.isscalar(value) and key not in ("SourceFrame",):
            atoms.info[key] = value
    if results:
        atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms
