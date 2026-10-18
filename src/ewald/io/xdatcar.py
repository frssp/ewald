"""Native lazy reader for VASP XDATCAR (fixed- and variable-cell)."""
from __future__ import annotations

import numpy as np

from ..data import Cell, DataCollection
from .base import FrameReader


def is_xdatcar(path) -> bool:
    import os

    return "XDATCAR" in os.path.basename(str(path)).upper()


class XdatcarReader(FrameReader):
    def __init__(self, path):
        self.path = str(path)
        self._frames: list[tuple[int, int]] = []  # (header offset, configuration-line offset)
        with open(self.path, "rb") as fh:
            header, natoms = 0, None
            while True:
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                if line.strip().lower().startswith((b"direct", b"cartesian")):
                    if natoms is None:
                        raise ValueError(f"{path}: configuration before header")
                    self._frames.append((header, offset))
                    for _ in range(natoms):
                        fh.readline()
                else:  # a (possibly repeated, variable-cell) 7-line header starts here
                    header = offset
                    for _ in range(6):
                        last = fh.readline()
                    natoms = sum(int(c) for c in last.split())
        if not self._frames:
            raise ValueError(f"{path}: no configurations found")

    def _read_header(self, offset):
        with open(self.path, "rb") as fh:
            fh.seek(offset)
            lines = [fh.readline().decode() for _ in range(7)]
        scale = float(lines[1].split()[0])
        lattice = np.array([l.split()[:3] for l in lines[2:5]], dtype=float)
        if scale < 0:  # negative scale = target volume
            lattice *= (-scale / abs(np.linalg.det(lattice))) ** (1 / 3)
        else:
            lattice *= scale
        species = lines[5].split()
        counts = [int(c) for c in lines[6].split()]
        return lattice, species, counts

    def __len__(self):
        return len(self._frames)

    def read(self, index: int) -> DataCollection:
        header, cfg = self._frames[index]
        lattice, species, counts = self._read_header(header)
        n = sum(counts)
        with open(self.path, "rb") as fh:
            fh.seek(cfg)
            kind = fh.readline().strip().lower()
            coords = np.array(b" ".join(fh.readline() for _ in range(n)).split()[: 3 * n], dtype=float)
        coords = coords.reshape(n, 3)
        cell = Cell(lattice, np.zeros(3), np.ones(3, bool))
        pos = cell.to_cartesian(coords) if kind.startswith(b"direct") else coords
        types = np.repeat(species, counts)
        data = DataCollection(pos, types, cell)
        data.particles["Particle Identifier"] = np.arange(1, n + 1)
        try:
            data.attributes["Timestep"] = int(kind.split(b"=")[-1])
        except ValueError:
            data.attributes["Timestep"] = index
        return data
