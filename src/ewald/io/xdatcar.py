"""Native lazy reader for VASP XDATCAR (fixed- and variable-cell).

Handles VASP 5/6 headers (including VASP 6.4 ``Si_GW/7a4c...`` species labels),
negative scale factors (target volume), Direct and Cartesian configurations,
compressed files and a truncated last configuration from a running/killed job.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from ..data import Cell, DataCollection
from .base import FrameReader, file_size, floats, head_lines, open_binary, strip_compression


def is_xdatcar(path) -> bool:
    if "XDATCAR" in os.path.basename(strip_compression(path)).upper():
        return True
    head = head_lines(path, 8)
    return len(head) == 8 and b"configuration" in head[7].lower() and \
        head[7].strip().lower().startswith((b"direct", b"cartesian"))


def species_name(label: str) -> str:
    """'Fe_pv/2c5c46b5' -> 'Fe', 'Si_GW' -> 'Si', 'H.75' -> 'H'."""
    base = label.split("/")[0].split("_")[0]
    letters = "".join(c for c in base if c.isalpha())
    return letters[:1].upper() + letters[1:2].lower() if letters else label


def _xyz(line: bytes) -> list:
    """Three coordinates, also when VASP runs the columns together (``-0.63265286-0.11227753``)."""
    xyz = line.split()[:3]
    return xyz if len(xyz) == 3 else floats(line)[:3]


def _has_xyz(line: bytes) -> bool:
    return len(_xyz(line)) == 3


class XdatcarReader(FrameReader):
    def __init__(self, path):
        self.path = str(path)
        self._size = file_size(self.path)
        self._frames, truncated = self._scan(0, 0, None)  # (header offset, configuration-line offset)
        if truncated:
            warnings.warn(f"{path}: last configuration is incomplete and was skipped", stacklevel=2)
        if not self._frames:
            raise ValueError(f"{path}: no configurations found")
        self._header_cache: dict[int, tuple] = {}

    def _scan(self, start, header, natoms) -> tuple[list[tuple[int, int]], bool]:
        """Frames from byte ``start`` on (``header``/``natoms``: the header in force there),
        and whether the scan stopped at an incomplete configuration or header."""
        frames = []
        with open_binary(self.path) as fh:
            if start:
                fh.seek(start)
            while True:
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                if line.strip().lower().startswith((b"direct", b"cartesian")):
                    if natoms is None:
                        raise ValueError(f"{self.path}: configuration before header")
                    complete = all(_has_xyz(fh.readline()) for _ in range(natoms))
                    if not complete:
                        return frames, True
                    frames.append((header, offset))
                else:  # a (possibly repeated, variable-cell) 7-line header starts here
                    header = offset
                    for _ in range(6):
                        last = fh.readline()
                    if not last.endswith(b"\n"):  # VASP is still writing it
                        return frames, True
                    try:
                        natoms = sum(int(c) for c in last.split())
                    except ValueError:
                        raise ValueError(f"{self.path}: cannot read atom counts from header line "
                                         f"{last.decode(errors='replace').strip()!r}") from None
        return frames, False

    def refresh(self) -> int | None:
        """Index configurations VASP appended since the last look; the old last one is
        scanned again (from its own line on) and counts as changed, since its last line
        may have been half written when it was read."""
        size = file_size(self.path)
        if size is None or size == self._size:
            return None
        if size < self._size:  # rewritten (a new run): index from scratch
            self._size = size
            frames, _ = self._scan(0, 0, None)
            if not frames:
                return None
            self._frames, self._header_cache = frames, {}
            return 0
        self._size = size
        header, cfg = self._frames[-1]
        natoms = sum(self._read_header(header)[3])
        frames, _ = self._scan(cfg, header, natoms)
        first = len(self._frames) - 1  # its last line may have been half written
        if frames:
            self._frames = self._frames[:-1] + frames
        return first

    def _read_header(self, offset):
        if offset in self._header_cache:
            return self._header_cache[offset]
        with open_binary(self.path) as fh:
            fh.seek(offset)
            lines = [fh.readline().decode() for _ in range(7)]
        scale = float(lines[1].split()[0])
        lattice = np.array([l.split()[:3] for l in lines[2:5]], dtype=float)
        if scale < 0:  # negative scale = target volume
            scale = (-scale / abs(np.linalg.det(lattice))) ** (1 / 3)
        lattice *= scale
        species = [species_name(s) for s in lines[5].split()]
        counts = [int(c) for c in lines[6].split()]
        if len(species) != len(counts):
            raise ValueError(f"{self.path}: species line {lines[5].strip()!r} does not match counts "
                             f"{lines[6].strip()!r} (VASP 4 XDATCAR without species is not supported)")
        self._header_cache[offset] = result = (lattice, scale, species, counts)
        return result

    def __len__(self):
        return len(self._frames)

    def read(self, index: int) -> DataCollection:
        header, cfg = self._frames[index]
        lattice, scale, species, counts = self._read_header(header)
        n = sum(counts)
        with open_binary(self.path) as fh:
            fh.seek(cfg)
            kind = fh.readline().strip().lower()
            coords = np.array([_xyz(fh.readline()) for _ in range(n)], dtype=float)
        coords = coords.reshape(n, 3)
        cell = Cell(lattice, np.zeros(3), np.ones(3, bool))
        pos = cell.to_cartesian(coords) if kind.startswith(b"direct") else coords * scale
        types = np.repeat(species, counts)
        data = DataCollection(pos, types, cell)
        data.particles["Particle Identifier"] = np.arange(1, n + 1)
        try:
            data.attributes["Timestep"] = int(kind.split(b"=")[-1])
        except ValueError:
            data.attributes["Timestep"] = index
        return data
