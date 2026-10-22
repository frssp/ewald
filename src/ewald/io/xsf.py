"""XCrySDen XSF: structures, ANIMSTEPS trajectories and 3D data grids (read and write).

XSF is what Quantum ESPRESSO ``pp.x`` (``output_format=5``), VESTA and XCrySDen exchange.
``CRYSTAL``/``SLAB``/``POLYMER`` set the periodic axes (3/2/1), ``MOLECULE`` (or no keyword)
none. ``PRIMCOORD`` atoms carry optional forces (columns 5-7, "Force"). Every
``DATAGRID_3D`` becomes a property of one :class:`VoxelGrid` in ``data.grids["xsf"]``,
named by its label. XSF grids are "general" grids: the last point of each axis repeats
the first, so for a periodic structure it is dropped and the grid matches the ewald
(CHGCAR) convention; for a molecule the grid keeps every point and spans the given
vectors. Animated files index the frame offsets once and parse a frame on demand.
"""
from __future__ import annotations

import os
import re

import numpy as np

from ..data import Cell, DataCollection, VoxelGrid
from .base import FrameReader, open_binary, strip_compression

_DIMENSION = {b"CRYSTAL": 3, b"SLAB": 2, b"POLYMER": 1, b"MOLECULE": 0}


def is_xsf(path) -> bool:
    return strip_compression(path).lower().endswith((".xsf", ".axsf"))


_KEYWORDS = {b"ANIMSTEPS", b"CRYSTAL", b"SLAB", b"POLYMER", b"MOLECULE", b"PRIMVEC", b"CONVVEC",
             b"PRIMCOORD", b"CONVCOORD", b"ATOMS"}


def _keyword(line: bytes):
    """The XSF keyword starting ``line`` (atom rows may start with an element symbol: not one)."""
    tok = line.split()
    if not tok:
        return None
    key = tok[0].upper()
    return key if key in _KEYWORDS or key.startswith((b"BEGIN_", b"END_", b"DATAGRID", b"BLOCK_")) else None


def _symbol(token: str) -> str:
    from ase.data import chemical_symbols

    if token.isdigit():
        z = int(token)
        return chemical_symbols[z] if 0 < z < len(chemical_symbols) else "X"
    return token[:1].upper() + token[1:].lower()


class XsfReader(FrameReader):
    """XSF/AXSF, lazy over ANIMSTEPS; grids are read with the frame (each frame carries them)."""

    def __init__(self, path):
        self.path = str(path)
        self.dimension = 0
        self._cells: dict[int, int] = {}     # step -> offset of its PRIMVEC
        self._coords: dict[int, tuple[int, bytes]] = {}  # step -> (offset, b"PRIMCOORD" | b"ATOMS")
        self._grids: list[int] = []          # offsets of BEGIN_BLOCK_DATAGRID_3D
        with open_binary(self.path) as fh:
            pos = 0
            for line in fh:
                key = _keyword(line)
                if key is not None:
                    tok = line.split()
                    step = int(tok[1]) if len(tok) > 1 and tok[1].isdigit() else 1
                    if key in _DIMENSION:
                        self.dimension = _DIMENSION[key]
                    elif key == b"PRIMVEC":
                        self._cells[step] = pos
                    elif key in (b"PRIMCOORD", b"ATOMS"):
                        self._coords[step] = (pos, key)
                    elif key in (b"BEGIN_BLOCK_DATAGRID_3D", b"BLOCK_DATAGRID_3D"):
                        self._grids.append(pos)
                pos += len(line)
        self._steps = sorted(self._coords)
        if not self._steps:
            raise ValueError(f"{path}: no PRIMCOORD or ATOMS block found")
        self._grid_cache = None

    def __len__(self):
        return len(self._steps)

    def _lines(self, offset, n=None):
        with open_binary(self.path) as fh:
            fh.seek(offset)
            fh.readline()
            if n is not None:
                return [fh.readline().decode() for _ in range(n)]
            out = []
            for line in fh:
                if _keyword(line) is not None:
                    break
                if line.strip() and not line.lstrip().startswith(b"#"):
                    out.append(line.decode())
            return out

    def read(self, index: int) -> DataCollection:
        step = self._steps[index]
        pbc = np.arange(3) < self.dimension
        cell = Cell(np.zeros((3, 3)), np.zeros(3), np.zeros(3, bool))
        cell_steps = [s for s in self._cells if s <= step] or list(self._cells)
        if self.dimension and cell_steps:
            vec = self._lines(self._cells[max(cell_steps)], 3)
            cell = Cell(np.array([v.split()[:3] for v in vec], float), np.zeros(3), pbc)
        offset, key = self._coords[step]
        if key == b"PRIMCOORD":
            n = int(self._lines(offset, 1)[0].split()[0])
            rows = [r.split() for r in self._lines(offset, n + 1)[1:]]
        else:
            rows = [r.split() for r in self._lines(offset)]
        types = [_symbol(r[0]) for r in rows]
        data = DataCollection(np.array([r[1:4] for r in rows], float).reshape(-1, 3), types, cell)
        data.particles["Particle Identifier"] = np.arange(1, len(rows) + 1)
        if rows and all(len(r) >= 7 for r in rows):
            data.particles["Force"] = np.array([r[4:7] for r in rows], float)
        data.attributes["Timestep"] = step
        if self._grids:
            data.grids.update(self._read_grids(pbc))
        return data

    def _read_grids(self, pbc):
        if self._grid_cache is None:
            grids = {}
            for offset in self._grids:
                for label, values, domain in _parse_grid_block(self.path, offset, pbc):
                    grid = next((g for g in grids.values() if g.shape == values.shape), None)
                    if grid is None:  # grids of another size go into their own VoxelGrid
                        grid = grids[f"xsf {len(grids) + 1}" if grids else "xsf"] = VoxelGrid(domain)
                    grid[label] = values
            self._grid_cache = grids
        return {k: g.copy() for k, g in self._grid_cache.items()}


_GRID_HEAD = re.compile(r"^(?:BEGIN_)?DATAGRID_3D_?(.*)$", re.IGNORECASE)


def _parse_grid_block(path, offset, pbc):
    """(label, values, domain) of each DATAGRID_3D in the block at ``offset``, in ewald's
    grid convention: periodic axes lose the repeated end point, other axes keep every
    point and the domain grows by one spacing."""
    with open_binary(path) as fh:
        fh.seek(offset)
        text = fh.read().decode(errors="replace").splitlines()
    i = 2  # skip BEGIN_BLOCK_DATAGRID_3D and the block name
    while i < len(text):
        head = text[i].strip()
        if head.upper().startswith("END_BLOCK_DATAGRID"):
            return
        match = _GRID_HEAD.match(head)
        if match and not head.upper().startswith("END"):
            label = match.group(1).replace("_", " ").strip() or "Value"
            shape = tuple(int(t) for t in text[i + 1].split()[:3])
            origin = np.array(text[i + 2].split()[:3], float)
            vectors = np.array([text[i + 3 + k].split()[:3] for k in range(3)], float)
            n = int(np.prod(shape))
            j, chunks, count = i + 6, [], 0
            while count < n:
                vals = np.array(text[j].split(), float)
                chunks.append(vals)
                count += len(vals)
                j += 1
            values = np.concatenate(chunks)[:n].reshape(shape[::-1]).transpose(2, 1, 0)  # x fastest
            for k in range(3):
                if pbc[k]:
                    values = np.take(values, range(shape[k] - 1), axis=k)
                elif shape[k] > 1:
                    vectors[k] *= shape[k] / (shape[k] - 1)
            yield label, values, Cell(vectors, origin, pbc.copy())
            i = j
        i += 1


def write_xsf(path, frames, property=None, grid=None):
    """Write one structure (``CRYSTAL``/``MOLECULE``) or an ``ANIMSTEPS`` trajectory.

    The first frame's grid (``grid=`` names which, ``property=`` one property; default all
    scalar ones) is written as a ``DATAGRID_3D`` block with the end points repeated, so
    VESTA and XCrySDen read it periodic.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("nothing to write")
    animated = len(frames) > 1
    first = frames[0]
    periodic = first.cell.is_valid and first.cell.pbc.any()
    with open(os.fspath(path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# written by ewald\n")
        if animated:
            fh.write(f"ANIMSTEPS {len(frames)}\n")
        if periodic:
            fh.write({3: "CRYSTAL", 2: "SLAB", 1: "POLYMER"}[int(first.cell.pbc.sum())] + "\n")
        else:
            fh.write("MOLECULE\n")
        for s, data in enumerate(frames, start=1):
            tag = f" {s}" if animated else ""
            if periodic:
                fh.write(f"PRIMVEC{tag}\n")
                for v in data.cell.matrix:
                    fh.write("  " + " ".join(f"{x:16.10f}" for x in v) + "\n")
                pos = data.positions - data.cell.origin
            else:
                pos = data.positions
            fh.write(f"PRIMCOORD{tag}\n{data.count} 1\n")
            force = data.particles.get("Force")
            for i, (t, p) in enumerate(zip(data.types, pos)):
                row = f"{t:>3s} " + " ".join(f"{x:16.10f}" for x in p)
                if force is not None:
                    row += " " + " ".join(f"{x:14.8f}" for x in force[i])
                fh.write(row + "\n")
        if first.grids:
            name = grid or next(iter(first.grids))
            g = first.grids[name]
            props = [property] if property else [p for p in g.properties if np.asarray(g[p]).ndim == 3]
            fh.write(f"BEGIN_BLOCK_DATAGRID_3D\n {name}\n")
            for prop in props:
                values = np.asarray(g[prop], float)
                if values.ndim != 3:
                    raise ValueError(f"{prop!r} has components; write one, e.g. '{prop}.Z'")
                vectors = g.domain.matrix.copy()
                for k in range(3):  # back to XSF's general grid (see _parse_grid_block)
                    if g.domain.pbc[k]:
                        values = np.concatenate([values, np.take(values, [0], axis=k)], axis=k)
                    elif values.shape[k] > 1:
                        vectors[k] *= (values.shape[k] - 1) / values.shape[k]
                label = prop.replace(" ", "_")
                fh.write(f" BEGIN_DATAGRID_3D_{label}\n  " + " ".join(str(n) for n in values.shape) + "\n")
                fh.write("  " + " ".join(f"{x:16.10f}" for x in g.domain.origin) + "\n")
                for v in vectors:
                    fh.write("  " + " ".join(f"{x:16.10f}" for x in v) + "\n")
                flat = values.transpose(2, 1, 0).ravel()
                rows = len(flat) // 6 * 6
                np.savetxt(fh, flat[:rows].reshape(-1, 6), fmt="%.8E")
                if rows < len(flat):
                    np.savetxt(fh, flat[rows:].reshape(1, -1), fmt="%.8E")
                fh.write(" END_DATAGRID_3D\n")
            fh.write("END_BLOCK_DATAGRID_3D\n")
