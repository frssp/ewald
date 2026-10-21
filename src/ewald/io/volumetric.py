"""Volumetric data: VASP CHGCAR/CHG/PARCHG/AECCAR/LOCPOT/ELFCAR and Gaussian cube files.

Each file becomes one frame: the atoms of its header as particles plus a
``VoxelGrid`` in ``data.grids``. VASP charge files store rho * V_cell; the reader
divides by the cell volume so "Charge Density" is in e/Angstrom^3 (as VESTA shows
it). A spin-polarised CHGCAR has a second block, "Magnetization" (rho_up - rho_down;
"Spin Up"/"Spin Down" are derived on access); a non-collinear one has three
(stored as one N x N x N x 3 "Magnetization"). Gaussian cube values are kept as
written (atomic units), positions are converted from Bohr to Angstrom.
"""
from __future__ import annotations

import os
import re
import warnings

import numpy as np

from ..data import Cell, DataCollection, VoxelGrid
from .base import FrameReader, IncompleteFileWarning, open_binary, strip_compression
from .xdatcar import species_name

BOHR = 0.529177210903  # Angstrom

_VASP_NAME = re.compile(r"(CHGCAR|^CHG|PARCHG|AECCAR|LOCPOT|ELFCAR)", re.IGNORECASE)
# kind -> (grid identifier, property name, divide by cell volume)
_VASP_KINDS = {"charge": ("charge-density", "Charge Density", True),
               "potential": ("potential", "Potential", False),
               "elf": ("elf", "ELF", False)}


def is_vasp_volumetric(path) -> bool:
    return bool(_VASP_NAME.search(os.path.basename(strip_compression(path))))


def is_cube(path) -> bool:
    return strip_compression(path).lower().endswith((".cube", ".cub"))


def vasp_kind(path) -> str:
    name = os.path.basename(strip_compression(path)).upper()
    return "potential" if "LOCPOT" in name else "elf" if "ELFCAR" in name else "charge"


def _read_text(path) -> list[str]:
    with open_binary(path) as fh:
        return fh.read().decode(errors="replace").splitlines()


def _is_dims(line: str, dims=None):
    tok = line.split()
    if len(tok) != 3 or not all(t.isdigit() for t in tok):
        return None
    d = tuple(int(t) for t in tok)
    return d if dims is None or d == dims else None


def _read_block(lines, start, dims):
    """Values of one VASP grid block (x fastest) starting at line ``start``; (array, next line)."""
    n = int(np.prod(dims))
    per_line = len(lines[start].split()) if start < len(lines) else 0
    if per_line == 0:
        return None, start
    stop = start + -(-n // per_line)
    values = np.fromstring(" ".join(lines[start:stop]), sep=" ")
    if values.size < n:
        return None, stop
    return values[:n].reshape(dims[::-1]).transpose(2, 1, 0), stop


def read_vasp_volumetric(path, kind: str | None = None) -> DataCollection:
    """Read a VASP volumetric file. ``kind``: 'charge' (CHGCAR, CHG, PARCHG, AECCAR*),
    'potential' (LOCPOT) or 'elf' (ELFCAR); guessed from the file name by default."""
    kind = kind or vasp_kind(path)
    if kind not in _VASP_KINDS:
        raise ValueError(f"kind must be one of {list(_VASP_KINDS)}, got {kind!r}")
    lines = _read_text(path)
    scale = float(lines[1].split()[0])
    lattice = np.array([l.split()[:3] for l in lines[2:5]], dtype=float)
    factor = (-scale / abs(np.linalg.det(lattice))) ** (1 / 3) if scale < 0 else scale
    lattice = lattice * factor
    tok = lines[5].split()
    if tok and not tok[0].lstrip("-").isdigit():
        species = [species_name(t) for t in tok]
        i = 6
    else:  # VASP 4: species names, if any, are in the comment line
        species = None
        i = 5
    counts = [int(c) for c in lines[i].split()]
    if species is None or len(species) != len(counts):
        guess = lines[0].split()
        species = [species_name(t) for t in guess] if len(guess) == len(counts) else \
            [f"X{k + 1}" for k in range(len(counts))]
    i += 1
    if lines[i].strip()[:1] in "sS":
        i += 1
    cartesian = lines[i].strip()[:1] in "cCkK"
    i += 1
    natoms = sum(counts)
    coords = np.array([l.split()[:3] for l in lines[i:i + natoms]], dtype=float).reshape(natoms, 3)
    i += natoms
    cell = Cell(lattice, np.zeros(3), np.ones(3, bool))
    positions = coords * factor if cartesian else cell.to_cartesian(coords)

    blocks, dims = [], None
    while i < len(lines):
        d = _is_dims(lines[i], dims)
        if d is None:
            if dims is None and lines[i].strip():
                raise ValueError(f"{path}: expected grid dimensions after the atom positions, "
                                 f"got {lines[i].strip()!r}")
            i += 1
            continue
        dims = d
        values, i = _read_block(lines, i + 1, dims)
        if values is None:
            warnings.warn(f"{path}: calculation incomplete: grid block {len(blocks) + 1} is cut short and was "
                          f"skipped", IncompleteFileWarning, stacklevel=2)
            break
        blocks.append(values)
    data = DataCollection(positions, np.repeat(species, counts), cell)
    data.particles["Particle Identifier"] = np.arange(1, natoms + 1)
    if not blocks:
        if dims is None:
            raise ValueError(f"{path}: no volumetric data found")
        return data  # VASP was killed while writing the grid: the atoms alone

    ident, prop, per_volume = _VASP_KINDS[kind]
    if per_volume:
        blocks = [b / cell.volume for b in blocks]
    grid = VoxelGrid(cell.copy())
    grid[prop] = blocks[0]
    if kind == "charge" and len(blocks) == 2:
        grid["Magnetization"] = blocks[1]
    elif kind == "charge" and len(blocks) == 4:
        grid["Magnetization"] = np.stack(blocks[1:], axis=-1)
    else:  # extra blocks of LOCPOT/ELFCAR (spin channels; their meaning depends on the VASP version)
        for k, b in enumerate(blocks[1:], start=2):
            grid[f"{prop} {k}"] = b
    data.grids[ident] = grid
    return data


def read_cube(path, pbc: bool = False) -> DataCollection:
    """Read a Gaussian cube file (also written by Quantum ESPRESSO pp.x, CP2K, vaspkit, ...).

    ``pbc=True`` marks the grid periodic (plane-wave codes); Gaussian output is not.
    Multi-value files (several orbitals) give an N x N x N x k "Value".
    """
    from ase.data import chemical_symbols

    lines = _read_text(path)
    head = lines[2].split()
    natoms = int(head[0])
    origin = np.array(head[1:4], float)
    nval = int(head[4]) if len(head) > 4 else 1
    axes = [lines[3 + k].split() for k in range(3)]
    shape = tuple(abs(int(a[0])) for a in axes)
    vectors = np.array([a[1:4] for a in axes], float)
    unit = BOHR if int(axes[0][0]) > 0 else 1.0
    atoms = np.array([l.split()[:5] for l in lines[6:6 + abs(natoms)]], float).reshape(-1, 5)
    i = 6 + abs(natoms)
    if natoms < 0:  # orbital cube: "nmo mo1 mo2 ..." line, possibly wrapped
        tok = lines[i].split()
        nmo = int(tok[0])
        while len(tok) < nmo + 1:
            i += 1
            tok += lines[i].split()
        nval = nmo
        i += 1
    values = np.fromstring(" ".join(lines[i:]), sep=" ")
    n = int(np.prod(shape)) * nval
    if values.size < n:
        raise ValueError(f"{path}: expected {n} values, found {values.size}")
    values = values[:n].reshape(shape + ((nval,) if nval > 1 else ()))
    cell = Cell(vectors * np.array(shape)[:, None] * unit, origin * unit, np.full(3, bool(pbc)))
    grid = VoxelGrid(cell.copy())
    grid["Value"] = values
    z = atoms[:, 0].astype(int)
    types = [chemical_symbols[k] if 0 < k < len(chemical_symbols) else "X" for k in z]
    data = DataCollection(atoms[:, 2:5] * unit, types, cell)
    data.particles["Particle Identifier"] = np.arange(1, len(z) + 1)
    data.grids["cube"] = grid
    return data


class VolumetricReader(FrameReader):
    """One-frame reader for a volumetric file (parsed on first access)."""

    def __init__(self, path, format: str | None = None, kind: str | None = None, pbc: bool = False):
        self.path = str(path)
        self.cube = format == "cube" or (format is None and is_cube(path))
        self.kind = kind
        self.pbc = pbc

    def __len__(self):
        return 1

    def read(self, index: int) -> DataCollection:
        if index != 0:
            raise IndexError(index)
        if self.cube:
            return read_cube(self.path, self.pbc)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IncompleteFileWarning)
            data = read_vasp_volumetric(self.path, self.kind)
        notes = []
        for w in caught:
            if issubclass(w.category, IncompleteFileWarning):
                notes.append(str(w.message).split("calculation incomplete: ", 1)[-1])
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        if not data.grids and not notes:
            notes.append("the grid is cut short; only the atoms were read")
        self._set_incomplete("; ".join(notes) or None, stacklevel=2)
        return data


def _pick_grid(data, grid=None, property=None):
    if not data.grids:
        raise ValueError("data has no volumetric grid")
    if grid is None:
        if property is not None:
            grid = next((k for k, g in data.grids.items() if property in g), next(iter(data.grids)))
        else:
            grid = next(iter(data.grids))
    return data.grids[grid]


def write_vasp_volumetric(path, data: DataCollection, property: str | None = None, grid: str | None = None):
    """Write ``data`` as a VASP CHGCAR-style file (VESTA reads it). Charge-like values
    ("Charge Density", "Magnetization", spin channels, differences) are multiplied by the
    cell volume as VASP expects; "Potential"/"ELF"/"Value" are written as they are."""
    g = _pick_grid(data, grid, property)
    prop = property or next(iter(g.properties))
    values = np.asarray(g[prop], float)
    if values.ndim != 3:
        raise ValueError(f"{prop!r} has components; write one, e.g. '{prop}.Z'")
    if not any(k in prop for k in ("Potential", "ELF", "Value")):
        values = values * g.domain.volume
    types = data.types
    species = list(dict.fromkeys(types))
    order = np.concatenate([np.flatnonzero(types == s) for s in species]) if data.count else np.zeros(0, int)
    frac = g.domain.to_fractional(data.positions[order]) if data.count else np.zeros((0, 3))
    with open(os.fspath(path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{prop} written by ewald\n   1.00000000000000\n")
        for v in g.domain.matrix:
            fh.write(" ".join(f"{x:12.6f}" for x in v) + "\n")
        fh.write("   " + "   ".join(species) + "\n")
        fh.write("   " + "   ".join(str(int(np.sum(types == s))) for s in species) + "\nDirect\n")
        for f in frac:
            fh.write(" ".join(f"{x:10.6f}" for x in f) + "\n")
        fh.write("\n" + " ".join(f"{n:5d}" for n in g.shape) + "\n")
        flat = values.transpose(2, 1, 0).ravel()
        rows = len(flat) // 5 * 5
        np.savetxt(fh, flat[:rows].reshape(-1, 5), fmt="%.11E")
        if rows < len(flat):
            np.savetxt(fh, flat[rows:].reshape(1, -1), fmt="%.11E")


def write_cube(path, data: DataCollection, property: str | None = None, grid: str | None = None):
    """Write one scalar grid property and the atoms as a Gaussian cube file (Bohr units)."""
    from ..radii import atomic_number

    g = _pick_grid(data, grid, property)
    prop = property or next(iter(g.properties))
    values = np.asarray(g[prop], float)
    if values.ndim != 3:
        raise ValueError(f"{prop!r} has components; write one, e.g. '{prop}.Z'")
    step = g.domain.matrix / np.array(g.shape)[:, None] / BOHR
    with open(os.fspath(path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{prop} written by ewald\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        fh.write(f"{data.count:5d} " + " ".join(f"{x:12.6f}" for x in g.domain.origin / BOHR) + "\n")
        for n, v in zip(g.shape, step):
            fh.write(f"{n:5d} " + " ".join(f"{x:12.6f}" for x in v) + "\n")
        for t, p in zip(data.types, data.positions / BOHR if data.count else []):
            z = atomic_number(t)
            fh.write(f"{z:5d} {float(z):12.6f} " + " ".join(f"{x:12.6f}" for x in p) + "\n")
        flat = values.reshape(-1, values.shape[2])
        for row in flat:
            for k in range(0, len(row), 6):
                fh.write(" ".join(f"{x:13.5E}" for x in row[k:k + 6]) + "\n")
