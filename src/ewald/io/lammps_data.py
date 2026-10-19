"""Native reader for LAMMPS data files (``read_data``/``write_data`` format).

Supports the common atom styles (atomic, charge, bond/angle/molecular, full,
sphere, dipole), optional image flags, orthogonal, restricted-triclinic
(``xy xz yz``) and general-triclinic (``avec bvec cvec abc origin``) cells,
``Velocities`` and ``Bonds`` sections, and particle type names from an
``Atom Type Labels`` section or from ``# Cu`` comments in ``Masses``.
Values stay in the file's own LAMMPS units.

Bonds are stored in ``data.bonds`` (shortest periodic image) and, for older code, as
``data.tables["Bonds"]`` with ``"Topology"`` (M x 2, zero-based particle indices into the
returned collection) and ``"Bond Type"``.
"""
from __future__ import annotations

import os
import re

import numpy as np

from ..bonds import Bonds, minimum_images
from ..data import Cell, DataCollection
from .base import FrameReader, head_lines, open_binary
from .lammps_dump import _cell_from_bounds

# atom style -> column names following the atom id (image flags may follow)
ATOM_STYLES = {
    "atomic": ("type", "x", "y", "z"),
    "charge": ("type", "q", "x", "y", "z"),
    "bond": ("mol", "type", "x", "y", "z"),
    "angle": ("mol", "type", "x", "y", "z"),
    "molecular": ("mol", "type", "x", "y", "z"),
    "full": ("mol", "type", "q", "x", "y", "z"),
    "sphere": ("type", "diameter", "density", "x", "y", "z"),
    "dipole": ("type", "q", "x", "y", "z", "mux", "muy", "muz"),
}
_PROPERTY = {"mol": "Molecule Identifier", "q": "Charge", "diameter": "Diameter", "density": "Density"}
_ATOMS_LINE = re.compile(rb"^\s*\d+\s+atoms\s*(#.*)?$")


def is_lammps_data(path) -> bool:
    """Content sniff: a '<N> atoms' header line among the first lines."""
    lines = head_lines(path, 40)
    return any(_ATOMS_LINE.match(line) for line in lines[1:])


class LammpsDataReader(FrameReader):
    def __init__(self, path, atom_style: str | None = None, type_map: dict | None = None,
                 sort_by_id: bool = True):
        self.path = str(path)
        self.atom_style = atom_style
        self.type_map = {str(k): v for k, v in (type_map or {}).items()}
        self.sort_by_id = sort_by_id
        self._data = None

    def __len__(self):
        return 1

    def read(self, index: int = 0) -> DataCollection:
        if index not in (0, -1):
            raise IndexError(index)
        if self._data is None:
            self._data = self._parse()
        return self._data.copy()

    # ------------------------------------------------------------------------
    def _parse(self) -> DataCollection:
        with open_binary(self.path) as fh:
            lines = fh.read().decode(errors="replace").splitlines()
        header, sections = _split_sections(lines[1:])
        counts, cell = _parse_header(header, self.path)

        labels = {}
        for body, _, _ in sections.get("Atom Type Labels", []):
            for row in body:
                labels[row[0]] = row[1]
        masses, mass_names = {}, {}
        for body, comments, _ in sections.get("Masses", []):
            for row, comment in zip(body, comments):
                masses[row[0]] = float(row[1])
                if comment and comment.split()[0].isalpha():
                    mass_names[row[0]] = comment.split()[0]

        atoms = sections.get("Atoms")
        n = counts.get("atoms", 0)
        if not atoms and n:
            raise ValueError(f"{self.path}: no Atoms section")
        rows, style_comment = (atoms[0][0], atoms[0][2]) if atoms else ([], None)
        if len(rows) != n:
            raise ValueError(f"{self.path}: header says {n} atoms, Atoms section has {len(rows)}")
        style = (self.atom_style or style_comment or _infer_style(rows, self.path)).split()[0]
        if style not in ATOM_STYLES:
            raise ValueError(f"{self.path}: atom style '{style}' is not supported "
                             f"(supported: {', '.join(ATOM_STYLES)})")
        cols = ATOM_STYLES[style]
        width = len(cols) + 1
        if any(len(r) < width for r in rows):
            raise ValueError(f"{self.path}: Atoms rows have fewer columns than atom style '{style}' needs")

        ids = np.array([int(r[0]) for r in rows], dtype=np.int64)
        table = {c: [r[1 + k] for r in rows] for k, c in enumerate(cols)}
        image = np.array([[int(v) for v in r[width:width + 3]] if len(r) >= width + 3 else [0, 0, 0]
                          for r in rows], dtype=int).reshape(n, 3)

        pos = np.array([table["x"], table["y"], table["z"]], dtype=float).T.reshape(n, 3)
        # Types may be numbers or (with a labelmap) the labels themselves.
        inverse = {v: k for k, v in labels.items()}
        type_index = np.array([t if t.isdigit() else inverse.get(t, "0") for t in table["type"]]).astype(int)
        names = {**mass_names, **labels}
        types = np.array([self.type_map.get(t) or names.get(t) or t
                          for t in type_index.astype(str)]) if n else np.array([], str)

        data = DataCollection(pos, types, cell)
        data.particles["Particle Identifier"] = ids
        data.particles["Type Index"] = type_index
        for c in cols:
            if c in _PROPERTY:
                values = np.array(table[c], dtype=float)
                data.particles[_PROPERTY[c]] = values.astype(np.int64) if c == "mol" else values
        if "mux" in cols:
            data.particles["Dipole Orientation"] = np.array([table["mux"], table["muy"], table["muz"]],
                                                            dtype=float).T.reshape(n, 3)
        data.particles["Periodic Image"] = image
        if masses:
            data.particles["Mass"] = np.array([masses.get(t, 0.0) for t in type_index.astype(str)])

        id_to_row = {int(i): k for k, i in enumerate(ids)}
        vel = sections.get("Velocities")
        if vel:
            v = np.zeros((n, 3))
            for row in vel[0][0]:
                v[id_to_row[int(row[0])]] = [float(x) for x in row[1:4]]
            data.particles["Velocity"] = v
        bonds = sections.get("Bonds")
        if bonds:
            body = bonds[0][0]
            topo = np.array([[id_to_row[int(r[2])], id_to_row[int(r[3])]] for r in body], dtype=np.int64)
            btype = np.array([r[1] for r in body])
            data.tables["Bonds"] = {"Topology": topo.reshape(-1, 2), "Bond Type": btype}
            topo = topo.reshape(-1, 2)
            data.bonds = Bonds(topo, minimum_images(data, topo[:, 0], topo[:, 1]), bond_type=btype)

        if self.sort_by_id and n:
            order = np.argsort(ids, kind="stable")
            if np.any(order != np.arange(n)):
                rank = np.empty(n, dtype=np.int64)
                rank[order] = np.arange(n)
                data = data.take(order)
                if "Bonds" in data.tables:
                    data.tables["Bonds"]["Topology"] = rank[data.tables["Bonds"]["Topology"]]
        data.attributes["Timestep"] = 0
        data.attributes["AtomStyle"] = style
        return data


def _split_sections(lines):
    """Split into header lines and {section name: [(rows, row comments, header comment)]}."""
    header, sections = [], {}
    current = None
    for raw in lines:
        text, _, comment = raw.partition("#")
        text = text.strip()
        if not text:
            continue
        if text[0].isalpha():  # section keyword such as 'Atoms' or 'Pair Coeffs'
            current = ([], [], comment.strip() or None)
            sections.setdefault(text, []).append(current)
        elif current is None:
            header.append(text)
        else:
            current[0].append(text.split())
            current[1].append(comment.strip())
    return header, sections


def _parse_header(header, path):
    counts, bounds, tilt = {}, {}, (0.0, 0.0, 0.0)
    vecs = {}
    triclinic = False
    for line in header:
        tok = line.split()
        keyword = " ".join(t for t in tok if not _is_number(t))
        nums = [float(t) for t in tok if _is_number(t)]
        if keyword in ("xlo xhi", "ylo yhi", "zlo zhi"):
            bounds[keyword[0]] = nums[:2]
        elif keyword == "xy xz yz":
            tilt, triclinic = tuple(nums[:3]), True
        elif keyword in ("avec", "bvec", "cvec", "abc origin"):
            vecs[keyword] = nums[:3]
        elif nums:
            counts[keyword] = int(nums[0])
    if len(vecs) == 4:
        return counts, Cell(np.array([vecs["avec"], vecs["bvec"], vecs["cvec"]]), vecs["abc origin"], True)
    if len(bounds) != 3:
        raise ValueError(f"{path}: missing xlo xhi / ylo yhi / zlo zhi in header")
    rows = np.array([bounds["x"] + [tilt[0]], bounds["y"] + [tilt[1]], bounds["z"] + [tilt[2]]])
    if triclinic:  # data files store the true (unexpanded) bounds, unlike dump files
        (xlo, xhi, xy), (ylo, yhi, xz), (zlo, zhi, yz) = rows
        matrix = np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])
        return counts, Cell(matrix, [xlo, ylo, zlo], True)
    return counts, _cell_from_bounds(rows, False, np.ones(3, bool))


def _infer_style(rows, path) -> str:
    if not rows:
        return "atomic"
    width = min(len(r) for r in rows)

    def integral(col):
        return all(_is_int(r[col]) for r in rows)

    candidates = [s for s, cols in ATOM_STYLES.items()
                  if width in (len(cols) + 1, len(cols) + 4) and s not in ("bond", "angle")]
    # Several styles share a width; tell them apart by which columns hold integers.
    if width in (5, 8):
        return "atomic"
    if width == 6:
        return "molecular" if integral(2) else "charge"
    if width == 7:
        return "full" if integral(1) and integral(2) else "sphere"
    if width == 9:
        if integral(6) and integral(7) and integral(8):
            return "molecular" if integral(2) else "charge"
        return "dipole"
    if width == 10:
        return "full" if integral(1) and integral(2) else "sphere"
    raise ValueError(f"{path}: cannot infer atom style from {width} columns "
                     f"(candidates {candidates}); pass atom_style='...'")


def _is_number(t: str) -> bool:
    try:
        float(t)
        return True
    except ValueError:
        return False


def _is_int(t: str) -> bool:
    return t.lstrip("+-").isdigit()


def lammps_box(matrix):
    """LAMMPS' restricted triclinic form of a cell: (lower-triangular matrix, rotation R)
    with ``matrix @ R == lower``; rotate positions, velocities and forces by ``R``."""
    a, b, c = np.asarray(matrix, float)
    if np.linalg.det(matrix) <= 0:
        raise ValueError("LAMMPS needs a right-handed cell (positive determinant)")
    ax = np.linalg.norm(a)
    bx = b @ a / ax
    by = np.sqrt(max(b @ b - bx * bx, 0.0))
    cx = c @ a / ax
    cy = (b @ c - bx * cx) / by
    cz = np.sqrt(max(c @ c - cx * cx - cy * cy, 0.0))
    lower = np.array([[ax, 0, 0], [bx, by, 0], [cx, cy, cz]])
    lower[np.abs(lower) < 1e-12 * ax] = 0.0
    return lower, np.linalg.solve(matrix, lower)


def write_lammps_data(path, data, atom_style: str | None = None, masses: bool = True):
    """Write one frame as a LAMMPS data file that LAMMPS ``read_data`` and ewald read back.

    Types are numbered in order of first appearance (numeric types such as "2" keep their
    number) and named in ``Masses`` comments (``1 63.546 # Cu``). ``atom_style`` is
    ``"atomic"``, or ``"charge"`` when the frame has a "Charge" property; ``"molecular"``
    and ``"full"`` take molecule ids from "Molecule Identifier" (0 without it). A "Velocity" property
    becomes a ``Velocities`` section. A cell that is not in LAMMPS' lower-triangular form
    is rotated rigidly into it.
    """
    from ase.data import atomic_masses, atomic_numbers

    types = np.asarray(data.types).astype(str)
    names = list(dict.fromkeys(types))
    if all(t.isdigit() for t in names):
        type_id = {t: int(t) for t in names}
    else:
        type_id = {t: i + 1 for i, t in enumerate(names)}
    style = atom_style or ("charge" if "Charge" in data.particles else "atomic")
    cell = data.cell
    if cell.is_valid:
        lower, rotation = lammps_box(cell.matrix)
        origin = cell.origin
    else:  # bounding box with a margin
        lo, hi = (data.positions.min(axis=0), data.positions.max(axis=0)) if data.count else (np.zeros(3), np.ones(3))
        origin, lower, rotation = lo - 1.0, np.diag(hi - lo + 2.0), np.eye(3)
    pos = (data.positions - origin) @ rotation
    ids = np.asarray(data.particles.get("Particle Identifier", np.arange(1, data.count + 1))).astype(int)
    charge = np.asarray(data.particles.get("Charge", np.zeros(data.count)), float)
    mol = np.asarray(data.particles.get("Molecule Identifier", np.zeros(data.count)), int)
    mass_prop = data.particles.get("Mass")
    with open(os.fspath(path), "w") as fh:
        fh.write("LAMMPS data file written by ewald\n\n")
        fh.write(f"{data.count} atoms\n{max(type_id.values(), default=0)} atom types\n\n")
        for k, axis in enumerate("xyz"):
            fh.write(f"0.0 {lower[k, k]:.10f} {axis}lo {axis}hi\n")
        if np.any(lower[[1, 2, 2], [0, 0, 1]]):
            fh.write(f"{lower[1, 0]:.10f} {lower[2, 0]:.10f} {lower[2, 1]:.10f} xy xz yz\n")
        if masses:
            fh.write("\nMasses\n\n")
            for t in names:
                if mass_prop is not None:
                    m = float(np.asarray(mass_prop)[types == t][0])
                else:
                    m = float(atomic_masses[atomic_numbers[t]]) if t in atomic_numbers else 1.0
                comment = f" # {t}" if not t.isdigit() else ""
                fh.write(f"{type_id[t]} {m:.6f}{comment}\n")
        fh.write(f"\nAtoms # {style}\n\n")
        for i in range(data.count):
            x, y, z = pos[i]
            t = type_id[types[i]]
            if style == "atomic":
                fh.write(f"{ids[i]} {t} {x:.10f} {y:.10f} {z:.10f}\n")
            elif style == "charge":
                fh.write(f"{ids[i]} {t} {charge[i]:.6f} {x:.10f} {y:.10f} {z:.10f}\n")
            elif style == "full":
                fh.write(f"{ids[i]} {mol[i]} {t} {charge[i]:.6f} {x:.10f} {y:.10f} {z:.10f}\n")
            elif style == "molecular":
                fh.write(f"{ids[i]} {mol[i]} {t} {x:.10f} {y:.10f} {z:.10f}\n")
            else:
                raise ValueError(f"atom_style {style!r} not supported (atomic, charge, molecular, full)")
        if "Velocity" in data.particles:
            vel = np.asarray(data["Velocity"], float) @ rotation
            fh.write("\nVelocities\n\n")
            for i in range(data.count):
                fh.write(f"{ids[i]} " + " ".join(f"{v:.10f}" for v in vel[i]) + "\n")
