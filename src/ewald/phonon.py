"""Phonon modes: eigenvectors from phonopy (band.yaml, mesh.yaml, qpoints.yaml) and VASP
(OUTCAR of an IBRION=5-8 run), and the displacement pattern of one mode in a supercell.

:class:`PhononModes` holds the primitive cell, the q-points (reduced coordinates), the
frequencies (THz; imaginary modes negative, as phonopy prints them) and the eigenvectors of
the dynamical matrix, shape ``(nq, nbands, natoms, 3)``. :meth:`PhononModes.displacements`
turns one mode into atomic displacements the way ``phonopy --modulation`` does,

\\[ \\mathbf u_{lj} = \\frac{A}{\\sqrt{m_j}}\\, \\mathbf e_j(\\mathbf q, \\nu)\\,
    e^{2\\pi i\\, \\mathbf q \\cdot \\mathbf x_{lj}}, \\]

with ``x_lj`` the position of atom ``j`` in cell ``l`` in fractional coordinates of the
primitive cell, a global phase chosen so that the largest component is real, and the
amplitude scaled so that the most displaced atom moves ``amplitude`` Angstrom. At time
``t`` (phase ``theta = 2 pi t / T``) the atoms sit at ``r0 + Re(u exp(-i theta))``.

The reader (:class:`ewald.io.phonon.PhononReader`) gives the primitive cell as a looping
trajectory and stores the modes in ``data.attributes["Phonon Modes"]``; the
:class:`ewald.modifiers.PhononMode` modifier animates one mode.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

from .data import Cell, DataCollection

THZ_TO_CM = 33.35641   # 1 THz in cm^-1
THZ_TO_MEV = 4.135668  # 1 THz in meV


@dataclass(repr=False)
class PhononModes:
    """Phonon modes of a crystal on a set of q-points."""

    cell: Cell                    # primitive cell
    positions: np.ndarray         # (natoms, 3) Cartesian, Angstrom
    types: np.ndarray             # (natoms,) element names
    masses: np.ndarray            # (natoms,) amu
    qpoints: np.ndarray           # (nq, 3) reduced coordinates of the primitive reciprocal cell
    frequencies: np.ndarray       # (nq, nbands) THz, imaginary < 0
    eigenvectors: np.ndarray      # (nq, nbands, natoms, 3) complex, each mode normalised over 3N
    distances: np.ndarray | None = None  # (nq,) path length along a band structure, 1/Angstrom
    ticks: list = field(default_factory=list)  # [(distance, label)] high-symmetry points of a path
    source: str = ""

    @property
    def num_qpoints(self) -> int:
        return len(self.qpoints)

    @property
    def num_bands(self) -> int:
        return self.frequencies.shape[1]

    @property
    def is_path(self) -> bool:
        """True for a band structure (q-points along a path), False for a q-point list or mesh."""
        return self.distances is not None and self.num_qpoints > 1

    def default_band(self) -> int:
        """The first optical branch when there is one (the acoustic modes at Gamma are rigid
        translations), else the highest branch."""
        return 3 if self.num_bands > 3 else self.num_bands - 1

    def check(self, qpoint: int, band: int) -> tuple[int, int]:
        nq, nb = self.frequencies.shape
        if not -nq <= qpoint < nq:
            raise IndexError(f"qpoint {qpoint} out of range (0..{nq - 1})")
        if not -nb <= band < nb:
            raise IndexError(f"band {band} out of range (0..{nb - 1})")
        return qpoint % nq, band % nb

    def frequency(self, qpoint: int, band: int) -> float:
        q, b = self.check(qpoint, band)
        return float(self.frequencies[q, b])

    def commensurate_supercell(self, qpoint: int, max_size: int = 8) -> tuple[int, int, int]:
        """Smallest diagonal supercell in which the mode at ``qpoint`` is periodic; along an
        axis where no repeat up to ``max_size`` fits, ``max_size`` (the wave then jumps at that
        cell face)."""
        q = self.qpoints[self.check(qpoint, 0)[0]]
        out = []
        for x in q:
            n = next((n for n in range(1, max_size + 1) if abs(n * x - round(n * x)) < 1e-4), max_size)
            out.append(n)
        return tuple(out)

    def displacements(self, qpoint: int, band: int, supercell=(1, 1, 1), amplitude: float = 0.3):
        """Complex displacement field of one mode in a diagonal ``supercell``.

        Returns ``(data, u)``: the supercell as a :class:`DataCollection` (atoms ordered
        cell by cell, like :class:`ewald.modifiers.Replicate`) and ``u`` (M x 3 complex,
        Angstrom). ``Re(u)`` is the displacement at phase 0, where the largest one is
        ``amplitude``."""
        q, b = self.check(qpoint, band)
        n = np.array(supercell, int).reshape(3)
        if (n < 1).any():
            raise ValueError(f"supercell must be positive, got {tuple(n)}")
        shifts = np.array([(i, j, k) for i in range(n[0]) for j in range(n[1]) for k in range(n[2])], float)
        frac = self.cell.to_fractional(self.positions)
        x = (frac[None] + shifts[:, None]).reshape(-1, 3)       # fractional in the primitive cell
        idx = np.tile(np.arange(len(self.positions)), len(shifts))
        phase = np.exp(2j * np.pi * x @ self.qpoints[q])
        u = self.eigenvectors[q, b][idx] * (phase / np.sqrt(self.masses[idx]))[:, None]
        flat = u.ravel()
        big = flat[np.argmax(np.abs(flat))]
        if abs(big) > 0:
            u = u / (big / abs(big))                             # largest component real, positive
        norm = np.sqrt((np.abs(u) ** 2).sum(axis=1)).max()
        if norm > 0:
            u = u * (amplitude / norm)
        cell = Cell(self.cell.matrix * n[:, None], self.cell.origin, self.cell.pbc)
        data = DataCollection(self.cell.to_cartesian(x), self.types[idx], cell)
        data.particles["Mass"] = self.masses[idx]
        data.particles["Primitive Index"] = idx.astype(np.int32)
        return data, u

    def table(self) -> dict:
        """Frequencies as a table: ``Distance`` (path) or ``Q-point`` (list, mesh) or ``Mode``
        (a single q-point) followed by one column per band (THz)."""
        if self.num_qpoints == 1:
            return {"Mode": np.arange(1, self.num_bands + 1), "Frequency": self.frequencies[0].copy()}
        x = self.distances if self.distances is not None else np.arange(self.num_qpoints, dtype=float)
        out = {"Distance" if self.distances is not None else "Q-point": np.asarray(x, float)}
        for b in range(self.num_bands):
            out[f"Band {b + 1}"] = self.frequencies[:, b].copy()
        return out

    def __repr__(self) -> str:  # shown in the GUI attribute inspector
        return (f"<PhononModes {self.num_qpoints} q-points x {self.num_bands} bands, "
                f"{len(self.positions)} atoms, {self.source}>")


# ------------------------------------------------------------------------ phonopy
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def is_phonopy_yaml(path) -> bool:
    from .io.base import head_lines, strip_compression

    name = os.path.basename(strip_compression(path)).lower()
    if not name.endswith((".yaml", ".yml")):
        return False
    head = b"".join(head_lines(path, 40))
    return b"nqpoint:" in head


def _bracket(text: str) -> list[float]:
    return [float(v) for v in re.findall(_NUM, text[text.index("[") + 1:text.index("]")])]


def read_phonopy_yaml(path, structure=None) -> PhononModes:
    """band.yaml / mesh.yaml / qpoints.yaml written with eigenvectors (``--eigvecs``,
    ``EIGENVECTORS = .TRUE.``). Files without the cell (qpoints.yaml) need ``structure=``:
    the primitive cell as a path ewald can read (POSCAR, ...) or a DataCollection."""
    from .io.base import open_binary

    with open_binary(path) as fh:
        text = fh.read().decode("utf-8", "replace")
    start = re.search(r"^phonon:\s*$", text, re.M)
    if start is None:
        raise ValueError(f"{path}: no 'phonon:' section; not a phonopy band/mesh/qpoints file")
    header, body = text[:start.start()], text[start.end():]

    lattice, points, labels, segments = None, [], [], []
    lines = header.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("lattice:"):
            lattice = [_bracket(lines[i + k]) for k in (1, 2, 3)]
            i += 4
            continue
        if line.startswith(("points:", "atoms:")):
            i += 1
            while i < len(lines) and lines[i].startswith(("- ", "  ")):
                if lines[i].startswith("- symbol:"):
                    points.append({"symbol": lines[i].split(":", 1)[1].split("#")[0].strip()})
                elif "coordinates:" in lines[i] or "position:" in lines[i]:
                    points[-1]["frac"] = _bracket(lines[i])
                elif "mass:" in lines[i]:
                    points[-1]["mass"] = float(lines[i].split(":", 1)[1])
                i += 1
            continue
        if line.startswith("labels:"):
            i += 1
            while i < len(lines) and lines[i].startswith("- ["):
                labels.append(re.findall(r"'((?:[^']|'')*)'", lines[i]))
                i += 1
            continue
        if line.startswith("segment_nqpoint:"):
            i += 1
            while i < len(lines) and lines[i].startswith("- "):
                segments.append(int(lines[i][2:]))
                i += 1
            continue
        i += 1

    if lattice is not None and points:
        cell = Cell(np.array(lattice))
        frac = np.array([p["frac"] for p in points])
        positions = cell.to_cartesian(frac)
        types = np.array([p["symbol"] for p in points])
        masses = np.array([p.get("mass", np.nan) for p in points])
    elif structure is not None:
        cell, positions, types, masses = _structure(structure)
    else:
        raise ValueError(f"{path} has no cell (lattice/points); pass structure= (the primitive cell, e.g. a POSCAR)")
    if np.isnan(masses).any():
        masses = _ase_masses(types)

    blocks = re.split(r"^- q-position:", body, flags=re.M)[1:]
    if not blocks:
        raise ValueError(f"{path}: no q-points in the 'phonon:' section")
    qpoints, distances, freqs, point_labels = [], [], [], []
    for blk in blocks:
        qpoints.append(_bracket(blk))
        m = re.search(r"^  distance:\s*(" + _NUM + ")", blk, re.M)
        distances.append(float(m.group(1)) if m else None)
        m = re.search(r"^  label:\s*'((?:[^']|'')*)'", blk, re.M)
        point_labels.append(m.group(1) if m else None)
        freqs.append([float(v) for v in re.findall(r"frequency:\s*(" + _NUM + ")", blk)])
    nq, natoms = len(blocks), len(types)
    freqs = np.array(freqs)
    nb = freqs.shape[1]
    vec = np.array(re.findall(r"^\s*- \[\s*(" + _NUM + r"),\s*(" + _NUM + r")\s*\]", body, re.M), float)
    if len(vec) != nq * nb * natoms * 3:
        raise ValueError(f"{path} has no eigenvectors (found {len(vec)} of {nq * nb * natoms * 3} components); "
                         f"rerun phonopy with --eigvecs (EIGENVECTORS = .TRUE.)")
    eig = (vec[:, 0] + 1j * vec[:, 1]).reshape(nq, nb, natoms, 3)

    dist = None if any(d is None for d in distances) else np.array(distances)
    ticks = _path_ticks(dist, point_labels, labels, segments) if dist is not None else []
    return PhononModes(cell, positions, types, masses, np.array(qpoints), freqs, eig, dist, ticks,
                       source=f"phonopy {os.path.basename(str(path))}")


def _path_ticks(dist, point_labels, labels, segments):
    """High-symmetry ticks: per-point labels when phonopy wrote them, else segment ends."""
    if any(point_labels):
        ticks = [(float(d), lab) for d, lab in zip(dist, point_labels) if lab]
    elif segments:
        ends = np.cumsum(segments)
        starts = np.concatenate([[0], ends[:-1]])
        ticks = []
        for k, (s, e) in enumerate(zip(starts, ends)):
            pair = labels[k] if k < len(labels) else ["", ""]
            ticks.append((float(dist[s]), pair[0] if pair else ""))
            ticks.append((float(dist[e - 1]), pair[1] if len(pair) > 1 else ""))
    else:
        return []
    out = []
    for d, lab in ticks:  # merge the end of one segment with the start of the next
        lab = lab.replace("''", "'")
        if out and abs(out[-1][0] - d) < 1e-9:
            if lab and lab not in out[-1][1].split("|"):
                out[-1] = (d, f"{out[-1][1]}|{lab}" if out[-1][1] else lab)
        else:
            out.append((d, lab))
    return out


def _structure(structure):
    if not isinstance(structure, DataCollection):
        from .io import import_file

        structure = import_file(str(structure)).compute(0)
    masses = np.asarray(structure.particles["Mass"], float) if "Mass" in structure else _ase_masses(structure.types)
    return structure.cell.copy(), np.asarray(structure.positions, float), np.asarray(structure.types), masses


def _ase_masses(types):
    from .radii import element_masses

    return element_masses(types)


# --------------------------------------------------------------------------- VASP
_IBRION = re.compile(rb"IBRION\s*=\s*(-?\d+)")


def is_outcar_phonon(path) -> bool:
    """An OUTCAR of a finite-difference or DFPT phonon run (IBRION = 5, 6, 7 or 8)."""
    from .io.base import head_lines
    from .io.vasp_output import is_outcar

    if not is_outcar(path):
        return False
    for line in head_lines(path, 8000):
        m = _IBRION.search(line)
        if m:
            return int(m.group(1)) in (5, 6, 7, 8)
    return False


_MODE = re.compile(r"^\s*(\d+)\s+(f  =|f/i=)\s*(" + _NUM + r")\s*THz", re.M)


def read_outcar_modes(path) -> PhononModes:
    """Gamma-point modes from "Eigenvectors and eigenvalues of the dynamical matrix" in an
    IBRION=5-8 OUTCAR, sorted by frequency like phonopy (imaginary modes first, negative).
    The structure is the first ionic step; masses come from POMASS."""
    from .io.base import open_binary
    from .io.vasp_output import OutcarReader

    ref = OutcarReader(path).read(0)
    with open_binary(path) as fh:
        text = fh.read().decode("utf-8", "replace")
    start = text.find("Eigenvectors and eigenvalues of the dynamical matrix")
    if start < 0:
        raise ValueError(f"{path}: no 'Eigenvectors and eigenvalues of the dynamical matrix' block "
                         "(phonon OUTCARs come from IBRION = 5, 6, 7 or 8)")
    stop = text.find("Eigenvectors after division by SQRT(mass)", start)
    block = text[start:stop if stop > 0 else len(text)]
    natoms = ref.count
    pos0 = np.asarray(ref.positions)
    freqs, vecs = [], []
    heads = list(_MODE.finditer(block))
    for k, m in enumerate(heads):
        seg = block[m.end():heads[k + 1].start() if k + 1 < len(heads) else len(block)]
        rows = []
        for line in seg.splitlines()[2:]:  # skip the rest of the header line and "X Y Z dx dy dz"
            vals = line.split()
            if len(vals) != 6:
                if rows:
                    break
                continue
            rows.append([float(v) for v in vals])
        rows = np.array(rows).reshape(-1, 6)
        e = np.zeros((natoms, 3))
        if len(rows) == natoms:
            e[:] = rows[:, 3:]
        else:  # selective dynamics: only the free atoms are listed; match them by position
            for r in rows:
                d = (pos0 - r[:3]) @ np.linalg.inv(ref.cell.matrix)
                d = (d - np.round(d)) @ ref.cell.matrix  # minimum image
                e[np.argmin(np.linalg.norm(d, axis=1))] = r[3:]
        f = float(m.group(3))
        freqs.append(-f if m.group(2) == "f/i=" else f)
        vecs.append(e)
    if not freqs:
        raise ValueError(f"{path}: the dynamical matrix block lists no modes")
    order = np.argsort(freqs, kind="stable")
    freqs = np.array(freqs)[order][None]
    eig = np.array(vecs)[order][None].astype(complex)
    masses = _pomass(text, ref.types)
    return PhononModes(ref.cell.copy(), pos0.copy(), np.asarray(ref.types), masses, np.zeros((1, 3)), freqs, eig,
                       None, [], source=f"VASP {os.path.basename(str(path))}")


def _pomass(text, types):
    """Masses per atom from the "Mass of Ions in am" POMASS line (one value per species, in
    POTCAR order); ASE's masses when it is missing."""
    from .io.base import floats

    species = list(dict.fromkeys(str(t) for t in types))
    lines = [ln for ln in re.findall(r"^\s*POMASS\s*=(.*)$", text, re.M) if ";" not in ln]
    if lines:
        vals = floats(lines[-1].encode())
        if len(vals) == len(species):
            table = dict(zip(species, vals))
            return np.array([table[str(t)] for t in types])
    return _ase_masses(types)


def read_modes(path, structure=None) -> PhononModes:
    """Phonon modes from a phonopy yaml file or a VASP phonon OUTCAR."""
    if is_phonopy_yaml(path):
        return read_phonopy_yaml(path, structure=structure)
    return read_outcar_modes(path)
