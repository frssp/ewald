"""Electronic structure from VASP: density of states (total and projected) and band structure.

:class:`ElectronicStructure` holds what vasprun.xml (``<dos>``, ``<eigenvalues>``,
``<projected>``) and vaspout.h5 (``results/electron_dos``, ``results/electron_eigenvalues``,
``results/projectors``) store about the final structure:

- the Fermi energy (eV),
- the total DOS ``dos`` (states/eV, shape ``(nspin, nE)``) on ``dos_energies``,
- the projected DOS ``pdos`` (``(natoms, nspin, norbitals, nE)``, LORBIT >= 10) with the
  orbital names ``orbitals`` (``s py pz px dxy ...``, or ``s p d`` for LORBIT = 10),
- the eigenvalues (``(nspin, nk, nbands)``), occupations, k-points (reduced) and weights,
- the band-path distances (1/Angstrom, without 2 pi, as phonopy and py4vasp), the
  high-symmetry ticks ``[(distance, label)]`` and the indices where the path jumps,
- the band projections (``(nspin, nk, nbands, natoms, norbitals)``), read when first used.

``nspin`` is 1 (also for non-collinear runs, where only the total is kept) or 2. Energies
in tables and plots are relative to the Fermi energy. In :meth:`dos_table` the spin-down
channel is negated, so it plots below the axis. The readers put the object in
``data.attributes["Electronic Structure"]``, the tables in ``data.tables["DOS"]`` and
``data.tables["Bands"]`` and the gap in the attributes ``Fermi Energy`` and ``Band Gap``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# orbital names by angular momentum, as VASP writes them (lm-decomposed, LORBIT = 11)
_L = {"s": ("s",), "p": ("p", "py", "pz", "px"),
      "d": ("d", "dxy", "dyz", "dz2", "dxz", "dx2", "x2-y2", "dx2-y2"),
      "f": ("f", "fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2", "fx3", "f-3", "f-2", "f-1", "f0", "f1", "f2", "f3")}
_GREEK = {"GAMMA": "Γ", "G": "Γ", "SIGMA": "Σ", "DELTA": "Δ", "LAMBDA": "Λ"}


def pretty_label(label: str) -> str:
    """High-symmetry point label for a plot: ``\\Gamma``, ``GAMMA``, ``G`` -> Γ; ``S_1`` -> S₁."""
    s = label.strip().strip("$").replace("\\mathrm", "").replace("{", "").replace("}", "")
    s = _GREEK.get(s.lstrip("\\").upper(), s.lstrip("\\"))
    return re.sub(r"_(\d)", lambda m: chr(0x2080 + int(m.group(1))), s)


def orbital_mask(orbitals: list[str], which) -> np.ndarray:
    """Boolean mask over ``orbitals`` for ``which``: None (all), ``"s"``/``"p"``/``"d"``/``"f"``
    (every orbital of that l), an orbital name (``"dxy"``) or a list of these."""
    names = [o.strip() for o in orbitals]
    if which is None:
        return np.ones(len(names), bool)
    picks = [which] if isinstance(which, str) else list(which)
    mask = np.zeros(len(names), bool)
    for w in picks:
        w = w.strip()
        hit = np.array([o == w or (w in _L and o in _L[w]) for o in names], bool)
        if not hit.any():
            raise KeyError(f"no orbital {w!r}; the projections have {', '.join(names)}")
        mask |= hit
    return mask


@dataclass(repr=False)
class ElectronicStructure:
    """Density of states and band structure of one calculation (see the module docstring)."""

    types: np.ndarray                         # (natoms,) element names, for per-element sums
    efermi: float = 0.0
    dos_energies: np.ndarray | None = None    # (nE,) absolute, eV
    dos: np.ndarray | None = None             # (nspin, nE) states/eV
    pdos: np.ndarray | None = None            # (natoms, nspin, norb, nE)
    orbitals: list = field(default_factory=list)
    kpoints: np.ndarray | None = None         # (nk, 3) reduced
    weights: np.ndarray | None = None         # (nk,)
    eigenvalues: np.ndarray | None = None     # (nspin, nk, nb) absolute, eV
    occupations: np.ndarray | None = None     # (nspin, nk, nb)
    distances: np.ndarray | None = None       # (nk,) along the path, 1/Angstrom
    ticks: list = field(default_factory=list)  # [(distance, label)]
    breaks: list = field(default_factory=list)  # k-point indices that start a new, disconnected segment
    source: str = ""
    projections_loader: Callable | None = None  # () -> (nspin, nk, nb, natoms, norb) or None
    _projections: np.ndarray | None = None

    # ------------------------------------------------------------------ properties
    @property
    def has_dos(self) -> bool:
        return self.dos is not None and self.dos_energies is not None

    @property
    def has_bands(self) -> bool:
        return self.eigenvalues is not None and self.eigenvalues.shape[1] > 0

    @property
    def num_spins(self) -> int:
        a = self.dos if self.dos is not None else self.eigenvalues
        return 0 if a is None else a.shape[0]

    @property
    def projections(self):
        """Band projections ``(nspin, nk, nb, natoms, norb)`` (read on first use), or None."""
        if self._projections is None and self.projections_loader is not None:
            loader, self.projections_loader = self.projections_loader, None
            self._projections = loader()
        return self._projections

    @property
    def has_projections(self) -> bool:
        return self._projections is not None or self.projections_loader is not None

    # ------------------------------------------------------------------ DOS
    def pdos_of(self, atoms=None, orbitals=None) -> np.ndarray:
        """Projected DOS ``(nspin, nE)`` summed over ``atoms`` (indices, a boolean mask or None
        for all) and ``orbitals`` (see :func:`orbital_mask`)."""
        if self.pdos is None:
            raise ValueError("no projected DOS (run VASP with LORBIT >= 10)")
        sel = self.pdos if atoms is None else self.pdos[_atom_index(atoms, len(self.pdos))]
        return sel[:, :, orbital_mask(self.orbitals, orbitals)].sum(axis=(0, 2))

    def dos_table(self) -> dict:
        """``Energy`` (E - E_F) and ``Total``, then one column per element and per element and
        l (``Fe d``) when there is a projected DOS. Spin-polarized runs get ``(up)`` and
        ``(down)`` columns, the down channel negated."""
        e = self.dos_energies - self.efermi
        out = {"Energy": np.asarray(e, float)}

        def add(name, values):
            if self.num_spins == 2:
                out[f"{name} (up)"] = values[0].copy()
                out[f"{name} (down)"] = -values[1]
            else:
                out[name] = values[0].copy()

        add("Total", self.dos)
        if self.pdos is not None:
            ls = [l for l in "spdf" if orbital_mask_any(self.orbitals, l)]
            for el in dict.fromkeys(self.types):
                atoms = np.asarray(self.types) == el
                add(el, self.pdos_of(atoms))
            for el in dict.fromkeys(self.types):
                atoms = np.asarray(self.types) == el
                for l in ls if len(ls) > 1 else []:
                    add(f"{el} {l}", self.pdos_of(atoms, l))
        return out

    # ------------------------------------------------------------------ bands
    def band_table(self) -> dict:
        """``Distance`` along the path (or ``K-point`` without a path) and one column per band
        (E - E_F, eV); spin-polarized runs get ``Band n (up)`` and ``Band n (down)``."""
        x = self.distances if self.distances is not None else np.arange(self.eigenvalues.shape[1], dtype=float)
        out = {"Distance" if self.distances is not None else "K-point": np.asarray(x, float)}
        e = self.eigenvalues - self.efermi
        for b in range(e.shape[2]):
            if self.num_spins == 2:
                out[f"Band {b + 1} (up)"] = e[0, :, b].copy()
                out[f"Band {b + 1} (down)"] = e[1, :, b].copy()
            else:
                out[f"Band {b + 1}"] = e[0, :, b].copy()
        return out

    def band_weights(self, atoms=None, orbitals=None) -> np.ndarray:
        """Weight of ``atoms`` and ``orbitals`` in every band state, ``(nspin, nk, nb)`` (fat bands)."""
        p = self.projections
        if p is None:
            raise ValueError("no band projections (run VASP with LORBIT >= 10)")
        sel = p if atoms is None else p[:, :, :, _atom_index(atoms, p.shape[3])]
        return sel[..., orbital_mask(self.orbitals, orbitals)].sum(axis=(3, 4))

    def band_gap(self, occupied: float = 0.5) -> dict | None:
        """``{"gap", "vbm", "cbm", "direct"}`` (eV, absolute) from the eigenvalues and
        occupations (a state with occupation above ``occupied`` (per spin channel) is filled),
        ``gap`` 0 for a metal; None without eigenvalues."""
        if not self.has_bands:
            return None
        e = self.eigenvalues
        occ = self.occupations if self.occupations is not None else (e <= self.efermi).astype(float)
        full = occ > occupied
        if not full.any() or full.all():
            return None
        vbm, cbm = float(e[full].max()), float(e[~full].min())
        if cbm <= vbm:
            return {"gap": 0.0, "vbm": vbm, "cbm": cbm, "direct": False}
        kv = np.argwhere(full & (e == vbm))[:, 1]
        kc = np.argwhere(~full & (e == cbm))[:, 1]
        direct = bool(np.intersect1d(kv, kc).size)
        if not direct and self.kpoints is not None:  # the same k-point listed twice (path ends)
            direct = bool(np.isclose(self.kpoints[kv][:, None], self.kpoints[kc][None]).all(-1).any())
        return {"gap": cbm - vbm, "vbm": vbm, "cbm": cbm, "direct": direct}

    # ------------------------------------------------------------------ for readers
    def attach(self, data):
        """Put this object, its tables and the Fermi energy / band gap into ``data``."""
        data.attributes["Electronic Structure"] = self
        data.attributes["Fermi Energy"] = float(self.efermi)
        tables = self._tables()
        data.tables.update(tables)
        gap = self.band_gap()
        if gap is not None:
            data.attributes["Band Gap"] = gap["gap"]
        return data

    def _tables(self) -> dict:
        if not hasattr(self, "_table_cache"):
            t = {}
            if self.has_dos:
                t["DOS"] = self.dos_table()
            if self.has_bands and self.eigenvalues.shape[1] > 1:
                t["Bands"] = self.band_table()
            self._table_cache = t
        return {k: dict(v) for k, v in self._table_cache.items()}

    def __repr__(self) -> str:  # shown in the GUI attribute inspector
        parts = []
        if self.has_dos:
            parts.append(f"DOS {len(self.dos_energies)} points" + (" + PDOS" if self.pdos is not None else ""))
        if self.has_bands:
            parts.append(f"{self.eigenvalues.shape[1]} k-points x {self.eigenvalues.shape[2]} bands")
        spin = " spin-polarized" if self.num_spins == 2 else ""
        return f"<ElectronicStructure{spin} {', '.join(parts)}, E_F {self.efermi:.4f} eV, {self.source}>"


def orbital_mask_any(orbitals, which) -> bool:
    try:
        return bool(orbital_mask(orbitals, which).any())
    except KeyError:
        return False


def _atom_index(atoms, n):
    a = np.asarray(atoms)
    if a.dtype == bool:
        if a.shape != (n,):
            raise ValueError(f"atom mask has shape {a.shape}, expected ({n},)")
        return np.flatnonzero(a)
    return a.astype(int).reshape(-1)


# ---------------------------------------------------------------------- k-point paths
def path_distances(kpoints, lattice, segment: int | None = None, starts=None) -> tuple[np.ndarray, list]:
    """Distance along the path (1/Angstrom, without 2 pi) of reduced ``kpoints`` in the
    reciprocal cell of ``lattice`` (rows are the cell vectors). With ``segment`` (points per
    line, VASP line mode), or ``starts`` (the first index of every line after the first), the
    distance does not grow from the end of one line to the start of the next. Returns
    ``(distances, breaks)``, ``breaks`` the indices that start a line which does not continue
    the previous one."""
    k = np.asarray(kpoints, float) @ np.linalg.inv(np.asarray(lattice, float)).T
    step = np.r_[0.0, np.linalg.norm(np.diff(k, axis=0), axis=1)]
    breaks = []
    if starts is None and segment and segment > 1 and len(k) % segment == 0:
        starts = range(segment, len(k), segment)
    if starts is not None:
        for s in starts:
            if step[s] > 1e-6:
                breaks.append(int(s))
            step[s] = 0.0
    return np.cumsum(step), breaks


def line_ticks(distances, segment: int, labels: list[str] | None, kpoints=None) -> list:
    """Ticks at the ends of every line of a line-mode path. ``labels`` are the two labels of
    each line in order (KPOINTS); without them the reduced coordinates label the ends."""
    n = len(distances)
    ends = []
    for s in range(0, n, segment):
        ends += [s, min(s + segment, n) - 1]
    ticks = []
    for i, idx in enumerate(ends):
        if labels and i < len(labels) and labels[i]:
            lab = pretty_label(labels[i])
        elif kpoints is not None:
            lab = _coord_label(kpoints[idx])
        else:
            lab = ""
        ticks.append((float(distances[idx]), lab))
    return merge_ticks(ticks)


def merge_ticks(ticks) -> list:
    """``(distance, label)`` ticks with the ticks at one distance merged: ``K|U`` where a line
    ends at K and the next starts at U."""
    out = []
    for d, lab in ticks:
        if out and abs(out[-1][0] - d) < 1e-9:
            if lab and lab != out[-1][1]:
                out[-1] = (d, f"{out[-1][1]}|{lab}" if out[-1][1] else lab)
        else:
            out.append((d, lab))
    return out


def _coord_label(k) -> str:
    from fractions import Fraction

    if np.allclose(k, 0):
        return "Γ"
    return "(" + " ".join(str(Fraction(float(x)).limit_denominator(12)) for x in k) + ")"


def kpoints_file_labels(path) -> tuple[int, list[str]] | None:
    """``(points per line, [label of every line end])`` from a line-mode KPOINTS file, else None."""
    try:
        with open(path, errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None
    if len(lines) < 5 or not lines[2].strip().lower().startswith("l"):
        return None
    try:
        n = int(lines[1].split()[0])
    except (ValueError, IndexError):
        return None
    labels = []
    for line in lines[4:]:
        if not line.strip():
            continue
        body, _, comment = line.partition("!")
        if len(body.split()) < 3:
            continue
        labels.append(comment.strip())
    return n, labels
