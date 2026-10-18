"""Core data model: a single simulation snapshot (``DataCollection``)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cell:
    """Simulation cell. ``matrix`` rows are the cell vectors a, b, c (Angstrom)."""

    matrix: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pbc: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=bool))

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=float).reshape(3, 3)
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)
        self.pbc = np.broadcast_to(np.asarray(self.pbc, dtype=bool), (3,)).copy()

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.matrix)))

    @property
    def is_valid(self) -> bool:
        return self.volume > 1e-12

    def perpendicular_widths(self) -> np.ndarray:
        """Distance between opposite cell faces along each cell direction."""
        a, b, c = self.matrix
        v = self.volume
        return np.array([v / np.linalg.norm(np.cross(b, c)),
                         v / np.linalg.norm(np.cross(c, a)),
                         v / np.linalg.norm(np.cross(a, b))])

    def to_fractional(self, positions: np.ndarray) -> np.ndarray:
        return (np.asarray(positions) - self.origin) @ np.linalg.inv(self.matrix)

    def to_cartesian(self, fractional: np.ndarray) -> np.ndarray:
        return np.asarray(fractional) @ self.matrix + self.origin

    def wrap(self, positions: np.ndarray) -> np.ndarray:
        f = self.to_fractional(positions)
        f[:, self.pbc] %= 1.0
        return self.to_cartesian(f)

    def copy(self) -> "Cell":
        return Cell(self.matrix.copy(), self.origin.copy(), self.pbc.copy())


class DataCollection:
    """One snapshot flowing through a pipeline.

    Per-particle data lives in ``particles``: a dict mapping OVITO-style property
    names ('Position', 'Particle Type', 'Particle Identifier', 'Color', 'Radius',
    'Selection', ...) to arrays whose first axis has length N.  Global values go
    into ``attributes`` and tabulated results (e.g. an RDF) into ``tables``.

    Modifiers receive a shallow copy, so they must *replace* arrays instead of
    writing into them in place; this keeps cached source frames untouched.
    """

    def __init__(self, positions=None, types=None, cell: Cell | None = None):
        self.particles: dict[str, np.ndarray] = {}
        self.attributes: dict[str, object] = {}
        self.tables: dict[str, dict[str, np.ndarray]] = {}
        self.cell = cell if cell is not None else Cell()
        if positions is not None:
            self.particles["Position"] = np.asarray(positions, dtype=float).reshape(-1, 3)
        if types is not None:
            self.particles["Particle Type"] = np.asarray(types).astype(str)

    # ---- convenience accessors -------------------------------------------------
    @property
    def count(self) -> int:
        p = self.particles.get("Position")
        return 0 if p is None else len(p)

    def __len__(self) -> int:
        return self.count

    @property
    def positions(self) -> np.ndarray:
        return self.particles["Position"]

    @positions.setter
    def positions(self, value):
        self.particles["Position"] = np.asarray(value, dtype=float).reshape(-1, 3)

    @property
    def types(self) -> np.ndarray:
        t = self.particles.get("Particle Type")
        return np.full(self.count, "X") if t is None else t

    @property
    def selection(self) -> np.ndarray:
        s = self.particles.get("Selection")
        return np.zeros(self.count, dtype=bool) if s is None else s.astype(bool)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.particles[name]

    def __setitem__(self, name: str, value) -> None:
        value = np.asarray(value)
        if value.ndim == 0:
            value = np.full(self.count, value)
        if len(value) != self.count:
            raise ValueError(f"property '{name}' has length {len(value)}, expected {self.count}")
        self.particles[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self.particles

    # ---- structural operations -------------------------------------------------
    def copy(self) -> "DataCollection":
        """Shallow copy: containers are new, arrays are shared (copy-on-write)."""
        new = DataCollection.__new__(DataCollection)
        new.particles = dict(self.particles)
        new.attributes = dict(self.attributes)
        new.tables = {k: dict(v) for k, v in self.tables.items()}
        new.cell = self.cell.copy()
        return new

    def take(self, index) -> "DataCollection":
        """Return a copy containing only the particles selected by ``index``."""
        new = self.copy()
        new.particles = {k: v[index] for k, v in self.particles.items()}
        return new

    def delete(self, mask) -> "DataCollection":
        return self.take(~np.asarray(mask, dtype=bool))

    # ---- interoperability --------------------------------------------------------
    def to_ase(self):
        from ase import Atoms

        cell = self.cell.matrix if self.cell.is_valid else None
        atoms = Atoms(symbols=[_ase_symbol(t) for t in self.types],
                      positions=self.positions - self.cell.origin,
                      cell=cell, pbc=self.cell.pbc if cell is not None else False)
        for name, arr in self.particles.items():
            if name not in ("Position", "Particle Type") and arr.dtype.kind in "biuf":
                atoms.arrays[name.replace(" ", "_")] = np.array(arr)
        return atoms

    @classmethod
    def from_ase(cls, atoms) -> "DataCollection":
        cell = Cell(np.array(atoms.cell), np.zeros(3), np.array(atoms.pbc))
        data = cls(atoms.get_positions(), atoms.get_chemical_symbols(), cell)
        skip = {"positions", "numbers"}
        for key, arr in atoms.arrays.items():
            if key not in skip and len(arr) == len(atoms):
                name = {"momenta": "Momentum", "masses": "Mass", "initial_charges": "Charge",
                        "initial_magmoms": "Magnetic Moment", "forces": "Force",
                        "id": "Particle Identifier"}.get(key, key)
                data.particles[name] = np.asarray(arr)
        calc = getattr(atoms, "calc", None)
        if calc is not None:
            for key, name in (("energy", "Energy"), ("forces", "Force"), ("stress", "Stress")):
                try:
                    val = calc.get_property(key, atoms, allow_calculation=False)
                except Exception:
                    val = None
                if val is None:
                    continue
                if name == "Force":
                    data.particles["Force"] = np.asarray(val)
                else:
                    data.attributes[name] = val
        data.attributes.update({k: v for k, v in atoms.info.items() if np.isscalar(v)})
        return data

    def summary(self, max_values: int = 6) -> str:
        """Human- (and LLM-) readable overview: cell, every property with dtype, shape and
        range (or most common values), attributes and tables. Cheap; print it before
        writing code against an unfamiliar frame."""
        from collections import Counter

        lines = [f"DataCollection: {self.count} particles"]
        if self.cell.is_valid:
            lengths = np.linalg.norm(self.cell.matrix, axis=1)
            lines.append(f"cell: a,b,c = {', '.join(f'{x:.4g}' for x in lengths)} A, volume {self.cell.volume:.6g} A^3, "
                         f"pbc {self.cell.pbc.tolist()} (rows of cell.matrix are the cell vectors)")
        else:
            lines.append("cell: none (non-periodic)")
        lines.append("particles:")
        for name, arr in self.particles.items():
            a = np.asarray(arr)
            shape = "x".join(map(str, a.shape))
            if a.dtype.kind in "biuf" and a.size:
                rng = f"min {np.nanmin(a):.4g}, max {np.nanmax(a):.4g}"
                if a.ndim == 1:
                    rng += f", mean {np.nanmean(a):.4g}"
                    if a.dtype.kind in "biu" and len(np.unique(a)) <= max_values:
                        rng += f", values {sorted(Counter(a.tolist()).items())}"
                lines.append(f"  {name!r}: {a.dtype} [{shape}] {rng}")
            else:
                common = Counter(str(x) for x in a.ravel()).most_common(max_values)
                lines.append(f"  {name!r}: {a.dtype} [{shape}] {common}")
        scalars = {k: v for k, v in self.attributes.items() if np.isscalar(v)}
        if scalars:
            lines.append("attributes: " + ", ".join(f"{k}={v!r}" for k, v in scalars.items()))
        for name, table in self.tables.items():
            lines.append(f"table {name!r}: columns {list(table)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        props = ", ".join(self.particles)
        return f"<DataCollection N={self.count} props=[{props}]>"


def _ase_symbol(t: str) -> str:
    from ase.data import chemical_symbols

    return t if t in chemical_symbols else "X"
