"""Core data model: a single simulation snapshot (``DataCollection``)."""
from __future__ import annotations

import copy
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

    def __getitem__(self, key):
        """OVITO's cell matrix, 3 x 4: columns a, b, c and the origin (``cell[:, :3]`` is
        ``matrix.T``), so code written for OVITO reads the same numbers. Read-only."""
        m = np.column_stack([self.matrix.T, self.origin])
        m.flags.writeable = False
        return m[key]

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

    def minimum_image(self, vectors: np.ndarray) -> np.ndarray:
        """Displacement vectors (..., 3) wrapped to their nearest periodic image; unchanged
        without a valid cell or periodic directions."""
        v = np.asarray(vectors, float)
        if not (self.is_valid and self.pbc.any()):
            return v
        f = v @ np.linalg.inv(self.matrix)
        return (f - np.round(f) * self.pbc) @ self.matrix

    def copy(self) -> "Cell":
        return Cell(self.matrix.copy(), self.origin.copy(), self.pbc.copy())


_COMPONENTS = {"X": 0, "Y": 1, "Z": 2}


class VoxelGrid:
    """Values on a regular 3D grid spanning ``domain`` (OVITO's VoxelGrid).

    Point ``(i, j, k)`` sits at ``domain.origin + (i/Nx, j/Ny, k/Nz) @ domain.matrix``,
    the VASP CHGCAR / Gaussian cube convention. ``properties`` maps names
    ("Charge Density", "Magnetization", "Potential", "ELF", ...) to arrays of shape
    ``shape`` or ``shape + (k,)``. ``domain.pbc`` says along which axes the grid repeats
    (isosurfaces close across those cell faces). Like particle arrays, property arrays
    are shared between copies: replace them, never write into them.
    """

    def __init__(self, domain: Cell, properties: dict | None = None, shape=None):
        self.domain = domain
        self.properties: dict[str, np.ndarray] = {}
        self.shape = tuple(int(n) for n in shape) if shape is not None else None
        for name, values in (properties or {}).items():
            self[name] = values

    def __setitem__(self, name: str, values) -> None:
        values = np.asarray(values)
        if self.shape is None:
            self.shape = tuple(values.shape[:3])
        if tuple(values.shape[:3]) != self.shape:
            raise ValueError(f"grid property '{name}' has shape {values.shape}, expected {self.shape}")
        self.properties[name] = values

    def __getitem__(self, name: str) -> np.ndarray:
        """A property; also "Name.X"/"Name.Y"/"Name.Z" (or "Name.1") components and, for a
        spin-polarised charge density, "Spin Up"/"Spin Down" = (total +- magnetization) / 2."""
        if name in self.properties:
            return self.properties[name]
        base, _, comp = name.rpartition(".")
        if base in self.properties and (comp.upper() in _COMPONENTS or comp.isdigit()):
            k = _COMPONENTS.get(comp.upper(), None)
            return self.properties[base][..., int(comp) - 1 if k is None else k]
        if name in ("Spin Up", "Spin Down") and "Charge Density" in self.properties \
                and self.properties.get("Magnetization", np.zeros(0)).ndim == 3:
            sign = 1.0 if name == "Spin Up" else -1.0
            return 0.5 * (self.properties["Charge Density"] + sign * self.properties["Magnetization"])
        raise KeyError(f"grid has no property {name!r} (has {list(self.properties)})")

    def __contains__(self, name: str) -> bool:
        try:
            self[name]
        except KeyError:
            return False
        return True

    def names(self) -> list[str]:
        """Property names, including the derived spin channels when available."""
        out = list(self.properties)
        if "Spin Up" in self and "Spin Up" not in out:
            out += ["Spin Up", "Spin Down"]
        return out

    @property
    def voxel_volume(self) -> float:
        return self.domain.volume / float(np.prod(self.shape))

    def integrate(self, name: str) -> float:
        """Sum of values times voxel volume (e.g. the electron count of a charge density)."""
        return float(np.sum(self[name]) * self.voxel_volume)

    def fractional_points(self) -> np.ndarray:
        """Fractional coordinates of every voxel point, shape ``shape + (3,)``."""
        axes = [np.arange(n) / n for n in self.shape]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    def interpolate(self, name: str, points, order: int = 1) -> np.ndarray:
        """Values at Cartesian ``points`` (M x 3) by spline interpolation (``order=1``: trilinear),
        periodic along ``domain.pbc`` axes, clamped elsewhere."""
        from scipy.ndimage import map_coordinates

        values = np.asarray(self[name], float)
        if values.ndim != 3:
            raise ValueError(f"{name!r} is not a scalar grid property; pick a component like '{name}.Z'")
        idx = (self.domain.to_fractional(np.asarray(points, float).reshape(-1, 3)) * self.shape).T
        if self.domain.pbc.all():
            return map_coordinates(values, idx, order=order, mode="grid-wrap")
        for k in range(3):
            if self.domain.pbc[k]:
                idx[k] %= self.shape[k]
        pad = [(0, 1) if p else (0, 0) for p in self.domain.pbc]
        return map_coordinates(np.pad(values, pad, mode="wrap"), idx, order=order, mode="nearest")

    def copy(self) -> "VoxelGrid":
        new = VoxelGrid(self.domain.copy(), shape=self.shape)
        new.properties = dict(self.properties)
        return new

    def __repr__(self) -> str:
        return f"<VoxelGrid {'x'.join(map(str, self.shape))} props={self.names()}>"


def _readonly(array):
    """A read-only view of a numpy array (anything else is returned unchanged)."""
    if isinstance(array, np.ndarray) and array.flags.writeable:
        array = array.view()
        array.flags.writeable = False
    return array


def _same_state(a, b) -> bool:
    """True when ase.Atoms ``a`` and ``b`` have exactly the same numbers, positions, cell and pbc."""
    if a is None or len(a) != len(b):
        return False
    return (np.array_equal(a.numbers, b.numbers) and np.array_equal(a.positions, b.positions)
            and np.array_equal(a.cell, b.cell) and np.array_equal(a.pbc, b.pbc))


class Particles(dict):
    """``DataCollection.particles``: a dict of property name -> array that also answers OVITO's
    ``particles.count`` and ``particles.positions`` (read-only shortcuts, so code written the OVITO
    way reads the same numbers)."""

    @property
    def count(self) -> int:
        p = self.get("Position")
        return 0 if p is None else len(p)

    @property
    def positions(self) -> np.ndarray:
        return self["Position"]


class DataCollection:
    """One snapshot flowing through a pipeline.

    Per-particle data lives in ``particles``: a dict mapping OVITO-style property
    names ('Position', 'Particle Type', 'Particle Identifier', 'Color', 'Radius',
    'Selection', ...) to arrays whose first axis has length N.  Global values go
    into ``attributes`` and tabulated results (e.g. an RDF) into ``tables``.
    ``bonds`` (an :class:`ewald.bonds.Bonds`, or None) holds bonds between particles,
    ``surfaces`` maps names to :class:`ewald.mesh.SurfaceMesh` geometry (polyhedra,
    isosurfaces, grid slices) and ``grids`` maps names to volumetric
    :class:`VoxelGrid` data (CHGCAR, LOCPOT, cube files). ``vectors`` maps particle property
    names to :class:`ewald.arrows.VectorStyle`: those properties are drawn as arrows (VectorVis).

    Modifiers receive a shallow copy, so they must *replace* arrays instead of
    writing into them in place; this keeps cached source frames untouched.
    """

    def __init__(self, positions=None, types=None, cell: Cell | None = None):
        self.particles = {}
        self.attributes: dict[str, object] = {}
        self.tables: dict[str, dict[str, np.ndarray]] = {}
        self.bonds = None
        self.surfaces: dict = {}
        self.grids: dict[str, VoxelGrid] = {}
        self.vectors: dict = {}
        self.cell = cell if cell is not None else Cell()
        if positions is not None:
            self.particles["Position"] = np.asarray(positions, dtype=float).reshape(-1, 3)
        if types is not None:
            self.particles["Particle Type"] = np.asarray(types).astype(str)

    @property
    def particles(self) -> Particles:
        """Property name -> per-particle array (a dict; any dict assigned is kept as :class:`Particles`)."""
        return self._particles

    @particles.setter
    def particles(self, value):
        self._particles = value if type(value) is Particles else Particles(value)

    # ---- convenience accessors -------------------------------------------------
    @property
    def count(self) -> int:
        p = self.particles.get("Position")
        return 0 if p is None else len(p)

    def __len__(self) -> int:
        return self.count

    @property
    def nbytes(self) -> int:
        """Bytes held by the per-particle arrays (what a cached frame costs)."""
        return sum(getattr(a, "nbytes", 0) for a in self.particles.values())

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
        # a table that carries labels or attributes (an ewald.verbs.Table: levels to draw) keeps them
        new.tables = {k: dict(v) if type(v) is dict else copy.copy(v) for k, v in self.tables.items()}
        new.bonds = None if getattr(self, "bonds", None) is None else self.bonds.copy()
        new.surfaces = dict(getattr(self, "surfaces", {}))
        new.grids = {k: v.copy() for k, v in getattr(self, "grids", {}).items()}
        new.vectors = dict(getattr(self, "vectors", {}))
        new.cell = self.cell.copy()
        return new

    def take(self, index) -> "DataCollection":
        """Return a copy containing only the particles selected by ``index``."""
        new = self.copy()
        new.particles = {k: v[index] for k, v in self.particles.items()}
        if new.bonds is not None:
            new.bonds = new.bonds.remap_particles(index, self.count)
        return new

    def delete(self, mask) -> "DataCollection":
        return self.take(~np.asarray(mask, dtype=bool))

    def freeze(self) -> "DataCollection":
        """Make the per-particle and grid arrays read-only views, in place; returns ``self``.

        The pipeline freezes the frames it caches, so a modifier that writes into an array
        (``data.positions += d``) fails loudly instead of silently changing the cached source
        frame. Replace arrays instead: ``data.positions = data.positions + d``. Only this
        collection's containers change; whoever else holds the original arrays can still write
        them. Arrays without numpy's ``writeable`` flag (CuPy) are left as they are."""
        self.particles = {k: _readonly(v) for k, v in self.particles.items()}
        for grid in getattr(self, "grids", {}).values():
            grid.properties = {k: _readonly(v) for k, v in grid.properties.items()}
        return self

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
        from ase.data import chemical_symbols

        # one symbol per distinct element, not per atom (atoms.get_chemical_symbols is a Python loop)
        elements, which = np.unique(atoms.numbers, return_inverse=True)
        symbols = np.asarray([chemical_symbols[z] for z in elements]).astype(str)[which.ravel()]
        data = cls(atoms.get_positions(), symbols, cell)
        skip = {"positions", "numbers"}
        for key, arr in atoms.arrays.items():
            if key not in skip and len(arr) == len(atoms):
                name = {"momenta": "Momentum", "masses": "Mass", "initial_charges": "Charge",
                        "initial_magmoms": "Magnetic Moment", "forces": "Force",
                        "id": "Particle Identifier"}.get(key, key)
                data.particles[name] = np.asarray(arr)
        calc = getattr(atoms, "calc", None)
        if calc is not None:
            # A calculator attached to these very atoms (what every ASE reader returns): take its
            # results directly; get_property's state check costs ~0.4 s per 1M atoms (np.isclose).
            fresh = _same_state(getattr(calc, "atoms", None), atoms) and isinstance(getattr(calc, "results", None), dict)
            for key, name in (("energy", "Energy"), ("forces", "Force"), ("stress", "Stress")):
                try:
                    val = calc.results.get(key) if fresh else calc.get_property(key, atoms, allow_calculation=False)
                except Exception:
                    val = None
                if val is None:
                    continue
                if fresh and isinstance(val, np.ndarray):
                    val = val.copy()  # get_property returns copies too
                if name == "Force":
                    data.particles["Force"] = np.asarray(val)
                else:
                    data.attributes[name] = val
        data.attributes.update({k: v for k, v in atoms.info.items() if np.isscalar(v)})
        return data

    def to_pymatgen(self):
        """This frame as a pymatgen ``Structure`` (a ``Molecule`` when there is no cell), via
        :meth:`to_ase`."""
        AseAtomsAdaptor = _pymatgen_adaptor("to_pymatgen")
        atoms = self.to_ase()
        if self.cell.is_valid:
            return AseAtomsAdaptor.get_structure(atoms)
        return AseAtomsAdaptor.get_molecule(atoms)

    @classmethod
    def from_pymatgen(cls, structure) -> "DataCollection":
        """A DataCollection from a pymatgen ``Structure`` or ``Molecule`` (via ASE)."""
        return cls.from_ase(_pymatgen_adaptor("from_pymatgen").get_atoms(structure))

    def apply(self, modifier) -> "DataCollection":
        """Run ``modifier`` (a :class:`~ewald.pipeline.Modifier` or an ``f(frame, data)`` function) on
        this frame alone and return the result, as OVITO's ``data.apply(mod)``; ``self`` stays as it
        was. Steps that read other frames (MSD, displacements) need a :class:`~ewald.pipeline.Pipeline`."""
        from .io import StaticSource
        from .pipeline import Pipeline

        pipe = Pipeline(StaticSource(self))
        pipe.cache_outputs = False
        pipe.add(modifier)
        return pipe.compute(0)

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
        if getattr(self, "bonds", None) is not None:
            lines.append(f"bonds: {self.bonds.count} (properties {list(self.bonds.properties)})")
        for name, mesh in getattr(self, "surfaces", {}).items():
            lines.append(f"surface {name!r}: {mesh.vertex_count} vertices, {mesh.face_count} faces, "
                         f"area {mesh.area:.4g} A^2")
        for name, style in getattr(self, "vectors", {}).items():
            from .arrows import summary_line

            lines.append(summary_line(name, style))
        for name, table in self.tables.items():
            lines.append(f"table {name!r}: columns {list(table)}")
        for name, grid in getattr(self, "grids", {}).items():
            lines.append(f"grid {name!r}: {'x'.join(map(str, grid.shape))} voxels")
            for prop in grid.names():
                a = np.asarray(grid[prop])
                comp = f" [{a.shape[3]} components]" if a.ndim == 4 else ""
                lines.append(f"  {prop!r}{comp}: min {a.min():.4g}, max {a.max():.4g}, mean {a.mean():.4g}")
        return "\n".join(lines)

    def _repr_png_(self):  # Jupyter shows the frame as a picture (ewald.display.options)
        from .display import data_png

        return data_png(self)

    def __repr__(self) -> str:
        props = ", ".join(self.particles)
        return f"<DataCollection N={self.count} props=[{props}]>"


def _ase_symbol(t: str) -> str:
    from ase.data import chemical_symbols

    return t if t in chemical_symbols else "X"


def _pymatgen_adaptor(what: str):
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        raise ImportError(f"{what} needs pymatgen: pip install pymatgen") from exc
    return AseAtomsAdaptor
