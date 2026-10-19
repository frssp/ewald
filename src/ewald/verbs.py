"""Analysis verbs: one call adds the analysis step and returns its table, for notebooks and scripts::

    from ewald import load, rdf, msd
    pipe = load("XDATCAR")
    g = rdf(pipe, cutoff=6)      # adds CoordinationAnalysis, returns the g(r) table
    g.x, g["g(r)"]               # columns are numpy arrays; pandas.DataFrame(g), g.plot()
    d = msd(pipe)                # MeanSquaredDisplacement over the trajectory
    cna(pipe).counts             # {"OTHER": 0, "FCC": 500, ...}

Calling a verb again replaces the step it added before (in a notebook, re-running the cell
does the same), so changing ``cutoff=`` and running again never stacks duplicate steps.
"""
from __future__ import annotations

import numpy as np

#: Insert › Analysis in the GUI: (verb, menu label, the code cell it inserts and runs)
VERB_CELLS = [
    ("rdf", "Radial distribution g(r)", "g = rdf(pipe, cutoff=6.0)\ng"),
    ("sq", "Structure factor S(q)", "s = sq(pipe, q_max=10.0)\ns"),
    ("xrd", "Powder diffraction (XRD)", "pattern = xrd(pipe)\npattern"),
    ("bond_angles", "Bond angles", "angles = bond_angles(pipe)\nangles"),
    ("cna", "Common neighbor analysis", "structures = cna(pipe)\nstructures"),
    ("ptm", "Polyhedral template matching", "structures = ptm(pipe)\nstructures"),
    ("voronoi", "Voronoi coordination", "vor = voronoi(pipe)\nvor"),
    ("clusters", "Clusters", "cl = clusters(pipe, cutoff=3.2)\ncl"),
    ("msd", "Mean squared displacement", "d = msd(pipe)\nd"),
    ("vacf", "Velocity autocorrelation", "v = vacf(pipe)\nv"),
]


def verb_cell(verb: str) -> str:
    """The code cell Insert › Analysis adds for ``verb`` (with its import)."""
    code = next(c for v, _label, c in VERB_CELLS if v == verb)
    return f"from ewald import {verb}\n{code}"


__all__ = ["Table", "bond_angles", "clusters", "cna", "msd", "ptm", "rdf", "sq", "vacf", "voronoi", "xrd"]


class Table(dict):
    """A data table (``{column: array}``, the first column is x) with where it came from."""

    def __init__(self, name: str, columns: dict, pipe=None, frame: int = 0, labels=None):
        super().__init__((k, np.asarray(v)) for k, v in columns.items())
        self.name, self.pipe, self.frame = name, pipe, frame
        self.labels = None if labels is None else list(labels)  # names of the rows (structure types)

    @property
    def counts(self) -> dict:
        """``{row: Count}`` for a counting table (structure types, coordination numbers, clusters);
        rows are named by ``labels`` when the table has them, else by the first column."""
        col = "Count" if "Count" in self else next((c for c in self if c.endswith("Size")), None)
        if col is None:
            raise KeyError(f"{self.name!r} has no Count column")
        keys = self.labels if self.labels is not None else [k.item() for k in self.x]
        return {k: int(v) for k, v in zip(keys, self[col])}

    @property
    def x(self) -> np.ndarray:
        return next(iter(self.values()))

    @property
    def columns(self) -> list[str]:
        return list(self)

    def plot(self, y: str | None = None):
        """Draw the table with matplotlib (:func:`ewald.views.show`)."""
        from .views import show

        show(self.pipe, table=self.name, y=y, frame=self.frame)

    def _repr_png_(self):  # the curve, in Jupyter and the GUI notebook
        from .display import table_png

        return table_png(self)

    def __repr__(self):
        n = len(self.x) if self else 0
        if self.labels is not None:
            return f"<Table {self.name!r}: {', '.join(f'{k} {v}' for k, v in self.counts.items())}>"
        return f"<Table {self.name!r}: {n} rows; {', '.join(self.columns)}>"


def _verb(pipe, cls, table: str, frame: int, labels=None, **params) -> Table:
    """Add ``cls(**params)`` to ``pipe`` (replacing the one an earlier verb call added) and return
    ``table`` at ``frame``."""
    mod = cls(**params)
    mod._verb = cls.__name__
    for i, m in enumerate(pipe.modifiers):
        if getattr(m, "_verb", None) == cls.__name__:
            pipe.modifiers[i] = mod
            break
    else:
        pipe.add(mod)
    data = pipe.compute(frame)
    if table not in data.tables:
        raise RuntimeError(f"{cls.__name__} wrote no {table!r} table: {mod.status or 'see the step for details'}")
    columns = data.tables[table]
    if callable(labels):
        labels = labels(mod, columns)
    return Table(table, columns, pipe, frame, labels)


def rdf(pipe, cutoff: float = 6.0, bins: int = 200, partial: bool = False, frame: int = 0) -> Table:
    """Radial distribution function g(r) up to ``cutoff`` (Angstrom), per type pair with
    ``partial=True``: adds :class:`~ewald.modifiers.CoordinationAnalysis` (which also writes the
    ``Coordination`` property) and returns its ``coordination-rdf`` table."""
    from .modifiers import CoordinationAnalysis

    return _verb(pipe, CoordinationAnalysis, "coordination-rdf", frame, cutoff=cutoff, number_of_bins=bins,
                 partial=partial)


def msd(pipe, frame: int = 0, **kwargs) -> Table:
    """Mean squared displacement over the trajectory (per type, with the Einstein fit):
    :class:`~ewald.modifiers.MeanSquaredDisplacement` with ``kwargs``; the ``msd`` table."""
    from .modifiers import MeanSquaredDisplacement

    return _verb(pipe, MeanSquaredDisplacement, "msd", frame, **kwargs)


def vacf(pipe, frame: int = 0, **kwargs) -> Table:
    """Velocity autocorrelation over the trajectory:
    :class:`~ewald.modifiers.VelocityAutocorrelation` with ``kwargs``; the ``vacf`` table."""
    from .modifiers import VelocityAutocorrelation

    return _verb(pipe, VelocityAutocorrelation, "vacf", frame, **kwargs)


def sq(pipe, q_max: float = 10.0, bins: int = 200, frame: int = 0, **kwargs) -> Table:
    """Structure factor S(q) up to ``q_max`` (1/Angstrom): :class:`~ewald.modifiers.StructureFactor`
    with ``kwargs`` (``weighting=``, ``experimental=``, ...); the ``structure-factor`` table."""
    from .modifiers import StructureFactor

    return _verb(pipe, StructureFactor, "structure-factor", frame, q_max=q_max, number_of_bins=bins, **kwargs)


def xrd(pipe, frame: int = 0, **kwargs) -> Table:
    """Powder diffraction pattern: :class:`~ewald.modifiers.PowderDiffraction` with ``kwargs``
    (wavelength, two-theta range, ...); the ``diffraction`` table."""
    from .modifiers import PowderDiffraction

    return _verb(pipe, PowderDiffraction, "diffraction", frame, **kwargs)


def bond_angles(pipe, bins: int = 180, frame: int = 0, **kwargs) -> Table:
    """Bond-angle distribution of the pipeline's bonds (add ``CreateBonds`` first):
    :class:`~ewald.modifiers.BondAngleDistribution`; the ``bond-angle-distr`` table."""
    from .modifiers import BondAngleDistribution

    return _verb(pipe, BondAngleDistribution, "bond-angle-distr", frame, bins=bins, **kwargs)


def _structure_names(mod, columns):
    return [mod._names[int(t)] for t in columns["Structure Type"]]


def cna(pipe, mode: str = "adaptive", frame: int = 0, **kwargs) -> Table:
    """Common neighbor analysis (FCC, HCP, BCC, ICO): adds
    :class:`~ewald.modifiers.CommonNeighborAnalysis` (``mode=`` adaptive, interval or fixed) and
    returns its ``structures`` table; ``.counts`` is ``{"OTHER": n, "FCC": n, ...}``."""
    from .modifiers import CommonNeighborAnalysis

    return _verb(pipe, CommonNeighborAnalysis, "structures", frame, _structure_names, mode=mode, **kwargs)


def ptm(pipe, frame: int = 0, **kwargs) -> Table:
    """Polyhedral template matching: :class:`~ewald.modifiers.PolyhedralTemplateMatching` with
    ``kwargs`` (``types=``, ``rmsd_cutoff=``, ...); the ``structures`` table, ``.counts`` by name."""
    from .modifiers import PolyhedralTemplateMatching

    return _verb(pipe, PolyhedralTemplateMatching, "structures", frame, _structure_names, **kwargs)


def voronoi(pipe, frame: int = 0, **kwargs) -> Table:
    """Voronoi tessellation: adds :class:`~ewald.modifiers.VoronoiAnalysis` (Atomic Volume,
    Coordination, ...) and returns the ``voronoi-coordination`` table (how many particles have
    each coordination number, also per type); ``.counts`` is ``{coordination: n}``."""
    from .modifiers import VoronoiAnalysis

    return _verb(pipe, VoronoiAnalysis, "voronoi-coordination", frame, **kwargs)


def clusters(pipe, cutoff: float = 3.2, frame: int = 0, **kwargs) -> Table:
    """Clusters of particles closer than ``cutoff``: adds :class:`~ewald.modifiers.ClusterAnalysis`
    (largest first) and returns its ``clusters`` table; ``.counts`` is ``{cluster id: size}``."""
    from .modifiers import ClusterAnalysis

    kwargs.setdefault("sort_by_size", True)
    return _verb(pipe, ClusterAnalysis, "clusters", frame, cutoff=cutoff, **kwargs)

