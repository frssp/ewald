"""Analysis verbs: one call adds the analysis step and returns its table, for notebooks and scripts::

    from ewald import load, rdf, msd
    pipe = load("XDATCAR")
    g = rdf(pipe, cutoff=6)      # adds CoordinationAnalysis, returns the g(r) table
    g.x, g["g(r)"]               # columns are numpy arrays; pandas.DataFrame(g), g.plot()
    d = msd(pipe)                # MeanSquaredDisplacement over the trajectory

Calling a verb again replaces the step it added before (in a notebook, re-running the cell
does the same), so changing ``cutoff=`` and running again never stacks duplicate steps.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Table", "rdf", "msd", "vacf", "sq", "xrd", "bond_angles"]


class Table(dict):
    """A data table (``{column: array}``, the first column is x) with where it came from."""

    def __init__(self, name: str, columns: dict, pipe=None, frame: int = 0):
        super().__init__((k, np.asarray(v)) for k, v in columns.items())
        self.name, self.pipe, self.frame = name, pipe, frame

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

    def __repr__(self):
        n = len(self.x) if self else 0
        return f"<Table {self.name!r}: {n} rows; {', '.join(self.columns)}>"


def _verb(pipe, cls, table: str, frame: int, **params) -> Table:
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
    return Table(table, data.tables[table], pipe, frame)


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
