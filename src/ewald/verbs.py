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

#: Insert › Quick analysis in the GUI: (verb, menu label, the code cell it inserts and runs)
VERB_CELLS = [
    ("check_run", "Run check (convergence, equilibration)", "check = check_run(pipe)\ncheck"),
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


#: verbs that need a trajectory (Insert › Quick analysis greys them out for a single frame)
TRAJECTORY_VERBS = ("msd", "vacf")


def verb_cell(verb: str, data=None) -> str:
    """The code cell Insert › Quick analysis adds for ``verb`` (with its import). With the current frame
    ``data`` the defaults fit it: the g(r) cutoff stays below half the periodic cell width, and
    bond angles first create bonds (covalent radii, like the default view) when there are none."""
    code = next(c for v, _label, c in VERB_CELLS if v == verb)
    if data is not None and verb == "rdf" and data.cell.is_valid and data.cell.pbc.any():
        half = float(np.min(data.cell.perpendicular_widths()[data.cell.pbc])) / 2
        if half <= 6.0:
            code = code.replace("cutoff=6.0", f"cutoff={np.floor(half * 0.99 * 10) / 10:.1f}")
    if data is not None and verb == "bond_angles" and data.bonds is None:
        return ("from ewald import bond_angles\nfrom ewald.modifiers import CreateBonds\n"
                f"pipe.add(CreateBonds(mode=\"covalent\"))\n{code}")
    return f"from ewald import {verb}\n{code}"


def load_table(path) -> Table:
    """A file that holds no structure, as a :class:`Table` to plot with ``show(table, y=...)``
    or use as ``table["column"]``: a LAMMPS log or VASP OSZICAR (the per-step thermo output;
    a log with several runs is one table joined by ``Step``), or a plain text table (columns of
    numbers, whitespace or comma separated, named by a header line or the last ``#`` comment
    with one word per column, as in ``#! FIELDS time phi psi``: CV and energy logs, analysis
    output, a table of runs). Relative paths are found as :func:`ewald.load` finds them."""
    import os

    from .io.text_table import read_text_table
    from .io.thermo import read_thermo
    from .notebook import resolve_data_path

    path = resolve_data_path(path)
    try:
        columns = read_thermo(path)
    except Exception:          # not a thermo log: a plain table (raises if it is not one either)
        columns = None
    if not columns:
        columns = read_text_table(path)
    return Table(os.path.basename(str(path)), columns)


__all__ = ["Table", "bond_angles", "clusters", "cna", "load_table", "msd", "ptm", "rdf", "sensitivity", "sq", "vacf",
           "voronoi", "xrd"]


class Table(dict):
    """A data table (``{column: array}``, the first column is x) with where it came from."""

    def __init__(self, name: str, columns: dict, pipe=None, frame: int = 0, labels=None, attributes=None):
        super().__init__((k, np.asarray(v)) for k, v in columns.items())
        self.name, self.pipe, self.frame = name, pipe, frame
        self.labels = None if labels is None else list(labels)  # names of the rows (structure types)
        self.attributes = dict(attributes or {})  # global results of the step (diffusion coefficient, ...)

    def __copy__(self):   # DataCollection.copy(): new containers, shared arrays, as for a plain table
        return Table(self.name, self, self.pipe, self.frame, self.labels, self.attributes)

    @property
    def counts(self) -> dict:
        """``{row: Count}`` for a counting table (structure types, coordination numbers, clusters);
        rows are named by ``labels`` when the table has them, else by the first column."""
        col = "Count" if "Count" in self else next((c for c in self if c.endswith("Size")), None)
        if col is None:
            raise KeyError(f"{self.name!r} has no Count column")
        keys = self.labels if self.labels is not None else [k.item() for k in self.x]
        return {k: int(v) for k, v in zip(keys, self[col])}

    def attribute(self, name: str):
        """One of the step's global results in :attr:`attributes` (as ``Runs.attribute`` names them)."""
        try:
            return self.attributes[name]
        except KeyError:
            raise KeyError(f"{self.name!r} has no attribute {name!r}; it has: "
                           f"{', '.join(self.attributes) or 'none'}") from None

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
        if self.labels is not None and ("Count" in self or any(c.endswith("Size") for c in self)):
            return f"<Table {self.name!r}: {', '.join(f'{k} {v}' for k, v in self.counts.items())}>"
        if self.labels is not None:
            return f"<Table {self.name!r}: rows {', '.join(self.labels)}; {', '.join(self.columns[1:])}>"
        return f"<Table {self.name!r}: {n} rows; {', '.join(self.columns)}>"


#: the verb that adds each step class (names a table the verb puts beside an earlier step's)
VERB_OF = {"CoordinationAnalysis": "rdf", "MeanSquaredDisplacement": "msd", "VelocityAutocorrelation": "vacf",
           "StructureFactor": "sq", "PowderDiffraction": "xrd", "BondAngleDistribution": "bond_angles",
           "CommonNeighborAnalysis": "cna", "PolyhedralTemplateMatching": "ptm", "VoronoiAnalysis": "voronoi",
           "ClusterAnalysis": "clusters"}


def _verb(pipe, cls, table: str, frame: int, labels=None, **params) -> Table:
    """Add ``cls(**params)`` to ``pipe`` (replacing the one an earlier verb call added) and return
    ``table`` at ``frame``. The step never changes what earlier steps wrote: a property, attribute
    or table they made keeps its value, and a table of the same name from this step is
    ``"<table> (<verb>)"`` (see :func:`ewald.pipeline._keep_upstream`). For :class:`~ewald.runs.Runs` the same on every run: a
    :class:`~ewald.runs.RunTables` of one table per run."""
    from .runs import Runs, RunTables

    if isinstance(pipe, Runs):
        return RunTables(table, {r.label: _verb(r.pipe, cls, table, frame, labels, **params) for r in pipe}, pipe)
    if "time_per_frame" in params and params["time_per_frame"] is None:
        from .runinfo import run_info

        params["time_per_frame"] = run_info(pipe).time_per_frame or 0.0
    mod = cls(**params)
    mod._verb = cls.__name__
    mod._verb_name = VERB_OF.get(cls.__name__, cls.__name__)
    for i, m in enumerate(pipe.modifiers):
        if getattr(m, "_verb", None) == cls.__name__:
            pipe.modifiers[i] = mod
            break
    else:
        pipe.add(mod)
    data = pipe.compute(frame)
    table_name = getattr(mod, "_verb_tables", {}).get(table, table)
    if table_name not in data.tables:
        raise RuntimeError(f"{cls.__name__} wrote no {table_name!r} table: {mod.status or 'see the step for details'}")
    columns = data.tables[table_name]
    if callable(labels):
        labels = labels(mod, columns)
    prefix = cls.__name__ + "."
    attrs = {k: v for k, v in {**data.attributes, **getattr(mod, "_verb_attributes", {})}.items()
             if k.startswith(prefix)}
    return Table(table_name, columns, pipe, frame, labels, attrs)


def rdf(pipe, cutoff: float = 6.0, bins: int = 200, partial: bool = False, frame: int = 0) -> Table:
    """Radial distribution function g(r) up to ``cutoff`` (Angstrom), per type pair with
    ``partial=True``: adds :class:`~ewald.modifiers.CoordinationAnalysis` (which also writes the
    ``Coordination`` property) and returns its ``coordination-rdf`` table."""
    from .modifiers import CoordinationAnalysis

    return _verb(pipe, CoordinationAnalysis, "coordination-rdf", frame, cutoff=cutoff, bins=bins,
                 partial=partial)


def msd(pipe, frame: int = 0, time_per_frame: float | None = None, **kwargs) -> Table:
    """Mean squared displacement over the trajectory (per type, with the Einstein fit):
    :class:`~ewald.modifiers.MeanSquaredDisplacement` with ``kwargs``; the ``msd`` table, whose
    ``attributes`` hold the diffusion coefficient. ``time_per_frame`` (ps) defaults to what the
    run's input files say (:func:`ewald.run_info`), else frames."""
    from .modifiers import MeanSquaredDisplacement

    return _verb(pipe, MeanSquaredDisplacement, "msd", frame, time_per_frame=time_per_frame, **kwargs)


def vacf(pipe, frame: int = 0, time_per_frame: float | None = None, **kwargs) -> Table:
    """Velocity autocorrelation over the trajectory:
    :class:`~ewald.modifiers.VelocityAutocorrelation` with ``kwargs``; the ``vacf`` table.
    ``time_per_frame`` (ps) defaults to the run's input files (:func:`ewald.run_info`) when the
    velocities are in A/ps (VASP, LAMMPS metal units), else frames."""
    from .modifiers import VelocityAutocorrelation
    from .runinfo import run_info
    from .runs import Runs, RunTables

    if isinstance(pipe, Runs):
        return RunTables("vacf", {r.label: vacf(r.pipe, frame, time_per_frame, **kwargs) for r in pipe}, pipe)
    if time_per_frame is None:
        info = run_info(pipe)
        time_per_frame = (info.time_per_frame or 0.0) if info.code == "vasp" or info.units == "metal" else 0.0

    return _verb(pipe, VelocityAutocorrelation, "vacf", frame, time_per_frame=time_per_frame, **kwargs)


def sq(pipe, q_max: float = 10.0, bins: int = 200, frame: int = 0, **kwargs) -> Table:
    """Structure factor S(q) up to ``q_max`` (1/Angstrom): :class:`~ewald.modifiers.StructureFactor`
    with ``kwargs`` (``weighting=``, ``experimental=``, ...); the ``structure-factor`` table."""
    from .modifiers import StructureFactor

    return _verb(pipe, StructureFactor, "structure-factor", frame, q_max=q_max, bins=bins, **kwargs)


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


def sensitivity(pipe, step, parameter: str, values, attributes=None, frame: int = 0) -> Table:
    """How much the results of ``step`` move when its ``parameter`` takes each of ``values``: an
    analysis choice (a fit window, a cutoff, where equilibration ends) shown next to its effect.

    ``step`` is a step of ``pipe`` (the modifier, or its class name for the first step of that
    class). Each value is tried on a copy of the step in its place, then the step is put back as
    it was. ``attributes`` (a name or list) are the results to report, by default every attribute
    the step writes (``MeanSquaredDisplacement.*``). Returns the ``sensitivity`` table: the
    parameter's values (numbers, or rows named by ``labels`` for tuples and strings), one column
    per attribute, and in ``attributes`` the step's own value of the parameter and the relative
    spread ``(max - min) / |value at the step's setting|`` of each result (``.spread.<attribute>``)::

        pipe.add(MeanSquaredDisplacement(time_per_frame=0.1, fit_time=(2, 8)))
        sensitivity(pipe, "MeanSquaredDisplacement", "fit_time", [(1, 5), (2, 8), (4, 12)],
                    "MeanSquaredDisplacement.diffusion_coefficient.Li")
    """
    import copy

    index = _step_index(pipe, step)
    original = pipe.modifiers[index]
    if parameter not in original.parameters():
        raise ValueError(f"{type(original).__name__} has no parameter {parameter!r}; it has "
                         f"{', '.join(original.parameters())}")
    prefix = type(original).__name__ + "."
    wanted = [attributes] if isinstance(attributes, str) else (list(attributes) if attributes else None)
    rows = []
    try:
        for value in values:
            trial = copy.copy(original)
            trial.__dict__.pop("_result", None)        # a trajectory step's cache belongs to the original
            setattr(trial, parameter, value)
            pipe.modifiers[index] = trial
            data = pipe.compute(frame)
            names = wanted or [k for k in data.attributes if k.startswith(prefix)]
            missing = [n for n in names if n not in data.attributes]
            if missing:
                raise KeyError(f"no attribute {missing[0]!r} after {type(original).__name__}; it wrote "
                               f"{', '.join(k for k in data.attributes if k.startswith(prefix)) or 'none'}")
            rows.append({n: data.attributes[n] for n in names})
    finally:
        pipe.modifiers[index] = original
    values = list(values)
    numeric = all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in values)
    columns = {parameter: np.asarray(values, float) if numeric else np.arange(len(values))}
    names = list(rows[0]) if rows else []
    for n in names:
        columns[n] = np.array([r.get(n, np.nan) for r in rows], dtype=float)
    attrs = {"sensitivity.parameter": parameter, "sensitivity.setting": getattr(original, parameter)}
    setting = getattr(original, parameter)
    at = next((i for i, v in enumerate(values) if _same(v, setting)), None)
    for n in names:
        col = columns[n]
        ref = col[at] if at is not None else np.nanmedian(col)
        attrs[f"sensitivity.spread.{n}"] = float((np.nanmax(col) - np.nanmin(col)) / abs(ref)) if ref else float("nan")
    return Table("sensitivity", columns, pipe, frame, None if numeric else [repr(v) for v in values], attrs)


def _same(a, b) -> bool:
    try:
        return bool(np.all(np.asarray(a) == np.asarray(b))) and np.shape(a) == np.shape(b)
    except Exception:
        return a == b


def _step_index(pipe, step) -> int:
    """Where ``step`` (a modifier of ``pipe`` or a class name) sits in ``pipe.modifiers``."""
    for i, m in enumerate(pipe.modifiers):
        if m is step or (isinstance(step, str) and type(m).__name__ == step):
            return i
    have = ", ".join(type(m).__name__ for m in pipe.modifiers) or "none"
    raise ValueError(f"{step if isinstance(step, str) else type(step).__name__} is not a step of this "
                     f"pipeline (its steps: {have})")
