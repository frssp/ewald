"""Several runs as one notebook: a temperature series, a composition series, a convergence test.

::

    from ewald import load_runs, msd, rdf, arrhenius
    runs = load_runs("T*/XDATCAR", label=r"T(\\d+)")   # one pipeline per file, labelled 300, 600, ...
    pipe = runs.pipe                 # the run the 3D view shows (the GUI's run picker switches it)
    runs.add(WrapAtoms())            # a step on every run
    g = rdf(runs, cutoff=5)          # every analysis verb takes runs: one table per run
    g.plot()                         # the runs overlaid; g["600"] is one run's Table
    d = msd(runs)                    # time per frame from each run's INCAR / log.lammps
    arrhenius(d)                     # activation energy from D(T)

The runs are ordinary :class:`~ewald.pipeline.Pipeline` objects, so anything that works on
``pipe`` works on ``runs["600"].pipe``. Verbs and :meth:`Runs.add` copy each step per run, so
the runs never share a modifier's cached results.
"""
from __future__ import annotations

import copy
import glob as _glob
import os
import re

import numpy as np

__all__ = ["Run", "Runs", "RunTables", "Arrhenius", "load_runs", "arrhenius"]

BOLTZMANN_EV = 8.617333262e-5  # eV/K


class Run:
    """One run of a :class:`Runs` series: its ``label``, the number in it (``value``, or None),
    the file ``path`` and its ``pipe``."""

    def __init__(self, label: str, path: str, pipe, value: float | None = None):
        self.label, self.path, self.pipe, self.value = label, path, pipe, value

    @property
    def info(self):
        """:class:`ewald.runinfo.RunInfo` of this run (from its INCAR / log.lammps)."""
        from .runinfo import run_info

        return run_info(self.pipe)

    def __repr__(self):
        return f"<Run {self.label!r}: {self.path}>"


def _number(text: str) -> float | None:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    return float(m.group(0)) if m else None


def _parts(path: str) -> list[str]:
    return os.path.normpath(path).split(os.sep)


def _default_labels(paths: list[str]) -> list[str]:
    """What tells the paths apart: the first path component that differs (``T300`` of
    ``T300/XDATCAR``), else the file names."""
    if len(paths) == 1:
        parts = _parts(paths[0])
        return [parts[-2] if len(parts) > 1 else parts[-1]]
    split = [_parts(p) for p in paths]
    n = min(len(s) for s in split)
    for k in range(n):
        names = [s[k] for s in split]
        if len(set(names)) > 1:
            return names
    return [os.path.basename(p) for p in paths]


class Runs:
    """A series of runs (:func:`load_runs`), in order of their labels' numbers when every label
    has one. Index by position or label: ``runs[0]``, ``runs["600"]``."""

    def __init__(self, runs):
        self.runs: list[Run] = list(runs)
        self.current = 0   # the run that ``pipe`` and the GUI's 3D view show

    # ---- access ---------------------------------------------------------
    def __len__(self):
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def __getitem__(self, key) -> Run:
        if isinstance(key, Run) and key in self.runs:
            return key
        if isinstance(key, (int, np.integer)):
            return self.runs[key]
        for r in self.runs:
            if r.label == str(key):
                return r
        number = _number(str(key)) if isinstance(key, str) else key
        hits = [r for r in self.runs if r.value is not None and isinstance(number, (float, int, np.floating))
                and r.value == number]
        if len(hits) == 1:          # "500" or 500.0 for the run labelled "T500"
            return hits[0]
        raise KeyError(f"no run {key!r}; the runs are {', '.join(self.labels)}")

    def keys(self) -> list[str]:
        """The labels, as for a dict ``{label: Run}``."""
        return self.labels

    def items(self) -> list:
        """``(label, Run)`` pairs, as for a dict ``{label: Run}``."""
        return [(r.label, r) for r in self.runs]

    @property
    def labels(self) -> list[str]:
        return [r.label for r in self.runs]

    @property
    def values(self) -> np.ndarray:
        """The number in each label (NaN where there is none)."""
        return np.array([np.nan if r.value is None else r.value for r in self.runs])

    @property
    def pipes(self) -> list:
        return [r.pipe for r in self.runs]

    @property
    def pipe(self):
        """The pipeline of the current run (the first, until :meth:`select` or the GUI changes it)."""
        return self.runs[self.current].pipe

    def select(self, key):
        """Make run ``key`` (label or position) the current one and return its pipeline."""
        run = self[key]
        self.current = self.runs.index(run)
        return run.pipe

    # ---- steps ----------------------------------------------------------
    def add(self, *modifiers):
        """Add each modifier (or ``def f(frame, data)`` Python step) to every run, as a copy per
        run; returns its argument, so ``@runs.add`` works as a decorator like ``@pipe.add``."""
        for r in self.runs:
            r.pipe.add(*[m if callable(m) and not hasattr(m, "modify") else copy.deepcopy(m) for m in modifiers])
        return modifiers[0] if len(modifiers) == 1 else modifiers

    def map(self, fn) -> dict:
        """``{label: fn(pipe)}`` over the runs."""
        return {r.label: fn(r.pipe) for r in self.runs}

    def attribute(self, name: str, frame: int = 0):
        """A global attribute of every run at ``frame`` (after the steps) as a table: the run
        value (or position) and the attribute."""
        from .verbs import Table

        vals = []
        for r in self.runs:
            v = r.pipe.compute(frame).attributes.get(name, np.nan)
            vals.append(float(v) if np.isscalar(v) else np.nan)
        return Table(name, {"Run": self._x(), name: np.array(vals)}, labels=self.labels)

    def _x(self) -> np.ndarray:
        x = self.values
        return x if np.isfinite(x).all() else np.arange(len(self.runs), dtype=float)

    # ---- run information ------------------------------------------------
    def info(self) -> dict:
        """``{label: RunInfo}`` (:func:`ewald.run_info` of each run)."""
        return {r.label: r.info for r in self.runs}

    def check(self) -> dict:
        """``{label: RunCheck}`` (:func:`ewald.check_run` of each run)."""
        from .runinfo import check_run

        return {r.label: check_run(r.pipe) for r in self.runs}

    def temperatures(self) -> np.ndarray:
        """Each run's target temperature from its input files (thermostat or TEEND), else the
        number in its label."""
        out = []
        for r in self.runs:
            t = r.info.temperature
            out.append(t[1] if t else (np.nan if r.value is None else r.value))
        return np.array(out, float)

    def __repr__(self):
        return f"<Runs {len(self)}: {', '.join(self.labels)}>"


def load_runs(paths, label=None, sort: bool = True, **kwargs) -> Runs:
    """Open several runs at once: ``paths`` is a wildcard (``"T*/XDATCAR"``, ``"cluster:/scratch/*/OUTCAR"``)
    or a list of files, each opened with :func:`ewald.load` (``kwargs`` such as ``type_map=`` go
    to it). ``label`` names the runs: a regular expression whose first group is the label
    (``r"T(\\d+)"``), a function of the path, or a list; by default the folder (or file) name that
    differs between the paths. Runs whose labels all hold a number are sorted by it."""
    from . import load
    from .io import remote

    if isinstance(paths, (str, os.PathLike)):
        pattern = str(paths)
        if remote.is_remote(pattern):
            files = remote.glob(pattern)
        else:
            files = sorted(_glob.glob(pattern))
            if not files and os.path.exists(pattern):
                files = [pattern]
        if not files:
            raise FileNotFoundError(f"no files match {pattern}")
    else:
        files = [str(p) for p in paths]
    local = [remote.parse_remote(f).path if remote.is_remote(f) else f for f in files]
    if label is None:
        labels = _default_labels(local)
    elif isinstance(label, str):
        rx = re.compile(label)
        labels = []
        for f in local:
            m = rx.search(f)
            if m is None:
                raise ValueError(f"label pattern {label!r} does not match {f}")
            labels.append(m.group(1) if m.groups() else m.group(0))
    elif callable(label):
        labels = [str(label(f)) for f in files]
    else:
        labels = [str(x) for x in label]
        if len(labels) != len(files):
            raise ValueError(f"{len(labels)} labels for {len(files)} files")
    runs = [Run(lab, f, load(f, **kwargs), _number(lab)) for lab, f in zip(labels, files)]
    if sort and all(r.value is not None for r in runs):
        runs.sort(key=lambda r: r.value)
    return Runs(runs)


# ----------------------------------------------------------------------- tables of runs
class RunTables(dict):
    """``{label: Table}``: one analysis table per run (what a verb returns for :class:`Runs`).
    The last line of a cell shows the runs overlaid; :meth:`plot` draws them with matplotlib."""

    def __init__(self, name: str, tables: dict, runs: Runs | None = None):
        super().__init__(tables)
        self.name, self.runs = name, runs

    def __getitem__(self, key):
        if isinstance(key, Run):
            key = key.label
        if key not in self.keys() and self.runs is not None and not isinstance(key, (int, np.integer)):
            try:
                key = self.runs[key].label     # "500" or 500.0 for the run labelled "T500"
            except KeyError:
                pass
        return super().__getitem__(key)

    def column(self, y: str | None = None) -> str:
        """The column drawn: ``y``, else the table's first data column."""
        first = next(iter(self.values()))
        return y if y is not None else first.columns[1]

    def attribute(self, name: str) -> np.ndarray:
        """A global attribute the analysis wrote, one value per run (NaN where missing; a name
        no run has is an error that lists the names there are)."""
        tables = list(self.values())
        if tables and not any(name in t.attributes for t in tables):
            have = sorted({k for t in tables for k in t.attributes})
            raise KeyError(f"no run has the attribute {name!r}; they have {', '.join(have) or 'none'}")
        return np.array([float(t.attributes.get(name, np.nan)) for t in tables])

    def figure(self, y: str | None = None, size=None):
        """The runs' curves of column ``y`` on one axis (a matplotlib Figure)."""
        from . import plotstyle
        from .plotstyle import axis_label

        y = self.column(y)
        fig = plotstyle.figure(size)
        with plotstyle.context(*fig._ewald_style):
            ax = fig.add_subplot()
            xname = None
            for lab, t in self.items():
                if y not in t:
                    continue
                xname = t.columns[0]
                yy = np.asarray(t[y], float)
                ok = np.isfinite(yy)
                ax.plot(np.asarray(t.x, float)[ok], yy[ok], label=lab)
            if xname:
                ax.set_xlabel(axis_label(xname))
            ax.set_ylabel(axis_label(y))
            if len(self) > 1:
                ax.legend(loc="best", title=None)
        return fig

    def plot(self, y: str | None = None):
        """Draw the runs overlaid (``plt.show()``)."""
        import matplotlib.pyplot as plt

        self.figure(y)
        plt.show()

    def _repr_png_(self):
        import io

        from . import plotstyle
        from .display import options

        if not options.get("enabled", True) or not self:
            return None
        try:
            fig = self.figure()
            buf = io.BytesIO()
            plotstyle.save(fig, buf, format="png", dpi=150)
        except Exception:  # noqa: BLE001 - the text repr instead
            return None
        return buf.getvalue()

    def __repr__(self):
        return f"<RunTables {self.name!r}: {len(self)} runs ({', '.join(self)})>"


# ----------------------------------------------------------------------- Arrhenius
class Arrhenius:
    """Fit of ``y = A exp(-Ea / kT)``: ``Ea`` (eV) with its standard error ``Ea_error``,
    the prefactor ``A`` and ``r2``; the cell shows ln y against 1000/T with the line."""

    def __init__(self, temperatures, values, name: str = "D"):
        t = np.asarray(temperatures, float)
        v = np.asarray(values, float)
        if t.shape != v.shape:
            raise ValueError(f"{t.size} temperatures for {v.size} values")
        ok = np.isfinite(t) & np.isfinite(v) & (v > 0) & (t > 0)
        if ok.sum() < 2 or len(set(t[ok])) < 2:
            pairs = ", ".join(f"{a:g} K: {b:.4g}" for a, b in zip(t, v))
            why = ("an Arrhenius fit needs runs at two or more different temperatures"
                   if ok.sum() >= 2 else "an Arrhenius fit needs at least two runs with a temperature and a positive value")
            raise ValueError(f"{why} (got {pairs or 'nothing'})")
        self.temperatures, self.values, self.name = t[ok], v[ok], name
        x, ly = 1.0 / self.temperatures, np.log(self.values)
        if ok.sum() > 2:
            (slope, icept), cov = np.polyfit(x, ly, 1, cov=True)
            self.Ea_error = float(np.sqrt(cov[0, 0]) * BOLTZMANN_EV)
        else:
            slope, icept = np.polyfit(x, ly, 1)
            self.Ea_error = float("nan")
        self.Ea = float(-slope * BOLTZMANN_EV)
        self.A = float(np.exp(icept))
        fit = slope * x + icept
        ss = float(np.sum((ly - ly.mean()) ** 2))
        self.r2 = 1.0 - float(np.sum((ly - fit) ** 2)) / ss if ss > 0 else 1.0

    def predict(self, temperature):
        """``A exp(-Ea / kT)`` at ``temperature`` (K)."""
        return self.A * np.exp(-self.Ea / (BOLTZMANN_EV * np.asarray(temperature, float)))

    def figure(self, size=None):
        from . import plotstyle

        fig = plotstyle.figure(size)
        with plotstyle.context(*fig._ewald_style):
            ax = fig.add_subplot()
            x = 1000.0 / self.temperatures
            ax.plot(x, self.values, "o", label=self.name)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, self.predict(1000.0 / xs), "--", lw=0.9, label=f"$E_a$ = {self.Ea:.3f} eV")
            ax.set_yscale("log")
            ax.set_xlabel("1000/$T$ (K$^{-1}$)")
            ax.set_ylabel(self.name)
            ax.legend(loc="best")
        return fig

    def plot(self, size=None):
        """Draw the Arrhenius plot (``plt.show()``; under a notebook cell it is the cell's figure)."""
        import matplotlib.pyplot as plt

        self.figure(size)
        plt.show()

    def savefig(self, path, dpi: int = 300, **kwargs):
        """Save the Arrhenius plot to ``path`` (PNG, PDF, SVG, ... by its extension)."""
        from . import plotstyle

        plotstyle.save(self.figure(), path, dpi=dpi, **kwargs)

    def _repr_png_(self):
        import io

        from . import plotstyle
        from .display import options

        if not options.get("enabled", True):
            return None
        try:
            buf = io.BytesIO()
            plotstyle.save(self.figure(), buf, format="png", dpi=150)
        except Exception:  # noqa: BLE001
            return None
        return buf.getvalue()

    def __repr__(self):
        err = "" if self.Ea_error != self.Ea_error else f" ± {self.Ea_error:.3f}"
        return f"<Arrhenius Ea = {self.Ea:.3f}{err} eV, A = {self.A:.3g}, r2 = {self.r2:.3f}, {len(self.temperatures)} runs>"


def arrhenius(data, values=None, attribute: str = "MeanSquaredDisplacement.diffusion_coefficient") -> Arrhenius:
    """Activation energy from runs at several temperatures. ``data`` is what ``msd(runs)``
    returned (temperatures from each run's thermostat target, else its label; values from
    ``attribute``), a list of temperatures (K) with ``values`` beside it, or ``{temperature: value}``."""
    if isinstance(data, dict) and not isinstance(data, RunTables) and values is None:
        return Arrhenius(list(data.keys()), list(data.values()), "value")
    if isinstance(data, RunTables):
        if data.runs is None:
            raise ValueError("these tables do not know their runs; pass temperatures and values")
        name = "$D$ (Å$^2$/ps)" if attribute.endswith("diffusion_coefficient") else attribute
        return Arrhenius(data.runs.temperatures(), data.attribute(attribute), name)
    if isinstance(data, Runs):
        return Arrhenius(data.temperatures(), np.asarray(values, float), "value")
    if values is None:
        raise TypeError("arrhenius(temperatures, values) needs the values too")
    return Arrhenius(data, values, "value")
