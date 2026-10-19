"""Views: ``show(...)`` puts a plot of a pipeline's data under a notebook cell.

One function for every kind of data view, so a notebook needs no fixed panels::

    show(pipe, table="coordination-rdf")          # any data table: g(r), XRD, DOS, Bands, Thermo, NEB
    show(pipe, table="DOS", y="Ga d")             # one column of it
    show(pipe, histogram="Coordination")          # histogram of a particle property

A cell that is a single ``show(...)`` call with literal arguments is a *view cell*. In the GUI
it becomes a live, linked plot (drag a range to select atoms or frames; the DOS and band
structure follow the atoms picked in the viewport; a phonon band click plays the mode). Run as
a script, it draws the same data with matplotlib, so the ``.py`` file stays reproducible.
"""
from __future__ import annotations

import numpy as np

KINDS = ("table", "histogram")


def view_data(source, table: str | None = None, y: str | None = None, histogram: str | None = None,
              frame: int = 0, bins: int = 50) -> tuple[str, dict]:
    """What a view plots, as ``(title, {column: values})``: the first column is x. For a
    histogram the columns are the bin centers and the counts. ``source`` is a Pipeline or a
    DataCollection."""
    data = source.compute(frame) if hasattr(source, "compute") else source
    if histogram is not None:
        if histogram not in data.particles:
            raise KeyError(f"no particle property {histogram!r}; this frame has: {', '.join(data.particles)}")
        values = np.asarray(data.particles[histogram], float)
        counts, edges = np.histogram(values[np.isfinite(values)], bins=bins)
        return histogram, {histogram: 0.5 * (edges[1:] + edges[:-1]), "count": counts}
    if table is None:
        if not data.tables:
            raise ValueError("no tables in this data: add an analysis step first, or pass histogram=")
        table = next(iter(data.tables))
    if table not in data.tables:
        raise KeyError(f"no table {table!r}; this frame has: {', '.join(data.tables) or 'none'}")
    columns = data.tables[table]
    names = list(columns)
    keep = [names[0]] + ([y] if y is not None else names[1:])
    return table, {n: np.asarray(columns[n]) for n in keep}


def _draw(ax, title: str, columns: dict, histogram: str | None, show_title: bool = True):
    """Draw a view onto ``ax`` in the active :mod:`ewald.plotstyle` (labels with units)."""
    from .plotstyle import axis_label, legend_labels, start_at_zero

    import matplotlib as mpl

    first = mpl.rcParams["axes.prop_cycle"].by_key()["color"][0]
    names = list(columns)
    x = np.asarray(columns[names[0]], float)
    if histogram is not None:
        width = x[1] - x[0] if len(x) > 1 else 1.0
        ax.bar(x, columns["count"], width=width, color=first, lw=0)
        ax.set_xlabel(axis_label(histogram))
        ax.set_ylabel("Count")
        ax.set_ylim(bottom=0)
        return
    ys = names[1:]
    sticks = title.endswith("-peaks")
    if sticks and "Intensity" in columns:
        ys = ["Intensity"]
    for name, label in zip(ys, legend_labels(ys)):
        y = np.asarray(columns[name], float)
        ok = np.isfinite(y)
        if sticks:
            ax.vlines(x[ok], 0, y[ok], label=label, lw=1.0, color=first)
        elif name == "Experimental":   # measured pattern: points under the calculated curve
            ax.plot(x[ok], y[ok], ls="none", marker="o", ms=2.0, mew=0, color="0.45", label=label, zorder=1)
        elif name.endswith(" fit"):
            ax.plot(x[ok], y[ok], ls="--", lw=0.9, label=label, zorder=3)
        else:
            ax.plot(x[ok], y[ok], label=label, zorder=2)
    ax.set_xlabel(axis_label(names[0]))
    start_at_zero(ax, x)
    if sticks or title in ("diffraction", "structure-factor") or all(n.startswith("Count") for n in ys):
        ax.set_ylim(bottom=0)
    units = {axis_label(n) for n in ys}
    if len(ys) == 1 or len(units) == 1:
        ax.set_ylabel(units.pop() if units else "")
    if len(ys) > 1:
        ax.legend(loc="best")
    if show_title:
        ax.set_title(title)


def figure(source, table: str | None = None, y: str | None = None, histogram: str | None = None,
           frame: int = 0, bins: int = 50, size: str | tuple | None = None, title: bool = False):
    """The matplotlib Figure of a view (what :func:`show` draws), without pyplot, in the
    publication style of :mod:`ewald.plotstyle` (``size``: a column width name or inches).
    Save it with :func:`ewald.plotstyle.save` to keep the text editable in PDF/SVG."""
    from . import plotstyle

    fig = plotstyle.figure(size)
    with plotstyle.context(*fig._ewald_style):
        _draw(fig.add_subplot(), *view_data(source, table, y, histogram, frame, bins), histogram, show_title=title)
    return fig


def show(source, table: str | None = None, y: str | None = None, histogram: str | None = None,
         frame: int = 0, bins: int = 50):
    """Plot a data table or a particle-property histogram of ``source`` (a Pipeline or a
    DataCollection) at ``frame`` with matplotlib (``plt.show()``), in the publication style."""
    import matplotlib.pyplot as plt

    from . import plotstyle

    title, columns = view_data(source, table, y, histogram, frame, bins)
    with plotstyle.context():
        _fig, ax = plt.subplots(figsize=plotstyle.figsize(), layout="constrained")
        _draw(ax, title, columns, histogram)
    plt.show()


def write_csv(path, columns: dict):
    """The columns of :func:`view_data` as a CSV file (header row, then one row per x)."""
    import csv

    names = list(columns)
    rows = np.column_stack([np.asarray(columns[n], float) for n in names])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        w.writerows(rows.tolist())


def view_args(source: str, function: str = "show") -> dict | None:
    """The arguments of a cell that is a single ``show(<name>, key=literal, ...)`` call
    (optionally after imports): ``{"source": name, **literal keywords}``; None otherwise."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    *head, last = tree.body or [None]
    if not isinstance(last, ast.Expr) or not all(isinstance(s, (ast.Import, ast.ImportFrom)) for s in head):
        return None
    call = last.value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == function
            and len(call.args) == 1 and isinstance(call.args[0], ast.Name)):
        return None
    args = {"source": call.args[0].id}
    for kw in call.keywords:
        if kw.arg is None:
            return None
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:
            return None
    return args


def view_source(source: str = "pipe", **kwargs) -> str:
    """Source of a view cell: ``show(pipe, table='DOS')``; None values are left out."""
    args = "".join(f", {k}={v!r}" for k, v in kwargs.items() if v is not None and v != "")
    return f"show({source}{args})"
