"""Views: ``show(...)`` puts a plot of a pipeline's data under a notebook cell.

One function for every kind of data view, so a notebook needs no fixed panels::

    show(pipe, table="coordination-rdf")          # any data table: g(r), XRD, DOS, Bands, Steps, NEB
    show(pipe, table="DOS", y="Ga d")             # one column of it
    show(pipe, histogram="Coordination")          # histogram of a particle property
    snapshot(pipe, frame=3, camera={...})         # a picture of the 3D view (Insert › 3D snapshot)

A cell that is a single ``show(...)`` call with literal arguments is a *view cell*. In the GUI
it becomes a live, linked plot (drag a range to select atoms or frames; the DOS and band
structure follow the atoms picked in the viewport; a phonon band click plays the mode). Run as
a script, it draws the same data with matplotlib, so the ``.py`` file stays reproducible.

Where the GUI shows such a plot (:func:`in_panel`): under its cell, like Jupyter, unless it
changes with the frame or is linked to the 3D view (g(r), histograms, time series, the NEB
profile, DOS and bands, phonon bands). Those open in the Plots panel and leave a thumbnail
under the cell. ``show(..., panel=True)`` or ``panel=False`` overrides the rule.
"""
from __future__ import annotations

import numpy as np

from .view3d import view  # noqa: F401 - notebooks import show and view from here

KINDS = ("table", "histogram")


def view_data(source, table: str | None = None, y: str | None = None, histogram: str | None = None,
              frame: int = 0, bins: int = 50) -> tuple[str, dict]:
    """What a view plots, as ``(title, {column: values})``: the first column is x. For a
    histogram the columns are the bin centers and the counts. ``source`` is a Pipeline, a
    DataCollection or a :class:`~ewald.verbs.Table` (an analysis result); a table comes back as a
    ``Table`` with its ``labels`` and ``attributes`` (the levels and regions drawn over it)."""
    from .verbs import Table

    if isinstance(source, Table):   # a result of its own (an analysis): the table is the data
        if table not in (None, source.name):
            raise KeyError(f"no table {table!r}; this result is {source.name!r}")
        names = list(source)
        keep = [names[0]] + ([y] if y is not None else names[1:])
        return source.name, Table(source.name, {n: source[n] for n in keep}, labels=source.labels,
                                  attributes=source.attributes)
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
    return table, Table(table, {n: columns[n] for n in keep}, labels=getattr(columns, "labels", None),
                        attributes=getattr(columns, "attributes", None))


COUNT_TABLES = {"structures": "Count", "voronoi-coordination": "Count", "clusters": "Cluster Size"}


def _draw_counts(ax, x, columns, column, labels, color):
    """Bars of a counting table; with per-type columns ('Count Cu', ...) side by side."""
    from .plotstyle import axis_label

    ys = [column] + [n for n in columns if n.startswith(column + " ")]
    if len(ys) > 1:
        ys = ys[1:]
    width = 0.8 / len(ys)
    for k, name in enumerate(ys):
        off = (k - (len(ys) - 1) / 2) * width
        ax.bar(x + off, np.asarray(columns[name], float), width=width, lw=0,
               color=color if len(ys) == 1 else None, label=name[len(column):].strip() or None)
    names = list(columns)
    if labels is not None:
        ax.set_xticks(x, labels, rotation=45 if len(labels) > 5 else 0, ha="right" if len(labels) > 5 else "center")
    else:
        ax.set_xlabel(axis_label(names[0]))
    ax.set_ylabel(axis_label(column))
    ax.set_ylim(bottom=0)
    if len(ys) > 1:
        ax.legend(loc="best")


def _named_rows(x, labels) -> bool:
    """A table of one value per named row (per surface, per run): its x is only the row number."""
    return labels is not None and len(labels) == len(x) and np.array_equal(x, np.arange(len(x)))


def _draw_named_rows(ax, columns, labels, color):
    """One bar per named row for the first value column, its value written on the bar; the
    other columns stay in the table. The axis starts near the values when they are far from 0,
    so 4.6 and 4.9 eV do not look alike."""
    from .plotstyle import axis_label

    name = list(columns)[1]
    y = np.asarray(columns[name], float)
    x = np.arange(len(y))
    ax.bar(x, y, width=0.6, color=color, lw=0)
    ok = y[np.isfinite(y)]
    if len(ok):
        lo, hi = ok.min(), ok.max()
        pad = max(0.1 * (hi - lo), 0.05 * max(abs(lo), abs(hi)), 1e-12)
        bottom = lo - 3 * pad if lo - 3 * pad > 0 else min(0.0, lo - pad)
        ax.set_ylim(bottom, max(0.0, hi) + pad)
        for xi, yi in zip(x, y):
            if np.isfinite(yi):
                ax.annotate(f"{yi:.3g}", (xi, yi), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize="small")
    ax.set_xticks(x, labels, rotation=45 if len(labels) > 5 else 0, ha="right" if len(labels) > 5 else "center")
    ax.set_xlim(-0.6, len(y) - 0.4)
    ax.set_ylabel(axis_label(name))


DECORATIONS = ("levels", "regions", "gaps", "energy_axis")   # what _draw_levels reads of a table's attributes


def decorations(table) -> dict:
    """The ``levels``/``regions``/``gaps``/``energy_axis`` a table's ``attributes`` put over its
    curves (empty for a plain table)."""
    attributes = getattr(table, "attributes", None) or {}
    return {k: attributes[k] for k in DECORATIONS if attributes.get(k)}


def _draw_levels(ax, attributes, x, colors, ink="0.15", muted="0.35"):
    """What a table's ``attributes`` put over its curves: ``regions`` (shaded x spans, named at the
    bottom), ``levels`` (horizontal lines over an x span or, as references like E_F, the whole
    width, named at their right end), ``gaps`` (arrows between two energies with their value) and
    ``energy_axis`` (the y label, in eV). ``ink`` draws the references and ``muted`` the region
    names (the Plots panel passes its theme's text colors)."""
    regions, levels, gaps = (attributes.get(k) or [] for k in ("regions", "levels", "gaps"))
    if attributes.get("energy_axis"):
        ax.set_ylabel(f"{attributes['energy_axis']} (eV)")
    if not (regions or levels or gaps):
        return
    x0, x1 = float(np.nanmin(x)), float(np.nanmax(x))
    for r in regions:
        ax.axvspan(*r["z"], color="0.5", alpha=0.12, lw=0, zorder=0)
        ax.text(sum(r["z"]) / 2, 0.02, r["name"], transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize="x-small", color=muted, zorder=4,
                bbox={"boxstyle": "round,pad=0.15", "fc": ax.get_facecolor(), "ec": "none", "alpha": 0.85})
    styles = {"solid": "-", "dashed": "--", "dotted": ":"}
    accent = colors[2 % len(colors)]
    for lv in levels:
        a, b = lv.get("z") or (x0, x1)
        color = ink if lv.get("z") is None else accent
        # a dense line, so the legend's "best" place keeps clear of it
        ax.plot(np.linspace(a, b, 64), np.full(64, float(lv["energy"])), color=color,
                ls=styles.get(lv.get("style"), "-"), lw=1.2, zorder=3)
        ax.annotate(lv["name"], (b, lv["energy"]), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize="x-small", color=color)
    ylo, yhi = ax.get_ylim()
    ylo, yhi = min([ylo] + [lv["energy"] for lv in levels]), max([yhi] + [lv["energy"] for lv in levels])
    top = 0.04
    for k, g in enumerate(gaps):
        gx = g.get("x")
        gx = x0 + (0.85 - 0.12 * k) * (x1 - x0) if gx is None else gx
        ax.annotate("", (gx, g["lower"]), (gx, g["upper"]),
                    arrowprops={"arrowstyle": "<->", "color": "C3", "lw": 1.0, "shrinkA": 0, "shrinkB": 0})
        small = g["upper"] - g["lower"] < 0.08 * (yhi - ylo)
        top = max(top, 0.12 if small else top)
        ax.annotate(g["name"], (gx, g["upper"] if small else 0.5 * (g["lower"] + g["upper"])),
                    xytext=(3, 6 if small else 0), textcoords="offset points", va="bottom" if small else "center",
                    fontsize="small", color="C3")
    # room on the right for the level names
    ax.set_xlim(x0, x1 + 0.2 * (x1 - x0))
    ax.set_ylim(ylo, yhi + top * (yhi - ylo))


def _draw(ax, title: str, columns: dict, histogram: str | None, show_title: bool = True, labels=None,
          attributes=None):
    """Draw a view onto ``ax`` in the active :mod:`ewald.plotstyle` (labels with units); a table
    of named rows as bars, and the ``levels``/``regions``/``gaps`` of its ``attributes`` over it."""
    from .plotstyle import axis_label, legend_labels, start_at_zero

    import matplotlib as mpl

    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    first = colors[0]
    labels = getattr(columns, "labels", None) if labels is None else labels
    attributes = getattr(columns, "attributes", None) if attributes is None else attributes
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
    if title in COUNT_TABLES:   # structure types, coordination numbers, cluster sizes: bars
        _draw_counts(ax, x, columns, COUNT_TABLES[title], labels, first)
        if show_title:
            ax.set_title(title)
        return
    if len(names) > 1 and _named_rows(x, labels):   # one value per surface, per run, ...: bars
        _draw_named_rows(ax, columns, labels, first)
        if show_title:
            ax.set_title(title)
        return
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
    _draw_levels(ax, attributes or {}, x, colors)
    if len(ys) > 1 and (attributes or {}).get("levels"):   # over the level lines where it must
        ax.legend(loc="best", frameon=True, framealpha=0.85, edgecolor="none")
    elif len(ys) > 1:
        ax.legend(loc="best")
    if show_title:
        ax.set_title(title)


def figure(source, table: str | None = None, y: str | None = None, histogram: str | None = None,
           frame: int = 0, bins: int = 50, size: str | tuple | None = None, title: bool = False,
           xlim: tuple | None = None, ylim: tuple | None = None):
    """The matplotlib Figure of a view (what :func:`show` draws), without pyplot, in the
    publication style of :mod:`ewald.plotstyle` (``size``: a column width name or inches;
    ``xlim``/``ylim``: the range shown, e.g. ``xlim=(-3, 3)`` for a DOS energy window).
    Save it with :func:`ewald.plotstyle.save` to keep the text editable in PDF/SVG."""
    from . import plotstyle

    fig = plotstyle.figure(size)
    with plotstyle.context(*fig._ewald_style):
        ax = fig.add_subplot()
        _draw(ax, *view_data(source, table, y, histogram, frame, bins), histogram, show_title=title)
        _limit(ax, xlim, ylim)
    return fig


def _limit(ax, xlim, ylim):
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def show(source, table: str | None = None, y: str | None = None, histogram: str | None = None,
         frame: int = 0, bins: int = 50, xlim: tuple | None = None, ylim: tuple | None = None,
         panel: bool | None = None, save=None):
    """Show ``source`` (a Pipeline or a DataCollection) at ``frame``.

    ``show(pipe)`` alone (a Pipeline or DataCollection; a table still plots) is a picture of the structure, drawn with the pipeline's view settings (a
    :func:`snapshot`; under a notebook cell or in Jupyter it shows as an image, from a script it
    opens in a matplotlib window). With ``table`` (and ``y``) or ``histogram`` it plots a data table
    or a particle-property histogram with matplotlib (``plt.show()``) in the publication style;
    ``xlim``/``ylim`` set the range shown. ``save="rdf.pdf"`` also writes the figure (editable text
    in PDF/SVG, see :func:`ewald.plotstyle.save`). Returns the :class:`Snapshot` or a :class:`Plot`
    (``.figure`` is the matplotlib Figure).

    In the GUI, a plot of the notebook's pipeline is a live plot linked to the 3D view wherever it is
    called (the image stays for files and Jupyter); in a cell that is only this call, zooming writes
    the range back into the arguments. ``panel``: True opens the GUI's plot in the Plots panel (a
    thumbnail stays under the cell), False keeps it under the cell; None (default) decides by
    :func:`in_panel`."""
    if table is None and y is None and histogram is None and _is_structure(source):
        return _show_structure(source, frame, save)
    import matplotlib.pyplot as plt

    from . import plotstyle

    title, columns = view_data(source, table, y, histogram, frame, bins)
    with plotstyle.context():
        fig, ax = plt.subplots(figsize=plotstyle.figsize(), layout="constrained")
        _draw(ax, title, columns, histogram)
        _limit(ax, xlim, ylim)
    # what was asked, for the GUI: a show() of the notebook's pipeline is a live plot there
    args = {"table": table, "y": y, "histogram": histogram, "bins": bins,
            "xlim": None if xlim is None else [float(v) for v in xlim],
            "ylim": None if ylim is None else [float(v) for v in ylim],
            "panel": None if panel is None else bool(panel)}
    fig._ewald_view = (source, {k: v for k, v in args.items() if v is not None})
    if save is not None:
        plotstyle.save(fig, save)
    plt.show()
    return Plot(fig, title)


class Plot:
    """What :func:`show` returns for a plot: ``.figure`` is the matplotlib Figure (``.savefig``,
    ``.axes`` and the other Figure attributes work on it directly). Its repr is one line, so a
    notebook cell ending in ``show(...)`` shows the plot once, not twice."""

    def __init__(self, figure, title: str = ""):
        self.figure, self.title = figure, title

    def save(self, target, **kwargs):
        """Write the figure with the plot style's settings (:func:`ewald.plotstyle.save`)."""
        from . import plotstyle

        plotstyle.save(self.figure, target, **kwargs)

    def __getattr__(self, name):
        if name == "figure":
            raise AttributeError(name)
        return getattr(self.figure, name)

    def _ipython_display_(self):
        pass   # plt.show() has drawn the figure already

    def __repr__(self):
        return f"<Plot {self.title!r}>" if self.title else "<Plot>"


def _is_structure(source) -> bool:
    from .data import DataCollection

    return isinstance(source, DataCollection) or (hasattr(source, "compute") and hasattr(source, "num_frames"))


def _show_structure(source, frame: int, save) -> "Snapshot":
    """``show(pipe)``: the structure picture; from a plain script (no IPython) also in a
    matplotlib window, since nothing else would display the returned :class:`Snapshot`."""
    shot = snapshot(source, frame)
    if save is not None:
        shot.save(save)
    try:
        from IPython import get_ipython

        interactive = get_ipython() is not None
    except ImportError:
        interactive = False
    if not interactive:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(layout="constrained")
        ax.imshow(shot.image())
        ax.set_axis_off()
        plt.show()
    return shot


DRAW_ARGS = ("table", "y", "histogram", "frame", "bins")   # the arguments of view_data()


def in_panel(data, args: dict, num_frames: int | None = None) -> bool:
    """Whether the GUI shows the plot of ``show(source, **args)`` in the Plots panel (True) or
    under its cell (False). ``panel=`` in ``args`` decides when given. Otherwise a plot goes to
    the panel when it changes with the frame (``num_frames`` > 1) or is linked to the 3D view:
    a range on it selects atoms or frames (histograms, g(r), bond lengths and angles, time
    series, the NEB profile; :func:`ewald.linking.link_for`), it follows the picked atoms (DOS
    and bands) or a click plays a phonon mode. ``data`` is the frame shown (None: under the cell)."""
    if args.get("panel") is not None:
        return bool(args["panel"])
    if data is None:
        return False
    if num_frames is not None and num_frames > 1:
        return True
    from .linking import link_for

    hist = args.get("histogram")
    table = args.get("table")
    if not hist and table is None:
        table = next(iter(data.tables), None)
    y = args.get("y")
    try:
        if link_for(data, table, y, hist, num_frames) is not None:
            return True
    except Exception:  # noqa: BLE001 - a table the linking cannot read is just not linked
        pass
    if hist:
        return False
    if table in ("DOS", "Bands") and hasattr(data.attributes.get("Electronic Structure"), "dos_table"):
        return True
    return table == "Phonon Bands" and hasattr(data.attributes.get("Phonon Modes"), "frequencies")


def set_keyword(source: str, key: str, value, index: int = 0, function: str = "show") -> str | None:
    """``source`` with keyword ``key=value`` set in its ``index``-th ``function(<name>, ...)`` call
    (in reading order; ``value`` None removes the keyword). None when there is no such call or
    the source does not parse. The rest of the text stays as it was."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    calls = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == function and n.args and isinstance(n.args[0], ast.Name)),
                   key=lambda n: (n.lineno, n.col_offset))
    if not 0 <= index < len(calls):
        return None
    call = calls[index]
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line.encode("utf-8")))
    raw = source.encode("utf-8")

    def offset(lineno, col):
        return starts[lineno - 1] + col

    old = next((kw for kw in call.keywords if kw.arg == key), None)
    if old is not None:
        if value is None:   # drop ", key=value" with the comma before it
            others = [a for a in (*call.args, *call.keywords) if a is not old]
            before = max((a for a in others if (a.lineno, a.col_offset) < (old.lineno, old.col_offset)),
                         key=lambda a: (a.end_lineno, a.end_col_offset))
            lo, hi = offset(before.end_lineno, before.end_col_offset), offset(old.end_lineno, old.end_col_offset)
            new = raw[:lo] + raw[hi:]
        else:
            lo, hi = offset(old.value.lineno, old.value.col_offset), offset(old.value.end_lineno, old.value.end_col_offset)
            new = raw[:lo] + repr(value).encode("utf-8") + raw[hi:]
    elif value is None:
        return source
    else:
        last = max((*call.args, *call.keywords), key=lambda a: (a.end_lineno, a.end_col_offset))
        at = offset(last.end_lineno, last.end_col_offset)
        new = raw[:at] + f", {key}={value!r}".encode("utf-8") + raw[at:]
    return new.decode("utf-8")


class Snapshot:
    """A picture of the 3D view of ``source`` at ``frame``: shows as an image under a notebook
    cell (and in Jupyter); :meth:`save` writes it, :meth:`image` returns the RGB array.
    ``kwargs`` go to :func:`ewald.render.render` (``camera=``, ``size=``, style keywords)."""

    def __init__(self, source, frame: int = 0, **kwargs):
        self.source, self.frame, self.kwargs = source, frame, kwargs

    def _data(self):
        return self.source.compute(self.frame) if hasattr(self.source, "compute") else self.source

    def image(self, **kwargs) -> np.ndarray:
        from .display import options
        from .render import render

        from .view3d import merge

        data = self._data()
        kw = {k: v for k, v in options.items() if k not in ("enabled", "size")}
        kw = merge({**kw, **self._view_kwargs(data)}, {**self.kwargs, **kwargs})
        if kw.get("quality") is None:
            kw.pop("quality", None)
        return render(data, **kw)

    def _view_kwargs(self, data) -> dict:
        """The pipeline's :attr:`~ewald.pipeline.Pipeline.view` settings (style, colors, bonds, camera, ...)."""
        v = getattr(self.source, "view", None)
        return v.render_kwargs(data, self.frame) if v is not None else {}

    def save(self, filename, **kwargs):
        """Write the picture (PNG, JPG, SVG/PDF, ...) with the same view; ``kwargs`` override."""
        from .render import render

        from .view3d import merge

        data = self._data()
        return render(data, filename, **merge(self._view_kwargs(data), {**self.kwargs, **kwargs}))

    def _repr_png_(self):
        import io

        from PIL import Image

        from .display import _quiet_vtk, options

        if not options.get("enabled", True):
            return None
        restore = _quiet_vtk()
        try:
            img = self.image()
        except Exception:  # noqa: BLE001 - the text repr instead
            return None
        finally:
            if restore:
                restore()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()

    def __repr__(self):
        return f"<Snapshot frame {self.frame}>"


def snapshot(source, frame: int | None = None, **kwargs) -> Snapshot:
    """A picture of ``source`` (a Pipeline or DataCollection) at ``frame`` (default: the view's)
    drawn with the pipeline's ``view`` settings; ``camera=``, ``size=`` and the other keywords of
    :func:`ewald.render.render` override them. The last line of a cell shows it. Insert › 3D
    snapshot in the GUI writes one for the current view."""
    kwargs.setdefault("size", (480, 360))
    if frame is None:
        v = getattr(source, "view", None)
        frame = v.frame if v is not None else 0
    return Snapshot(source, frame, **kwargs)


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
    if function == "show" and args["source"] == "pipe" and not {"table", "y", "histogram"} & set(args):
        return None   # show(pipe) alone is a picture of the structure, not a plot (show(table) still plots)
    return args


def view_source(source: str = "pipe", **kwargs) -> str:
    """Source of a view cell: ``show(pipe, table='DOS')``; None values are left out."""
    args = "".join(f", {k}={v!r}" for k, v in kwargs.items() if v is not None and v != "")
    return f"show({source}{args})"
