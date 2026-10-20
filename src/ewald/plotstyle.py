"""The look of every plot ewald draws with matplotlib (GUI Plots tab, notebook view cells,
``Table`` pictures in Jupyter, figures saved into ``.ewd`` bundles)::

    from ewald import plotstyle
    fig = plotstyle.figure()                 # 1-column (3.5 in) figure in the publication style
    ax = fig.add_subplot()
    ax.plot(r, g)
    ax.set_xlabel(plotstyle.axis_label("r"))  # "$r$ (Å)"
    plotstyle.save(fig, "gr.pdf")            # vector text (Type 42 / SVG text), tight box

    with plotstyle.context():                # the style for your own pyplot figures
        plt.plot(...)
    plotstyle.use()                          # ... or for every figure from now on

The style follows journal figure guides: sans-serif text at 8 pt for a 3.5 in column, ticks
inside on all four sides with minor ticks, thin axes, 1.25 pt lines, no frame around the
legend, and a colorblind-safe color cycle (checked for deutan/protan/tritan separation and
contrast on its surface) that starts with the UI accent. ``tokens`` takes the GUI theme
colors (:data:`ewald.gui.theme.TOKENS`) so the Plots tab matches Paper or Graphite; files are
always written with :data:`PRINT` colors (white background, near-black ink).
"""
from __future__ import annotations

import contextlib

# Categorical cycles, in a fixed order. LIGHT is for white/Paper surfaces, DARK for Graphite.
LIGHT = ("#008C7E", "#D55E00", "#3A64C8", "#A0348C", "#B08600", "#2C8DD6")
DARK = ("#1FA090", "#DD6F2A", "#5B82E0", "#C055B0", "#B88E1A", "#3A9BE0")

# Colors of a saved figure: the GUI token names the plots use, for print.
PRINT = {"bg": "#FFFFFF", "panel": "#FFFFFF", "raised": "#FFFFFF", "text": "#1A1A1A", "secondary": "#333333",
         "muted": "#666666", "border": "#B3B3B3", "accent": LIGHT[0], "data": LIGHT[1], "pick": LIGHT[1],
         "selection": LIGHT[4]}

# Figure sizes (inches) by journal column width.
SIZES = {"1-column (3.5 in)": (3.5, 2.6), "1.5-column (5.0 in)": (5.0, 3.3), "2-column (7.2 in)": (7.2, 3.8)}
DEFAULT_SIZE = "1-column (3.5 in)"
FONT_SIZE = 8.0                       # pt, at the 1-column size
DPI = {"draft": 150, "standard": 300, "publication": 600}   # raster exports (PNG); vector is unaffected


SANS = ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "TeX Gyre Heros", "DejaVu Sans")


def _sans() -> str:
    """The first installed font of :data:`SANS` (mathtext needs a real family name, so that
    $r$ or 10$^3$ is set in the same face as the rest of the text)."""
    global _SANS
    if _SANS is None:
        from matplotlib import font_manager

        names = {f.name for f in font_manager.fontManager.ttflist}
        _SANS = next((n for n in SANS if n in names), "DejaVu Sans")
    return _SANS


_SANS = None


def _dark(color: str) -> bool:
    c = color.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5


def colors(tokens: dict | None = None) -> tuple:
    """The color cycle for a surface: :data:`DARK` on a dark ``tokens['bg']``, else :data:`LIGHT`.
    A colorful ``tokens['accent']`` (a UI theme's) leads the cycle in place of the cycle colors
    close to it in hue (at least the nearest), so the first line of a plot has the theme's accent."""
    import colorsys

    base = DARK if tokens and _dark(tokens.get("bg", "#FFFFFF")) else LIGHT
    accent = (tokens or {}).get("accent", "").upper()
    if not accent or accent in base:
        return base

    def hls(c):
        return colorsys.rgb_to_hls(*(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)))

    h, _, sat = hls(accent)
    if sat < 0.25:   # a gray accent says nothing about hue
        return base
    dist = [min(abs(hls(c)[0] - h), 1 - abs(hls(c)[0] - h)) for c in base]
    near = {k for k, d in enumerate(dist) if d < 0.07} or {dist.index(min(dist))}
    return (accent,) + tuple(c for k, c in enumerate(base) if k not in near)


def rc(tokens: dict | None = None, font_size: float = FONT_SIZE) -> dict:
    """matplotlib rcParams of the style; ``tokens`` are theme colors (default :data:`PRINT`)."""
    from matplotlib import cycler

    t = dict(PRINT, **(tokens or {}))
    fs = float(font_size)
    small = fs - 1 if fs > 6 else fs
    ink = t["secondary"]
    sans = _sans()
    return {
        "font.family": "sans-serif", "font.sans-serif": list(SANS),
        "mathtext.fontset": "custom", "mathtext.rm": sans, "mathtext.it": f"{sans}:italic",
        "mathtext.bf": f"{sans}:bold", "mathtext.cal": sans, "mathtext.sf": sans, "mathtext.default": "it",
        "font.size": fs, "axes.labelsize": fs, "axes.titlesize": fs, "figure.titlesize": fs,
        "xtick.labelsize": small, "ytick.labelsize": small, "legend.fontsize": small,
        "axes.titleweight": "normal", "axes.labelpad": 3.0, "axes.titlepad": 4.0,
        "text.color": t["text"], "axes.labelcolor": t["text"], "axes.titlecolor": t["text"],
        "axes.edgecolor": ink, "xtick.color": ink, "ytick.color": ink,
        "xtick.labelcolor": t["text"], "ytick.labelcolor": t["text"],
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "legend.labelcolor": t["text"],
        "axes.linewidth": 0.8, "axes.grid": False, "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=list(colors(tokens))),
        "axes.xmargin": 0.0, "axes.ymargin": 0.04,
        "axes.formatter.use_mathtext": True, "axes.formatter.limits": (-3, 4),
        "axes.unicode_minus": True,
        "lines.linewidth": 1.25, "lines.markersize": 4.0, "lines.markeredgewidth": 0.8,
        "patch.linewidth": 0.6, "hatch.linewidth": 0.6, "errorbar.capsize": 2.0,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.major.pad": 3.0, "ytick.major.pad": 3.0,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.4, "legend.labelspacing": 0.3, "legend.columnspacing": 1.2,
        "image.cmap": "viridis",
        "figure.dpi": 150, "figure.constrained_layout.use": False,
        "savefig.dpi": DPI["publication"], "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }


def context(tokens: dict | None = None, font_size: float = FONT_SIZE):
    """``with plotstyle.context(): ...``: the style for figures made (and saved) inside."""
    import matplotlib as mpl

    return mpl.rc_context(rc(tokens, font_size))


STYLES = ("publication", "default")   # the user-plot styles of :class:`Session` and :func:`use`
_SKIP = ("backend", "backend_fallback", "interactive")   # never swapped around a cell


def _style_rc(style: str = "publication", tokens: dict | None = None, font_size: float = FONT_SIZE) -> dict:
    """What ``style`` sets on top of the process defaults: the publication rc plus a 1-column
    figure size and constrained layout for plain pyplot figures; nothing for 'default'."""
    if style == "default":
        return {}
    if style != "publication":
        raise ValueError(f"unknown plot style {style!r}; use one of {', '.join(STYLES)}")
    out = rc(tokens, font_size)
    out.update({"figure.figsize": SIZES[DEFAULT_SIZE], "figure.constrained_layout.use": True,
                "savefig.bbox": "tight"})
    return out


def use(style: str | dict | None = "publication", tokens: dict | None = None, font_size: float = FONT_SIZE):
    """Set matplotlib's rcParams for your own figures: ``use()`` the publication style (in
    ``tokens`` colors), ``use("default")`` matplotlib's own defaults. In an ewald notebook or
    the GUI console the change lasts for that notebook only (see :class:`Session`)."""
    import matplotlib as mpl

    if isinstance(style, dict) or style is None:   # use(tokens) of the first version
        style, tokens = "publication", style
    if style == "default":
        dict.update(mpl.rcParams, {k: v for k, v in mpl.rcParamsDefault.items() if k not in _SKIP})
    else:
        mpl.rcParams.update(_style_rc(style, tokens, font_size))


class Session:
    """The matplotlib rcParams of one notebook or console, swapped in around each cell::

        session = Session()                # publication style
        with session.active():
            exec(cell)                     # plt.plot(...) here is drawn in the style

    What a cell changes (``plt.rcParams[...] = ...``, ``plt.style.use(...)``, :func:`use`) is
    kept for the next cells of that notebook, and :meth:`configure` (another style, or a UI theme
    change) leaves those keys alone. Outside :meth:`active` the process rcParams are untouched,
    so other notebooks and ewald's own renderers never see a notebook's style."""

    def __init__(self, style: str = "publication", tokens: dict | None = None):
        self.style, self.tokens = "default", None
        self._state = None   # this session's rcParams (all keys but _SKIP), made on first use
        self._applied = {}   # what the style set last, to tell user changes from ours
        self.configure(style, tokens)

    def _base(self) -> dict:
        import matplotlib as mpl

        return {k: v for k, v in mpl.rcParams.items() if k not in _SKIP}

    def configure(self, style: str = "publication", tokens: dict | None = None):
        """Switch the style or its colors; keys a cell set itself keep their values."""
        import matplotlib as mpl

        style = (style or "publication").lower()
        new = dict(mpl.RcParams(_style_rc(style, tokens)))   # validated, as rcParams holds them
        if self._state is not None:
            defaults = self._base()
            for k in set(self._applied) | set(new):
                ours = self._applied.get(k, defaults.get(k))
                if k in self._state and self._state[k] == ours:   # untouched by the user
                    self._state[k] = new.get(k, defaults.get(k))
        self.style, self.tokens, self._applied = style, tokens, new

    @contextlib.contextmanager
    def active(self):
        """Inside: this session's rcParams; after: the process's again, the session keeps changes."""
        import matplotlib as mpl

        saved = self._base()
        if self._state is None:
            self._state = dict(saved, **self._applied)
        dict.update(mpl.rcParams, self._state)
        try:
            yield self
        finally:
            self._state = self._base()
            dict.update(mpl.rcParams, saved)


def figsize(size: str | tuple | None = None) -> tuple:
    """Inches of a :data:`SIZES` name (a prefix such as "2-column" is enough) or a (w, h) pair."""
    if size is None:
        return SIZES[DEFAULT_SIZE]
    if isinstance(size, str):
        for name, wh in SIZES.items():
            if name.startswith(size):
                return wh
        raise ValueError(f"unknown plot size {size!r}; use one of {', '.join(SIZES)}")
    return tuple(size)


def figure(size: str | tuple | None = None, tokens: dict | None = None, font_size: float | None = None):
    """A matplotlib Figure (no pyplot) in the style, laid out with constrained layout.

    ``font_size`` defaults to :data:`FONT_SIZE` scaled with the width, so a 2-column figure is
    not drawn with 1-column type."""
    from matplotlib.figure import Figure

    w, h = figsize(size)
    fs = FONT_SIZE * max(1.0, (w / 3.5) ** 0.5) if font_size is None else font_size
    with context(tokens, fs):
        fig = Figure(figsize=(w, h), layout="constrained")
    fig._ewald_style = (tokens, fs)
    return fig


def save(fig, target, format: str | None = None, quality: str = "publication", dpi: float | None = None, **kwargs):
    """``fig.savefig`` with the style's output settings: editable text in PDF/EPS/SVG, a tight
    box, and :data:`DPI` of ``quality`` for raster formats (``dpi=`` wins)."""
    tokens, fs = getattr(fig, "_ewald_style", (None, FONT_SIZE))
    with context(tokens, fs):
        fig.savefig(target, format=format, dpi=dpi or DPI[quality.lower()], **kwargs)


def no_minor_x(ax):
    """Drop the minor x ticks (axes with labeled points: band paths, mode numbers)."""
    from matplotlib.ticker import NullLocator

    ax.xaxis.set_minor_locator(NullLocator())


# Axis labels with units for the column names of ewald's tables. Legends keep the raw names.
LABELS = {
    "r": "$r$ (Å)", "g(r)": "$g(r)$", "Pair separation": "$r$ (Å)",
    "q": "$q$ (Å$^{-1}$)", "S(q)": "$S(q)$", "2theta": r"$2\theta$ (°)", "Intensity": "Intensity (arb. units)",
    "d": "$d$ (Å)", "Experimental": "Intensity (arb. units)",
    "Bond Length": "Bond length (Å)", "Angle": "Bond angle (°)", "Cosine": r"$\cos\theta$",
    "MSD": "MSD (Å$^2$)", "MSD fit": "MSD (Å$^2$)", "Charge MSD": "Charge MSD (Å$^2$)",
    "MQD": "MQD (Å$^4$)", "alpha2": r"$\alpha_2$", "VACF (normalized)": "VACF (normalized)",
    "Reaction Coordinate": "Reaction coordinate (Å)", "Energy": "Energy (eV)",
    "Frequency": "Frequency (THz)", "Count": "Count", "count": "Count",
    "Coordination": "Coordination number", "Displacement Magnitude": "Displacement (Å)",
    "CSP": "Centrosymmetry (Å$^2$)", "RMSD": "RMSD",
}


def _split(name: str) -> tuple[str, str]:
    """``("g(r)", "Cu-Zr")`` for a per-pair or per-type column such as "g(r) Cu-Zr" or "Count Cu"."""
    for key in sorted(LABELS, key=len, reverse=True):
        if name.startswith(key + " ") and len(name) > len(key) + 1:
            return key, name[len(key) + 1:]
    return name, ""


def axis_label(name: str) -> str:
    """The axis label of table column ``name``: :data:`LABELS` (also for per-pair/per-type
    columns like "g(r) Cu-Zr" or "Count Cu"), else the name itself."""
    base, _ = _split(name)
    return LABELS.get(base, name)


def legend_labels(names) -> list[str]:
    """Legend entries for columns drawn on one axis: when they all share a quantity
    ("g(r)", "g(r) Cu-Cu", ...), just what tells them apart ("Total", "Cu-Cu", ...)."""
    names = list(names)
    parts = [_split(n) for n in names]
    if len(names) > 1 and len({b for b, _ in parts}) == 1 and parts[0][0] in LABELS:
        fit = any(rest == "fit" for _, rest in parts)
        return [n if not rest and fit else "Total" if not rest else "Fit" if rest == "fit" else rest.replace("-", "–")
                for n, (_, rest) in zip(names, parts)]
    return names


def start_at_zero(ax, x):
    """Put the x axis origin at 0 when the data start just above it (bin centers of g(r), counts)."""
    import numpy as np

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) > 1 and 0 <= x.min() <= 0.05 * (x.max() - x.min()):
        ax.set_xlim(left=0)
