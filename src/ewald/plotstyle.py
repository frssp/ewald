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
    plotstyle.use("user")                    # ... with your style file on top (see below)

The style follows journal figure guides: sans-serif text at 8 pt for a 3.5 in column, ticks
inside on all four sides with minor ticks, thin axes, 1.25 pt lines, no frame around the
legend, and a colorblind-safe color cycle (checked for deutan/protan/tritan separation and
contrast on its surface) that starts with the UI accent. ``tokens`` takes the GUI theme
colors (:data:`ewald.gui.theme.TOKENS`) so the Plots tab matches Paper or Graphite; files are
always written with :data:`PRINT` colors (white background, near-black ink).

Your own style is a matplotlib ``.mplstyle`` file, ``<config dir>/plots/user.mplstyle``
(:func:`user_style_path`), drawn over the publication style: in the GUI it applies to the Plots
panel, notebook cells and the console alike. The file is the source; Preferences › Plots shows
its common keys as a form (:data:`FORM`) and writes back only what the form changed
(:func:`write_style`). Scripts never read it implicitly: ``use("user")``, ``use("acs")`` (any
``<config dir>/plots/<name>.mplstyle``) or ``use("path/to/file.mplstyle")`` ask for it. On
screen the UI theme keeps the background and text colors (:data:`THEME_KEYS`); saved figures
use the file's.
"""
from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path

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


def _sans(families=None) -> str:
    """The first installed font of ``families`` (default :data:`SANS`; mathtext needs a real
    family name, so that $r$ or 10$^3$ is set in the same face as the rest of the text)."""
    global _SANS, _FONTS
    if _FONTS is None:
        from matplotlib import font_manager

        _FONTS = {f.name.lower(): f.name for f in font_manager.fontManager.ttflist}
    if families:
        found = next((_FONTS[n.lower()] for n in families if n.lower() in _FONTS), None)
        if found:
            return found
    if _SANS is None:
        _SANS = next((_FONTS[n.lower()] for n in SANS if n.lower() in _FONTS), "DejaVu Sans")
    return _SANS


_SANS = _FONTS = None


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


def rc(tokens: dict | None = None, font_size: float = FONT_SIZE, user: dict | None = None) -> dict:
    """matplotlib rcParams of the style; ``tokens`` are theme colors (default :data:`PRINT`).
    ``user`` (a style file's rcParams, :func:`read_style`) goes on top; its ``font.size`` and
    ``font.sans-serif`` also set the smaller tick and legend sizes and the math font."""
    from matplotlib import cycler

    user = validated(user)
    t = dict(PRINT, **(tokens or {}))
    fs = float(user.get("font.size", font_size))
    small = fs - 1 if fs > 6 else fs
    ink = t["secondary"]
    sans = _sans(user.get("font.sans-serif"))
    out = {
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
    out.update(user)
    return out


def context(tokens: dict | None = None, font_size: float = FONT_SIZE, user: dict | None = None):
    """``with plotstyle.context(): ...``: the style for figures made (and saved) inside."""
    import matplotlib as mpl

    return mpl.rc_context(rc(tokens, font_size, user))


STYLES = ("publication", "default")   # the user-plot styles of :class:`Session` and :func:`use`
_SKIP = ("backend", "backend_fallback", "interactive")   # never swapped around a cell


def _style_rc(style: str = "publication", tokens: dict | None = None, font_size: float = FONT_SIZE,
              user: dict | None = None) -> dict:
    """What ``style`` sets on top of the process defaults: the publication rc plus a 1-column
    figure size and constrained layout for plain pyplot figures; nothing for 'default'. ``user``
    (a style file's rcParams) goes on top of either, without :data:`THEME_KEYS` when the colors
    are a UI theme's (``tokens``)."""
    if tokens is not None:
        user = screen_rc(user)
    if style == "default":
        return validated(user)
    if style != "publication":
        raise ValueError(f"unknown plot style {style!r}; use one of {', '.join(STYLES)}")
    out = rc(tokens, font_size, user)
    extra = {"figure.figsize": SIZES[DEFAULT_SIZE], "figure.constrained_layout.use": True, "savefig.bbox": "tight"}
    out.update({k: v for k, v in extra.items() if k not in (user or {})})
    return out


def use(style: str | dict | None = "publication", tokens: dict | None = None, font_size: float = FONT_SIZE):
    """Set matplotlib's rcParams for your own figures: ``use()`` the publication style (in
    ``tokens`` colors), ``use("default")`` matplotlib's own defaults, ``use("user")`` the
    publication style with your style file (:func:`user_style_path`) on top, ``use("acs")`` with
    ``<config dir>/plots/acs.mplstyle``, ``use("my.mplstyle")`` with that file. In an ewald
    notebook or the GUI console the change lasts for that notebook only (see :class:`Session`)."""
    import warnings

    import matplotlib as mpl

    if isinstance(style, dict) or style is None:   # use(tokens) of the first version
        style, tokens = "publication", style
    user = None
    if style not in STYLES:
        path = style_file(style)
        if path is None:
            raise ValueError(f"unknown plot style {style!r}; use one of {', '.join(STYLES + ('user',))}, "
                             f"a name in {user_style_path().parent} or a .mplstyle file")
        user, problems = read_style(path)
        if not path.is_file():
            problems = [f"{path} does not exist"] if style != "user" else []
        for problem in problems:
            warnings.warn(f"plot style {path.name}: {problem}", stacklevel=2)
        style = "publication"
    if style == "default":
        dict.update(mpl.rcParams, {k: v for k, v in mpl.rcParamsDefault.items() if k not in _SKIP})
    else:
        mpl.rcParams.update(_style_rc(style, tokens, font_size, user))


_ACTIVE_LOCK = threading.RLock()
_ACTIVE = [0, None]   # sessions active now (cells of several notebooks on threads), process rcParams


class Session:
    """The matplotlib rcParams of one notebook or console, swapped in around each cell::

        session = Session()                # publication style
        with session.active():
            exec(cell)                     # plt.plot(...) here is drawn in the style

    What a cell changes (``plt.rcParams[...] = ...``, ``plt.style.use(...)``, :func:`use`) is
    kept for the next cells of that notebook, and :meth:`configure` (another style, or a UI theme
    change) leaves those keys alone. Outside :meth:`active` the process rcParams are untouched,
    so other notebooks and ewald's own renderers never see a notebook's style."""

    def __init__(self, style: str = "publication", tokens: dict | None = None, user: dict | None = None):
        self.style, self.tokens, self.user = "default", None, None
        self._state = None   # this session's rcParams (all keys but _SKIP), made on first use
        self._applied = {}   # what the style set last, to tell user changes from ours
        self.configure(style, tokens, user)

    def _base(self) -> dict:
        import matplotlib as mpl

        return {k: v for k, v in mpl.rcParams.items() if k not in _SKIP}

    def configure(self, style: str = "publication", tokens: dict | None = None, user: dict | None = None):
        """Switch the style, its colors or the style file's rcParams (``user``); keys a cell set
        itself keep their values."""
        import matplotlib as mpl

        style = (style or "publication").lower()
        new = dict(mpl.RcParams(_style_rc(style, tokens, user=user)))   # validated, as rcParams holds them
        if self._state is not None:
            defaults = self._base()
            for k in set(self._applied) | set(new):
                ours = self._applied.get(k, defaults.get(k))
                if k in self._state and self._state[k] == ours:   # untouched by the user
                    self._state[k] = new.get(k, defaults.get(k))
        self.style, self.tokens, self.user, self._applied = style, tokens, user, new

    @contextlib.contextmanager
    def active(self):
        """Inside: this session's rcParams; after: the process's again, the session keeps changes."""
        import matplotlib as mpl

        with _ACTIVE_LOCK:
            if not _ACTIVE[0]:   # the process's own rcParams, restored when the last session ends
                _ACTIVE[1] = self._base()
            _ACTIVE[0] += 1
            saved = _ACTIVE[1]
            if self._state is None:
                self._state = dict(saved, **self._applied)
            dict.update(mpl.rcParams, self._state)
        try:
            yield self
        finally:
            with _ACTIVE_LOCK:
                self._state = self._base()
                _ACTIVE[0] -= 1
                if not _ACTIVE[0]:   # while another notebook's cell runs on another thread, its rcParams stay
                    dict.update(mpl.rcParams, saved)


# ---------------------------------------------------------------- style files

STYLE_DIR = "plots"              # <config dir>/plots/*.mplstyle: named styles for use(name)
USER_STYLE = "user.mplstyle"     # the one the GUI draws with and Preferences › Plots edits
# On a UI theme's colors (the screen) the theme keeps these; saved figures take the file's.
THEME_KEYS = ("figure.facecolor", "axes.facecolor", "savefig.facecolor", "text.color", "axes.labelcolor",
              "axes.titlecolor", "axes.edgecolor", "xtick.color", "ytick.color", "xtick.labelcolor",
              "ytick.labelcolor", "legend.labelcolor", "legend.facecolor", "legend.edgecolor", "grid.color")


def user_style_path() -> Path:
    """``<config dir>/plots/user.mplstyle`` (``EWALD_CONFIG_DIR`` moves it with the settings)."""
    from .settings import config_dir

    return config_dir() / STYLE_DIR / USER_STYLE


def named_styles() -> dict:
    """``{name: path}`` of the ``.mplstyle`` files in ``<config dir>/plots`` (names lower-case)."""
    folder = user_style_path().parent
    try:
        entries = sorted(os.scandir(folder), key=lambda e: e.name.lower())
    except OSError:
        return {}
    return {os.path.splitext(e.name)[0].lower(): Path(e.path) for e in entries
            if e.is_file() and e.name.lower().endswith(".mplstyle")}


def style_file(name) -> Path | None:
    """The file of a style name: "user", a name in ``<config dir>/plots`` (any case) or a path
    ending in ``.mplstyle``; None for anything else."""
    text = os.fspath(name)
    if text.lower().endswith(".mplstyle"):
        return Path(os.path.expanduser(text))
    if text.lower() == "user":
        return user_style_path()
    return named_styles().get(text.lower())


def _strip_comment(line: str) -> str:
    """``line`` up to a ``#`` that is not inside quotes (as matplotlibrc reads it)."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "#":
            return line[:i]
    return line


def _key_of(line: str, commented: bool = False) -> str | None:
    """The rcParams key a line sets (``commented``: of a ``# key: value`` line), else None."""
    import matplotlib as mpl

    text = line.strip()
    if commented:
        if not text.startswith("#"):
            return None
        text = text.lstrip("#")
    key, sep, _ = _strip_comment(text).partition(":")
    key = key.strip()
    return key if sep and key in mpl.rcParams.validate else None


def parse_style(text: str) -> tuple[dict, list]:
    """``(rc, problems)`` of ``.mplstyle`` text: ``{key: value text}`` of every line matplotlib
    accepts (a later line wins), and one message per line it would not ("line 3: ...")."""
    import matplotlib as mpl

    rc_, problems = {}, []
    for n, line in enumerate(text.lstrip("\ufeff").splitlines(), 1):   # CRLF and a BOM read alike
        body = _strip_comment(line).strip()
        if not body:
            continue
        key, sep, value = body.partition(":")
        key, value = key.strip(), value.strip()
        if not sep or not key:
            problems.append(f"line {n}: expected 'key: value', not {body!r}")
        elif key not in mpl.rcParams.validate:
            problems.append(f"line {n}: {key!r} is not a matplotlib rcParam")
        elif key in _SKIP:
            problems.append(f"line {n}: {key} cannot be set by a style")
        else:
            try:
                mpl.rcParams.validate[key](value)
            except (ValueError, TypeError) as exc:
                problems.append(f"line {n}: {key}: {exc}")
            else:
                rc_[key] = value
    return rc_, problems


def read_style(path=None) -> tuple[dict, list]:
    """:func:`parse_style` of a file (default :func:`user_style_path`); ``({}, [])`` when it does
    not exist, so a fresh install draws the publication style."""
    path = Path(path) if path is not None else user_style_path()
    try:
        with open(path, encoding="utf-8-sig") as fh:
            text = fh.read()
    except FileNotFoundError:
        return {}, []
    except (OSError, UnicodeDecodeError) as exc:
        return {}, [f"{path}: {exc}"]
    return parse_style(text)


def validated(user: dict | None) -> dict:
    """``user`` as rcParams values (``{key: text}`` or values already); keys matplotlib would
    refuse are left out (a style file is checked when read, see :func:`parse_style`)."""
    import matplotlib as mpl

    out = {}
    for key, value in (user or {}).items():
        if key in _SKIP or key not in mpl.rcParams.validate:
            continue
        try:
            out[key] = mpl.rcParams.validate[key](value)
        except (ValueError, TypeError):
            pass
    return out


def screen_rc(user: dict | None) -> dict:
    """``user`` without :data:`THEME_KEYS`, for plots drawn in a UI theme's colors."""
    return {k: v for k, v in (user or {}).items() if k not in THEME_KEYS}


def format_value(value) -> str:
    """An rcParams value as ``.mplstyle`` text (hex colors without ``#``, which starts a comment)."""
    def one(v):
        text = str(v)
        return text[1:] if text.startswith("#") and len(text) in (4, 5, 7, 9) else text

    if hasattr(value, "by_key"):   # a cycler
        return "cycler(" + ", ".join(f"{k}=[{', '.join(repr(one(v)) for v in vs)}]"
                                     for k, vs in value.by_key().items()) + ")"
    if isinstance(value, (list, tuple)):
        return ", ".join(one(v) for v in value)
    return one(value)


def template() -> str:
    """A new style file: how it works, then every key of the publication style commented out."""
    lines = ["# EWALD plot style: matplotlib rcParams drawn over EWALD's publication style in the",
             "# Plots panel, notebook cells and the console (help: matplotlib 'customizing').",
             "# This file is the source: Preferences > Plots shows its common keys and writes back",
             "# only what you change there. Uncomment a line to set it; a line you add wins.",
             "# On screen the UI theme keeps the background and text colors; saved figures use yours.",
             "# In a script or notebook: ewald.plotstyle.use(\"user\").",
             "#",
             "# The publication style's values:"]
    lines += [f"# {k}: {format_value(v)}" for k, v in sorted(rc().items())]
    return "\n".join(lines) + "\n"


def write_style(updates: dict, path=None) -> Path:
    """Set keys of a style file (default :func:`user_style_path`) from ``{key: value}``; None
    comments a key out. Other lines and comments stay; a key that is only commented out is set
    on that line; a new file starts from :func:`template`."""
    from .fileops import write_text

    path = Path(path) if path is not None else user_style_path()
    try:
        with open(path, encoding="utf-8-sig") as fh:
            lines = fh.read().splitlines()
    except FileNotFoundError:
        lines = template().splitlines()
    for key, value in updates.items():
        active = [i for i, line in enumerate(lines) if _key_of(line) == key]
        if value is None or active:
            for i in active[value is not None:]:   # one line per key: the first takes the value
                lines[i] = "# " + lines[i]
        if value is None:
            continue
        text = f"{key}: {value if isinstance(value, str) else format_value(value)}"
        commented = [i for i, line in enumerate(lines) if _key_of(line, commented=True) == key]
        if active:
            lines[active[0]] = text
        elif commented:
            lines[commented[0]] = text
        else:
            lines.append(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_text(path, "\n".join(lines) + "\n")


@dataclass(frozen=True)
class FormField:
    """One row of the Preferences › Plots form: a value for ``keys`` of the style file.
    ``kind``: "text", "float", "choice" (of ``choices``), "bool" or "colors" (a color cycle)."""
    label: str
    keys: tuple
    kind: str
    help: str = ""
    choices: tuple = ()

    def default(self):
        """The publication style's value (the form's placeholder)."""
        return self.value({k: v for k, v in rc().items() if k in self.keys}) if self.keys else None

    def value(self, rc_: dict):
        """This field's value in a style file's rcParams (None: not set there)."""
        if self.keys[0] not in rc_:
            return None
        v = validated({self.keys[0]: rc_[self.keys[0]]}).get(self.keys[0])
        if v is None:
            return None
        if self.kind == "colors":
            from matplotlib.colors import to_hex

            return ", ".join(to_hex(c).upper() for c in v.by_key().get("color", []))
        if self.kind == "text":
            return ", ".join(v) if isinstance(v, (list, tuple)) else str(v)
        return float(v) if self.kind == "float" else v

    def updates(self, value) -> dict:
        """``{key: text or None}`` for :func:`write_style` (None or "" unsets the keys). A bad
        value raises ValueError with a message for the user."""
        if value is None or value == "":
            return dict.fromkeys(self.keys)
        if self.kind == "colors":
            from matplotlib.colors import to_hex

            def color(c):   # "D55E00" as matplotlibrc writes it, "#D55E00", or a name
                return to_hex("#" + c if len(c) in (3, 6, 8) and all(x in "0123456789abcdefABCDEF" for x in c) else c)

            try:
                hexes = [color(c.strip()).upper()[1:] for c in str(value).split(",") if c.strip()]
            except ValueError as exc:
                raise ValueError(f"{self.label}: {exc}") from None
            text = "cycler(color=[" + ", ".join(repr(h) for h in hexes) + "])"
        elif self.kind == "bool":
            text = "True" if value else "False"
        else:
            text = format_value(value)
        problems = parse_style(f"{self.keys[0]}: {text}")[1]
        if problems:
            raise ValueError(f"{self.label}: {problems[0].split(': ', 2)[-1]}")
        return dict.fromkeys(self.keys, text)


# The common keys of a style file, shown as a form in Preferences › Plots (the rest: Open as text).
FORM = (
    FormField("Font", ("font.sans-serif",), "text", "Sans-serif fonts in order of preference; the first "
              "installed one is used, also for math."),
    FormField("Font size", ("font.size",), "float", "Points; tick labels and legends are 1 pt smaller."),
    FormField("Line width", ("lines.linewidth",), "float", "Points."),
    FormField("Axes line width", ("axes.linewidth",), "float", "Points; ticks follow the axes."),
    FormField("Tick direction", ("xtick.direction", "ytick.direction"), "choice", choices=("in", "out", "inout")),
    FormField("Minor ticks", ("xtick.minor.visible", "ytick.minor.visible"), "bool"),
    FormField("Legend frame", ("legend.frameon",), "bool"),
    FormField("Color cycle", ("axes.prop_cycle",), "colors", "Line colors in order, as hex codes separated by "
              "commas; empty = the colorblind-safe cycle."),
)


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
    return bind(fig, (tokens, fs))


def _draw_styled(fig, renderer):
    style = fig._ewald_style
    with context(*(style() if callable(style) else style)):
        return type(fig).draw(fig, renderer)


def bind(fig, style=(None, FONT_SIZE)):
    """Draw ``fig`` in the style wherever it is drawn: a GUI canvas paints it from the Qt event
    loop, and ``savefig`` may run after the ``with context()`` that made its artists. Text made
    in the style sets math in the style's ``mathtext.fontset`` ("custom"), which reads the
    ``mathtext.rm``/``it``/``cal``... families when drawn; outside the style those are matplotlib's
    defaults (``cal`` = "cursive": Apple Chancery on macOS, with a findfont warning). ``style``
    is ``(tokens, font_size)`` or ``(tokens, font_size, user)``, or a function returning one
    (a theme or style file that may change)."""
    import functools

    fig._ewald_style = style
    fig.draw = functools.partial(_draw_styled, fig)
    return fig


def save(fig, target, format: str | None = None, quality: str = "publication", dpi: float | None = None, **kwargs):
    """``fig.savefig`` with the style's output settings: editable text in PDF/EPS/SVG, a tight
    box, and :data:`DPI` of ``quality`` for raster formats (``dpi=`` wins)."""
    style = getattr(fig, "_ewald_style", (None, FONT_SIZE))
    with context(*(style() if callable(style) else style)):
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
