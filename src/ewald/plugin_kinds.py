"""What each kind of plugin is: one dataclass per registry (readers, writers, modifiers, visual
elements, overlays, color maps, themes, UI themes, AI tools, GUI panels, settings pages), the
built-in entries and
:class:`ListReader`. Stateless; the registry itself and the loading live in :mod:`ewald.plugins`,
which re-exports everything here."""
from __future__ import annotations

import fnmatch
import inspect
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable

from .io.base import FrameReader, strip_compression

log = logging.getLogger("ewald.plugins")

_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]+):")   # 2+ characters: "C:" is a Windows drive


def path_scheme(path) -> str | None:
    """The URI scheme of ``path`` (``"aiida"`` for ``aiida:1234``), or None. A single letter is
    a Windows drive (``C:\\data``), never a scheme, on every platform."""
    m = _SCHEME.match(path) if isinstance(path, str) else None
    return m.group(1).lower() if m else None


def _matches(name: str, patterns) -> bool:
    """File-name patterns ignore case on every platform (``fnmatch.fnmatch`` ignores it only
    on Windows): ``POSCAR*`` matches ``poscar`` everywhere."""
    low = name.lower()
    return any(fnmatch.fnmatchcase(low, p.lower()) for p in patterns)


@dataclass
class ReaderPlugin:
    name: str
    factory: Callable
    patterns: tuple = ()
    sniff: Callable | None = None
    description: str = ""
    origin: str = ""             # "built-in", entry point, file path, or "python" for runtime registration
    priority: int | None = 50    # lower is tried first; None = only by explicit format=
    aliases: tuple = ()
    takes_format: bool = False   # factory receives the requested format name (the ASE fallback)
    category: str = "structure"  # "structure" or "volumetric" (groups the GUI open filter)
    schemes: tuple = ()          # URI schemes it serves: ("aiida",) takes every "aiida:..." path

    def matches(self, path) -> bool:
        if self.schemes:
            return path_scheme(path) in self.schemes
        if self.sniff is not None:
            try:
                return bool(self.sniff(str(path)))
            except Exception as exc:  # a broken sniffer must not block other formats
                log.debug("sniff of %s failed for %s: %s", self.name, path, exc)
                return False
        base = os.path.basename(strip_compression(path))
        return _matches(base, self.patterns)

    def open(self, path, format: str | None = None, **kwargs):
        if self.takes_format:
            kwargs["format"] = format
        if isinstance(self.factory, type) and issubclass(self.factory, FrameReader):
            return self.factory(path, **kwargs)
        result = self.factory(path, **kwargs)
        return result if isinstance(result, FrameReader) else ListReader(result)


@dataclass
class WriterPlugin:
    """A file writer, the export-side twin of :class:`ReaderPlugin`.

    ``factory(path, frames, **kwargs)`` gets an iterator of ``DataCollection`` (frames are
    computed one at a time, so a streaming writer never holds the whole trajectory); with
    ``multi_frame=False`` it gets one ``DataCollection`` instead: ``factory(path, data, **kwargs)``.
    Without ``format=``, writers are matched on the output file name ``patterns`` by
    ``priority``; ASE (100) takes everything else.
    """
    name: str
    factory: Callable
    patterns: tuple = ()
    description: str = ""
    origin: str = ""
    priority: int | None = 50
    aliases: tuple = ()
    takes_format: bool = False   # factory receives the requested format name (the ASE fallback)
    multi_frame: bool = True     # False: one frame per file (POSCAR, CHGCAR, cube, LAMMPS data)
    category: str = "structure"

    def matches(self, path) -> bool:
        base = os.path.basename(strip_compression(path))
        return _matches(base, self.patterns)

    def write(self, path, frames, format: str | None = None, **kwargs):
        """Write ``frames`` (iterable of DataCollection) to ``path``; a ``.gz``/``.bz2``/``.xz``
        name is written uncompressed next to it first, then compressed."""
        plain = strip_compression(path)
        if plain != str(path):
            import shutil
            import tempfile

            from .io.base import _OPENERS

            folder, base = os.path.split(plain)
            fd, tmp = tempfile.mkstemp(prefix=".ewald-", suffix="-" + base, dir=folder or ".")
            os.close(fd)
            try:
                result = self.write(tmp, frames, format=format, **kwargs)
                opener = _OPENERS[str(path)[len(plain):]]
                with open(tmp, "rb") as src, opener(str(path), "wb") as dst:
                    shutil.copyfileobj(src, dst)
            finally:
                os.remove(tmp)
            return result
        if self.takes_format:
            kwargs["format"] = format
        if self.multi_frame:
            return self.factory(path, iter(frames), **kwargs)
        it = iter(frames)
        try:
            first = next(it)
        except StopIteration:
            raise ValueError("nothing to write") from None
        if next(it, None) is not None:
            raise ValueError(f"format {self.name!r} holds one frame; pick one, e.g. frames=[-1]")
        return self.factory(path, first, **kwargs)


def _builtin_readers() -> dict:
    from . import io

    return {p.name: p for p in io.builtin_readers()}


def _builtin_writers() -> dict:
    from . import io

    return {p.name: p for p in io.builtin_writers()}


@dataclass
class ModifierPlugin:
    name: str
    cls: type
    category: str = "Plugins"    # submenu of the GUI's Add modification menu
    description: str = ""
    origin: str = ""


@dataclass
class VisualPlugin:
    name: str
    cls: type
    category: str = "Plugins"
    description: str = ""
    origin: str = ""
    builtin: bool = False


def _builtin_visuals() -> dict:
    from . import scene

    out = {}
    for cls in scene.BUILTIN:
        doc = inspect.cleandoc(cls.__doc__ or "").split("\n\n")[0]
        out[cls.name] = VisualPlugin(cls.name, cls, "Built-in", " ".join(doc.split()), "built-in", True)
    return out


@dataclass
class OverlayPlugin:
    name: str
    cls: type
    category: str = "Plugins"
    description: str = ""
    origin: str = ""
    builtin: bool = False


def _builtin_overlays() -> dict:
    from . import overlays

    out = {}
    for cls in overlays.BUILTIN:
        doc = inspect.cleandoc(cls.__doc__ or "").split("\n\n")[0]
        out[cls.name] = OverlayPlugin(cls.name, cls, "Built-in", " ".join(doc.split()), "built-in", True)
    return out


@dataclass
class ColormapPlugin:
    """A named color map; ``factory()`` returns a matplotlib Colormap or a list/array of colors."""

    name: str
    factory: Callable
    kind: str = "sequential"
    description: str = ""
    origin: str = ""
    _cmap: object = field(default=None, repr=False)

    def colormap(self):
        if self._cmap is None:
            from .colormaps import as_colormap

            value = self.factory() if callable(self.factory) and not hasattr(self.factory, "N") else self.factory
            self._cmap = as_colormap(self.name, value)
        return self._cmap


@dataclass
class ThemePlugin:
    """A named theme; ``factory()`` returns a :class:`ewald.themes.Theme` (or it is one)."""

    name: str
    factory: object
    description: str = ""
    origin: str = ""
    builtin: bool = False

    def theme(self):
        from .themes import Theme

        t = self.factory if isinstance(self.factory, Theme) else self.factory()
        if not isinstance(t, Theme):
            raise TypeError(f"theme {self.name!r} returned {type(t).__name__}, not a Theme")
        return t


def _builtin_themes() -> dict:
    from .themes import BUILTIN

    return {k: ThemePlugin(k, t, t.description, "built-in", True) for k, t in BUILTIN.items()}


@dataclass
class UiThemePlugin:
    """A theme of the desktop app: ``tokens`` (a dict, or a function returning one) overrides
    the design tokens of the ``base`` palette (a built-in theme such as "paper" or "graphite"),
    and may also set fonts, sizes, corner radius, spacing, icon line width and extra style
    sheet rules (:meth:`style`)."""

    name: str
    tokens: object
    base: str = "paper"
    description: str = ""
    origin: str = ""
    builtin: bool = False

    def _own(self) -> dict:
        own = self.tokens() if callable(self.tokens) else self.tokens
        if not isinstance(own, dict):
            raise TypeError(f"UI theme {self.name!r} returned {type(own).__name__}, not a dict of tokens")
        return own

    def style(self) -> dict:
        """Everything but colors (``ewald.ui_tokens.STYLE_DEFAULTS``: font, mono_font, font_size,
        radius, density, icon_stroke, qss), defaults filled in."""
        from .ui_tokens import STYLE_DEFAULTS, check_style

        return check_style(self.name, {k: v for k, v in self._own().items() if k in STYLE_DEFAULTS})

    def palette(self) -> dict:
        """Every design token as ``#rrggbb``: the base palette with this theme's overrides."""
        from .ui_tokens import PALETTES, STYLE_DEFAULTS

        own = {k: v for k, v in self._own().items() if k not in STYLE_DEFAULTS}
        base = PALETTES.get(self.base)
        if base is None:
            raise ValueError(f"UI theme {self.name!r}: base must be one of {', '.join(PALETTES)}")
        unknown = sorted(set(own) - set(base))
        if unknown:
            raise ValueError(f"UI theme {self.name!r}: unknown tokens {', '.join(unknown)} "
                             f"(known: {', '.join(base)})")
        bad = [k for k, v in own.items() if not (isinstance(v, str) and _HEX.fullmatch(v))]
        if bad:
            raise ValueError(f"UI theme {self.name!r}: tokens {', '.join(bad)} must be colors like '#1e1e2e'")
        if not own:
            return dict(base)
        from .ui_tokens import derive

        return derive({**base, **own}, own)


_HEX = re.compile(r"#[0-9a-fA-F]{6}")


def _builtin_ui_themes() -> dict:
    from .ui_tokens import DESCRIPTIONS, PALETTES

    return {name: UiThemePlugin(name, {}, name, DESCRIPTIONS.get(name, ""), "built-in", True) for name in PALETTES}


def _builtin_workflows() -> dict:
    from .workflow_folder import FolderSource

    src = FolderSource()
    return {"folders": WorkflowSourcePlugin("folders", src, src.title, (), "", " ".join(src.__doc__.split()),
                                            "built-in")}


def _builtin_panels() -> dict:
    from .workflow import workflow_panel

    f = workflow_panel("folders", title="Workflows", icon="workflow",
                       tooltip="Workflows: folders of VASP, LAMMPS, NEB and phonopy calculations")
    return {"workflows": PanelPlugin("workflows", f, f.title, f.icon, f.tooltip, "", 320,
                                     "Folders of calculations as workflows: their steps, outputs and one notebook",
                                     "built-in")}


def _builtin_codes() -> dict:
    from .codes import builtin_codes   # hooks are lazy "module:function" strings

    return builtin_codes()


def _builtin_ai_providers() -> dict:
    from .llm.providers import BUILTIN   # no SDK import

    return dict(BUILTIN)


@dataclass
class ToolPlugin:
    """An AI tool: ``function(toolset, **arguments)`` returns JSON-ready data (or a
    ``ewald.llm.tools.ToolResult``); ``schema`` is the JSON schema of the arguments."""

    name: str
    function: Callable
    description: str = ""
    schema: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    origin: str = ""


@dataclass
class PanelPlugin:
    """A panel of the desktop app: with ``area="side"`` a side bar that a button on the activity
    bar (left edge) shows or hides, with ``area="bottom"`` a tab under the 3D view (next to
    Python console, View, Data inspector, Plots and Log); either has an entry under Window ›
    Plugin panels. ``factory(context)`` builds its ``QWidget`` the first time it is opened (not at
    start), with ``context`` an ``ewald.gui.window_plugins.PanelContext``. ``icon`` is the name of
    a built-in icon or an SVG/PNG file (resolved next to the plugin)."""

    name: str
    factory: Callable
    title: str = ""
    icon: str = "plugins"
    tooltip: str = ""
    shortcut: str = ""
    width: int = 320
    description: str = ""
    origin: str = ""
    area: str = "side"


PANEL_AREAS = ("side", "bottom")
#: Where a ``@menu_item`` can go: a menu of the menu bar, the main toolbar, or the right-click
#: menu of the 3D view, of a notebook cell (its ⋯ button) or of a plot.
MENUS = ("File", "Edit", "View", "Notebook", "Insert", "Tools", "Window", "Help", "Toolbar", "3D view",
         "Notebook cell", "Plot")
#: The menus whose entries get ``context.target`` (the cell, the plot).
TARGET_MENUS = ("Notebook cell", "Plot")
MENU_WHEN = ("", "data", "picks")


@dataclass
class PlotTarget:
    """What a ``@menu_item(menu="Plot")`` entry gets as ``context.target``: the plot it was opened
    on. ``table`` is the name of the table (or the histogrammed property) on it, ``values`` that
    ``Table`` (None for a histogram or an electronic-structure plot), ``columns`` the plotted
    columns, ``xlim``/``ylim`` the axis ranges on screen and ``data`` the frame it shows."""

    table: str = ""
    values: object = None
    columns: list = field(default_factory=list)
    xlim: tuple | None = None
    ylim: tuple | None = None
    data: object = None


@dataclass
class MenuItemPlugin:
    """An entry a plugin adds to the desktop app: ``menu`` is one of :data:`MENUS`, optionally
    with a submenu path (``"Tools/Sim agent"``; on the toolbar a drop-down button); choosing it
    calls ``callback(context)`` with an ``ewald.gui.window_plugins.PanelContext``, whose
    ``target`` is the notebook cell or plot it was opened on. ``when`` is "" (always on), "data"
    (a structure is loaded) or "picks" (atoms are picked); in a right-click menu an entry whose
    ``when`` does not hold is left out, in the menu bar and the toolbar it is greyed out."""

    name: str
    callback: Callable
    title: str
    menu: str = "Tools"
    shortcut: str = ""
    icon: str = ""
    when: str = ""
    description: str = ""
    origin: str = ""


@dataclass
class WorkflowSourcePlugin:
    """A place workflows come from (sim-agent, an AiiDA profile): ``source`` has ``account()``,
    ``list(query, cursor) -> WorkflowPage`` and ``describe(ref) -> Workflow`` (:mod:`ewald.workflow`),
    and optionally ``analyses(wf) -> [Analysis]`` for a described workflow;
    ``schemes`` are the ``load()`` path schemes its workflows use, ``settings_page`` the plugin
    settings page that sets it up ("" if none)."""

    name: str
    source: object
    title: str = ""
    schemes: tuple = ()
    settings_page: str = ""
    description: str = ""
    origin: str = ""

    def account(self) -> str:
        fn = getattr(self.source, "account", None)
        return str(fn() or "") if callable(fn) else "ready"

    def list(self, query: str | None = None, cursor: str | None = None):
        return self._named(self.source.list(query=query, cursor=cursor))

    def describe(self, ref: str):
        wf = self.source.describe(ref)
        wf.source = wf.source or self.name
        offer = getattr(self.source, "analyses", None)
        if callable(offer) and getattr(wf, "complete", False) and not wf.analyses:
            try:
                wf.analyses = list(offer(wf) or [])
            except Exception as exc:  # noqa: BLE001 - a broken recipe must not hide the workflow
                from .workflow import Analysis

                wf.analyses = [Analysis("analyses", reason=f"{self.name} could not offer analyses: {exc}")]
        return wf

    def _named(self, page):
        for wf in page.workflows:
            wf.source = wf.source or self.name
        return page


class ListReader(FrameReader):
    """FrameReader over frames already in memory (what function readers return)."""

    def __init__(self, frames):
        from .data import DataCollection

        if not isinstance(frames, (list, tuple)):
            frames = [frames]
        if not frames:
            raise ValueError("reader plugin returned no frames")
        self.frames = [f if isinstance(f, DataCollection) else DataCollection.from_ase(f) for f in frames]

    def __len__(self):
        return len(self.frames)

    def read(self, index):
        return self.frames[index].copy()


#: Kinds of :class:`SettingsField`: a line of text, a password-style secret, a check box, a
#: choice from ``choices``, several lines of text, a whole number.
FIELD_KINDS = ("text", "secret", "bool", "choice", "multiline", "int")


@dataclass
class SettingsField:
    """One entry of a plugin's settings page; the dialog is generated from these.

    ``secret`` fields never show what is stored: ``load()`` returns a true value when a secret
    is saved (the field then says so) and ``save()`` gets ``""`` when the user left it empty,
    meaning "keep the saved one". Keep secrets in :mod:`ewald.keychain`, never in a file.

    ``advanced`` fields (rarely changed: a CA file, host aliases) start folded under
    **Advanced**, which says how many of them hold a value. ``help`` is the field's tooltip."""

    name: str
    label: str = ""
    kind: str = "text"
    help: str = ""
    choices: tuple = ()
    default: object = None
    placeholder: str = ""
    advanced: bool = False

    def __post_init__(self):
        if self.kind not in FIELD_KINDS:
            raise ValueError(f"settings field {self.name!r}: kind must be one of {', '.join(FIELD_KINDS)}, "
                             f"not {self.kind!r}")
        if self.kind == "choice" and not self.choices:
            raise ValueError(f"settings field {self.name!r}: a choice needs choices")
        self.label = self.label or self.name.replace("_", " ").capitalize()
        self.choices = tuple(self.choices)

    def empty(self):
        """The value of this field when ``load()`` says nothing about it."""
        if self.default is not None:
            return self.default
        return {"bool": False, "int": 0, "choice": self.choices[0] if self.choices else ""}.get(self.kind, "")


@dataclass
class SettingsPagePlugin:
    """A plugin's own page of settings (Plugins side bar › Settings…, Edit › Plugin settings).

    ``load()`` returns the current values by field name; ``save(values)`` checks and stores them
    and returns a short message for the user, or raises ``ValueError`` with what to fix.
    ``actions`` are extra buttons, label -> ``f(values)`` returning a message (``Test``, ``Log out``)."""

    name: str
    title: str
    fields: list
    load: Callable
    save: Callable
    description: str = ""
    actions: dict = field(default_factory=dict)
    origin: str = ""

    def values(self) -> dict:
        """``load()`` with every field present (missing ones get their empty value)."""
        got = dict(self.load() or {})
        return {f.name: got.get(f.name, f.empty()) for f in self.fields}
