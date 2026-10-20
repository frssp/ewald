"""What each kind of plugin is: one dataclass per registry (readers, writers, modifiers, visual
elements, overlays, color maps, themes, UI themes, AI tools, GUI panels and settings pages), the
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

_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")


def path_scheme(path) -> str | None:
    """The URI scheme of ``path`` (``"aiida"`` for ``aiida:1234``), or None."""
    m = _SCHEME.match(path) if isinstance(path, str) else None
    return m.group(1).lower() if m else None


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
        return any(fnmatch.fnmatch(base, p) for p in self.patterns)

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
        return any(fnmatch.fnmatch(base, p) or fnmatch.fnmatch(base.lower(), p.lower()) for p in self.patterns)

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
    """A color theme of the desktop app: ``tokens`` (a dict, or a function returning one) overrides
    the design tokens of the ``base`` palette (a built-in theme such as "paper" or "graphite")."""

    name: str
    tokens: object
    base: str = "paper"
    description: str = ""
    origin: str = ""
    builtin: bool = False

    def palette(self) -> dict:
        """Every design token as ``#rrggbb``: the base palette with this theme's overrides."""
        from .ui_tokens import PALETTES

        own = self.tokens() if callable(self.tokens) else self.tokens
        if not isinstance(own, dict):
            raise TypeError(f"UI theme {self.name!r} returned {type(own).__name__}, not a dict of tokens")
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
    """A side bar of the desktop app: a button on the activity bar (left edge) shows or hides it,
    and an entry under Window › Plugin panels. ``factory(context)`` builds its ``QWidget`` the first
    time it is opened (not at start), with ``context`` an ``ewald.gui.window_plugins.PanelContext``.
    ``icon`` is the name of a built-in icon or an SVG/PNG file (resolved next to the plugin)."""

    name: str
    factory: Callable
    title: str = ""
    icon: str = "plugins"
    tooltip: str = ""
    shortcut: str = ""
    width: int = 320
    description: str = ""
    origin: str = ""


@dataclass
class SettingsPagePlugin:
    """A page of Preferences: ``options`` are ``ewald.settings.Option`` under one key prefix of
    the plugin's own (``simagent.server``, ``simagent.timeout``, ...), stored in settings.json
    with the core settings; ``sections`` groups their keys (default: one section, ``name``)."""

    name: str
    options: tuple
    description: str = ""
    sections: tuple = ()          # ((title, (key, ...)), ...)
    icon: str = ""
    origin: str = ""

    @property
    def prefix(self) -> str:
        return self.options[0].key.split(".")[0]

    def layout(self) -> tuple:
        """``(description, [(section, [keys])])`` like ``ewald.gui.preferences.LAYOUT``."""
        sections = [(t, list(keys)) for t, keys in self.sections] or [(self.name, [o.key for o in self.options])]
        return self.description, sections


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
