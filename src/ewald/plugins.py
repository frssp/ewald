"""Plugins: file readers (and later modifiers, tools, panels) supplied outside ewald.

A plugin registers itself in one of two ways.

Installed package, via a Python entry point in its ``pyproject.toml``::

    [project.entry-points."ewald.readers"]
    cp2k-xyz = "ewald_cp2k:Cp2kXyzReader"

A single file dropped into ``<config dir>/plugins/`` (``~/.config/ewald/plugins`` by
default, ``$EWALD_CONFIG_DIR/plugins`` when set), using the decorator::

    # ~/.config/ewald/plugins/cp2k.py
    import numpy as np
    from ewald import DataCollection
    from ewald.plugins import reader

    @reader("cp2k-xyz", patterns=("*-pos-*.xyz",), description="CP2K position trajectory")
    def read_cp2k(path, **kwargs):
        ...                                   # return a list of DataCollection or ase.Atoms

Then ``import_file("run-pos-1.xyz")``, the GUI's File > Open and the Claude tools all
pick the reader. A reader is one of

- a ``FrameReader`` subclass (lazy: ``__len__`` + ``read(i)``), constructed as ``cls(path, **kwargs)``;
- a function ``f(path, **kwargs)`` returning a ``FrameReader``, one frame, or a list of frames
  (``DataCollection`` or ``ase.Atoms``), which are then held in memory.

ewald's own formats are registered the same way (origin ``built-in``), so ``ewald plugins``,
the GUI open filter and the Claude tools list every format from this one registry.

Format selection: ``format="name"`` (or an alias such as ``"lammps/dump"``) picks that reader;
an unknown name is handed to ASE. Without a format, readers are tried by ``priority``: the
volumetric readers (cube 5, VASP CHGCAR/LOCPOT/ELFCAR 6), the native LAMMPS dump/data
and XDATCAR readers (10-12), then plugins (50 unless given), then
ASE (100) for everything else. A reader with ``sniff(path)`` is matched on content;
one without is matched on the file-name ``patterns`` (compression suffix ignored). Registering
an existing name replaces that reader. A plugin that fails to import is recorded in
:func:`status` (and ``ewald plugins``) instead of breaking ewald. Set ``EWALD_NO_PLUGINS=1``
to skip loading external plugins.
"""
from __future__ import annotations

import fnmatch
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .io.base import FrameReader, strip_compression

log = logging.getLogger(__name__)

READER_GROUP = "ewald.readers"


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

    def matches(self, path) -> bool:
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


def _builtin_readers() -> dict:
    from . import io

    return {p.name: p for p in io.builtin_readers()}


@dataclass
class _Registry:
    readers: dict = field(default_factory=_builtin_readers)
    errors: dict = field(default_factory=dict)      # origin -> message
    loaded: bool = False


_registry = _Registry()


def plugin_dir() -> Path:
    from .settings import config_dir

    return config_dir() / "plugins"


def reader(name: str, patterns=(), sniff: Callable | None = None, description: str = "",
           priority: int | None = 50, aliases=(), category: str = "structure"):
    """Decorator registering a reader class or function under the format ``name``."""

    def deco(factory):
        register_reader(name, factory, patterns=patterns, sniff=sniff, description=description,
                        priority=priority, aliases=aliases, category=category)
        return factory

    return deco


def register_reader(name: str, factory: Callable, patterns=(), sniff: Callable | None = None,
                    description: str = "", origin: str = "python", priority: int | None = 50,
                    aliases=(), category: str = "structure") -> ReaderPlugin:
    if isinstance(patterns, str):
        patterns = (patterns,)
    # attributes on the factory fill in what the call left out (entry-point classes carry their own)
    patterns = tuple(patterns or getattr(factory, "patterns", ()))
    sniff = sniff or getattr(factory, "sniff", None)
    doc = (factory.__doc__ or "").strip().splitlines()
    description = description or getattr(factory, "description", "") or (doc[0] if doc else "")
    plugin = ReaderPlugin(name.lower(), factory, patterns, sniff, description,
                          getattr(_loading, "origin", None) or origin, priority,
                          tuple(a.lower() for a in aliases), category=category)
    _registry.readers[plugin.name] = plugin
    return plugin


class _Loading:
    origin: str | None = None


_loading = _Loading()


def load_plugins(force: bool = False):
    """Import entry points and files in :func:`plugin_dir` once per process (``force`` reloads)."""
    if _registry.loaded and not force:
        return
    _registry.loaded = True
    if force:
        keep = {k: v for k, v in _registry.readers.items() if v.origin == "python"}
        _registry.readers = {**_builtin_readers(), **keep}
        _registry.errors.clear()
    if os.environ.get("EWALD_NO_PLUGINS"):
        return
    _load_entry_points()
    _load_directory(plugin_dir())


def _load_entry_points():
    from importlib.metadata import entry_points

    try:
        eps = entry_points(group=READER_GROUP)
    except Exception as exc:
        _registry.errors[READER_GROUP] = str(exc)
        return
    for ep in eps:
        origin = f"entry point {ep.value}"
        try:
            factory = ep.load()
            _loading.origin = origin
            if ep.name.lower() not in _registry.readers or _registry.readers[ep.name.lower()].factory is not factory:
                register_reader(ep.name, factory)
        except Exception as exc:
            _registry.errors[origin] = f"{type(exc).__name__}: {exc}"
        finally:
            _loading.origin = None


def _load_directory(folder: Path):
    if not folder.is_dir():
        return
    for path in sorted(folder.iterdir()):
        if path.name.startswith(("_", ".")):
            continue
        if path.suffix == ".py":
            target = path
        elif path.is_dir() and (path / "__init__.py").exists():
            target = path / "__init__.py"
        else:
            continue
        module_name = f"ewald_plugin_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(
                module_name, target, submodule_search_locations=[str(path)] if path.is_dir() else None)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            _loading.origin = str(path)
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            _registry.errors[str(path)] = f"{type(exc).__name__}: {exc}"
        finally:
            _loading.origin = None


def readers() -> dict[str, ReaderPlugin]:
    load_plugins()
    return dict(_registry.readers)


def find_reader(path, format: str | None = None) -> ReaderPlugin:
    """The reader for ``format`` (name or alias; unknown names go to ASE), else the first by
    priority that recognises ``path``."""
    plugins = readers()
    if format:
        fmt = format.lower()
        for plugin in plugins.values():
            if fmt == plugin.name or fmt in plugin.aliases:
                return plugin
        return plugins["ase"]          # unknown names are ASE format names
    ranked = sorted((p for p in plugins.values() if p.priority is not None), key=lambda p: p.priority)
    for plugin in ranked:
        if plugin.matches(path):
            return plugin
    raise ValueError(f"no reader recognises {path}")


def open_reader(path, format: str | None = None, **kwargs) -> FrameReader:
    return find_reader(path, format).open(path, format=format, **kwargs)


def file_patterns(category: str | None = None) -> list[str]:
    """Glob patterns of all readers (of one ``category``), in priority order (for file dialogs)."""
    out = []
    for plugin in sorted(readers().values(), key=lambda p: (p.priority is None, p.priority or 0)):
        if category and plugin.category != category:
            continue
        out += [p for p in plugin.patterns if p not in out]
    return out


def status() -> dict:
    """What is installed and what failed to load, as plain data (CLI, GUI, LLM tools)."""
    load_plugins()
    return {"plugin_dir": str(plugin_dir()),
            "readers": [{"name": p.name, "aliases": list(p.aliases), "patterns": list(p.patterns),
                         "description": p.description, "origin": p.origin, "priority": p.priority}
                        for p in sorted(_registry.readers.values(),
                                        key=lambda p: (p.priority is None, p.priority or 0, p.name))],
            "errors": dict(_registry.errors)}


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


__all__ = ["reader", "register_reader", "load_plugins", "readers", "find_reader", "open_reader", "file_patterns",
           "status", "plugin_dir", "ReaderPlugin", "ListReader"]
