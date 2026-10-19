"""Find, install and remove plugins: the data behind the GUI's Plugins panel (like VS Code's
Extensions view), without Qt.

Installed plugins come from the registry (:func:`ewald.plugins.status`), grouped by where they
were loaded from: a file or folder in the plugin folder, an installed package (entry point), or
code run in this session. Available plugins come from catalogs:

- the example plugins that ship with ewald's source (``examples/plugins``), and
- plugin indexes: JSON files at the URLs (or local paths) in the ``plugins.catalogs`` setting,
  so a group can publish its own list::

    {"plugins": [
      {"name": "Local strain", "summary": "Atomic strain against a reference frame",
       "kinds": ["modifier"], "source": "https://example.org/plugins/local_strain.py",
       "version": "0.2", "author": "Kim lab", "homepage": "https://example.org"},
      {"name": "ewald-cp2k", "summary": "CP2K readers", "kinds": ["reader"], "source": "pip:ewald-cp2k"}
    ]}

A ``source`` is a ``.py`` file (path or http(s) URL), copied into the plugin folder, or
``pip:<requirement>``, installed with pip into the Python ewald runs in. Removing a file
plugin moves it to ``<plugin folder>/.removed`` (put it back to restore it); a package is
removed with pip.
"""
from __future__ import annotations

import ast
import json
import shutil
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import plugins

KINDS = ("modifier", "reader", "writer", "visual", "overlay", "colormap", "theme")
_DECORATORS = {k: k for k in KINDS} | {f"register_{k}": k for k in KINDS}
_STATUS_KEYS = {"modifiers": "modifier", "readers": "reader", "writers": "writer", "visuals": "visual",
                "overlays": "overlay", "colormaps": "colormap", "themes": "theme"}


@dataclass
class Entry:
    """A plugin offered by a catalog."""

    name: str
    summary: str = ""
    kinds: list = field(default_factory=list)
    source: str = ""                 # .py path or URL, or "pip:<requirement>"
    version: str = ""
    author: str = ""
    homepage: str = ""
    description: str = ""
    catalog: str = ""                # where the entry came from

    @property
    def is_pip(self) -> bool:
        return self.source.startswith("pip:")

    @property
    def file_name(self) -> str:
        return self.source.rstrip("/").rsplit("/", 1)[-1] if not self.is_pip else ""


@dataclass
class Installed:
    """One installed plugin (a plugin-folder file or folder, a package), or the plugins
    registered by code run in this session; built from :func:`ewald.plugins.installed`."""

    name: str
    origin: str                      # file path, distribution name or "python"
    provides: dict = field(default_factory=dict)   # kind -> [registered names]
    error: str = ""
    package: str = ""                # distribution name for package plugins
    path: str = ""                   # the file or folder for plugin-folder plugins
    id: str = ""                     # plugins.installed() id; "" for this session's code
    version: str = ""
    summary: str = ""
    publisher: str = ""
    state: str = "loaded"            # "loaded", "disabled", "incompatible", "failed", "not loaded"

    @property
    def removable(self) -> bool:
        return bool(self.path or self.package)

    @property
    def enabled(self) -> bool:
        return self.state != "disabled"

    @property
    def kinds(self) -> list:
        return [k for k in KINDS if self.provides.get(k)]


# ---------------------------------------------------------------- installed
def installed() -> list[Installed]:
    """Every non-built-in plugin (disabled and broken ones too), plus one item for plugins
    registered by code run in this session."""
    folder = plugins.plugin_dir().resolve()
    items = []
    for pid, p in plugins.installed().items():
        if "@" in pid:                        # a duplicate id, already reported on the first one
            continue
        path = ""
        if p.kind != "package":
            src = Path(p.origin)
            if not src.exists():              # removed: its classes stay registered until restart
                continue
            path = str(src) if src.resolve().parent == folder else ""
        items.append(Installed(
            name=p.name or pid, origin=p.origin, id=pid, version=p.version, summary=p.description,
            publisher=p.publisher, state=p.state, error="; ".join(p.errors),
            package=p.origin if p.kind == "package" else "", path=path,
            provides={k.rstrip("s"): list(v) for k, v in p.contributions.items() if v}))
    session = Installed(name="This session", origin="python")
    status = plugins.status()
    for key, kind in _STATUS_KEYS.items():
        for entry in status[key]:
            if entry.get("origin") == "python":
                session.provides.setdefault(kind, []).append(entry["name"])
    if session.provides:
        items.append(session)
    return sorted(items, key=lambda i: i.name.lower())


def set_enabled(item: Installed, enabled: bool):
    """Switch a plugin on or off (kept across sessions); see :func:`ewald.plugins.set_enabled`."""
    plugins.set_enabled(item.id, enabled)


# ---------------------------------------------------------------- catalogs
def examples_dir() -> Path | None:
    """``examples/plugins`` of a source checkout (editable installs), if there is one."""
    here = Path(__file__).resolve()
    for parent in here.parents[1:4]:
        d = parent / "examples" / "plugins"
        if d.is_dir():
            return d
    return None


def describe_file(path) -> Entry:
    """An Entry for a plugin ``.py`` file, read without importing it: the docstring's first
    paragraph and the kinds of plugin its decorators register."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return Entry(name=_title(path.stem), source=str(path))
    doc = ast.get_docstring(tree) or ""
    kinds = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            for d in node.decorator_list:
                target = d.func if isinstance(d, ast.Call) else d
                name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
                kind = _DECORATORS.get(name)
                if kind and kind not in kinds:
                    kinds.append(kind)
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            kind = _DECORATORS.get(name) if name.startswith("register_") else None
            if kind and kind not in kinds:
                kinds.append(kind)
    first = " ".join(doc.split("\n\n")[0].split())
    summary = (first.split(": ", 1)[-1] if first.lower().startswith("example") else first).replace("``", "")
    summary = summary[:1].upper() + summary[1:]
    return Entry(name=_title(path.stem), summary=summary, kinds=[k for k in KINDS if k in kinds],
                 source=str(path), description=doc)


def _title(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip().capitalize()


def catalog(urls=(), timeout: float = 10.0) -> tuple[list[Entry], dict]:
    """Available plugins: the bundled examples and every index in ``urls``. Returns the entries
    and ``{url: error}`` for indexes that could not be read."""
    entries, errors = [], {}
    ex = examples_dir()
    if ex is not None:
        for path in sorted(ex.glob("*.py")):
            if not path.name.startswith("_"):
                e = describe_file(path)
                e.catalog = "examples"
                entries.append(e)
    for url in urls:
        url = url.strip()
        if not url:
            continue
        try:
            data = json.loads(_read(url, timeout))
            for raw in data.get("plugins", []):
                e = Entry(**{k: v for k, v in raw.items() if k in Entry.__dataclass_fields__})
                e.catalog = url
                if e.source and not e.is_pip and "://" not in e.source and not Path(e.source).is_absolute():
                    e.source = _join(url, e.source)  # relative to the index
                entries.append(e)
        except Exception as exc:
            errors[url] = f"{type(exc).__name__}: {exc}"
    return entries, errors


def _join(base: str, rel: str) -> str:
    if "://" in base:
        return base.rsplit("/", 1)[0] + "/" + rel
    return str(Path(base).expanduser().parent / rel)


def _read(source: str, timeout: float) -> str:
    if "://" in source:
        with urllib.request.urlopen(source, timeout=timeout) as r:  # noqa: S310 (user-configured URL)
            return r.read().decode("utf-8")
    return Path(source).expanduser().read_text(encoding="utf-8")


def search(entries, text: str) -> list:
    """Entries (or Installed items) whose name, summary or kinds contain every word of ``text``."""
    words = text.lower().split()

    def hay(e):
        return " ".join([e.name, getattr(e, "summary", ""), " ".join(e.kinds),
                         " ".join(n for names in getattr(e, "provides", {}).values() for n in names)]).lower()
    return [e for e in entries if all(w in hay(e) for w in words)]


# ---------------------------------------------------------------- install / remove
def is_installed(entry: Entry, items=None) -> bool:
    items = installed() if items is None else items
    if entry.is_pip:
        req = entry.source[4:].split("[")[0].split("=")[0].split("<")[0].split(">")[0].strip().lower()
        return any(i.package.lower().replace("_", "-") == req.replace("_", "-") for i in items if i.package)
    return (plugins.plugin_dir() / entry.file_name).exists()


def install_file(entry_or_source, timeout: float = 30.0) -> list[str]:
    """Copy a plugin ``.py`` (path or URL) into the plugin folder and load it; returns the
    names it registered. Replaces a file of the same name (an update)."""
    source = entry_or_source.source if isinstance(entry_or_source, Entry) else str(entry_or_source)
    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    name = source.rstrip("/").rsplit("/", 1)[-1]
    if not name.endswith(".py"):
        raise ValueError(f"not a plugin file: {source} (a .py file or pip:<package>)")
    target = folder / name
    if "://" in source:
        target.write_text(_read(source, timeout), encoding="utf-8")
    else:
        src = Path(source).expanduser().resolve()
        if src != target.resolve():
            shutil.copyfile(src, target)
    return plugins.load_plugin_file(target)


def remove_file(item: Installed) -> Path:
    """Move a plugin-folder plugin to ``.removed`` in the plugin folder; returns where it went.
    Its classes stay loaded until ewald restarts."""
    src = Path(item.path)
    trash = plugins.plugin_dir() / ".removed"
    trash.mkdir(parents=True, exist_ok=True)
    dest = trash / src.name
    if dest.exists():
        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
    shutil.move(str(src), dest)
    return dest


def pip_command(action: str, requirement: str) -> list[str]:
    """``python -m pip install|uninstall ...`` for the Python ewald runs in."""
    if action == "install":
        return [sys.executable, "-m", "pip", "install", requirement]
    return [sys.executable, "-m", "pip", "uninstall", "-y", requirement]
