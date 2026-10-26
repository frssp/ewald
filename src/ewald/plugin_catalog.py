"""Find, install and remove plugins: the data behind the GUI's Plugins panel (like VS Code's
Extensions view), without Qt.

Installed plugins come from the registry (:func:`ewald.plugins.status`), grouped by where they
were loaded from: a file or folder in the plugin folder, an installed package (entry point), or
code run in this session. Available plugins come from catalogs:

- the example plugins that ship with ewald's source (``examples/plugins``), and
- plugin indexes: the default index (:data:`DEFAULT_CATALOG`, or ``EWALD_PLUGIN_INDEX``) and JSON files at the URLs (or
  local paths) in the ``python.plugin_catalogs`` setting, so a group can publish its own list::

    {"format": 1, "plugins": [
      {"id": "local_strain", "name": "Local strain", "summary": "Atomic strain against a reference frame",
       "kinds": ["modifier"], "source": "https://example.org/plugins/local_strain.py",
       "version": "0.2", "requires_ewald": ">=0.1", "author": "Kim lab", "homepage": "https://example.org",
       "sha256": "9f2c..."},
      {"id": "ewald-cp2k", "name": "ewald-cp2k", "summary": "CP2K readers", "kinds": ["reader"],
       "source": "pip:ewald-cp2k"}
    ]}

An entry matches an installed plugin by ``id`` (the plugin's id: its file stem or package
name), then by name. A newer ``version`` offers an update; an entry whose ``requires_ewald``
this ewald does not meet is listed but cannot be installed. A file plugin without an
``__ewald__`` version has the version of the entry it was installed from (remembered in
``<config dir>/plugin-sources.json`` with the file's SHA-256), and a file whose content is the
entry's ``sha256`` is up to date whatever its version says.

Indexes and files on GitHub (``raw.githubusercontent.com``, ``api.github.com``) are read with a
token when there is one, from ``GITHUB_TOKEN`` or ``gh auth token``, so a private index works.
The token is sent to those hosts only and is never stored; for a ``pip:git+https://github.com/...``
source it goes to the pip process through git's environment, not its command line.

A ``source`` is a ``.py`` file (path or http(s) URL), copied into the plugin folder after its
``sha256`` (when the entry has one) matches, or
``pip:<requirement>``, installed with pip into the Python ewald runs in. Removing a file
plugin moves it to ``<plugin folder>/.removed`` (put it back to restore it); a package is
removed with pip.
"""
from __future__ import annotations

import ast
import base64
import importlib.metadata
import importlib.util
import json
import os
import posixpath
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from . import plugins
from ._tls import ssl_context
from .trust import sha256, verify_sha256

KINDS = ("modifier", "reader", "writer", "visual", "overlay", "colormap", "theme", "tool", "ui_theme", "ai_provider",
         "settings_page", "panel", "menu_item", "workflow_source", "simulation_code")
_DECORATORS = {k: k for k in KINDS} | {f"register_{k}": k for k in KINDS} | {"claude_tool": "tool", "ai_tool": "tool"}
DEFAULT_CATALOG = "https://raw.githubusercontent.com/frssp/ewald-plugins/main/index.json"
TOKEN_HOSTS = ("raw.githubusercontent.com", "api.github.com")
_STATUS_KEYS = {"modifiers": "modifier", "readers": "reader", "writers": "writer", "visuals": "visual",
                "overlays": "overlay", "colormaps": "colormap", "themes": "theme", "tools": "tool",
                "ui_themes": "ui_theme", "ai_providers": "ai_provider", "settings_pages": "settings_page",
                "panels": "panel", "menu_items": "menu_item", "workflows": "workflow_source",
                "codes": "simulation_code"}


@dataclass
class Entry:
    """A plugin offered by a catalog."""

    name: str
    summary: str = ""
    kinds: list = field(default_factory=list)
    source: str = ""                 # .py path or URL, or "pip:<requirement>"
    version: str = ""
    id: str = ""                     # the installed plugin's id (file stem or package name)
    requires_ewald: str = ""         # e.g. ">=0.2, <1"
    author: str = ""
    homepage: str = ""
    description: str = ""
    sha256: str = ""                 # hex SHA-256 of a .py source; checked before it is installed
    catalog: str = ""                # where the entry came from

    @property
    def is_pip(self) -> bool:
        return self.source.startswith("pip:")

    @property
    def file_name(self) -> str:
        return "" if self.is_pip else source_name(self.source)

    @property
    def package(self) -> str:
        """The distribution name of a ``pip:`` source ("" for git URLs and files)."""
        if not self.is_pip or "://" in self.source:
            return ""
        req = self.source[4:]
        for sep in "[=<>~!; ":
            req = req.split(sep)[0]
        return req.strip()

    @property
    def compatible(self) -> bool:
        """Whether this ewald meets ``requires_ewald`` (an unreadable requirement does not)."""
        try:
            return plugins.version_satisfies(plugins.ewald_version(), self.requires_ewald)
        except ValueError:
            return False


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
        path, version = "", p.version
        if p.kind != "package":
            src = Path(p.origin)
            if not src.exists():              # removed: its classes stay registered until restart
                continue
            path = str(src) if src.resolve().parent == folder else ""
            if path and not version:          # a catalog file without __ewald__: the version it was installed as
                version = installed_version(path)
        items.append(Installed(
            name=p.name or pid, origin=p.origin, id=pid, version=version, summary=p.description,
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


def default_catalog() -> str:
    """The default plugin index: ``EWALD_PLUGIN_INDEX`` if set (empty turns it off), else
    :data:`DEFAULT_CATALOG`."""
    return os.environ.get("EWALD_PLUGIN_INDEX", DEFAULT_CATALOG).strip()


def catalog_urls(setting: str = "", default: bool = True) -> list[str]:
    """The default index followed by the ``;``- or newline-separated URLs of ``setting``."""
    urls = [default_catalog()] if default and default_catalog() else []
    for u in setting.replace("\n", ";").split(";"):
        if u.strip() and u.strip() not in urls:
            urls.append(u.strip())
    return urls


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
            if int(data.get("format", 1)) > 1:
                raise ValueError(f"index format {data['format']} is newer than this ewald reads (1)")
            for raw in data.get("plugins", []):
                e = Entry(**{k: v for k, v in raw.items() if k in Entry.__dataclass_fields__})
                e.catalog = url
                if e.source and not e.is_pip and "://" not in e.source and not Path(e.source).is_absolute():
                    e.source = _join(url, e.source)  # relative to the index
                entries.append(e)
        except Exception as exc:
            errors[url] = f"{type(exc).__name__}: {exc}"
    return entries, errors


def source_name(source: str) -> str:
    """The file name of a plugin source: the last part of a URL's path, or of a local path
    with either separator (``C:\\plugins\\x.py`` gives ``x.py`` on every platform)."""
    if "://" in source:
        return posixpath.basename(urllib.parse.urlsplit(source).path.rstrip("/"))
    return PureWindowsPath(source.rstrip("/\\")).name   # splits on / and on backslashes


def _join(base: str, rel: str) -> str:
    if "://" in base:
        return base.rsplit("/", 1)[0] + "/" + rel
    return str(Path(base).expanduser().parent / rel)


def github_token() -> str:
    """A GitHub token from ``GITHUB_TOKEN`` or ``gh auth token``; "" when there is none.
    Read each time it is needed and never stored."""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token
    gh = shutil.which("gh")
    if gh is None:
        return ""
    try:
        out = subprocess.run([gh, "auth", "token"], capture_output=True, timeout=5,
                             encoding="utf-8", errors="replace",
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))  # no console flash on Windows
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _wants_token(url: str) -> bool:
    u = urllib.parse.urlsplit(url)
    return u.scheme == "https" and (u.hostname or "").lower() in TOKEN_HOSTS


def _request(url: str) -> urllib.request.Request:
    req = urllib.request.Request(url)
    token = github_token() if _wants_token(url) else ""
    if token:
        req.add_unredirected_header("Authorization", f"token {token}")  # not sent on to a redirect
    return req


def _read(source: str, timeout: float) -> str:
    if "://" in source:
        with urllib.request.urlopen(_request(source), timeout=timeout, context=ssl_context()) as r:  # noqa: S310 (user-configured URL)
            return r.read().decode("utf-8")
    return Path(source).expanduser().read_text(encoding="utf-8")


def is_missing(error: str) -> bool:
    """Whether a :func:`catalog` error means the index is not there (HTTP 404, no such file)."""
    return "404" in error or error.startswith("FileNotFoundError")


def search(entries, text: str) -> list:
    """Entries (or Installed items) whose name, summary or kinds contain every word of ``text``."""
    words = text.lower().split()

    def hay(e):
        return " ".join([e.name, getattr(e, "summary", ""), " ".join(e.kinds),
                         " ".join(n for names in getattr(e, "provides", {}).values() for n in names)]).lower()
    return [e for e in entries if all(w in hay(e) for w in words)]


# ---------------------------------------------------------------- install / remove
def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def match(entry: Entry, items=None) -> Installed | None:
    """The installed plugin a catalog entry describes: by id, then name, then the package or
    plugin-folder file it would install."""
    items = installed() if items is None else items
    for key, attr in ((entry.id, "id"), (entry.name, "name")):
        if key:
            for i in items:
                if getattr(i, attr) and _norm(getattr(i, attr)) == _norm(key):
                    return i
    if entry.package:
        for i in items:
            if i.package and _norm(i.package) == _norm(entry.package):
                return i
    if entry.file_name:
        target = plugins.plugin_dir() / entry.file_name
        for i in items:
            if i.path and Path(i.path) == target:
                return i
    return None


def is_installed(entry: Entry, items=None) -> bool:
    if match(entry, items) is not None:
        return True
    return bool(entry.file_name) and (plugins.plugin_dir() / entry.file_name).exists()


def update_for(item: Installed, entries) -> Entry | None:
    """A compatible catalog entry with a newer version of ``item``, if any. A plugin-folder file
    whose content is the entry's (same ``sha256``) is up to date; one with no version at all is
    offered an entry only when its content differs from the entry's ``sha256``."""
    best = None
    current = _file_sha256(item.path) if item.path and Path(item.path).is_file() else ""
    for e in entries:
        if not e.version or not e.compatible or match(e, [item]) is None:
            continue
        if current and e.sha256 and current == e.sha256.lower():
            continue                          # this very file is installed
        if not item.version and not (current and e.sha256):
            continue                          # nothing to compare with: do not offer it forever
        if not item.version or plugins._version_tuple(e.version) > plugins._version_tuple(item.version):
            if best is None or plugins._version_tuple(e.version) > plugins._version_tuple(best.version):
                best = e
    return best


def _file_sha256(path) -> str:
    try:
        return sha256(Path(path).read_bytes())
    except OSError:
        return ""


def _sources_path() -> Path:
    from .settings import config_dir

    return config_dir() / "plugin-sources.json"


def _sources() -> dict:
    try:
        data = json.loads(_sources_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _record_source(path: Path, entry: Entry | None):
    """Remember which catalog entry a plugin-folder file came from (``<config dir>/plugin-sources.json``,
    keyed by the lower-cased file name), or forget it (``entry`` None)."""
    data = _sources()
    key = path.name.lower()
    if entry is None:
        if data.pop(key, None) is None:
            return
    else:
        data[key] = {"version": entry.version, "source": entry.source, "sha256": _file_sha256(path)}
    target = _sources_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8", newline="\n")


def installed_version(path) -> str:
    """The catalog version a plugin-folder file was installed as ("" when it was not installed
    from a catalog, or has been edited since)."""
    rec = _sources().get(Path(path).name.lower(), {})
    if rec.get("version") and rec.get("sha256") == _file_sha256(path):
        return str(rec["version"])
    return ""


def install_file(entry_or_source, timeout: float = 30.0) -> list[str]:
    """Copy a plugin ``.py`` (path or URL) into the plugin folder and load it; returns the
    names it registered. Replaces a file of the same name (an update)."""
    return plugins.load_plugin_file(fetch_file(entry_or_source, timeout))


def fetch_file(entry_or_source, timeout: float = 30.0) -> Path:
    """The download-and-copy half of :func:`install_file` (no import), safe to run off the GUI
    thread; returns the file in the plugin folder."""
    source = entry_or_source.source if isinstance(entry_or_source, Entry) else str(entry_or_source)
    expected = entry_or_source.sha256 if isinstance(entry_or_source, Entry) else ""
    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    name = source_name(source)
    if not name.endswith(".py"):
        raise ValueError(f"not a plugin file: {source} (a .py file or pip:<package>)")
    target = folder / name
    if "://" in source:
        text = _read(source, timeout)
        verify_sha256(text, expected, source)   # before anything is written or imported
        target.write_text(text, encoding="utf-8", newline="\n")
    else:
        src = Path(source).expanduser().resolve()
        if expected:
            verify_sha256(src.read_bytes(), expected, source)
        if src != target.resolve():
            shutil.copyfile(src, target)
    _record_source(target, entry_or_source if isinstance(entry_or_source, Entry) and entry_or_source.version else None)
    return target


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
    _record_source(src, None)
    return dest


ARCHIVE_SUFFIXES = (".zip", ".whl", ".tar.gz")   # a downloaded package that pip installs (a GitHub "Download ZIP" of a plugin repo)


def _uv() -> str | None:
    """The uv executable when this Python has no pip (an environment made by uv), else None."""
    return shutil.which("uv") if importlib.util.find_spec("pip") is None else None


# uv reads the OS certificate store (a company proxy's root certificate) when one of these is
# set. Older uv knows only UV_NATIVE_TLS, newer uv also UV_SYSTEM_CERTS; each ignores the other.
# The command-line flags differ the same way (--native-tls vs --system-certs) and an unknown flag
# is an error, so the environment is the form that works with every uv version.
UV_TLS_ENV = {"UV_NATIVE_TLS": "1", "UV_SYSTEM_CERTS": "1"}


def pip_command(action: str, requirement: str | list[str]) -> list[str]:
    """``python -m pip install|uninstall ...`` for the Python ewald runs in; ``"update"``
    installs with ``--upgrade``. ``requirement`` may be a list (several packages at once).
    An environment made by uv has no pip: then ``uv pip ... --python <this python>`` is used
    (with :data:`UV_TLS_ENV` from :func:`pip_env`, so a company proxy's root certificate in the
    OS store is trusted)."""
    reqs = [requirement] if isinstance(requirement, str) else list(requirement)
    flags = {"install": ["install"], "update": ["install", "--upgrade"], "uninstall": ["uninstall", "-y"]}
    verb, *rest = flags[action if action in flags else "uninstall"]
    if uv := _uv():
        if verb == "uninstall":
            rest = []                                    # uv pip uninstall has no -y
        return [uv, "pip", verb, "--python", sys.executable, *rest, *reqs]
    return [sys.executable, "-m", "pip", verb, *rest, *reqs]


def pip_env(requirement: str | list[str]) -> dict:
    """Environment variables for the :func:`pip_command` process: :data:`UV_TLS_ENV` when it
    is uv, and for a ``git+https://github.com/...`` requirement and a GitHub token, a git config
    entry that sends the token to github.com, so it never appears in the command or in pip's
    output."""
    env = dict(UV_TLS_ENV) if _uv() else {}
    return env | _github_env(requirement)


def _github_env(requirement: str | list[str]) -> dict:
    if not isinstance(requirement, str):
        return next((env for r in requirement if (env := _github_env(r))), {})
    url = requirement.split("@", 1)[-1].strip() if " @ " in requirement else requirement
    url = url.removeprefix("git+")
    if not url.startswith("https://github.com/"):
        return {}
    token = github_token()
    if not token:
        return {}
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    n = int(os.environ.get("GIT_CONFIG_COUNT", "0") or 0)
    return {"GIT_CONFIG_COUNT": str(n + 1), f"GIT_CONFIG_KEY_{n}": "http.https://github.com/.extraheader",
            f"GIT_CONFIG_VALUE_{n}": f"AUTHORIZATION: basic {basic}"}


# ---------------------------------------------------------------- restart
def _module_distributions() -> dict[str, list[str]]:
    """Top-level module name -> the distributions that install it (read fresh from disk).

    Reads each distribution's ``top_level.txt`` or ``RECORD`` and never looks at the files they
    list: :func:`importlib.metadata.packages_distributions` checks that every file of every
    installed package exists (Python 3.12+), tens of thousands of ``stat`` calls that took the
    Plugins side bar seconds, and much longer in a big environment on Windows."""
    importlib.invalidate_caches()
    out: dict[str, list[str]] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        for top in _top_level_names(dist) | {_norm(name).replace("-", "_")}:  # the name: editable installs
            if name not in out.setdefault(top, []):
                out[top].append(name)
    return out


def _top_level_names(dist) -> set[str]:
    """The importable top-level names a distribution installs, from its metadata files only."""
    text = dist.read_text("top_level.txt")
    if text:
        return {line.strip() for line in text.splitlines() if line.strip()}
    names = set()
    for line in (dist.read_text("RECORD") or "").splitlines():
        path = line.rsplit(",", 2)[0].strip().strip('"').replace("\\", "/")
        first = PurePosixPath(path).parts[0] if path else ""
        if not first or first in ("..", "__pycache__") or first.endswith((".dist-info", ".data", ".egg-info")):
            continue
        if first.endswith((".py", ".pyd", ".so")):
            names.add(first.partition(".")[0])     # a module, or an extension module.cpython-313-....so
        elif "." not in first:
            names.add(first)                       # a package folder
    return names


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return ""


def loaded_versions() -> dict[str, str]:
    """``{distribution: version}`` of the installed packages that have a module imported in
    this process. Take it before pip runs and pass it to :func:`restart_reason` afterwards."""
    top = {name.partition(".")[0] for name in list(sys.modules)}
    dists = {d for name, ds in _module_distributions().items() if name in top for d in ds}
    return {d: _version(d) for d in sorted(dists)}


def _imported(package: str) -> bool:
    """Whether a module of the installed distribution ``package`` is imported in this process."""
    want = _norm(package)
    top = {name.partition(".")[0] for name in list(sys.modules)}
    return any(_norm(d) == want for name, ds in _module_distributions().items() if name in top for d in ds)


def restart_reason(action: str, item: Installed | None = None, before: dict | None = None) -> str:
    """Why ewald should restart after ``action`` ("install", "update", "uninstall", "enable",
    "disable", "remove") on ``item``, or "" when the change already took effect.

    Python keeps a module it has imported, so a restart is needed when pip changed or removed a
    package whose code is already loaded (``before`` = :func:`loaded_versions` from before pip
    ran; a numpy upgrade a plugin pulled in counts too), when a package plugin that was imported
    earlier is switched back on (its module is not run again), and when a plugin folder (not a
    single file) is updated. Installing something new, and every change to a single-file
    plugin, takes effect at once."""
    if before is not None and action in ("install", "update", "uninstall"):
        changed = []
        for dist, old in before.items():
            new = _version(dist)
            if new != old:
                changed.append(f"{dist} {old} -> {new}" if new else f"{dist} {old} (removed)")
        if changed:
            return ("pip changed packages that are already loaded, and the running ewald keeps "
                    "their old code: " + ", ".join(changed))
    if item is None:
        return ""
    if action == "enable" and item.package and _imported(item.package):
        return f"{item.package} was loaded before it was switched off; its contributions come back after a restart"
    if action == "update" and item.path and Path(item.path).is_dir():
        return f"{item.name} is a folder plugin whose modules are already loaded"
    return ""


def restart_command(paths=()) -> list[str]:
    """The command line that starts ewald's desktop app again, opening ``paths``. A frozen app
    (Windows installer, EWALD.app) is its own executable; otherwise this Python runs
    ``-m ewald gui`` (the same interpreter, so the same environment)."""
    paths = [os.fspath(p) for p in paths]
    if getattr(sys, "frozen", False):
        return [sys.executable, *paths]
    return [sys.executable, "-m", "ewald", "gui", *paths]
