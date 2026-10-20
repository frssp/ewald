"""Plugin manifests and versions: :class:`Plugin` (one installed plugin and its manifest),
reading ``ewald-plugin.toml`` and a file's ``__ewald__`` without importing it, and the version
checks. Stateless; :mod:`ewald.plugins` re-exports everything here."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Plugin:
    """One installed plugin: a pip distribution with ewald entry points, or a file or folder in
    :func:`plugin_dir`. Its metadata comes from its manifest (``ewald-plugin.toml`` in a package,
    ``__ewald__ = {...}`` in a plugin file), read without importing the plugin's code."""

    id: str
    name: str = ""               # display name
    version: str = ""
    description: str = ""
    publisher: str = ""
    requires_ewald: str = ""     # e.g. ">=0.2, <1"
    plugin_api: object = None    # the ewald.plugin_api version it was written for (None: not declared)
    kind: str = "folder"         # "built-in" (ships with ewald), "package" (pip), "folder" (plugin folder)
    #                              or "forge" (written by Modifier Forge)
    origin: str = ""             # distribution name or file path
    enabled: bool = True
    compatible: bool = True
    loaded: bool = False
    contributions: dict = field(default_factory=dict)   # registry ("readers", "modifiers", ...) -> names
    errors: list = field(default_factory=list)
    manifest: dict = field(default_factory=dict, repr=False)

    @property
    def state(self) -> str:
        """"disabled", "incompatible", "failed", "loaded" or "not loaded"."""
        if not self.enabled:
            return "disabled"
        if not self.compatible:
            return "incompatible"
        if self.errors:
            return "failed"
        return "loaded" if self.loaded else "not loaded"

    def apply_manifest(self, manifest: dict):
        self.manifest = dict(manifest)

        def get(*keys):
            return next((manifest[k] for k in keys if manifest.get(k) not in (None, "")), "")

        self.id = str(get("id") or self.id)
        self.name = str(get("display-name", "display_name", "name") or self.name or self.id)
        self.version = str(get("version") or self.version)
        self.description = str(get("description") or self.description)
        self.publisher = str(get("publisher") or self.publisher)
        self.requires_ewald = str(get("requires-ewald", "requires_ewald") or self.requires_ewald)
        api = get("plugin-api", "plugin_api")
        self.plugin_api = api if api != "" else self.plugin_api
        if manifest.get("source") == "forge":
            self.kind = "forge"

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name or self.id, "version": self.version,
                "description": self.description, "publisher": self.publisher,
                "requires_ewald": self.requires_ewald, "plugin_api": self.plugin_api, "kind": self.kind, "origin": self.origin,
                "state": self.state, "enabled": self.enabled, "compatible": self.compatible,
                "contributions": {k: list(v) for k, v in self.contributions.items()},
                "errors": list(self.errors)}


def _read_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    return tomllib.loads(path.read_text(encoding="utf-8"))


def file_manifest(path) -> dict:
    """The ``__ewald__ = {...}`` dict at the top level of a plugin file, read with ``ast``
    (the file is not run); ``{}`` when it has none."""
    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else \
            [node.target] if isinstance(node, ast.AnnAssign) and node.value is not None else []
        if any(isinstance(t, ast.Name) and t.id == "__ewald__" for t in targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise TypeError(f"__ewald__ in {path.name} must be a dict of literals")
            return value
    return {}


def _version_tuple(version: str) -> tuple:
    out = []
    for part in str(version).split("."):
        m = re.match(r"\d+", part)
        if not m:
            break
        out.append(int(m.group()))
    return tuple(out)


def version_satisfies(version: str, requirement: str) -> bool:
    """Whether ``version`` meets ``requirement``: comma-separated clauses such as ``">=0.2, <1"``
    with ``>= > <= < == != ~=`` on numeric release versions (pre/dev suffixes are ignored)."""
    v = _version_tuple(version)
    for clause in str(requirement or "").split(","):
        clause = clause.strip()
        if not clause:
            continue
        m = re.fullmatch(r"(~=|==|!=|>=|<=|>|<)?\s*(\d+(?:\.\d+)*)", clause)
        if not m:
            raise ValueError(f"cannot read the version requirement {clause!r} (use e.g. '>=0.2, <1')")
        op, t = m.group(1) or "==", _version_tuple(m.group(2))
        n = max(len(v), len(t))
        a, b = v + (0,) * (n - len(v)), t + (0,) * (n - len(t))
        ok = {"==": a == b, "!=": a != b, ">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b,
              "~=": a >= b and v[:len(t) - 1] == t[:len(t) - 1]}[op]
        if not ok:
            return False
    return True


def ewald_version() -> str:
    from . import __version__

    return __version__
