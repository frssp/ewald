"""The public Python API as text, one name per line (``ewald api --signatures``).

What a notebook, a script or a plugin may rely on is listed here with its call signature:
the names of ``ewald.__all__``, the functions ``llms.txt`` documents, every modifier of
``ewald.modifiers.__all__``, the plugin surface ``ewald.plugin_api``, the keywords of
``render()``, the settings of the 3D view and the options of ``ewald.display``. Two snapshots
of this text live in ``tests/data/api/`` and ``tests/test_api_stability.py`` reads them:

- ``current.txt`` is the API of the checked-out code. A change to a signature makes the test
  fail until the file is regenerated, so every change of the public API shows up in the diff
  of the pull request that makes it.
- ``released.txt`` is the API of the last release. What a release promised and the code no
  longer keeps (a name or parameter that is gone, a default that changed, a parameter that
  became required) has to be named in ``CHANGELOG.md`` above that release.

Signatures are written without annotations and with plain values only, so the text is the same
on every Python version and platform. docs/developer/api-stability.md has the rules.
"""
from __future__ import annotations

import ast
import dataclasses
import enum
import inspect
import numbers
import re

HEADER = "# The public API of ewald {version}, one name per line (python -m ewald api --signatures)."

_SAFE = (type(None), bool, int, float, complex, str, bytes)


# ---------------------------------------------------------------- one value, one signature
def _value(value, depth: int = 0) -> str:
    """A default as text that does not depend on the Python version, numpy version or run:
    plain values as their repr, anything else as ``<TypeName>``."""
    if isinstance(value, enum.Enum):
        return f"<{type(value).__name__}.{value.name}>"
    if isinstance(value, bool) or value is None or isinstance(value, (str, bytes)):
        return repr(value)
    for abstract, kind in ((numbers.Integral, int), (numbers.Real, float), (numbers.Complex, complex)):
        if isinstance(value, abstract):   # numpy scalars too: they print differently in numpy 1 and 2
            return repr(kind(value))
    if depth < 3 and type(value) in (tuple, list, frozenset, set, dict):
        if isinstance(value, dict):
            return "{" + ", ".join(f"{_value(k, depth + 1)}: {_value(v, depth + 1)}" for k, v in value.items()) + "}"
        items = [_value(v, depth + 1) for v in value]
        if isinstance(value, (set, frozenset)):
            body = "{" + ", ".join(sorted(items)) + "}" if items else "set()"
            return body if isinstance(value, set) else f"frozenset({body})" if items else "frozenset()"
        if isinstance(value, tuple):
            return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
        return "[" + ", ".join(items) + "]"
    if value is dataclasses.MISSING or type(value).__name__ == "_HAS_DEFAULT_FACTORY_CLASS":
        return "<factory>"
    return f"<{type(value).__name__}>"


def _parameters(params) -> str:
    out, star = [], False
    params = list(params)
    for i, p in enumerate(params):
        if p.kind is p.VAR_POSITIONAL:
            out.append(f"*{p.name}")
            star = True
            continue
        if p.kind is p.VAR_KEYWORD:
            out.append(f"**{p.name}")
            continue
        if p.kind is p.KEYWORD_ONLY and not star:
            out.append("*")
            star = True
        out.append(p.name if p.default is p.empty else f"{p.name}={_value(p.default)}")
        if p.kind is p.POSITIONAL_ONLY and (i + 1 == len(params) or params[i + 1].kind is not p.POSITIONAL_ONLY):
            out.append("/")
    return "(" + ", ".join(out) + ")"


def _own(cls: type, name: str) -> bool:
    """Whether a class of ewald in the MRO of ``cls`` defines ``name`` (not object, dict, Exception)."""
    return any(name in vars(base) and (base.__module__ or "").split(".")[0] == "ewald" for base in cls.__mro__)


def signature(obj) -> str:
    """``(a, b=1, *, c=None)`` for a function or a class's constructor; ``(...)`` when Python
    cannot tell (a compiled function)."""
    try:
        if isinstance(obj, type):
            if not (_own(obj, "__init__") or _own(obj, "__new__") or dataclasses.is_dataclass(obj)):
                return "()"
            init = obj.__init__ if _own(obj, "__init__") or dataclasses.is_dataclass(obj) else obj.__new__
            return _parameters(list(inspect.signature(init).parameters.values())[1:])
        return _parameters(inspect.signature(obj).parameters.values())
    except (TypeError, ValueError):
        return "(...)"


# ---------------------------------------------------------------- one name, its lines
def _members(cls: type) -> dict:
    """Public members of ``cls`` that ewald defines: methods, properties and plain class values."""
    found = {}
    for base in reversed(cls.__mro__):
        if (base.__module__ or "").split(".")[0] != "ewald":
            continue
        for name, raw in vars(base).items():
            if not name.startswith("_"):
                found[name] = raw
    return found


def _describe(prefix: str, obj, lines: dict, seen: dict, members: bool = True):
    """The lines of one public name; a class seen before is written as an alias of its first name."""
    if isinstance(obj, type):
        if obj in seen and seen[obj] != prefix:
            lines[prefix] = f"{prefix} = {seen[obj]}"
            return
        seen.setdefault(obj, prefix)
        if issubclass(obj, enum.Enum):
            lines[prefix] = prefix
            for member in obj:
                lines[f"{prefix}.{member.name}"] = f"{prefix}.{member.name} = {_value(member.value)}"
            return
        lines[prefix] = prefix + signature(obj)
        if not members:
            return
        fields = {f.name for f in dataclasses.fields(obj)} if dataclasses.is_dataclass(obj) else set()
        for name, raw in sorted(_members(obj).items()):
            if name in fields:   # a dataclass field: the constructor line has its name and default
                continue
            key = f"{prefix}.{name}"
            if isinstance(raw, (staticmethod, classmethod)):
                params = list(inspect.signature(raw.__func__).parameters.values())
                lines[key] = key + _parameters(params[1:] if isinstance(raw, classmethod) else params)
            elif inspect.isfunction(raw):
                lines[key] = key + _parameters(list(inspect.signature(raw).parameters.values())[1:])
            elif isinstance(raw, property) or inspect.isdatadescriptor(raw):
                lines[key] = key
            elif isinstance(raw, _SAFE) or type(raw) in (tuple, frozenset):
                lines[key] = f"{key} = {_value(raw)}"
            elif not isinstance(raw, type):
                lines[key] = key
    elif callable(obj):
        lines[prefix] = prefix + signature(obj)
    elif isinstance(obj, _SAFE) or type(obj) in (tuple, frozenset):
        lines[prefix] = f"{prefix} = {_value(obj)}"
    else:
        lines[prefix] = prefix


def _instance(prefix: str, obj, lines: dict):
    """Public attributes an instance gets in its constructor (``pipe.modifiers``, ``data.cell``)."""
    for name in sorted(vars(obj)):
        if not name.startswith("_"):
            lines.setdefault(f"{prefix}.{name}", f"{prefix}.{name}")


# ---------------------------------------------------------------- the whole surface
def public_api() -> dict:
    """``{name: line}`` for every public name, in the order of the snapshot file."""
    import numpy as np

    import ewald

    from . import display, modifiers, plugin_api, verbs, view3d
    from .llm.reference import FUNCTIONS, _function_modules
    from .render import render_keywords

    lines: dict = {}
    seen: dict = {}
    for name in sorted(ewald.__all__):
        _describe(f"ewald.{name}", getattr(ewald, name), lines, seen)
    _describe("ewald.verbs.Table", verbs.Table, lines, seen)
    data = ewald.DataCollection(np.zeros((1, 3)), ["Cu"], ewald.Cell(np.eye(3)))
    pipe = ewald.Pipeline(ewald.StaticSource(data))
    for prefix, obj in (("ewald.Pipeline", pipe), ("ewald.DataCollection", data), ("ewald.Cell", data.cell),
                        ("ewald.verbs.Table", verbs.Table("t", {"x": [0.0]}))):
        _instance(prefix, obj, lines)
    mods = _function_modules()
    for module, names in FUNCTIONS:
        for name in names:
            _describe(f"{module}.{name}", getattr(mods[module], name), lines, seen)
    for name in sorted(modifiers.__all__):
        # a modifier's methods are the Modifier contract (listed once, above); its constructor is its API
        _describe(f"ewald.modifiers.{name}", getattr(modifiers, name), lines, seen, members=False)
    for name in sorted(plugin_api.__all__):
        _describe(f"ewald.plugin_api.{name}", getattr(plugin_api, name), lines, seen)
    for name in sorted(render_keywords()):
        lines[f"render({name}=)"] = f"render({name}=)"
    for name, setting in sorted(view3d.SETTINGS.items()):
        lines[f"View.{name}"] = f"View.{name} = {_value(setting.default)}"
    for name in sorted(display._BASE):
        lines[f"ewald.display.options[{name!r}]"] = f"ewald.display.options[{name!r}]"
    return dict(sorted(lines.items(), key=lambda item: _order(item[0])))


_SECTIONS = ("ewald.modifiers.", "ewald.plugin_api.", "render(", "View.", "ewald.display.")


def _order(key: str) -> tuple:
    section = next((i + 1 for i, s in enumerate(_SECTIONS) if key.startswith(s)), 0)
    return section, key


def render() -> str:
    """The snapshot text: a header line with the version, then one line per name."""
    from . import __version__

    return "\n".join([HEADER.format(version=__version__), *public_api().values()]) + "\n"


# ---------------------------------------------------------------- comparing two snapshots
def parse(text: str) -> dict:
    """``{name: line}`` of a snapshot text (comment and empty lines skipped)."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out[_key(line)] = line
    return out


def version_of(text: str) -> str | None:
    """The ewald version a snapshot text was made with (from its header line)."""
    m = re.match(r"# The public API of ewald (\S+),", text)
    return m.group(1) if m else None


def _key(line: str) -> str:
    if line.startswith("render("):
        return line
    cut = min((i for i in (line.find("("), line.find(" = ")) if i >= 0), default=len(line))
    return line[:cut]


def _arguments(line: str):
    """The ``ast.arguments`` of a callable's line, or None for a line without a signature."""
    start = line.find("(")
    if start < 0 or line.startswith("render(") or line.endswith("(...)"):
        return None
    source = re.sub(r"<([A-Za-z_][\w.]*)>", lambda m: "_" + m.group(1).replace(".", "_") + "_", line[start:])
    try:
        return ast.parse(f"def f{source}: pass").body[0].args
    except SyntaxError:
        return None


def _params(args) -> dict:
    """``{name: (kind, position or None, default text or None)}`` of parsed arguments."""
    out = {}
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for i, (a, d) in enumerate(zip(positional, defaults)):
        kind = "positional-only" if i < len(args.posonlyargs) else "positional"
        out[a.arg] = (kind, i, None if d is None else ast.unparse(d))
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        out[a.arg] = ("keyword-only", None, None if d is None else ast.unparse(d))
    if args.vararg:
        out["*" + args.vararg.arg] = ("var-positional", None, None)
    if args.kwarg:
        out["**" + args.kwarg.arg] = ("var-keyword", None, None)
    return out


def breaking(old: str, new: str) -> list:
    """What code written for the API of text ``old`` can no longer rely on in ``new``:
    ``[(name, parameter or None, what happened)]``. Additions (a new name, a new parameter with
    a default at the end or keyword-only) are not listed."""
    before, after = parse(old), parse(new)
    out = []
    for name, line in before.items():
        if name not in after:
            out.append((name, None, "removed"))
            continue
        if after[name] == line:
            continue
        a, b = _arguments(line), _arguments(after[name])
        if a is None or b is None:
            out.append((name, None, f"changed from `{line}` to `{after[name]}`"))
            continue
        pa, pb = _params(a), _params(b)
        for p, (kind, pos, default) in pa.items():
            if p not in pb:
                if p.startswith("*") or b.kwarg is None:
                    out.append((name, p.lstrip("*"), f"parameter `{p}` removed"))
                elif kind != "keyword-only":
                    out.append((name, p, f"parameter `{p}` is keyword-only now (taken by `**{b.kwarg.arg}`)"))
                continue
            nkind, npos, ndefault = pb[p]
            if default != ndefault:
                what = (f"parameter `{p}` is required now (was {default})" if ndefault is None else
                        f"parameter `{p}` got the default {ndefault} (was required)" if default is None else
                        f"default of `{p}` changed from {default} to {ndefault}")
                if default is not None or ndefault is None:   # gaining a default breaks nothing
                    out.append((name, p, what))
            if kind != nkind and not (kind == "positional-only" and nkind == "positional"):
                out.append((name, p, f"parameter `{p}` is {nkind} now (was {kind})"))
        # positional order: the parameters both have, in the old and in the new order
        kept = [p for p, v in pa.items() if v[1] is not None and p in pb and pb[p][1] is not None]
        if kept != sorted(kept, key=lambda p: pb[p][1]):
            out += [(name, p, f"parameter `{p}` moved from position {pa[p][1] + 1} to {pb[p][1] + 1}")
                    for p in kept if pa[p][1] != pb[p][1]]
        else:
            last = max((pb[p][1] for p in kept if pb[p][1] != pa[p][1]), default=-1)
            out += [(name, p, f"new parameter `{p}` at position {v[1] + 1} moves the positional parameters after it")
                    for p, v in pb.items() if p not in pa and v[1] is not None and v[1] < last]
        for p, (nkind, npos, ndefault) in pb.items():
            if p not in pa and ndefault is None and not p.startswith("*"):
                out.append((name, p, f"new required parameter `{p}`"))
    return out


def unrecorded(old: str, new: str, changelog: str) -> list:
    """The :func:`breaking` changes that ``changelog`` (the part of CHANGELOG.md above the release
    of ``old``) does not name: an entry counts when it has the name's last part (``msd``,
    ``CoordinationAnalysis``, ``frames``) and, for a parameter, the parameter as words of their own."""
    def named(word: str) -> bool:
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])", changelog) is not None

    out = []
    for name, parameter, what in breaking(old, new):
        option = re.fullmatch(r"ewald\.display\.options\['(.+)'\]", name)
        words = ["options", option.group(1)] if option else [re.split(r"[.(]", name.rstrip("=)"))[-1], parameter]
        if not all(named(w) for w in words if w):
            out.append((name, parameter, what))
    return out


def changelog_above(changelog: str, version: str | None) -> str:
    """CHANGELOG.md down to (not including) the section of release ``version``; all of it when
    that release has no section."""
    m = re.search(rf"^## \[{re.escape(version)}\]", changelog, flags=re.M) if version else None
    return changelog[:m.start()] if m else changelog
