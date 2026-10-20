"""OVITO-style per-particle expressions evaluated with numpy.

Property names are used as in OVITO: ``Position.Z``, ``ParticleType``,
``ParticleIdentifier``, ``ParticleIndex`` (0-based row), ``Velocity.X``, ``Coordination``; globals ``N``,
``Frame``, ``Timestep``, ``CellSize.X``. Combine conditions with numpy
operators: ``(Position.Z > 10) & (ParticleType == "Cu")``; ``^`` is a power, as in OVITO
(``Force.X^2``); numpy is ``np``.
"""
from __future__ import annotations

import ast
import re

import numpy as np

_COMPONENTS = {"Color": "RGB"}


def namespace(data, frame=None) -> dict:
    ns: dict[str, object] = {}
    for name, arr in data.particles.items():
        key = name.replace(" ", "")
        ns[key] = arr
        if arr.ndim == 2 and arr.shape[1] in (2, 3):
            letters = _COMPONENTS.get(name, "XYZ")
            for k in range(arr.shape[1]):
                ns[f"{key}.{letters[k]}"] = arr[:, k]
    ns.setdefault("ParticleIndex", np.arange(data.count))
    m = data.cell.matrix
    ns.update({"N": data.count, "CellSize.X": m[0, 0], "CellSize.Y": m[1, 1], "CellSize.Z": m[2, 2],
               "CellOrigin.X": data.cell.origin[0], "CellOrigin.Y": data.cell.origin[1],
               "CellOrigin.Z": data.cell.origin[2]})
    for k, v in data.attributes.items():
        if np.isscalar(v):
            ns.setdefault(k.replace(" ", ""), v)
    if frame is not None:
        ns["Frame"] = frame
    return ns


# numpy names an expression may use as ``np.<name>``: every ufunc plus array functions that
# take and return arrays. Expressions come from session files, plots and Claude's tools,
# so nothing that reaches files, modules or Python objects (np.load, np.lib, np.testing) is allowed.
_NP_NAMES = frozenset(n for n in dir(np) if not n.startswith("_") and isinstance(getattr(np, n), np.ufunc)) | {
    "isin", "where", "clip", "any", "all", "sum", "mean", "std", "var", "median", "min", "max", "amin", "amax",
    "nanmin", "nanmax", "nanmean", "nansum", "round", "around", "arange", "linspace", "zeros", "ones", "full",
    "zeros_like", "ones_like", "full_like", "unique", "cumsum", "diff", "sort", "argsort", "argmin", "argmax",
    "nonzero", "count_nonzero", "asarray", "array", "dot", "cross", "stack", "column_stack", "concatenate",
    "select", "percentile", "quantile", "histogram", "digitize", "bincount", "searchsorted", "interp",
    "float64", "float32", "int64", "int32", "bool_", "nan", "inf", "pi", "e", "newaxis", "linalg"}
# attributes allowed on values (arrays, numbers): reductions and shape
_VALUE_ATTRS = frozenset({
    "T", "real", "imag", "shape", "size", "ndim", "sum", "mean", "std", "var", "min", "max", "any", "all",
    "astype", "round", "clip", "cumsum", "argmin", "argmax", "argsort", "nonzero", "reshape", "ravel"})
_NODES = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Attribute, ast.Call, ast.keyword,
          ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Subscript, ast.Slice,
          ast.Tuple, ast.List, ast.operator, ast.unaryop, ast.boolop, ast.cmpop)
_SUBSTITUTED = re.compile(r"__v\d+$")


def _check(code: str, expr: str):
    """Reject anything but arithmetic, comparisons, indexing and calls of allowed numpy
    functions: ``eval`` without builtins is no sandbox once a module or a dunder attribute
    is reachable (``np.testing...os.system``, ``().__class__``)."""
    try:
        tree = ast.parse(code, "<expression>", "eval")
    except SyntaxError as exc:
        raise SyntaxError(f"invalid expression {expr!r}: {exc.msg}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _NODES):
            raise ValueError(f"{expr!r}: {type(node).__name__} is not allowed in an expression")
        if isinstance(node, ast.Name) and node.id.startswith("_") and not _SUBSTITUTED.match(node.id):
            raise ValueError(f"{expr!r}: name {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute):
            owner = node.value
            if isinstance(owner, ast.Name) and owner.id == "np":
                ok = node.attr in _NP_NAMES
            elif isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name) \
                    and owner.value.id == "np" and owner.attr == "linalg":
                ok = node.attr == "norm"
            else:
                ok = node.attr in _VALUE_ATTRS
            if not ok and isinstance(owner, ast.Name) and owner.id != "np" and node.attr[:1].isupper():
                raise NameError(f"name '{owner.id}.{node.attr}' is not defined")  # e.g. Velocity.X without velocities
            if not ok:
                raise ValueError(f"{expr!r}: attribute {node.attr!r} is not allowed in an expression")


def _numpy_import(name, *args, **kw):
    # array methods (Position.Z.mean()) import numpy internals through the calling frame's
    # builtins; expressions themselves cannot name __import__ (see _check)
    if name != "numpy" and not name.startswith("numpy."):
        raise ImportError(f"import of {name!r} in an expression")
    return __import__(name, *args, **kw)


_BUILTINS = {"__import__": _numpy_import}


_SPACED_NAME = re.compile(r"[A-Za-z][\w.]*( +[\w.]+)+")


def evaluate(expr: str, data, frame=None):
    ns = namespace(data, frame)
    if _SPACED_NAME.fullmatch(expr.strip()):
        try:
            ast.parse(expr.strip(), "<expression>", "eval")
        except SyntaxError:
            # a property written as it is named ("Particle Identifier"); expressions drop the spaces
            key = expr.strip().replace(" ", "")
            if key in ns:
                return ns[key]
            raise NameError(f"no property {expr.strip()!r} in this frame. "
                            f"Available: {', '.join(sorted(ns))}") from None
    dotted = sorted((k for k in ns if "." in k), key=len, reverse=True)
    env = {"np": np, "abs": np.abs, "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
           "sin": np.sin, "cos": np.cos, "min": np.minimum, "max": np.maximum, "pi": np.pi}
    code = expr.replace("^", "**")   # OVITO's power operator ('Force.X^2'); never XOR here
    for i, k in enumerate(dotted):
        code = re.sub(r"(?<![\w.])" + re.escape(k) + r"(?![\w])", f"__v{i}", code)
        env[f"__v{i}"] = ns[k]
    env.update({k: v for k, v in ns.items() if "." not in k})
    try:
        _check(code, expr)
        return eval(compile(code, "<expression>", "eval"), {"__builtins__": _BUILTINS}, env)
    except ValueError as exc:
        if "truth value" in str(exc):
            raise ValueError(f"{expr!r}: use '&', '|', '~' with parentheses instead of "
                             "'and', 'or', 'not' for per-particle conditions") from exc
        raise
    except NameError as exc:
        avail = ", ".join(sorted(ns))
        raise NameError(f"{exc} in {expr!r}. Available: {avail}") from exc
