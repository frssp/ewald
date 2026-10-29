"""OVITO-style per-particle expressions evaluated with numpy.

Property names are used as in OVITO: ``Position.Z``, ``ParticleType``,
``ParticleIdentifier``, ``ParticleIndex`` (0-based row), ``Velocity.X``, ``Coordination``; globals ``N``,
``Frame``, ``Timestep``, ``CellSize.X``. Combine conditions with numpy
operators: ``(Position.Z > 10) & (ParticleType == "Cu")``; ``^`` is a power, as in OVITO
(``Force.X^2``); ``cond ? a : b`` is OVITO's conditional (``np.where``); a name with spaces may be
written in backticks (```Displacement Magnitude```); numpy is ``np``.
"""
from __future__ import annotations

import ast
import re

import numpy as np

_COMPONENTS = {"Color": "RGB"}


class _TypeNames(np.ndarray):
    """``ParticleType`` (type names) in an expression. OVITO's ParticleType is a number, so
    ``ParticleType == 2`` compares a LAMMPS type number with names that are numbers ("2"), and
    is an error, not silently no match, when the types have element names."""

    def _compare(self, other, op):
        plain = self.view(np.ndarray)
        if isinstance(other, (int, float, np.integer, np.floating)) and not isinstance(other, (bool, np.bool_)):
            names = np.unique(plain)
            if all(str(n).lstrip("-").isdigit() for n in names):
                return op(plain.astype(np.int64), other)
            raise ValueError(f"ParticleType holds type names here ({', '.join(map(str, names[:8]))}), not "
                             f"OVITO's type numbers: compare with a name, e.g. ParticleType == {str(names[-1])!r}")
        return op(plain, other.view(np.ndarray) if isinstance(other, _TypeNames) else other)

    def __eq__(self, other):
        return self._compare(other, np.equal)

    def __ne__(self, other):
        return self._compare(other, np.not_equal)

    __hash__ = None


def _key(name: str) -> str:
    """A property's name in expressions: spaces and other characters a Python name cannot hold
    dropped ("Particle Type" -> ParticleType, "SqDisp (Average)" -> SqDispAverage)."""
    return re.sub(r"[^\w.]", "", name)


def namespace(data, frame=None) -> dict:
    ns: dict[str, object] = {}
    for name, arr in data.particles.items():
        key = _key(name)
        ns[key] = arr.view(_TypeNames) if key == "ParticleType" and arr.dtype.kind in "US" else arr
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


def _matching(s: str, i: int) -> int:
    """Index of the bracket closing the one at ``s[i]`` (quotes skipped), or -1."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack, k = [], i
    while k < len(s):
        c = s[k]
        if c in "'\"":
            end = s.find(c, k + 1)
            k = len(s) if end < 0 else end
        elif c in pairs:
            stack.append(pairs[c])
        elif stack and c == stack[-1]:
            stack.pop()
            if not stack:
                return k
        k += 1
    return -1


def _conditional(s: str) -> str:
    """OVITO's C-style ``cond ? a : b`` (right-associative, lowest precedence, also inside
    brackets) as ``np.where(cond, a, b)``."""
    if "?" not in s:
        return s
    out, top, k = [], [], 0      # top: positions of ? and : outside brackets, in ``out`` text
    while k < len(s):
        c = s[k]
        if c in "([{":
            end = _matching(s, k)
            if end < 0:
                raise SyntaxError(f"unbalanced {c!r}")
            out.append(c + _conditional(s[k + 1:end]) + s[end])
            k = end + 1
            continue
        if c in "'\"":
            end = s.find(c, k + 1)
            end = len(s) - 1 if end < 0 else end
            out.append(s[k:end + 1])
            k = end + 1
            continue
        if c in "?:":
            top.append((len(out), c))
        out.append(c)
        k += 1
    q = next((i for i, (_, c) in enumerate(top) if c == "?"), None)
    if q is None:
        return "".join(out)
    depth = 0
    for j in range(q + 1, len(top)):
        depth += 1 if top[j][1] == "?" else -1
        if depth < 0:
            qi, ci = top[q][0], top[j][0]
            cond, a, b = "".join(out[:qi]), "".join(out[qi + 1:ci]), "".join(out[ci + 1:])
            return f"np.where(({cond.strip()}), ({_conditional(a).strip()}), ({_conditional(b).strip()}))"
    raise SyntaxError("'?' without its ':' (OVITO's conditional is 'cond ? a : b')")


_BACKTICKED = re.compile(r"`([^`]+)`")
_SPACED_NAME = re.compile(r"[A-Za-z][\w.]*( +[\w.]+)+")


def evaluate(expr: str, data, frame=None):
    ns = namespace(data, frame)
    if expr.strip() in data.particles and _key(expr.strip()) in ns:   # a property as named: "SqDisp (Average)"
        return ns[_key(expr.strip())]
    if "`" in expr:     # `Displacement Magnitude`: a name as it is written, spaces and all
        expr = _BACKTICKED.sub(lambda m: _key(m.group(1)), expr)
    if _SPACED_NAME.fullmatch(expr.strip()):
        try:
            ast.parse(expr.strip(), "<expression>", "eval")
        except SyntaxError:
            # a property written as it is named ("Particle Identifier"); expressions drop the spaces
            key = _key(expr.strip())
            if key in ns:
                return ns[key]
            raise NameError(f"no property {expr.strip()!r} in this frame. "
                            f"Available: {', '.join(sorted(ns))}") from None
    # properties written as they are named inside a longer expression: "(Net Displacement > 6)",
    # "SqDisp (Average) > 20" (models write them so); longest first, whole names only
    spelled = sorted((n for n in data.particles.keys() if _key(n) != n and _key(n) in ns), key=len, reverse=True)
    for name in spelled:
        expr = re.sub(r"(?<![\w.])" + re.escape(name) + r"(?![\w(])", _key(name), expr)
    dotted = sorted((k for k in ns if "." in k), key=len, reverse=True)
    env = {"np": np, "abs": np.abs, "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
           "sin": np.sin, "cos": np.cos, "min": np.minimum, "max": np.maximum, "pi": np.pi}
    code = expr.replace("^", "**")   # OVITO's power operator ('Force.X^2'); never XOR here
    try:
        code = _conditional(code)
    except SyntaxError as exc:
        raise SyntaxError(f"invalid expression {expr!r}: {exc}") from None
    for i, k in enumerate(dotted):
        code = re.sub(r"(?<![\w.])" + re.escape(k) + r"(?![\w])", f"__v{i}", code)
        env[f"__v{i}"] = ns[k]
    env.update({k: v for k, v in ns.items() if "." not in k})
    try:
        _check(code, expr)
        out = eval(compile(code, "<expression>", "eval"), {"__builtins__": _BUILTINS}, env)
        return out.view(np.ndarray) if isinstance(out, _TypeNames) else out
    except ValueError as exc:
        if "truth value" in str(exc):
            raise ValueError(f"{expr!r}: use '&', '|', '~' with parentheses instead of "
                             "'and', 'or', 'not' for per-particle conditions") from exc
        raise
    except NameError as exc:
        avail = ", ".join(sorted(ns))
        raise NameError(f"{exc} in {expr!r}. Available: {avail}") from exc
