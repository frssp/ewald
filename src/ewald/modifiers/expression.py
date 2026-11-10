"""OVITO-style per-particle expressions evaluated with numpy.

Property names are used as in OVITO: ``Position.Z``, ``ParticleType``,
``ParticleIdentifier``, ``ParticleIndex`` (0-based row), ``Velocity.X``, ``Coordination``; globals ``N``,
``Frame``, ``Timestep``, ``CellSize.X``. Combine conditions with numpy
operators: ``(Position.Z > 10) & (ParticleType == "Cu")``, or with OVITO's ``&&`` and ``||``, which
need no parentheses (``Position.Z > 10 && ParticleType == "Cu"``); ``^`` is a power, as in OVITO
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


_OPERATORS = ("&&", "||", "==", "!=", "<=", ">=", "<<", ">>", "**", "//")   # read as one token
_COMPARISONS = frozenset({"<", ">", "<=", ">=", "==", "!="})
_PYTHON_LOGIC = re.compile(r"\b(and|or|if|else|lambda)\b")   # bind looser than && and || would: never regrouped


def _plain(text: str, inner) -> str:
    """A token of :func:`_tokens` as text of its own bracket level: a blank for a group or a string."""
    return " " if inner is not None or text[0] in "'\"" else text


def _tokens(s: str):
    """``s`` at its own bracket level, as ``(text, inner)``: a bracket group (``text`` is its two
    brackets, ``inner`` what is between them), else with ``inner`` None a quoted string taken
    whole, a two-character operator or one character."""
    k = 0
    while k < len(s):
        c = s[k]
        end = _matching(s, k) if c in "([{" else -1   # unbalanced: one character, Python's parser says so
        if end >= 0:
            yield c + s[end], s[k + 1:end]
            k = end + 1
            continue
        end = k + 1
        if c in "'\"":
            while end < len(s) and s[end] != c:
                end += 2 if s[end] == "\\" else 1
            end = min(end + 1, len(s))
        elif s[k:k + 2] in _OPERATORS:
            end = k + 2
        yield s[k:end], None
        k = end


def _fold(function: str, operands: list) -> str:
    """``operands`` (source text) folded left to right with ``function(a, b)``; one operand is
    returned as it is written."""
    if len(operands) == 1:
        return operands[0]
    if not all(o.strip() for o in operands):
        raise SyntaxError("'&&' and '||' need a condition on both sides")
    out = f"({operands[0].strip()})"
    for o in operands[1:]:
        out = f"{function}({out}, ({o.strip()}))"
    return out


def _logical(s: str) -> str:
    """OVITO's C-style ``a && b`` and ``a || b`` (both below comparisons, ``||`` lowest, so neither
    side needs parentheses; also inside brackets) as ``np.logical_and(a, b)`` / ``np.logical_or``:
    any nonzero value counts as true, as in OVITO, which ``&`` on numbers would not give."""
    if "&&" not in s and "||" not in s:
        return s
    args, ors, ands, piece = [], [], [], []   # arguments between commas; || operands; && operands; their text
    keyword, plain = [], []                   # a keyword argument's "name="; the argument without groups and strings

    def end(level: int):   # close the && operand being read (0), the || operand (1), the argument (2)
        ands.append("".join(piece))
        piece.clear()
        if level >= 1:
            ors.append(_fold("np.logical_and", ands))
            ands.clear()
        if level == 2:
            if ("&&" in plain or "||" in plain) and _PYTHON_LOGIC.search("".join(plain)):
                raise SyntaxError("'&&' or '||' beside 'and', 'or' or 'if': put each part in parentheses")
            args.append("".join(keyword) + _fold("np.logical_or", ors))
            ors.clear()
            keyword.clear()
            plain.clear()

    for text, inner in _tokens(s):
        plain.append(_plain(text, inner))
        if inner is not None:
            piece.append(text[0] + _logical(inner) + text[1])
        elif text in ("&&", "||", ","):
            end(("&&", "||", ",").index(text))
        elif text == "=" and not (ors or ands or keyword) and "".join(piece).strip().isidentifier():
            keyword[:] = [*piece, "="]   # np.where(..., where=a && b): the condition starts after the name
            piece.clear()
        else:
            piece.append(text)
    end(2)
    return ",".join(args)


def _double_equals(s: str) -> str:
    """``s`` with every single ``=`` written ``==`` (``==``, ``<=``, ``>=``, ``!=`` and quoted
    strings stay as they are)."""
    return "".join("==" if text == "=" else text if inner is None else text[0] + _double_equals(inner) + text[1]
                   for text, inner in _tokens(s))


def _parenthesized(s: str) -> str | None:
    """``s`` with parentheses around every comparison that stands beside ``&`` or ``|`` at its
    bracket level (``a > 1 | b > 2`` -> ``(a > 1) | (b > 2)``, what was meant: Python reads it as
    ``a > (1 | b) > 2``); None when there is no such comparison."""
    out, run, changed = [], [["", False]], False   # run: [text, holds a comparison] per operand of & and |
    plain = []                                     # the run without its groups and strings

    def end():   # a run of operands ends at a separator of lower precedence, or with ``s``
        nonlocal changed
        wrap = len(run) > 1 and any(compares for _, compares in run) and not _PYTHON_LOGIC.search("".join(plain))
        changed = changed or wrap
        plain.clear()
        for k, (text, compares) in enumerate(run):
            if k:
                text, op = text[1:], text[0]
                out.append(op)
            if wrap and compares:
                body = text.strip()
                at = text.index(body)
                text = f"{text[:at]}({body}){text[at + len(body):]}"
            out.append(text)
        run[:] = [["", False]]

    for text, inner in _tokens(s):
        plain.append(_plain(text, inner))
        if inner is not None:
            fixed = _parenthesized(inner)
            changed = changed or fixed is not None
            run[-1][0] += text[0] + (inner if fixed is None else fixed) + text[1]
        elif text in ("&", "|"):
            run.append([text, False])
        elif text in (",", "&&", "||", "?", ":"):
            end()
            out.append(text)
        else:
            run[-1][0] += text
            run[-1][1] = run[-1][1] or text in _COMPARISONS
    end()
    return "".join(out) if changed else None


def _suggest(fix, expr: str, data, frame) -> str | None:
    """``fix(expr)`` when it is another expression that evaluates on ``data``; "" when it only
    parses (it fails for another reason: say the rule, not this text); None when ``fix`` changes
    nothing or does not make it parse. A correction goes into an error message only after it ran."""
    try:
        fixed = fix(expr)
        if fixed is None or fixed == expr or fix(fixed) not in (None, fixed):   # stable, so trying it ends
            return None
        evaluate(fixed, data, frame)
    except SyntaxError:
        return None
    except Exception:  # noqa: BLE001 - an unknown name, a keyword argument's "=" that was doubled too, ...
        return ""
    return fixed


_BINDS_TIGHTER = ("'&' and '|' bind tighter than comparisons in Python (a > 1 | b > 2 is read as a > (1 | b) > 2), "
                  "so put each condition in parentheses")
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
        code = _logical(_conditional(code))
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
    except SyntaxError:
        fixed = _suggest(_double_equals, expr, data, frame)   # "Coordination = 12"
        if fixed is None:
            raise
        raise ValueError(f"{expr!r}: a single '=' does not compare; use '=='"
                         + (f": {fixed!r}" if fixed else ", as in Coordination == 12")) from None
    except TypeError as exc:
        # "Position.Z > 1 | Position.X > 1" is 1 | Position.X first: numpy's "ufunc 'bitwise_or'
        # not supported" says nothing about the parentheses that are missing
        fixed = _suggest(_parenthesized, expr, data, frame)
        if fixed:
            raise ValueError(f"{expr!r}: {_BINDS_TIGHTER}: {fixed!r}") from exc
        if "bitwise_" in str(exc):
            raise ValueError(f"{expr!r}: a side of '&' or '|' is neither a condition nor a whole number. "
                             f"{_BINDS_TIGHTER}, e.g. (Position.Z > 5) & (Coordination < 12)") from exc
        raise
    except ValueError as exc:
        if "truth value" in str(exc):
            fixed = _suggest(_parenthesized, expr, data, frame)   # whole numbers: a == 1 | b == 2 chains
            if fixed:
                raise ValueError(f"{expr!r}: {_BINDS_TIGHTER}: {fixed!r}") from exc
            raise ValueError(f"{expr!r}: use '&', '|', '~' with parentheses instead of "
                             "'and', 'or', 'not' for per-particle conditions") from exc
        raise
    except NameError as exc:
        avail = ", ".join(sorted(ns))
        raise NameError(f"{exc} in {expr!r}. Available: {avail}") from exc
