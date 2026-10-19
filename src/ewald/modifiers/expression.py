"""OVITO-style per-particle expressions evaluated with numpy.

Property names are used as in OVITO: ``Position.Z``, ``ParticleType``,
``ParticleIdentifier``, ``ParticleIndex`` (0-based row), ``Velocity.X``, ``Coordination``; globals ``N``,
``Frame``, ``Timestep``, ``CellSize.X``. Combine conditions with numpy
operators: ``(Position.Z > 10) & (ParticleType == "Cu")``; numpy is ``np``.
"""
from __future__ import annotations

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


def evaluate(expr: str, data, frame=None):
    ns = namespace(data, frame)
    dotted = sorted((k for k in ns if "." in k), key=len, reverse=True)
    env = {"np": np, "abs": np.abs, "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
           "sin": np.sin, "cos": np.cos, "min": np.minimum, "max": np.maximum, "pi": np.pi}
    code = expr
    for i, k in enumerate(dotted):
        code = re.sub(r"(?<![\w.])" + re.escape(k) + r"(?![\w])", f"__v{i}", code)
        env[f"__v{i}"] = ns[k]
    env.update({k: v for k, v in ns.items() if "." not in k})
    try:
        return eval(compile(code, "<expression>", "eval"), {"__builtins__": {}}, env)
    except ValueError as exc:
        if "truth value" in str(exc):
            raise ValueError(f"{expr!r}: use '&', '|', '~' with parentheses instead of "
                             "'and', 'or', 'not' for per-particle conditions") from exc
        raise
    except NameError as exc:
        avail = ", ".join(sorted(ns))
        raise NameError(f"{exc} in {expr!r}. Available: {avail}") from exc
