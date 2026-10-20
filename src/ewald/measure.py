"""Distances, angles and dihedrals between atoms, with the minimum image convention.

The view's ``measurements`` setting (``view(pipe, measurements=[[0, 1], [0, 1, 2]])``) draws them in the
viewport and in pictures (:class:`ewald.scene.MeasurementsVis` for the lines,
:class:`ewald.overlays.MeasurementLabels` for the values); :meth:`ewald.view3d.View.measure` adds one and
returns its value. An item of 2 atoms is a distance (Angstrom), 3 an angle at the middle atom and 4 a
dihedral about the middle bond (degrees)::

    from ewald.measure import measure
    measure(data, [0, 1])          # 2.761
    measure(data, [1, 0, 2])       # 90.0
"""
from __future__ import annotations

import numpy as np

KINDS = {2: "distance", 3: "angle", 4: "dihedral"}


def check(measurements, count: int | None = None) -> list[list[int]]:
    """``measurements`` as a list of index lists (2, 3 or 4 atoms each); ``count`` also checks the range."""
    if measurements is None:
        return []
    items = np.asarray(measurements, dtype=object)
    if items.ndim == 1 and len(items) and not isinstance(items[0], (list, tuple, np.ndarray)):
        measurements = [measurements]   # one item given bare: [0, 1]
    out = []
    for item in measurements:
        idx = [int(i) for i in item]
        if len(idx) not in KINDS:
            raise ValueError(f"a measurement is 2 (distance), 3 (angle) or 4 (dihedral) atoms, not {item!r}")
        if len(set(idx)) != len(idx):
            raise ValueError(f"a measurement needs different atoms, not {item!r}")
        if min(idx) < 0 or (count is not None and max(idx) >= count):
            raise IndexError(f"measurement {item!r}: atom index out of range" +
                             (f" (0..{count - 1})" if count is not None else ""))
        out.append(idx)
    return out


def mic(data, v) -> np.ndarray:
    """Vectors ``v`` folded to their shortest periodic image along the periodic cell directions."""
    v = np.asarray(v, float)
    cell = data.cell
    if not cell.is_valid:
        return v
    f = v @ np.linalg.inv(cell.matrix)
    f = f - np.round(f) * np.asarray(cell.pbc, float)
    return f @ cell.matrix


def points(data, indices) -> np.ndarray:
    """The atoms of one measurement as a connected chain (each next atom at its nearest image of the
    one before), from the first atom's position: what is drawn."""
    pos = np.asarray(data.positions, float)[list(indices)]
    out = [pos[0]]
    for k in range(1, len(pos)):
        out.append(out[-1] + mic(data, pos[k] - pos[k - 1]))
    return np.array(out)


def measure(data, indices) -> float:
    """Distance (Angstrom), angle or dihedral (degrees) of 2, 3 or 4 atoms ``indices``."""
    (idx,) = check([indices], data.count)
    p = points(data, idx)
    if len(p) == 2:
        return float(np.linalg.norm(p[1] - p[0]))
    if len(p) == 3:
        a, b = p[0] - p[1], p[2] - p[1]
        return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1n = b1 / np.linalg.norm(b1)
    v, w = b0 - (b0 @ b1n) * b1n, b2 - (b2 @ b1n) * b1n
    return float(np.degrees(np.arctan2(np.cross(b1n, v) @ w, v @ w)))


def text(data, indices) -> str:
    """The value as a label: '2.761 Å' or '90.0°'."""
    value = measure(data, indices)
    return f"{value:.3f} Å" if len(indices) == 2 else f"{value:.1f}°"


def anchor(data, indices) -> np.ndarray:
    """Where the label goes: the middle of a distance or of a dihedral's middle bond, the vertex of an angle."""
    p = points(data, indices)
    return p[1] if len(p) == 3 else p[len(p) // 2 - 1: len(p) // 2 + 1].mean(0)


def table(data, measurements) -> list[dict]:
    """``[{atoms, kind, value}]`` for ``measurements`` (get_view, the GUI)."""
    return [{"atoms": idx, "kind": KINDS[len(idx)], "value": round(measure(data, idx), 6)}
            for idx in check(measurements, data.count)]
