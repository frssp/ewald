"""Describe a few particles the user picked, and their surroundings, for Claude.

This is the headless half of "Ask about selection": the GUI (or a script) passes the
indices of the picked or selected particles, and these functions return a compact,
JSON-ready description of the selection and of the shell of radius ``radius`` around it:
per-atom neighbor lists and coordination numbers, structure types from CNA/PTM if the
pipeline has them, property values next to their frame averages, and the distance to the
nearest defect. :func:`selection_scene` cuts the same region out for a close-up render.

>>> from ewald.llm.selection import describe_selection
>>> describe_selection(pipe.compute(0), [17, 18], radius=6.0)["selection"]["types"]
{'Cu': 2}
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from ..data import Cell, DataCollection
from ..neighbors import find_pairs
from .describe import _round, suggest_cutoff

#: Highlight color of the selected particles in :func:`selection_scene` (the GUI pick color).
HIGHLIGHT = (1.0, 0.847, 0.302)

_SKIP = {"Position", "Particle Type", "Particle Identifier", "Color", "Selection", "Structure Type",
         "Transparency"}
_COUNTS = re.compile(r"^(?P<prefix>.+)\.counts\.(?P<name>[^.]+)$")


def structure_names(data: DataCollection) -> list[str] | None:
    """Names of the 'Structure Type' ids, read from the ``<Modifier>.counts.<NAME>`` attributes
    that every structure-identification modifier writes in id order (the last one wins)."""
    if "Structure Type" not in data:
        return None
    groups: dict[str, list[str]] = {}
    for key in data.attributes:
        m = _COUNTS.match(key)
        if m:
            groups.setdefault(m["prefix"], []).append(m["name"])
    return list(groups.values())[-1] if groups else None


def _structure_label(names, t) -> str:
    t = int(t)
    return names[t] if names and 0 <= t < len(names) else str(t)


def _scalar_properties(data: DataCollection, limit: int = 12) -> list[str]:
    out = []
    for k, v in data.particles.items():
        if k in _SKIP or v.ndim != 1 or v.dtype.kind not in "biuf":
            continue
        out.append(k)
    return out[:limit]


def _ids(data: DataCollection) -> np.ndarray:
    return np.asarray(data["Particle Identifier"]) if "Particle Identifier" in data else np.arange(data.count)


def _distances_to(data: DataCollection, center: np.ndarray) -> np.ndarray:
    return np.linalg.norm(data.cell.minimum_image(data.positions - center), axis=1)


def _environment_mask(data: DataCollection, idx: np.ndarray, centroid: np.ndarray, extent: float,
                      radius: float) -> np.ndarray:
    """Particles within ``radius`` of any selected particle (of the centroid sphere for big selections)."""
    if len(idx) > 256:
        return _distances_to(data, centroid) <= extent + radius
    near = np.zeros(data.count, bool)
    for s in idx:
        near |= _distances_to(data, data.positions[s]) <= radius
    return near


def _cutoff(data: DataCollection, cutoff) -> float:
    if cutoff in (None, "auto"):
        return suggest_cutoff(data) or 3.0
    return float(cutoff)


def describe_selection(data: DataCollection, indices, radius: float = 6.0, cutoff: float | str = "auto",
                       max_atoms: int = 12, max_neighbors: int = 16, frame: int | None = None) -> dict:
    """JSON-ready description of particles ``indices`` (0-based) and their surroundings.

    ``cutoff`` (Angstrom, or "auto": the first minimum of g(r)) defines neighbors and
    coordination numbers; ``radius`` the environment shell around the selection. Up to
    ``max_atoms`` selected particles are listed one by one with ``max_neighbors`` neighbors.
    """
    idx = np.unique(np.asarray(indices, dtype=int).ravel())
    if not len(idx):
        raise ValueError("the selection is empty; pick particles or select them with ExpressionSelection")
    if idx[0] < 0 or idx[-1] >= data.count:
        raise ValueError(f"particle index out of range (the frame has {data.count} particles)")
    pos, cell = data.positions, data.cell
    types = np.asarray(data.types).astype(str)
    ids = _ids(data)
    rc = _cutoff(data, cutoff)
    names = structure_names(data)
    struct = np.asarray(data["Structure Type"]) if "Structure Type" in data else None
    props = _scalar_properties(data)

    # coordination numbers of every particle (frame averages are the reference for the selection)
    i, j, _, _ = find_pairs(pos, cell, rc, half=True)
    i, j = np.asarray(i), np.asarray(j)
    cn = np.bincount(i, minlength=data.count) + np.bincount(j, minlength=data.count)

    ref = pos[idx[0]]
    rel = cell.minimum_image(pos[idx] - ref)
    centroid = ref + rel.mean(axis=0)
    extent = float(np.linalg.norm(rel - rel.mean(axis=0), axis=1).max())
    selected = np.zeros(data.count, bool)
    selected[idx] = True

    atoms = []
    for s in idx[:max_atoms]:
        d = _distances_to(data, pos[s])
        d[s] = np.inf
        order = np.argsort(d)
        within = order[d[order] <= rc]
        entry = {"index": int(s), "id": int(ids[s]), "type": types[s],
                 "position": _round(list(pos[s])), "cn": int(cn[s]),
                 "cn_by_type": dict(Counter(types[within].tolist())),
                 "nearest_distance": _round(d[order[0]]) if np.isfinite(d[order[0]]) else None,
                 "neighbors": [{"id": int(ids[k]), "type": types[k], "d": _round(d[k], 4),
                                **({"selected": True} if selected[k] else {})}
                               for k in order[:max_neighbors] if d[k] <= max(rc * 1.5, rc + 1.0)]}
        if struct is not None:
            entry["structure"] = _structure_label(names, struct[s])
        for p in props:
            entry[p] = _round(data[p][s])
        atoms.append(entry)

    sel = {"count": int(len(idx)), "types": dict(Counter(types[idx].tolist())),
           "centroid": _round(list(centroid)), "extent": _round(extent),
           "cn_mean_by_type": {t: _round(cn[idx][types[idx] == t].mean()) for t in np.unique(types[idx])}}
    if len(idx) > max_atoms:
        sel["listed"] = max_atoms
    if struct is not None:
        sel["structures"] = dict(Counter(_structure_label(names, t) for t in struct[idx]))
    if 2 <= len(idx) <= 6:
        sel["distances"] = [{"ids": [int(ids[a]), int(ids[b])],
                             "d": _round(np.linalg.norm(cell.minimum_image(pos[b] - pos[a])), 5)}
                            for n, a in enumerate(idx) for b in idx[n + 1:]]
    if len(idx) == 3:
        a, b, c = idx   # angle at the middle particle, as the viewport's angle tool measures it
        v1, v2 = cell.minimum_image(pos[a] - pos[b]), cell.minimum_image(pos[c] - pos[b])
        sel["angle_at_middle"] = _round(np.degrees(np.arccos(np.clip(
            v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))), 5)

    near = _environment_mask(data, idx, centroid, extent, radius) & ~selected
    env = {"radius": _round(radius), "count": int(near.sum()), "types": dict(Counter(types[near].tolist())),
           "cn_mean_by_type": {t: _round(cn[near][types[near] == t].mean()) for t in np.unique(types[near])}}
    frame_info = {"count": int(data.count), "types": dict(Counter(types.tolist())),
                  "cn_mean_by_type": {t: _round(cn[types == t].mean()) for t in np.unique(types)}}
    if struct is not None:
        env["structures"] = dict(Counter(_structure_label(names, t) for t in struct[near]))
        frame_info["structures"] = dict(Counter(_structure_label(names, t) for t in struct))
    for p in props:
        v = np.asarray(data[p], float)
        sel.setdefault("property_mean", {})[p] = _round(v[idx].mean())
        if near.any():
            env.setdefault("property_mean", {})[p] = _round(v[near].mean())
        frame_info.setdefault("property_mean", {})[p] = _round(v.mean())

    out = {"frame": frame, "cutoff": _round(rc), "selection": sel, "atoms": atoms, "environment": env,
           "frame_reference": frame_info}
    if struct is not None:
        bulk = Counter(struct.tolist()).most_common(1)[0][0]
        defect = (struct != bulk) & ~selected
        out["bulk_structure"] = _structure_label(names, bulk)
        if defect.any():
            d = _distances_to(data, centroid)
            k = int(np.argmin(np.where(defect, d, np.inf)))
            out["nearest_defect"] = {"id": int(ids[k]), "type": types[k],
                                     "structure": _structure_label(names, struct[k]),
                                     "distance_from_centroid": _round(d[k]),
                                     "defect_atoms_in_environment": int((defect & near).sum())}
    else:
        out["hint"] = ("no 'Structure Type' in the pipeline; add CommonNeighborAnalysis or "
                       "PolyhedralTemplateMatching to classify the local structure")
    return out


def selection_scene(data: DataCollection, indices, radius: float = 6.0) -> DataCollection:
    """The selection and the particles within ``radius`` of it, unwrapped around the selection
    (no periodic cell), selected particles in :data:`HIGHLIGHT` and the others drawn small so the
    selection stays visible inside a crystal: the scene for a close-up render."""
    from ..render import default_colors, default_radii

    idx = np.unique(np.asarray(indices, dtype=int).ravel())
    pos, cell = data.positions, data.cell
    ref = pos[idx[0]]
    rel = cell.minimum_image(pos[idx] - ref)
    centroid = ref + rel.mean(axis=0)
    extent = float(np.linalg.norm(rel - rel.mean(axis=0), axis=1).max())
    mask = _environment_mask(data, idx, centroid, extent, radius)
    mask[idx] = True
    keep = np.nonzero(mask)[0]
    colors = np.array(default_colors(data), float)
    colors[idx] = HIGHLIGHT
    radii = np.array(default_radii(data), float) * 0.4
    radii[idx] /= 0.4
    out = data.take(keep)
    out.bonds = None
    out.surfaces = {}
    out.grids = {}
    out.particles["Position"] = centroid + cell.minimum_image(pos[keep] - centroid)
    out.particles["Color"] = colors[keep]
    out.particles["Radius"] = radii[keep]
    out.cell = Cell(cell.matrix.copy(), cell.origin.copy(), np.zeros(3, bool))
    return out


def selection_prompt(question: str, description: dict) -> str:
    """The user turn for a question about a selection: the question plus the description."""
    import json

    n = description["selection"]["count"]
    return (f"The user picked {n} particle{'s' if n != 1 else ''} in the viewport"
            + (f" (frame {description['frame']})" if description.get("frame") is not None else "")
            + " and asks about them. Below is ewald's description of the selection and its surroundings"
            + " (distances in Angstrom; ids are Particle Identifiers, or indices when the file has none), and a close-up render with the"
            + " selected particles highlighted in yellow. Answer from these numbers and the image; call"
            + " tools (describe_selection with another radius or cutoff, add_modifier, render_image) when"
            + " something needs checking, and say which modifier would confirm your reading.\n\n"
            + f"Question: {question.strip()}\n\n<selection>\n{json.dumps(description, indent=1)}\n</selection>")


__all__ = ["describe_selection", "selection_scene", "selection_prompt", "structure_names", "HIGHLIGHT"]
