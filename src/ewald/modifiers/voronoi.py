"""Voronoi analysis (port of OVITO's VoronoiAnalysisModifier, Copyright 2026 OVITO GmbH,
used under the MIT option of its GPLv3-or-MIT license), built on voro++."""
from __future__ import annotations

import numpy as np

from ..data import Cell
from ..pipeline import Modifier
from ._knn import PeriodicKDTree
from .structure import _native


class VoronoiAnalysis(Modifier):
    """Per-particle Voronoi cells: volume, coordination, Voronoi indices.

    Writes 'Atomic Volume', 'Coordination', 'Max Face Order', 'Cavity Radius'
    and, if ``compute_indices``, 'Voronoi Index' (N, max face order) whose
    column n-1 counts faces with n edges (so <0,0,0,0,12> -> index[:, 4] == 12).
    ``use_radii`` gives the radical (polydisperse) tessellation from 'Radius'.
    Faces smaller than ``face_threshold`` (absolute) or ``relative_face_threshold``
    times the cell surface area, and edges shorter than ``edge_threshold``,
    are ignored for coordination and indices, as in OVITO.

    With ``compute_bonds``, the table ``tables['voronoi-faces']`` lists every
    face as (i, j, area, order).  Attributes: 'Voronoi.max_face_order' and
    'Voronoi.volume_sum' (equals the cell volume for a periodic system).
    """

    def __init__(self, compute_indices: bool = False, use_radii: bool = False, edge_threshold: float = 0.0,
                 face_threshold: float = 0.0, relative_face_threshold: float = 0.0,
                 compute_bonds: bool = False, only_selected: bool = False, num_threads: int = 0):
        self.compute_indices = compute_indices
        self.use_radii = use_radii
        self.edge_threshold = edge_threshold
        self.face_threshold = face_threshold
        self.relative_face_threshold = relative_face_threshold
        self.compute_bonds = compute_bonds
        self.only_selected = only_selected
        self.num_threads = num_threads

    def modify(self, data, ctx):
        native = _native()
        n_all = data.count
        subset = np.nonzero(data.selection)[0] if self.only_selected and "Selection" in data else None
        pos = data.positions if subset is None else data.positions[subset]
        cell = data.cell
        if not cell.is_valid:  # no cell: non-periodic bounding box with a small margin
            lo, hi = pos.min(0) - 1.0, pos.max(0) + 1.0
            cell = Cell(np.diag(hi - lo), lo, np.zeros(3, bool))
        radii = None
        if self.use_radii:
            if "Radius" not in data:
                raise ValueError("use_radii=True needs a 'Radius' property")
            radii = np.asarray(data["Radius"] if subset is None else data["Radius"][subset], float)
        n = len(pos)
        res = None
        if n:
            tree = PeriodicKDTree(pos, cell)
            todo = np.arange(n)
            # non-periodic: k = n leaves an empty (inf) column, telling the kernel the list is complete
            kmax = 1 << 14 if tree.periodic else n
            k = min(32, kmax)
            res = {}
            while len(todo):
                index, delta, dist = tree.query(pos[todo], k + 1)
                # drop the self entry (distance 0) of each query
                index, delta, dist = index[:, 1:], delta[:, 1:], dist[:, 1:]
                count = np.isfinite(dist).sum(1).astype(np.int32)
                part = native.voronoi_analysis(pos[todo], cell.matrix, cell.origin, cell.pbc, todo, index, delta,
                                               count, radii, self.edge_threshold, self.face_threshold,
                                               self.relative_face_threshold, self.compute_bonds, self.num_threads)
                done = part["status"] != 1
                for key, val in part.items():
                    if key.startswith("face_"):
                        continue
                    if key not in res:
                        res[key] = np.zeros((n,) + val.shape[1:], val.dtype)
                    res[key][todo[done]] = val[done]
                if self.compute_bonds:
                    keep = np.isin(part["face_i"], todo[done])
                    for key in ("face_i", "face_j", "face_order", "face_area"):
                        res.setdefault(key, []).append(part[key][keep])
                todo = todo[~done]
                if k >= kmax:
                    break
                k = min(2 * k, kmax)
            if np.any(res["status"] == 2):
                bad = int(np.count_nonzero(res["status"] == 2))
                raise RuntimeError(f"voro++ could not compute the Voronoi cell of {bad} particle(s); make sure "
                                   "particles lie strictly inside non-periodic cell boundaries")

        def full(key, shape=(), dtype=float):
            out = np.zeros((n_all,) + shape, dtype)
            if res is not None:
                if subset is None:
                    out[...] = res[key][(slice(None),) + tuple(slice(0, s) for s in shape)]
                else:
                    out[subset] = res[key][(slice(None),) + tuple(slice(0, s) for s in shape)]
            return out

        data.particles["Atomic Volume"] = full("volume")
        data.particles["Coordination"] = full("coordination", dtype=np.int32)
        data.particles["Cavity Radius"] = full("cavity_radius")
        mfo = full("max_face_order", dtype=np.int32)
        data.particles["Max Face Order"] = mfo
        max_order = int(mfo.max()) if n_all else 0
        data.attributes["Voronoi.max_face_order"] = max_order
        data.attributes["Voronoi.volume_sum"] = float(data.particles["Atomic Volume"].sum())
        if self.compute_indices:
            width = min(max(max_order, 1), native.FACE_ORDER_STORAGE_LIMIT)
            data.particles["Voronoi Index"] = full("indices", (width,), np.int32)
        if self.compute_bonds:
            fi = np.concatenate(res["face_i"]) if res else np.zeros(0, np.int64)
            fj = np.concatenate(res["face_j"]) if res else np.zeros(0, np.int64)
            if subset is not None:
                fi, fj = subset[fi], subset[fj]
            data.tables["voronoi-faces"] = {
                "i": fi, "j": fj,
                "area": np.concatenate(res["face_area"]) if res else np.zeros(0),
                "order": np.concatenate(res["face_order"]) if res else np.zeros(0, np.int32)}
