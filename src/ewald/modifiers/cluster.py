"""Cluster analysis (port of OVITO's ClusterAnalysisModifier, Copyright 2026 OVITO GmbH,
used under the MIT option of its GPLv3-or-MIT license)."""
from __future__ import annotations

import colorsys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

from ..neighbors import find_pairs
from ..pipeline import Modifier


class ClusterAnalysis(Modifier):
    """Group particles into clusters connected by distances below ``cutoff``.

    Writes 'Cluster' (ids from 1; 0 for particles excluded by ``only_selected``)
    and the table ``tables['clusters']`` with 'Cluster Identifier' and
    'Cluster Size', plus 'Center of Mass' (``compute_com``) and 'Radius of
    Gyration' / 'Gyration Tensor' (XX, YY, ZZ, XY, XZ, YZ; ``compute_gyration``).
    Centers of mass and gyration use 'Mass' if present and coordinates unwrapped
    across periodic boundaries within each cluster; ``unwrap_particles`` also
    writes those coordinates to 'Position'.  Without ``sort_by_size`` clusters
    are numbered in order of their lowest particle index, as in OVITO;
    ``sort_by_size`` renumbers them largest first and sets
    'ClusterAnalysis.largest_size'.  'ClusterAnalysis.cluster_count' is always set.
    """

    def __init__(self, cutoff: float = 3.2, sort_by_size: bool = False, compute_com: bool = False,
                 compute_gyration: bool = False, unwrap_particles: bool = False, only_selected: bool = False,
                 cluster_coloring: bool = False):
        self.cutoff = cutoff
        self.sort_by_size = sort_by_size
        self.compute_com = compute_com
        self.compute_gyration = compute_gyration
        self.unwrap_particles = unwrap_particles
        self.only_selected = only_selected
        self.cluster_coloring = cluster_coloring

    def modify(self, data, ctx):
        be = ctx.backend
        n_all = data.count
        subset = np.nonzero(data.selection)[0] if self.only_selected and "Selection" in data else np.arange(n_all)
        pos = data.positions[subset]
        n = len(pos)
        i, j, delta, _ = find_pairs(pos, data.cell, self.cutoff, be)
        i, j, delta = be.to_numpy(i), be.to_numpy(j), be.to_numpy(delta)
        graph = csr_matrix((np.ones(len(i), np.int8), (i, j)), shape=(n, n))
        ncl, labels = connected_components(graph, directed=False)
        # Number clusters by their lowest particle index (OVITO's seed order).
        first = np.full(ncl, n, dtype=np.int64)
        np.minimum.at(first, labels, np.arange(n))
        rank = np.empty(ncl, dtype=np.int64)
        rank[np.argsort(first, kind="stable")] = np.arange(ncl)
        cluster = rank[labels] + 1
        sizes = np.bincount(cluster, minlength=ncl + 1)[1:]

        need_unwrap = self.compute_com or self.compute_gyration or self.unwrap_particles
        if need_unwrap:
            unwrapped = self._unwrap(pos, i, j, delta, first[np.argsort(first, kind="stable")])
            mass = np.asarray(data["Mass"], float)[subset] if "Mass" in data else np.ones(n)
            msum = np.bincount(cluster - 1, weights=mass, minlength=ncl)
            msafe = np.where(msum > 0, msum, 1.0)
            com = np.stack([np.bincount(cluster - 1, weights=mass * unwrapped[:, d], minlength=ncl)
                            for d in range(3)], 1) / msafe[:, None]
            com[msum <= 0] = 0.0
            if self.compute_gyration:
                dr = unwrapped - com[cluster - 1]
                comps = [dr[:, 0] ** 2, dr[:, 1] ** 2, dr[:, 2] ** 2,
                         dr[:, 0] * dr[:, 1], dr[:, 0] * dr[:, 2], dr[:, 1] * dr[:, 2]]
                gyr = np.stack([np.bincount(cluster - 1, weights=mass * c, minlength=ncl) for c in comps], 1)
                gyr /= msafe[:, None]
                rg = np.sqrt(gyr[:, :3].sum(1))

        order = np.arange(ncl)
        if self.sort_by_size and ncl:
            order = np.argsort(-sizes, kind="stable")
            remap = np.empty(ncl + 1, dtype=np.int64)
            remap[0] = 0
            remap[order + 1] = np.arange(1, ncl + 1)
            cluster = remap[cluster]
            data.attributes["ClusterAnalysis.largest_size"] = int(sizes[order[0]])

        out = np.zeros(n_all, dtype=np.int64)
        out[subset] = cluster
        data.particles["Cluster"] = out
        data.attributes["ClusterAnalysis.cluster_count"] = int(ncl)
        table = {"Cluster Identifier": np.arange(1, ncl + 1, dtype=np.int64), "Cluster Size": sizes[order]}
        if self.compute_com:
            table["Center of Mass"] = com[order]
        if self.compute_gyration:
            table["Radius of Gyration"] = rg[order]
            table["Gyration Tensor"] = gyr[order]
        data.tables["clusters"] = table
        if self.unwrap_particles:
            p = data.positions.copy()
            p[subset] = unwrapped
            data.positions = p
        if self.cluster_coloring:
            rng = np.random.default_rng(1)
            hsv = np.stack([rng.random(ncl + 1), 1 - 0.4 * rng.random(ncl + 1), 1 - 0.3 * rng.random(ncl + 1)], 1)
            colors = np.array([colorsys.hsv_to_rgb(*c) for c in hsv])
            colors[0] = 0.8
            data.particles["Color"] = colors[out]

    @staticmethod
    def _unwrap(pos, i, j, delta, seeds):
        """Coordinates made contiguous across periodic boundaries by a BFS from each cluster seed."""
        n = len(pos)
        # A virtual root linked to every seed turns the forest into one BFS tree.
        rows = np.concatenate([i, np.full(len(seeds), n)])
        cols = np.concatenate([j, seeds])
        graph = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n + 1, n + 1))
        order, pred = breadth_first_order(graph, n, directed=True, return_predecessors=True)
        unwrapped = pos.copy()
        key = i * n + j                                      # find_pairs output is sorted by (i, j)
        nodes = order[1:]
        parent = pred[nodes]
        tree = parent != n
        nodes, parent = nodes[tree], parent[tree]
        step = delta[np.searchsorted(key, parent * n + nodes)]
        done = np.zeros(n, bool)
        done[seeds] = True
        while len(nodes):
            ready = done[parent]
            unwrapped[nodes[ready]] = unwrapped[parent[ready]] + step[ready]
            done[nodes[ready]] = True
            nodes, parent, step = nodes[~ready], parent[~ready], step[~ready]
        return unwrapped
