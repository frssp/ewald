"""Cluster analysis (port of OVITO's ClusterAnalysisModifier, Copyright 2026 OVITO GmbH,
used under the MIT option of its GPLv3-or-MIT license; see ewald/_csrc/LICENSE.OVITO-MIT.txt), CPU/GPU."""
from __future__ import annotations

import colorsys

import numpy as np

from ..kernels import cluster_properties, connected_components, unwrap_clusters
from ..neighbors import find_pairs
from ..pipeline import Modifier


class ClusterAnalysis(Modifier):
    """Group particles into clusters connected by distances below ``cutoff``.

    Writes 'Cluster' (ids from 1; 0 for particles excluded by ``only_selected``)
    and the table ``tables['clusters']`` with 'Cluster Identifier' and
    'Cluster Size', plus 'Center of Mass' (``compute_com``) and 'Radius of
    Gyration' / 'Gyration Tensor' (XX, YY, ZZ, XY, XZ, YZ; ``compute_gyration``).
    Centers of mass and gyration use 'Mass' if present and coordinates unwrapped
    across periodic boundaries within each cluster (walking bonds from its
    lowest-index particle); ``unwrap_particles`` also writes those coordinates
    to 'Position'. Without ``sort_by_size`` clusters are numbered in order of
    their lowest particle index, as in OVITO; ``sort_by_size`` renumbers them
    largest first and sets 'ClusterAnalysis.largest_size'.
    'ClusterAnalysis.cluster_count' is always set.
    """

    def __init__(self, cutoff: float = 3.2, sort_by_size: bool = False, compute_com: bool = False,
                 compute_gyration: bool = False, unwrap_particles: bool = False,
                 only_selected: bool = False, cluster_coloring: bool = False):
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
        subset = np.nonzero(data.selection)[0] if self.only_selected else np.arange(n_all)
        pos = data.positions[subset]
        i, j, delta, _ = find_pairs(pos, data.cell, self.cutoff, be, half=True)
        labels, ncl = connected_components(len(pos), i, j, be)   # numbered by lowest member
        sizes = be.to_numpy(be.xp.bincount(labels, minlength=ncl))

        if self.compute_com or self.compute_gyration or self.unwrap_particles:
            unwrapped = unwrap_clusters(pos, i, j, delta, labels, ncl, be)
            mass = np.asarray(data["Mass"], float)[subset] if "Mass" in data else None
            _, com, gyr, rg = cluster_properties(unwrapped, labels, ncl, mass, be)
            unwrapped = be.to_numpy(unwrapped)
        cluster = be.to_numpy(labels) + 1

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
        table = {"Cluster Identifier": np.arange(1, ncl + 1, dtype=np.int64),
                 "Cluster Size": sizes[order]}
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
            hsv = np.stack([rng.random(ncl + 1), 1 - 0.4 * rng.random(ncl + 1),
                            1 - 0.3 * rng.random(ncl + 1)], 1)
            colors = np.array([colorsys.hsv_to_rgb(*c) for c in hsv])
            colors[0] = 0.8
            from .. import color_rule

            data.particles["Color"] = colors[out]
            color_rule.record(data, {"kind": "categorical", "source": "the ClusterAnalysis step (cluster_coloring=True)",
                                     "property": "Cluster", "note": "one random color per cluster; "
                                     "gray #cccccc = no cluster (Cluster 0)"})
