"""Volumetric data: isosurfaces, slice planes and grid differences (CHGCAR, LOCPOT, cube)."""
from __future__ import annotations

import os

import numpy as np

from ..mesh import SurfaceMesh
from ..pipeline import Modifier


def _grid(data, grid, property):
    from ..io.volumetric import _pick_grid

    g = _pick_grid(data, grid, property)
    if property not in g:
        raise KeyError(f"grid has no property {property!r}; available: {g.names()}")
    values = np.asarray(g[property], float)
    if values.ndim != 3:
        raise ValueError(f"{property!r} has {values.shape[3]} components; pick one, e.g. '{property}.Z'")
    return g, values


def isosurface(grid, values, level: float) -> SurfaceMesh:
    """Marching-cubes (VTK flying edges) surface of ``values`` at ``level`` on ``grid``,
    closed across the periodic cell faces and mapped into the (possibly triclinic) cell."""
    import pyvista as pv

    padded = values
    for axis in range(3):  # repeat the first layer so the surface closes across periodic faces
        if grid.domain.pbc[axis]:
            padded = np.concatenate([padded, np.take(padded, [0], axis=axis)], axis=axis)
    img = pv.ImageData(dimensions=padded.shape)
    img.point_data["v"] = np.ascontiguousarray(padded.ravel(order="F"))
    mesh = img.contour([level], scalars="v", method="flying_edges")
    if mesh.n_points == 0:
        return SurfaceMesh()
    mesh = mesh.triangulate()
    frac = np.asarray(mesh.points, float) / np.array(grid.shape)
    mesh.points = frac @ grid.domain.matrix + grid.domain.origin
    mesh = mesh.compute_normals(cell_normals=False, split_vertices=False, consistent_normals=False,
                                auto_orient_normals=False)
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    return SurfaceMesh(np.asarray(mesh.points), faces,
                       vertex_properties={"Normal": np.asarray(mesh.point_data["Normals"], float)})


class CreateIsosurface(Modifier):
    """VESTA-style isosurface of a grid property (OVITO's CreateIsosurfaceModifier).

    ``isolevel`` in the property's units (e/A^3 for a CHGCAR "Charge Density"); None uses
    the mean of |values|. ``property`` can be "Charge Density", "Magnetization", "Spin Up",
    "Spin Down", "Potential", "ELF", "Value" (cube) or a component like "Magnetization.Z".
    ``signed=True`` also draws the surface at -isolevel in ``negative_color``: the usual
    view of charge-density differences and spin densities. The surfaces go into
    ``data.surfaces[identifier]`` (and ``identifier + "-"``); the level used is stored
    in the attribute ``"<identifier>.isolevel"``.
    """

    def __init__(self, isolevel: float | None = None, property: str = "Charge Density", grid: str | None = None,
                 signed: bool = False, color=(1.0, 0.86, 0.0), negative_color=(0.0, 0.75, 1.0),
                 transparency: float = 0.0, identifier: str = "isosurface"):
        self.isolevel = isolevel
        self.property = property
        self.grid = grid
        self.signed = signed
        self.color = color
        self.negative_color = negative_color
        self.transparency = transparency
        self.identifier = identifier

    def _surface(self, grid, values, level, color):
        from .coloring import _to_rgb

        surf = isosurface(grid, values, level)
        surf.face_properties["Color"] = np.tile(_to_rgb(color), (surf.face_count, 1))
        surf.transparency = float(self.transparency)
        return surf

    def modify(self, data, ctx):
        g, values = _grid(data, self.grid, self.property)
        level = float(np.mean(np.abs(values))) if self.isolevel is None else float(self.isolevel)
        surf = self._surface(g, values, level, self.color)
        data.surfaces[self.identifier] = surf
        if self.signed:
            data.surfaces[self.identifier + "-"] = self._surface(g, values, -abs(level), self.negative_color)
        data.attributes[f"{self.identifier}.isolevel"] = level
        data.attributes[f"{self.identifier}.area"] = surf.area


def _plane_polygon(miller, distance):
    """Corners (fractional) of the plane h*fa + k*fb + l*fc = distance inside the unit cell."""
    n = np.asarray(miller, float)
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    pts = []
    for a in range(8):
        for b in range(a + 1, 8):
            if np.abs(corners[a] - corners[b]).sum() != 1:
                continue  # not a cube edge
            fa, fb = corners[a] @ n - distance, corners[b] @ n - distance
            for f, p in ((fa, corners[a]), (fb, corners[b])):
                if abs(f) < 1e-12:
                    pts.append(p)
            if fa * fb < 0:
                pts.append(corners[a] + fa / (fa - fb) * (corners[b] - corners[a]))
    pts = np.unique(np.round(np.array(pts).reshape(-1, 3), 12), axis=0)
    return pts if len(pts) >= 3 else None


def _triangulate_polygon(points, step):
    """Fan-triangulate a convex polygon (Cartesian, unordered) and subdivide every triangle
    so that no edge is longer than ``step``. Returns (vertices, faces, unit normal)."""
    c = points.mean(0)
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    for k in range(3, len(points)):
        if np.linalg.norm(normal) > 1e-9:
            break
        normal = np.cross(points[1] - points[0], points[k] - points[0])
    normal /= np.linalg.norm(normal)
    u = points[0] - c
    u /= np.linalg.norm(u)
    w = np.cross(normal, u)
    ang = np.arctan2((points - c) @ w, (points - c) @ u)
    ring = points[np.argsort(ang)]
    verts, faces = [], []
    offset = 0
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        m = max(1, int(np.ceil(max(np.linalg.norm(a - c), np.linalg.norm(b - c), np.linalg.norm(b - a)) / step)))
        # barycentric lattice of the triangle (c, a, b) with m subdivisions
        idx = {}
        for i in range(m + 1):
            for j in range(m + 1 - i):
                idx[i, j] = len(idx)
                verts.append(c + (a - c) * i / m + (b - c) * j / m)
        for i in range(m):
            for j in range(m - i):
                faces.append((offset + idx[i, j], offset + idx[i + 1, j], offset + idx[i, j + 1]))
                if j < m - i - 1:
                    faces.append((offset + idx[i + 1, j], offset + idx[i + 1, j + 1], offset + idx[i, j + 1]))
        offset += len(idx)
    return np.array(verts), np.array(faces, np.int64), normal


class GridSlice(Modifier):
    """Colour-mapped cross-section of a grid property on a lattice plane (VESTA's 2D slice).

    The plane is ``h*x + k*y + l*z = distance`` in fractional coordinates for
    ``miller=(h, k, l)``, i.e. ``distance`` is in units of the interplanar spacing d_hkl
    (``(0, 0, 1)``, 0.5 cuts the cell halfway along c). Values are trilinearly
    interpolated (periodically) onto a mesh with ``resolution`` Angstrom spacing (None:
    one voxel) and mapped through the matplotlib colormap ``cmap`` over ``range``
    (None: the min/max on the plane); ``scale="log"`` maps log10(|value|), which shows
    the valence density between nuclei. The slice goes into ``data.surfaces[identifier]``
    and its colour range into the ``ColorCoding.*`` attributes, so a legend is drawn
    (unless a particle ColorCoding already set one).
    """

    def __init__(self, miller=(0, 0, 1), distance: float = 0.5, property: str = "Charge Density",
                 grid: str | None = None, cmap: str = "viridis", range=None, scale: str = "linear",
                 resolution: float | None = None, transparency: float = 0.0, identifier: str = "slice"):
        self.miller = miller
        self.distance = distance
        self.property = property
        self.grid = grid
        self.cmap = cmap
        self.range = range
        self.scale = scale
        self.resolution = resolution
        self.transparency = transparency
        self.identifier = identifier

    def modify(self, data, ctx):
        import matplotlib

        if self.scale not in ("linear", "log"):
            raise ValueError(f"scale must be 'linear' or 'log', got {self.scale!r}")
        g, _ = _grid(data, self.grid, self.property)
        if not np.any(self.miller):
            raise ValueError("miller indices must not all be zero")
        frac = _plane_polygon(self.miller, float(self.distance))
        if frac is None:
            raise ValueError(f"plane {tuple(self.miller)} at distance {self.distance} does not cut the cell")
        step = self.resolution or float(np.min(g.domain.perpendicular_widths() / np.array(g.shape)))
        verts, faces, normal = _triangulate_polygon(frac @ g.domain.matrix + g.domain.origin, step)
        values = g.interpolate(self.property, verts)
        label = self.property
        if self.scale == "log":
            values = np.log10(np.maximum(np.abs(values), 1e-12))
            label = f"log10 {self.property}"
        lo, hi = (float(values.min()), float(values.max())) if self.range is None else map(float, self.range)
        t = np.clip((values - lo) / (hi - lo if hi != lo else 1.0), 0, 1)
        colors = matplotlib.colormaps[self.cmap](t)[:, :3]
        data.surfaces[self.identifier] = SurfaceMesh(
            verts, faces, transparency=self.transparency,
            vertex_properties={"Normal": np.broadcast_to(normal, verts.shape), "Color": colors})
        data.attributes[f"{self.identifier}.range"] = (lo, hi)
        if "ColorCoding.range" not in data.attributes:
            data.attributes["ColorCoding.property"] = label
            data.attributes["ColorCoding.cmap"] = self.cmap
            data.attributes["ColorCoding.range"] = (lo, hi)


class GridDifference(Modifier):
    """Charge-density difference: subtract the grids of other files, e.g.
    rho(AB) - rho(A) - rho(B) with ``GridDifference(["A/CHGCAR", "B/CHGCAR"])``.

    The files must have the same grid shape. ``property`` is subtracted from the same
    property of each file's first grid (the files are read once and cached). The result
    replaces ``property`` (or goes into ``output`` when given), so a following
    ``CreateIsosurface(signed=True)`` shows accumulation and depletion.
    """

    def __init__(self, subtract=(), property: str = "Charge Density", grid: str | None = None,
                 output: str | None = None):
        self.subtract = subtract
        self.property = property
        self.grid = grid
        self.output = output
        self._cache: dict = {}

    def _other(self, path):
        from ..io import open_reader

        key = (os.path.abspath(path), os.path.getmtime(path))
        if key not in self._cache:
            other = open_reader(path).read(0)
            g, values = _grid(other, None, self.property)
            self._cache[key] = values
        return self._cache[key]

    def modify(self, data, ctx):
        paths = [self.subtract] if isinstance(self.subtract, (str, os.PathLike)) else list(self.subtract)
        g, values = _grid(data, self.grid, self.property)
        result = values
        for path in paths:
            other = self._other(os.fspath(path))
            if other.shape != values.shape:
                raise ValueError(f"{path}: grid {other.shape} does not match {values.shape}")
            result = result - other
        name = next(k for k, v in data.grids.items() if v is g)
        new = g.copy()
        new[self.output or self.property] = result
        data.grids[name] = new
