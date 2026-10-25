"""Triangle meshes in world coordinates (OVITO's ``data.surfaces``, simplified).

Modifiers that produce geometry (coordination polyhedra now; surface meshes and
isosurfaces later) store a ``SurfaceMesh`` in ``data.surfaces[name]``; the OpenGL
renderer and the ray tracer draw every entry.
"""
from __future__ import annotations

import numpy as np


class SurfaceMesh:
    """``vertices`` (V x 3), triangular ``faces`` (F x 3 vertex indices), per-face
    properties ('Color' F x 3, 'Region' F) and per-vertex properties ('Normal' V x 3 for
    smooth shading, e.g. isosurfaces; 'Color' V x 3, interpolated across faces, e.g. grid
    slices; it overrides the face colours). ``edges`` (E x 2 vertex indices) are outline
    lines drawn on top (VESTA's polyhedron edges); ``transparency`` 0..1 applies to the faces.

    ``lit``: whether the colours are shaded by the lights. None (default) shades face colours and keeps
    vertex colours true, as colour maps (grid slices) should read; True shades vertex colours too
    (smoothly coloured solids such as protein cartoons), False keeps face colours true.
    """

    def __init__(self, vertices=None, faces=None, face_properties=None, edges=None, *,
                 transparency: float = 0.0, edge_color=(0.25, 0.25, 0.25), edge_width: float = 0.04,
                 vertex_properties=None, lit: bool | None = None):
        self.vertices = np.zeros((0, 3)) if vertices is None else np.asarray(vertices, float).reshape(-1, 3)
        self.faces = np.zeros((0, 3), np.int64) if faces is None else np.asarray(faces, np.int64).reshape(-1, 3)
        self.face_properties: dict[str, np.ndarray] = dict(face_properties or {})
        self.vertex_properties: dict[str, np.ndarray] = dict(vertex_properties or {})
        self.edges = None if edges is None else np.asarray(edges, np.int64).reshape(-1, 2)
        self.transparency = float(transparency)
        self.edge_color = edge_color
        self.edge_width = float(edge_width)
        self.lit = lit

    @property
    def is_lit(self) -> bool:
        """Whether renderers shade this mesh (see ``lit``)."""
        return ("Color" not in self.vertex_properties) if self.lit is None else bool(self.lit)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def area(self) -> float:
        v = self.vertices[self.faces]
        return float(0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum())

    def face_colors(self, default=(0.6, 0.6, 0.6)) -> np.ndarray:
        c = self.face_properties.get("Color")
        return np.tile(np.asarray(default, float), (self.face_count, 1)) if c is None else np.asarray(c, float)

    def copy(self) -> "SurfaceMesh":
        new = SurfaceMesh.__new__(SurfaceMesh)
        new.__dict__.update(self.__dict__)
        new.face_properties = dict(self.face_properties)
        new.vertex_properties = dict(self.vertex_properties)
        return new

    def to_pyvista(self):
        import pyvista as pv

        faces = np.column_stack([np.full(self.face_count, 3), self.faces]).ravel()
        return pv.PolyData(self.vertices, faces=faces)

    def __repr__(self):
        return f"<SurfaceMesh V={self.vertex_count} F={self.face_count} props=[{', '.join(self.face_properties)}]>"
