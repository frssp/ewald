"""OpenGL 1.x renderer for the viewport over SSH X11 forwarding (``ssh -X``).

A forwarded display offers only indirect GLX, i.e. OpenGL 1.x, which VTK cannot use. VESTA works
there because it draws with fixed-function OpenGL and display lists: the geometry crosses the
network once, the user's own GPU draws it, and rotating only sends a new matrix. This module does
the same for an :class:`ewald.scene.Scene` (spheres, cylinders, lines, meshes, arrows), calling
``libGL`` through ctypes (no PyOpenGL), so it needs nothing beyond what the GUI already has.

The camera is a ``vtkCamera`` (pure math, no OpenGL), wrapped by :class:`CameraPlotter` so the
rest of the GUI (``setup_camera``, ``camera_state``, sessions, the render page) treats it like a
PyVista plotter.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import sys

import numpy as np

from ..render import FLAT_OUTLINE, STICK_OUTLINE

# ----------------------------------------------------------------------------- libGL through ctypes
GL_COMPILE = 0x1300
GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT = 0x4000, 0x0100
GL_DEPTH_TEST, GL_LIGHTING, GL_LIGHT0, GL_LIGHT1, GL_COLOR_MATERIAL = 0x0B71, 0x0B50, 0x4000, 0x4001, 0x0B57
GL_NORMALIZE, GL_BLEND, GL_CULL_FACE, GL_LINE_SMOOTH = 0x0BA1, 0x0BE2, 0x0B44, 0x0B20
GL_POLYGON_OFFSET_FILL, GL_POINT_SMOOTH = 0x8037, 0x0B10
GL_MODELVIEW, GL_PROJECTION = 0x1700, 0x1701
GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, GL_SPECULAR, GL_SHININESS = 0x0408, 0x1602, 0x1202, 0x1601
GL_POSITION, GL_DIFFUSE, GL_AMBIENT, GL_LIGHT_MODEL_AMBIENT, GL_LIGHT_MODEL_TWO_SIDE = 0x1203, 0x1201, 0x1200, 0x0B53, 0x0B52
GL_SMOOTH, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA = 0x1D01, 0x0302, 0x0303
GL_TRIANGLES, GL_LINES, GL_POINTS, GL_LINE = 0x0004, 0x0001, 0x0000, 0x1B01
GL_FLOAT, GL_DOUBLE, GL_UNSIGNED_INT = 0x1406, 0x140A, 0x1405
GL_VERTEX_ARRAY, GL_NORMAL_ARRAY, GL_COLOR_ARRAY = 0x8074, 0x8075, 0x8076
GL_VERSION, GL_RENDERER, GL_VENDOR = 0x1F02, 0x1F01, 0x1F00
GL_FILL = 0x1B02
GL_FRONT = 0x0404

_SIGNATURES = {
    "glClearColor": [ctypes.c_float] * 4, "glClear": [ctypes.c_uint], "glEnable": [ctypes.c_uint],
    "glDisable": [ctypes.c_uint], "glViewport": [ctypes.c_int] * 4, "glMatrixMode": [ctypes.c_uint],
    "glLoadMatrixd": [ctypes.c_void_p], "glMultMatrixd": [ctypes.c_void_p], "glLoadIdentity": [],
    "glPushMatrix": [], "glPopMatrix": [], "glTranslated": [ctypes.c_double] * 3,
    "glScaled": [ctypes.c_double] * 3, "glColor4d": [ctypes.c_double] * 4,
    "glGenLists": [ctypes.c_int], "glNewList": [ctypes.c_uint, ctypes.c_uint], "glEndList": [],
    "glCallList": [ctypes.c_uint], "glDeleteLists": [ctypes.c_uint, ctypes.c_int],
    "glLightfv": [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p],
    "glLightModelfv": [ctypes.c_uint, ctypes.c_void_p], "glLightModeli": [ctypes.c_uint, ctypes.c_int],
    "glMaterialfv": [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p],
    "glMaterialf": [ctypes.c_uint, ctypes.c_uint, ctypes.c_float],
    "glColorMaterial": [ctypes.c_uint, ctypes.c_uint], "glShadeModel": [ctypes.c_uint],
    "glLineWidth": [ctypes.c_float], "glPointSize": [ctypes.c_float],
    "glBlendFunc": [ctypes.c_uint, ctypes.c_uint], "glDepthMask": [ctypes.c_ubyte],
    "glPolygonMode": [ctypes.c_uint, ctypes.c_uint], "glPolygonOffset": [ctypes.c_float] * 2,
    "glEnableClientState": [ctypes.c_uint], "glDisableClientState": [ctypes.c_uint],
    "glVertexPointer": [ctypes.c_int, ctypes.c_uint, ctypes.c_int, ctypes.c_void_p],
    "glNormalPointer": [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p],
    "glColorPointer": [ctypes.c_int, ctypes.c_uint, ctypes.c_int, ctypes.c_void_p],
    "glDrawElements": [ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p],
    "glDrawArrays": [ctypes.c_uint, ctypes.c_int, ctypes.c_int],
    "glGetString": [ctypes.c_uint], "glFinish": [], "glGetError": [], "glCullFace": [ctypes.c_uint],
}


class GL:
    """The fixed-function OpenGL 1.1 entry points this renderer uses, from the system libGL."""

    def __init__(self, lib=None):
        if lib is None:
            if sys.platform == "win32":
                lib = ctypes.WinDLL("opengl32")
            elif sys.platform == "darwin":
                lib = ctypes.CDLL("/System/Library/Frameworks/OpenGL.framework/OpenGL")
            else:
                lib = ctypes.CDLL(ctypes.util.find_library("GL") or "libGL.so.1")
        self.lib = lib
        for name, args in _SIGNATURES.items():
            fn = getattr(lib, name)
            fn.argtypes = args
            fn.restype = {"glGetString": ctypes.c_char_p, "glGenLists": ctypes.c_uint,
                          "glGetError": ctypes.c_uint}.get(name)
            setattr(self, name, fn)

    def info(self) -> dict:
        def s(e):
            v = self.glGetString(e)
            return v.decode(errors="replace") if v else ""
        return {"version": s(GL_VERSION), "renderer": s(GL_RENDERER), "vendor": s(GL_VENDOR)}


def _ptr(a: np.ndarray):
    return ctypes.c_void_p(a.ctypes.data)


# ----------------------------------------------------------------------------- unit geometry
def unit_sphere(res: int):
    """Latitude/longitude sphere of radius 1: vertices (= normals) and triangle indices."""
    nlat, nlon = max(res // 2, 3), max(res, 4)
    th = np.linspace(0, np.pi, nlat + 1)
    ph = np.linspace(0, 2 * np.pi, nlon + 1)
    t, p = np.meshgrid(th, ph, indexing="ij")
    v = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1).reshape(-1, 3)
    i = np.arange(nlat)[:, None] * (nlon + 1) + np.arange(nlon)[None, :]
    a, b, c, d = i, i + 1, i + nlon + 1, i + nlon + 2
    faces = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    return np.ascontiguousarray(v), np.ascontiguousarray(faces.astype(np.uint32))


def unit_cylinder(res: int, cone: bool = False):
    """Open cylinder (or cone) of radius 1 from z=0 to z=1: vertices, normals, triangle indices."""
    n = max(res, 4)
    ang = np.linspace(0, 2 * np.pi, n + 1)
    ring = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], -1)
    if cone:  # side normals tilted 45 degrees for a cone with height = radius
        top = np.zeros_like(ring) + [0, 0, 1]
        normals = np.vstack([ring + [0, 0, 1], ring + [0, 0, 1]]) / np.sqrt(2)
        verts = np.vstack([ring, top])
    else:
        verts = np.vstack([ring, ring + [0, 0, 1]])
        normals = np.vstack([ring, ring])
    k = np.arange(n)
    faces = np.concatenate([np.stack([k, k + 1, k + n + 1], -1), np.stack([k + 1, k + n + 2, k + n + 1], -1)])
    return np.ascontiguousarray(verts), np.ascontiguousarray(normals), np.ascontiguousarray(faces.astype(np.uint32))


def unit_disk(res: int, top: bool):
    """Flat cap of the unit cylinder: a disk of radius 1 at z=1 facing +z (``top``) or at z=0 facing -z."""
    n = max(res, 4)
    ang = np.linspace(0, 2 * np.pi, n + 1)
    z = 1.0 if top else 0.0
    verts = np.vstack([np.stack([np.cos(ang), np.sin(ang), np.full_like(ang, z)], -1), [[0, 0, z]]])
    normals = np.tile([0.0, 0.0, 1.0 if top else -1.0], (len(verts), 1))
    k = np.arange(n)
    faces = np.stack([np.full(n, n + 1), k, k + 1] if top else [np.full(n, n + 1), k + 1, k], -1)
    return np.ascontiguousarray(verts), np.ascontiguousarray(normals), np.ascontiguousarray(faces.astype(np.uint32))


def align_matrices(start, end, radius):
    """Column-major 4x4 matrices taking the unit cylinder (z from 0 to 1, radius 1) to each segment."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    axis = end - start
    length = np.linalg.norm(axis, axis=1)
    z = axis / np.where(length > 0, length, 1)[:, None]
    helper = np.where(np.abs(z[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x, axis=1)[:, None]
    y = np.cross(z, x)
    r = np.broadcast_to(np.asarray(radius, float), (len(start),))
    m = np.zeros((len(start), 4, 4))
    m[:, 0, :3] = x * r[:, None]        # columns of the matrix are rows of this (column-major)
    m[:, 1, :3] = y * r[:, None]
    m[:, 2, :3] = z * length[:, None]
    m[:, 3, :3] = start
    m[:, 3, 3] = 1
    return np.ascontiguousarray(m)


# ----------------------------------------------------------------------------- camera
class CameraPlotter:
    """The few PyVista ``Plotter`` members the GUI uses, on a bare ``vtkCamera``.

    ``setup_camera``, ``camera_state``/``apply_camera_state`` and the viewport's view buttons
    then work unchanged. ``bounds`` is set by the renderer from the scene."""

    def __init__(self, on_render=None):
        import pyvista as pv

        self.camera = pv.Camera()
        self.bounds = np.array([-1.0, 1, -1, 1, -1, 1])
        self._on_render = on_render or (lambda: None)
        self.renderer = None

    def render(self):
        self._on_render()

    def enable_parallel_projection(self):
        self.camera.parallel_projection = True

    def disable_parallel_projection(self):
        self.camera.parallel_projection = False

    def reset_camera(self, render=False):
        """Like ``vtkRenderer::ResetCamera``: look at the bounds' center from the current direction."""
        b = self.bounds
        center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        radius = 0.5 * np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]) or 1.0
        cam = self.camera
        d = np.asarray(cam.GetDirectionOfProjection(), float)
        angle = np.radians(cam.view_angle)
        dist = radius / np.sin(angle / 2)
        cam.focal_point = tuple(center)
        cam.position = tuple(center - d * dist)
        cam.parallel_scale = radius
        self.reset_camera_clipping_range()

    def reset_camera_clipping_range(self):
        b = self.bounds
        center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        radius = 0.5 * np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]) or 1.0
        cam = self.camera
        d = np.linalg.norm(np.asarray(cam.position) - center)
        near = max(d - radius * 1.5, radius * 1e-3)
        self.camera.clipping_range = (near, d + radius * 1.5)

    def matrices(self, aspect: float):
        """(projection, modelview) as column-major float64 arrays for glLoadMatrixd."""
        cam = self.camera
        self.reset_camera_clipping_range()
        proj = _vtk_to_np(cam.GetProjectionTransformMatrix(aspect, -1, 1))
        view = _vtk_to_np(cam.GetViewTransformMatrix())
        return np.ascontiguousarray(proj.T), np.ascontiguousarray(view.T)

    def close(self):
        pass


def _vtk_to_np(m) -> np.ndarray:
    return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])


def pick_ray(plotter: CameraPlotter, x: float, y: float, width: int, height: int):
    """World-space ray (origin, unit direction) under window pixel (x, y), y from the top."""
    proj, view = plotter.matrices(width / max(height, 1))
    inv = np.linalg.inv(proj.T @ view.T)
    ndc_x, ndc_y = 2 * x / width - 1, 1 - 2 * y / height
    near = inv @ [ndc_x, ndc_y, -1, 1]
    far = inv @ [ndc_x, ndc_y, 1, 1]
    near, far = near[:3] / near[3], far[:3] / far[3]
    d = far - near
    return near, d / np.linalg.norm(d)


def ray_spheres(origin, direction, centers, radii):
    """Index of the first sphere hit by the ray, or None."""
    oc = centers - origin
    t = oc @ direction
    d2 = np.einsum("ij,ij->i", oc, oc) - t * t
    hit = d2 <= radii * radii
    if not hit.any():
        return None
    t_hit = np.where(hit, t - np.sqrt(np.maximum(radii * radii - d2, 0)), np.inf)
    i = int(np.argmin(t_hit))
    return i if np.isfinite(t_hit[i]) else None


# ----------------------------------------------------------------------------- renderer
class LegacyRenderer:
    """Compiles a :class:`ewald.scene.Scene` into OpenGL 1.x display lists and draws them.

    Every call needs a current GL context. ``compile`` runs once per frame or style change (the
    geometry is sent to the X server then); ``draw`` runs on every camera change and only sends
    matrices and one ``glCallList`` per layer group."""

    POINTS_ABOVE = 400_000   # more spheres than this are drawn as round points

    def __init__(self, gl: GL | None = None):
        self.gl = gl or GL()
        self.background = (1.0, 1.0, 1.0)
        self.ink = (0.0, 0.0, 0.0)
        self.sphere_resolution = 16
        self.light = None                # key light (ewald.render.light_params); None: the head light
        self.flat = False                # the flat (ASE) style: unlit atoms and sticks in dark outlines
        self._lists: list[int] = []      # opaque display lists
        self._glass: list[int] = []      # transparent display lists, drawn last
        self._overlay: list[int] = []    # picks / measurements, drawn without depth test
        self._line_width: dict = {}      # line lists: ("pixels", Lines.width) or ("world", Angstrom), set per draw
        self._unit: dict = {}
        self.centers = np.zeros((0, 3))  # for picking: sphere centers, radii and particle indices
        self.radii = np.zeros(0)
        self.index = np.zeros(0, int)
        self.bounds = np.array([-1.0, 1, -1, 1, -1, 1])

    # ---------------------------------------------------------------- helpers
    def _new_list(self, target):
        gl = self.gl
        n = gl.glGenLists(1)
        gl.glNewList(n, GL_COMPILE)
        target.append(n)
        return n

    def _free(self, lists):
        for n in lists:
            self.gl.glDeleteLists(n, 1)
        lists.clear()

    def _unit_list(self, key, make):
        """A display list for a unit shape, compiled once per context."""
        if key not in self._unit:
            gl = self.gl
            verts, normals, faces = make()
            n = gl.glGenLists(1)
            gl.glNewList(n, GL_COMPILE)
            self._draw_arrays(verts, faces, normals)
            gl.glEndList()
            self._unit[key] = n
        return self._unit[key]

    def _draw_arrays(self, verts, faces, normals=None, colors=None, mode=GL_TRIANGLES):
        gl = self.gl
        v = np.ascontiguousarray(verts, np.float64)
        gl.glEnableClientState(GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, GL_DOUBLE, 0, _ptr(v))
        keep = [v]
        if normals is not None:
            nrm = np.ascontiguousarray(normals, np.float64)
            keep.append(nrm)
            gl.glEnableClientState(GL_NORMAL_ARRAY)
            gl.glNormalPointer(GL_DOUBLE, 0, _ptr(nrm))
        if colors is not None:
            col = np.ascontiguousarray(colors, np.float64)
            keep.append(col)
            gl.glEnableClientState(GL_COLOR_ARRAY)
            gl.glColorPointer(col.shape[1], GL_DOUBLE, 0, _ptr(col))
        if faces is None:
            gl.glDrawArrays(mode, 0, len(v))
        else:
            idx = np.ascontiguousarray(faces, np.uint32).ravel()
            keep.append(idx)
            gl.glDrawElements(mode, len(idx), GL_UNSIGNED_INT, _ptr(idx))
        for state in (GL_VERTEX_ARRAY, GL_NORMAL_ARRAY, GL_COLOR_ARRAY):
            gl.glDisableClientState(state)
        return keep

    def sphere_list(self, res):
        def make():
            v, f = unit_sphere(res)
            return v, v, f
        return self._unit_list(("sphere", res), make)

    def cylinder_list(self, res, cone=False):
        return self._unit_list(("cone" if cone else "cylinder", res), lambda: unit_cylinder(res, cone))

    def disk_list(self, res, top):
        return self._unit_list(("disk", res, top), lambda: unit_disk(res, top))

    # ---------------------------------------------------------------- compile
    def compile(self, scene, *, sphere_resolution=None, background=None, ink=None, light=None, flat=False):
        """Replace the drawn geometry with ``scene`` (needs a current context); ``light`` is the view's key
        light (None: the default key light); ``flat`` draws the flat (ASE) style as the other renderers do."""
        self.light = light
        self.flat = bool(flat)
        if background is not None:
            self.background = tuple(background)
        if ink is not None:
            self.ink = tuple(ink)
        if sphere_resolution:
            self.sphere_resolution = int(sphere_resolution)
        self._free(self._lists)
        self._free(self._glass)
        self._line_width.clear()
        centers, radii, index, lo, hi = [], [], [], [], []
        for layer in scene.layers:
            prim, kind = layer.primitive, layer.kind
            if kind == "spheres" and prim.count:
                self._compile_spheres(prim)
                if layer.name == "atoms" or not len(centers):
                    centers.append(prim.centers)
                    radii.append(prim.sizes())
                    index.append(prim.index)
                pts = prim.centers
            elif kind == "cylinders" and prim.count:
                self._compile_cylinders(prim.start, prim.end, prim.radius, prim.rgb, prim.alpha,
                                        flat_caps=(prim.start_cap == 1, prim.end_cap == 1), outline=self.flat)
                center, _, radius, rgb, alpha, _ = prim.caps("round")
                if len(center):  # a sphere of the tube radius shows as a hemisphere past the end
                    from ..scene import Spheres

                    self._compile_spheres(Spheres(center, radius, rgb, alpha))
                pts = np.vstack([prim.start, prim.end])
            elif kind == "lines" and prim.count:
                self._compile_lines(prim)
                pts = prim.points
            elif kind == "mesh" and prim.face_count:
                self._compile_mesh(prim)
                pts = prim.vertices
            elif kind == "arrows" and prim.count:
                self._compile_arrows(prim)
                pts = np.vstack([prim.base, prim.tip])
            else:
                continue
            if len(pts):
                lo.append(pts.min(0))
                hi.append(pts.max(0))
        self.centers = np.vstack(centers) if centers else np.zeros((0, 3))
        self.radii = np.concatenate(radii) if radii else np.zeros(0)
        self.index = np.concatenate(index) if index else np.zeros(0, int)
        if lo:
            pad = float(self.radii.max()) if len(self.radii) else 0.0
            lo, hi = np.min(lo, 0) - pad, np.max(hi, 0) + pad
            self.bounds = np.array([lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])

    def _sphere_res(self, n):
        """Fewer triangles per sphere for big systems: the list stays small on the X server."""
        res = self.sphere_resolution
        if n > 20_000:
            res = min(res, 10)
        if n > 100_000:
            res = min(res, 6)
        return max(res, 4)

    def _compile_spheres(self, s):
        gl = self.gl
        rgb, rad, alpha = s.colors(), s.sizes(), np.asarray(s.alpha, float)
        keep = alpha > 0
        if not keep.any():
            return
        if s.count > self.POINTS_ABOVE:  # round points, one glDrawArrays
            self._new_list(self._lists)
            gl.glDisable(GL_LIGHTING)
            gl.glEnable(GL_POINT_SMOOTH)
            gl.glPointSize(3.0)
            self._draw_arrays(s.centers[keep], None, colors=rgb[keep], mode=GL_POINTS)
            gl.glEnable(GL_LIGHTING)
            gl.glEndList()
            return
        unit = self.sphere_list(self._sphere_res(s.count))
        for glass in (False, True):
            sel = keep & ((alpha < 1) if glass else (alpha >= 1))
            if not sel.any():
                continue
            self._new_list(self._glass if glass else self._lists)
            key = np.round(np.column_stack([rgb[sel], alpha[sel]]), 4)
            order = np.lexsort(key.T[::-1])
            c, r, k = s.centers[sel][order], rad[sel][order], key[order]
            if self.flat:  # unlit discs inside a dark rim (5 % of the radius, as the other renderers)
                gl.glDisable(GL_LIGHTING)
                self._spheres(unit, c, r * 0.95, k)
                gl.glEnable(GL_CULL_FACE)
                gl.glCullFace(GL_FRONT)          # the inverted hull: only its far side shows, past the rim
                self._spheres(unit, c, r, np.column_stack([np.full((len(c), 3), FLAT_OUTLINE), k[:, 3]]))
                gl.glDisable(GL_CULL_FACE)
                gl.glEnable(GL_LIGHTING)
            else:
                self._spheres(unit, c, r, k)
            gl.glEndList()

    def _spheres(self, unit, c, r, k):
        """Unit-sphere calls at centers ``c`` scaled by ``r`` in RGBA colors ``k`` (inside a list)."""
        gl = self.gl
        last = None
        for i in range(len(c)):
            if last is None or not np.array_equal(k[i], last):
                gl.glColor4d(*k[i])
                last = k[i]
            gl.glPushMatrix()
            gl.glTranslated(*c[i])
            gl.glScaled(r[i], r[i], r[i])
            gl.glCallList(unit)
            gl.glPopMatrix()

    def _compile_cylinders(self, start, end, radius, rgb, alpha, cone=False, res=None, flat_caps=None, outline=False):
        """Cylinders (or cones) as display lists; ``flat_caps`` (start mask, end mask) closes those ends;
        ``outline``: unlit, in a dark hull 30 % of the radius wider (the flat style)."""
        gl = self.gl
        alpha = np.broadcast_to(np.asarray(alpha, float), (len(start),))
        res = res or max(self._sphere_res(len(start)) // 2 + 2, 6)
        unit = self.cylinder_list(res, cone)
        no = np.zeros(len(start), bool)
        caps = [(np.broadcast_to(m, (len(start),)), self.disk_list(res, top))
                for m, top in zip(flat_caps or (no, no), (False, True)) if np.any(m)]
        mats = align_matrices(start, end, radius)
        hull = align_matrices(start, end, (1.0 + STICK_OUTLINE) * np.asarray(radius, float)) if outline else None
        rgb = np.asarray(rgb, float)
        base = mats.ctypes.data
        for glass in (False, True):
            sel = np.flatnonzero((alpha > 0) & ((alpha < 1) if glass else (alpha >= 1)))
            if not len(sel):
                continue
            self._new_list(self._glass if glass else self._lists)
            if outline:
                gl.glDisable(GL_LIGHTING)
                gl.glEnable(GL_CULL_FACE)
                gl.glCullFace(GL_FRONT)
                for i in sel:
                    gl.glColor4d(FLAT_OUTLINE, FLAT_OUTLINE, FLAT_OUTLINE, alpha[i])
                    gl.glPushMatrix()
                    gl.glMultMatrixd(ctypes.c_void_p(hull.ctypes.data + int(i) * 128))
                    gl.glCallList(unit)
                    gl.glPopMatrix()
                gl.glDisable(GL_CULL_FACE)
            last = None
            for i in sel:
                col = (rgb[i, 0], rgb[i, 1], rgb[i, 2], alpha[i])
                if col != last:
                    gl.glColor4d(*col)
                    last = col
                gl.glPushMatrix()
                gl.glMultMatrixd(ctypes.c_void_p(base + int(i) * 128))
                gl.glCallList(unit)
                for mask, disk in caps:
                    if mask[i]:
                        gl.glCallList(disk)
                gl.glPopMatrix()
            if outline:
                gl.glEnable(GL_LIGHTING)
            gl.glEndList()

    def _compile_lines(self, lines):
        gl = self.gl
        seg = lines.segments()
        if not len(seg):
            return
        colors = lines.vertex_colors(self.ink)[seg.ravel()]   # per point: color maps blend along a piece
        n = self._new_list(self._lists)
        self._line_width[n] = ("pixels", float(lines.width))
        gl.glDisable(GL_LIGHTING)
        self._draw_arrays(lines.points[seg.ravel()], None, colors=colors, mode=GL_LINES)
        gl.glEnable(GL_LIGHTING)
        gl.glEndList()

    def _compile_mesh(self, mesh):
        gl = self.gl
        v, f = mesh.vertices, mesh.faces
        tri = v[f]                                    # F x 3 x 3, flat-shaded unless normals are given
        if "Normal" in mesh.vertex_properties:
            normals = np.asarray(mesh.vertex_properties["Normal"], float)[f]
        else:
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
            normals = np.repeat(n[:, None], 3, axis=1)
        if "Color" in mesh.vertex_properties:
            rgb = np.asarray(mesh.vertex_properties["Color"], float)[f]
        else:  # the face colors, with the other renderers' default grey
            rgb = np.repeat(np.clip(mesh.face_colors(), 0, 1)[:, None], 3, axis=1)
        # the VTK surface material (ambient 0.3, diffuse 0.7, specular 0.2; color-mapped slices 0.6, 0.45, 0, so
        # they read true) from the lights' diffuse 0.8: colors scaled to the diffuse part, ambient light to match
        mapped = "Color" in mesh.vertex_properties
        amb, dif, spec = (0.6, 0.45, 0.0) if mapped else (0.3, 0.7, 0.2)
        alpha = 1.0 - mesh.transparency
        colors = np.concatenate([rgb * (dif / 0.8), np.full(rgb.shape[:2] + (1,), alpha)], -1).reshape(-1, 4)
        self._new_list(self._glass if alpha < 1 else self._lists)
        f4 = ctypes.c_float * 4
        self._mesh_light = [f4(*([amb * 0.8 / dif] * 3), 1), f4(*([spec / 0.3] * 3), 1)]
        gl.glLightModelfv(GL_LIGHT_MODEL_AMBIENT, self._mesh_light[0])
        gl.glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, self._mesh_light[1])
        self._draw_arrays(tri.reshape(-1, 3), None, normals.reshape(-1, 3), colors)
        gl.glLightModelfv(GL_LIGHT_MODEL_AMBIENT, self._keep[4])
        gl.glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, self._keep[5])
        gl.glEndList()
        if mesh.edges is not None and len(mesh.edges):
            n = self._new_list(self._lists)
            self._line_width[n] = ("world", float(mesh.edge_width))
            gl.glDisable(GL_LIGHTING)
            ec = np.broadcast_to(np.asarray(mesh.edge_color, float), (len(mesh.edges) * 2, 3))
            self._draw_arrays(v[mesh.edges.ravel()], None, colors=ec, mode=GL_LINES)
            gl.glEnable(GL_LIGHTING)
            gl.glEndList()

    def _compile_arrows(self, a):
        neck = a.neck
        self._compile_cylinders(a.base, neck, a.radius, a.rgb, a.alpha)
        self._compile_cylinders(neck, a.tip, a.head_radius, a.rgb, a.alpha, cone=True)

    def set_overlay(self, centers=None, radii=None, line=None, rgb=(1.0, 0.75, 0.2)):
        """Picked atoms (wire spheres) and a measurement polyline, drawn over everything."""
        gl = self.gl
        self._free(self._overlay)
        if centers is None or not len(centers):
            return
        self._new_list(self._overlay)
        gl.glDisable(GL_LIGHTING)
        gl.glColor4d(rgb[0], rgb[1], rgb[2], 1.0)
        gl.glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        unit = self.sphere_list(12)
        for c, r in zip(np.asarray(centers, float), np.asarray(radii, float)):
            gl.glPushMatrix()
            gl.glTranslated(*c)
            s = float(r) * 1.18
            gl.glScaled(s, s, s)
            gl.glCallList(unit)
            gl.glPopMatrix()
        gl.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        if line is not None and len(line) >= 2:
            gl.glLineWidth(3.0)
            pts = np.asarray(line, float)
            seg = np.column_stack([np.arange(len(pts) - 1), np.arange(1, len(pts))]).ravel()
            self._draw_arrays(pts[seg], None, mode=GL_LINES)
        gl.glEnable(GL_LIGHTING)
        gl.glEndList()

    # ---------------------------------------------------------------- draw
    def setup_state(self):
        """Lights and material state, once per context."""
        gl = self.gl
        gl.glEnable(GL_DEPTH_TEST)
        gl.glEnable(GL_LIGHTING)
        gl.glEnable(GL_LIGHT0)
        gl.glEnable(GL_LIGHT1)
        gl.glEnable(GL_NORMALIZE)
        gl.glEnable(GL_COLOR_MATERIAL)
        gl.glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        gl.glShadeModel(GL_SMOOTH)
        gl.glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1)
        f4 = lambda *v: (ctypes.c_float * 4)(*v)  # noqa: E731
        self._keep = [None, f4(0.8, 0.8, 0.8, 1), f4(0.3, 0.3, 0.3, 1),
                      f4(0.0, 0.0, 0.0, 1), f4(0.25, 0.25, 0.25, 1), f4(1.0, 1.0, 1.0, 1)]
        gl.glLightModelfv(GL_LIGHT_MODEL_AMBIENT, self._keep[4])   # ambient 0.25, diffuse 0.8, specular 0.3 x 40
        gl.glLightfv(GL_LIGHT0, GL_AMBIENT, self._keep[3])
        gl.glLightfv(GL_LIGHT0, GL_DIFFUSE, self._keep[1])
        gl.glLightfv(GL_LIGHT0, GL_SPECULAR, self._keep[2])
        gl.glLightfv(GL_LIGHT1, GL_AMBIENT, self._keep[3])
        gl.glLightfv(GL_LIGHT1, GL_SPECULAR, self._keep[3])
        gl.glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, self._keep[5])
        gl.glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 40.0)
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(GL_LINE_SMOOTH)

    def draw(self, plotter: CameraPlotter, width: int, height: int, axes: bool = True):
        gl = self.gl
        plotter.bounds = self.bounds
        gl.glViewport(0, 0, width, height)
        gl.glClearColor(*self.background, 1.0)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        proj, view = plotter.matrices(width / max(height, 1))
        gl.glMatrixMode(GL_PROJECTION)
        gl.glLoadMatrixd(_ptr(proj))
        gl.glMatrixMode(GL_MODELVIEW)
        gl.glLoadIdentity()
        from ..render import FILL_LIGHT, light_params

        d, k = light_params(self.light)  # the key light of every renderer (D51), eye space, plus a head light
        f4 = ctypes.c_float * 4
        key, fill, spec = 0.8 * k, 0.8 * FILL_LIGHT * k, 0.3 * k   # the VTK mesh material
        self._light_now = [f4(*d, 0.0), f4(key, key, key, 1), f4(spec, spec, spec, 1),
                           f4(0.0, 0.0, 1.0, 0.0), f4(fill, fill, fill, 1)]
        gl.glLightfv(GL_LIGHT0, GL_POSITION, self._light_now[0])
        gl.glLightfv(GL_LIGHT0, GL_DIFFUSE, self._light_now[1])
        gl.glLightfv(GL_LIGHT0, GL_SPECULAR, self._light_now[2])
        gl.glLightfv(GL_LIGHT1, GL_POSITION, self._light_now[3])
        gl.glLightfv(GL_LIGHT1, GL_DIFFUSE, self._light_now[4])
        gl.glLoadMatrixd(_ptr(view))
        gl.glEnable(GL_DEPTH_TEST)
        if self._line_width:  # D51: lines scale with the view height, mesh edges with the scene
            from ..render import camera_state, pixel_size
            from ..scene import line_pixels

            px = pixel_size(camera_state(plotter), height)
        for n in self._lists:
            if n in self._line_width:  # smooth (blended) lines, as wide as in the other renderers
                kind, value = self._line_width[n]
                gl.glLineWidth(line_pixels(value, height) if kind == "pixels" else max(1.0, value / px))
                gl.glEnable(GL_BLEND)
                gl.glCallList(n)
                gl.glDisable(GL_BLEND)
                continue
            gl.glCallList(n)
        if self._glass:
            gl.glEnable(GL_BLEND)
            gl.glDepthMask(0)
            for n in self._glass:
                gl.glCallList(n)
            gl.glDepthMask(1)
            gl.glDisable(GL_BLEND)
        if self._overlay:
            gl.glDisable(GL_DEPTH_TEST)
            for n in self._overlay:
                gl.glCallList(n)
            gl.glEnable(GL_DEPTH_TEST)
        if axes:
            self._draw_axes(view, width, height)

    def _draw_axes(self, view, width, height):
        """x/y/z triad (red/green/blue) in the lower-left corner, rotated with the view."""
        gl = self.gl
        size = max(min(width, height) // 6, 40)
        gl.glViewport(0, 0, size, size)
        gl.glClear(GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glMatrixMode(GL_MODELVIEW)
        rot = np.array(view, dtype=float).reshape(4, 4)
        rot[3, :3] = 0  # column-major storage: the translation is the last stored column
        gl.glLoadMatrixd(_ptr(rot))
        gl.glScaled(0.8, 0.8, 0.8)
        gl.glDisable(GL_LIGHTING)
        gl.glLineWidth(2.0)
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1.0]])
        cols = np.array([[0.9, 0.2, 0.2]] * 2 + [[0.2, 0.75, 0.25]] * 2 + [[0.25, 0.45, 0.95]] * 2)
        self._draw_arrays(pts, None, colors=cols, mode=GL_LINES)
        gl.glEnable(GL_LIGHTING)
        gl.glViewport(0, 0, width, height)
