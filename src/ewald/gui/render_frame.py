"""The Render frame on the 3D view: the part of the view the picture will show, outside it dimmed.

The frame keeps the picture's aspect ratio and a fixed place on screen, so zooming or turning the view
changes what is inside it, and the picture is exactly what the frame covers (:mod:`ewald.framing`).
Drag a corner to resize it, a side's middle handle to change the picture's aspect (its width or height
in pixels), and the frame's edge to move it; anywhere else the mouse turns and zooms the view as usual.
Drawn as VTK 2D actors in the viewport's renderer; the OpenGL 1.x viewport has no frame yet.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui

from .. import framing
from .theme import TOKENS

GRAB = 7          # pixels around an edge or handle that grab it
HANDLE = 4        # half size of the square handles, pixels
MIN_SIZE = 24     # smallest frame side, pixels
_CURSORS = {"move": QtCore.Qt.SizeAllCursor, "nw": QtCore.Qt.SizeFDiagCursor, "se": QtCore.Qt.SizeFDiagCursor,
            "ne": QtCore.Qt.SizeBDiagCursor, "sw": QtCore.Qt.SizeBDiagCursor, "n": QtCore.Qt.SizeVerCursor,
            "s": QtCore.Qt.SizeVerCursor, "e": QtCore.Qt.SizeHorCursor, "w": QtCore.Qt.SizeHorCursor}


def hit_test(rect, x: float, y: float) -> str | None:
    """What of a frame ``(left, top, width, height)`` the point (x, y) grabs: a corner ('nw', 'ne',
    'sw', 'se'), a side's middle handle ('n', 's', 'e', 'w'), the edge ('move') or nothing."""
    left, top, w, h = rect
    right, bottom = left + w, top + h
    near_x = {"w": abs(x - left) <= GRAB, "e": abs(x - right) <= GRAB}
    near_y = {"n": abs(y - top) <= GRAB, "s": abs(y - bottom) <= GRAB}
    for vy, ny in near_y.items():
        for vx, nx in near_x.items():
            if ny and nx:
                return vy + vx
    cx, cy = left + w / 2, top + h / 2
    if abs(x - cx) <= GRAB + HANDLE:
        for side, near in near_y.items():
            if near:
                return side
    if abs(y - cy) <= GRAB + HANDLE:
        for side, near in near_x.items():
            if near:
                return side
    inside_x, inside_y = left - GRAB <= x <= right + GRAB, top - GRAB <= y <= bottom + GRAB
    if (inside_y and any(near_x.values())) or (inside_x and any(near_y.values())):
        return "move"
    return None


def drag_rect(rect, part: str, dx: float, dy: float, aspect: float) -> tuple[tuple, float]:
    """The frame after dragging ``part`` by (dx, dy) pixels, and its aspect (width / height): corners
    keep the aspect with the opposite corner fixed, sides change it, 'move' shifts the frame."""
    left, top, w, h = rect
    if part == "move":
        return (left + dx, top + dy, w, h), aspect
    if part in ("e", "w"):
        nw = max(MIN_SIZE, w + (dx if part == "e" else -dx))
        return ((left if part == "e" else left + w - nw), top, nw, h), nw / h
    if part in ("n", "s"):
        nh = max(MIN_SIZE, h + (dy if part == "s" else -dy))
        return (left, (top if part == "s" else top + h - nh), w, nh), w / nh
    sx = dx if "e" in part else -dx
    sy = dy if "s" in part else -dy
    nh = max(MIN_SIZE / max(aspect, 1e-9), MIN_SIZE, h + max(sy, sx / aspect))  # the larger pull wins
    nw = nh * aspect
    nl = left if "e" in part else left + w - nw
    nt = top if "s" in part else top + h - nh
    return (nl, nt, nw, nh), aspect


def keep_inside(rect, view_size, part: str):
    """A dragged frame kept inside the view: moved back when moved out, clipped when resized out."""
    left, top, w, h = rect
    vw, vh = view_size
    if part == "move":
        return min(max(left, 0.0), max(vw - w, 0.0)), min(max(top, 0.0), max(vh - h, 0.0)), w, h
    if part in ("e", "w", "n", "s"):
        l2, t2 = max(left, 0.0), max(top, 0.0)
        r2, b2 = min(left + w, vw), min(top + h, vh)
        return l2, t2, max(r2 - l2, MIN_SIZE), max(b2 - t2, MIN_SIZE)
    # corners keep the aspect: shrink about the fixed corner until it fits
    scale = 1.0
    if "e" in part:
        scale = min(scale, (vw - left) / w)
    else:
        scale = min(scale, (left + w) / w)
    if "s" in part:
        scale = min(scale, (vh - top) / h)
    else:
        scale = min(scale, (top + h) / h)
    if scale >= 1.0:
        return rect
    nw, nh = w * scale, h * scale
    return (left if "e" in part else left + w - nw), (top if "s" in part else top + h - nh), nw, nh


class RenderFrame(QtCore.QObject):
    """The frame overlay of one VTK viewport. ``region`` None follows the largest frame that fits."""
    regionChanged = QtCore.Signal(object)   # (x, y, height) after a drag
    aspectChanged = QtCore.Signal(float, str)  # width / height after dragging a side, and the side

    def __init__(self, viewport, parent=None):
        super().__init__(parent)
        self.viewport = viewport
        self.aspect = 4 / 3
        self.region = None
        self.visible = False
        self._actors = None
        self._drag = None            # (part, press position, rect at press)
        self._observer = None
        widget = getattr(viewport.plotter, "interactor", None)
        self.widget = widget
        if widget is not None:
            widget.installEventFilter(self)
            widget.setMouseTracking(True)

    # ------------------------------------------------------------------ state
    def view_size(self) -> tuple[int, int]:
        w = self.widget
        return (max(w.width(), 1), max(w.height(), 1)) if w is not None else (800, 600)

    def effective_region(self) -> tuple[float, float, float]:
        """The region the picture shows: the set one, or the largest centered frame that fits."""
        return framing.as_region(self.region) if self.region is not None else \
            framing.fit_region(self.view_size(), self.aspect)

    def rect(self):
        return framing.frame_rect(self.effective_region(), self.view_size(), self.aspect)

    def set_aspect(self, aspect: float):
        aspect = max(float(aspect), 1e-3)
        if abs(aspect - self.aspect) < 1e-9:
            return
        if self.region is not None:  # keep the frame inside the view when it gets wider
            x, y, h = self.region
            w, vh = self.view_size()
            h = min(h, (w / vh) / aspect)
            self.region = (x, y, h)
        self.aspect = aspect
        self.update()

    def set_region(self, region):
        self.region = None if region is None else framing.as_region(region)
        self.update()

    def set_visible(self, on: bool):
        if on == self.visible:
            return
        self.visible = on
        if not on and self.widget is not None:
            self.widget.unsetCursor()
        self.update()

    # ------------------------------------------------------------------ drawing
    def _ensure_actors(self):
        if self._actors is not None:
            return self._actors
        import vtk

        ren = self.viewport.plotter.renderer
        actors = {}
        for name, opacity in (("dim", 0.42), ("line", 1.0), ("handles", 1.0)):
            poly = vtk.vtkPolyData()
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputData(poly)
            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetOpacity(opacity)
            actor.SetPickable(False)
            ren.AddViewProp(actor)
            actors[name] = (actor, poly)
        actors["line"][0].GetProperty().SetLineWidth(1.5)
        self._actors = actors
        self._observer = ren.AddObserver("StartEvent", lambda *_: self._layout())
        return actors

    def update(self):
        if self._actors is None and not self.visible:
            return
        actors = self._ensure_actors()
        for actor, _ in actors.values():
            actor.SetVisibility(self.visible)
        self.viewport.render()

    def _layout(self):
        """Place the frame for the current window size (called before each render)."""
        if not self.visible or self._actors is None:
            return
        import vtk

        ren = self.viewport.plotter.renderer
        W, H = ren.GetSize()
        if W <= 1 or H <= 1:
            return
        ratio = W / max(self.view_size()[0], 1)
        left, top, fw, fh = (v * ratio for v in self.rect())
        x0, x1, y1, y0 = left, left + fw, H - top, H - top - fh   # VTK display y goes up
        bg = QtGui.QColor(self.viewport.style.background)
        dark = bg.lightnessF() < 0.5
        dim = (0.0, 0.0, 0.0) if not dark else (0.55, 0.57, 0.6)   # black on black would not show
        accent = QtGui.QColor(TOKENS["accent"])
        acc = (accent.redF(), accent.greenF(), accent.blueF())

        def quads(poly, rects):
            pts, cells = vtk.vtkPoints(), vtk.vtkCellArray()
            for a, b, c, d in rects:
                ids = [pts.InsertNextPoint(px, py, 0) for px, py in ((a, b), (c, b), (c, d), (a, d))]
                cells.InsertNextCell(4, ids)
            poly.SetPoints(pts)
            poly.SetPolys(cells)
            poly.Modified()

        dim_actor, dim_poly = self._actors["dim"]
        quads(dim_poly, [(0, 0, W, max(y0, 0)), (0, min(y1, H), W, H),
                         (0, max(y0, 0), max(x0, 0), min(y1, H)), (min(x1, W), max(y0, 0), W, min(y1, H))])
        dim_actor.GetProperty().SetColor(*dim)
        dim_actor.GetProperty().SetOpacity(0.3 if dark else 0.38)
        line_actor, line_poly = self._actors["line"]
        pts, lines = vtk.vtkPoints(), vtk.vtkCellArray()
        ids = [pts.InsertNextPoint(px, py, 0) for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
        lines.InsertNextCell(len(ids), ids)
        line_poly.SetPoints(pts)
        line_poly.SetLines(lines)
        line_poly.Modified()
        line_actor.GetProperty().SetColor(*acc)
        line_actor.GetProperty().SetLineWidth(1.5 * ratio)
        h_actor, h_poly = self._actors["handles"]
        s = HANDLE * ratio
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        quads(h_poly, [(px - s, py - s, px + s, py + s) for px, py in
                       ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (mx, y0), (mx, y1), (x0, my), (x1, my))])
        h_actor.GetProperty().SetColor(*acc)

    # ------------------------------------------------------------------ mouse
    def eventFilter(self, obj, e):
        if not self.visible:
            return False
        t = e.type()
        if t == QtCore.QEvent.MouseButtonPress and e.button() == QtCore.Qt.LeftButton:
            p = e.position()
            part = hit_test(self.rect(), p.x(), p.y())
            if part is None:
                return False
            self._drag = (part, (p.x(), p.y()), self.rect(), self.aspect)
            return True
        if t == QtCore.QEvent.MouseMove:
            p = e.position()
            if self._drag is None:
                part = hit_test(self.rect(), p.x(), p.y())
                if part is None:
                    obj.unsetCursor()
                    return False
                obj.setCursor(_CURSORS[part])
                return False
            part, (px, py), rect, aspect = self._drag
            new, new_aspect = drag_rect(rect, part, p.x() - px, p.y() - py, aspect)
            new = keep_inside(new, self.view_size(), part)
            if part in ("e", "w", "n", "s"):
                new_aspect = new[2] / new[3]
            self.region = framing.rect_region(new, self.view_size())
            if abs(new_aspect - self.aspect) > 1e-9:
                self.aspect = new_aspect
                self.aspectChanged.emit(new_aspect, part)
            self.viewport.render()
            return True
        if t == QtCore.QEvent.MouseButtonRelease and self._drag is not None and e.button() == QtCore.Qt.LeftButton:
            self._drag = None
            self.regionChanged.emit(self.region)
            return True
        return False

    def fit(self, data, padding: float = 0.06):
        """Move and zoom the view (same direction) so the structure fills the frame."""
        from ..render import apply_camera_state, camera_state

        if data is None or not data.count:
            return
        vp = self.viewport
        pts = np.asarray(data.positions, float)
        radii = np.asarray(vp._hover_radii_for(data), float) if hasattr(vp, "_hover_radii_for") else 0.0
        if vp.style.show_cell and getattr(data, "cell", None) is not None and data.cell.is_valid:
            m, o = np.asarray(data.cell.matrix, float), np.asarray(data.cell.origin, float)
            corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) @ m + o
            radii = np.concatenate([np.broadcast_to(radii, (len(pts),)), np.zeros(8)])
            pts = np.vstack([pts, corners])
        cam = framing.fit_camera(camera_state(vp.plotter), pts, self.effective_region(), self.aspect,
                                 padding, radii=radii)
        apply_camera_state(vp.plotter, cam)
        vp.render()
