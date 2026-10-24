"""The viewport on OpenGL 1.x (:mod:`ewald.gui.legacy_gl`), for SSH X11 forwarding.

Same toolbar, tools, picking and camera API as :class:`~ewald.gui.viewport.Viewport`; the 3D view
is a ``QOpenGLWindow`` drawing display lists instead of a VTK render window. It has no impostors,
FXAA, depth peeling, color bar or 3D text, which forwarded GLX could not do anyway.
"""
from __future__ import annotations

import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtOpenGL import QOpenGLWindow

from ..render import default_radii
from .legacy_gl import CameraPlotter, LegacyRenderer, pick_ray, ray_spheres
from .theme import TOKENS
from .viewport import EMPTY_HINT, Viewport, _is_dark, control_click


class GLView(QOpenGLWindow):
    """Draws the renderer's display lists; mouse input goes to the owning viewport."""

    def __init__(self, viewport: LegacyViewport):
        super().__init__(QOpenGLWindow.NoPartialUpdate)
        fmt = QtGui.QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setProfile(QtGui.QSurfaceFormat.NoProfile)
        fmt.setVersion(1, 1)       # fixed function only: what indirect GLX offers
        self.setFormat(fmt)
        self.viewport = viewport
        self.renderer: LegacyRenderer | None = None
        self.pending = None         # (scene, kwargs) waiting for the context
        self.pending_overlay = None
        self.info = {"version": "", "renderer": "", "vendor": ""}

    def initializeGL(self):
        self.renderer = LegacyRenderer()
        self.renderer.setup_state()
        self.info = self.renderer.gl.info()

    def paintGL(self):
        r, vp = self.renderer, self.viewport
        if self.pending is not None:
            scene, kw = self.pending
            self.pending = None
            r.compile(scene, **kw)
            vp.plotter.bounds = r.bounds
            if vp._first_pending:
                vp._first_pending = False
                vp._reset_camera_now()
        if self.pending_overlay is not None:
            r.set_overlay(*self.pending_overlay)
            self.pending_overlay = None
        t = time.perf_counter()
        ratio = self.devicePixelRatio()
        r.draw(vp.plotter, round(self.width() * ratio), round(self.height() * ratio), axes=vp.show_axes)
        vp._render_time = time.perf_counter() - t
        vp._camera_check()

    # mouse: VTK trackball-camera conventions (left rotate, middle/Shift+left pan, right/wheel zoom)
    def mousePressEvent(self, e):
        self.viewport._dragging = True
        self.viewport.hover_leave()
        self._last = self._press = e.position()
        self._button = e.button()
        self._shift = bool(e.modifiers() & QtCore.Qt.ShiftModifier)

    def mouseMoveEvent(self, e):
        if getattr(self, "_press", None) is None:  # no button down: the hover readout
            self.viewport.hover_move(e.position().x(), e.position().y(), self.width(), self.height())
            return
        pos = e.position()
        dx, dy = pos.x() - self._last.x(), pos.y() - self._last.y()
        self._last = pos
        cam, h = self.viewport.plotter.camera, max(self.height(), 1)
        if self._button == QtCore.Qt.LeftButton and not self._shift:
            cam.Azimuth(-dx * 0.4)
            cam.Elevation(dy * 0.4)
            cam.OrthogonalizeViewUp()
        elif self._button in (QtCore.Qt.MiddleButton, QtCore.Qt.LeftButton):
            self.viewport.pan(dx / h, dy / h)
        elif self._button == QtCore.Qt.RightButton:
            self.viewport.zoom(1.01 ** -dy)
        self.update()

    def mouseReleaseEvent(self, e):
        press, button = getattr(self, "_press", None), getattr(self, "_button", None)
        self._press = None
        self.viewport._dragging = False
        if press is None:
            return
        pos = e.position()
        click = abs(pos.x() - press.x()) + abs(pos.y() - press.y()) <= 4
        if click and (button == QtCore.Qt.RightButton
                      or button == QtCore.Qt.LeftButton and control_click(e.modifiers())):
            self.viewport.contextMenuRequested.emit()
        elif click and button == QtCore.Qt.LeftButton:
            self.viewport.click(pos.x(), pos.y(), self.width(), self.height(),
                                extend=bool(e.modifiers() & QtCore.Qt.ShiftModifier))

    def keyPressEvent(self, e):
        # key events go to this native window, not to the widgets around it: Esc clears the picks
        if e.key() == QtCore.Qt.Key_Escape:
            self.viewport._escape()
        else:
            super().keyPressEvent(e)

    def leaveEvent(self, e):
        self.viewport.hover_leave()

    def wheelEvent(self, e):
        self.viewport.zoom(1.1 ** (e.angleDelta().y() / 120))
        self.update()


class LegacyViewport(Viewport):
    """:class:`Viewport` drawing with OpenGL 1.x display lists (``Preferences › 3D view``)."""

    backend = "legacy-gl"
    uses_vtk = False   # its GLView asks for its own OpenGL 1.x format
    lazy_view = False  # cheap to build, and its GL window and caption are used right away

    def _create_view(self) -> QtWidgets.QWidget:
        self.anti_aliasing = False
        self._paint_driven = False  # render() -> plotter.render() -> one QOpenGLWindow update
        self._first_pending = True
        self.view = GLView(self)
        self.plotter = CameraPlotter(self.view.update)
        self.plotter.enable_parallel_projection()
        container = self._container = QtWidgets.QWidget.createWindowContainer(self.view, self)
        container.setFocusPolicy(QtCore.Qt.StrongFocus)
        container.setMinimumSize(200, 150)
        # a native GL window can't have widgets drawn over it: the frame text goes above it
        self.caption = QtWidgets.QLabel(objectName="muted")
        self.caption.setContentsMargins(6, 2, 6, 2)
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.caption)
        lay.addWidget(container, 1)
        return box

    def _gl_widget(self):
        return self._container

    def _gl_image(self):
        return self.view.grabFramebuffer()

    # ------------------------------------------------------------------ scene
    def redraw(self, coarse: bool = False):
        from ..scene import SceneContext, build

        t0 = time.perf_counter()
        light, flat = None, False
        if self.data is None:
            from ..scene import Scene

            scene = Scene()
        else:
            from ..color_rule import take

            kw = dict(self.style.scene_kwargs())
            light = kw.pop("light", None)
            flat = kw.get("atom_style") == "flat"
            scene = build(take(self.data, kw), ctx=SceneContext(), visuals=kw.pop("visuals", None), **kw)
        dark = _is_dark(self.style.background)
        from matplotlib.colors import to_rgb

        ink = to_rgb(self.style.cell_color or ("#AEB4BE" if dark else "black"))
        self.view.pending = (scene, {"sphere_resolution": min(self.sphere_resolution, 12),
                                     "background": to_rgb(self.style.background), "ink": ink, "light": light,
                                     "flat": flat})
        if self._first and self.data is not None:
            self._first = False
            self._first_pending = True
        self._set_controls(self.data is not None)
        self.caption.setText(EMPTY_HINT if self.data is None else self.overlay_text if self.show_overlay else "")
        self.caption.setVisible(bool(self.caption.text()))
        self._draw_picks()
        self.coarse = False
        self.view.update()
        self._build_time = time.perf_counter() - t0

    def _reset_camera_now(self):
        from ..render import setup_camera

        if self.data is not None:
            setup_camera(self.plotter, self.data, self.initial_view, self.ortho)

    def reset_view(self):
        self._first = True

    def refine(self):
        self.coarse = False

    def _go_coarse_for_motion(self):
        pass

    def opengl_info(self) -> dict:
        info = dict(self.view.info)
        info["software"] = "llvmpipe" in info["renderer"].lower() or "softpipe" in info["renderer"].lower()
        info["backend"] = self.backend
        return info

    def set_anti_aliasing(self, on: bool):
        self.anti_aliasing = False  # not available on OpenGL 1.x

    def set_axes_visible(self, on):
        self.show_axes = on
        self.view.update()

    # ------------------------------------------------------------------ camera
    def pan(self, fx: float, fy: float):
        """Move the view by fractions of the window height."""
        cam = self.plotter.camera
        pos, focal = np.asarray(cam.position), np.asarray(cam.focal_point)
        up = np.asarray(cam.up)
        d = focal - pos
        dist = np.linalg.norm(d)
        right = np.cross(d / dist, up)
        span = 2 * cam.parallel_scale if cam.parallel_projection else 2 * dist * np.tan(np.radians(cam.view_angle) / 2)
        shift = (-fx * right + fy * up) * span
        cam.position = tuple(pos + shift)
        cam.focal_point = tuple(focal + shift)

    def zoom(self, factor: float):
        cam = self.plotter.camera
        if cam.parallel_projection:
            cam.parallel_scale = cam.parallel_scale / factor
        else:
            cam.Dolly(factor)

    def view_along(self, axis: str):
        super().view_along(axis)

    def set_axis_plane(self, corners):
        pass   # no see-through plane on OpenGL 1.x yet (docs/todo.md); alignment and slabs work

    # ------------------------------------------------------------------ picking
    def click(self, x, y, width, height, extend=False):
        if self.data is not None:
            self._click(x, y, extend, lambda x, y: self.pick_at(x, y, width, height))

    def pick_at(self, x, y, width=None, height=None):
        """Particle under window position (x, y from the top-left), by ray casting the spheres."""
        r = self.view.renderer
        if r is None or not len(r.centers):
            return None
        width = width or self.view.width()
        height = height or self.view.height()
        origin, direction = pick_ray(self.plotter, x, y, width, height)
        i = ray_spheres(origin, direction, r.centers, r.radii)
        return None if i is None else int(r.index[i])

    def _draw_picks(self):
        if self.data is None or not self._picks or max(self._picks) >= self.data.count:
            self._picks = []
            self.view.pending_overlay = (None,)
            return
        pos = self.data.positions[self._picks]
        rad = default_radii(self.data, **self.style.radius_kwargs())[self._picks] * self.style.radius_scale
        line = pos if len(self._picks) >= 2 and self.tool in ("distance", "angle") else None
        from matplotlib.colors import to_rgb

        self.view.pending_overlay = (pos, rad, line, to_rgb(TOKENS["pick"]))
