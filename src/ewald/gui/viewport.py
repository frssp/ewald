"""3D viewport: VESTA-style view toolbar, tool rail (orbit/pick/measure), overlays."""
from __future__ import annotations

import time

import numpy as np

from . import check_qt_binding

check_qt_binding()  # before PySide6: a clear error when another Qt binding is loaded

from PySide6 import QtCore, QtGui, QtWidgets

from ..render import (
    _RADIUS_KEYS,
    _add_colorbar,
    add_atoms,
    atom_cloud,
    build_scene,
    default_radii,
    needs_depth_peeling,
    setup_camera,
)
from . import header
from .style import STYLE_PRESETS, ViewStyle
from .theme import TOKENS


def _button(text, tip, checkable=False, parent=None):
    b = QtWidgets.QToolButton(parent)
    b.setText(text)
    b.setToolTip(tip)
    b.setCheckable(checkable)
    b.setAutoRaise(True)
    return b


def camera_ray(camera, x, y, width, height):
    """World-space ray (origin, unit direction) under window position (x, y from the top-left)
    of a ``vtkCamera`` drawn into a width x height view."""
    m = camera.GetCompositeProjectionTransformMatrix(width / height, -1, 1)
    inv = np.linalg.inv(np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)]))
    ndc = (2 * x / width - 1, 1 - 2 * y / height)
    near = inv @ [ndc[0], ndc[1], -1, 1]
    far = inv @ [ndc[0], ndc[1], 1, 1]
    near, far = near[:3] / near[3], far[:3] / far[3]
    d = far - near
    return near, d / np.linalg.norm(d)


def ray_spheres(origin, direction, centers, radii):
    """Index of the first sphere hit by the ray, or None."""
    oc = centers - origin
    t = oc @ direction
    d2 = np.einsum("ij,ij->i", oc, oc) - t * t
    r2 = radii * radii
    hit = np.flatnonzero(d2 <= r2)
    if not len(hit):
        return None
    t_hit = t[hit] - np.sqrt(np.maximum(r2[hit] - d2[hit], 0))
    return int(hit[np.argmin(t_hit)])


class Viewport(QtWidgets.QWidget):
    """QtInteractor wrapped with the view toolbar, tool rail and picking."""

    picked = QtCore.Signal(str)          # status text for the picked atom / measurement
    bondEdit = QtCore.Signal(int, int)   # bond tool: add or delete the bond between two atoms
    styleChanged = QtCore.Signal()
    contextMenuRequested = QtCore.Signal()  # right click without dragging
    hovered = QtCore.Signal(str)         # one-line description of the atom under the cursor, "" for none

    TOOLS = (("orbit", "Orbit", "Orbit / pan / zoom (mouse)"),
             ("pick", "Pick", "Pick atom: shows id, type, position and properties; Shift+click picks several"),
             ("distance", "Dist", "Measure distance: click two atoms"),
             ("angle", "Angle", "Measure angle: click three atoms (vertex in the middle)"),
             ("bond", "Bond", "Edit bonds: click two atoms to add or delete the bond between them"))

    def __init__(self, style: ViewStyle, parent=None):
        super().__init__(parent)
        self.style = style
        self.data = None
        self.ortho = True
        self.tool = "orbit"
        self.overlay_text = ""
        # set from Preferences (ewald.settings)
        self.rotation_step = 15.0
        self.sphere_drawing = "impostor"  # 'auto' styles: GPU impostors, or 'mesh' up to mesh_limit atoms
        self.sphere_resolution = 16
        self.mesh_limit = 150_000
        self.arrow_resolution = 6
        self.show_overlay = True
        self.show_axes = True
        self.initial_view = "iso"
        # interactive level of detail (Preferences › Viewports › Performance)
        self.detail_mode = "auto"         # 'auto': coarse while moving if a full frame misses target_fps
        self.target_fps = 30.0
        self.interactive_atoms = 1_000_000  # coarse scenes draw at most this many atoms
        self.refine_delay = 0.3           # seconds of idle before the full-detail scene returns
        self._render_time = 0.0           # last full-detail render, seconds
        self._build_time = 0.0            # last full-detail scene build + render, seconds
        self._pending_build = None        # build time of a scene not rendered yet
        self.coarse = False               # the scene on screen is the coarse one
        self._dragging = False
        self._lod_cloud = None            # subsampled impostor cloud of the current scene, built on first drag
        self._refine_timer = QtCore.QTimer(self, singleShot=True)
        self._refine_timer.timeout.connect(self.refine)
        self._picks: list[int] = []
        self._press = None
        # hover readout (Preferences › Viewports › Hover): ray cast once the cursor rests
        self.hover_info = True
        self.hover_delay = 0.25           # seconds the cursor rests before the readout appears
        self._hover_at = None             # (x, y, width, height) of the resting cursor, y from the top
        self._hover_global = None         # same position on screen, for the tooltip
        self._hover_index = None
        self._hover_radii = (None, None)  # (key, radii) cached per frame and radius style
        self._hover_timer = QtCore.QTimer(self, singleShot=True)
        self._hover_timer.timeout.connect(self._hover_fire)
        self._first = True

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        self.styled_buttons = []  # (button, text in the default style); MainWindow.set_button_style
        grid.addWidget(self._build_viewbar(), 0, 0, 1, 2)
        grid.addWidget(self._build_rail(), 1, 0)
        grid.addWidget(self._create_view(), 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(1, 1)

    def _create_view(self) -> QtWidgets.QWidget:
        """The 3D widget: a VTK QtInteractor (``self.plotter``) wired to the pick and LOD handlers."""
        from pyvistaqt import QtInteractor

        # always interactive, even when PYVISTA_OFF_SCREEN is set for headless rendering
        # no auto-update timer: it re-rendered the scene 5 times a second even when idle
        self.plotter = QtInteractor(self, off_screen=False, auto_update=False)
        self.plotter.set_background(self.style.background)
        self.plotter.enable_parallel_projection()
        self.anti_aliasing = False
        self.set_anti_aliasing(True)
        self._paint_driven = self._render_on_paint()
        iren = self.plotter.iren
        iren.add_observer("LeftButtonPressEvent", self._on_press)
        iren.add_observer("LeftButtonReleaseEvent", self._on_release)
        for button in ("Middle", "Right"):
            iren.add_observer(f"{button}ButtonPressEvent", self._on_drag_start)
            iren.add_observer(f"{button}ButtonReleaseEvent", self._on_drag_end)
        iren.add_observer("MouseMoveEvent", self._on_move)
        iren.add_observer("LeaveEvent", lambda *_: self.hover_leave())
        for wheel in ("MouseWheelForwardEvent", "MouseWheelBackwardEvent"):
            iren.add_observer(wheel, self._on_wheel)
        return self.plotter.interactor

    # ------------------------------------------------------------------ chrome
    def _build_viewbar(self):
        bar = QtWidgets.QFrame(objectName="viewbar")
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(2)
        axes = QtWidgets.QFrame(objectName="axisGroup")  # one compact segmented group, no label
        row = QtWidgets.QHBoxLayout(axes)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        for axis in ("a", "b", "c", "a*", "b*", "c*"):
            b = _button(axis, f"View along {axis}")
            b.clicked.connect(lambda _=False, ax=axis: self.view_along(ax))
            row.addWidget(b)
        lay.addWidget(axes)
        lay.addSpacing(8)
        for name, tip, fn in (("arrow_left", "Rotate one step (Preferences › Viewports) left", lambda: self.rotate("azimuth", -self.rotation_step)),
                              ("arrow_right", "Rotate one step right about the screen vertical", lambda: self.rotate("azimuth", self.rotation_step)),
                              ("arrow_up", "Tilt up one step", lambda: self.rotate("elevation", self.rotation_step)),
                              ("arrow_down", "Tilt down one step", lambda: self.rotate("elevation", -self.rotation_step)),
                              ("roll", "Roll one step in the screen plane", lambda: self.rotate("roll", self.rotation_step))):
            b = header.button(name, tip)
            b.clicked.connect(fn)
            lay.addWidget(b)
        lay.addSpacing(8)
        self.ortho_button = _button("Ortho", "Orthographic / perspective projection", True)
        self.ortho_button.setChecked(True)
        self.ortho_button.toggled.connect(self.set_ortho)
        lay.addWidget(self.ortho_button)
        lay.addSpacing(8)
        lay.addWidget(QtWidgets.QLabel("Style", objectName="muted"))
        self.style_combo = QtWidgets.QComboBox()
        self.style_combo.addItems(list(STYLE_PRESETS))
        self.style_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.style_combo.setCurrentText(self.style.preset)
        self.style_combo.currentTextChanged.connect(self._preset)
        lay.addWidget(self.style_combo)
        lay.addStretch()
        fit = _button("Fit", "Zoom to fit (Ctrl+0)")
        fit.setProperty("ewald_icon", "fit")
        fit.clicked.connect(self.zoom_fit)
        lay.addWidget(fit)
        self.styled_buttons.append((fit, False))
        return bar

    def _build_rail(self):
        rail = QtWidgets.QFrame(objectName="rail")
        rail.setFixedWidth(46)
        lay = QtWidgets.QVBoxLayout(rail)
        lay.setContentsMargins(4, 6, 4, 6)
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        self.tool_buttons = {}
        for key, label, tip in self.TOOLS:
            b = _button(label, tip, True)
            b.setProperty("ewald_icon", key)
            b.clicked.connect(lambda _=False, k=key: self.set_tool(k))
            group.addButton(b)
            lay.addWidget(b)
            self.tool_buttons[key] = b
            self.styled_buttons.append((b, False))
        self.tool_buttons["orbit"].setChecked(True)
        lay.addStretch()
        clear = _button("Clear", "Clear picks and measurements")
        clear.setProperty("ewald_icon", "clear")
        clear.clicked.connect(self.clear_picks)
        lay.addWidget(clear)
        self.styled_buttons.append((clear, False))
        self._rail = rail
        return rail

    def set_button_style(self, style: str):
        """Size the tool rail for icons (36 px squares) or text labels (see ``icons.button_style``)."""
        text = style == "Text"
        self._rail.setFixedWidth(58 if text else 46)
        for b, _ in self.styled_buttons:
            if b.parent() is self._rail:
                b.setFixedSize(50 if text else 36, 28 if text else 36)
                b.setIconSize(QtCore.QSize(20, 20))

    # ------------------------------------------------------------------ scene
    def set_data(self, data, interactive: bool = False):
        """Show ``data``; ``interactive`` (scrubbing, playback) allows a coarse scene that
        is refined once frames stop arriving. A parked viewport keeps the data and draws it
        when it is shown again."""
        self.data = data
        if self.parked:
            self._parked_stale = True
            return
        self.redraw(coarse=interactive and self.needs_coarse(scrub=True))

    parked = False
    _parked_stale = False

    def set_parked(self, parked: bool):
        """Parked: behind another tab or in a folded panel (the window's workbench says so).
        Frames that arrive meanwhile build no scene; the last one is drawn on return."""
        was, self.parked = self.parked, parked
        if was and not parked and self._parked_stale:
            self._parked_stale = False
            self.redraw()

    def snapshot(self) -> QtGui.QPixmap:
        """The widget as a picture, 3D view included (``grab`` leaves a GL window black); shown
        in its place while a tab is dragged over it."""
        pix = self.grab()
        try:
            gl, img = self._gl_widget(), self._gl_image()
        except Exception:  # no read-back on this driver: the toolbars alone still help
            return pix
        painter = QtGui.QPainter(pix)
        painter.drawImage(QtCore.QRect(gl.mapTo(self, QtCore.QPoint(0, 0)), gl.size()), img)
        painter.end()
        return pix

    def _gl_widget(self) -> QtWidgets.QWidget:
        return self.plotter  # the QtInteractor is the GL widget

    def _gl_image(self) -> QtGui.QImage:
        arr = np.ascontiguousarray(self.plotter.screenshot(None, return_img=True)[..., :3])
        h, w = arr.shape[:2]
        return QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()

    def needs_coarse(self, scrub: bool = False) -> bool:
        """Whether moving the view (or changing frames when ``scrub``) should use the coarse scene."""
        if self.data is None or self.detail_mode == "full":
            return False
        if self.detail_mode == "coarse":
            return True
        budget = 1.0 / max(self.target_fps, 1e-3)
        return (self._build_time if scrub else self._render_time) > budget

    def _coarse_mask(self):
        """Every k-th atom, so a coarse scene holds at most ``interactive_atoms``."""
        n = self.data.count
        visible = None
        if "Transparency" in self.data:
            visible = np.asarray(self.data["Transparency"]) < 1
        step = -(-n // max(int(self.interactive_atoms), 1))
        if step <= 1:
            return visible
        mask = np.zeros(n, bool)
        mask[::step] = True
        return mask if visible is None else mask & visible

    def redraw(self, coarse: bool = False):
        p = self.plotter
        self._refine_timer.stop()
        self._lod_cloud = None
        stale = [n for n in p.actors if n.startswith(("surface:", "surface-edges:", "vectors:", "visual:", "bonds:"))]
        for name in ("atoms", "atoms:glass", "atoms_lod", "bonds", "cell", "trajectories", "_legend", "_overlay", *stale):
            p.remove_actor(name, render=False)
        if p.scalar_bars:
            p.remove_scalar_bar()
        if self.data is None:
            self.render()
            return
        t0 = time.perf_counter()
        dark = _is_dark(self.style.background)
        p.set_background(self.style.background)
        kw = self.style.scene_kwargs()
        kw["cell_color"] = self.style.cell_color or ("#AEB4BE" if dark else "black")
        kw["sphere_resolution"] = self.sphere_resolution  # interactive; image export uses full quality
        if kw["atom_style"] == "auto" and self.sphere_drawing == "impostor":
            kw["atom_style"] = "impostor"
        kw["arrow_resolution"] = self.arrow_resolution
        scene = self.data
        if coarse:  # impostors only, no bonds: cheap to build and to draw
            kw.update(atom_style="flat" if kw["atom_style"] == "flat" else "impostor",
                      only_visible=self._coarse_mask(), bond_pairs=None)
            if scene.bonds is not None or scene.vectors:
                scene = scene.copy()
                scene.bonds, scene.vectors = None, {}
        drawn = build_scene(p, scene, mesh_limit=self.mesh_limit, **kw)
        # boundary images are hovered through the drawn spheres (each knows its real atom)
        atoms = drawn["atoms"].primitive if "boundary" in kw and not coarse and "atoms" in drawn else None
        self._boundary_atoms = None if atoms is None else (atoms.centers, atoms.index, atoms.sizes())
        if (any(s.transparency > 0 for s in self.data.surfaces.values()) or needs_depth_peeling(drawn)) \
                and not getattr(self, "_peeling", False):
            p.enable_depth_peeling()
            self._peeling = True
        if "ColorCoding.range" in self.data.attributes:
            _add_colorbar(p, self.data, font_size=12)
            sb = p.scalar_bar
            if sb is not None and dark:
                for prop in (sb.GetTitleTextProperty(), sb.GetLabelTextProperty()):
                    prop.SetColor(0.9, 0.91, 0.93)
        if self.overlay_text and self.show_overlay:
            p.add_text(self.overlay_text, position="upper_left", font_size=7, name="_overlay",
                       color="#E7E9EC" if dark else "black", render=False)
        if self._first:
            setup_camera(p, self.data, self.initial_view, self.ortho)
            self._first = False
        if self.show_axes and getattr(self, "_axes", None) != dark:  # new, or the background flipped
            p.add_axes(color="#E7E9EC" if dark else "black", line_width=2)
            self._axes = dark
        self._draw_picks()
        self.coarse = coarse
        if coarse:
            self._refine_timer.start(int(self.refine_delay * 1000))
        if self._paint_driven:
            self._pending_build = None if coarse else time.perf_counter() - t0
            self.render()  # timed by _render_ended when the paint draws it
            return
        t1 = time.perf_counter()
        p.render()
        if not coarse:
            t2 = time.perf_counter()
            self._render_time, self._build_time = t2 - t1, t2 - t0

    def refine(self):
        """Replace a coarse scene with full detail (called after ``refine_delay`` of idle)."""
        if not self.coarse or self.data is None:
            return
        p = self.plotter
        lod = p.actors.get("atoms_lod")
        if lod is None:  # the whole scene was built coarse (scrubbing): rebuild it
            self.redraw()
            return
        p.remove_actor("atoms_lod", render=False)  # rotation only swapped the atoms
        for name in ("atoms", "bonds", "bonds:outline"):
            if name in p.actors:
                p.actors[name].SetVisibility(True)
        self.coarse = False
        self.render()

    def _go_coarse_for_motion(self):
        """Swap the full atoms (and bonds) for subsampled impostors while the camera moves."""
        p = self.plotter
        if self.coarse or self.data is None or not self.data.count or not self.needs_coarse():
            return
        if self._lod_cloud is None:
            kw = self.style.scene_kwargs()
            extra = {k: kw[k] for k in (*_RADIUS_KEYS, "type_colors") if k in kw}
            self._lod_cloud, _ = atom_cloud(self.data, radius_scale=kw["radius_scale"],
                                            only_visible=self._coarse_mask(), **extra)
        add_atoms(p, self._lod_cloud, atom_style="flat" if self.style.atom_style == "flat" else "impostor",
                  name="atoms_lod")
        for name in ("atoms", "atoms:glass", "bonds", "bonds:outline"):
            if name in p.actors:
                p.actors[name].SetVisibility(False)
        self.coarse = True

    def _on_drag_start(self, obj, event):
        self._dragging = True
        self.hover_leave()
        self._refine_timer.stop()
        self._right_press = QtGui.QCursor.pos() if event == "RightButtonPressEvent" else None

    def _on_drag_end(self, obj, event):
        self._dragging = False
        start = getattr(self, "_right_press", None)
        if event == "RightButtonReleaseEvent" and start is not None \
                and (QtGui.QCursor.pos() - start).manhattanLength() < 5:
            self.contextMenuRequested.emit()  # a click, not a zoom drag
        if self.coarse:
            self._refine_timer.start(int(self.refine_delay * 1000))

    def _on_move(self, obj, event):
        if self._dragging:
            if not self.coarse:
                self._go_coarse_for_motion()
            return
        w = self.plotter.interactor
        pos = w.mapFromGlobal(QtGui.QCursor.pos())
        self.hover_move(pos.x(), pos.y(), w.width(), w.height())

    def _on_wheel(self, obj, event):
        self._go_coarse_for_motion()
        if self.coarse:
            self._refine_timer.start(int(self.refine_delay * 1000))

    def opengl_info(self) -> dict:
        """The OpenGL driver of the viewport (see :func:`ewald.render.opengl_info`)."""
        from ..render import opengl_info

        info = getattr(self, "_gl_info", None)
        if info is None:
            info = opengl_info(self.plotter.render_window)
            if info["renderer"]:  # known once the window has a GL context
                self._gl_info = info
        return info

    def set_anti_aliasing(self, on: bool):
        """FXAA on or off (Preferences › Viewports › Smooth edges)."""
        if on == self.anti_aliasing:
            return
        if on:
            self.plotter.enable_anti_aliasing("fxaa")
        else:
            self.plotter.disable_anti_aliasing()
        self.anti_aliasing = on

    def reset_view(self):
        self._first = True

    def _render_on_paint(self) -> bool:
        """Draw each change once, in the widget's next paint.

        pyvistaqt renders when asked (camera moves, ``plotter.render()``) and then again in
        the paint that shows the result, so every frame and every mouse move cost two
        renders. Here the interactor only schedules a paint (as VTK's own Qt widget does,
        through ``RenderEvent``), requests made before the paint collapse into one, and a
        hidden viewport renders nothing. Returns False on a pyvistaqt without the paint
        flag this relies on, which keeps its own behavior."""
        widget, iren = self.plotter.interactor, self.plotter.iren.interactor
        if not hasattr(widget, "_in_paint"):
            return False
        rw = self.plotter.render_window

        def on_render(*_):
            if widget._in_paint:
                rw.Render()  # the paint's own render
            else:
                widget.update()

        iren.EnableRenderOff()
        iren.AddObserver("RenderEvent", on_render)
        rw.AddObserver("StartEvent", self._render_started)
        rw.AddObserver("EndEvent", self._render_ended)
        return True

    def _render_started(self, *_):
        self._t_render = time.perf_counter()

    def _render_ended(self, *_):
        if self.coarse or getattr(self, "_t_render", None) is None:
            return
        dt = time.perf_counter() - self._t_render
        self._render_time = dt
        if self._pending_build is not None:  # the first render of a rebuilt scene
            self._build_time, self._pending_build = self._pending_build + dt, None

    def render(self):
        """Show the current scene: schedules one paint (see ``_render_on_paint``)."""
        if self._paint_driven:
            self.plotter.interactor.update()
        else:
            self.plotter.render()

    # ------------------------------------------------------------------ camera
    def view_along(self, axis: str):
        if self.data is None or not self.data.cell.is_valid:
            return
        m = self.data.cell.matrix
        if axis.endswith("*"):
            rec = np.linalg.inv(m).T  # reciprocal vectors (rows), up to 2*pi
            d = rec["abc".index(axis[0])]
        else:
            d = m["abc".index(axis)]
        d = d / np.linalg.norm(d)
        # up: next lattice vector projected onto the screen plane (c for a/b, b for c)
        up = m[2] if axis[0] in "ab" else m[1]
        up = up - np.dot(up, d) * d
        if np.linalg.norm(up) < 1e-6:
            up = np.cross(d, [1.0, 0, 0]) if abs(d[0]) < 0.9 else np.cross(d, [0, 1.0, 0])
        setup_camera(self.plotter, self.data, tuple(d), self.ortho)
        self.plotter.camera.up = tuple(up / np.linalg.norm(up))
        self.plotter.reset_camera(render=False)
        self.render()

    def view_direction(self, direction):
        if self.data is not None:
            setup_camera(self.plotter, self.data, direction, self.ortho)
            self.render()

    def rotate(self, kind, degrees):
        cam = self.plotter.camera
        {"azimuth": cam.Azimuth, "elevation": cam.Elevation, "roll": cam.Roll}[kind](degrees)
        cam.OrthogonalizeViewUp()
        self.plotter.reset_camera_clipping_range()
        self.render()

    def set_ortho(self, on):
        self.ortho = on
        self.ortho_button.setText("Ortho" if on else "Persp")
        (self.plotter.enable_parallel_projection if on else self.plotter.disable_parallel_projection)()
        self.render()

    def set_axes_visible(self, on):
        self.show_axes = on
        if getattr(self, "_axes", None) is not None:
            (self.plotter.show_axes if on else self.plotter.hide_axes)()
            self.render()

    def zoom_fit(self):
        self.plotter.reset_camera(render=False)
        self.render()

    def _preset(self, name):
        self.style.apply_preset(name, self.data.types if self.data is not None else ())
        if self.style.bonds_enabled and self.data is not None:
            self.style.ensure_bond_pairs(self.data.types)
        self.styleChanged.emit()

    # ------------------------------------------------------------------ picking
    def set_tool(self, tool):
        self.tool = tool
        self.tool_buttons[tool].setChecked(True)
        self._picks = []
        self._draw_picks()
        self.render()

    @property
    def picks(self) -> list[int]:
        """Indices of the picked particles in the current frame."""
        return list(self._picks)

    def set_picks(self, indices, emit: bool = True):
        """Select particles ``indices`` (e.g. a range dragged on a plot), drawn like picks;
        the pick tool's Shift+click then adds to or removes from them."""
        n = self.data.count if self.data is not None else 0
        self._picks = [int(i) for i in indices if 0 <= int(i) < n]
        if self._picks and self.tool not in ("pick", "orbit"):
            self.tool = "pick"
            self.tool_buttons["pick"].setChecked(True)
        self._draw_picks()
        self.render()
        if emit:
            self.picked.emit(self.describe_picks())

    def clear_picks(self):
        self._picks = []
        self._draw_picks()
        self.render()
        self.picked.emit("")

    def _event_position(self):
        """Window position (x, y from the top-left, in Qt pixels) of the current VTK mouse event.

        Read from the render window interactor, not the observer's ``obj``: pyvista sends button
        release observers to the interactor style, which has no event position."""
        x, y = self.plotter.iren.interactor.GetEventPosition()
        w = self.plotter.interactor
        ratio = w.devicePixelRatioF() or 1.0
        return x / ratio, w.height() - y / ratio

    def _on_press(self, obj, event):
        self._press = self._event_position()
        self._on_drag_start(obj, event)

    def _on_release(self, obj, event):
        self._on_drag_end(obj, event)
        if self.tool == "orbit" or self._press is None or self.data is None:
            return
        x, y = self._event_position()
        if abs(x - self._press[0]) + abs(y - self._press[1]) > 4:
            return  # it was a drag (rotation), not a click
        idx = self.pick_at(x, y)
        if idx is None:
            return
        shift = bool(self.plotter.iren.interactor.GetShiftKey())
        self.add_pick(idx, extend=self.tool == "pick" and shift)

    def add_pick(self, idx, extend=False):
        """Pick particle ``idx``; ``extend`` (Shift+click with the pick tool) adds it to the picks,
        or removes it when already picked, for Ask AI about selection."""
        if extend:
            self._picks = [p for p in self._picks if p != idx] if idx in self._picks else self._picks + [idx]
            self._draw_picks()
            self.render()
            self.picked.emit(self.describe_picks())
            return
        need = {"pick": 1, "distance": 2, "angle": 3, "bond": 2}[self.tool]
        self._picks = (self._picks + [idx])[-need:] if len(self._picks) < need else [idx]
        if self.tool == "bond" and len(self._picks) == 2:
            a, b = self._picks
            self._picks = []
            self._draw_picks()
            if a != b:
                self.bondEdit.emit(a, b)
            return
        self._draw_picks()
        self.render()
        self.picked.emit(self.describe_picks())

    def pick_at(self, x, y):
        """Atom under window position (x, y from the top-left): the front-most atom sphere on the
        camera ray (as for hover), else the atom nearest to the surface VTK picks there (a bond,
        a polyhedron, an isosurface)."""
        import vtk

        w = self.plotter.interactor
        idx = self.hover_index(x, y, w.width(), w.height())
        if idx is not None:
            return idx
        ratio = w.devicePixelRatioF() or 1.0
        picker = vtk.vtkPropPicker()
        if not picker.Pick(x * ratio, (w.height() - y) * ratio, 0, self.plotter.renderer):
            return None
        world = np.array(picker.GetPickPosition())
        pos = self.data.positions
        if not len(pos):
            return None
        diff = pos - world
        if self.style.boundary and self.data.cell.is_valid:  # a boundary ghost picks its real atom
            cell = self.data.cell
            f = diff @ np.linalg.inv(cell.matrix)
            diff = (f - np.rint(f) * cell.pbc) @ cell.matrix
        d = np.linalg.norm(diff, axis=1)
        i = int(np.argmin(d))
        rmax = float(default_radii(self.data, **self.style.radius_kwargs()).max()) * self.style.radius_scale
        return i if d[i] <= 1.5 * rmax + 0.3 else None

    # ------------------------------------------------------------------ hover
    def hover_move(self, x, y, width, height):
        """The cursor moved to window position (x, y from the top-left) without a button down.

        Nothing is computed while it moves: the ray cast runs once it rests for ``hover_delay``,
        so hovering never slows rotation, even with millions of atoms.
        """
        if not self.hover_info or self._dragging or self.data is None:
            return
        last = self._hover_at
        if last is not None and abs(x - last[0]) + abs(y - last[1]) <= 2 and self._hover_index is not None:
            return  # jitter over the same atom keeps the tooltip
        self._hover_at = (x, y, width, height)
        self._hover_global = QtGui.QCursor.pos()
        if self._hover_index is not None:
            QtWidgets.QToolTip.hideText()
        self._hover_timer.start(int(self.hover_delay * 1000))

    def hover_leave(self):
        """Cursor left the view or a drag started: drop the readout."""
        self._hover_timer.stop()
        self._hover_at = None
        if self._hover_index is not None:
            self._hover_index = None
            QtWidgets.QToolTip.hideText()
            self.hovered.emit("")

    def _hover_fire(self):
        if self._hover_at is None or self._dragging or self.data is None or not self.hover_info:
            return
        idx = self.hover_index(*self._hover_at)
        if idx == self._hover_index:
            return
        self._hover_index = idx
        if idx is None:
            QtWidgets.QToolTip.hideText()
            self.hovered.emit("")
            return
        lines = self.describe_atom(idx)
        QtWidgets.QToolTip.showText(self._hover_global, "\n".join(lines))
        self.hovered.emit("   ".join(lines))

    def hover_index(self, x, y, width, height):
        """Front-most atom under window position (x, y from the top-left), or None.

        Ray-sphere test on the CPU against the camera, so it does not depend on the renderer
        (VTK meshes or impostors, the OpenGL 1.x viewport) and needs no extra render pass.
        """
        d = self.data
        if d is None or not d.count or width <= 0 or height <= 0:
            return None
        origin, direction = camera_ray(self.plotter.camera, x, y, width, height)
        ghosts = getattr(self, "_boundary_atoms", None)
        if ghosts is not None and self.style.boundary:
            idx = ray_spheres(origin, direction, ghosts[0], ghosts[2])
            return None if idx is None else int(ghosts[1][idx])
        idx = ray_spheres(origin, direction, d.positions, self._hover_radii_for(d))
        return None if idx is None else int(idx)

    def _hover_radii_for(self, d):
        kw = self.style.radius_kwargs()
        key = (id(d), d.count, repr(sorted(kw.items())), self.style.radius_scale)
        if self._hover_radii[0] != key:
            self._hover_radii = (key, np.asarray(default_radii(d, **kw), float) * self.style.radius_scale)
        return self._hover_radii[1]

    def describe_atom(self, i) -> list[str]:
        """Hover readout of particle ``i``: type and index, Cartesian and (with a cell) fractional position."""
        d = self.data
        pid = d.particles.get("Particle Identifier")
        head = f"{d.types[i]}  index {i}" + (f"  id {int(pid[i])}" if pid is not None else "")
        x, y, z = d.positions[i]
        lines = [head, f"xyz  {x:.4f}  {y:.4f}  {z:.4f} Å"]
        if d.cell.is_valid:
            f = d.cell.to_fractional(d.positions[i])
            lines.append(f"frac {f[0]:.4f}  {f[1]:.4f}  {f[2]:.4f}")
        return lines

    def _mic(self, v):
        cell = self.data.cell
        if not cell.is_valid:
            return v
        f = v @ np.linalg.inv(cell.matrix)
        f -= np.round(f) * cell.pbc
        return f @ cell.matrix

    def describe_picks(self) -> str:
        d, P = self.data, self._picks
        if not P:
            return ""
        def label(i):
            pid = d.particles.get("Particle Identifier")
            return f"{d.types[i]}#{int(pid[i]) if pid is not None else i}"
        if self.tool in ("pick", "orbit") and len(P) > 1:
            shown = ", ".join(label(i) for i in P[:6]) + (" …" if len(P) > 6 else "")
            return f"{len(P)} atoms picked: {shown}"
        if self.tool == "pick" or len(P) == 1:
            i = P[0]
            extra = [f"{k}={d[k][i]:.4g}" for k in d.particles
                     if k not in ("Position", "Particle Type", "Particle Identifier", "Color")
                     and d[k].ndim == 1 and d[k].dtype.kind in "biuf"][:4]
            x, y, z = d.positions[i]
            return f"{label(i)}  ({x:.3f}, {y:.3f}, {z:.3f}) Å  " + "  ".join(extra)
        if self.tool == "distance" and len(P) == 2:
            r = np.linalg.norm(self._mic(d.positions[P[1]] - d.positions[P[0]]))
            return f"d({label(P[0])}, {label(P[1])}) = {r:.4f} Å"
        if self.tool == "angle" and len(P) == 3:
            v1 = self._mic(d.positions[P[0]] - d.positions[P[1]])
            v2 = self._mic(d.positions[P[2]] - d.positions[P[1]])
            ang = np.degrees(np.arccos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1)))
            return f"∠({label(P[0])}, {label(P[1])}, {label(P[2])}) = {ang:.2f}°"
        return f"{len(P)} atom(s) picked"

    def _draw_picks(self):
        import pyvista as pv

        p = self.plotter
        for name in ("_pick", "_measure", "_measure_label"):
            p.remove_actor(name, render=False)
        if self.data is None or not self._picks or max(self._picks) >= self.data.count:
            self._picks = []
            return
        pos = self.data.positions[self._picks]
        r = default_radii(self.data, **self.style.radius_kwargs())[self._picks] * self.style.radius_scale
        cloud = pv.PolyData(np.asarray(pos, float))
        if len(pos) > 20000:  # a plot range can select very many atoms: dots, not spheres
            p.add_mesh(cloud, color=TOKENS["pick"], point_size=4, render_points_as_spheres=True,
                       name="_pick", render=False, pickable=False)
        else:
            res = 16 if len(pos) <= 500 else 8
            cloud["scale"] = np.asarray(r, float) * 1.18
            rings = cloud.glyph(scale="scale", orient=False,
                                geom=pv.Sphere(radius=1.0, theta_resolution=res, phi_resolution=res))
            p.add_mesh(rings, color=TOKENS["pick"], style="wireframe", line_width=1.5, opacity=0.9,
                       name="_pick", render=False, pickable=False)
        if len(self._picks) >= 2 and self.tool in ("distance", "angle"):
            line = pv.lines_from_points(pos)
            p.add_mesh(line, color=TOKENS["pick"], line_width=3, name="_measure", render=False, pickable=False)
            text = self.describe_picks().split("=")[-1].strip()
            mid = pos.mean(0) if self.tool == "distance" else pos[1]
            p.add_point_labels([mid], [text], font_size=14, text_color=TOKENS["pick"], shape_color="#14161A",
                               shape_opacity=0.7, name="_measure_label", render=False, always_visible=True)


def _is_dark(color: str) -> bool:
    from matplotlib.colors import to_rgb

    return sum(to_rgb(color)) < 1.5
