"""3D viewport: VESTA-style view toolbar, tool rail (orbit/pick/measure), overlays."""
from __future__ import annotations

import time

import numpy as np
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
from .style import STYLE_PRESETS, ViewStyle
from .theme import TOKENS


def _button(text, tip, checkable=False, parent=None):
    b = QtWidgets.QToolButton(parent)
    b.setText(text)
    b.setToolTip(tip)
    b.setCheckable(checkable)
    b.setAutoRaise(True)
    return b


class Viewport(QtWidgets.QWidget):
    """QtInteractor wrapped with the view toolbar, tool rail and picking."""

    picked = QtCore.Signal(str)          # status text for the picked atom / measurement
    bondEdit = QtCore.Signal(int, int)   # bond tool: add or delete the bond between two atoms
    styleChanged = QtCore.Signal()
    contextMenuRequested = QtCore.Signal()  # right click without dragging

    TOOLS = (("orbit", "⟲", "Orbit / pan / zoom (mouse)"),
             ("pick", "⌖", "Pick atom: shows id, type, position and properties; Shift+click picks several"),
             ("distance", "↔", "Measure distance: click two atoms"),
             ("angle", "∠", "Measure angle: click three atoms (vertex in the middle)"),
             ("bond", "⌇", "Edit bonds: click two atoms to add or delete the bond between them"))

    def __init__(self, style: ViewStyle, parent=None):
        super().__init__(parent)
        from pyvistaqt import QtInteractor

        self.style = style
        self.data = None
        self.ortho = True
        self.tool = "orbit"
        self.overlay_text = ""
        # set from Preferences (ewald.settings)
        self.rotation_step = 15.0
        self.sphere_resolution = 10
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
        self.coarse = False               # the scene on screen is the coarse one
        self._dragging = False
        self._lod_cloud = None            # subsampled impostor cloud of the current scene, built on first drag
        self._refine_timer = QtCore.QTimer(self, singleShot=True)
        self._refine_timer.timeout.connect(self.refine)
        self._picks: list[int] = []
        self._press = None
        self._first = True

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        grid.addWidget(self._build_viewbar(), 0, 0, 1, 2)
        grid.addWidget(self._build_rail(), 1, 0)
        # always interactive, even when PYVISTA_OFF_SCREEN is set for headless rendering
        self.plotter = QtInteractor(self, off_screen=False)
        grid.addWidget(self.plotter.interactor, 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(1, 1)
        self.plotter.set_background(style.background)
        self.plotter.enable_parallel_projection()
        iren = self.plotter.iren
        iren.add_observer("LeftButtonPressEvent", self._on_press)
        iren.add_observer("LeftButtonReleaseEvent", self._on_release)
        for button in ("Middle", "Right"):
            iren.add_observer(f"{button}ButtonPressEvent", self._on_drag_start)
            iren.add_observer(f"{button}ButtonReleaseEvent", self._on_drag_end)
        iren.add_observer("MouseMoveEvent", self._on_move)
        for wheel in ("MouseWheelForwardEvent", "MouseWheelBackwardEvent"):
            iren.add_observer(wheel, self._on_wheel)

    # ------------------------------------------------------------------ chrome
    def _build_viewbar(self):
        bar = QtWidgets.QFrame(objectName="viewbar")
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(2)
        lab = QtWidgets.QLabel("View along", objectName="muted")
        lay.addWidget(lab)
        for axis in ("a", "b", "c", "a*", "b*", "c*"):
            b = _button(axis, f"View along {axis}")
            b.clicked.connect(lambda _=False, ax=axis: self.view_along(ax))
            lay.addWidget(b)
        lay.addSpacing(8)
        for text, tip, fn in (("←", "Rotate one step (Preferences › Viewports) left", lambda: self.rotate("azimuth", -self.rotation_step)),
                              ("→", "Rotate one step right about the screen vertical", lambda: self.rotate("azimuth", self.rotation_step)),
                              ("↑", "Tilt up one step", lambda: self.rotate("elevation", self.rotation_step)),
                              ("↓", "Tilt down one step", lambda: self.rotate("elevation", -self.rotation_step)),
                              ("↻", "Roll one step in the screen plane", lambda: self.rotate("roll", self.rotation_step))):
            b = _button(text, tip)
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
        self.style_combo.setCurrentText(self.style.preset)
        self.style_combo.currentTextChanged.connect(self._preset)
        lay.addWidget(self.style_combo)
        lay.addStretch()
        fit = _button("Fit", "Zoom to fit (Ctrl+0)")
        fit.clicked.connect(self.zoom_fit)
        lay.addWidget(fit)
        return bar

    def _build_rail(self):
        rail = QtWidgets.QFrame(objectName="rail")
        rail.setFixedWidth(46)
        lay = QtWidgets.QVBoxLayout(rail)
        lay.setContentsMargins(4, 6, 4, 6)
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        self.tool_buttons = {}
        for key, icon, tip in self.TOOLS:
            b = _button(icon, tip, True)
            b.setFixedSize(36, 36)
            b.clicked.connect(lambda _=False, k=key: self.set_tool(k))
            group.addButton(b)
            lay.addWidget(b)
            self.tool_buttons[key] = b
        self.tool_buttons["orbit"].setChecked(True)
        lay.addStretch()
        clear = _button("✕", "Clear picks and measurements")
        clear.setFixedSize(36, 36)
        clear.clicked.connect(self.clear_picks)
        lay.addWidget(clear)
        return rail

    # ------------------------------------------------------------------ scene
    def set_data(self, data, interactive: bool = False):
        """Show ``data``; ``interactive`` (scrubbing, playback) allows a coarse scene that
        is refined once frames stop arriving."""
        self.data = data
        self.redraw(coarse=interactive and self.needs_coarse(scrub=True))

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
        stale = [n for n in p.actors if n.startswith(("surface:", "surface-edges:", "vectors:", "visual:"))]
        for name in ("atoms", "atoms:glass", "atoms_lod", "bonds", "cell", "trajectories", "_legend", "_overlay", *stale):
            p.remove_actor(name, render=False)
        if p.scalar_bars:
            p.remove_scalar_bar()
        if self.data is None:
            p.render()
            return
        t0 = time.perf_counter()
        dark = _is_dark(self.style.background)
        p.set_background(self.style.background)
        kw = self.style.scene_kwargs()
        kw["cell_color"] = "#AEB4BE" if dark else "black"
        kw["sphere_resolution"] = self.sphere_resolution  # interactive; image export uses full quality
        kw["arrow_resolution"] = self.arrow_resolution
        scene = self.data
        if coarse:  # impostors only, no bonds: cheap to build and to draw
            kw.update(atom_style="impostor", only_visible=self._coarse_mask(), bond_pairs=None)
            if scene.bonds is not None or scene.vectors:
                scene = scene.copy()
                scene.bonds, scene.vectors = None, {}
        drawn = build_scene(p, scene, mesh_limit=self.mesh_limit, **kw)
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
            if self.show_axes and not getattr(self, "_axes", False):
                p.add_axes(color="#E7E9EC" if dark else "black", line_width=2)
                self._axes = True
            self._first = False
        self._draw_picks()
        t1 = time.perf_counter()
        p.render()
        self.coarse = coarse
        if coarse:
            self._refine_timer.start(int(self.refine_delay * 1000))
        else:
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
        for name in ("atoms", "bonds"):
            if name in p.actors:
                p.actors[name].SetVisibility(True)
        self.coarse = False
        p.render()

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
        add_atoms(p, self._lod_cloud, atom_style="impostor", name="atoms_lod")
        for name in ("atoms", "atoms:glass", "bonds"):
            if name in p.actors:
                p.actors[name].SetVisibility(False)
        self.coarse = True

    def _on_drag_start(self, obj, event):
        self._dragging = True
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
        if self._dragging and not self.coarse:
            self._go_coarse_for_motion()

    def _on_wheel(self, obj, event):
        self._go_coarse_for_motion()
        if self.coarse:
            self._refine_timer.start(int(self.refine_delay * 1000))

    def reset_view(self):
        self._first = True

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
        self.plotter.reset_camera()
        self.plotter.render()

    def view_direction(self, direction):
        if self.data is not None:
            setup_camera(self.plotter, self.data, direction, self.ortho)
            self.plotter.render()

    def rotate(self, kind, degrees):
        cam = self.plotter.camera
        {"azimuth": cam.Azimuth, "elevation": cam.Elevation, "roll": cam.Roll}[kind](degrees)
        cam.OrthogonalizeViewUp()
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def set_ortho(self, on):
        self.ortho = on
        self.ortho_button.setText("Ortho" if on else "Persp")
        (self.plotter.enable_parallel_projection if on else self.plotter.disable_parallel_projection)()
        self.plotter.render()

    def set_axes_visible(self, on):
        self.show_axes = on
        if getattr(self, "_axes", False):
            (self.plotter.show_axes if on else self.plotter.hide_axes)()
            self.plotter.render()

    def zoom_fit(self):
        self.plotter.reset_camera()
        self.plotter.render()

    def _preset(self, name):
        self.style.apply_preset(name)
        if self.style.bonds_enabled and self.data is not None:
            self.style.ensure_bond_pairs(self.data.types)
        self.styleChanged.emit()

    # ------------------------------------------------------------------ picking
    def set_tool(self, tool):
        self.tool = tool
        self.tool_buttons[tool].setChecked(True)
        self._picks = []
        self._draw_picks()
        self.plotter.render()

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
        self.plotter.render()
        if emit:
            self.picked.emit(self.describe_picks())

    def clear_picks(self):
        self._picks = []
        self._draw_picks()
        self.plotter.render()
        self.picked.emit("")

    def _on_press(self, obj, event):
        self._press = obj.GetEventPosition()
        self._on_drag_start(obj, event)

    def _on_release(self, obj, event):
        self._on_drag_end(obj, event)
        if self.tool == "orbit" or self._press is None or self.data is None:
            return
        x, y = obj.GetEventPosition()
        if abs(x - self._press[0]) + abs(y - self._press[1]) > 4:
            return  # it was a drag (rotation), not a click
        idx = self.pick_at(x, y)
        if idx is None:
            return
        self.add_pick(idx, extend=self.tool == "pick" and bool(obj.GetShiftKey()))

    def add_pick(self, idx, extend=False):
        """Pick particle ``idx``; ``extend`` (Shift+click with the pick tool) adds it to the picks,
        or removes it when already picked, for Ask Claude about selection."""
        if extend:
            self._picks = [p for p in self._picks if p != idx] if idx in self._picks else self._picks + [idx]
            self._draw_picks()
            self.plotter.render()
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
        self.plotter.render()
        self.picked.emit(self.describe_picks())

    def pick_at(self, x, y):
        """Atom under display position (x, y): ray-cast to the surface, then nearest atom center."""
        import vtk

        picker = vtk.vtkPropPicker()
        if not picker.Pick(x, y, 0, self.plotter.renderer):
            return None
        world = np.array(picker.GetPickPosition())
        pos = self.data.positions
        if not len(pos):
            return None
        d = np.linalg.norm(pos - world, axis=1)
        i = int(np.argmin(d))
        rmax = float(default_radii(self.data, **self.style.radius_kwargs()).max()) * self.style.radius_scale
        return i if d[i] <= 1.5 * rmax + 0.3 else None

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
