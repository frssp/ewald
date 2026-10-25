"""3D viewport: VESTA-style view controls and tools (orbit/pick/measure) as chips over the view, overlays."""
from __future__ import annotations

import sys
import time
import weakref

import numpy as np

from . import check_qt_binding

check_qt_binding()  # before PySide6: a clear error when another Qt binding is loaded

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from ..render import (
    _RADIUS_KEYS,
    add_atoms,
    atom_cloud,
    build_scene,
    default_radii,
    needs_depth_peeling,
    setup_camera,
)
from . import header
from .application import use_vtk_surface_format, vtk_surface_format  # noqa: F401  (moved there)
from .palette import native_text
from .style import STYLE_PRESETS, ViewStyle
from .theme import TOKENS


def _button(text, tip, checkable=False, parent=None):
    b = QtWidgets.QToolButton(parent)
    b.setText(text)
    b.setToolTip(tip)
    b.setCheckable(checkable)
    b.setAutoRaise(True)
    return b


def control_click(modifiers, platform: str | None = None) -> bool:
    """Whether a left click is the macOS one-button context click, Control+click (Qt reports the
    Control key as ``MetaModifier`` there; elsewhere Ctrl+click stays a normal click)."""
    return (platform or sys.platform) == "darwin" and bool(modifiers & QtCore.Qt.MetaModifier)


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


class _Placeholder(QOpenGLWidget):
    """Holds the 3D view's place until it is built: its first paint (the window is on screen
    now) schedules ``build`` for the next turn of the event loop. An OpenGL widget itself, so
    the window composites OpenGL from its first frame: adding the VTK widget to a window
    shown without one makes Qt rebuild the window's backing store, which crashes some drivers."""

    def __init__(self, build, background):
        super().__init__()
        self.setObjectName("viewportPlaceholder")
        self._build = build
        self._background = QtGui.QColor(background)
        self._scheduled = False

    def paintGL(self):
        ctx = self.context().functions()
        c = self._background
        ctx.glClearColor(c.redF(), c.greenF(), c.blueF(), 1.0)
        ctx.glClear(0x00004000)  # GL_COLOR_BUFFER_BIT
        if not self._scheduled:
            self._scheduled = True
            QtCore.QTimer.singleShot(0, self._build)


def release_gl(widget) -> None:
    """Drop pyvistaqt's Python wrappers of a QOpenGLWidget's context and surface.

    pyvistaqt keeps ``context().surface()``, the widget's internal QOffscreenSurface, which Qt
    owns and deletes with the widget. PySide does not notice, so while the QtInteractor object
    lives on (a closed window still referenced from the console, a closure, a test), the stale
    wrapper stays registered for that address. A later Qt object allocated at the same address
    then comes back to Python as that wrapper (a QHeaderView as a "QOffscreenSurface"), and
    calls through it write into the wrong object: random heap corruption (issue #145).
    """
    forget = getattr(widget, "_forget_context", None)   # pyvistaqt >= 0.12
    if forget is not None:
        forget()
    elif widget is not None:
        for name in ("_surface", "_ctx"):
            if hasattr(widget, name):
                setattr(widget, name, None)


def _image_actor(rgba, dx, dy, anchor):
    """A 2D actor showing an RGBA image with its lower-left corner (dx, dy) display pixels from ``anchor``
    (normalized viewport coordinates)."""
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkRenderingCore import vtkActor2D, vtkCoordinate, vtkImageMapper

    h, w = rgba.shape[:2]
    img = vtkImageData()
    img.SetDimensions(w, h, 1)
    arr = numpy_to_vtk(np.ascontiguousarray(rgba[::-1]).reshape(-1, 4), deep=True)
    arr.SetName("rgba")
    img.GetPointData().SetScalars(arr)
    mapper = vtkImageMapper()
    mapper.SetInputData(img)
    mapper.SetColorWindow(255.0)
    mapper.SetColorLevel(127.5)
    actor = vtkActor2D()
    actor.SetMapper(mapper)
    ref = vtkCoordinate()
    ref.SetCoordinateSystemToNormalizedViewport()
    ref.SetValue(*anchor)
    c = actor.GetPositionCoordinate()
    c.SetCoordinateSystemToDisplay()
    c.SetReferenceCoordinate(ref)
    c.SetValue(round(dx), round(dy))
    return actor


class Viewport(QtWidgets.QWidget):
    """QtInteractor wrapped with the view toolbar, tool rail and picking."""

    picked = QtCore.Signal(str)          # status text for the picked atom / measurement
    bondEdit = QtCore.Signal(int, int)   # bond tool: add or delete the bond between two atoms
    styleChanged = QtCore.Signal()
    contextMenuRequested = QtCore.Signal()  # right click without dragging
    hovered = QtCore.Signal(str)         # one-line description of the atom under the cursor, "" for none
    viewBuilt = QtCore.Signal()          # the 3D widget exists now (see ensure_view)
    lazy_view = True                     # build the 3D widget after the first paint
    uses_vtk = True                      # a pyvistaqt QtInteractor (needs vtk_surface_format)

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
        self._want_anti_aliasing = True  # applied when the 3D widget is built
        self.tool = "orbit"
        self.overlay_text = ""
        # set from Preferences (ewald.settings)
        self.rotation_step = 15.0
        self.sphere_drawing = "impostor"  # 'auto' styles: GPU impostors, or 'mesh' up to mesh_limit atoms
        self.sphere_resolution = 16
        self.mesh_limit = 150_000
        self.bond_limit = 250_000  # bonds.auto_limit: a style preset turns bonds on only up to this many atoms
        self.arrow_resolution = 6
        self.show_overlay = True
        self.show_legend = True           # the atoms' color bar or category swatches
        self._has_colorbar = False
        self._colorbar_box = None         # (x0, y0, x1, y1) of the color bar cards, Qt px from the top right corner
        self.show_axes = True
        self.initial_view = "iso"
        # interactive level of detail (Preferences › 3D view › Performance)
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
        self._empty_at = None             # (time, x, y) of the last click on empty space
        # hover readout (Preferences › 3D view › Hover): ray cast once the cursor rests
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
        # the view controls are chips floating over the 3D view (D61): view directions top left
        # (``_viewbar``), projection, style and Fit top right (``_displaybar``), tools on the left
        # (``_rail``); they fold into menus as the view gets small (see _fit_hud)
        self._viewbar, self._displaybar = self._build_viewbar()
        self._build_rail()
        if not self.hud_overlay:   # a native GL window can't have widgets over it: the chips go around it
            strip = QtWidgets.QWidget(objectName="hudStrip")
            row = QtWidgets.QHBoxLayout(strip)
            row.setContentsMargins(self.HUD_MARGIN, self.HUD_MARGIN, self.HUD_MARGIN, 0)
            row.addWidget(self._viewbar)
            row.addStretch()
            row.addWidget(self._displaybar)
            grid.addWidget(strip, 0, 0, 1, 2)
            col = QtWidgets.QWidget(objectName="hudStrip")
            lay = QtWidgets.QVBoxLayout(col)
            lay.setContentsMargins(self.HUD_MARGIN, self.HUD_MARGIN, 0, self.HUD_MARGIN)
            lay.addWidget(self._rail)
            lay.addStretch()
            grid.addWidget(col, 1, 0)
        # the 3D widget (VTK: importing pyvistaqt alone takes about a second) is built after the
        # window's first paint, or on first use of ``plotter``; a plain frame holds its place
        self._grid = grid
        self._plotter = None
        self._placeholder = None
        if self.uses_vtk:
            use_vtk_surface_format()  # before the placeholder: see use_vtk_surface_format
        if self.lazy_view:
            self._placeholder = _Placeholder(self.ensure_view, self.style.background)
            grid.addWidget(self._placeholder, 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(1, 1)
        # Esc clears the picks, measurements and selection, like a double click on empty space
        esc = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self,
                              context=QtCore.Qt.WidgetWithChildrenShortcut)
        esc.activated.connect(self._escape)
        if self.data is None:
            self._set_controls(False)
        if not self.lazy_view:
            self.ensure_view()

    @property
    def plotter(self):
        """The 3D view (a pyvistaqt ``QtInteractor``, or the legacy GL camera plotter), built on
        first access."""
        if self._plotter is None:
            self.ensure_view()
        return self._plotter

    @plotter.setter
    def plotter(self, value):
        self._plotter = value

    @property
    def view_built(self) -> bool:
        return self._plotter is not None

    def ensure_view(self):
        """Build the 3D widget now if it is not yet (normally right after the first paint)."""
        if self._plotter is not None or getattr(self, "_building", False):
            return
        self._building = True
        try:
            widget = self._create_view()
        finally:
            self._building = False
        if self._placeholder is None:
            self._grid.addWidget(widget, 1, 1)
        else:
            self._grid.replaceWidget(self._placeholder, widget)
            self._placeholder.deleteLater()
            self._placeholder = None
        self._place_hud()
        self.redraw()
        self.viewBuilt.emit()

    def _create_view(self) -> QtWidgets.QWidget:
        """The 3D widget: a VTK QtInteractor (``self.plotter``) wired to the pick and LOD handlers."""
        from pyvistaqt import QtInteractor

        # always interactive, even when PYVISTA_OFF_SCREEN is set for headless rendering
        # no auto-update timer: it re-rendered the scene 5 times a second even when idle
        self.plotter = QtInteractor(self, off_screen=False, auto_update=False)
        # drop its wrapper of the GL surface as soon as the widget goes (see release_gl)
        self.plotter.destroyed.connect(lambda *_, ref=weakref.ref(self.plotter): release_gl(ref()))
        self.plotter.set_background(self.style.background)
        # no MSAA (FXAA is the view's anti-aliasing): depth peeling into pyvistaqt's multisampled
        # framebuffers drops every see-through layer (NEB overlays, Transparency), on Mesa and macOS
        self.plotter.render_window.SetMultiSamples(0)
        if self.ortho:
            self.plotter.enable_parallel_projection()
        self.anti_aliasing = False
        self.set_anti_aliasing(self._want_anti_aliasing)
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
    # ------------------------------------------------------------------ HUD
    # The view controls float over the 3D view as chips (D61) and fold as the view gets small;
    # each level hides more, widest first. Fit always stays.
    HUD_LEVELS = (
        "full",          # everything
        "folded",        # a*/b*/c* under a chevron, the five rotations under one menu button
        "axis-menu",     # the six directions under one "a" menu button, Style under ⋯
        "minimal",       # rotations and Ortho go into ⋯ too
    )
    HUD_MARGIN = 8       # px between the chips and the edges of the view
    hud_overlay = True   # chips over the GL widget (a QOpenGLWidget composites them); False: around it
    AXES = ("a", "b", "c", "a*", "b*", "c*")

    def _chip(self, name, vertical=False):
        chip = QtWidgets.QFrame(self, objectName="hudChip")
        chip.setProperty("hudName", name)
        chip.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        chip.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        lay = (QtWidgets.QVBoxLayout if vertical else QtWidgets.QHBoxLayout)(chip)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        lay.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)   # the chip is as large as what it shows
        return chip, lay

    @staticmethod
    def _divider(lay, vertical=False):
        d = QtWidgets.QFrame(objectName="hudDivider")
        if vertical:
            d.setFixedHeight(1)
        else:
            d.setFixedSize(1, 16)
        lay.addWidget(d, 0, QtCore.Qt.AlignCenter)
        return d

    def _build_viewbar(self):
        """The two top chips: view directions and rotations (left), projection, style and Fit (right)."""
        view, lay = self._chip("view")
        self.axis_buttons = {}
        for axis in self.AXES:
            b = _button(axis, f"Look along {axis}")
            b.setObjectName("hudAxis")
            b.clicked.connect(lambda _=False, ax=axis: self.view_along(ax))
            lay.addWidget(b)
            self.axis_buttons[axis] = b
        self.axis_more = self._menu_button("chevron_down", "Look along a reciprocal axis", "hudChevron")
        for axis in ("a*", "b*", "c*"):
            self.axis_more.menu().addAction(f"Look along {axis}", lambda ax=axis: self.view_along(ax))
        lay.addWidget(self.axis_more)
        self.axis_menu = self._menu_button("chevron_down", "Look along a cell axis", "hudAxisMenu")
        self.axis_menu.setText("a")
        self.axis_menu.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.axis_menu.setLayoutDirection(QtCore.Qt.RightToLeft)   # the chevron after the letter
        self.axis_menu.setIconSize(QtCore.QSize(12, 12))
        for axis in self.AXES:
            self.axis_menu.menu().addAction(f"Look along {axis}", lambda ax=axis: self.view_along(ax))
        lay.addWidget(self.axis_menu)
        self._rotate_divider = self._divider(lay)
        self.rotations = (
            ("arrow_left", "Rotate left", "Rotate one step (Preferences › 3D view) left", ("azimuth", -1)),
            ("arrow_right", "Rotate right", "Rotate one step right about the screen vertical", ("azimuth", 1)),
            ("arrow_up", "Tilt up", "Tilt up one step", ("elevation", 1)),
            ("arrow_down", "Tilt down", "Tilt down one step", ("elevation", -1)),
            ("roll", "Roll", "Roll one step in the screen plane", ("roll", 1)))
        self.rotate_buttons = []
        for name, _label, tip, (kind, sign) in self.rotations:
            b = header.button(name, tip)
            b.clicked.connect(lambda _=False, k=kind, s=sign: self.rotate(k, s * self.rotation_step))
            lay.addWidget(b)
            self.rotate_buttons.append(b)
        self.rotate_menu = self._menu_button("roll", "Rotate or tilt one step", "hudMenu")
        self._add_rotations(self.rotate_menu.menu())
        lay.addWidget(self.rotate_menu)

        display, lay = self._chip("display")
        self.ortho_button = _button("Ortho", "Orthographic / perspective projection", True)
        self.ortho_button.setChecked(True)
        self.ortho_button.toggled.connect(self.set_ortho)
        lay.addWidget(self.ortho_button)
        self.style_combo = QtWidgets.QComboBox(objectName="hudCombo", toolTip="Style")
        self.style_combo.addItems(list(STYLE_PRESETS))
        self.style_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.style_combo.setCurrentText(self.style.preset)
        self.style_combo.currentTextChanged.connect(self._preset)
        lay.addWidget(self.style_combo)
        self.more_button = self._menu_button("more", "More view controls", "hudMenu")
        self.more_button.menu().aboutToShow.connect(self._fill_more_menu)
        lay.addWidget(self.more_button)
        fit = _button("Fit", native_text("Zoom to fit (Ctrl+0)"))
        fit.setProperty("ewald_icon", "fit")
        fit.clicked.connect(self.zoom_fit)
        lay.addWidget(fit)
        self.fit_button = fit
        self.styled_buttons.append((fit, False))
        self.hud_level = None
        self._view_chip, self._display_chip = view, display
        self._set_hud_level(0, view, display)
        return view, display

    def _menu_button(self, icon_name, tip, name):
        """A chip button that opens a menu at once (folded axes, rotations, ⋯)."""
        b = QtWidgets.QToolButton(toolTip=tip, autoRaise=True, objectName=name)
        b.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        b.setMenu(QtWidgets.QMenu(b))
        if icon_name:
            from .icons import icon

            b.setProperty("ewald_icon", icon_name)
            b.setIcon(icon(icon_name))
            b.setIconSize(QtCore.QSize(header.ICON, header.ICON))
            b.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        return b

    def _add_rotations(self, menu):
        for _name, label, tip, (kind, sign) in self.rotations:
            a = menu.addAction(label, lambda k=kind, s=sign: self.rotate(k, s * self.rotation_step))
            a.setToolTip(tip)

    def _set_hud_level(self, level: int, view=None, display=None):
        """Show the top chips at fold ``level`` (an index into ``HUD_LEVELS``)."""
        if level == self.hud_level:
            return
        self.hud_level = level
        for b in ("a", "b", "c"):
            self.axis_buttons[b].setVisible(level < 2)
        for b in ("a*", "b*", "c*"):
            self.axis_buttons[b].setVisible(level < 1)
        self.axis_more.setVisible(level == 1)
        self.axis_menu.setVisible(level >= 2)
        for b in self.rotate_buttons:
            b.setVisible(level < 1)
        self.rotate_menu.setVisible(level in (1, 2))
        self._rotate_divider.setVisible(level < 3)
        self.style_combo.setVisible(level < 2)
        self.ortho_button.setVisible(level < 3)
        self.more_button.setVisible(level >= 2)
        for chip in (view or self._viewbar, display or self._displaybar):
            chip.layout().activate()
            chip.adjustSize()

    def hud_width(self, level: int) -> int:
        """The width the two top chips need at fold ``level``, with their margins."""
        current = self.hud_level
        self._set_hud_level(level)
        width = self._viewbar.sizeHint().width() + self._displaybar.sizeHint().width() + 3 * self.HUD_MARGIN
        if not self.hud_overlay:
            width -= self.HUD_MARGIN   # the strip has no gap at its right edge
        self._set_hud_level(current)
        return width

    def _fit_hud(self, width: int):
        """Fold the top chips just enough to fit ``width``."""
        level = next((i for i in range(len(self.HUD_LEVELS)) if self.hud_width(i) <= width),
                     len(self.HUD_LEVELS) - 1)
        self._set_hud_level(level)

    def _fill_more_menu(self):
        """The ⋯ menu of the folded chips: what they have no room for now."""
        menu = self.more_button.menu()
        menu.clear()
        if not self.rotate_menu.isVisible():
            self._add_rotations(menu.addMenu("Rotate"))
        if not self.ortho_button.isVisible():
            ortho = menu.addAction("Orthographic projection")
            ortho.setCheckable(True)
            ortho.setChecked(self.ortho_button.isChecked())
            ortho.toggled.connect(self.ortho_button.setChecked)
        style = menu.addMenu("Style")
        for name in STYLE_PRESETS:
            a = style.addAction(name, lambda n=name: self.style_combo.setCurrentText(n))
            a.setCheckable(True)
            a.setChecked(name == self.style_combo.currentText())

    def _build_rail(self):
        """The tool chip on the left: orbit, pick, measure, bond tools, and Clear."""
        rail, lay = self._chip("tools", vertical=True)
        self._rail = rail
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
        self.rail_more = _button("More", "More tools", True)
        self.rail_more.setProperty("ewald_icon", "more")
        self.rail_more.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.rail_more.setMenu(QtWidgets.QMenu(self.rail_more))
        self.rail_more.menu().aboutToShow.connect(self._fill_rail_menu)
        self.rail_more.menu().aboutToHide.connect(self._update_rail_more)
        lay.addWidget(self.rail_more)
        self.styled_buttons.append((self.rail_more, False))
        self._clear_divider = self._divider(lay, vertical=True)
        clear = _button("Clear", "Clear picks and measurements")
        clear.setProperty("ewald_icon", "clear")
        clear.clicked.connect(self.clear_picks)
        lay.addWidget(clear)
        self.clear_button = clear
        self.styled_buttons.append((clear, False))
        self.rail_tools = len(self.TOOLS)   # tools shown on the chip, the others are under ⋯
        self._set_rail_tools(len(self.TOOLS), True)
        return rail

    def _set_rail_tools(self, shown: int, clear: bool):
        """Show the first ``shown`` tools (and Clear if ``clear``) on the tool chip, the rest under ⋯."""
        self.rail_tools = shown
        for i, key in enumerate(self.tool_buttons):
            self.tool_buttons[key].setVisible(i < shown)
        self.clear_button.setVisible(clear)
        self._clear_divider.setVisible(clear)
        self.rail_more.setVisible(shown < len(self.TOOLS) or not clear)
        self._update_rail_more()
        self._rail.layout().activate()
        self._rail.adjustSize()

    def _update_rail_more(self):
        """⋯ looks checked while the current tool is one of those folded into it."""
        if hasattr(self, "rail_more"):
            self.rail_more.setChecked(self.tool in [k for k, *_ in self.TOOLS[self.rail_tools:]])

    def _fit_rail(self, height: int):
        """Fold the tool chip's last tools (Clear first) into ⋯ until the rest fit ``height``."""
        lay = self._rail.layout()
        m = lay.contentsMargins()
        step = max(max(b.sizeHint().height(), b.minimumHeight()) for b in self.tool_buttons.values()) + lay.spacing()
        room = height - m.top() - m.bottom() + lay.spacing()
        n = len(self.TOOLS)
        if n * step + step + lay.spacing() + 1 <= room:      # all tools, a divider and Clear
            self._set_rail_tools(n, True)
        else:
            self._set_rail_tools(max(1, min(n - 1, room // step - 1)), False)   # one slot for ⋯

    def _fill_rail_menu(self):
        menu = self.rail_more.menu()
        menu.clear()
        for key, label, tip in self.TOOLS[self.rail_tools:]:
            a = menu.addAction(label, lambda k=key: self.set_tool(k))
            a.setToolTip(tip)
            a.setCheckable(True)
            a.setChecked(key == self.tool)
        if not self.clear_button.isVisible():
            menu.addSeparator()
            menu.addAction("Clear picks and measurements", self.clear_picks)

    def hud_insets(self) -> tuple[int, int]:
        """(left, top) px of the 3D view the chips cover; 2D overlays drawn in the view start below and
        right of them. Zero when the chips sit around the view (``hud_overlay`` False)."""
        if not self.hud_overlay:
            return 0, 0
        top = max(self._viewbar.sizeHint().height(), self._displaybar.sizeHint().height())
        return self._rail.sizeHint().width() + self.HUD_MARGIN, top + self.HUD_MARGIN

    def _place_hud(self):
        """Fold the chips to the 3D view's size and put them over its corners."""
        if not self.hud_overlay:
            self._fit_hud(self.width())
            self._fit_rail(max(0, self.height() - self._viewbar.sizeHint().height() - 3 * self.HUD_MARGIN))
            return
        r = self.rect()
        m = self.HUD_MARGIN
        self._fit_hud(r.width())
        self._viewbar.move(r.left() + m, r.top() + m)
        self._displaybar.move(r.right() + 1 - m - self._displaybar.sizeHint().width(), r.top() + m)
        top = r.top() + 2 * m + self._viewbar.sizeHint().height()
        # the orientation axes take the bottom left fifth of the view (pyvista's add_axes)
        bottom = r.bottom() + 1 - (round(0.2 * r.height()) if self.show_axes else m)
        self._fit_rail(max(0, bottom - top))
        self._rail.move(r.left() + m, top)
        for chip in (self._viewbar, self._displaybar, self._rail):
            chip.raise_()

    def sizeHint(self):
        # the GL widget has no size hint of its own, and the chips float outside the layout
        return QtCore.QSize(640, 480)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_hud()

    def showEvent(self, event):
        super().showEvent(event)
        self._place_hud()

    def _set_controls(self, on: bool):
        """The view chips act on the structure: greyed out while there is none."""
        for chip in (self._viewbar, self._displaybar, self._rail):
            chip.setEnabled(on)

    def _show_empty(self, empty: bool):
        """With nothing loaded: a centered hint in the view, and the view controls greyed out."""
        self._set_controls(not empty)
        p = self.plotter
        p.remove_actor("_empty", render=False)
        if empty:
            dark = _is_dark(self.style.background)
            actor = p.add_text(EMPTY_HINT, position=(0.5, 0.5), viewport=True, font_size=8,
                               color="#8B919A" if dark else "#6B7079", name="_empty")
            prop = actor.GetTextProperty()
            prop.SetJustificationToCentered()
            prop.SetVerticalJustificationToCentered()

    def set_button_style(self, style: str):
        """Size the tool chip for icons (30 px squares) or text labels (see ``icons.button_style``)."""
        text = style == "Text"
        for b, _ in self.styled_buttons:
            if b.parent() is self._rail:
                b.setFixedSize(50 if text else 30, 26 if text else 30)
                b.setIconSize(QtCore.QSize(18, 18))
        self._rail.adjustSize()
        self._place_hud()

    # ------------------------------------------------------------------ scene
    def set_data(self, data, interactive: bool = False):
        """Show ``data``; ``interactive`` (scrubbing, playback) allows a coarse scene that
        is refined once frames stop arriving. A parked viewport keeps the data and draws it
        when it is shown again."""
        self.data = data
        if self.parked:
            self._parked_stale = True
            return
        coarse = interactive and self.needs_coarse(scrub=True)
        if interactive and self.playing:
            # one level of detail for the whole playback: a frame time near the budget made
            # some frames full and others coarse, so bonds flickered (see set_playing)
            if self._play_coarse is None:
                self._play_coarse = coarse
            coarse = self._play_coarse
        self.redraw(coarse=coarse)

    playing = False
    _play_coarse = None

    def set_playing(self, playing: bool):
        """The timeline plays: frames keep the level of detail the first one got and are not
        refined in between (a refine between two frames switched bonds on and off). Stopping
        refines the last frame."""
        self.playing, self._play_coarse = playing, None
        if playing:
            self._refine_timer.stop()
        elif self.coarse:
            self._refine_timer.start(int(self.refine_delay * 1000))

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

    def release_gl(self):
        """Forget the GL context and surface (after ``plotter.close()``, before Qt deletes them)."""
        release_gl(self.plotter)

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

    def _coarse_step(self) -> int:
        """A coarse scene draws every k-th atom; k > 1 only above ``interactive_atoms``."""
        return -(-self.data.count // max(int(self.interactive_atoms), 1))

    def _coarse_mask(self):
        """Every k-th atom, so a coarse scene holds at most ``interactive_atoms``."""
        n = self.data.count
        visible = None
        if "Transparency" in self.data:
            visible = np.asarray(self.data["Transparency"]) < 1
        step = self._coarse_step()
        if step <= 1:
            return visible
        mask = np.zeros(n, bool)
        mask[::step] = True
        return mask if visible is None else mask & visible

    def redraw(self, coarse: bool = False):
        if self._plotter is None:  # drawn when the 3D widget is built
            return
        p = self.plotter
        self._refine_timer.stop()
        self._lod_cloud = None
        stale = [n for n in p.actors if n.startswith(("surface:", "surface-edges:", "vectors:", "visual:", "bonds:", "_legend"))]
        for name in ("atoms", "atoms:glass", "atoms_lod", "bonds", "cell", "trajectories", "_legend", "_overlay",
                     "_labels-labels", "measurements", "_measurements-labels", *stale):
            p.remove_actor(name, render=False)
        for title in list(p.scalar_bars.keys()):
            p.remove_scalar_bar(title, render=False)
        self._show_empty(self.data is None)
        if self.data is None:
            self.render()
            return
        t0 = time.perf_counter()
        dark = _is_dark(self.style.background)
        p.set_background(self.style.background)
        kw = self.style.scene_kwargs()
        labels = kw.pop("labels", None)
        kw["cell_color"] = self.style.cell_color or ("#AEB4BE" if dark else "black")
        kw["sphere_resolution"] = self.sphere_resolution  # interactive; image export uses full quality
        if kw["atom_style"] == "auto" and self.sphere_drawing == "impostor":
            kw["atom_style"] = "impostor"
        kw["arrow_resolution"] = self.arrow_resolution
        from ..color_rule import take

        scene = shown = take(self.data, kw)   # the view's color_by (display only)
        if coarse:  # impostors: cheap to build and to draw
            kw.update(atom_style="flat" if kw["atom_style"] == "flat" else "impostor",
                      only_visible=self._coarse_mask())
            if self._coarse_step() > 1:
                # subsampled atoms would leave bonds and arrows dangling: drop them. Otherwise
                # they stay, so a coarse frame shows what a full one does (bonds no longer flicker)
                kw["bond_pairs"] = None
                if scene.bonds is not None or scene.vectors:
                    scene = scene.copy()
                    scene.bonds, scene.vectors = None, {}
        drawn = build_scene(p, scene, mesh_limit=self.mesh_limit, **kw)
        # boundary and replicated images are hovered through the drawn spheres (each knows its real atom)
        atoms = drawn["atoms"].primitive if ("boundary" in kw or "replicate" in kw) and not coarse and "atoms" in drawn else None
        self._boundary_atoms = None if atoms is None else (atoms.centers, atoms.index, atoms.sizes())
        if (any(s.transparency > 0 for s in self.data.surfaces.values()) or needs_depth_peeling(drawn)) \
                and not getattr(self, "_peeling", False):
            p.enable_depth_peeling()
            self._peeling = True
        self._has_colorbar = False
        self._colorbar_box = None
        if self.show_legend:
            self._draw_colorbars(shown, drawn.legends)
            self._draw_color_legend(shown, dark)
        if self.overlay_text and self.show_overlay:
            self._draw_caption(self.overlay_text, dark)
        if self._first:
            setup_camera(p, self.data, self.initial_view, self.ortho)
            self._first = False
        if self.show_axes and getattr(self, "_axes", None) != dark:  # new, or the background flipped
            p.add_axes(color="#E7E9EC" if dark else "black", line_width=2)
            self._axes = dark
        if labels and not coarse:
            self._draw_labels(labels, dark)
        if kw.get("measurements") and not coarse:
            self._draw_measurements(kw["measurements"])
        self._draw_picks()
        self.coarse = coarse
        if coarse and not self.playing:
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
        """Swap the full atoms for (subsampled) impostors while the camera moves; bonds go only when
        the atoms are subsampled."""
        p = self.plotter
        if self.coarse or self.data is None or not self.data.count or not self.needs_coarse():
            return
        if self._lod_cloud is None:
            from ..color_rule import take

            kw = self.style.scene_kwargs()
            shown = take(self.data, kw)
            extra = {k: kw[k] for k in (*_RADIUS_KEYS, "type_colors") if k in kw}
            self._lod_cloud, _ = atom_cloud(shown, radius_scale=kw["radius_scale"],
                                            only_visible=self._coarse_mask(), **extra)
        add_atoms(p, self._lod_cloud, atom_style="flat" if self.style.atom_style == "flat" else "impostor",
                  name="atoms_lod")
        hide = ("atoms", "atoms:glass") + (("bonds", "bonds:outline") if self._coarse_step() > 1 else ())
        for name in hide:  # bonds stay unless the atoms are subsampled
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
        if event == "RightButtonReleaseEvent":
            # pyvista observes right releases on the interactor style, and a VTK style that has an
            # observer for an event skips its own handler: end the zoom drag here, or every later
            # mouse move (after a right-click menu, too) keeps zooming
            self.plotter.iren.style.OnRightButtonUp()
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
        if self._hover_colorbar(pos.x(), pos.y()):
            return
        self.hover_move(pos.x(), pos.y(), w.width(), w.height())

    def _hover_colorbar(self, x, y) -> bool:
        """Over the color bar cards: a pointing hand and what a click does instead of the atom readout."""
        w = self.plotter.interactor
        on = self.on_colorbar(x, y)
        if on != getattr(self, "_over_colorbar", False):
            self._over_colorbar = on
            if on:
                self.hover_leave()
                w.setCursor(QtCore.Qt.PointingHandCursor)
                tip = "Unfold the color bars" if self.style.colorbar_collapsed else "Fold the color bars"
                QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), tip, w)
            else:
                w.unsetCursor()
                QtWidgets.QToolTip.hideText()
        return on

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
        """FXAA on or off (Preferences › 3D view › Smooth edges)."""
        self._want_anti_aliasing = on
        if self._plotter is None or on == self.anti_aliasing:
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
        if self._plotter is None:  # drawn when the 3D widget is built
            return
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
        if self._plotter is None:  # applied when the 3D widget is built
            return
        (self.plotter.enable_parallel_projection if on else self.plotter.disable_parallel_projection)()
        self.render()

    def set_axes_visible(self, on):
        self.show_axes = on
        self._place_hud()   # the tool chip may use the corner the axes leave
        if getattr(self, "_axes", None) is not None:
            (self.plotter.show_axes if on else self.plotter.hide_axes)()
            self.render()

    def zoom_fit(self):
        self.plotter.reset_camera(render=False)
        self.render()

    def _preset(self, name):
        self.style.apply_preset(name, self.data.types if self.data is not None else ())
        if self.style.bonds_enabled and self.data is not None and self.data.count > self.bond_limit:
            self.style.bonds_enabled = False  # too many atoms for automatic bonds; Bonds turns them on
        if self.style.bonds_enabled and self.data is not None:
            self.style.ensure_bond_pairs(self.data.types)
        self.styleChanged.emit()

    # ------------------------------------------------------------------ picking
    def set_tool(self, tool):
        self.tool = tool
        self.tool_buttons[tool].setChecked(True)
        self._update_rail_more()
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
        if self._press is None:
            return
        x, y = self._event_position()
        if abs(x - self._press[0]) + abs(y - self._press[1]) > 4:
            return  # it was a drag (rotation), not a click
        if control_click(QtWidgets.QApplication.keyboardModifiers()):
            self.contextMenuRequested.emit()  # macOS Control+click
            return
        if self.on_colorbar(x, y):
            self.toggle_colorbars()   # a click on a color bar card folds or unfolds them, never picks behind
            return
        if self.data is None:
            return
        shift = bool(self.plotter.iren.interactor.GetShiftKey())
        self._click(x, y, shift, self.pick_at)

    def _click(self, x, y, shift, pick_at):
        """A left click at window position (x, y): pick with the pick and measure tools; a double
        click on empty space clears the picks in any tool."""
        if self.tool == "orbit" and not self._picks:
            return
        idx = pick_at(x, y)
        if idx is None:
            self._empty_click(x, y)
        else:
            self._empty_at = None
            if self.tool != "orbit":
                self.add_pick(idx, extend=self.tool == "pick" and shift)

    def _empty_click(self, x, y):
        """Click on empty space: the second one within the double-click interval clears the picks."""
        now, last = time.monotonic(), self._empty_at
        self._empty_at = (now, x, y)
        interval = QtWidgets.QApplication.doubleClickInterval() / 1000
        if last and now - last[0] <= interval and abs(x - last[1]) + abs(y - last[2]) <= 6:
            self._empty_at = None
            if self._picks:
                self.clear_picks()

    def _escape(self):
        if self._picks:
            self.clear_picks()

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
        from vtkmodules.vtkRenderingCore import (
            vtkPropPicker,  # not ``import vtk``: that loads every VTK module
        )

        w = self.plotter.interactor
        idx = self.hover_index(x, y, w.width(), w.height())
        if idx is not None:
            return idx
        ratio = w.devicePixelRatioF() or 1.0
        picker = vtkPropPicker()
        if not picker.Pick(x * ratio, (w.height() - y) * ratio, 0, self.plotter.renderer):
            return None
        world = np.array(picker.GetPickPosition())
        pos = self.data.positions
        if not len(pos):
            return None
        diff = pos - world
        if (self.style.boundary or self.style.replicate) and self.data.cell.is_valid:  # an image picks its real atom
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
        if ghosts is not None and (self.style.boundary or self.style.replicate):
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

    COLORBAR_FONT = 12   # px at a device pixel ratio of 1

    def _draw_colorbars(self, data, legends):
        """The color bars (:func:`ewald.overlays.color_legends`) as one image pinned to the top right corner:
        the items pictures draw (:mod:`ewald.colorbars`), so the view and the rendered figure match."""
        from ..colorbars import colorbar_items, items_rgba
        from ..overlays import color_legends, ink_color

        bars = color_legends(data, legends)
        if not bars:
            return
        w = self.plotter.interactor
        ratio = w.devicePixelRatioF() or 1.0
        W, H = max(1, int(w.width() * ratio)), max(1, int(w.height() * ratio))
        ink = ink_color(self.style.background, False)
        items = colorbar_items(bars, (W, H), unit=self.COLORBAR_FONT * ratio, ink=ink, background=self.style.background,
                               collapsed=bool(getattr(self.style, "colorbar_collapsed", False)), chevrons=True)
        rgba, x0, y0 = items_rgba(items, ink)
        y0 += self._inset_px()[1]   # below the view chips at the top
        self._colorbar_box = ((x0 - W) / ratio, y0 / ratio, (x0 + rgba.shape[1] - W) / ratio,
                              (y0 + rgba.shape[0]) / ratio)
        # VTK display coordinates start at the bottom left; the offset is kept from the top right corner, so the
        # cards stay there while the view is resized
        self.plotter.add_actor(_image_actor(rgba, x0 - W, -y0 - rgba.shape[0], (1.0, 1.0)), name="_legend",
                               reset_camera=False, render=False)
        self._has_colorbar = True

    def _inset_px(self) -> tuple[float, float]:
        """:meth:`hud_insets` in VTK display pixels."""
        ratio = self.plotter.interactor.devicePixelRatioF() or 1.0
        left, top = self.hud_insets()
        return left * ratio, top * ratio

    def _draw_caption(self, text, dark: bool):
        """The frame caption (``overlay_text``) at the top left, below the view chips."""
        from vtkmodules.vtkRenderingCore import vtkCoordinate, vtkTextActor

        a = vtkTextActor()
        a.SetInput(text)
        tp = a.GetTextProperty()
        tp.SetFontSize(round(12 * (self.plotter.interactor.devicePixelRatioF() or 1.0)))
        tp.SetColor((0.9, 0.91, 0.93) if dark else (0.0, 0.0, 0.0))
        tp.SetVerticalJustificationToTop()
        left, top = self._inset_px()
        ref = vtkCoordinate()
        ref.SetCoordinateSystemToNormalizedViewport()
        ref.SetValue(0.0, 1.0)
        c = a.GetPositionCoordinate()
        c.SetCoordinateSystemToDisplay()
        c.SetReferenceCoordinate(ref)
        c.SetValue(left + 4, -top - 4)
        self.plotter.add_actor(a, name="_overlay", reset_camera=False, render=False)

    def on_colorbar(self, x, y) -> bool:
        """Window position (x, y from the top left, Qt px) is on the color bar cards."""
        box, w = self._colorbar_box, self.plotter.interactor if self._plotter is not None else None
        if box is None or w is None:
            return False
        dx = x - w.width()
        return box[0] <= dx <= box[2] and box[1] <= y <= box[3]

    def toggle_colorbars(self):
        """Fold the color bar cards to chips, or unfold them (the view's ``colorbar_collapsed``)."""
        self.style.colorbar_collapsed = not self.style.colorbar_collapsed
        self.styleChanged.emit()   # the window redraws and writes the view cell

    def _draw_color_legend(self, data, dark: bool):
        """Swatches of the atoms' color rule (structure types, painted conditions; :mod:`ewald.color_rule`):
        top right, or top left under the caption when a color bar takes the right edge. Text actors placed in
        pixels from a corner, so they keep their size and place when the view is resized."""
        from vtkmodules.vtkRenderingCore import vtkCoordinate, vtkTextActor

        from ..color_rule import legend_entries, legend_title, rule

        r = rule(data, self.style.type_colors)
        rows = legend_entries(r)
        if not rows:
            return
        right = not self._has_colorbar
        ink = (0.9, 0.91, 0.93) if dark else (0.1, 0.1, 0.1)
        size, step = 12, 18
        left, top = self._inset_px()   # below the view chips, right of the tool chip
        x0, y = (-14, -14 - top) if right else (12 + left, -40 - top)
        lines = ([(legend_title(r), None)] if r.get("kind") == "categorical" else []) + list(rows)

        def actor(text, dx, dy, bold=False, fill=None):
            a = vtkTextActor()
            a.SetInput(text)
            tp = a.GetTextProperty()
            tp.SetFontSize(size)
            tp.SetBold(bold)
            tp.SetColor(*ink)
            tp.SetJustificationToRight() if right and fill is None else tp.SetJustificationToLeft()
            tp.SetVerticalJustificationToTop()
            if fill is not None:
                tp.SetBackgroundColor(*fill)
                tp.SetBackgroundOpacity(1.0)
                tp.SetFrame(True)
                tp.SetFrameColor(*ink)
            ref = vtkCoordinate()
            ref.SetCoordinateSystemToNormalizedViewport()
            ref.SetValue(1.0 if right else 0.0, 1.0)
            c = a.GetPositionCoordinate()
            c.SetCoordinateSystemToDisplay()
            c.SetReferenceCoordinate(ref)
            c.SetValue(dx, dy)
            return a

        for i, (text, rgb) in enumerate(lines):
            if rgb is None:
                acts = [actor(str(text), x0, y, bold=True)]
            else:
                swatch = actor("   ", x0 - 16 if right else x0, y, fill=tuple(float(c) for c in rgb[:3]))
                acts = [swatch, actor(str(text), x0 - 22 if right else x0 + 24, y)]
            for k, a in enumerate(acts):
                self.plotter.add_actor(a, name=f"_legend-colors-{i}-{k}", reset_camera=False, render=False)
            y -= step

    def _draw_labels(self, labels, dark: bool):
        """Atom labels (the view's ``labels`` setting) anchored at the atoms, drawn over them."""
        from ..overlays import atom_label_mask, atom_label_texts, label_params

        try:
            params = label_params(labels)
            idx = np.flatnonzero(atom_label_mask(self.data, params.get("types")))[: int(params.get("max_labels", 2000))]
            texts = atom_label_texts(self.data, idx, params.get("text", "{type}"))
        except (KeyError, TypeError, ValueError) as exc:
            self.picked.emit(f"Labels: {exc}")
            return
        if not len(idx):
            return
        color = params.get("color") or ("#E7E9EC" if dark else "black")
        self.plotter.add_point_labels(np.asarray(self.data.positions, float)[idx], texts,
                                      font_size=int(params.get("size") or 12), text_color=color, bold=True,
                                      show_points=False, shape_opacity=0.0, shadow=True, always_visible=True,
                                      name="_labels",
                                      render=False, pickable=False, justification_horizontal="center",
                                      justification_vertical="center")

    def _draw_measurements(self, measurements):
        """The values of the view's ``measurements`` at their lines (the lines are the Measurements element)."""
        from .. import measure
        from ..scene import MEASUREMENT_COLOR

        try:
            items = [m for m in measure.check(measurements) if max(m) < self.data.count]
        except (TypeError, ValueError, IndexError) as exc:
            self.picked.emit(f"Measurements: {exc}")
            return
        if not items:
            return
        self.plotter.add_point_labels(np.array([measure.anchor(self.data, m) for m in items]),
                                      [measure.text(self.data, m) for m in items], font_size=14,
                                      text_color=MEASUREMENT_COLOR, bold=True, show_points=False,
                                      shape_color="#14161A", shape_opacity=0.6, always_visible=True,
                                      name="_measurements", render=False, pickable=False)

    def keep_measurement(self) -> bool:
        """Add the distance or angle measured with the tool to the view's ``measurements`` (drawn in the
        viewport and pictures, written to the view cell); False when no measurement is complete."""
        need = {"distance": 2, "angle": 3}.get(self.tool)
        if need is None or len(self._picks) != need:
            return False
        if self._picks not in self.style.measurements:
            self.style.measurements = [*self.style.measurements, list(self._picks)]
        self.styleChanged.emit()   # the window redraws and writes the view cell
        return True

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


EMPTY_HINT = native_text("Open a structure, trajectory or notebook\n"
                         "File › Open (Ctrl+O), File › Open example, or drop a file here")


def _is_dark(color: str) -> bool:
    from matplotlib.colors import to_rgb

    return sum(to_rgb(color)) < 1.5
