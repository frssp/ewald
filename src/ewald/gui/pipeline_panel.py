"""Step menus (Insert › Step, the step context checks) and the View tab (:class:`DisplayPanel`): how
the structure is drawn (atoms, types, bonds, cell and boundary, plugin elements, camera) as cards."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import modifiers as mods
from . import header
from .params import ParameterForm
from .theme import TOKENS

# Add-modification menu: category -> label -> factory. Grouped like OVITO's list.
MODIFIER_MENU = {
    "Analysis": {
        "Coordination analysis / RDF": lambda: mods.CoordinationAnalysis(cutoff=3.2),
        "Displacement vectors": lambda: mods.CalculateDisplacements(0),
        "Histogram": lambda: mods.Histogram("Position.Z"),
        "Spatial binning (profiles, density maps)": lambda: mods.SpatialBinning("", "sumvol", "Z", 50),
        "Reduce property (mean, sum, ...)": lambda: mods.ReduceProperty("Position.Z"),
        "Steinhardt order parameters (q4, q6)": mods.SteinhardtOrderParameters,
    },
    "Structure identification": {
        "Common neighbor analysis": mods.CommonNeighborAnalysis,
        "Polyhedral template matching": mods.PolyhedralTemplateMatching,
        "Ackland-Jones bond angles": mods.AcklandJones,
        "Identify diamond structure": mods.IdentifyDiamond,
        "Chill+ (ice, hydrate)": mods.ChillPlus,
        "Centrosymmetry": mods.CentroSymmetry,
        "Atomic strain": mods.AtomicStrain,
        "Elastic strain (crystal)": lambda: mods.ElasticStrain(lattice_constant=3.615),
        "Grain segmentation (after PTM)": mods.GrainSegmentation,
        "fcc planar faults (after PTM)": mods.IdentifyFCCPlanarFaults,
    },
    "Symmetry & diffraction": {
        "Space group / Wyckoff positions (spglib)": mods.SymmetryAnalysis,
        "Powder XRD pattern": mods.PowderDiffraction,
        "Powder neutron diffraction": lambda: mods.PowderDiffraction(radiation="neutron", wavelength="1.54"),
        "Structure factor S(q)": mods.StructureFactor,
    },
    "Trajectory": {
        "Mean squared displacement / diffusion": mods.MeanSquaredDisplacement,
        "Velocity autocorrelation": mods.VelocityAutocorrelation,
        "Time averaging (g(r), attributes)": mods.TimeAveraging,
        "Time series of attributes": mods.TimeSeries,
        "Trajectory lines / paths": mods.GenerateTrajectoryLines,
        "NEB energy profile / barrier": mods.NebEnergyProfile,
        "Overlay frames (see-through images)": mods.OverlayFrames,
        "Unwrap trajectories": mods.UnwrapTrajectories,
        "Freeze property (values of frame 0)": mods.FreezeProperty,
        "Load trajectory (topology + dump)": mods.LoadTrajectory,
    },
    "Ion transport": {
        "Ionic conductivity / Haven ratio": mods.IonicConductivity,
        "Van Hove correlation G(r, t)": mods.VanHoveCorrelation,
        "Self intermediate scattering Fs(q, t)": mods.SelfIntermediateScattering,
        "Non-Gaussian parameter": mods.NonGaussianParameter,
        "Site hopping / residence time": mods.SiteHopping,
    },
    "Bonds & polyhedra": {
        "Create bonds": lambda: mods.CreateBonds(mode="covalent"),
        "Edit bonds (add / delete)": mods.EditBonds,
        "Coordination polyhedra": mods.CoordinationPolyhedra,
        "Bond angle distribution": mods.BondAngleDistribution,
        "Bond length distribution": mods.BondLengthDistribution,
        "Ring statistics": mods.FindRings,
    },
    "Volume (CHGCAR, LOCPOT, cube)": {
        "Isosurface": mods.CreateIsosurface,
        "Isosurface, signed (density difference / spin)": lambda: mods.CreateIsosurface(signed=True),
        "Slice plane (2D map)": mods.GridSlice,
        "Charge density difference": lambda: mods.GridDifference(["CHGCAR_A", "CHGCAR_B"]),
    },
    "Vectors (arrows)": {
        "Force / displacement / velocity arrows": mods.VectorVis,
        "Magnetic moment arrows": lambda: mods.VectorVis("Magnetic Moment", scaling=0.5, alignment="center",
                                                         color=(0.85, 0.15, 0.15)),
        "Arrows colored by length": lambda: mods.VectorVis(color_mapping_property="Magnitude"),
    },
    "Coloring": {
        "Color coding": lambda: mods.ColorCoding("Position.Z"),
        "Assign color": lambda: mods.AssignColor((1.0, 0.2, 0.2)),
    },
    "Modification": {
        "Slice": lambda: mods.Slice((0, 0, 1), 0.0),
        "Replicate": lambda: mods.Replicate((2, 2, 2)),
        "Wrap at periodic boundaries": mods.WrapAtoms,
        "Affine transformation": lambda: mods.AffineTransformation([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        "Compute property": lambda: mods.ComputeProperty("MyProperty", "Position.Z"),
        "Delete selected": mods.DeleteSelected,
        "Combine datasets (add particles from a file)": mods.CombineDatasets,
    },
    "Selection": {
        "Expression selection": lambda: mods.ExpressionSelection("Position.Z > 0"),
        "Select type": lambda: mods.SelectType("Cu"),
        "Expand selection (neighbors)": mods.ExpandSelection,
        "Invert selection": mods.InvertSelection,
        "Clear selection": mods.ClearSelection,
    },
    "Python": {
        "Python script": mods.PythonScript,
    },
}


def register_modifier(category: str, label: str, factory):
    """Add an entry to the Add-modification menu (used by plugins and new modifiers)."""
    MODIFIER_MENU.setdefault(category, {})[label] = factory


_ANIONS = {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P", "As", "H", "C"}
_VECTORS = ("Force", "Displacement", "Velocity", "Magnetic Moment", "Dipole Orientation")
_BOND_STEPS = ("BondAngleDistribution", "BondLengthDistribution", "FindRings")
_TRAJECTORY_STEPS = ("MeanSquaredDisplacement", "VelocityAutocorrelation", "TimeSeries", "TimeAveraging",
                     "GenerateTrajectoryLines", "NebEnergyProfile", "OverlayFrames", "IonicConductivity",
                     "VanHoveCorrelation", "SelfIntermediateScattering", "NonGaussianParameter", "SiteHopping")


def step_unavailable(mod, data, pipeline=None) -> str | None:
    """Why step ``mod`` (as Insert › Step creates it) cannot work on ``data`` (the current frame,
    after the pipeline), or None. The menus grey such steps out with this reason as tooltip."""
    if data is None:
        return None
    name = type(mod).__name__
    frames = pipeline.num_frames if pipeline is not None else 1
    if name in _TRAJECTORY_STEPS and frames < 2:
        return "Needs a trajectory: the data has a single frame"
    if name in ("CreateIsosurface", "GridSlice", "GridDifference") and not data.grids:
        return "Needs a volumetric grid (CHGCAR, LOCPOT, cube)"
    if name == "VectorVis":
        prop = getattr(mod, "property", None)
        wanted = [prop] if prop else list(_VECTORS)
        if not any(p in data.particles for p in wanted):
            return f"Needs a vector property ({', '.join(wanted)})"
    if name in ("GrainSegmentation", "IdentifyFCCPlanarFaults") and "Orientation" not in data.particles:
        return "Needs Polyhedral template matching with orientations first"
    if name == "TimeAveraging" and str(getattr(mod, "operate_on", "")).startswith("table:"):
        table = mod.operate_on.split(":", 1)[1]
        if table not in data.tables:
            return f"Needs the {table!r} table first (Coordination analysis / RDF)"
    if name == "FreezeProperty" and getattr(mod, "source_property", "Selection") not in data.particles:
        return f"Needs the {getattr(mod, 'source_property', 'Selection')!r} property (select atoms first)"
    if name == "CoordinationPolyhedra" and _polyhedra_centers(data) is None and "Selection" not in data.particles:
        return "Needs center atoms: select them first (no anion to build polyhedra around)"
    return None


def _polyhedra_centers(data):
    """Default polyhedron centers: the smallest non-anion type of a compound with anions (Ti in SrTiO3)."""
    from ..radii import _table_radius, element_of

    types = [str(t) for t in dict.fromkeys(np.asarray(data.types).astype(str))]
    elements = {t: element_of(t) for t in types}
    cations = [t for t in types if elements[t] and elements[t] not in _ANIONS]
    if not cations or len(cations) == len(types):
        return None
    return [min(cations, key=lambda t: _table_radius(elements[t], "covalent") or 9.0)]


def prepare_step(mod, data) -> list:
    """The steps to add for ``mod``: bond analyses get ``CreateBonds`` (covalent, like the view)
    first when the data has no bonds; Coordination polyhedra get default center types."""
    if data is None:
        return [mod]
    name = type(mod).__name__
    if name in _BOND_STEPS and data.bonds is None:
        return [mods.CreateBonds(mode="covalent"), mod]
    if name == "CoordinationPolyhedra" and mod.center_types is None and "Selection" not in data.particles:
        mod.center_types = _polyhedra_centers(data)
    return [mod]


def fill_modifier_menu(menu, add, context=None):
    """(Re)populate ``menu`` with MODIFIER_MENU; ``add`` receives the new modifier. ``context``
    returns ``(data, pipeline)``: steps that cannot work on that data are greyed out with the
    reason, and the added steps get what they need first (:func:`prepare_step`)."""
    menu.clear()
    for sub in menu.findChildren(QtWidgets.QMenu, options=QtCore.Qt.FindDirectChildrenOnly):
        sub.deleteLater()
    data, pipeline = context() if context is not None else (None, None)

    def run(factory):
        for m in prepare_step(factory(), data):
            add(m)

    for group, items in MODIFIER_MENU.items():
        sub = menu.addMenu(group.replace("&", "&&"))  # a single "&" would be a mnemonic
        sub.setToolTipsVisible(True)
        for label, factory in items.items():
            act = sub.addAction(label, lambda f=factory: run(f))
            try:
                reason = step_unavailable(factory(), data, pipeline)
            except Exception:
                reason = None
            if reason:
                act.setEnabled(False)
                act.setToolTip(reason)


def _discover():
    """Pick up modifiers other threads add to ewald.modifiers without touching this file."""
    known = {type(f()).__name__ for items in MODIFIER_MENU.values() for f in items.values()}
    for name in getattr(mods, "__all__", []):
        cls = getattr(mods, name, None)
        if (isinstance(cls, type) and issubclass(cls, mods.Modifier) and cls not in (mods.Modifier, mods.PythonModifier)
                and name not in known):
            try:
                cls()
            except Exception:
                continue  # needs arguments we cannot guess
            register_modifier("Analysis" if "Analysis" in name or "Structure" in name else "Other", name, cls)
    from .. import plugins

    for name, plugin in plugins.modifiers().items():
        if name in known:
            continue
        try:
            plugin.cls()
        except Exception:
            continue  # needs arguments we cannot guess
        register_modifier(plugin.category, name, plugin.cls)


class FitWidthScrollArea(QtWidgets.QScrollArea):
    """Scrolls vertically only: the page is always as wide as the area, like a document."""

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.widget() is not None:
            self.widget().setFixedWidth(self.viewport().width())

    def setWidget(self, widget):
        super().setWidget(widget)
        if widget is not None:
            widget.setFixedWidth(self.viewport().width())


class FlowLayout(QtWidgets.QLayout):
    """Lays its items out in rows that wrap at the available width (the add-cell bar, the View
    tab's cards). An item wider than the row is narrowed to it."""

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self._items = []
        self._gap = spacing

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientations()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._place(QtCore.QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._place(rect, move=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _place(self, rect, move):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, row_h = r.x(), r.y(), 0
        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            w = max(1, min(hint.width(), r.width()))
            h = item.heightForWidth(w) if item.hasHeightForWidth() else hint.height()
            if x + w > r.right() + 1 and row_h:
                x, y, row_h = r.x(), y + row_h + self._gap, 0
            if move:
                item.setGeometry(QtCore.QRect(x, y, w, h))
            x += w + self._gap
            row_h = max(row_h, h)
        return y + row_h - rect.y() + m.bottom()


class MasonryLayout(QtWidgets.QLayout):
    """The View tab's cards in equal-width columns: as many columns of at least ``column`` px as fit,
    stretched to the full width, each card going into the shortest column so the columns end
    close together. Cards keep their order within a column."""

    def __init__(self, parent=None, spacing=10, column=300):
        super().__init__(parent)
        self._items = []
        self._gap = spacing
        self._column = column

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientations()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._place(QtCore.QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._place(rect, move=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        m = self.contentsMargins()
        w = min(self._column, max((i.minimumSize().width() for i in self._items), default=0))
        return QtCore.QSize(w + m.left() + m.right(), m.top() + m.bottom())

    def columns(self, width: int) -> int:
        m = self.contentsMargins()
        inner = width - m.left() - m.right()
        shown = sum(not i.isEmpty() for i in self._items)
        return max(1, min(shown, (inner + self._gap) // (self._column + self._gap)))

    def _place(self, rect, move):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        n = self.columns(rect.width())
        w = max(1, (r.width() - (n - 1) * self._gap) // n)
        tops = [r.y()] * n
        for item in self._items:
            if item.isEmpty():
                continue
            h = item.heightForWidth(w) if item.hasHeightForWidth() else item.sizeHint().height()
            k = tops.index(min(tops))
            if move:
                item.setGeometry(QtCore.QRect(r.x() + k * (w + self._gap), tops[k], w, h))
            tops[k] += h + self._gap
        return max(tops) - self._gap - rect.y() + m.bottom()


class _FlowPage(QtWidgets.QWidget):
    """The page of a :class:`FitWidthScrollArea` whose layout wraps: as tall as its rows."""

    def _fit(self):
        h = self.layout().heightForWidth(self.width())
        if h != self.minimumHeight():
            self.setMinimumHeight(h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def event(self, event):
        done = super().event(event)
        if event.type() == QtCore.QEvent.LayoutRequest and self.layout() is not None:
            self._fit()        # a card folded or grew
        return done


class _CardHead(QtWidgets.QWidget):
    """A card's title row; a click anywhere on it folds the card."""

    clicked = QtCore.Signal()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class _Group(QtWidgets.QWidget):
    """A switchable sub-section of a View tab card (Replicate cell, Boundary, Light): a rule, a check box
    title, and rows that show only while it is checked. Answers like a checkable QGroupBox."""

    toggled = QtCore.Signal(bool)

    def __init__(self, title, checked, tip):
        super().__init__(objectName="viewGroup", toolTip=tip)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(8)
        rule = QtWidgets.QFrame(objectName="viewRule")
        rule.setFixedHeight(1)
        lay.addWidget(rule)
        self.check = QtWidgets.QCheckBox(title, checked=checked, objectName="viewGroupTitle")
        lay.addWidget(self.check)
        self.rows = QtWidgets.QWidget(objectName="viewBody")
        self.rows.setVisible(checked)
        lay.addWidget(self.rows)
        self.check.toggled.connect(self.rows.setVisible)
        self.check.toggled.connect(self.toggled)

    def title(self):
        return self.check.text()

    def isChecked(self):
        return self.check.isChecked()

    def setChecked(self, on):
        self.check.setChecked(on)

    def event(self, event):
        if event.type() == QtCore.QEvent.DynamicPropertyChange:   # marked "set": restyle the title
            self.check.setProperty("set", self.property("set"))
            self.check.style().unpolish(self.check)
            self.check.style().polish(self.check)
        return super().event(event)


class _ElidedCheckBox(QtWidgets.QCheckBox):
    """A check box whose label is shortened with … in the middle when the cell is too narrow."""

    def minimumSizeHint(self):
        return QtCore.QSize(40, super().minimumSizeHint().height())

    def paintEvent(self, event):
        opt = QtWidgets.QStyleOptionButton()
        self.initStyleOption(opt)
        label = self.style().subElementRect(QtWidgets.QStyle.SE_CheckBoxContents, opt, self)
        opt.text = self.fontMetrics().elidedText(self.text(), QtCore.Qt.ElideMiddle, max(0, label.width()))
        QtWidgets.QStylePainter(self).drawControl(QtWidgets.QStyle.CE_CheckBox, opt)


class _Card(QtWidgets.QFrame):
    """A View tab card: a foldable title row over its settings, as wide as its column."""

    def __init__(self):
        super().__init__(objectName="viewSection")
        self.setMinimumWidth(200)
        self.folded = False


# View tab cards, in order -> the pipe.view settings they show (marked when the notebook sets one)
_SECTION_KEYS = {
    "Atoms": ("style", "theme", "atom_style", "radius_scale", "radius_preset", "oxidation_states", "atoms"),
    "Types": ("colors", "radii", "transparency"),
    "Bonds": ("bonds", "bond_radius", "bond_pairs", "bond_factor"),
    "Surfaces": ("surfaces",),
    "Cell": ("cell", "cell_width", "cell_color", "replicate", "replicate_cell", "boundary", "boundary_bonds"),
    "Labels": ("labels", "selection"),
    "Camera": ("frame", "direction", "zoom", "orthographic", "camera", "light"),
    "Elements": ("visuals", "vectors", "trajectory_width", "trajectory_color_by", "trajectory_colormap",
                 "trajectory_color_range", "trajectory_time_range", "trajectory_trail"),
}
# widest form label of the View tab: every card's label column is this wide, so fields line up
_LABEL_WIDEST = "Radius scale"


def _label_text(labels) -> str:
    """The Labels field for a ``labels`` setting."""
    if labels is None or labels is False:
        return ""
    if labels is True:
        return "{type}"
    return str(labels.get("text", "{type}")) if isinstance(labels, dict) else str(labels)


class DisplayPanel(QtWidgets.QWidget):
    """The View tab: how the structure is drawn, as foldable cards (Atoms, Types, Bonds, Surfaces, Cell,
    Labels, Camera, Elements) in equal-width columns below the 3D view, one column when the tab is
    narrow. Every form shares one label column width so fields line up across cards. Every value is a
    ``pipe.view`` setting; the ones the notebook's view cell sets are marked, and a change made
    here goes back into that cell. Steps live in notebook cells; this tab only changes the look."""

    styleChanged = QtCore.Signal()               # visual elements changed -> redraw only
    modifierRequested = QtCore.Signal(object)    # e.g. "Use as Create bonds modifier": a notebook step
    cameraRequested = QtCore.Signal()            # Camera › Save camera: write it into the view cell

    def __init__(self, style=None, parent=None):
        super().__init__(parent)
        from .style import ViewStyle

        _discover()
        self.pipeline = None
        self.style = style or ViewStyle()
        self.data = None
        self._emitting = False
        self._marks = []          # (label, keys): marked when the notebook's view sets one of keys
        self._forms = {}          # plugin visual index -> its ParameterForm
        self._form = None         # the last plugin visual form built (tests)
        self.folded = set()       # titles of folded cards; kept across rebuilds
        self._label_w = self.fontMetrics().horizontalAdvance(_LABEL_WIDEST) + 8
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.visual_menu = QtWidgets.QMenu(self)
        self.visual_menu.aboutToShow.connect(self._fill_visual_menu)
        self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)   # right-click on a card title or between cards
        self.addAction(QtGui.QAction("Fold all cards", self, triggered=lambda: self.fold_all(True)))
        self.addAction(QtGui.QAction("Unfold all cards", self, triggered=lambda: self.fold_all(False)))
        self.scroll = FitWidthScrollArea(objectName="viewScroll")
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        lay.addWidget(self.scroll, 1)
        self.sections = {}

    # ------------------------------------------------------------------ model
    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        self.refresh()

    def set_data(self, data, refresh: bool = False):
        old = _types(self.data)
        self.data = data
        if refresh or _types(data) != old:  # another file, or code changed pipe.view: rebuild
            self.refresh()
        else:
            self.update_status()

    def _emit(self):
        """A change made here: redraw and write it into the view cell, without rebuilding the
        cards (the widget being edited stays), then mark what the notebook now sets."""
        self._emitting = True
        try:
            self.styleChanged.emit()
        finally:
            self._emitting = False
        self.update_marks()

    def refresh(self, select=None):
        """Rebuild the cards from the style (``select`` is accepted for older callers)."""
        if self._emitting:
            self.update_marks()
            return
        self._marks, self._forms = [], {}
        fm = self.fontMetrics()
        self._label_w = fm.horizontalAdvance(_LABEL_WIDEST) + 8
        page = _FlowPage(objectName="viewPage")
        flow = MasonryLayout(page, spacing=10, column=max(280, fm.averageCharWidth() * 40))
        flow.setContentsMargins(10, 10, 10, 10)
        self.sections = {}
        builders = {"Atoms": self._particles_editor, "Types": self._types_editor, "Bonds": self._bonds_editor,
                    "Surfaces": self._surfaces_editor, "Cell": self._cell_editor, "Labels": self._labels_editor,
                    "Camera": self._camera_editor, "Elements": self._elements_editor}
        for title in _SECTION_KEYS:
            card = self._section(title, builders[title]())
            self.sections[title] = card
            flow.addWidget(card)
        old = self.scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self.scroll.setWidget(page)
        self.update_status()
        self.update_marks()

    def _section(self, title, body):
        card = _Card()
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(12, 6, 12, 12)
        lay.setSpacing(8)
        head = _CardHead(objectName="viewSectionHead", toolTip=f"Fold or unfold {title}")
        head.setCursor(QtCore.Qt.PointingHandCursor)
        row = QtWidgets.QHBoxLayout(head)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        fold = header.button("chevron_down", f"Fold {title}", name="viewFold")
        fold.setFixedSize(20, 20)
        fold.setIconSize(QtCore.QSize(14, 14))
        name = QtWidgets.QLabel(title.upper(), objectName="viewSectionTitle")
        mark = QtWidgets.QLabel("", objectName="viewSectionMark")
        row.addWidget(fold)
        row.addWidget(name)
        row.addStretch(1)
        row.addWidget(mark)
        lay.addWidget(head)
        content = QtWidgets.QWidget(objectName="viewBody")
        inner = QtWidgets.QVBoxLayout(content)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(8)
        body.setObjectName("viewBody")
        inner.addWidget(body)
        card.status = QtWidgets.QLabel("", objectName="viewStatus", wordWrap=True)
        card.status.hide()
        inner.addWidget(card.status)
        lay.addWidget(content)
        card.content, card.fold = content, fold
        head.clicked.connect(lambda t=title: self.set_folded(t, t not in self.folded))
        fold.clicked.connect(lambda _=False, t=title: self.set_folded(t, t not in self.folded))
        self._show_folded(card, title in self.folded)
        self._marks.append((mark, _SECTION_KEYS[title]))
        return card

    def set_folded(self, title: str, folded: bool):
        """Fold (only the title row shows) or unfold the card ``title``; stays so across rebuilds."""
        (self.folded.add if folded else self.folded.discard)(title)
        if title in self.sections:
            self._show_folded(self.sections[title], folded)

    def fold_all(self, folded: bool = True):
        """Fold (or unfold) every card; the window keeps the folded cards for the next start."""
        for title in list(self.sections) or list(_SECTION_KEYS):
            self.set_folded(title, folded)

    @staticmethod
    def _show_folded(card, folded):
        card.folded = folded
        card.content.setVisible(not folded)
        header.set_icon(card.fold, "chevron_right" if folded else "chevron_down")
        card.fold.setToolTip(("Unfold " if folded else "Fold ") + card.fold.toolTip().split(" ", 1)[1])

    def _form_layout(self, parent=None, top=0):
        """A form whose label column is as wide as every other card's."""
        f = QtWidgets.QFormLayout(parent)
        f.setContentsMargins(0, top, 0, 0)
        f.setHorizontalSpacing(10)
        f.setVerticalSpacing(6)
        f.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        f.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        f.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        return f

    def _group(self, title, checked, tip):
        """A switchable sub-section of a card (Replicate cell, Boundary, Light): a title with a check box
        over a rule, its rows in the card's own label and field columns."""
        box = _Group(title, checked, tip)
        return box, self._form_layout(box.rows)

    def _buttons(self, *buttons):
        """A row of card actions, left-aligned."""
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for b in buttons:
            row.addWidget(b)
        row.addStretch(1)
        return row

    @staticmethod
    def _fit_table(table, rows=8):
        """A table exactly as tall as its header and up to ``rows`` rows: no empty band below the rows."""
        table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        table.setShowGrid(False)
        table.setAlternatingRowColors(False)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setFocusPolicy(QtCore.Qt.ClickFocus)
        vh = table.verticalHeader()
        vh.setDefaultSectionSize(max(26, table.fontMetrics().height() + 12))
        n = min(rows, table.rowCount())
        h = table.horizontalHeader().sizeHint().height() + n * vh.defaultSectionSize() + 2 * table.frameWidth()
        table.setFixedHeight(h)

    @staticmethod
    def _cell_spin(spin):
        """A number edited inside a table cell, right-aligned; theme.py hides its arrows (wheel and keys
        still step). Not ``NoButtons``: with a style sheet that leaves the editor 1 px wide."""
        spin.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return spin

    @staticmethod
    def _numeric_headers(table, *columns):
        for c in columns:
            table.horizontalHeaderItem(c).setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

    @staticmethod
    def _swatch(rgb, tip):
        """A color chip for a table cell; returns (cell widget, button)."""
        cell = QtWidgets.QWidget(objectName="viewCell")
        lay = QtWidgets.QHBoxLayout(cell)
        lay.setContentsMargins(6, 4, 6, 4)
        btn = QtWidgets.QPushButton(toolTip=tip, objectName="viewSwatch")
        btn.setFixedHeight(16)
        btn.setMinimumWidth(24)
        _paint_swatch(btn, QtGui.QColor.fromRgbF(*np.clip(rgb, 0, 1)).name())
        lay.addWidget(btn)
        return cell, btn

    def _row(self, form, label, widget, *keys):
        """A form row whose label is marked when the notebook's view sets one of ``keys``."""
        lab = QtWidgets.QLabel(label, objectName="viewLabel")
        lab.setFixedWidth(self._label_w)
        form.addRow(lab, widget)
        if keys:
            self._marks.append((lab, keys))
        return lab

    def _set_keys(self) -> dict:
        view = getattr(self.pipeline, "view", None) if self.pipeline is not None else None
        try:
            return view.changed() if view is not None else {}
        except Exception:  # noqa: BLE001 - a view a plugin replaced
            return {}

    def update_marks(self):
        """Mark rows and cards whose settings the notebook's view cell sets (tooltip: the values)."""
        values = self._set_keys()
        for lab, keys in self._marks:
            try:
                set_here = [k for k in keys if k in values]
            except RuntimeError:  # a card deleted by a rebuild
                continue
            if lab.objectName() == "viewSectionMark":
                lab.setText("in notebook" if set_here else "")
            lab.setProperty("set", bool(set_here))
            lab.setToolTip("Set in the notebook's view cell:\n" + "\n".join(
                f"{k}={_short(values[k])}" for k in set_here) if set_here else "Default (not in the notebook)")
            lab.style().unpolish(lab)
            lab.style().polish(lab)

    def update_status(self):
        """Counts and periodicity of the current frame under the cards (no rebuild)."""
        if self.data is None or not self.sections:
            return
        types = self.sections["Types"].status
        types.setText(f"{self.data.count:,} atoms")
        types.show()
        cell = self.sections["Cell"].status
        if self.data.cell.is_valid:
            cell.setText(cell_summary(self.data.cell))
            cell.show()

    # ------------------------------------------------------------------ editing
    def add_modifier(self, mod):
        """Buttons here that make a step (Use as Create bonds modifier) ask the notebook for it."""
        self.modifierRequested.emit(mod)

    def _fill_visual_menu(self):
        from .. import plugins

        self.visual_menu.clear()
        extra = [p for p in plugins.visuals().values() if not p.builtin]
        if not extra:
            act = self.visual_menu.addAction("No visual element plugins installed")
            act.setEnabled(False)
            return
        for p in sorted(extra, key=lambda p: (p.category, p.name)):
            act = self.visual_menu.addAction(p.name)
            act.setToolTip(p.description)
            act.triggered.connect(lambda _=False, p=p: self.add_visual(p.name))

    def add_visual(self, name: str):
        """Add a registered visual element (constructed with its defaults) to the style."""
        from ..scene import coerce_visuals, visual_to_dict

        (el,) = coerce_visuals([name])
        self.style.visuals.append(visual_to_dict(el))
        self.styleChanged.emit()
        self.refresh()
        self._form = self._forms.get(len(self.style.visuals) - 1)

    def remove_visual(self, k: int):
        del self.style.visuals[k]
        self.styleChanged.emit()
        self.refresh()

    def set_visual_enabled(self, k: int, on: bool):
        self.style.visuals[k]["enabled"] = on
        self._emit()

    def _set(self, name, value):
        setattr(self.style, name, value)
        self._emit()

    # ------------------------------------------------------------------ cards
    def _elements_editor(self):
        """Plugin visual elements (cell axes, halos, ...): a switch, their parameters and Remove."""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._trajectory_editor())
        for k, entry in enumerate(self.style.visuals):
            lay.addWidget(self._plugin_visual_editor(k, entry))
        if not self.style.visuals:
            lay.addWidget(QtWidgets.QLabel("Cell axes, halos and other drawn extras from plugins.",
                                           objectName="muted", wordWrap=True))
        add = QtWidgets.QPushButton("Add element", objectName="viewAdd",
                                    toolTip="Add a visual element (cell axes, halos, … from plugins)")
        add.setMenu(self.visual_menu)
        self.visual_button = add
        lay.addLayout(self._buttons(add))
        return w

    def _trajectory_editor(self):
        """Trajectory lines (a GenerateTrajectoryLines step): width, and a color map by Time, Frame, step
        length or a property, over a Time window or a trail that follows the current frame."""
        from ..colormaps import names as colormap_names
        from ..scene import TRAJECTORY_COLOR_BY

        st = self.style
        w = QtWidgets.QWidget()
        f = self._form_layout(w)
        width = QtWidgets.QDoubleSpinBox(minimum=0, maximum=20, singleStep=0.5, value=st.trajectory_width,
                                         suffix=" px", toolTip="Line width in pixels; 0 hides the trajectory lines")
        width.valueChanged.connect(lambda v: self._set("trajectory_width", float(v)))
        self._row(f, "Trajectories", width, "trajectory_width")
        box, g = self._group("Color map", st.trajectory_color_by is not None,
                             "Color the trajectory lines along their path by Time, Frame, step length or a "
                             "particle property, with a color bar")
        by = QtWidgets.QComboBox(editable=True, toolTip="Time (timestep), Frame, Step Length (distance per step), "
                                 "a column of the trajectory-lines table or a numeric particle property")
        extra = []
        if self.data is not None:
            table = self.data.tables.get("trajectory-lines", {})
            extra = [c for c in table if c not in ("Position", "Particle Identifier", *TRAJECTORY_COLOR_BY)]
            extra += [n for n, v in self.data.particles.items()
                      if np.ndim(v) == 1 and np.issubdtype(np.asarray(v).dtype, np.number) and n not in extra]
        by.addItems([*TRAJECTORY_COLOR_BY, *extra])
        by.setCurrentText(st.trajectory_color_by or "Time")
        cmap = QtWidgets.QComboBox()
        cmap.addItems(colormap_names())
        if st.trajectory_colormap not in colormap_names():
            cmap.addItem(st.trajectory_colormap)
        cmap.setCurrentText(st.trajectory_colormap)

        def pair(value, tips):
            row, edits = QtWidgets.QHBoxLayout(), []
            row.setSpacing(6)
            for k, tip in enumerate(tips):
                e = QtWidgets.QLineEdit(placeholderText="auto", toolTip=tip)
                if value is not None and value[k] is not None:
                    e.setText(f"{value[k]:g}")
                row.addWidget(e, 1)
                edits.append(e)
            return _wrap(row), edits

        rng_row, rng = pair(st.trajectory_color_range, ("Value at the low end of the map; empty: automatic",
                                                         "Value at the high end of the map; empty: automatic"))
        win_row, win = pair(st.trajectory_time_range, ("Draw from this Time (timestep); empty: the start",
                                                       "Draw up to this Time (timestep); empty: the end"))
        trail = QtWidgets.QLineEdit(placeholderText="whole path", clearButtonEnabled=True,
                                    toolTip="Follow the current frame: 'all' draws the paths up to it, a number only "
                                            "that much Time before it (a tail); empty: the whole paths")
        if st.trajectory_trail is True:
            trail.setText("all")
        elif st.trajectory_trail:
            trail.setText(f"{st.trajectory_trail:g}")

        def number(edit):
            try:
                return float(edit.text()) if edit.text().strip() else None
            except ValueError:
                return None

        def both(edits):
            lo, hi = (number(e) for e in edits)
            if lo is not None and hi is not None and lo > hi:
                lo, hi = hi, lo
            return None if lo is None and hi is None else [lo, hi]

        def apply(*_):
            text = trail.text().strip().lower()
            tail = True if text in ("all", "true", "yes") else number(trail)
            values = {"trajectory_color_by": by.currentText().strip() or None if box.isChecked() else None,
                      "trajectory_colormap": cmap.currentText(), "trajectory_color_range": both(rng),
                      "trajectory_time_range": both(win), "trajectory_trail": tail if tail else None}
            if all(getattr(st, k) == v for k, v in values.items()):
                return
            for k, v in values.items():
                setattr(st, k, v)
            self._emit()

        box.toggled.connect(apply)
        by.currentTextChanged.connect(apply)
        cmap.currentTextChanged.connect(apply)
        for e in (*rng, *win, trail):
            e.editingFinished.connect(apply)
        self._row(g, "Color by", by, "trajectory_color_by")
        self._row(g, "Color map", cmap, "trajectory_colormap")
        self._row(g, "Range", rng_row, "trajectory_color_range")
        self._marks.append((box, ("trajectory_color_by",)))
        f.addRow(box)
        self._row(f, "Time window", win_row, "trajectory_time_range")
        self._row(f, "Trail", trail, "trajectory_trail")
        self.trajectory_widgets = {"width": width, "group": box, "color_by": by, "colormap": cmap, "range": rng,
                                   "window": win, "trail": trail}
        return w

    def _plugin_visual_editor(self, k, entry):
        """Parameter form of a plugin visual element, built from its constructor like a modifier's."""
        from ..scene import coerce_visuals, visual_to_dict

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 4)
        head = QtWidgets.QHBoxLayout()
        on = QtWidgets.QCheckBox(entry["name"], checked=entry.get("enabled", True))
        on.toggled.connect(lambda v, k=k: self.set_visual_enabled(k, v))
        head.addWidget(on, 1)
        rm = header.button("delete", f"Remove {entry['name']}", name="headerButton")
        rm.clicked.connect(lambda _=False, k=k: self.remove_visual(k))
        head.addWidget(rm)
        lay.addLayout(head)
        try:
            (el,) = coerce_visuals([entry])
        except Exception as exc:  # plugin no longer installed, or its parameters changed
            lay.addWidget(QtWidgets.QLabel(f"{type(exc).__name__}: {exc}", wordWrap=True, objectName="viewError"))
            return w
        form = self._forms[k] = self._form = ParameterForm(el)
        lay.addWidget(form)

        def apply():
            form.apply()
            self.style.visuals[k] = {**visual_to_dict(el), "enabled": self.style.visuals[k].get("enabled", True)}
            self._emit()

        button = QtWidgets.QPushButton("Apply  (Ctrl+Enter)")
        button.setObjectName("primary")
        button.clicked.connect(apply)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), w, activated=apply)
        lay.addWidget(button, 0, QtCore.Qt.AlignLeft)
        return w

    def _particles_editor(self):
        from ..radii import IONIC, PRESETS, format_oxidation_states
        from ..view3d import STYLE_PRESETS

        w = QtWidgets.QWidget()
        f = self._form_layout(w)
        style = QtWidgets.QComboBox()
        style.addItems(list(STYLE_PRESETS))
        style.setCurrentText(self.style.preset)
        style.currentTextChanged.connect(self._set_style_preset)
        self._row(f, "Style", style, "style", "theme")
        scale = QtWidgets.QDoubleSpinBox(minimum=0.05, maximum=5, singleStep=0.05, value=self.style.radius_scale)
        scale.valueChanged.connect(lambda v: self._set("radius_scale", v))
        self._row(f, "Radius scale", scale, "radius_scale")
        preset = QtWidgets.QComboBox()
        preset.addItems(list(PRESETS))
        for k, cite in enumerate(PRESETS.values()):
            preset.setItemData(k, cite, QtCore.Qt.ToolTipRole)
        preset.setCurrentText(self.style.radius_preset)
        preset.currentTextChanged.connect(self._set_radius_preset)
        self._row(f, "Radii", preset, "radius_preset")
        oxi = QtWidgets.QLineEdit(placeholderText="auto, e.g. Sr=2, Ti=4, O=-2")
        oxi.setText(format_oxidation_states(self.style.oxidation_states))
        oxi.setToolTip("Oxidation state per type for the ionic presets; empty types are guessed "
                       "(type name, charges, bond valence, charge balance)")
        oxi.editingFinished.connect(lambda e=oxi: self._set_oxidation_states(e))
        oxi.setEnabled(self.style.radius_preset in IONIC)
        self._row(f, "Oxidation", oxi, "oxidation_states")
        shape = QtWidgets.QComboBox()
        shape.addItems(["auto", "mesh", "impostor", "flat"])
        shape.setCurrentText(self.style.atom_style)
        shape.setToolTip("mesh: best shading for figures; impostor: GPU sprites for millions of atoms; "
                         "flat: ASE-GUI look, unshaded discs with outlines")
        shape.currentTextChanged.connect(lambda v: self._set("atom_style", v))
        self._row(f, "Drawn as", shape, "atom_style")
        show = QtWidgets.QCheckBox("Draw atoms", checked=bool(self.style.show_atoms),
                                   toolTip="Off: bonds, polyhedra and the rest stay (a stick model)")
        show.toggled.connect(lambda v: self._set("show_atoms", v))
        self._row(f, "", show, "atoms")
        return w

    def _labels_editor(self):
        """Text on each atom and the selection highlight."""
        w = QtWidgets.QWidget()
        f = self._form_layout(w)
        labels = QtWidgets.QLineEdit(placeholderText="off, e.g. {type}{index}", clearButtonEnabled=True)
        labels.setText(_label_text(self.style.labels))
        labels.setToolTip("A label on each atom: {type}, {index}, {id} or a per-atom property such as "
                          "{Charge:.2f}; empty: no labels")
        labels.editingFinished.connect(lambda e=labels: self._set_labels(e.text()))
        self._row(f, "Text", labels, "labels")
        sel = QtWidgets.QCheckBox("Highlight selection", checked=self.style.selection is not False,
                                  toolTip="Draw the atoms a selection step selected in red, as OVITO does")
        sel.toggled.connect(lambda v: self._set("selection", bool(v)))
        self._row(f, "Selection", sel, "selection")
        return w

    def _set_labels(self, text: str):
        text = text.strip()
        value = None if not text else (True if text == "{type}" else text)
        if value is not None and isinstance(self.style.labels, dict):
            value = {**self.style.labels, "text": text}   # keep types, size and color set in code
        if value == self.style.labels:
            return
        self._set("labels", value)

    def _surfaces_editor(self):
        """Polyhedra, isosurfaces and slices (``data.surfaces``) by name: shown, color, transparency,
        edges; the view's ``surfaces`` setting, so the step that made them stays as it is."""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        names = [] if self.data is None else [n for n, m in getattr(self.data, "surfaces", {}).items() if m.face_count]
        if not names:
            lay.addWidget(QtWidgets.QLabel("Polyhedra, isosurfaces and slices that steps make appear here.",
                                           objectName="muted", wordWrap=True))
            return w
        table = QtWidgets.QTableWidget(len(names), 4, objectName="viewTable")
        table.setHorizontalHeaderLabels(["Surface", "Color", "Transp.", "Edges"])
        table.verticalHeader().setVisible(False)
        head = table.horizontalHeader()
        head.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        head.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        head.setDefaultAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        for r, name in enumerate(names):
            mesh = self.data.surfaces[name]
            st = self.style.surfaces.get(name, {})
            shown = _ElidedCheckBox(name, checked=st.get("visible", True), toolTip=f"Draw {name}")
            shown.toggled.connect(lambda v, n=name: self.set_surface(n, visible=v))
            table.setCellWidget(r, 0, shown)
            color = st.get("color")
            if color is None:
                fc = mesh.face_colors() if mesh.face_count else np.array([[0.6, 0.6, 0.6]])
                rgb = np.clip(np.asarray(fc, float).reshape(-1, 3).mean(0), 0, 1)
            else:
                from ..render import _rgb

                rgb = np.clip(_rgb(color), 0, 1)
            cell, btn = self._swatch(rgb, f"Color of {name} (Shift+click: the step's colors)")
            btn.clicked.connect(lambda _=False, n=name, b=btn: self._pick_surface_color(n, b))
            table.setCellWidget(r, 1, cell)
            tr = QtWidgets.QDoubleSpinBox(minimum=0, maximum=1, singleStep=0.05,
                                          value=float(st.get("transparency", mesh.transparency)))
            tr.valueChanged.connect(lambda v, n=name: self.set_surface(n, transparency=round(v, 3)))
            table.setCellWidget(r, 2, self._cell_spin(tr))
            edges = QtWidgets.QCheckBox(checked=st.get("edges", mesh.edges is not None and len(mesh.edges) > 0))
            edges.setEnabled(mesh.edges is not None and len(mesh.edges) > 0)
            edges.toggled.connect(lambda v, n=name: self.set_surface(n, edges=v))
            table.setCellWidget(r, 3, _centered(edges))
        self._numeric_headers(table, 2)
        self._fit_table(table)
        lay.addWidget(table)
        reset = QtWidgets.QPushButton("Reset", toolTip="Back to the look the steps gave the surfaces")
        reset.clicked.connect(lambda: (self.style.surfaces.clear(), self.styleChanged.emit(), self.refresh()))
        lay.addLayout(self._buttons(reset))
        return w

    def set_surface(self, name: str, **style):
        """Restyle the surface ``name`` (visible, color, transparency, edges) and redraw."""
        self.style.surfaces = {**self.style.surfaces, name: {**self.style.surfaces.get(name, {}), **style}}
        self._emit()

    def _pick_surface_color(self, name, button):
        if QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier:
            st = {k: v for k, v in self.style.surfaces.get(name, {}).items() if k != "color"}
            self.style.surfaces = {**{k: v for k, v in self.style.surfaces.items() if k != name},
                                   **({name: st} if st else {})}
            self._emit()
            self.refresh()
            return
        c = QtWidgets.QColorDialog.getColor(parent=self)
        if c.isValid():
            _paint_swatch(button, c.name())
            self.set_surface(name, color=c.name())

    def _types_editor(self):
        from ..render import type_color, type_radius

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        types = [] if self.data is None else list(dict.fromkeys(self.data.types))
        table = QtWidgets.QTableWidget(len(types), 5, objectName="viewTable")
        table.setHorizontalHeaderLabels(["Type", "Color", "Radius", "Transp.", "Count"])
        table.verticalHeader().setVisible(False)
        head = table.horizontalHeader()
        head.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        for c in (0, 1, 4):
            head.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        head.setMinimumSectionSize(44)
        head.setDefaultAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        for r, t in enumerate(types):
            item = QtWidgets.QTableWidgetItem(t)
            if t in self.style.type_colors or t in self.style.type_radii:
                item.setToolTip("Color or radius set in the notebook")
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            table.setItem(r, 0, item)
            color = self.style.type_colors.get(t)
            rgb = type_color(t, r) if color is None else np.asarray(QtGui.QColor(color).getRgbF()[:3] if isinstance(color, str) else color)
            cell, btn = self._swatch(rgb, f"Color of {t}")
            btn.clicked.connect(lambda _=False, t=t, b=btn: self._pick_color(t, b))
            table.setCellWidget(r, 1, cell)
            rad = QtWidgets.QDoubleSpinBox(minimum=0.05, maximum=5, singleStep=0.05,
                                           value=self.style.type_radii.get(t, type_radius(t, preset=self.style.radius_preset)))
            rad.valueChanged.connect(lambda v, t=t: (self.style.type_radii.__setitem__(t, v), self._emit()))
            table.setCellWidget(r, 2, self._cell_spin(rad))
            tr = QtWidgets.QDoubleSpinBox(minimum=0, maximum=1, singleStep=0.1,
                                          value=float(self.style.type_transparency.get(t, 0.0)),
                                          toolTip=f"Transparency of {t}: 0 opaque, 1 hidden")
            tr.valueChanged.connect(lambda v, t=t: self.set_type_transparency(t, v))
            table.setCellWidget(r, 3, self._cell_spin(tr))
            count = QtWidgets.QTableWidgetItem(f"{int((self.data.types == t).sum()):,}")
            count.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            table.setItem(r, 4, count)
        self._numeric_headers(table, 2, 3, 4)
        self._fit_table(table)
        lay.addWidget(table)
        imp = QtWidgets.QPushButton("VESTA…", toolTip="Import colors and radii from a VESTA elements.ini file")
        imp.clicked.connect(self._import_elements_ini)
        reset = QtWidgets.QPushButton("Reset", toolTip="Back to the default colors and radii of every type")
        reset.clicked.connect(lambda: (self.style.type_colors.clear(), self.style.type_radii.clear(),
                                       self.style.type_transparency.clear(), self.styleChanged.emit(),
                                       self.refresh()))
        lay.addLayout(self._buttons(imp, reset))
        return w

    def set_type_transparency(self, t: str, value: float):
        table = {k: v for k, v in self.style.type_transparency.items() if k != t}
        if value > 0:
            table[t] = round(float(value), 3)
        self.style.type_transparency = table
        self._emit()

    def _set_style_preset(self, name):
        types = self.data.types if self.data is not None else ()
        self.style.apply_preset(name, types)
        if self.style.bonds_enabled and self.data is not None:
            self.style.ensure_bond_pairs(types)
        self.styleChanged.emit()
        QtCore.QTimer.singleShot(0, self.refresh)  # radius, bonds and drawing follow the preset

    def _set_radius_preset(self, name):
        self.style.radius_preset = name
        self.styleChanged.emit()
        QtCore.QTimer.singleShot(0, self.refresh)  # rebuilds this editor; not from inside the combo's signal

    def _set_oxidation_states(self, edit):
        from ..radii import parse_oxidation_states

        try:
            states = parse_oxidation_states(edit.text())
        except ValueError as exc:
            edit.setToolTip(str(exc))
            edit.setProperty("invalid", True)
            edit.style().polish(edit)
            return
        edit.setProperty("invalid", False)
        edit.style().polish(edit)
        if states != self.style.oxidation_states:
            self.style.oxidation_states = states
            self.styleChanged.emit()
            QtCore.QTimer.singleShot(0, self.refresh)

    def _pick_color(self, t, button):
        c = QtWidgets.QColorDialog.getColor(parent=self, title=f"Color of {t}")
        if c.isValid():
            self.style.type_colors[t] = (c.redF(), c.greenF(), c.blueF())
            _paint_swatch(button, c.name())
            self._emit()

    def _import_elements_ini(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "VESTA elements.ini", "", "elements.ini (*.ini);;All (*)")
        if path:
            colors, radii = read_vesta_elements(path)
            self.style.type_colors.update(colors)
            self.style.type_radii.update(radii)
            self.styleChanged.emit()
            self.refresh()

    def _bonds_editor(self):
        w = QtWidgets.QWidget()
        f = self._form_layout(w)
        on = QtWidgets.QCheckBox("Show bonds", checked=self.style.bonds_enabled)
        rad = QtWidgets.QDoubleSpinBox(minimum=0.01, maximum=2, singleStep=0.02, value=self.style.bond_radius, suffix=" Å")
        rad.valueChanged.connect(lambda v: self._set("bond_radius", v))
        self._row(f, "", on, "bonds")
        self._row(f, "Radius", rad, "bond_radius")
        pairs = self.style.bond_pairs
        if not pairs and self.data is not None:   # what the automatic table would be; stored once edited
            from .style import ViewStyle

            auto = ViewStyle(bond_factor=self.style.bond_factor)
            auto.ensure_bond_pairs(self.data.types)
            pairs = auto.bond_pairs
        table = QtWidgets.QTableWidget(len(pairs), 4, objectName="viewTable")
        table.setHorizontalHeaderLabels(["A", "B", "min Å", "max Å"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        for r, (a, b, lo, hi) in enumerate(pairs):
            for c, v in enumerate((a, b, f"{lo:.2f}", f"{hi:.2f}")):
                item = QtWidgets.QTableWidgetItem(str(v))
                if c >= 2:
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                table.setItem(r, c, item)
        self._numeric_headers(table, 2, 3)
        self._fit_table(table, rows=6)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)   # cells are edited in place
        table.setToolTip("Bond search: pairwise cutoffs, * = any type")

        def commit():
            rows = []
            for r in range(table.rowCount()):
                try:
                    rows.append([table.item(r, 0).text().strip(), table.item(r, 1).text().strip(),
                                 float(table.item(r, 2).text()), float(table.item(r, 3).text())])
                except (AttributeError, ValueError):
                    continue
            self.style.bond_pairs = rows
            self._emit()

        table.itemChanged.connect(lambda *_: commit())
        on.toggled.connect(lambda v: (setattr(self.style, "bonds_enabled", v), self.styleChanged.emit(),
                                      QtCore.QTimer.singleShot(0, self.refresh)))
        pairs_label = QtWidgets.QLabel("Pairs (cutoffs, * = any type)", objectName="viewLabel")
        self._marks.append((pairs_label, ("bond_pairs",)))
        f.addRow(pairs_label)
        f.addRow(table)
        add = QtWidgets.QPushButton("Add pair")
        def edit(change):
            if not self.style.bond_pairs:
                self.style.bond_pairs = [list(p) for p in pairs]
            change(self.style.bond_pairs)
            self.styleChanged.emit()
            self.refresh()

        add.clicked.connect(lambda: edit(lambda rows: rows.append(["*", "*", 0.0, 2.5])))
        rm = QtWidgets.QPushButton("Remove")
        rm.clicked.connect(lambda: edit(lambda rows: table.currentRow() >= 0 and rows.pop(table.currentRow())))
        to_mod = QtWidgets.QPushButton("As step")
        to_mod.setToolTip("Add a Create bonds step with this table to the notebook, so bonds can be edited and "
                          "used by Coordination polyhedra")
        to_mod.clicked.connect(lambda: self.add_modifier(
            mods.CreateBonds(mode="pairwise", pairs=[tuple(p) for p in pairs])))
        f.addRow(self._buttons(add, rm, to_mod))
        if self.data is not None and self.data.bonds is not None:
            f.addRow(QtWidgets.QLabel(f"{self.data.bonds.count:,} bonds from a step (drawn instead of the table)",
                                      objectName="viewStatus", wordWrap=True))
        elif self.data is not None and self.style.bonds_enabled:
            from ..render import bond_list

            try:
                n = len(bond_list(self.data, pairs=pairs)[0]) // 2
                f.addRow(QtWidgets.QLabel(f"{n:,} bonds", objectName="viewStatus"))
            except ValueError as exc:
                f.addRow(QtWidgets.QLabel(str(exc), wordWrap=True, objectName="viewError"))
        return w

    def _cell_editor(self):
        w = QtWidgets.QWidget()
        f = self._form_layout(w)
        show = QtWidgets.QCheckBox("Draw the cell", checked=self.style.show_cell)
        show.toggled.connect(lambda v: self._set("show_cell", v))
        width = QtWidgets.QDoubleSpinBox(minimum=0.5, maximum=10, singleStep=0.5, value=self.style.cell_width)
        width.valueChanged.connect(lambda v: self._set("cell_width", v))
        self._row(f, "", show, "cell")
        self._row(f, "Line width", width, "cell_width")
        for box, keys in ((self._replicate_editor(), ("replicate", "replicate_cell")),
                          (self._boundary_editor(), ("boundary", "boundary_bonds"))):
            self._marks.append((box, keys))
            f.addRow(box)
        return w

    def _replicate_editor(self):
        """OVITO's Replicate as a display setting: na x nb x nc copies of the cell are drawn."""
        from ..replicate import CELL_MODES

        n = list(self.style.replicate or (1, 1, 1))
        box, g = self._group("Replicate cell", bool(self.style.replicate),
                             "Draw copies of the cell side by side, with bonds and polyhedra across them. "
                             "Display only: analysis and export keep the original cell.")
        spins = []
        for k, axis in enumerate("abc"):
            s = QtWidgets.QSpinBox(minimum=1, maximum=20, value=int(n[k]) if box.isChecked() else 2,
                                   toolTip=f"Copies along {axis}")
            s.setPrefix(f"{axis}  ")
            s.setMinimumWidth(44)
            spins.append(s)
        lines = QtWidgets.QComboBox()
        lines.addItems(list(CELL_MODES))
        lines.setCurrentText(self.style.replicate_cell)
        for k, tip in enumerate(CELL_MODES.values()):
            lines.setItemData(k, tip, QtCore.Qt.ToolTipRole)

        def apply(*_):
            reps = [s.value() for s in spins]
            self.style.replicate = reps if box.isChecked() and reps != [1, 1, 1] else []
            self.style.replicate_cell = lines.currentText()
            self._emit()

        box.toggled.connect(apply)
        for s in spins:
            s.valueChanged.connect(apply)
        lines.currentTextChanged.connect(apply)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        for s in spins:
            row.addWidget(s, 1)
        self._row(g, "Copies", _wrap(row))
        self._row(g, "Cell lines", lines)
        return box

    def _boundary_editor(self):
        """VESTA's boundary range: periodic images inside lo..hi (fractional) are drawn too."""
        from ..boundary import BOND_MODES

        box, g = self._group("Boundary (VESTA)", bool(self.style.boundary),
                             "Also draw the periodic images of atoms inside this fractional range, and complete "
                             "bonds and polyhedra cut by the cell. Display only: analysis keeps the real atoms.")
        lo_hi = (self.style.boundary if np.ndim(self.style.boundary) == 1 else self.style.boundary[0]) \
            if self.style.boundary else (0.0, 1.0)
        lo = QtWidgets.QDoubleSpinBox(minimum=-5, maximum=1, singleStep=0.05, decimals=3, value=float(lo_hi[0]))
        hi = QtWidgets.QDoubleSpinBox(minimum=0, maximum=6, singleStep=0.05, decimals=3, value=float(lo_hi[1]))
        mode = QtWidgets.QComboBox()
        mode.addItems(list(BOND_MODES))
        mode.setCurrentText(self.style.boundary_bonds)
        for k, tip in enumerate(BOND_MODES.values()):
            mode.setItemData(k, tip, QtCore.Qt.ToolTipRole)

        def apply(*_):
            self.style.boundary = [lo.value(), max(hi.value(), lo.value())] if box.isChecked() else []
            self.style.boundary_bonds = mode.currentText()
            self._emit()

        box.toggled.connect(apply)
        lo.valueChanged.connect(apply)
        hi.valueChanged.connect(apply)
        mode.currentTextChanged.connect(apply)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(lo, 1)
        row.addWidget(QtWidgets.QLabel("to", objectName="viewLabel"))
        row.addWidget(hi, 1)
        self._row(g, "Range", _wrap(row))
        self._row(g, "Bonds out", mode)
        return box

    def _camera_editor(self):
        """What the notebook's view says about the camera and frame; Save writes the current one."""
        w = QtWidgets.QWidget()
        f = self._form_layout(w)
        view = getattr(self.pipeline, "view", None) if self.pipeline is not None else None
        values = self._set_keys()

        def shown(key, default):
            return _short(values[key]) if key in values else default

        if view is not None and "camera" in values:
            self._row(f, "Camera", QtWidgets.QLabel("saved exactly", objectName="viewValue"), "camera")
        else:
            self._row(f, "Direction", QtWidgets.QLabel(shown("direction", "iso"), objectName="viewValue"), "direction")
            self._row(f, "Zoom", QtWidgets.QLabel(shown("zoom", "1.0"), objectName="viewValue"), "zoom")
        ortho = values.get("orthographic", True)
        self._row(f, "Projection", QtWidgets.QLabel("orthographic" if ortho else "perspective",
                                                    objectName="viewValue"), "orthographic")
        self._row(f, "Frame", QtWidgets.QLabel(shown("frame", "the timeline's"), objectName="viewValue"), "frame")
        box = self._light_editor()
        self._marks.append((box, ("light",)))
        f.addRow(box)
        save = QtWidgets.QPushButton("Save camera", toolTip="Write the current camera and frame into the "
                                                            "notebook's view(pipe, ...) cell")
        save.clicked.connect(self.cameraRequested.emit)
        f.addRow(self._buttons(save))
        return w

    def _light_editor(self):
        """The key light (the view's ``light``) as azimuth/elevation from the viewer and intensity; unchecked,
        each renderer lights the scene its own default way."""
        import math

        from ..render import light_params

        d, k = light_params(self.style.light)
        box, g = self._group("Light", self.style.light is not None,
                             "Key light for the viewport and every renderer, relative to the viewer: azimuth turns "
                             "it right, elevation up. Unchecked: the default light (upper left).")
        az = QtWidgets.QDoubleSpinBox(minimum=-180, maximum=180, singleStep=15, decimals=0, suffix="°",
                                      value=round(math.degrees(math.atan2(d[0], d[2]))))
        el = QtWidgets.QDoubleSpinBox(minimum=-90, maximum=90, singleStep=15, decimals=0, suffix="°",
                                      value=round(math.degrees(math.asin(max(-1.0, min(1.0, d[1]))))))
        strength = QtWidgets.QDoubleSpinBox(minimum=0, maximum=3, singleStep=0.1, decimals=2, value=k)
        self.light_widgets = {"box": box, "azimuth": az, "elevation": el, "intensity": strength}

        def apply(*_):
            self.style.light = {"azimuth": az.value(), "elevation": el.value(), "intensity": strength.value()} \
                if box.isChecked() else None
            self._emit()

        box.toggled.connect(apply)
        for spin in (az, el, strength):
            spin.valueChanged.connect(apply)
        self._row(g, "Azimuth", az)
        self._row(g, "Elevation", el)
        self._row(g, "Intensity", strength)
        return box


def _paint_swatch(button, color: str):
    """Fill a color chip (the color is data, so this one style is set inline)."""
    button.setStyleSheet(f"QPushButton {{ background: {color}; border: 1px solid {TOKENS['border']}; "
                         f"border-radius: 3px; padding: 0; min-height: 0; }}")


def _centered(widget):
    """``widget`` centered in a table cell."""
    cell = QtWidgets.QWidget(objectName="viewCell")
    lay = QtWidgets.QHBoxLayout(cell)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(widget, 0, QtCore.Qt.AlignCenter)
    return cell


def _wrap(layout):
    """A layout as a form field."""
    w = QtWidgets.QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w


def _short(value) -> str:
    """A setting value for a tooltip or a camera row: floats rounded, long values cut."""
    if isinstance(value, str):
        text = value
    elif isinstance(value, float):
        text = f"{value:.4g}"
    elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
        text = "[" + ", ".join(f"{x:.4g}" for x in value) + "]"
    else:
        text = repr(value)
    return text if len(text) <= 60 else text[:57] + "…"


def _types(data):
    from ..render import type_groups

    return None if data is None else (data.count, tuple(sorted(map(str, type_groups(data.types)[0]))))


def cell_summary(cell) -> str:
    m = cell.matrix
    a, b, c = np.linalg.norm(m, axis=1)
    ang = lambda u, v: np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
    return (f"a={a:.4f}  b={b:.4f}  c={c:.4f} Å\n"
            f"α={ang(m[1], m[2]):.2f}°  β={ang(m[0], m[2]):.2f}°  γ={ang(m[0], m[1]):.2f}°\n"
            f"V={cell.volume:.2f} Å³   PBC {' '.join('T' if p else 'F' for p in cell.pbc)}")


def read_vesta_elements(path):
    """VESTA elements.ini rows: index symbol radius vdw ionic R G B."""
    colors, radii = {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 8 and parts[0].isdigit():
                sym = parts[1]
                radii[sym] = float(parts[2])
                colors[sym] = tuple(float(x) for x in parts[5:8])
    return colors, radii
