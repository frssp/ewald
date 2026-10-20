"""Step menus (Insert › Step, the step context checks) and the Display panel: visual elements
(Particles, Bonds, Simulation cell, plugin elements) with their editors."""
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


KIND = QtCore.Qt.UserRole + 1



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

class DisplayPanel(QtWidgets.QWidget):
    """Visual elements (Particles, Bonds, Simulation cell, plugin elements) with their editors.
    Steps live in notebook cells; this panel only changes how the data is drawn."""

    styleChanged = QtCore.Signal()               # visual elements changed -> redraw only
    modifierRequested = QtCore.Signal(object)    # e.g. "Use as Create bonds modifier": a notebook step

    def __init__(self, style=None, parent=None):
        super().__init__(parent)
        from .style import ViewStyle

        _discover()
        self.pipeline = None
        self.style = style or ViewStyle()
        self.data = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        # the Display header shows the add-visual menu as an icon (app.py moves it there)
        self.visual_button = QtWidgets.QToolButton(objectName="headerButton",
                                                   toolTip="Add a visual element (cell axes, halos, … from plugins)")
        self.visual_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        header.set_icon(self.visual_button, "plus")
        self.visual_button.setIconSize(QtCore.QSize(header.ICON, header.ICON))
        self.visual_button.setFixedSize(header.BUTTON, header.BUTTON)
        self.visual_menu = QtWidgets.QMenu(self)
        self.visual_menu.aboutToShow.connect(self._fill_visual_menu)
        self.visual_button.setMenu(self.visual_menu)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(3)
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(8)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
        self.tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.tree.setColumnWidth(0, 52)
        self.tree.itemChanged.connect(self._item_toggled)
        self.tree.currentItemChanged.connect(lambda cur, prev: self._show_editor(cur))
        QtGui.QShortcut(QtGui.QKeySequence.Delete, self.tree, activated=self._remove)
        lay.addWidget(self.tree)

        self.editor_title = QtWidgets.QLabel("Properties", objectName="section")
        lay.addWidget(self.editor_title)
        self.scroll = FitWidthScrollArea()  # editors fit the panel width instead of being cut off
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        lay.addWidget(self.scroll, 1)
        self._form = None

    # ------------------------------------------------------------------ model
    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        self.refresh()

    def set_data(self, data):
        old = _types(self.data)
        self.data = data
        self.update_status()
        if _types(data) != old:  # another file or notebook: the type table and counts change
            self._show_editor(self.tree.currentItem())

    def _current(self):
        item = self.tree.currentItem()
        return None if item is None else item.data(0, KIND)

    def refresh(self, select=None):
        """Rebuild the rows; keeps the selection unless ``select`` (a row kind) is given."""
        if select is None:
            select = self._current() or ("visual", "particles")
        self.tree.blockSignals(True)
        self.tree.clear()
        target = self._visual_items(select)
        self.tree.blockSignals(False)
        self._fit_tree()
        self.update_status()
        if target is not None and target is not self.tree.currentItem():
            self.tree.setCurrentItem(target)
        else:
            self._show_editor(self.tree.currentItem())

    def _fit_tree(self):
        """As tall as its few rows (Particles, Bonds, Cell, plugins), so the editor gets the rest."""
        rows = self.tree.topLevelItemCount()
        row = self.tree.sizeHintForRow(0) if rows else 24
        self.tree.setFixedHeight(max(1, rows) * row + 2 * self.tree.frameWidth() + 4)

    def _visual_items(self, select):
        """Visual element rows (Particles, Bonds, Cell, plugin elements); returns the row to select."""
        target = None
        for key, label, on in (("particles", "Particles", True), ("bonds", "Bonds", self.style.bonds_enabled),
                               ("cell", "Simulation cell", self.style.show_cell)):
            it = QtWidgets.QTreeWidgetItem(self.tree, ["", label, ""])
            it.setData(0, KIND, ("visual", key))
            if key != "particles":
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(0, QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
            if select == ("visual", key):
                target = it
        for k, entry in enumerate(self.style.visuals):
            it = QtWidgets.QTreeWidgetItem(self.tree, ["", entry["name"], ""])
            it.setData(0, KIND, ("plugin-visual", k))
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(0, QtCore.Qt.Checked if entry.get("enabled", True) else QtCore.Qt.Unchecked)
            if select == ("plugin-visual", k):
                target = it
        return target

    def update_status(self):
        """Right column: particle count and periodicity of the current frame (no editor rebuild)."""
        if self.data is None:
            return
        self.tree.blockSignals(True)
        for k in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(k)
            kind = it.data(0, KIND)
            if kind == ("visual", "particles"):
                it.setText(2, f"{self.data.count}")
            elif kind == ("visual", "cell") and self.data.cell.is_valid:
                it.setText(2, "PBC " + "".join("xyz"[i] for i in range(3) if self.data.cell.pbc[i]))
        self.tree.blockSignals(False)

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
        self.refresh(select=("plugin-visual", len(self.style.visuals) - 1))
        self.styleChanged.emit()

    def _remove(self):
        cur = self._current()
        if cur and cur[0] == "plugin-visual":
            del self.style.visuals[cur[1]]
            self.refresh(select=("visual", "particles"))
            self.styleChanged.emit()

    def _item_toggled(self, item, column):
        kind = item.data(0, KIND)
        if not kind or column != 0:
            return
        on = item.checkState(0) == QtCore.Qt.Checked
        if kind[0] == "visual":
            if kind[1] == "bonds":
                self.style.bonds_enabled = on
                if on and self.data is not None:
                    self.style.ensure_bond_pairs(self.data.types)
            elif kind[1] == "cell":
                self.style.show_cell = on
            self.styleChanged.emit()
        elif kind[0] == "plugin-visual":
            self.style.visuals[kind[1]]["enabled"] = on
            self.styleChanged.emit()

    # ------------------------------------------------------------------ properties
    def _set_editor(self, title, widget):
        self.editor_title.setText(title.upper())
        old = self.scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        page = QtWidgets.QWidget()  # the editor at its own height on top, not spread over the panel
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(widget)
        lay.addStretch(1)
        self.scroll.setWidget(page)

    def _show_editor(self, item):
        self._form = None
        kind = None if item is None else item.data(0, KIND)
        if not kind:
            self._set_editor("Properties", QtWidgets.QWidget())
            return
        if kind[0] == "visual":
            build = {"particles": self._particles_editor, "bonds": self._bonds_editor, "cell": self._cell_editor}
            self._set_editor(item.text(1), build[kind[1]]())
        elif kind[0] == "plugin-visual":
            self._set_editor(item.text(1), self._plugin_visual_editor(kind[1]))

    def _plugin_visual_editor(self, k):
        """Parameter form of a plugin visual element, built from its constructor like a modifier's."""
        from ..scene import coerce_visuals, visual_to_dict

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        try:
            (el,) = coerce_visuals([self.style.visuals[k]])
        except Exception as exc:  # plugin no longer installed, or its parameters changed
            err = QtWidgets.QLabel(f"{type(exc).__name__}: {exc}", wordWrap=True)
            err.setStyleSheet(f"color:{TOKENS['error']}")
            lay.addWidget(err)
            lay.addStretch()
            return w
        doc = (type(el).__doc__ or "").strip().split("\n\n")[0]
        if doc:
            lay.addWidget(QtWidgets.QLabel(" ".join(doc.replace("``", "").split()), wordWrap=True, objectName="muted"))
        self._form = ParameterForm(el)
        lay.addWidget(self._form)

        def apply():
            self._form.apply()
            self.style.visuals[k] = {**visual_to_dict(el), "enabled": self.style.visuals[k].get("enabled", True)}
            self.styleChanged.emit()

        button = QtWidgets.QPushButton("Apply  (Ctrl+Enter)")
        button.setObjectName("primary")
        button.clicked.connect(apply)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), w, activated=apply)
        lay.addWidget(button)
        lay.addStretch()
        return w

    def _particles_editor(self):
        from ..radii import IONIC, PRESETS, format_oxidation_states
        from ..render import type_color, type_radius

        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        scale = QtWidgets.QDoubleSpinBox(minimum=0.05, maximum=5, singleStep=0.05, value=self.style.radius_scale)
        scale.valueChanged.connect(lambda v: (setattr(self.style, "radius_scale", v), self.styleChanged.emit()))
        f.addRow("Radius scale", scale)
        preset = QtWidgets.QComboBox()
        preset.addItems(list(PRESETS))
        for k, cite in enumerate(PRESETS.values()):
            preset.setItemData(k, cite, QtCore.Qt.ToolTipRole)
        preset.setCurrentText(self.style.radius_preset)
        preset.currentTextChanged.connect(self._set_radius_preset)
        f.addRow("Radius preset", preset)
        oxi = QtWidgets.QLineEdit(placeholderText="auto, e.g. Sr=2, Ti=4, O=-2")
        oxi.setText(format_oxidation_states(self.style.oxidation_states))
        oxi.setToolTip("Oxidation state per type for the ionic presets; empty types are guessed "
                       "(type name, charges, bond valence, charge balance)")
        oxi.editingFinished.connect(lambda e=oxi: self._set_oxidation_states(e))
        oxi.setEnabled(self.style.radius_preset in IONIC)
        f.addRow("Oxidation states", oxi)
        shape = QtWidgets.QComboBox()
        shape.addItems(["auto", "mesh", "impostor", "flat"])
        shape.setCurrentText(self.style.atom_style)
        shape.setToolTip("mesh: best shading for figures; impostor: GPU sprites for millions of atoms; "
                         "flat: ASE-GUI look, unshaded discs with outlines")
        shape.currentTextChanged.connect(lambda v: (setattr(self.style, "atom_style", v), self.styleChanged.emit()))
        f.addRow("Rendering", shape)
        types = [] if self.data is None else list(dict.fromkeys(self.data.types))
        table = QtWidgets.QTableWidget(len(types), 4)
        table.setHorizontalHeaderLabels(["Type", "Color", "Radius", "Count"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table.horizontalHeader().setMinimumSectionSize(36)
        for r, t in enumerate(types):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(t))
            color = self.style.type_colors.get(t)
            rgb = type_color(t, r) if color is None else np.asarray(QtGui.QColor(color).getRgbF()[:3] if isinstance(color, str) else color)
            btn = QtWidgets.QPushButton()
            btn.setStyleSheet(f"background: {QtGui.QColor.fromRgbF(*rgb).name()}; border: 1px solid {TOKENS['border']};")
            btn.clicked.connect(lambda _=False, t=t, b=btn: self._pick_color(t, b))
            table.setCellWidget(r, 1, btn)
            rad = QtWidgets.QDoubleSpinBox(minimum=0.05, maximum=5, singleStep=0.05,
                                           value=self.style.type_radii.get(t, type_radius(t, preset=self.style.radius_preset)))
            rad.valueChanged.connect(lambda v, t=t: (self.style.type_radii.__setitem__(t, v), self.styleChanged.emit()))
            table.setCellWidget(r, 2, rad)
            table.setItem(r, 3, QtWidgets.QTableWidgetItem(str(int((self.data.types == t).sum()))))
        table.setFixedHeight(min(260, 34 * (len(types) + 1) + 6))  # no empty rows below the types
        f.addRow(QtWidgets.QLabel("Types", objectName="section"))
        f.addRow(table)
        row = QtWidgets.QHBoxLayout()
        imp = QtWidgets.QPushButton("VESTA…", toolTip="Import colors and radii from a VESTA elements.ini file")
        imp.clicked.connect(self._import_elements_ini)
        reset = QtWidgets.QPushButton("Reset", toolTip="Back to the default colors and radii of every type")
        reset.clicked.connect(lambda: (self.style.type_colors.clear(), self.style.type_radii.clear(),
                                       self.styleChanged.emit(), self.refresh()))
        row.addWidget(imp); row.addWidget(reset)
        f.addRow(row)
        return w

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
            edit.setStyleSheet(f"border: 1px solid {TOKENS['error']};")
            return
        edit.setStyleSheet("")
        if states != self.style.oxidation_states:
            self.style.oxidation_states = states
            self.styleChanged.emit()
            QtCore.QTimer.singleShot(0, self.refresh)

    def _pick_color(self, t, button):
        c = QtWidgets.QColorDialog.getColor(parent=self, title=f"Color of {t}")
        if c.isValid():
            self.style.type_colors[t] = (c.redF(), c.greenF(), c.blueF())
            button.setStyleSheet(f"background: {c.name()}; border: 1px solid {TOKENS['border']};")
            self.styleChanged.emit()

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
        f = QtWidgets.QFormLayout(w)
        on = QtWidgets.QCheckBox("Show bonds", checked=self.style.bonds_enabled)
        rad = QtWidgets.QDoubleSpinBox(minimum=0.01, maximum=2, singleStep=0.02, value=self.style.bond_radius, suffix=" Å")
        rad.valueChanged.connect(lambda v: (setattr(self.style, "bond_radius", v), self.styleChanged.emit()))
        f.addRow(on)
        f.addRow("Bond radius", rad)
        f.addRow(QtWidgets.QLabel("Bond search (pairwise cutoffs, * = any)", objectName="section"))
        if self.data is not None:
            self.style.ensure_bond_pairs(self.data.types)
        table = QtWidgets.QTableWidget(len(self.style.bond_pairs), 4)
        table.setHorizontalHeaderLabels(["A", "B", "min Å", "max Å"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        for r, (a, b, lo, hi) in enumerate(self.style.bond_pairs):
            for c, v in enumerate((a, b, f"{lo:.2f}", f"{hi:.2f}")):
                table.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
        table.setMinimumHeight(min(240, 32 * (len(self.style.bond_pairs) + 1) + 6))

        def commit():
            rows = []
            for r in range(table.rowCount()):
                try:
                    rows.append([table.item(r, 0).text().strip(), table.item(r, 1).text().strip(),
                                 float(table.item(r, 2).text()), float(table.item(r, 3).text())])
                except (AttributeError, ValueError):
                    continue
            self.style.bond_pairs = rows
            self.styleChanged.emit()

        table.itemChanged.connect(lambda *_: commit())
        on.toggled.connect(lambda v: (setattr(self.style, "bonds_enabled", v), self.styleChanged.emit(), self.refresh()))
        f.addRow(table)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add pair")
        add.clicked.connect(lambda: (self.style.bond_pairs.append(["*", "*", 0.0, 2.5]), self.styleChanged.emit(), self.refresh()))
        rm = QtWidgets.QPushButton("Remove")
        rm.clicked.connect(lambda: (table.currentRow() >= 0 and self.style.bond_pairs.pop(table.currentRow()),
                                    self.styleChanged.emit(), self.refresh()))
        row.addWidget(add); row.addWidget(rm)
        f.addRow(row)
        to_mod = QtWidgets.QPushButton("Use as Create bonds modifier")
        to_mod.setToolTip("Add a CreateBonds modifier with this table, so bonds can be edited and used "
                          "by Coordination polyhedra")
        to_mod.clicked.connect(lambda: self.add_modifier(
            mods.CreateBonds(mode="pairwise", pairs=[tuple(p) for p in self.style.bond_pairs])))
        f.addRow(to_mod)
        if self.data is not None and self.data.bonds is not None:
            f.addRow("Pipeline bonds", QtWidgets.QLabel(f"{self.data.bonds.count} (drawn instead of the table)"))
        elif self.data is not None and self.style.bonds_enabled:
            from ..render import bond_list

            try:
                n = len(bond_list(self.data, pairs=self.style.bond_pairs)[0]) // 2
                f.addRow("Bonds", QtWidgets.QLabel(str(n)))
            except ValueError as exc:
                f.addRow(QtWidgets.QLabel(str(exc), wordWrap=True))
        return w

    def _cell_editor(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        show = QtWidgets.QCheckBox("Render cell", checked=self.style.show_cell)
        show.toggled.connect(lambda v: (setattr(self.style, "show_cell", v), self.styleChanged.emit(), self.refresh()))
        width = QtWidgets.QDoubleSpinBox(minimum=0.5, maximum=10, singleStep=0.5, value=self.style.cell_width)
        width.valueChanged.connect(lambda v: (setattr(self.style, "cell_width", v), self.styleChanged.emit()))
        f.addRow(show)
        f.addRow("Line width", width)
        f.addRow(self._boundary_editor())
        if self.data is not None and self.data.cell.is_valid:
            f.addRow(QtWidgets.QLabel(cell_summary(self.data.cell), objectName="muted"))
        return w

    def _boundary_editor(self):
        """VESTA's boundary range: periodic images inside lo..hi (fractional) are drawn too."""
        from ..boundary import BOND_MODES

        box = QtWidgets.QGroupBox("Boundary (VESTA)", checkable=True, checked=bool(self.style.boundary))
        box.setToolTip("Also draw the periodic images of atoms inside this fractional range, and complete "
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
            self.styleChanged.emit()

        box.toggled.connect(apply)
        lo.valueChanged.connect(apply)
        hi.valueChanged.connect(apply)
        mode.currentTextChanged.connect(apply)
        g = QtWidgets.QFormLayout(box)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(lo)
        row.addWidget(QtWidgets.QLabel("to"))
        row.addWidget(hi)
        g.addRow("Range", row)
        g.addRow("Bonds leaving it", mode)
        return box


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
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 8 and parts[0].isdigit():
                sym = parts[1]
                radii[sym] = float(parts[2])
                colors[sym] = tuple(float(x) for x in parts[5:8])
    return colors, radii
