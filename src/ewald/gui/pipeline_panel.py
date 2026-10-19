"""Command panel: pipeline tree (visual elements / modifications / source) + properties editor."""
from __future__ import annotations

import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import modifiers as mods
from .params import ParameterForm
from .theme import TOKENS

# Add-modification menu: category -> label -> factory. Grouped like OVITO's list.
MODIFIER_MENU = {
    "Analysis": {
        "Coordination analysis / RDF": lambda: mods.CoordinationAnalysis(cutoff=3.2),
        "Displacement vectors": lambda: mods.CalculateDisplacements(0),
    },
    "Trajectory": {
        "Mean squared displacement / diffusion": mods.MeanSquaredDisplacement,
        "Velocity autocorrelation": mods.VelocityAutocorrelation,
        "Time averaging (g(r), attributes)": mods.TimeAveraging,
        "Time series of attributes": mods.TimeSeries,
        "Trajectory lines / paths": mods.GenerateTrajectoryLines,
    },
    "Bonds & polyhedra": {
        "Create bonds": lambda: mods.CreateBonds(mode="covalent"),
        "Edit bonds (add / delete)": mods.EditBonds,
        "Coordination polyhedra": mods.CoordinationPolyhedra,
    },
    "Volume (CHGCAR, LOCPOT, cube)": {
        "Isosurface": mods.CreateIsosurface,
        "Isosurface, signed (density difference / spin)": lambda: mods.CreateIsosurface(signed=True),
        "Slice plane (2D map)": mods.GridSlice,
        "Charge density difference": lambda: mods.GridDifference(["CHGCAR_A", "CHGCAR_B"]),
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
    },
    "Selection": {
        "Expression selection": lambda: mods.ExpressionSelection("Position.Z > 0"),
        "Select type": lambda: mods.SelectType("Cu"),
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


def fill_modifier_menu(menu, add):
    """(Re)populate ``menu`` with MODIFIER_MENU; ``add`` receives the new modifier."""
    menu.clear()
    for sub in menu.findChildren(QtWidgets.QMenu, options=QtCore.Qt.FindDirectChildrenOnly):
        sub.deleteLater()
    for group, items in MODIFIER_MENU.items():
        sub = menu.addMenu(group)
        for label, factory in items.items():
            sub.addAction(label, lambda f=factory: add(f()))


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


BADGES = {"CoordinationAnalysis": "CN", "CalculateDisplacements": "DV", "ColorCoding": "CC", "AssignColor": "AC",
          "Slice": "SL", "Replicate": "RP", "WrapAtoms": "WR", "AffineTransformation": "AT", "ComputeProperty": "CP",
          "DeleteSelected": "DS", "ExpressionSelection": "ES", "SelectType": "ST", "InvertSelection": "IS",
          "ClearSelection": "CS", "PythonScript": "Py", "PythonModifier": "Py",
          "MeanSquaredDisplacement": "MS", "VelocityAutocorrelation": "VA", "TimeAveraging": "TA",
          "TimeSeries": "TS", "GenerateTrajectoryLines": "TL", "CreateBonds": "CB", "EditBonds": "EB",
          "CoordinationPolyhedra": "PH",
          "CreateIsosurface": "SF", "GridSlice": "GS", "GridDifference": "GD"}

KIND = QtCore.Qt.UserRole + 1


class PipelinePanel(QtWidgets.QWidget):
    changed = QtCore.Signal()        # pipeline changed -> recompute
    styleChanged = QtCore.Signal()   # visual elements changed -> redraw only

    def __init__(self, style=None, parent=None):
        super().__init__(parent)
        from .style import ViewStyle

        _discover()
        self.pipeline = None
        self.style = style or ViewStyle()
        self.data = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        bar = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("Add modification  ▾")
        self.add_button.setObjectName("primary")
        self.add_menu = QtWidgets.QMenu(self)
        # rebuilt on every open, so entries registered later (startup script, plugins) show up
        self.add_menu.aboutToShow.connect(lambda: fill_modifier_menu(self.add_menu, self.add_modifier))
        fill_modifier_menu(self.add_menu, self.add_modifier)
        self.add_button.setMenu(self.add_menu)
        bar.addWidget(self.add_button, 1)
        for text, tip, slot in (("▲", "Move up (later in the pipeline)", lambda: self._move(-1)),
                                ("▼", "Move down (earlier)", lambda: self._move(1)),
                                ("✕", "Delete modifier (Del)", self._remove)):
            b = QtWidgets.QToolButton(text=text, toolTip=tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
        lay.addLayout(bar)

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
        lay.addWidget(self.tree, 2)

        self.editor_title = QtWidgets.QLabel("Properties", objectName="section")
        lay.addWidget(self.editor_title)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        lay.addWidget(self.scroll, 3)
        self._form = None
        self._script = None

    # ------------------------------------------------------------------ model
    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        self.refresh()

    def set_data(self, data):
        self.data = data
        self.update_status()

    def _current(self):
        item = self.tree.currentItem()
        return None if item is None else item.data(0, KIND)

    def refresh(self, select=None):
        """Rebuild the tree; keeps the selection unless ``select`` (a modifier) is given."""
        if select is None:
            select = self._current()
        self.tree.blockSignals(True)
        self.tree.clear()
        bold = QtGui.QFont(self.font()); bold.setBold(True)

        def section(title):
            it = QtWidgets.QTreeWidgetItem(self.tree, ["", title.upper(), ""])
            it.setFlags(QtCore.Qt.ItemIsEnabled)
            it.setForeground(1, QtGui.QBrush(QtGui.QColor(TOKENS["muted"])))
            f = QtGui.QFont(self.font()); f.setPixelSize(10); f.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 1); it.setFont(1, f)
            return it

        target = None
        section("Visual elements")
        for key, label, on in (("particles", "Particles", True), ("bonds", "Bonds", self.style.bonds_enabled),
                               ("cell", "Simulation cell", self.style.show_cell)):
            it = QtWidgets.QTreeWidgetItem(self.tree, ["", label, ""])
            it.setData(0, KIND, ("visual", key))
            if key != "particles":
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(0, QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
            if select == ("visual", key):
                target = it
        section("Modifications")
        if self.pipeline is not None:
            for mod in reversed(self.pipeline.modifiers):
                badge = BADGES.get(type(mod).__name__, type(mod).__name__[:2])
                it = QtWidgets.QTreeWidgetItem(self.tree, [badge, mod.name, ""])
                it.setData(0, KIND, ("mod", mod))
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(0, QtCore.Qt.Checked if mod.enabled else QtCore.Qt.Unchecked)
                it.setForeground(0, QtGui.QBrush(QtGui.QColor(TOKENS["accent"])))
                if select == ("mod", mod) or select is mod:
                    target = it
        section("Data source")
        if self.pipeline is not None:
            it = QtWidgets.QTreeWidgetItem(self.tree, ["", _source_label(self.pipeline.source), ""])
            it.setData(0, KIND, ("source", None))
            it.setFont(1, bold)
            if select == ("source", None):
                target = it
        self.tree.blockSignals(False)
        self.update_status()
        if target is not None:
            if target is self.tree.currentItem():
                self._show_editor(target)
            else:
                self.tree.setCurrentItem(target)
        else:
            self._show_editor(self.tree.currentItem())

    def modifier_items(self):
        out = []
        for k in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(k)
            kind = it.data(0, KIND)
            if kind and kind[0] == "mod":
                out.append(it)
        return out

    def update_status(self):
        """Status column: time / counts / errors after a compute (no editor rebuild)."""
        self.tree.blockSignals(True)
        for it in self.modifier_items():
            mod = it.data(0, KIND)[1]
            if mod.status:
                it.setText(2, "error")
                it.setForeground(2, QtGui.QBrush(QtGui.QColor(TOKENS["error"])))
                it.setForeground(1, QtGui.QBrush(QtGui.QColor(TOKENS["error"])))
                it.setToolTip(1, mod.status)
            else:
                t = mod.last_time
                it.setText(2, "" if (t is None or not mod.enabled) else (f"{t * 1000:.0f} ms" if t >= 0.001 else "<1 ms"))
                it.setForeground(2, QtGui.QBrush(QtGui.QColor(TOKENS["muted"])))
                it.setForeground(1, QtGui.QBrush(QtGui.QColor(TOKENS["text"] if mod.enabled else TOKENS["muted"])))
                it.setToolTip(1, "")
        for k in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(k)
            kind = it.data(0, KIND)
            if kind and kind[0] == "visual" and self.data is not None:
                if kind[1] == "particles":
                    it.setText(2, f"{self.data.count}")
                elif kind[1] == "cell" and self.data.cell.is_valid:
                    it.setText(2, "PBC " + "".join("xyz"[i] for i in range(3) if self.data.cell.pbc[i]))
            if kind and kind[0] == "source" and self.pipeline is not None:
                it.setText(2, f"{self.pipeline.num_frames} fr")
                it.setForeground(2, QtGui.QBrush(QtGui.QColor(TOKENS["muted"])))
        self.tree.blockSignals(False)

    # ------------------------------------------------------------------ editing
    def add_modifier(self, mod):
        if self.pipeline is None:
            return
        cur = self._current()
        if cur and cur[0] == "mod" and cur[1] in self.pipeline.modifiers:
            pos = self.pipeline.modifiers.index(cur[1]) + 1  # insert above the selection
        else:
            pos = len(self.pipeline.modifiers)
        self.pipeline.modifiers.insert(pos, mod)
        self.refresh(select=("mod", self.pipeline.modifiers[pos]))
        self.changed.emit()

    def _remove(self):
        cur = self._current()
        if not cur or cur[0] != "mod":
            return
        idx = self.pipeline.modifiers.index(cur[1])
        del self.pipeline.modifiers[idx]
        rest = self.pipeline.modifiers
        self.refresh(select=("mod", rest[min(idx, len(rest) - 1)]) if rest else ("source", None))
        self.changed.emit()

    def _move(self, direction):
        cur = self._current()
        if not cur or cur[0] != "mod":
            return
        m = self.pipeline.modifiers
        idx = m.index(cur[1])
        new = idx - direction  # "up" in the list = later in the pipeline
        if not 0 <= new < len(m):
            return
        m[idx], m[new] = m[new], m[idx]
        self.refresh(select=("mod", cur[1]))
        self.changed.emit()

    def _item_toggled(self, item, column):
        kind = item.data(0, KIND)
        if not kind or column != 0:
            return
        on = item.checkState(0) == QtCore.Qt.Checked
        if kind[0] == "mod":
            kind[1].enabled = on
            self.changed.emit()
        elif kind[0] == "visual":
            if kind[1] == "bonds":
                self.style.bonds_enabled = on
                if on and self.data is not None:
                    self.style.ensure_bond_pairs(self.data.types)
            elif kind[1] == "cell":
                self.style.show_cell = on
            self.styleChanged.emit()

    # ------------------------------------------------------------------ properties
    def _set_editor(self, title, widget):
        self.editor_title.setText(title.upper())
        old = self.scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self.scroll.setWidget(widget)

    def _show_editor(self, item):
        self._form = self._script = None
        kind = None if item is None else item.data(0, KIND)
        if not kind:
            self._set_editor("Properties", QtWidgets.QWidget())
            return
        if kind[0] == "mod":
            self._set_editor(kind[1].name, self._modifier_editor(kind[1]))
        elif kind[0] == "visual":
            build = {"particles": self._particles_editor, "bonds": self._bonds_editor, "cell": self._cell_editor}
            self._set_editor(item.text(1), build[kind[1]]())
        else:
            self._set_editor("Data source", self._source_editor())

    def _modifier_editor(self, mod):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        if mod.status:
            err = QtWidgets.QLabel(mod.status, wordWrap=True)
            err.setStyleSheet(f"color:{TOKENS['error']}")
            lay.addWidget(err)
        doc = (type(mod).__doc__ or "").strip().split("\n\n")[0]
        if doc:
            d = QtWidgets.QLabel(" ".join(doc.replace("``", "").split()), wordWrap=True, objectName="muted")
            lay.addWidget(d)
        self._form = ParameterForm(mod)
        lay.addWidget(self._form)
        if isinstance(mod, mods.PythonScript):
            self._script = QtWidgets.QPlainTextEdit(mod.source)
            self._script.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
            self._script.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
            self._script.setTabStopDistance(4 * self._script.fontMetrics().horizontalAdvance(" "))
            self._script.setMinimumHeight(260)
            lay.addWidget(self._script, 1)
            ok = QtWidgets.QLabel(objectName="muted")
            ok.setText("OK" if not mod.status else "")
            lay.addWidget(ok)
        elif hasattr(mod, "function"):
            lay.addWidget(QtWidgets.QLabel("Python function modifier (edit it in the console)", objectName="muted"))
        apply = QtWidgets.QPushButton("Run  (Ctrl+Enter)" if isinstance(mod, mods.PythonScript) else "Apply  (Ctrl+Enter)")
        apply.setObjectName("primary")
        apply.clicked.connect(lambda: self._apply(mod))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), w, activated=lambda: self._apply(mod))
        lay.addWidget(apply)
        lay.addStretch()
        return w

    def _apply(self, mod):
        if self._form is not None:
            self._form.apply()
        if self._script is not None:
            mod.source = self._script.toPlainText()
        self.changed.emit()
        self.refresh(select=("mod", mod))

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
        shape.addItems(["auto", "mesh", "impostor"])
        shape.setCurrentText(self.style.atom_style)
        shape.setToolTip("mesh: best shading for figures; impostor: GPU sprites for millions of atoms")
        shape.currentTextChanged.connect(lambda v: (setattr(self.style, "atom_style", v), self.styleChanged.emit()))
        f.addRow("Rendering", shape)
        types = [] if self.data is None else list(dict.fromkeys(self.data.types))
        table = QtWidgets.QTableWidget(len(types), 4)
        table.setHorizontalHeaderLabels(["Type", "Color", "Radius", "Count"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
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
        table.setMinimumHeight(min(260, 34 * (len(types) + 1) + 6))
        f.addRow(QtWidgets.QLabel("Types", objectName="section"))
        f.addRow(table)
        row = QtWidgets.QHBoxLayout()
        imp = QtWidgets.QPushButton("Import VESTA elements.ini…")
        imp.clicked.connect(self._import_elements_ini)
        reset = QtWidgets.QPushButton("Reset")
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
        if self.data is not None and self.data.cell.is_valid:
            f.addRow(QtWidgets.QLabel(cell_summary(self.data.cell), objectName="muted"))
        return w

    def _source_editor(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        src = self.pipeline.source if self.pipeline else None
        paths = getattr(src, "paths", None)
        if paths:
            f.addRow("File", QtWidgets.QLabel(os.path.basename(paths[0]) + (f" (+{len(paths) - 1})" if len(paths) > 1 else "")))
            readers = {type(r).__name__ for r in src.readers}
            f.addRow("Reader", QtWidgets.QLabel(", ".join(sorted(readers))))
        if self.pipeline:
            f.addRow("Frames", QtWidgets.QLabel(f"{self.pipeline.num_frames}, loaded lazily"))
        if self.data is not None:
            f.addRow("Particles", QtWidgets.QLabel(str(self.data.count)))
            if self.data.cell.is_valid:
                f.addRow(QtWidgets.QLabel(cell_summary(self.data.cell), objectName="muted"))
            props = QtWidgets.QLabel(", ".join(self.data.particles), wordWrap=True, objectName="muted")
            f.addRow("Properties", props)
        return w


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


def _source_label(source):
    paths = getattr(source, "paths", None)
    if paths:
        return os.path.basename(paths[0]) + (f" (+{len(paths) - 1})" if len(paths) > 1 else "")
    return type(source).__name__
