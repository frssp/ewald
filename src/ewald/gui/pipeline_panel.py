"""Pipeline editor: modifier list (checkable), add/remove/reorder, parameter forms."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .. import modifiers as mods
from .params import ParameterForm

# Menu entries: label -> factory. Grouped like OVITO's modifier list.
MODIFIER_MENU = {
    "Analysis": {
        "Coordination analysis / RDF": lambda: mods.CoordinationAnalysis(cutoff=3.2),
        "Displacement vectors": lambda: mods.CalculateDisplacements(0),
    },
    "Selection": {
        "Expression selection": lambda: mods.ExpressionSelection("Position.Z > 0"),
        "Select type": lambda: mods.SelectType("Cu"),
        "Invert selection": mods.InvertSelection,
        "Clear selection": mods.ClearSelection,
        "Delete selected": mods.DeleteSelected,
    },
    "Modification": {
        "Slice": lambda: mods.Slice((0, 0, 1), 0.0),
        "Replicate": lambda: mods.Replicate((2, 2, 2)),
        "Wrap at periodic boundaries": mods.WrapAtoms,
        "Affine transformation": lambda: mods.AffineTransformation([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        "Compute property": lambda: mods.ComputeProperty("MyProperty", "Position.Z"),
    },
    "Coloring": {
        "Color coding": lambda: mods.ColorCoding("Position.Z"),
        "Assign color": lambda: mods.AssignColor((1.0, 0.2, 0.2)),
    },
    "Python": {
        "Python script": mods.PythonScript,
    },
}


class PipelinePanel(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pipeline = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        bar = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QToolButton(text="Add modification…")
        self.add_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(self)
        for group, items in MODIFIER_MENU.items():
            sub = menu.addMenu(group)
            for label, factory in items.items():
                sub.addAction(label, lambda f=factory: self.add_modifier(f()))
        self.add_button.setMenu(menu)
        bar.addWidget(self.add_button)
        bar.addStretch()
        for text, tip, slot in (("▲", "Move up", lambda: self._move(-1)),
                                ("▼", "Move down", lambda: self._move(1)),
                                ("✕", "Delete modifier", self._remove)):
            b = QtWidgets.QToolButton(text=text, toolTip=tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
        lay.addLayout(bar)

        # Top of the list = last modifier, as in OVITO (data flows bottom -> top).
        self.list = QtWidgets.QListWidget()
        self.list.itemChanged.connect(self._item_toggled)
        self.list.currentRowChanged.connect(self._show_editor)
        lay.addWidget(self.list, 1)

        self.editor_box = QtWidgets.QGroupBox("Parameters")
        self.editor_layout = QtWidgets.QVBoxLayout(self.editor_box)
        lay.addWidget(self.editor_box, 2)
        self._form = None
        self._script = None

    # list index <-> modifier index (list is reversed, last row = source)
    def _mod_index(self, row):
        n = len(self.pipeline.modifiers)
        return None if row < 0 or row >= n else n - 1 - row

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        self.refresh()

    def refresh(self, select=None):
        """Rebuild the list; keeps the selected modifier selected unless ``select`` is given."""
        if select is None and self.pipeline is not None:
            idx = self._mod_index(self.list.currentRow())
            current = self.pipeline.modifiers[idx] if idx is not None and idx < len(self.pipeline.modifiers) else None
            if current is not None and current in self.pipeline.modifiers:
                select = self.pipeline.modifiers.index(current)
        self.list.blockSignals(True)
        self.list.clear()
        if self.pipeline is not None:
            for mod in reversed(self.pipeline.modifiers):
                item = QtWidgets.QListWidgetItem(mod.name)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked if mod.enabled else QtCore.Qt.Unchecked)
                if mod.status:
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#c62828")))
                    item.setToolTip(mod.status)
                self.list.addItem(item)
            src = QtWidgets.QListWidgetItem(f"Source: {_source_label(self.pipeline.source)}")
            src.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            font = src.font(); font.setBold(True); src.setFont(font)
            self.list.addItem(src)
        self.list.blockSignals(False)
        if select is not None and self.pipeline is not None:
            row = len(self.pipeline.modifiers) - 1 - select
            if row == self.list.currentRow():
                self._show_editor(row)
            else:
                self.list.setCurrentRow(row)
        else:
            self._show_editor(self.list.currentRow())

    def update_status(self):
        """Recolor items after a compute without rebuilding the editor."""
        if self.pipeline is None:
            return
        self.list.blockSignals(True)  # recoloring fires itemChanged
        for row in range(self.list.count() - 1):
            mod = self.pipeline.modifiers[self._mod_index(row)]
            item = self.list.item(row)
            item.setForeground(QtGui.QBrush(QtGui.QColor("#c62828" if mod.status else self.palette().text().color())))
            item.setToolTip(mod.status or "")
        self.list.blockSignals(False)

    def add_modifier(self, mod):
        if self.pipeline is None:
            return
        row = self.list.currentRow()
        idx = self._mod_index(row)
        pos = len(self.pipeline.modifiers) if idx is None else idx + 1  # insert above selection
        self.pipeline.modifiers.insert(pos, mod)
        self.refresh(select=pos)
        self.changed.emit()

    def _remove(self):
        idx = self._mod_index(self.list.currentRow())
        if idx is None:
            return
        del self.pipeline.modifiers[idx]
        self.refresh()
        self.changed.emit()

    def _move(self, direction):
        idx = self._mod_index(self.list.currentRow())
        if idx is None:
            return
        new = idx - direction  # "up" in the list = later in the pipeline
        if not 0 <= new < len(self.pipeline.modifiers):
            return
        m = self.pipeline.modifiers
        m[idx], m[new] = m[new], m[idx]
        self.refresh(select=new)
        self.changed.emit()

    def _item_toggled(self, item):
        idx = self._mod_index(self.list.row(item))
        if idx is not None:
            self.pipeline.modifiers[idx].enabled = item.checkState() == QtCore.Qt.Checked
            self.changed.emit()

    def _show_editor(self, row):
        while self.editor_layout.count():
            w = self.editor_layout.takeAt(0).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self._form = self._script = None
        idx = None if self.pipeline is None else self._mod_index(row)
        if idx is None:
            self.editor_box.setTitle("Parameters")
            return
        mod = self.pipeline.modifiers[idx]
        self.editor_box.setTitle(mod.name)
        if mod.status:
            err = QtWidgets.QLabel(mod.status, wordWrap=True)
            err.setStyleSheet("color:#c62828")
            self.editor_layout.addWidget(err)
        self._form = ParameterForm(mod)
        self.editor_layout.addWidget(self._form)
        if isinstance(mod, mods.PythonScript):
            self._script = QtWidgets.QPlainTextEdit(mod.source)
            self._script.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
            self._script.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
            self._script.setTabStopDistance(4 * self._script.fontMetrics().horizontalAdvance(" "))
            self.editor_layout.addWidget(self._script, 1)
        elif hasattr(mod, "function"):
            self.editor_layout.addWidget(QtWidgets.QLabel("Python function modifier (edit in the console)"))
        apply = QtWidgets.QPushButton("Apply")
        apply.setShortcut("Ctrl+Return")
        apply.clicked.connect(lambda: self._apply(mod))
        self.editor_layout.addWidget(apply)
        self.editor_layout.addStretch()

    def _apply(self, mod):
        if self._form is not None:
            self._form.apply()
        if self._script is not None:
            mod.source = self._script.toPlainText()
        row = self.list.currentRow()
        self.list.item(row).setText(mod.name)
        self.changed.emit()
        self._show_editor(row)


def _source_label(source):
    import os

    paths = getattr(source, "paths", None)
    if paths:
        return os.path.basename(paths[0]) + (f" (+{len(paths) - 1})" if len(paths) > 1 else "")
    return type(source).__name__
