"""Periodic table of per-element color and radius defaults (Preferences > Elements)."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..settings import Settings
from .pref_widgets import _ColorButton, _hex
from .theme import TOKENS


class ElementTable(QtWidgets.QWidget):
    """Per-element overrides (VESTA Edit › Elements); only edited rows end up in settings."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.overrides: dict = {}
        self.base = Settings()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        bar = QtWidgets.QHBoxLayout()
        self.filter = QtWidgets.QLineEdit(placeholderText="Filter elements, e.g. Cu O", objectName="field")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._filter)
        imp = QtWidgets.QPushButton("Import elements.ini…", objectName="secondary")
        imp.setToolTip("Read colors and radii from a VESTA elements.ini file")
        imp.clicked.connect(self._import)
        reset = QtWidgets.QPushButton("Reset all", objectName="secondary")
        reset.clicked.connect(lambda: self.set_overrides({}, self.base) or self.changed.emit())
        bar.addWidget(self.filter, 1)
        bar.addWidget(imp)
        bar.addWidget(reset)
        lay.addLayout(bar)
        self.table = QtWidgets.QTableWidget(0, 5, objectName="elements")
        self.table.setHorizontalHeaderLabels(["Z", "Element", "Color", "Radius (Å)", ""])
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.setShowGrid(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(QtCore.Qt.NoFocus)
        hdr = self.table.horizontalHeader()
        hdr.setDefaultAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        hdr.setStretchLastSection(True)
        for col, width in enumerate((44, 80, 128, 120)):
            self.table.setColumnWidth(col, width)
        self.table.setMinimumHeight(320)
        lay.addWidget(self.table)

    def set_overrides(self, overrides: dict, base: Settings):
        from ase.data import chemical_symbols

        self.overrides = {k: dict(v) for k, v in overrides.items()}
        self.base = base
        symbols = chemical_symbols[1:97]
        self.table.setRowCount(len(symbols))
        for r, sym in enumerate(symbols):
            o = self.overrides.get(sym, {})
            for col, text in ((0, str(r + 1)), (1, sym)):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(QtCore.Qt.ItemIsEnabled)
                if col == 0:
                    item.setForeground(QtGui.QColor(TOKENS["muted"]))
                self.table.setItem(r, col, item)
            btn = _ColorButton(o.get("color") or self._scheme(sym, "color"))
            btn.changed.connect(lambda s=sym, b=btn: self._edit(s, "color", b.color))
            self.table.setCellWidget(r, 2, self._cell(btn))
            spin = QtWidgets.QDoubleSpinBox(minimum=0.05, maximum=5, singleStep=0.05, decimals=3)
            spin.setFixedWidth(100)
            spin.setValue(o.get("radius") or self._scheme(sym, "radius"))
            spin.valueChanged.connect(lambda v, s=sym: self._edit(s, "radius", round(v, 3)))
            self.table.setCellWidget(r, 3, self._cell(spin))
            self._mark(r, sym)
        self._filter(self.filter.text())

    @staticmethod
    def _cell(w):
        holder = QtWidgets.QWidget(objectName="cell")
        lay = QtWidgets.QHBoxLayout(holder)
        lay.setContentsMargins(0, 2, 6, 2)
        lay.addWidget(w)
        lay.addStretch()
        return holder

    def _scheme(self, sym, what):
        probe = self.base.copy()
        ov = probe["elements.overrides"]
        ov.pop(sym, None)
        probe["elements.overrides"] = ov
        return probe.element_color(sym) if what == "color" else probe.element_radius(sym)

    def _mark(self, r, sym):
        item = QtWidgets.QTableWidgetItem("● Edited" if sym in self.overrides else "")
        item.setForeground(QtGui.QColor(TOKENS["accent"]))
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.table.setItem(r, 4, item)

    def _edit(self, sym, field, value):
        self.overrides.setdefault(sym, {})[field] = value
        from ase.data import atomic_numbers

        self._mark(atomic_numbers[sym] - 1, sym)
        self.changed.emit()

    def _filter(self, text):
        wanted = {t.strip().lower() for t in text.replace(",", " ").split() if t.strip()}
        for r in range(self.table.rowCount()):
            sym = self.table.item(r, 1).text().lower()
            self.table.setRowHidden(r, bool(wanted) and not any(sym.startswith(w) for w in wanted))

    def _import(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "VESTA elements.ini", "", "elements.ini (*.ini);;All (*)")
        if path:
            self.import_vesta(path)

    def import_vesta(self, path):
        from .pipeline_panel import read_vesta_elements

        colors, radii = read_vesta_elements(path)
        for sym in set(colors) | set(radii):
            o = self.overrides.setdefault(sym, {})
            if sym in colors:
                o["color"] = _hex(colors[sym])
            if sym in radii:
                o["radius"] = radii[sym]
        self.set_overrides(self.overrides, self.base)
        self.changed.emit()
