"""Data inspector: particle property table and global attributes."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets


class ParticleTableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.columns: list[tuple[str, np.ndarray]] = []
        self.rows = 0
        self.filter_index = None

    def set_data(self, data, filter_index=None):
        self.beginResetModel()
        cols = []
        if data is not None:
            for name, arr in data.particles.items():
                if arr.ndim == 2:
                    letters = "RGB" if name == "Color" else "XYZ"
                    for k in range(min(arr.shape[1], 3)):
                        cols.append((f"{name}.{letters[k]}", arr[:, k]))
                else:
                    cols.append((name, arr))
        self.columns = cols
        self.filter_index = filter_index
        self.rows = 0 if data is None else (data.count if filter_index is None else len(filter_index))
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return self.rows

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.columns)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole or not index.isValid():
            return None
        row = index.row() if self.filter_index is None else int(self.filter_index[index.row()])
        v = self.columns[index.column()][1][row]
        return f"{v:.5g}" if isinstance(v, (float, np.floating)) else str(v)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return self.columns[section][0]
        return str(section if self.filter_index is None else int(self.filter_index[section]))


class DataInspector(QtWidgets.QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        # particles tab with an expression filter
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(2, 2, 2, 2)
        self.filter = QtWidgets.QLineEdit(placeholderText="Filter expression, e.g. Coordination < 12")
        self.filter.returnPressed.connect(self._apply_filter)
        self.filter_status = QtWidgets.QLabel()
        self.model = ParticleTableModel(self)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        lay.addWidget(self.filter)
        lay.addWidget(self.filter_status)
        lay.addWidget(self.table)
        self.addTab(page, "Particles")
        self.attributes = QtWidgets.QTableWidget(0, 2)
        self.attributes.setHorizontalHeaderLabels(["Attribute", "Value"])
        self.attributes.horizontalHeader().setStretchLastSection(True)
        self.addTab(self.attributes, "Attributes")
        self.tables = QtWidgets.QPlainTextEdit(readOnly=True)
        self.addTab(self.tables, "Tables")

    def set_data(self, data):
        self._data = data
        self._apply_filter()
        attrs = {} if data is None else data.attributes
        self.attributes.setRowCount(len(attrs))
        for r, (k, v) in enumerate(attrs.items()):
            self.attributes.setItem(r, 0, QtWidgets.QTableWidgetItem(str(k)))
            self.attributes.setItem(r, 1, QtWidgets.QTableWidgetItem(str(v)))
        lines = []
        for name, table in ({} if data is None else data.tables).items():
            keys = list(table)
            lines.append(f"# {name}\n# " + "\t".join(keys))
            for row in zip(*(table[k] for k in keys)):
                lines.append("\t".join(f"{x:.5g}" for x in row))
        self.tables.setPlainText("\n".join(lines))

    def _apply_filter(self):
        from ..modifiers.expression import evaluate

        text = self.filter.text().strip()
        index = None
        self.filter_status.setText("")
        if text and self._data is not None:
            try:
                mask = np.broadcast_to(np.asarray(evaluate(text, self._data), bool), (self._data.count,))
                index = np.nonzero(mask)[0]
                self.filter_status.setText(f"{len(index)} of {self._data.count} particles")
            except Exception as exc:
                self.filter_status.setText(f"<span style='color:#c00'>{exc}</span>")
        self.model.set_data(self._data, index)
