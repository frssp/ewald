"""Auto-generated parameter editor for a Modifier, built from its constructor signature."""
from __future__ import annotations

import ast

from PySide6 import QtWidgets

_CMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "coolwarm_r", "jet",
          "rainbow", "turbo", "RdBu", "RdBu_r", "Spectral", "hot", "Greys"]


class ParameterForm(QtWidgets.QWidget):
    """Edits ``modifier.parameters()``; emits nothing itself, call ``apply()``."""

    def __init__(self, modifier, parent=None):
        super().__init__(parent)
        self.modifier = modifier
        self.editors: dict[str, tuple[QtWidgets.QWidget, type]] = {}
        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        for name, value in modifier.parameters().items():
            if name in ("source", "params"):
                continue  # script source has its own editor
            editor = self._editor_for(name, value)
            layout.addRow(name, editor)

    def _editor_for(self, name, value):
        if isinstance(value, bool):
            w = QtWidgets.QCheckBox()
            w.setChecked(value)
            self.editors[name] = (w, bool)
        elif isinstance(value, int):
            w = QtWidgets.QSpinBox()
            w.setRange(-10**9, 10**9)
            w.setValue(value)
            self.editors[name] = (w, int)
        elif isinstance(value, float):
            w = QtWidgets.QDoubleSpinBox()
            w.setDecimals(4)
            w.setRange(-1e9, 1e9)
            w.setSingleStep(0.1)
            w.setValue(value)
            self.editors[name] = (w, float)
        elif name == "cmap":
            w = QtWidgets.QComboBox()
            w.setEditable(True)
            w.addItems(_CMAPS)
            w.setCurrentText(str(value))
            self.editors[name] = (w, str)
        elif isinstance(value, str) and (name in getattr(self.modifier, "multiline_parameters", ()) or "\n" in value):
            w = QtWidgets.QPlainTextEdit(value)   # e.g. a PLUMED input
            w.setMinimumHeight(110)
            w.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
            self.editors[name] = (w, str)
        elif isinstance(value, str):
            w = QtWidgets.QLineEdit(value)
            self.editors[name] = (w, str)
        else:  # tuples, lists, None, ...: Python literal
            w = QtWidgets.QLineEdit(repr(value))
            w.setToolTip("Python literal, e.g. (0, 0, 1) or None")
            self.editors[name] = (w, object)
        return w

    def apply(self):
        for name, (w, kind) in self.editors.items():
            if kind is bool:
                value = w.isChecked()
            elif kind in (int, float):
                value = w.value()
            elif isinstance(w, QtWidgets.QComboBox):
                value = w.currentText()
            elif isinstance(w, QtWidgets.QPlainTextEdit):
                value = w.toPlainText()
            elif kind is str:
                value = w.text()
            else:
                text = w.text().strip()
                try:
                    value = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    value = text
            setattr(self.modifier, name, value)
