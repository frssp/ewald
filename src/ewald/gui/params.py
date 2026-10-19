"""Auto-generated parameter editor for a Modifier, built from its constructor signature."""
from __future__ import annotations

import ast

from PySide6 import QtCore, QtGui, QtWidgets

_CHOICES = {"alignment": ("base", "center", "head"), "shading": ("normal", "flat"),
            "corner": ("top-left", "top-right", "bottom-left", "bottom-right"), "axes": ("xyz", "abc")}


def colormap_icon(name: str, width: int = 64, height: int = 12):
    """A strip of the color map ``name`` for menus and combo boxes (None if it does not resolve)."""
    import numpy as np

    from ..colormaps import get_colormap

    try:
        rgba = (get_colormap(name)(np.linspace(0, 1, width)) * 255).astype(np.uint8)
    except (KeyError, ValueError):
        return None
    strip = np.ascontiguousarray(np.repeat(rgba[None], height, axis=0))
    image = QtGui.QImage(strip.data, width, height, 4 * width, QtGui.QImage.Format_RGBA8888).copy()
    return QtGui.QIcon(QtGui.QPixmap.fromImage(image))


def colormap_combo(value: str) -> QtWidgets.QComboBox:
    """Editable combo box of color maps (common matplotlib maps, then plugins) with preview strips;
    the tooltip says whether a map reads in grayscale."""
    from .. import plugins
    from ..colormaps import is_perceptual, names

    w = QtWidgets.QComboBox()
    w.setEditable(True)
    w.setIconSize(QtCore.QSize(64, 12))
    kinds = {k: p.kind for k, p in plugins.colormaps().items()}
    for k, name in enumerate(names()):
        icon = colormap_icon(name)
        w.addItem(icon, name) if icon is not None else w.addItem(name)
        try:
            ok = is_perceptual(name, kinds.get(name))
        except (KeyError, ValueError):
            continue
        tip = "reads in grayscale (monotonic lightness)" if ok else "not monotonic in lightness"
        w.setItemData(k, f"{name}: {tip}" + (" · plugin" if name in kinds else ""), QtCore.Qt.ToolTipRole)
    w.setCurrentText(str(value))
    return w


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
            w = colormap_combo(value)
            self.editors[name] = (w, str)
        elif isinstance(value, str) and name in (choices := {**_CHOICES, **getattr(self.modifier, "choice_parameters", {})}):
            w = QtWidgets.QComboBox()
            w.addItems(choices[name])
            w.setCurrentText(value)
            self.editors[name] = (w, str)
        elif isinstance(value, str) and name in getattr(self.modifier, "file_parameters", ()):
            line = QtWidgets.QLineEdit(value)
            w = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(w)
            row.setContentsMargins(0, 0, 0, 0)
            browse = QtWidgets.QPushButton("…", toolTip="Choose a file")
            browse.setFixedWidth(28)
            browse.clicked.connect(lambda _=False, e=line: self._browse(e))
            row.addWidget(line, 1)
            row.addWidget(browse)
            self.editors[name] = (line, str)
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

    def _browse(self, line):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a file", line.text(),
                                                        "Data (*.xy *.xye *.dat *.txt *.csv *.chi);;All files (*)")
        if path:
            line.setText(path)

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
