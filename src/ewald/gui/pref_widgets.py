"""Small widgets of the Preferences dialog: toggle switch, color and path fields, icons."""
from __future__ import annotations


from PySide6 import QtCore, QtGui, QtWidgets

from .theme import STYLE, TOKENS


# 24x24 line icons (stroke = currentColor), drawn in the Lucide style
_ICONS = {
    "General": '<path d="M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1"/><circle cx="15" cy="6" r="2"/>'
               '<circle cx="9" cy="12" r="2"/><circle cx="17" cy="18" r="2"/>',
    "Viewports": '<rect x="3" y="4" width="18" height="14" rx="2"/><path d="M8 21h8M12 18v3"/>'
                 '<path d="M12 7l4 2.3v4.4L12 16l-4-2.3V9.3z"/>',
    "Elements": '<circle cx="12" cy="12" r="1.5"/><ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(30 12 12)"/>'
                '<ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(-30 12 12)"/>'
                '<ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(90 12 12)"/>',
    "Rendering": '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    "Compute": '<rect x="6" y="6" width="12" height="12" rx="1.5"/><rect x="9.5" y="9.5" width="5" height="5"/>'
               '<path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "Python": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3M12 15h5"/>',
    "AI": '<path d="M12 3l1.8 5.4L19 10l-5.2 1.6L12 17l-1.8-5.4L5 10l5.2-1.6z"/><path d="M18 16l.8 2.2L21 19l-2.2.8L18 22l-.8-2.2L15 19l2.2-.8z"/>',
}


def _icon(name, color, size=18) -> QtGui.QPixmap:
    from PySide6 import QtSvg

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
           f'stroke-width="{STYLE["icon_stroke"]:g}" stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>')
    ratio = 2
    pix = QtGui.QPixmap(size * ratio, size * ratio)
    pix.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pix)
    QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode())).render(painter)
    painter.end()
    pix.setDevicePixelRatio(ratio)
    return pix


def _chevron() -> str:
    """Down-arrow image for combo boxes (QSS needs a file), in the active palette."""
    from .theme import image

    return image("chevron")


def _hex(c) -> str:
    if isinstance(c, str):
        return QtGui.QColor(c).name()
    return QtGui.QColor.fromRgbF(*[float(x) for x in c[:3]]).name()


def _restyle(w, name):
    w.setObjectName(name)
    w.style().unpolish(w)
    w.style().polish(w)


class Toggle(QtWidgets.QCheckBox):
    """iOS/macOS-style switch; still a QCheckBox for signals and isChecked()."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedSize(36, 20)

    def sizeHint(self):
        return QtCore.QSize(36, 20)

    def hitButton(self, pos):
        return self.rect().contains(pos)

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        on = self.isChecked()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(TOKENS["accent"] if on else TOKENS["border"]))
        p.drawRoundedRect(QtCore.QRectF(0, 0, 36, 20), 10, 10)
        p.setBrush(QtGui.QColor(TOKENS["on_accent"] if on else TOKENS["secondary"]))
        p.drawEllipse(QtCore.QRectF(19 if on else 3, 3, 14, 14))


class _ColorButton(QtWidgets.QPushButton):
    changed = QtCore.Signal()

    def __init__(self, color="#ffffff", parent=None):
        super().__init__(parent)
        self.setObjectName("swatch")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedWidth(112)
        self.clicked.connect(self._pick)
        self.set(color)

    def set(self, color):
        self.color = _hex(color)
        self.setText(" " + self.color.upper())
        pix = QtGui.QPixmap(28, 28)
        pix.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtGui.QPen(QtGui.QColor(TOKENS["muted"]), 1.5))
        p.setBrush(QtGui.QColor(self.color))
        p.drawRoundedRect(QtCore.QRectF(1, 1, 26, 26), 6, 6)
        p.end()
        pix.setDevicePixelRatio(2)
        self.setIcon(QtGui.QIcon(pix))
        self.setIconSize(QtCore.QSize(14, 14))

    def _pick(self):
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.color), self)
        if c.isValid():
            self.set(c.name())
            self.changed.emit()


class _PathEdit(QtWidgets.QWidget):
    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.kind = kind
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.edit = QtWidgets.QLineEdit(placeholderText="Not set", objectName="field")
        browse = QtWidgets.QPushButton("Browse…", objectName="secondary")
        browse.clicked.connect(self._browse)
        lay.addWidget(self.edit, 1)
        lay.addWidget(browse)

    def _browse(self):
        if self.kind == "file":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Python script", self.edit.text(), "Python (*.py)")
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder", self.edit.text())
        if path:
            self.edit.setText(path)


class _ProfileBox(QtWidgets.QWidget):
    """Saved AI connections (``llm.profiles``): a picker plus Save as… and Delete. Picking one
    emits ``chosen(name)``; the dialog copies its fields into the rows below."""

    changed = QtCore.Signal()
    chosen = QtCore.Signal(str)
    save_requested = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.profiles: dict = {}
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.combo = QtWidgets.QComboBox()
        self.combo.activated.connect(self._activated)
        self.save = QtWidgets.QPushButton("Save as…", objectName="secondary",
                                          toolTip="Save the provider, model and connection below under a name")
        self.save.clicked.connect(self._ask_name)
        self.delete = QtWidgets.QPushButton("Delete", objectName="secondary", toolTip="Forget the chosen connection")
        self.delete.clicked.connect(lambda: self.remove(self.combo.currentData()))
        lay.addWidget(self.combo, 1)
        lay.addWidget(self.save)
        lay.addWidget(self.delete)
        self.set({})

    def set(self, profiles: dict, current: str | None = None):
        self.profiles = {k: dict(v) for k, v in (profiles or {}).items()}
        current = current if current is not None else self.combo.currentData()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("No saved connections" if not self.profiles else "Choose a saved connection…", None)
        for name, p in sorted(self.profiles.items()):
            self.combo.addItem(name, name)
            self.combo.setItemData(self.combo.count() - 1,
                                   " · ".join(str(v) for v in (p.get("provider"), p.get("model"), p.get("base_url")) if v),
                                   QtCore.Qt.ToolTipRole)
        self.combo.setCurrentIndex(max(0, self.combo.findData(current)) if current else 0)
        self.combo.blockSignals(False)
        self.delete.setEnabled(bool(self.combo.currentData()))

    def _activated(self, _index):
        name = self.combo.currentData()
        self.delete.setEnabled(bool(name))
        if name:
            self.chosen.emit(name)

    def _ask_name(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save connection", "Name:",
                                                  text=self.combo.currentData() or "")
        if ok and name.strip():
            self.save_requested.emit(name.strip())

    def remove(self, name):
        if name in self.profiles:
            del self.profiles[name]
            self.set(self.profiles, "")
            self.changed.emit()
