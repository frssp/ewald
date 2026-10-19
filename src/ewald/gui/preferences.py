"""Preferences dialog (Edit › Preferences, Ctrl+,): OVITO-style page list + VESTA element table.

Design: /design/settings-menu.md. Rows are generated from ``ewald.settings.OPTIONS`` and
grouped into sections by ``LAYOUT``; ``apply_settings`` pushes a Settings object into a
running MainWindow.
"""
from __future__ import annotations

import base64
import json
import os
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import settings as settings_mod
from ..settings import DEVICES, OPTION, OPTIONS, PAGES, Settings
from .style import PRESET_THEMES
from .theme import TOKENS

_CHOICE_LABELS = {"auto": "Auto", "coarse": "Always coarse", "full": "Always full", "ortho": "Orthographic", "persp": "Perspective", "raytrace": "Ray tracing (AO + shadows)",
                  "opengl": "OpenGL (fast preview)", "iso": "Isometric", "x": "Along +x", "y": "Along +y",
                  "z": "Along +z (top)", "front": "Front (−y)", "ask": "Ask when opening", "never": "Use default map",
                  "covalent": "Covalent", "vdW": "van der Waals", "white": "White", "black": "Black",
                  "transparent": "Transparent", "viewport": "Same as viewport"}

# page -> (subtitle, [(section, [option keys])])
LAYOUT = {
    "General": ("Startup behaviour and file handling.", [
        ("Startup", ["general.startup_workspace", "general.restore_layout"]),
        ("Appearance", ["general.font_size", "general.toolbar_style"]),
        ("LAMMPS files", ["general.lammps_type_prompt", "general.lammps_type_map"]),
        ("Help", ["general.docs_url"])]),
    "Viewports": ("Defaults for the interactive 3D view.", [
        ("View", ["viewport.background", "viewport.projection", "viewport.initial_view", "viewport.style_preset"]),
        ("Navigation", ["viewport.rotation_step"]),
        ("Overlays", ["viewport.show_axes", "viewport.show_overlay"]),
        ("Performance", ["viewport.interactive_detail", "viewport.target_fps", "viewport.interactive_atoms",
                         "viewport.refine_delay", "viewport.sphere_drawing", "viewport.anti_aliasing",
                         "viewport.sphere_resolution", "viewport.arrow_resolution",
                         "viewport.mesh_limit"])]),
    "Elements": ("Default atom colors, radii and bond search, like VESTA's elements.ini.", [
        ("Atoms", ["elements.color_scheme", "elements.radius_source", "elements.radius_scale"]),
        ("Bonds", ["bonds.length_factor", "bonds.radius", "bonds.auto_limit"]),
        ("Element table", ["elements.overrides"])]),
    "Rendering": ("Starting values for the Render workspace.", [
        ("Renderer", ["render.renderer", "render.quality", "render.movie_quality", "render.samples",
                      "render.ao_samples", "render.shadows"]),
        ("Output", ["render.size_preset", "render.dpi", "render.background", "render.output_dir"]),
        ("Overlays and movies", ["render.type_legend", "render.scale_bar", "render.movie_format"])]),
    "Compute": ("Where pipelines run, and which optional packages are available.", [
        ("Device", ["compute.device", "compute.threads"]),
        ("Memory", ["performance.cache_memory"])]),
    "Python": ("Console and module search path.", [
        ("Console", ["python.startup_script"]),
        ("Modules", ["python.extra_paths"])]),
}

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
}


def _icon(name, color, size=18) -> QtGui.QPixmap:
    from PySide6 import QtSvg

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
           f'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>')
    ratio = 2
    pix = QtGui.QPixmap(size * ratio, size * ratio)
    pix.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pix)
    QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode())).render(painter)
    painter.end()
    pix.setDevicePixelRatio(ratio)
    return pix


QSS = """
QDialog#prefs {{ background: {bg}; }}
QFrame#sidebar {{ background: {panel}; border: none; border-right: 1px solid {border}; }}
QLineEdit#search {{ background: {bg}; border: 1px solid {border}; border-radius: 6px; padding: 6px 8px; }}
QLineEdit#search:focus {{ border-color: {accent}; }}
QListWidget#nav {{ background: transparent; border: none; outline: none; }}
QListWidget#nav::item {{ color: {secondary}; padding: 7px 8px; margin: 1px 0; border-radius: 6px; border: none; }}
QListWidget#nav::item:hover {{ background: {raised}; color: {text}; }}
QListWidget#nav::item:selected {{ background: {raised}; color: {text}; border-left: 2px solid {accent}; }}
QScrollArea#content, QWidget#contentInner {{ background: {bg}; border: none; }}
QWidget#contentInner QLabel, QWidget#contentInner QWidget#noteHolder {{ background: transparent; }}
QFrame#card QAbstractSpinBox::up-button, QFrame#card QAbstractSpinBox::down-button {{ width: 0px; border: none; }}
QFrame#card QComboBox::drop-down {{ border: none; width: 22px; }}
QFrame#card QComboBox::down-arrow {{ image: url("{chevron}"); width: 10px; height: 10px; margin-right: 8px; }}
QLabel#pageTitle {{ font-size: 20px; font-weight: 600; color: {text}; }}
QLabel#pageSubtitle {{ font-size: 12px; color: {muted}; }}
QLabel#sectionTitle {{ font-size: 11px; font-weight: 600; color: {muted}; letter-spacing: 0.6px; }}
QFrame#card {{ background: {panel}; border: 1px solid {border}; border-radius: 8px; }}
QFrame#row {{ background: transparent; border: none; border-top: 1px solid {border}; }}
QFrame#rowFirst {{ background: transparent; border: none; }}
QLabel#rowLabel {{ font-size: 13px; color: {text}; background: transparent; }}
QLabel#rowLabelChanged {{ font-size: 13px; color: {text}; font-weight: 600; background: transparent; }}
QLabel#rowHelp {{ font-size: 11px; color: {muted}; background: transparent; }}
QLabel#dot {{ color: {accent}; font-size: 10px; background: transparent; }}
QToolButton#rowReset {{ color: {muted}; border: none; background: transparent; padding: 0 4px; font-size: 13px; }}
QToolButton#rowReset:hover {{ color: {accent}; }}
QFrame#card QComboBox, QFrame#card QSpinBox, QFrame#card QDoubleSpinBox, QFrame#card QLineEdit#field {{
    background: {bg}; border: 1px solid {border}; border-radius: 6px; padding: 4px 8px; min-height: 18px; }}
QFrame#card QComboBox:hover, QFrame#card QSpinBox:hover, QFrame#card QDoubleSpinBox:hover,
QFrame#card QLineEdit#field:hover {{ border-color: {muted}; }}
QFrame#card QComboBox:focus, QFrame#card QSpinBox:focus, QFrame#card QDoubleSpinBox:focus,
QFrame#card QLineEdit#field:focus {{ border-color: {accent}; }}
QPushButton#swatch {{ background: {bg}; border: 1px solid {border}; border-radius: 6px; padding: 4px 8px;
    text-align: left; font-family: "IBM Plex Mono", "Menlo", "Consolas", monospace; }}
QPushButton#swatch:hover {{ border-color: {muted}; }}
QPushButton#secondary {{ background: {raised}; border: 1px solid {border}; border-radius: 6px; padding: 5px 12px; }}
QPushButton#secondary:hover {{ border-color: {muted}; }}
QFrame#footer {{ background: {panel}; border: none; border-top: 1px solid {border}; }}
QPushButton#footerButton {{ background: {raised}; border: 1px solid {border}; border-radius: 6px; padding: 6px 16px;
    min-width: 64px; }}
QPushButton#footerButton:hover {{ border-color: {muted}; }}
QPushButton#footerPrimary {{ background: {accent}; color: #0B1220; border: none; border-radius: 6px; padding: 6px 18px;
    min-width: 64px; font-weight: 600; }}
QPushButton#footerPrimary:hover {{ background: #78B9FF; }}
QPushButton#link {{ background: transparent; border: none; color: {secondary}; padding: 4px 6px; }}
QPushButton#link:hover {{ color: {accent}; }}
QLabel#path {{ color: {muted}; font-size: 11px; font-family: "IBM Plex Mono", "Menlo", "Consolas", monospace; }}
QLabel#badgeOk {{ color: {ok}; background: rgba(91, 208, 138, 0.12); border-radius: 9px; padding: 2px 9px;
    font-size: 11px; font-weight: 600; }}
QLabel#badgeOff {{ color: {muted}; background: {raised}; border-radius: 9px; padding: 2px 9px; font-size: 11px; }}
QLabel#mono {{ color: {secondary}; font-size: 11px; background: transparent;
    font-family: "IBM Plex Mono", "Menlo", "Consolas", monospace; }}
QTableWidget#elements {{ background: {bg}; border: 1px solid {border}; border-radius: 6px; gridline-color: transparent; }}
QWidget#cell {{ background: transparent; }}
QTableWidget#elements::item {{ border-bottom: 1px solid {panel}; padding: 0 6px; }}
"""


def _chevron() -> str:
    """Down-arrow image for combo boxes (QSS needs a file)."""
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "ewald-chevron.svg")
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
                     f'stroke="{TOKENS["secondary"]}" stroke-width="2.5" stroke-linecap="round" '
                     'stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>')
    return path.replace("\\", "/")


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
        p.setBrush(QtGui.QColor("#0B1220" if on else TOKENS["secondary"]))
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


class PreferencesDialog(QtWidgets.QDialog):
    """Edits a copy of ``settings``; Apply/OK copy it back and emit ``applied``."""

    applied = QtCore.Signal(object)
    EDITOR_WIDTH = 240

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setObjectName("prefs")
        self.setWindowTitle("Preferences")
        self.setStyleSheet(QSS.format(chevron=_chevron(), **TOKENS))
        self.resize(940, 680)
        self.setMinimumSize(760, 520)
        self.settings = settings
        self.work = settings.copy()
        self.editors: dict[str, tuple] = {}   # key -> (editor, label, row)
        self.row_parts: dict[str, tuple] = {}  # key -> (dot, reset button)
        self.cards: list[tuple] = []           # (section title, card, [keys])
        self._loading = False
        self._pending_note = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(0)
        outer.addLayout(body, 1)

        sidebar = QtWidgets.QFrame(objectName="sidebar")
        sidebar.setFixedWidth(220)
        side = QtWidgets.QVBoxLayout(sidebar)
        side.setContentsMargins(12, 14, 12, 12)
        side.setSpacing(10)
        self.search = QtWidgets.QLineEdit(placeholderText="Search settings", objectName="search")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._search)
        side.addWidget(self.search)
        self.pages = QtWidgets.QListWidget(objectName="nav")
        self.pages.setIconSize(QtCore.QSize(18, 18))
        self.pages.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        side.addWidget(self.pages, 1)
        body.addWidget(sidebar)

        self.stack = QtWidgets.QStackedWidget()
        body.addWidget(self.stack, 1)
        for name in PAGES:
            self._build_page(name)
        self.pages.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.pages.setCurrentRow(0)
        outer.addWidget(self._footer())
        self._load()

    # ---------------------------------------------------------------- building
    def _footer(self):
        bar = QtWidgets.QFrame(objectName="footer")
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(8)
        for text, tip, fn in (("Import…", "Load settings from a JSON file", self._import),
                              ("Export…", "Save these settings to a JSON file", self._export),
                              ("Reset page", "Restore this page to its defaults", self.restore_page)):
            b = QtWidgets.QPushButton(text, objectName="link", toolTip=tip)
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.clicked.connect(fn)
            lay.addWidget(b)
        self.path_label = QtWidgets.QLabel(objectName="path")
        self.path_label.setText(str(self.settings.path))
        self.path_label.setToolTip("Settings file")
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.path_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        lay.addSpacing(8)
        lay.addWidget(self.path_label, 1)
        for text, name, fn in (("Cancel", "footerButton", self.reject), ("Apply", "footerButton", self.apply),
                               ("OK", "footerPrimary", self.accept)):
            b = QtWidgets.QPushButton(text, objectName=name)
            b.clicked.connect(fn)
            b.setAutoDefault(name == "footerPrimary")
            b.setDefault(name == "footerPrimary")
            lay.addWidget(b)
        return bar

    def _build_page(self, name):
        subtitle, sections = LAYOUT[name]
        inner = QtWidgets.QWidget(objectName="contentInner")
        col = QtWidgets.QVBoxLayout(inner)
        col.setContentsMargins(32, 24, 32, 32)
        col.setSpacing(6)
        col.addWidget(QtWidgets.QLabel(name, objectName="pageTitle"))
        col.addWidget(QtWidgets.QLabel(subtitle, objectName="pageSubtitle"))
        for title, keys in sections:
            col.addSpacing(16)
            heading = QtWidgets.QLabel(title.upper(), objectName="sectionTitle")
            col.addWidget(heading)
            card, rows = self._card()
            for key in keys:
                editor = self._device_combo() if key == "compute.device" else self._editor(OPTION[key])
                self._add_row(rows, OPTION[key], editor)
            col.addWidget(card)
            self.cards.append((heading, card, keys))
        if name == "Compute":
            col.addSpacing(16)
            heading = QtWidgets.QLabel("OPTIONAL COMPONENTS", objectName="sectionTitle")
            col.addWidget(heading)
            card, rows = self._card()
            self._components(rows)
            col.addWidget(card)
            self.components_card = (heading, card)
        col.addStretch()
        scroll = QtWidgets.QScrollArea(objectName="content")
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.stack.addWidget(scroll)
        item = QtWidgets.QListWidgetItem(name)
        icon = QtGui.QIcon()
        icon.addPixmap(_icon(name, TOKENS["secondary"]), QtGui.QIcon.Normal)
        icon.addPixmap(_icon(name, TOKENS["accent"]), QtGui.QIcon.Selected)
        item.setIcon(icon)
        self.pages.addItem(item)

    @staticmethod
    def _card():
        card = QtWidgets.QFrame(objectName="card")
        rows = QtWidgets.QVBoxLayout(card)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(0)
        return card, rows

    def _row_frame(self, rows):
        row = QtWidgets.QFrame(objectName="rowFirst" if rows.count() == 0 else "row")
        rows.addWidget(row)
        return row

    def _add_row(self, rows, opt, editor):
        row = self._row_frame(rows)
        wide = opt.kind == "table"
        grid = QtWidgets.QGridLayout(row)
        grid.setContentsMargins(16, 12, 16, 12)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(2)
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(6)
        dot = QtWidgets.QLabel("●", objectName="dot", toolTip="Changed from the default")
        label = QtWidgets.QLabel(opt.label, objectName="rowLabel")
        label.setToolTip(opt.key)
        reset = QtWidgets.QToolButton(text="↺", objectName="rowReset", toolTip="Reset to default")
        reset.setCursor(QtCore.Qt.PointingHandCursor)
        reset.clicked.connect(lambda _=False, k=opt.key: self.reset_key(k))
        head.addWidget(dot)
        head.addWidget(label)
        head.addWidget(reset)
        head.addStretch()
        text.addLayout(head)
        if opt.help:
            h = QtWidgets.QLabel(opt.help, objectName="rowHelp", wordWrap=True)
            text.addWidget(h)
        if key_note := getattr(self, "_pending_note", None):
            text.addWidget(key_note)
            self._pending_note = None
        grid.addLayout(text, 0, 0)
        if wide:
            grid.addWidget(editor, 1, 0, 1, 2)
        else:
            if isinstance(editor, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                editor.setFixedWidth(140)
            elif not isinstance(editor, (Toggle, _ColorButton)):
                editor.setFixedWidth(self.EDITOR_WIDTH if not isinstance(editor, _PathEdit) else 320)
            grid.addWidget(editor, 0, 1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        grid.setColumnStretch(0, 1)
        self.editors[opt.key] = (editor, label, row)
        self.row_parts[opt.key] = (dot, reset)

    def _editor(self, opt):
        d = opt.default
        if opt.kind == "table":
            w = ElementTable()
            w.changed.connect(lambda k=opt.key: self._changed(k))
        elif opt.kind == "color":
            w = _ColorButton(d)
            w.changed.connect(lambda k=opt.key: self._changed(k))
        elif opt.kind in ("file", "dir"):
            w = _PathEdit(opt.kind)
            w.edit.textChanged.connect(lambda _, k=opt.key: self._changed(k))
        elif opt.choices:
            w = QtWidgets.QComboBox()
            for c in opt.choices:
                w.addItem(_CHOICE_LABELS.get(c, c), c)
            w.currentIndexChanged.connect(lambda _, k=opt.key: self._changed(k))
        elif isinstance(d, bool):
            w = Toggle()
            w.toggled.connect(lambda _, k=opt.key: self._changed(k))
        elif isinstance(d, int):
            w = QtWidgets.QSpinBox(minimum=int(opt.minimum or 0), maximum=int(opt.maximum or 10**9))
            w.setGroupSeparatorShown(d >= 1000)
            if opt.key == "compute.threads":
                w.setSpecialValueText(f"All cores ({os.cpu_count()})")
            w.valueChanged.connect(lambda _, k=opt.key: self._changed(k))
        elif isinstance(d, float):
            w = QtWidgets.QDoubleSpinBox(minimum=opt.minimum or 0, maximum=opt.maximum or 1e9,
                                         decimals=1 if opt.key == "viewport.rotation_step" else 2,
                                         singleStep=0.05 if (opt.maximum or 10) <= 5 else 1)
            w.valueChanged.connect(lambda _, k=opt.key: self._changed(k))
        else:
            w = QtWidgets.QLineEdit(placeholderText="Not set", objectName="field")
            w.textChanged.connect(lambda _, k=opt.key: self._changed(k))
        if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            w.setAlignment(QtCore.Qt.AlignRight)
            suffix = {"general.font_size": " px", "viewport.rotation_step": "°", "bonds.radius": " Å",
                      "render.dpi": " dpi", "viewport.mesh_limit": " atoms",
                      "viewport.interactive_atoms": " atoms", "viewport.target_fps": " fps",
                      "viewport.refine_delay": " s", "performance.cache_memory": " MB"}.get(opt.key)
            if suffix:
                w.setSuffix(suffix)
        return w

    def _device_combo(self):
        combo = QtWidgets.QComboBox()
        self.device_detail = {}
        for key, label, probe in DEVICES:
            ok, detail = probe()
            combo.addItem(label, key)
            idx = combo.count() - 1
            combo.setItemData(idx, detail, QtCore.Qt.ToolTipRole)
            if not ok:
                combo.model().item(idx).setEnabled(False)
            self.device_detail[key] = (ok, detail)
        combo.currentIndexChanged.connect(lambda _: (self._changed("compute.device"), self._device_note()))
        note = QtWidgets.QHBoxLayout()
        self.device_badge = QtWidgets.QLabel()
        self.device_note_label = QtWidgets.QLabel(objectName="rowHelp")
        holder = QtWidgets.QWidget(objectName="noteHolder")
        note.setContentsMargins(0, 4, 0, 0)
        note.addWidget(self.device_badge)
        note.addWidget(self.device_note_label, 1)
        holder.setLayout(note)
        self._pending_note = holder
        return combo

    def _device_note(self):
        combo = self.editors["compute.device"][0]
        ok, detail = self.device_detail.get(combo.currentData(), (True, ""))
        self.device_badge.setText("Available" if ok else "Unavailable")
        _restyle(self.device_badge, "badgeOk" if ok else "badgeOff")
        self.device_note_label.setText(detail)

    def _components(self, rows):
        self.component_rows = {}
        for mod, purpose, found, version, cmd in settings_mod.component_status():
            row = self._row_frame(rows)
            lay = QtWidgets.QHBoxLayout(row)
            lay.setContentsMargins(16, 10, 16, 10)
            text = QtWidgets.QVBoxLayout()
            text.setSpacing(2)
            text.addWidget(QtWidgets.QLabel(purpose, objectName="rowLabel"))
            text.addWidget(QtWidgets.QLabel(mod, objectName="mono"))
            lay.addLayout(text, 1)
            if found:
                lay.addWidget(QtWidgets.QLabel(f"Installed {version}".strip(), objectName="badgeOk"))
            else:
                lay.addWidget(QtWidgets.QLabel(cmd, objectName="mono"))
                copy = QtWidgets.QPushButton("Copy", objectName="secondary", toolTip="Copy the install command")
                copy.clicked.connect(lambda _=False, c=cmd: QtWidgets.QApplication.clipboard().setText(c))
                lay.addWidget(copy)
                lay.addWidget(QtWidgets.QLabel("Not installed", objectName="badgeOff"))
            self.component_rows[mod] = row

    # ---------------------------------------------------------------- values
    def _load(self):
        self._loading = True
        for key, (w, _l, _r) in self.editors.items():
            self._set_editor(w, OPTION[key], self.work[key])
        self._loading = False
        self._device_note()
        self._mark_all()

    def _set_editor(self, w, opt, value):
        if isinstance(w, ElementTable):
            w.set_overrides(value, self.work)
        elif isinstance(w, _ColorButton):
            w.set(value)
        elif isinstance(w, _PathEdit):
            w.edit.setText(value)
        elif isinstance(w, QtWidgets.QComboBox):
            w.setCurrentIndex(max(0, w.findData(value)))
        elif isinstance(w, QtWidgets.QCheckBox):
            w.setChecked(value)
        elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            w.setValue(value)
        else:
            w.setText(value)

    @staticmethod
    def _value(w):
        if isinstance(w, ElementTable):
            return w.overrides
        if isinstance(w, _ColorButton):
            return w.color
        if isinstance(w, _PathEdit):
            return w.edit.text()
        if isinstance(w, QtWidgets.QComboBox):
            return w.currentData()
        if isinstance(w, QtWidgets.QCheckBox):
            return w.isChecked()
        if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            return w.value()
        return w.text()

    def _changed(self, key):
        if self._loading:
            return
        w = self.editors[key][0]
        try:
            self.work[key] = self._value(w)
        except ValueError:
            return
        if key.startswith("elements.") and key != "elements.overrides":
            self._refresh_table()
        self._mark(key)

    def _refresh_table(self):
        table = self.editors["elements.overrides"][0]
        self._loading = True
        table.set_overrides(table.overrides, self.work)
        self._loading = False

    def _mark(self, key):
        changed = not self.work.is_default(key)
        dot, reset = self.row_parts[key]
        label = self.editors[key][1]
        dot.setVisible(changed)
        reset.setVisible(changed)
        if label.objectName() != ("rowLabelChanged" if changed else "rowLabel"):
            _restyle(label, "rowLabelChanged" if changed else "rowLabel")

    def _mark_all(self):
        for key in self.editors:
            self._mark(key)

    def set_value(self, key, value):
        """Programmatic edit (tests, command palette): updates the editor and the working copy."""
        self._set_editor(self.editors[key][0], OPTION[key], value)
        self.work[key] = value
        self._mark(key)

    def reset_key(self, key):
        self.work.reset([key])
        self._loading = True
        self._set_editor(self.editors[key][0], OPTION[key], self.work[key])
        self._loading = False
        if key.startswith("elements."):
            self._refresh_table()
        if key == "compute.device":
            self._device_note()
        self._mark(key)

    def restore_page(self):
        page = PAGES[self.stack.currentIndex()]
        self.work.reset([o.key for o in OPTIONS if o.page == page])
        self._load()

    def _search(self, text):
        text = text.strip().lower()
        matched_pages = set()
        for heading, card, keys in self.cards:
            first = True
            for key in keys:
                opt = OPTION[key]
                hit = not text or any(text in s.lower() for s in (opt.label, opt.key, opt.help, opt.page))
                row = self.editors[key][2]
                row.setVisible(hit)
                if hit:
                    matched_pages.add(opt.page)
                    want = "rowFirst" if first else "row"
                    if row.objectName() != want:
                        _restyle(row, want)
                    first = False
            heading.setVisible(not first)
            card.setVisible(not first)
        comp = not text or any(text in s for s in ("component", "install", "package", "ovito", "dxa", "cupy"))
        self.components_card[0].setVisible(comp)
        self.components_card[1].setVisible(comp)
        if comp:
            matched_pages.add("Compute")
        first_page = None
        for i, name in enumerate(PAGES):
            hidden = name not in matched_pages
            self.pages.item(i).setHidden(hidden)
            if not hidden and first_page is None:
                first_page = i
        if first_page is not None and self.pages.item(self.pages.currentRow()).isHidden():
            self.pages.setCurrentRow(first_page)

    def _import(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import settings", "", "JSON (*.json)")
        if path:
            self.work.update(settings_mod.load(path))
            self._load()

    def _export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export settings", "ewald-settings.json", "JSON (*.json)")
        if path:
            self.work.save(path)

    def apply(self):
        self.settings.update(self.work)
        try:
            self.settings.save()
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Preferences", f"Could not save {self.settings.path}:\n{exc}")
        self.applied.emit(self.settings)

    def accept(self):
        self.apply()
        super().accept()


# ---------------------------------------------------------------- applying to the main window
def apply_element_defaults(style, data, s: Settings):
    """Fill per-type colors/radii from the settings without touching what the user set in the session.

    Returns the :class:`ewald.radii.RadiusTable` used for the radii (None when nothing was filled).
    """
    from ..radii import IONIC, element_of

    # drop what we filled last time (unless the user has since edited it) so scheme changes take effect
    for t, (c, r) in getattr(style, "_pref_filled", {}).items():
        if c is not None and style.type_colors.get(t) == c:
            style.type_colors.pop(t, None)
        if r is not None and style.type_radii.get(t) == r:
            style.type_radii.pop(t, None)
    filled = {}
    style._pref_filled = filled
    overrides = s["elements.overrides"]
    fill_colors = not s.is_default("elements.color_scheme") or any("color" in o for o in overrides.values())
    # ionic radii depend on the structure: resolve them once per composition rather than on every draw
    fill_radii = (not s.is_default("elements.radius_scale") or any("radius" in o for o in overrides.values())
                  or style.radius_preset in IONIC)
    if not (fill_colors or fill_radii):
        return None  # built-in defaults already match; keep generated scripts free of redundant type_colors
    table = None
    if fill_radii:
        uniq, counts = np.unique(np.asarray(data.types).astype(str), return_counts=True)
        key = (style.radius_preset, tuple(sorted(style.oxidation_states.items())), tuple(uniq), tuple(counts),
               s["elements.radius_scale"], repr(sorted(overrides.items())))
        cache = getattr(style, "_radius_cache", None)
        if cache is None or cache[0] != key:
            cache = style._radius_cache = (key, *s.type_radii(data, style.radius_preset, style.oxidation_states))
        radii, table = cache[1], cache[2]
    for t in dict.fromkeys(str(t) for t in data.types):
        sym = element_of(t)
        if sym is None:
            continue
        c = r = None
        if fill_colors and t not in style.type_colors:
            c = style.type_colors[t] = _hex(s.element_color(sym))
        if fill_radii and t not in style.type_radii and t in radii:
            r = style.type_radii[t] = radii[t]
        if c is not None or r is not None:
            filled[t] = (c, r)
    return table


def apply_settings(win, s: Settings, startup: bool = False):
    """Push settings into a MainWindow: viewport, style, render page, device, font."""
    from . import theme

    vp, style, page = win.viewport, win.style, win.render_page
    win.settings = s
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.setStyleSheet(theme.stylesheet().replace("font-size: 12px", f"font-size: {s['general.font_size']}px"))

    win.set_button_style(s["general.toolbar_style"])
    if style.preset not in PRESET_THEMES:  # those presets keep their theme's background
        style.background = s["viewport.background"]
    style.bond_radius = s["bonds.radius"]
    if startup or style.preset != s["viewport.style_preset"]:
        style.apply_preset(s["viewport.style_preset"], win.data.types if win.data is not None else ())
    vp.rotation_step = s["viewport.rotation_step"]
    vp.sphere_drawing = s["viewport.sphere_drawing"]
    vp.sphere_resolution = s["viewport.sphere_resolution"]
    vp.set_anti_aliasing(s["viewport.anti_aliasing"])
    vp.mesh_limit = s["viewport.mesh_limit"]
    vp.arrow_resolution = s["viewport.arrow_resolution"]
    vp.detail_mode = s["viewport.interactive_detail"]
    vp.target_fps = s["viewport.target_fps"]
    vp.interactive_atoms = s["viewport.interactive_atoms"]
    vp.refine_delay = s["viewport.refine_delay"]
    if win.pipeline is not None:
        win.pipeline.cache_memory = s["performance.cache_memory"] * 2**20
    vp.show_overlay = s["viewport.show_overlay"]
    vp.initial_view = s["viewport.initial_view"]
    vp.set_axes_visible(s["viewport.show_axes"])
    vp.ortho_button.setChecked(s["viewport.projection"] == "ortho")

    page.set_renderer(s["render.renderer"])
    page.samples.setValue(s["render.samples"])
    page.ao_samples.setValue(s["render.ao_samples"])
    page.shadows.setChecked(s["render.shadows"])
    page.quality.setCurrentText(s["render.quality"])  # after the Custom values: a preset overwrites them
    page._quality_changed(s["render.quality"])
    page.movie_quality.setCurrentText(s["render.movie_quality"])
    page.dpi.setValue(s["render.dpi"])
    page.preset.setCurrentText(s["render.size_preset"])
    page.background.setCurrentText(s["render.background"])
    page.type_legend.setChecked(s["render.type_legend"])
    page.scale_bar.setChecked(s["render.scale_bar"])
    page.movie_format.blockSignals(True)
    page.movie_format.setCurrentText(s["render.movie_format"])
    page.movie_format.blockSignals(False)
    if s["render.output_dir"] and startup:
        page.output.setText(os.path.join(s["render.output_dir"], os.path.basename(page.output.text() or "render.png")))

    if s["compute.threads"]:
        try:
            import numba

            numba.set_num_threads(min(s["compute.threads"], numba.config.NUMBA_NUM_THREADS))
        except Exception as exc:
            win.message(f"Could not set CPU threads: {exc}", error=True)
    if startup and win.device is None:
        win.device = s["compute.device"]
        probe = dict((k, p) for k, _label, p in DEVICES).get(win.device)
        ok, detail = probe() if probe else (False, "unknown device")
        if not ok:
            win.message(f"Default device {win.device!r} unavailable ({detail}); using CPU", error=True)
            win.device = "cpu"
        label = {"cpu": "CPU", "cuda": "CUDA", "auto": "auto"}.get(win.device, "CPU")
        win.device_combo.blockSignals(True)
        win.device_combo.setCurrentText(label)
        win.device_combo.blockSignals(False)
    for p in filter(None, (x.strip() for x in s["python.extra_paths"].split(";"))):
        if p not in sys.path:
            sys.path.append(p)

    if getattr(style, "bond_factor", None) != s["bonds.length_factor"]:
        style.bond_factor = s["bonds.length_factor"]
        style.bond_pairs.clear()  # re-seeded from the new factor on the next draw
    if startup or getattr(style, "_pref_preset", None) != s["elements.radius_source"]:
        style.radius_preset = style._pref_preset = s["elements.radius_source"]
    if win.data is not None:
        if style.bonds_enabled:
            style.ensure_bond_pairs(win.data.types)
        apply_element_defaults(style, win.data, s)
    win._style_changed()


def save_layout(win, s: Settings):
    if s["general.restore_layout"]:
        state = {"geometry": bytes(win.saveGeometry().toBase64()).decode(),
                 "state": bytes(win.saveState().toBase64()).decode()}
        try:
            layout_path(s).parent.mkdir(parents=True, exist_ok=True)
            layout_path(s).write_text(json.dumps(state))
        except OSError:
            pass


def restore_layout(win, s: Settings):
    p = layout_path(s)
    if not s["general.restore_layout"] or not p.exists():
        return
    try:
        state = json.loads(p.read_text())
        win.restoreGeometry(QtCore.QByteArray(base64.b64decode(state["geometry"])))
        win.restoreState(QtCore.QByteArray(base64.b64decode(state["state"])))
    except Exception:
        pass


def layout_path(s: Settings):
    return s.path.with_name("layout.json")


def run_startup_script(win, s: Settings):
    path = s["python.startup_script"]
    if not path:
        return
    shell = getattr(win.console, "shell", None)
    try:
        src = open(path).read()
    except OSError as exc:
        win.message(f"Startup script {path}: {exc}", error=True)
        return
    try:
        if hasattr(shell, "run_cell"):
            shell.run_cell(src, store_history=False)
        elif shell is not None:
            exec(compile(src, path, "exec"), shell.locals)
        win.message(f"Ran startup script {path}")
    except Exception as exc:
        win.message(f"Startup script {path}: {exc}", error=True)
