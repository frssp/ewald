"""Preferences dialog (Edit › Preferences, Ctrl+,): OVITO-style page list + VESTA element table.

Design: /design/settings-menu.md. The form is generated from ``ewald.settings.OPTIONS``;
``apply_settings`` pushes a Settings object into a running MainWindow.
"""
from __future__ import annotations

import base64
import json
import os
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from .. import settings as settings_mod
from ..settings import DEVICES, OPTION, OPTIONS, PAGES, Settings
from .theme import TOKENS

_CHOICE_LABELS = {"ortho": "Orthographic", "persp": "Perspective", "raytrace": "Ray tracing (AO + shadows)",
                  "opengl": "OpenGL (fast preview)", "iso": "Isometric", "x": "+x", "y": "+y", "z": "+z (top)",
                  "front": "Front (-y)", "ask": "Ask when opening", "never": "Never ask"}


def _hex(c) -> str:
    if isinstance(c, str):
        return QtGui.QColor(c).name()
    return QtGui.QColor.fromRgbF(*[float(x) for x in c[:3]]).name()


class _ColorButton(QtWidgets.QPushButton):
    changed = QtCore.Signal()

    def __init__(self, color="#ffffff", parent=None):
        super().__init__(parent)
        self.setFixedWidth(84)
        self.clicked.connect(self._pick)
        self.set(color)

    def set(self, color):
        self.color = _hex(color)
        self.setText(self.color)
        light = QtGui.QColor(self.color).lightnessF() > 0.55
        self.setStyleSheet(f"background: {self.color}; color: {'#0B1220' if light else '#E7E9EC'}; "
                           f"border: 1px solid {TOKENS['border']};")

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
        self.edit = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(30)
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
        self.filter = QtWidgets.QLineEdit(placeholderText="Filter elements (e.g. Cu, O)")
        self.filter.textChanged.connect(self._filter)
        lay.addWidget(self.filter)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Element", "Color", "Radius (Å)", ""])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 70)
        self.table.setColumnWidth(1, 96)
        self.table.setMinimumHeight(300)
        lay.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        imp = QtWidgets.QPushButton("Import VESTA elements.ini…")
        imp.clicked.connect(self._import)
        reset = QtWidgets.QPushButton("Reset all elements")
        reset.clicked.connect(lambda: self.set_overrides({}, self.base) or self.changed.emit())
        row.addWidget(imp)
        row.addWidget(reset)
        row.addStretch()
        lay.addLayout(row)

    def set_overrides(self, overrides: dict, base: Settings):
        from ase.data import chemical_symbols

        self.overrides = {k: dict(v) for k, v in overrides.items()}
        self.base = base
        symbols = chemical_symbols[1:97]
        self.table.setRowCount(len(symbols))
        for r, sym in enumerate(symbols):
            o = self.overrides.get(sym, {})
            item = QtWidgets.QTableWidgetItem(sym)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.table.setItem(r, 0, item)
            btn = _ColorButton(o.get("color") or self._scheme(sym, "color"))
            btn.changed.connect(lambda s=sym, b=btn: self._edit(s, "color", b.color))
            self.table.setCellWidget(r, 1, btn)
            spin = QtWidgets.QDoubleSpinBox(minimum=0.05, maximum=5, singleStep=0.05, decimals=3)
            spin.setValue(o.get("radius") or self._scheme(sym, "radius"))
            spin.valueChanged.connect(lambda v, s=sym: self._edit(s, "radius", round(v, 3)))
            self.table.setCellWidget(r, 2, spin)
            self._mark(r, sym)
        self._filter(self.filter.text())

    def _scheme(self, sym, what):
        probe = self.base.copy()
        ov = probe["elements.overrides"]
        ov.pop(sym, None)
        probe["elements.overrides"] = ov
        return probe.element_color(sym) if what == "color" else probe.element_radius(sym)

    def _mark(self, r, sym):
        edited = sym in self.overrides
        item = QtWidgets.QTableWidgetItem("● edited" if edited else "")
        item.setForeground(QtGui.QColor(TOKENS["accent"]))
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.table.setItem(r, 3, item)

    def _edit(self, sym, field, value):
        self.overrides.setdefault(sym, {})[field] = value
        from ase.data import atomic_numbers

        self._mark(atomic_numbers[sym] - 1, sym)
        self.changed.emit()

    def _filter(self, text):
        wanted = {t.strip().lower() for t in text.replace(",", " ").split() if t.strip()}
        for r in range(self.table.rowCount()):
            sym = self.table.item(r, 0).text().lower()
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

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(820, 600)
        self.settings = settings
        self.work = settings.copy()
        self.editors: dict[str, tuple] = {}   # key -> (widget, row label, row widget)
        self._loading = False

        outer = QtWidgets.QVBoxLayout(self)
        self.search = QtWidgets.QLineEdit(placeholderText="Search settings (e.g. background, cuda, radius)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._search)
        outer.addWidget(self.search)
        body = QtWidgets.QHBoxLayout()
        self.pages = QtWidgets.QListWidget()
        self.pages.setFixedWidth(160)
        self.stack = QtWidgets.QStackedWidget()
        body.addWidget(self.pages)
        body.addWidget(self.stack, 1)
        outer.addLayout(body, 1)
        self.page_forms: dict[str, QtWidgets.QFormLayout] = {}
        for name in PAGES:
            self._build_page(name)
        self.pages.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.pages.setCurrentRow(0)

        buttons = QtWidgets.QHBoxLayout()
        for text, fn in (("Restore page defaults", self.restore_page), ("Import…", self._import),
                         ("Export…", self._export)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            buttons.addWidget(b)
        buttons.addStretch()
        self.path_label = QtWidgets.QLabel(str(settings.path), objectName="muted")
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        buttons.addWidget(self.path_label)
        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Apply
                                         | QtWidgets.QDialogButtonBox.Cancel)
        box.button(QtWidgets.QDialogButtonBox.Ok).setObjectName("primary")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        box.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.apply)
        buttons.addWidget(box)
        outer.addLayout(buttons)
        self._load()

    # ---------------------------------------------------------------- building
    def _build_page(self, name):
        inner = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(inner)
        form.setContentsMargins(12, 8, 12, 12)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        title = QtWidgets.QLabel(name)
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        form.addRow(title)
        if name == "Compute":
            self._device_rows(form)
        for opt in OPTIONS:
            if opt.page != name or opt.key == "compute.device":
                continue
            self._add_row(form, opt, self._editor(opt))
        if name == "Compute":
            form.addRow(QtWidgets.QLabel("Optional components", objectName="section"))
            form.addRow(self._components_table())
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.stack.addWidget(scroll)
        self.pages.addItem(name)
        self.page_forms[name] = form

    def _add_row(self, form, opt, editor):
        holder = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(holder)
        v.setContentsMargins(0, 0, 0, 4)
        v.setSpacing(1)
        v.addWidget(editor)
        if opt.help:
            h = QtWidgets.QLabel(opt.help, objectName="muted", wordWrap=True)
            v.addWidget(h)
        label = QtWidgets.QLabel(opt.label)
        label.setToolTip(opt.key)
        form.addRow(label, holder)
        self.editors[opt.key] = (editor, label, holder)

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
            w = QtWidgets.QCheckBox()
            w.toggled.connect(lambda _, k=opt.key: self._changed(k))
        elif isinstance(d, int):
            w = QtWidgets.QSpinBox(minimum=int(opt.minimum or 0), maximum=int(opt.maximum or 10**9))
            if opt.key == "compute.threads":
                w.setSpecialValueText(f"all ({os.cpu_count()})")
            w.valueChanged.connect(lambda _, k=opt.key: self._changed(k))
        elif isinstance(d, float):
            w = QtWidgets.QDoubleSpinBox(minimum=opt.minimum or 0, maximum=opt.maximum or 1e9, decimals=3,
                                         singleStep=0.05 if (opt.maximum or 10) <= 5 else 1)
            w.valueChanged.connect(lambda _, k=opt.key: self._changed(k))
        else:
            w = QtWidgets.QLineEdit()
            w.textChanged.connect(lambda _, k=opt.key: self._changed(k))
        return w

    def _device_rows(self, form):
        opt = OPTION["compute.device"]
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
        combo.currentIndexChanged.connect(lambda _: (self._changed(opt.key), self._device_note()))
        self._add_row(form, opt, combo)
        self.device_note = QtWidgets.QLabel(objectName="muted", wordWrap=True)
        self.editors[opt.key][2].layout().addWidget(self.device_note)

    def _device_note(self):
        combo = self.editors["compute.device"][0]
        ok, detail = self.device_detail.get(combo.currentData(), (True, ""))
        self.device_note.setText(("● " if ok else "✕ unavailable: ") + detail)

    def _components_table(self):
        rows = settings_mod.component_status()
        t = QtWidgets.QTableWidget(len(rows), 3)
        t.setHorizontalHeaderLabels(["Component", "Status", "Install"])
        t.verticalHeader().hide()
        t.horizontalHeader().setStretchLastSection(True)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for r, (mod, purpose, found, version, cmd) in enumerate(rows):
            t.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{purpose}  ({mod})"))
            st = QtWidgets.QTableWidgetItem(f"✓ {version}" if found else "not installed")
            st.setForeground(QtGui.QColor(TOKENS["ok"] if found else TOKENS["muted"]))
            t.setItem(r, 1, st)
            t.setItem(r, 2, QtWidgets.QTableWidgetItem("" if found else cmd))
        t.resizeColumnsToContents()
        t.setMinimumHeight(34 * (len(rows) + 1))
        self.components = t
        return t

    # ---------------------------------------------------------------- values
    def _load(self):
        self._loading = True
        for key, (w, _l, _h) in self.editors.items():
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
            table = self.editors["elements.overrides"][0]
            self._loading = True
            table.set_overrides(table.overrides, self.work)
            self._loading = False
        self._mark(key)

    def _mark(self, key):
        opt, label = OPTION[key], self.editors[key][1]
        label.setText(opt.label if self.work.is_default(key) else f"{opt.label}  ●")
        label.setStyleSheet("" if self.work.is_default(key) else f"color: {TOKENS['accent']};")

    def _mark_all(self):
        for key in self.editors:
            self._mark(key)

    def set_value(self, key, value):
        """Programmatic edit (tests, command palette): updates the editor and the working copy."""
        self._set_editor(self.editors[key][0], OPTION[key], value)
        self.work[key] = value
        self._mark(key)

    def restore_page(self):
        page = PAGES[self.stack.currentIndex()]
        self.work.reset([o.key for o in OPTIONS if o.page == page])
        self._load()

    def _search(self, text):
        text = text.strip().lower()
        matched_pages = set()
        for key, (_w, label, holder) in self.editors.items():
            opt = OPTION[key]
            hit = not text or any(text in s.lower() for s in (opt.label, opt.key, opt.help, opt.page))
            label.setVisible(hit)
            holder.setVisible(hit)
            if hit:
                matched_pages.add(opt.page)
        if text == "" or "component" in text:
            matched_pages.add("Compute")
        first = None
        for i, name in enumerate(PAGES):
            hidden = name not in matched_pages
            self.pages.item(i).setHidden(hidden)
            if not hidden and first is None:
                first = i
        if first is not None and self.pages.item(self.pages.currentRow()).isHidden():
            self.pages.setCurrentRow(first)

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
def apply_element_defaults(style, types, s: Settings):
    """Fill per-type colors/radii from the settings without touching what the user set in the session."""
    # drop what we filled last time (unless the user has since edited it) so scheme changes take effect
    for t, (c, r) in getattr(style, "_pref_filled", {}).items():
        if style.type_colors.get(t) == c and style.type_radii.get(t) == r:
            style.type_colors.pop(t, None)
            style.type_radii.pop(t, None)
    filled = {}
    element_keys = ("elements.color_scheme", "elements.radius_source", "elements.radius_scale", "elements.overrides")
    if all(s.is_default(k) for k in element_keys):
        style._pref_filled = filled
        return  # built-in defaults already match; keep generated scripts free of redundant type_colors
    for t in dict.fromkeys(types):
        if t in style.type_colors or t in style.type_radii:
            continue
        c, r = s.element_color(t), s.element_radius(t)
        if c is None:
            continue
        style.type_colors[t] = _hex(c)
        style.type_radii[t] = r
        filled[t] = (style.type_colors[t], r)
    style._pref_filled = filled


def apply_settings(win, s: Settings, startup: bool = False):
    """Push settings into a MainWindow: viewport, style, render page, device, font."""
    from . import theme

    vp, style, page = win.viewport, win.style, win.render_page
    win.settings = s
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.setStyleSheet(theme.stylesheet().replace("font-size: 12px", f"font-size: {s['general.font_size']}px"))

    style.background = s["viewport.background"]
    style.bond_radius = s["bonds.radius"]
    if startup or style.preset != s["viewport.style_preset"]:
        style.apply_preset(s["viewport.style_preset"])
    vp.rotation_step = s["viewport.rotation_step"]
    vp.sphere_resolution = s["viewport.sphere_resolution"]
    vp.mesh_limit = s["viewport.mesh_limit"]
    vp.show_overlay = s["viewport.show_overlay"]
    vp.initial_view = s["viewport.initial_view"]
    vp.set_axes_visible(s["viewport.show_axes"])
    vp.ortho_button.setChecked(s["viewport.projection"] == "ortho")

    page.renderer.setCurrentIndex(0 if s["render.renderer"] == "raytrace" else 1)
    page.samples.setValue(s["render.samples"])
    page.ao_samples.setValue(s["render.ao_samples"])
    page.shadows.setChecked(s["render.shadows"])
    page.dpi.setValue(s["render.dpi"])
    page.preset.setCurrentText(s["render.size_preset"])
    page.background.setCurrentText(s["render.background"])
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
    if win.data is not None:
        if style.bonds_enabled:
            style.ensure_bond_pairs(win.data.types)
        apply_element_defaults(style, win.data.types, s)
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
