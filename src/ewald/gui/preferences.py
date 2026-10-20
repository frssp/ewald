"""Preferences dialog (Edit › Preferences, Ctrl+,): OVITO-style page list + VESTA element table.

Design: /design/settings-menu.md. Rows are generated from ``ewald.settings.OPTIONS`` and
grouped into sections by ``LAYOUT``; ``apply_settings`` pushes a Settings object into a
running MainWindow.
"""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from .. import settings as settings_mod
from ..settings import DEVICES, OPTION, OPTIONS, PAGES, Settings
from . import header
from .element_table import ElementTable  # noqa: F401 (imported from here by callers)
from .pref_widgets import Toggle, _ColorButton, _PathEdit, _ProfileBox, _chevron, _icon, _restyle
from .prefs_apply import (apply_element_defaults, apply_settings, layout_path, restore_layout,  # noqa: F401
                          run_startup_script, save_layout)
from .theme import TOKENS

_CHOICE_LABELS = {"auto": "Auto", "coarse": "Always coarse", "full": "Always full", "ortho": "Orthographic", "persp": "Perspective", "raytrace": "Ray tracing (AO + shadows)",
                  "opengl": "OpenGL (fast preview)", "iso": "Isometric", "x": "Along +x", "y": "Along +y",
                  "z": "Along +z (top)", "front": "Front (−y)", "ask": "Ask when opening", "never": "Use default map",
                  "covalent": "Covalent", "vdW": "van der Waals", "white": "White", "black": "Black",
                  "transparent": "Transparent", "viewport": "Same as viewport",
                  "claude-opus-5-5": "Claude Opus 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5 (faster)",
                  "claude-haiku-5-5": "Claude Haiku 5.5 (fastest)",
                  "claude-fable-5-1": "Claude Fable 5.1 (most capable)", "low": "Low", "medium": "Medium",
                  "high": "High", "xhigh": "Extra high", "max": "Max"}

# page -> (subtitle, [(section, [option keys])])
LAYOUT = {
    "General": ("Startup behaviour and file handling.", [
        ("Startup", ["general.startup_workspace", "general.restore_layout"]),
        ("Appearance", ["general.ui_theme", "general.font_size", "general.toolbar_style"]),
        ("LAMMPS files", ["general.lammps_type_prompt", "general.lammps_type_map"]),
        ("Running jobs", ["general.live_reload", "general.live_reload_interval"]),
        ("Help", ["general.docs_url"])]),
    "Viewports": ("Defaults for the interactive 3D view.", [
        ("View", ["viewport.follow_theme", "viewport.background", "viewport.projection", "viewport.initial_view", "viewport.style_preset"]),
        ("Navigation", ["viewport.rotation_step"]),
        ("Overlays", ["viewport.show_axes", "viewport.show_overlay"]),
        ("Hover", ["viewport.hover_info", "viewport.hover_delay"]),
        ("Performance", ["viewport.backend", "viewport.interactive_detail", "viewport.target_fps", "viewport.interactive_atoms",
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
        ("Overlays and movies", ["render.type_legend", "render.scale_bar", "render.movie_format"]),
        ("Plots", ["plots.size", "plots.font_size"])]),
    "Compute": ("Where pipelines run, and which optional packages are available.", [
        ("Device", ["compute.device", "compute.threads", "performance.background_compute"]),
        ("Memory", ["performance.cache_memory"])]),
    "Python": ("Notebooks, console and module search path.", [
        ("Notebooks", ["python.run_notebooks", "python.plot_style", "python.save_figures",
                        "python.bundle_data_mb"]),
        ("Plugins", ["python.plugin_catalogs"]),
        ("Console", ["python.startup_script"]),
        ("Modules", ["python.extra_paths"])]),
    "AI": ("The AI service behind Ask AI and the AI chat, its API key, model and connection.", [
        ("Saved connections", ["llm.profiles"]),
        ("Provider", ["llm.provider"]),
        ("Model", ["llm.model", "llm.effort"]),
        ("Connection", ["llm.base_url", "llm.route_prefix", "llm.user_id", "llm.system_name", "llm.headers",
                        "llm.timeout"])]),
}

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
QFrame#card QAbstractSpinBox::up-arrow, QFrame#card QAbstractSpinBox::down-arrow {{ image: none; width: 0px; }}
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
QToolButton#rowReset {{ border: none; background: transparent; padding: 0; }}
QToolButton#rowReset:hover {{ background: {hover}; }}
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
QPushButton#footerPrimary {{ background: {accent}; color: {on_accent}; border: none; border-radius: 6px; padding: 6px 18px;
    min-width: 64px; font-weight: 600; }}
QPushButton#footerPrimary:hover {{ background: {accent}; }}
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
            if keys == ["llm.provider"]:  # the key of the chosen provider comes right after it
                col.addSpacing(16)
                heading = QtWidgets.QLabel("API KEY", objectName="sectionTitle")
                col.addWidget(heading)
                card, rows = self._card()
                self._api_key_rows(rows)
                col.addWidget(card)
                self.api_key_card = (heading, card)
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
        wide = opt.kind in ("table", "ai_profiles")
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
        reset = header.button("reset", "Reset to default", name="rowReset")
        reset.setFixedSize(20, 20)
        reset.setIconSize(QtCore.QSize(14, 14))
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
        elif opt.kind == "ai_provider":
            from ..llm.providers import providers

            w = QtWidgets.QComboBox()
            for name, p in providers().items():
                w.addItem(p.label, name)
                w.setItemData(w.count() - 1, p.description, QtCore.Qt.ToolTipRole)
            w.currentIndexChanged.connect(lambda _, k=opt.key: (self._changed(k), self._provider_changed()))
        elif opt.kind == "ai_profiles":
            w = _ProfileBox()
            w.changed.connect(lambda k=opt.key: self._changed(k))
            w.chosen.connect(self.load_profile)
            w.save_requested.connect(self.save_profile)
        elif opt.kind == "ai_model":
            w = QtWidgets.QComboBox(editable=True)
            w.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
            w.currentTextChanged.connect(lambda _, k=opt.key: self._changed(k))
        elif opt.choices or opt.kind == "ui_theme":
            w = QtWidgets.QComboBox()
            for c in opt.choices or _ui_theme_names(d):
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

    def _api_key_rows(self, rows):
        """Masked key field, where the key comes from, Test connection and Remove.

        The saved key is never loaded into the field; Apply/OK save what was typed into the
        keychain (``ewald.llm.credentials``), never into settings.json.
        """
        from ..llm import credentials

        self._key_removed = False
        self._key_worker = None
        row = self._row_frame(rows)
        grid = QtWidgets.QGridLayout(row)
        grid.setContentsMargins(16, 12, 16, 12)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(2)
        self.key_label = QtWidgets.QLabel("API key", objectName="rowLabel")
        text.addWidget(self.key_label)
        where = credentials.keyring_name()
        text.addWidget(QtWidgets.QLabel(
            f"Saved in {where}, never in the settings file." if where else
            "No system keychain here: saved to credentials.json in the settings folder, readable "
            "only by you.", objectName="rowHelp", wordWrap=True))
        self.key_status = QtWidgets.QLabel(objectName="rowHelp", wordWrap=True)
        text.addWidget(self.key_status)
        grid.addLayout(text, 0, 0)
        grid.setColumnStretch(0, 1)
        self.key_edit = QtWidgets.QLineEdit(objectName="field")
        self.key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.key_edit.setFixedWidth(self.EDITOR_WIDTH)
        self.key_edit.textChanged.connect(lambda _: self._key_note(""))
        grid.addWidget(self.key_edit, 0, 1, QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch()
        show = QtWidgets.QPushButton("Show", objectName="secondary", checkable=True, toolTip="Show the typed key")
        show.toggled.connect(lambda on: self.key_edit.setEchoMode(
            QtWidgets.QLineEdit.Normal if on else QtWidgets.QLineEdit.Password))
        buttons.addWidget(show)
        self.key_test = QtWidgets.QPushButton("Test connection", objectName="secondary",
                                              toolTip="Ask the API whether the key and model work (no tokens used)")
        self.key_test.clicked.connect(self.test_api_key)
        self.key_remove = QtWidgets.QPushButton("Remove", objectName="secondary",
                                                toolTip="Forget the saved key when you press Apply or OK")
        self.key_remove.clicked.connect(self.remove_api_key)
        buttons.addWidget(self.key_test)
        buttons.addWidget(self.key_remove)
        grid.addLayout(buttons, 1, 0, 1, 2)
        self.key_result = QtWidgets.QLabel(objectName="rowHelp", wordWrap=True)
        grid.addWidget(self.key_result, 2, 0, 1, 2)
        self.key_result.hide()
        self._key_status()

    def _provider(self):
        """The provider chosen on the AI page (Anthropic when the setting names an unknown one)."""
        from ..llm import providers

        try:
            return providers.get(self.work["llm.provider"])
        except KeyError:
            return providers.get()

    def _provider_changed(self, reset_model: bool = True):
        """Show the chosen provider's key, model suggestions and server address."""
        p = self._provider()
        if reset_model and not self._loading:
            self.key_edit.clear()
            self._key_removed = False
            self._key_note("")
            if "llm.model" in self.editors:
                self.set_value("llm.model", "")   # a model id of the old provider means nothing to the new one
        self.key_label.setText(f"{p.label} {p.key_title}" + ("" if p.key_required else " (optional)"))
        self.key_edit.setPlaceholderText(p.key_hint)
        self.key_edit.setEnabled(bool(p.key_name))
        if "llm.model" in self.editors:
            combo = self.editors["llm.model"][0]
            loading, self._loading = self._loading, True
            text = combo.currentText()
            combo.clear()
            for model_id, label in p.models.items():
                combo.addItem(model_id)
                combo.setItemData(combo.count() - 1, label, QtCore.Qt.ToolTipRole)
            combo.setEditText(text)
            combo.lineEdit().setPlaceholderText(f"Default: {p.default_model}" if p.default_model else "Model id")
            self._loading = loading
        self._connection_rows(p)
        self._key_status()

    # rows the chosen provider uses: llm.<field> -> the Provider.options entry it needs
    _PROVIDER_ROWS = {"llm.route_prefix": "route_prefix", "llm.user_id": "user_id",
                      "llm.system_name": "system_name"}

    def _row_applies(self, key) -> bool:
        p = self._provider()
        if key == "llm.effort":
            return p.effort
        return key not in self._PROVIDER_ROWS or self._PROVIDER_ROWS[key] in p.options

    def _connection_rows(self, p):
        """Placeholders saying what an empty field falls back to; hide rows the provider ignores."""
        from ..llm.providers import gateway_env

        if "llm.base_url" not in self.editors:
            return
        url = p.base_url or (p.model_url(self.work["llm.model"] or p.default_model) if p.model_url else "")
        self.editors["llm.base_url"][0].setPlaceholderText(
            url or ("Required: the gateway address" if p.model_url else "The provider's own address"))
        self.editors["llm.route_prefix"][0].setPlaceholderText(p.model_prefix() or "None")
        self.editors["llm.user_id"][0].setPlaceholderText(
            gateway_env("USER_ID") or os.environ.get("USER") or os.environ.get("USERNAME") or "Not set")
        self.editors["llm.system_name"][0].setPlaceholderText(gateway_env("SYSTEM_NAME", "ewald"))
        self.editors["llm.headers"][0].setPlaceholderText("Name: value; Other: value")
        self.editors["llm.timeout"][0].setSpecialValueText("Provider default")
        if hasattr(self, "search"):
            self._search(self.search.text())

    def save_profile(self, name: str):
        """Save the provider, model and connection shown now as ``name`` (Apply/OK keep it)."""
        from ..llm.providers import profile_from_settings

        box = self.editors["llm.profiles"][0]
        profiles = {**box.profiles, name: profile_from_settings(self.work)}
        box.set(profiles, name)
        self.work["llm.profiles"] = profiles
        self._mark("llm.profiles")

    def load_profile(self, name: str):
        """Fill the AI rows from the saved connection ``name``."""
        from ..llm.providers import PROFILE_FIELDS

        saved = self.work["llm.profiles"].get(name)
        if not saved:
            return
        for field in PROFILE_FIELDS:   # provider first: switching it clears the model
            key = f"llm.{field}"
            self.set_value(key, saved.get(field, OPTION[key].default))
        self.editors["llm.profiles"][0].set(self.work["llm.profiles"], name)

    def _key_status(self):
        from ..llm import credentials

        p = self._provider()
        key, source = (None, None) if self._key_removed else credentials.stored_api_key(p)
        var, env = next(((v, os.environ[v].strip()) for v in p.env_vars if os.environ.get(v, "").strip()),
                        (p.env_var, ""))
        if self._key_removed:
            text = "Will be removed when you press Apply."
        elif key:
            text = f"Saved key {credentials.mask(key)}."
            if env:
                text += f" It is used instead of ${var}."
        elif env:
            text = f"Using ${var} from the environment ({credentials.mask(env)})."
        else:
            text = "Not set. " + (p.key_help or (f"Or set ${p.env_var}." if p.env_var else ""))
        self.key_status.setText(text.strip())
        self.key_remove.setEnabled(bool(key) and not self._key_removed)

    def _key_note(self, text, ok=None):
        self.key_result.setVisible(bool(text))
        self.key_result.setText(text)
        self.key_result.setStyleSheet("" if ok is None else f"color: {TOKENS['ok' if ok else 'error']};")

    def remove_api_key(self):
        self._key_removed = True
        self.key_edit.clear()
        self._key_status()

    def test_api_key(self, wait: bool = False):
        """Check the typed key (or the current one) against the Models API in a worker thread."""
        from ..llm import credentials

        if self._key_worker is not None and self._key_worker.isRunning():
            return
        key = self.key_edit.text().strip() or None
        from ..llm.providers import from_settings

        try:
            p = from_settings(self.work)["provider"]   # with the address, headers and options shown
        except KeyError:
            p = self._provider()
        model = self.work["llm.model"] or None
        self.key_test.setEnabled(False)
        self._key_note("Testing…")
        worker = QtCore.QThread(self)
        result = {}
        worker.run = lambda: result.update(r=credentials.check(key, model, provider=p))

        def done():
            ok, msg = result.get("r", (False, "no result"))
            self._key_note(msg, ok)
            self.key_test.setEnabled(True)

        worker.finished.connect(done)
        self._key_worker = worker
        worker.start()
        if wait:
            worker.wait()
            QtWidgets.QApplication.processEvents()

    def _apply_api_key(self):
        from ..llm import credentials

        typed = self.key_edit.text().strip()
        p = self._provider()
        try:
            if typed and p.key_name:
                credentials.save_api_key(typed, p)
                self.key_edit.clear()
            elif self._key_removed:
                credentials.delete_api_key(p)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Preferences", f"Could not save the API key:\n{exc}")
        self._key_removed = False
        self._key_status()

    # ---------------------------------------------------------------- values
    def _load(self):
        self._loading = True
        for key, (w, _l, _r) in self.editors.items():
            self._set_editor(w, OPTION[key], self.work[key])
        self._loading = False
        self._device_note()
        self._provider_changed(reset_model=False)
        self._mark_all()

    def _set_editor(self, w, opt, value):
        if isinstance(w, _ProfileBox):
            w.set(value)
        elif isinstance(w, ElementTable):
            w.set_overrides(value, self.work)
        elif isinstance(w, _ColorButton):
            w.set(value)
        elif isinstance(w, _PathEdit):
            w.edit.setText(value)
        elif isinstance(w, QtWidgets.QComboBox) and w.isEditable():
            w.setEditText(value)
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
        if isinstance(w, _ProfileBox):
            return w.profiles
        if isinstance(w, ElementTable):
            return w.overrides
        if isinstance(w, _ColorButton):
            return w.color
        if isinstance(w, _PathEdit):
            return w.edit.text()
        if isinstance(w, QtWidgets.QComboBox) and w.isEditable():
            return w.currentText().strip()
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
                hit = (not text or any(text in s.lower() for s in (opt.label, opt.key, opt.help, opt.page))) \
                    and self._row_applies(key)
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
        key_hit = not text or any(text in s for s in ("api key", "anthropic", "claude", "openai", "gemini",
                                                      "ollama", "token", "keychain", "gateway", "credential"))
        self.api_key_card[0].setVisible(key_hit)
        self.api_key_card[1].setVisible(key_hit)
        if key_hit:
            matched_pages.add("AI")
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
        self._apply_api_key()
        self.settings.update(self.work)
        try:
            self.settings.save()
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Preferences", f"Could not save {self.settings.path}:\n{exc}")
        self.applied.emit(self.settings)

    def accept(self):
        self.apply()
        super().accept()


def _ui_theme_names(default: str) -> list[str]:
    """Built-in and plugin UI themes for the combo box (the built-ins when plugins fail)."""
    from .. import plugins

    try:
        return list(plugins.ui_themes())
    except Exception:
        from .theme import PALETTES

        return list(PALETTES)
