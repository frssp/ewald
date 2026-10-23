"""Render workspace: journal-size presets, preview, overlays, and 'Python for this figure'."""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from .busy import BusyLabel
from .theme import TOKENS

SIZE_PRESETS = {
    # label: (width inches, height inches) or (px, px) when 'px'
    "1-column (3.5 in, 4:3)": (3.5, 2.625),
    "2-column (7.2 in, 16:9)": (7.2, 4.05),
    "Square (3.5 in)": (3.5, 3.5),
    "Slide 16:9 (1920×1080)": ("px", 1920, 1080),
    "Custom (px)": None,
}


RENDERERS = ("raytrace", "opengl", "vector")
QUALITIES = ("Draft", "Standard", "Publication", "Custom")
VECTOR_EXTS = (".svg", ".pdf", ".eps", ".ps")
MOVIE_FORMATS = {"MP4 (H.264)": ".mp4", "GIF": ".gif", "AVI (Motion JPEG)": ".avi", "MOV (H.264)": ".mov",
                 "PNG sequence": ".png"}
MOVIE_EXTS = (".mp4", ".gif", ".avi", ".mov", ".webm", ".m4v")
OUTPUT_FILTER = ("Images (*.png *.jpg *.tif);;Vector (*.svg *.pdf *.eps);;"
                 "Movies (*.mp4 *.gif *.avi *.mov);;All files (*)")


class RenderPage(QtWidgets.QWidget):
    renderRequested = QtCore.Signal()
    cancelRequested = QtCore.Signal()  # the Render button while a cancellable render runs
    regionWritten = QtCore.Signal()   # the user changed the frame: the window writes it to pipe.view

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        # left: the live 3D view with the picture's frame (the workbench lends it the viewport), or
        # the last rendered picture
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        bar = QtWidgets.QFrame(objectName="viewbar")
        hb = QtWidgets.QHBoxLayout(bar)
        hb.setContentsMargins(6, 2, 6, 2)
        hb.setSpacing(4)
        self.live_button = QtWidgets.QToolButton(text="3D view", checkable=True, checked=True, autoRaise=True)
        self.live_button.setToolTip("The 3D view with the picture's frame: zoom and turn the view, drag the "
                                    "frame's corners, sides or edge to resize, reshape or move it")
        self.result_button = QtWidgets.QToolButton(text="Last render", checkable=True, autoRaise=True)
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for b in (self.live_button, self.result_button):
            group.addButton(b)
            hb.addWidget(b)
        self.live_button.toggled.connect(lambda on: self.show_live(on))
        hb.addSpacing(12)
        self.fit_button = QtWidgets.QToolButton(text="Fit to structure", autoRaise=True)
        self.fit_button.setToolTip("Zoom and center the view so the structure fills the frame")
        self.fit_button.clicked.connect(self.fit_frame)
        self.reset_frame_button = QtWidgets.QToolButton(text="Reset frame", autoRaise=True)
        self.reset_frame_button.setToolTip("The largest centered frame that fits the view")
        self.reset_frame_button.clicked.connect(lambda: self.set_region(None, write=True))
        hb.addWidget(self.fit_button)
        hb.addWidget(self.reset_frame_button)
        hb.addStretch(1)
        self.busy_label = BusyLabel("Rendering…")   # beside the picture size, where the eye is
        self.busy_label.setVisible(False)
        hb.addWidget(self.busy_label)
        hb.addSpacing(8)
        self.frame_label = QtWidgets.QLabel(objectName="muted")
        hb.addWidget(self.frame_label)
        lv.addWidget(bar)
        self.left_stack = QtWidgets.QStackedWidget()
        self.live_slot = QtWidgets.QWidget(objectName="nativeSlot")
        self.live_slot.setMinimumSize(400, 300)
        self.preview = QtWidgets.QLabel("Press Render to preview the final image", alignment=QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(400, 300)
        self.preview.setStyleSheet(f"background: {TOKENS['bg']}; color: {TOKENS['muted']};")
        self.left_stack.addWidget(self.live_slot)
        self.left_stack.addWidget(self.preview)
        lv.addWidget(self.left_stack, 1)
        outer.addWidget(left, 3)
        self.frame = None        # RenderFrame on the viewport, set by attach_frame()
        self._region = None      # the frame's region before a viewport is attached

        side = QtWidgets.QWidget()
        side.setMaximumWidth(460)
        form = QtWidgets.QFormLayout(side)
        form.setContentsMargins(12, 12, 12, 12)
        self.renderer = QtWidgets.QComboBox()
        self.renderer.addItem("Ray tracing (AO + shadows)")
        self.renderer.addItem("OpenGL (fast preview)")
        self.renderer.addItem("Vector (SVG / PDF / EPS)")
        self.renderer.addItem("OSPRay path tracing (needs a VTK build with OSPRay)")
        self.renderer.model().item(3).setEnabled(False)
        form.addRow("Renderer", self.renderer)
        self.quality = QtWidgets.QComboBox()
        self.quality.addItems(QUALITIES)
        self.quality.setToolTip("Draft: half size, no AO or shadows (fast previews). Standard: balanced. "
                                "Publication: 16 rays/pixel, 8 AO rays per hit, shadows, SSAO, fine spheres. "
                                "Custom: the values below.")
        form.addRow("Quality", self.quality)
        self.samples = QtWidgets.QSpinBox(minimum=1, maximum=256, value=8, suffix=" rays/pixel")
        self.ao_samples = QtWidgets.QSpinBox(minimum=0, maximum=512, value=16, suffix=" AO rays")
        self.noise = QtWidgets.QDoubleSpinBox(minimum=0.0, maximum=0.1, singleStep=0.005, decimals=3, value=0.01,
                                              specialValueText="Off (every pixel takes all rays)")
        self.noise.setToolTip("Adaptive sampling: a pixel stops taking rays once its estimated error "
                              "(standard error, 0-1 color units) is below this; edges, shadows and AO keep all.")
        self.shadows = QtWidgets.QCheckBox("Shadows", checked=True)
        form.addRow("Anti-aliasing", self.samples)
        form.addRow("Ambient occl.", self.ao_samples)
        form.addRow("Noise threshold", self.noise)
        form.addRow(self.shadows)
        self.ao = QtWidgets.QCheckBox("Ambient occlusion", toolTip="OpenGL screen-space ambient occlusion (SSAO)")
        self.aa = QtWidgets.QCheckBox("Supersampling", checked=True, toolTip="OpenGL anti-aliasing")
        form.addRow(self.ao)
        form.addRow(self.aa)
        self.renderer.currentIndexChanged.connect(self._renderer_changed)
        form.addRow(QtWidgets.QLabel("Output size", objectName="section"))
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(list(SIZE_PRESETS))
        self.dpi = QtWidgets.QSpinBox(minimum=72, maximum=2400, value=600, suffix=" dpi")
        self.width = QtWidgets.QSpinBox(minimum=64, maximum=16384, value=2100)
        self.height = QtWidgets.QSpinBox(minimum=64, maximum=16384, value=1575)
        size = QtWidgets.QHBoxLayout(); size.addWidget(self.width); size.addWidget(QtWidgets.QLabel("×")); size.addWidget(self.height)
        form.addRow("Preset", self.preset)
        form.addRow("Resolution", self.dpi)
        form.addRow("Pixels", size)
        self.background = QtWidgets.QComboBox()
        self.background.addItems(["white", "transparent", "black", "viewport"])
        form.addRow("Background", self.background)
        form.addRow(QtWidgets.QLabel("Overlays", objectName="section"))
        self.text = QtWidgets.QLineEdit(placeholderText="Text label, e.g. t = [Time] ps   frame [Frame]")
        self.time_per_frame = QtWidgets.QDoubleSpinBox(minimum=0, maximum=1e9, value=1.0, decimals=4, suffix=" ps/frame")
        self.legend = QtWidgets.QCheckBox("Color legend", checked=True)
        self.type_legend = QtWidgets.QCheckBox("Particle type legend")
        self.scale_bar = QtWidgets.QCheckBox("Scale bar")
        self.scale_length = QtWidgets.QDoubleSpinBox(minimum=0, maximum=1e6, value=0, decimals=1, suffix=" Å",
                                                     specialValueText="auto")
        bar = QtWidgets.QHBoxLayout(); bar.addWidget(self.scale_bar); bar.addWidget(self.scale_length, 1)
        form.addRow("Text", self.text)
        form.addRow("[Time] =", self.time_per_frame)
        form.addRow(self.legend)
        form.addRow(self.type_legend)
        form.addRow(bar)
        self.overlays: list[dict] = []   # overlay elements: {name, params, enabled}
        self.overlay_list = QtWidgets.QListWidget()
        self.overlay_list.setMaximumHeight(90)
        self.overlay_list.itemChanged.connect(self._overlay_toggled)
        self.overlay_list.currentRowChanged.connect(self._show_overlay_form)
        self.overlay_menu = QtWidgets.QMenu(self)
        self.overlay_menu.aboutToShow.connect(self._fill_overlay_menu)
        add = QtWidgets.QToolButton(text="+ Overlay", popupMode=QtWidgets.QToolButton.InstantPopup)
        add.setMenu(self.overlay_menu)
        remove = QtWidgets.QPushButton("Remove")
        remove.clicked.connect(lambda: self.remove_overlay(self.overlay_list.currentRow()))
        row = QtWidgets.QHBoxLayout(); row.addWidget(add); row.addWidget(remove); row.addStretch()
        form.addRow(self.overlay_list)
        form.addRow(row)
        self.overlay_form_holder = QtWidgets.QVBoxLayout()
        form.addRow(self.overlay_form_holder)
        self._overlay_form = None
        form.addRow(QtWidgets.QLabel("Output", objectName="section"))
        out = QtWidgets.QHBoxLayout()
        self.output = QtWidgets.QLineEdit("render.png")
        browse = QtWidgets.QPushButton("…")
        browse.clicked.connect(self._browse)
        out.addWidget(self.output, 1); out.addWidget(browse)
        form.addRow("File", out)
        self.animation = QtWidgets.QCheckBox("Render animation",
                                             toolTip="All frames, at the timeline's fps and stride")
        form.addRow(self.animation)
        self.movie_format = QtWidgets.QComboBox()
        self.movie_format.addItems(list(MOVIE_FORMATS))
        self.movie_format.currentTextChanged.connect(self._movie_format_changed)
        form.addRow("Movie format", self.movie_format)
        self.movie_quality = QtWidgets.QComboBox()
        self.movie_quality.addItems(["Same as image", "Draft", "Standard", "Publication"])
        form.addRow("Movie quality", self.movie_quality)
        go = self.render_button = QtWidgets.QPushButton("Render")
        go.setObjectName("primary")
        go.clicked.connect(self._render_clicked)
        self.cancellable = False
        form.addRow(go)
        form.addRow(QtWidgets.QLabel("Python for this figure", objectName="section"))
        self.code = QtWidgets.QPlainTextEdit(readOnly=True)
        self.code.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.code.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.code.setMinimumHeight(220)
        form.addRow(self.code)
        copy = QtWidgets.QPushButton("Copy code")
        copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.code.toPlainText()))
        form.addRow(copy)
        # narrow column: rows put their label above the field, and combo boxes shrink to a short
        # elided text, so the form never scrolls sideways (the window's minimum width stays)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        for spin, px in ((self.time_per_frame, 100), (self.width, 70), (self.height, 70), (self.scale_length, 70)):
            spin.setMinimumWidth(px)                   # their wide ranges would ask for much more
        for combo in side.findChildren(QtWidgets.QComboBox):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
        scroll = QtWidgets.QScrollArea()
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setMaximumWidth(480)
        outer.addWidget(scroll, 2)

        for w in (self.width, self.height):
            w.valueChanged.connect(self._size_changed)
        self.preset.currentTextChanged.connect(self._apply_preset)
        self.dpi.valueChanged.connect(lambda _: self._apply_preset(self.preset.currentText()))
        for w in (self.ao, self.aa, self.legend, self.animation, self.shadows, self.type_legend, self.scale_bar):
            w.toggled.connect(self.update_code)
        self.renderer.currentIndexChanged.connect(self.update_code)
        for w in (self.width, self.height, self.dpi, self.time_per_frame, self.samples, self.ao_samples,
                  self.noise, self.scale_length):
            w.valueChanged.connect(self.update_code)
        for w in (self.text, self.output):
            w.textChanged.connect(self.update_code)
        self.background.currentTextChanged.connect(self.update_code)
        self._apply_preset(self.preset.currentText())
        self._renderer_changed()
        self._syncing_quality = False
        self.quality.currentTextChanged.connect(self._quality_changed)
        for w in (self.samples, self.ao_samples, self.noise):
            w.valueChanged.connect(self._quality_edited)
        for w in (self.shadows, self.ao, self.aa):
            w.toggled.connect(self._quality_edited)
        self.quality.currentTextChanged.connect(self.update_code)
        self.movie_quality.currentTextChanged.connect(self.update_code)
        self.quality.setCurrentText("Publication")
        self._quality_changed("Publication")

    # ---------------------------------------------------------------- frame on the 3D view
    def attach_frame(self, frame):
        """Use ``frame`` (a :class:`~ewald.gui.render_frame.RenderFrame` on the viewport)."""
        self.frame = frame
        frame.set_aspect(self.aspect)
        frame.set_region(self._region)
        frame.regionChanged.connect(self._frame_dragged)
        frame.aspectChanged.connect(self._frame_reshaped)
        self._show_frame_label()

    @property
    def aspect(self) -> float:
        return self.width.value() / max(self.height.value(), 1)

    @property
    def region(self):
        """The frame's (x, y, height) as set, or None for the largest frame that fits."""
        return self.frame.region if self.frame is not None else self._region

    def effective_region(self):
        if self.frame is not None:
            return self.frame.effective_region()
        from ..framing import as_region

        return as_region(self._region) if self._region is not None else None

    def set_region(self, region, write: bool = False):
        from ..framing import as_region

        self._region = None if region is None else as_region(region)
        if self.frame is not None:
            self.frame.set_region(self._region)
        self.update_code()
        if write:
            self.regionWritten.emit()

    def _frame_dragged(self, region):
        self._region = region
        self.update_code()
        self.regionWritten.emit()

    def _frame_reshaped(self, aspect, side):
        """Dragging a side of the frame: the picture gets that aspect (Custom size, the other side kept)."""
        w, h = self.width.value(), self.height.value()
        if self.preset.currentText() != "Custom (px)":
            self.preset.setCurrentText("Custom (px)")
        self._reshaping = True
        try:
            if side in ("e", "w"):
                self.width.setValue(max(64, round(h * aspect)))
            else:
                self.height.setValue(max(64, round(w / aspect)))
        finally:
            self._reshaping = False
        self._show_frame_label()

    def _size_changed(self, *_):
        if self.frame is not None and not getattr(self, "_reshaping", False):
            self.frame.set_aspect(self.aspect)
        self._show_frame_label()

    def _show_frame_label(self):
        self.frame_label.setText(f"{self.width.value()} × {self.height.value()} px")

    def _render_clicked(self):
        (self.cancelRequested if self.cancellable else self.renderRequested).emit()

    def set_busy(self, text: str = "", cancellable: bool = False):
        """While a picture renders, a spinner beside the picture size says what runs, and the Render
        button greys out, or turns into Cancel when the render can be stopped (the ray tracer)."""
        self.cancellable = bool(text) and cancellable
        self.render_button.setEnabled(not text or self.cancellable)
        self.render_button.setText("Cancel" if self.cancellable else text or "Render")
        self.render_button.setToolTip("Stop the ray trace" if self.cancellable else "")
        self.busy_label.setText(text)
        self.busy_label.setVisible(bool(text))

    def show_live(self, on: bool = True):
        """Show the 3D view with the frame (True) or the last rendered picture."""
        (self.live_button if on else self.result_button).setChecked(True)
        self.left_stack.setCurrentIndex(0 if on else 1)
        for b in (self.fit_button, self.reset_frame_button):
            b.setEnabled(on)
        if self.window is not None and hasattr(self.window, "_render_frame_visibility"):
            self.window._render_frame_visibility()

    def fit_frame(self):
        if self.frame is not None:
            self.frame.fit(getattr(self.window, "data", None))

    @property
    def raytrace(self) -> bool:
        return self.renderer.currentIndex() == 0

    # ---------------------------------------------------------------- quality presets
    def _quality_changed(self, name):
        from ..render import QUALITY_PRESETS

        preset = QUALITY_PRESETS.get(name.lower())
        if preset is None:  # Custom keeps whatever is in the fields
            return
        self._syncing_quality = True
        try:
            self.samples.setValue(preset["samples"])
            self.ao_samples.setValue(preset["ao_samples"])
            self.noise.setValue(preset["noise_threshold"])
            self.shadows.setChecked(preset["shadows"])
            self.ao.setChecked(preset["ambient_occlusion"])
            self.aa.setChecked(preset["antialiasing"])
        finally:
            self._syncing_quality = False

    def _quality_edited(self, *_):
        if not self._syncing_quality and self.quality.currentText() != "Custom":
            self.quality.blockSignals(True)
            self.quality.setCurrentText("Custom")
            self.quality.blockSignals(False)
            self.update_code()

    def quality_kwargs(self, movie: bool = False) -> dict:
        """render() keywords for the sampling settings: a preset name, or the explicit Custom values."""
        name = self.quality.currentText()
        if movie and self.movie_quality.currentText() != "Same as image":
            name = self.movie_quality.currentText()
        if name != "Custom":
            return {"quality": name.lower()}
        if self.renderer_name == "raytrace":
            return {"samples": self.samples.value(), "ao_samples": self.ao_samples.value(),
                    "noise_threshold": self.noise.value(), "shadows": self.shadows.isChecked()}
        if self.renderer_name == "opengl":
            return {"ambient_occlusion": self.ao.isChecked(), "antialiasing": self.aa.isChecked()}
        return {}

    @property
    def renderer_name(self) -> str:
        return RENDERERS[min(self.renderer.currentIndex(), len(RENDERERS) - 1)]

    def set_renderer(self, name: str):
        self.renderer.setCurrentIndex(RENDERERS.index(name) if name in RENDERERS else 0)

    def _renderer_changed(self, *_):
        for w in (self.samples, self.ao_samples, self.noise, self.shadows):
            w.setEnabled(self.raytrace)
        for w in (self.ao, self.aa):
            w.setEnabled(self.renderer_name == "opengl")
        out = self.output.text()
        root, ext = os.path.splitext(out)
        if self.renderer_name == "vector" and ext.lower() not in VECTOR_EXTS + MOVIE_EXTS:
            self.output.setText(root + ".svg")
        elif self.renderer_name != "vector" and ext.lower() in VECTOR_EXTS:
            self.output.setText(root + ".png")

    def _movie_format_changed(self, label):
        ext = MOVIE_FORMATS[label]
        root, old = os.path.splitext(self.output.text() or "movie")
        self.output.setText(root + ext)
        self.animation.setChecked(True)

    @property
    def is_movie(self) -> bool:
        ext = os.path.splitext(self.output.text())[1].lower()
        return self.animation.isChecked() or ext in MOVIE_EXTS

    def _apply_preset(self, name):
        spec = SIZE_PRESETS.get(name)
        custom = spec is None
        self.width.setEnabled(custom); self.height.setEnabled(custom); self.dpi.setEnabled(not custom and spec[0] != "px")
        if spec and spec[0] == "px":
            self.width.setValue(spec[1]); self.height.setValue(spec[2])
        elif spec:
            self.width.setValue(round(spec[0] * self.dpi.value())); self.height.setValue(round(spec[1] * self.dpi.value()))
        self.update_code()

    def _browse(self):
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(self, "Output", self.output.text(),
                                                             OUTPUT_FILTER)
        if path:
            from .file_dialogs import suffixed

            path = suffixed(path, chosen)
            self.output.setText(path)

    # ---------------------------------------------------------------- overlay elements
    def _fill_overlay_menu(self):
        from .. import plugins

        self.overlay_menu.clear()
        for name, p in plugins.overlays().items():
            act = self.overlay_menu.addAction(name, lambda n=name: self.add_overlay(n))
            act.setToolTip(p.description)
        self.overlay_menu.setToolTipsVisible(True)

    def add_overlay(self, name: str, params: dict | None = None):
        """Add an overlay element (built-in or plugin) drawn on every rendered figure."""
        from ..overlays import coerce_overlays, overlay_to_dict

        (el,) = coerce_overlays([{"name": name, "params": params or {}}])
        self.overlays.append(overlay_to_dict(el))
        self._refresh_overlays(len(self.overlays) - 1)

    def remove_overlay(self, k: int):
        if 0 <= k < len(self.overlays):
            self.overlays.pop(k)
            self._refresh_overlays(min(k, len(self.overlays) - 1))

    def _refresh_overlays(self, current: int = -1):
        self.overlay_list.blockSignals(True)
        self.overlay_list.clear()
        for ov in self.overlays:
            item = QtWidgets.QListWidgetItem(ov["name"])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if ov.get("enabled", True) else QtCore.Qt.Unchecked)
            self.overlay_list.addItem(item)
        self.overlay_list.blockSignals(False)
        self.overlay_list.setCurrentRow(current)
        self._show_overlay_form(current)
        self.update_code()

    def _overlay_toggled(self, item):
        k = self.overlay_list.row(item)
        self.overlays[k]["enabled"] = item.checkState() == QtCore.Qt.Checked
        self.update_code()

    def _show_overlay_form(self, k: int):
        from ..overlays import coerce_overlays, overlay_to_dict
        from .params import ParameterForm

        while self.overlay_form_holder.count():
            w = self.overlay_form_holder.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self._overlay_form = None
        if not 0 <= k < len(self.overlays):
            return
        try:
            (el,) = coerce_overlays([self.overlays[k]])
        except Exception as exc:  # plugin no longer installed, or its parameters changed
            self.overlay_form_holder.addWidget(QtWidgets.QLabel(f"{type(exc).__name__}: {exc}", wordWrap=True))
            return
        form = ParameterForm(el)
        self._overlay_form = form

        def apply():
            form.apply()
            self.overlays[k] = {**overlay_to_dict(el), "enabled": self.overlays[k].get("enabled", True)}
            self.update_code()

        button = QtWidgets.QPushButton("Apply overlay")
        button.clicked.connect(apply)
        self.overlay_form_holder.addWidget(form)
        self.overlay_form_holder.addWidget(button)

    def label_text(self, frame: int, data) -> str:
        from ..overlays import format_label

        return format_label(self.text.text(), frame, data, self.time_per_frame.value())

    def render_kwargs(self, movie: bool = False) -> dict:
        bg = self.background.currentText()
        kw = {"size": (self.width.value(), self.height.value()), "legend": self.legend.isChecked()}
        region = self.effective_region()
        if region is not None and tuple(region) != (0.0, 0.0, 1.0):
            kw["region"] = tuple(round(v, 4) for v in region)
        if self.type_legend.isChecked():
            kw["type_legend"] = True
        if self.scale_bar.isChecked():
            kw["scale_bar"] = self.scale_length.value() or True
        if self.overlays:
            kw["overlays"] = [dict(o) for o in self.overlays]
        if self.renderer_name != "opengl":
            kw["renderer"] = self.renderer_name
        kw.update(self.quality_kwargs(movie))
        if self.preset.currentText() != "Custom (px)" and SIZE_PRESETS.get(self.preset.currentText(), ("px",))[0] != "px":
            kw["dpi"] = self.dpi.value()
        if bg == "transparent":
            kw.update(background="white", transparent_background=True)
        elif bg == "viewport":
            kw["background"] = self.window.style.background
        else:
            kw["background"] = bg
        return kw

    def update_code(self, *_):
        self.code.setPlainText(figure_code(self.window, self))

    # ---------------------------------------------------------------- sessions
    def state(self) -> dict:
        """Everything on this page, for session files (GUI keys next to render() keywords)."""
        return {"renderer_name": self.renderer_name, "quality": self.quality.currentText(),
                "movie_quality": self.movie_quality.currentText(), "samples": self.samples.value(),
                "ao_samples": self.ao_samples.value(), "noise_threshold": self.noise.value(),
                "shadows": self.shadows.isChecked(),
                "ambient_occlusion": self.ao.isChecked(), "antialiasing": self.aa.isChecked(),
                "preset": self.preset.currentText(), "dpi": self.dpi.value(),
                "size": (self.width.value(), self.height.value()), "background_choice": self.background.currentText(),
                "text_template": self.text.text(), "time_per_frame": self.time_per_frame.value(),
                "legend": self.legend.isChecked(), "type_legend": self.type_legend.isChecked(),
                "scale_bar_on": self.scale_bar.isChecked(), "scale_bar_length": self.scale_length.value(),
                "output": self.output.text(), "animation": self.animation.isChecked(),
                "overlays": [dict(o) for o in self.overlays],
                "region": None if self.region is None else list(self.region)}

    def set_state(self, st: dict):
        if "renderer_name" in st:
            self.set_renderer(st["renderer_name"])
        if st.get("movie_quality"):
            self.movie_quality.setCurrentText(st["movie_quality"])
        for key, w in (("samples", self.samples), ("ao_samples", self.ao_samples), ("noise_threshold", self.noise),
                       ("dpi", self.dpi),
                       ("time_per_frame", self.time_per_frame), ("scale_bar_length", self.scale_length)):
            if key in st:
                w.setValue(st[key])
        for key, w in (("shadows", self.shadows), ("ambient_occlusion", self.ao), ("antialiasing", self.aa),
                       ("legend", self.legend), ("type_legend", self.type_legend), ("scale_bar_on", self.scale_bar),
                       ("animation", self.animation)):
            if key in st:
                w.setChecked(bool(st[key]))
        if st.get("preset") in SIZE_PRESETS:
            self.preset.setCurrentText(st["preset"])
        if "size" in st and self.preset.currentText() == "Custom (px)":
            self.width.setValue(int(st["size"][0])); self.height.setValue(int(st["size"][1]))
        if "background_choice" in st:
            self.background.setCurrentText(st["background_choice"])
        if "text_template" in st:
            self.text.setText(st["text_template"])
        if st.get("output"):
            self.output.setText(st["output"])
        if "overlays" in st:
            self.overlays = [dict(o) for o in st["overlays"] or []]
            self._refresh_overlays()
        if "region" in st:
            self.set_region(st["region"])
        if st.get("quality") in QUALITIES:  # last: a preset overwrites the fields, Custom keeps them
            self.quality.setCurrentText(st["quality"])
            self._quality_changed(st["quality"])
        self.update_code()


def figure_code(window, page) -> str:
    """Script reproducing the current figure with the public ewald API."""
    pipe = window.pipeline
    lines = ["from ewald import import_file", "from ewald.modifiers import *", "from ewald.render import render", ""]
    src = getattr(pipe, "source", None) if pipe else None
    if src is not None and hasattr(src, "path"):
        args = [repr(src.path)] + ([f"format={src.format!r}"] if src.format else []) + \
               ([f"join={src.join!r}"] if getattr(src, "join", "timestep") != "timestep" else []) + \
               [f"{k}={v!r}" for k, v in src.kwargs.items()]
        lines.append(f"pipe = import_file({', '.join(args)})")
    else:
        lines.append("pipe = ...  # in-memory source")
    for m in (pipe.modifiers if pipe else []):
        if not m.enabled:
            continue
        if hasattr(m, "function"):
            lines.append(f"pipe.modifiers.append({m.name})  # Python function from the console")
        else:
            lines.append(f"pipe.modifiers.append({m!r})")
    frame = getattr(window, "frame", 0)
    device = window.device
    lines.append(f"data = pipe.compute({frame}" + (f", device={device!r})" if device and device != "cpu" else ")"))
    kw = dict(window.style.scene_kwargs())
    kw.update(page.render_kwargs(movie=page.is_movie))
    from ..render import camera_state

    if getattr(window.viewport, "view_built", True):  # no camera yet while the 3D view is being built
        kw["camera"] = {k: (tuple(round(x, 4) for x in v) if isinstance(v, tuple) else round(v, 4)
                            if isinstance(v, float) else v)
                        for k, v in camera_state(window.viewport.plotter).items()}
    out = page.output.text() or "render.png"
    if page.is_movie and pipe is not None:
        lines[2] = "from ewald.render import render_animation"
        lines.pop()  # no single-frame compute
        stride = window.timeline.stride.value() if hasattr(window, "timeline") else 1
        kw["size"] = (kw["size"][0] // 2 * 2, kw["size"][1] // 2 * 2)
        if page.text.text():
            kw["text"] = page.text.text()
            kw["time_per_frame"] = page.time_per_frame.value()
        fps = window.timeline.fps.value() if hasattr(window, "timeline") else 10
        head = f"range(0, pipe.num_frames, {stride})"
        body = ",\n                 ".join(f"{k}={v!r}" for k, v in kw.items())
        dev = f", device={device!r}" if device and device != "cpu" else ""
        lines.append(f"render_animation(pipe, {out!r}, frames={head}, fps={fps}{dev},\n                 {body})")
        return "\n".join(lines) + "\n"
    text = page.label_text(frame, window.data)
    if text:
        kw["text"] = text
    body = ",\n       ".join(f"{k}={v!r}" for k, v in kw.items())
    lines.append(f"render(data, {out!r},\n       {body})")
    return "\n".join(lines) + "\n"
