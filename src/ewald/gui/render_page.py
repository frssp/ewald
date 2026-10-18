"""Render workspace: journal-size presets, preview, overlays, and 'Python for this figure'."""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from .theme import TOKENS

SIZE_PRESETS = {
    # label: (width inches, height inches) or (px, px) when 'px'
    "1-column (3.5 in, 4:3)": (3.5, 2.625),
    "2-column (7.2 in, 16:9)": (7.2, 4.05),
    "Square (3.5 in)": (3.5, 3.5),
    "Slide 16:9 (1920×1080)": ("px", 1920, 1080),
    "Custom (px)": None,
}


class RenderPage(QtWidgets.QWidget):
    renderRequested = QtCore.Signal()

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.preview = QtWidgets.QLabel("Press Render to preview the final image", alignment=QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(400, 300)
        self.preview.setStyleSheet(f"background: {TOKENS['bg']}; color: {TOKENS['muted']};")
        outer.addWidget(self.preview, 3)

        side = QtWidgets.QWidget()
        side.setMaximumWidth(460)
        form = QtWidgets.QFormLayout(side)
        form.setContentsMargins(12, 12, 12, 12)
        self.renderer = QtWidgets.QComboBox()
        self.renderer.addItem("OpenGL (VTK)")
        self.renderer.addItem("OSPRay path tracing (needs a VTK build with OSPRay)")
        self.renderer.model().item(1).setEnabled(_has_ospray())
        form.addRow("Renderer", self.renderer)
        self.ao = QtWidgets.QCheckBox("Ambient occlusion")
        self.aa = QtWidgets.QCheckBox("Supersampling anti-aliasing", checked=True)
        form.addRow(self.ao)
        form.addRow(self.aa)
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
        form.addRow("Text", self.text)
        form.addRow("[Time] =", self.time_per_frame)
        form.addRow(self.legend)
        form.addRow(QtWidgets.QLabel("Output", objectName="section"))
        out = QtWidgets.QHBoxLayout()
        self.output = QtWidgets.QLineEdit("render.png")
        browse = QtWidgets.QPushButton("…")
        browse.clicked.connect(self._browse)
        out.addWidget(self.output, 1); out.addWidget(browse)
        form.addRow("File", out)
        self.animation = QtWidgets.QCheckBox("Render animation (all frames, uses timeline fps/stride)")
        form.addRow(self.animation)
        go = QtWidgets.QPushButton("Render")
        go.setObjectName("primary")
        go.clicked.connect(self.renderRequested)
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
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setMaximumWidth(480)
        outer.addWidget(scroll, 2)

        self.preset.currentTextChanged.connect(self._apply_preset)
        self.dpi.valueChanged.connect(lambda _: self._apply_preset(self.preset.currentText()))
        for w in (self.ao, self.aa, self.legend, self.animation):
            w.toggled.connect(self.update_code)
        for w in (self.width, self.height, self.dpi, self.time_per_frame):
            w.valueChanged.connect(self.update_code)
        for w in (self.text, self.output):
            w.textChanged.connect(self.update_code)
        self.background.currentTextChanged.connect(self.update_code)
        self._apply_preset(self.preset.currentText())

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
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Output", self.output.text(),
                                                        "Images (*.png *.jpg *.tif);;Movies (*.mp4 *.gif)")
        if path:
            self.output.setText(path)

    def label_text(self, frame: int, data) -> str:
        t = self.text.text()
        if not t:
            return ""
        ts = data.attributes.get("Timestep", frame) if data is not None else frame
        return (t.replace("[Frame]", str(frame)).replace("[Timestep]", str(ts))
                 .replace("[Time]", f"{frame * self.time_per_frame.value():g}"))

    def render_kwargs(self) -> dict:
        bg = self.background.currentText()
        kw = {"size": (self.width.value(), self.height.value()), "ambient_occlusion": self.ao.isChecked(),
              "antialiasing": self.aa.isChecked(), "legend": self.legend.isChecked()}
        if bg == "transparent":
            kw.update(background="white", transparent_background=True)
        elif bg == "viewport":
            kw["background"] = self.window.style.background
        else:
            kw["background"] = bg
        return kw

    def update_code(self, *_):
        self.code.setPlainText(figure_code(self.window, self))


def _has_ospray() -> bool:
    try:
        import vtk

        return hasattr(vtk, "vtkOSPRayPass")
    except ImportError:
        return False


def figure_code(window, page) -> str:
    """Script reproducing the current figure with the public ewald API."""
    pipe = window.pipeline
    lines = ["from ewald import import_file", "from ewald.modifiers import *", "from ewald.render import render", ""]
    src = getattr(pipe, "source", None) if pipe else None
    if src is not None and hasattr(src, "path"):
        args = [repr(src.path)] + ([f"format={src.format!r}"] if src.format else []) + \
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
    kw.update(page.render_kwargs())
    from ..render import camera_state

    kw["camera"] = {k: (tuple(round(x, 4) for x in v) if isinstance(v, tuple) else round(v, 4) if isinstance(v, float) else v)
                    for k, v in camera_state(window.viewport.plotter).items()}
    text = page.label_text(frame, window.data)
    if text:
        kw["text"] = text
    out = page.output.text() or "render.png"
    body = ",\n       ".join(f"{k}={v!r}" for k, v in kw.items())
    lines.append(f"render(data, {out!r},\n       {body})")
    return "\n".join(lines) + "\n"
