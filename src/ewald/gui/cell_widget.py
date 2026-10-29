"""One notebook cell in the window: :class:`CellWidget` (gutter, editor, step header, outputs)
with its :class:`OutputArea`, result chips and mini plots."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..notebook import Cell, Notebook, step_call
from . import header
from .code_editor import CodeEditor, OutputText, mono_font
from .icons import icon
from .palette import native_text
from .theme import TOKENS


class OutputArea(QtWidgets.QWidget):
    """Printed text, values, errors and images under a code cell. An image ``show(pipe, ...)``
    drew is a live plot here (``live``): it follows the frame and is linked to the 3D view, under
    the cell or in the Plots panel (a :class:`PlotSlot`)."""

    MAX_IMAGE_HEIGHT = 360   # a figure or snapshot never takes the whole panel
    plotCreated = QtCore.Signal(object)   # a live PlotsPanel: the main window links it
    placeRequested = QtCore.Signal(int, bool)   # the k-th live plot moved: True = Plots panel

    def __init__(self, parent=None, live: bool = True):
        super().__init__(parent)
        self.setObjectName("cellOutput")
        self.setAttribute(QtCore.Qt.WA_StyledBackground)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.rule = OutputRule()   # the rule beside the outputs: click scrolls or folds, double-click hides
        self.rule.clicked.connect(self.click_rule)
        self.rule.doubleClicked.connect(self.toggle_hidden)
        row.addWidget(self.rule)
        column = QtWidgets.QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        row.addLayout(column, 1)
        self.hidden_note = QtWidgets.QPushButton(objectName="cellOutputHidden", flat=True)
        self.hidden_note.setCursor(QtCore.Qt.PointingHandCursor)
        self.hidden_note.clicked.connect(lambda: self.set_hidden(False))
        self.hidden_note.hide()
        column.addWidget(self.hidden_note, 0, QtCore.Qt.AlignLeft)
        self.lay = QtWidgets.QVBoxLayout()
        self.lay.setContentsMargins(0, 2, 0, 2)
        self.lay.setSpacing(4)
        column.addLayout(self.lay)
        self.scrolled = None    # long text outputs: None by their length, True / False set by hand
        self.hidden = False     # every output folded away (Jupyter's collapsed output)
        self.live = live
        self.key = ""        # the cell's id: names the Plots panel tabs of its plots
        self._images = []    # (label, full-size pixmap): rescaled when the cell's width changes
        self.panels = []     # (PlotsPanel, show() arguments) of the live plots
        self.slots = []      # PlotSlot of each live plot
        self._widgets = []   # one per output, so a running cell's outputs update in place

    def set_outputs(self, outputs):
        outputs = list(outputs)
        old = getattr(self, "outputs", None)
        if old is not None and len(old) == len(outputs) and all(a is b for a, b in zip(old, outputs)):
            return   # the same outputs (every frame of playback): rebuilding them made the cell flicker
        if old is not None and len(self._widgets) == len(old) and len(old) <= len(outputs) and \
                all(a is b or _same_widget(a, b) for a, b in zip(old, outputs)):
            # a running cell's text grew, a progress bar moved or an output was added: only those change
            for k, (a, b) in enumerate(zip(old, outputs)):
                if a is not b:
                    _update_widget(self._widgets[k], b)
            for out in outputs[len(old):]:
                self._add(out)
            self.outputs = outputs
            self._fit_images()
            self.setVisible(bool(outputs))
            self._update_rule()
            return
        self.outputs = outputs
        for slot in self.slots:
            slot.release()
        while self.lay.count():
            w = self.lay.takeAt(0).widget()
            if w is not None:
                w.hide()          # gone now, not at the next event loop pass (no stale copy drawn)
                w.deleteLater()
        self._images, self.panels, self.slots, self._widgets = [], [], [], []
        for out in outputs:
            self._add(out)
        self._fit_images()
        self.setVisible(bool(outputs))
        self._update_rule()

    def _add(self, out):
        if out.kind == "image" and out.view is not None and self.live:
            k = len(self.slots)
            w = PlotSlot(out.view, out.data, f"{self.key}-{k}")
            w.placeRequested.connect(lambda on, k=k: self.placeRequested.emit(k, on))
            self.slots.append(w)
            self.panels.append((w.panel, out.view))
            self.plotCreated.emit(w.panel)
        elif out.kind == "image":
            w = self._picture(out.data)
        elif out.kind == "rich":
            w = self._rich(out)
        else:
            w = _text_label(out)
        if isinstance(w, OutputText):
            w.set_scrolled(self.scrolled)
            w.fitted.connect(self._update_rule)
        self._widgets.append(w)
        self.lay.addWidget(w)
        w.setVisible(not self.hidden)
        self._update_rule()

    def _picture(self, data: bytes, fmt=None):
        pix = QtGui.QPixmap()
        pix.loadFromData(data, fmt)
        w = QtWidgets.QLabel(objectName="cellImage")
        w.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self._images.append((w, pix))
        return w

    def _rich(self, out):
        """The richest representation Qt can show: a progress bar, HTML, Markdown, LaTeX (drawn by
        matplotlib's mathtext), SVG, JPEG or JSON; the plain text if that fails."""
        from ..shell import PROGRESS_MIME

        mime = out.mime or {}
        if PROGRESS_MIME in mime:
            return ProgressOutput(mime[PROGRESS_MIME])
        html = mime.get("text/html")
        if isinstance(html, str) and "<script" not in html.lower():
            return _rich_label(html, QtCore.Qt.RichText)
        if isinstance(mime.get("text/markdown"), str):
            return _rich_label(mime["text/markdown"], QtCore.Qt.MarkdownText)
        if isinstance(mime.get("text/latex"), str):
            png = latex_png(mime["text/latex"], TOKENS["text"])
            if png:
                return self._picture(png, "PNG")
        if isinstance(mime.get("image/svg+xml"), str):
            png = svg_png(mime["image/svg+xml"])
            if png:
                return self._picture(png, "PNG")
        if isinstance(mime.get("image/jpeg"), bytes):
            return self._picture(mime["image/jpeg"], "JPG")
        if "application/json" in mime and "text/plain" not in mime:
            import json

            return _text_label(type(out)("value", json.dumps(mime["application/json"], indent=1, default=str)))
        return _text_label(out)

    def texts(self) -> list:
        return [w for w in self._widgets if isinstance(w, OutputText)]

    def can_scroll(self) -> bool:
        """A text output is taller than the scrolled box (so the rule can scroll or expand it)."""
        return any(w.can_scroll() for w in self.texts())

    def is_scrolled(self) -> bool:
        return any(w.is_scrolled() for w in self.texts())

    def click_rule(self):
        """A click on the rule, as on Jupyter's output prompt: a hidden output comes back, a long
        one scrolls in its box or shows every line, a short one folds away."""
        if self.hidden:
            self.set_hidden(False)
        elif self.can_scroll():
            self.toggle_scrolled()
        else:
            self.set_hidden(True)

    def toggle_scrolled(self):
        """Jupyter's Shift+O: the text outputs in a box of ``python.output_box_lines`` lines, or
        every line, whatever their length."""
        if self.can_scroll():
            self.set_scrolled(not self.is_scrolled())

    def set_scrolled(self, on):
        self.scrolled = on
        for w in self.texts():
            w.set_scrolled(on)
        self._update_rule()

    def toggle_hidden(self):
        """Jupyter's O (and a double-click on the rule): fold every output away, or show it again."""
        self.set_hidden(not self.hidden)

    def set_hidden(self, on: bool):
        self.hidden = bool(on) and bool(self._widgets)
        for w in self._widgets:
            w.setVisible(not self.hidden)
        self._update_rule()

    def _update_rule(self):
        texts, hidden = self.texts(), self.hidden and bool(self._widgets)
        lines = sum(w.line_count() for w in texts)
        if hidden:
            n = len(self._widgets)
            what = f"{lines} line" + ("s" if lines != 1 else "") if len(texts) == n else f"{n} output" + ("s" if n > 1 else "")
            self.hidden_note.setText(f"Output hidden ({what}), click to show")
        self.hidden_note.setVisible(hidden)
        state = ("hidden" if hidden else "" if not self._widgets else
                 ("scrolled" if self.is_scrolled() else "expanded") if self.can_scroll() else "short")
        self.rule.set_state(state, lines)

    def _fit_images(self):
        width = max(120, self.width() - self.rule.width() - 6)
        for label, pix in self._images:
            if pix.isNull():
                continue
            scale = min(1.0, width / pix.width(), self.MAX_IMAGE_HEIGHT / pix.height())
            shown = pix if scale >= 1.0 else pix.scaled(int(pix.width() * scale), int(pix.height() * scale),
                                                       QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            label.setPixmap(shown)
            label.setFixedHeight(shown.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._images and event.oldSize().width() != event.size().width():
            self._fit_images()


class OutputRule(QtWidgets.QWidget):
    """The thin rule left of a cell's outputs, Jupyter's output prompt: a click scrolls a long
    output or shows it whole (a short one folds away), a double-click hides every output."""

    clicked = QtCore.Signal()
    doubleClicked = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cellOutputRule")
        self.setAttribute(QtCore.Qt.WA_StyledBackground)
        self.setAttribute(QtCore.Qt.WA_Hover)
        self.setFixedWidth(8)    # 2 px drawn, the rest is room to hit it
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self._click = QtCore.QTimer(self, singleShot=True)   # a click waits to see if it was a double-click
        self._click.timeout.connect(self.clicked.emit)
        self.set_state("", 0)

    TIPS = {"scrolled": "Show all {} lines (Shift+O); double-click hides the output (O)",
            "expanded": "Scroll the output (Shift+O); double-click hides it (O)",
            "short": "Hide the output (O)",
            "hidden": "Show the output (O)"}

    def set_state(self, state: str, lines: int):
        """``scrolled``, ``expanded``, ``short`` (nothing to scroll), ``hidden`` or "" (no output)."""
        if self.property("state") != state:
            self.setProperty("state", state)
            self.style().unpolish(self)
            self.style().polish(self)
        self.setCursor(QtCore.Qt.PointingHandCursor if state else QtCore.Qt.ArrowCursor)
        self.setToolTip(self.TIPS.get(state, "").format(lines))

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.property("state"):
            if not self._click.isActive():
                self._click.start(QtWidgets.QApplication.doubleClickInterval())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.property("state"):
            self._click.stop()
            self.doubleClicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


def _text_label(out):
    return OutputText(str(out.data).rstrip("\n"), out.kind)


def _rich_label(text: str, fmt):
    """The output as rich text; an HTML table (a ``workflow_table``, a pandas frame) keeps its
    width and scrolls sideways instead of being cut off or wrapped."""
    w = QtWidgets.QLabel(objectName="cellRich")
    w.setTextFormat(fmt)
    w.setText(text)
    w.setOpenExternalLinks(True)
    w.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
    if fmt != QtCore.Qt.RichText or "<table" not in text.lower():
        w.setWordWrap(True)
        return w
    w.adjustSize()
    area = QtWidgets.QScrollArea(objectName="cellRichScroll")
    area.setFrameShape(QtWidgets.QFrame.NoFrame)
    area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    area.setWidget(w)
    area.setFixedHeight(w.sizeHint().height() + area.horizontalScrollBar().sizeHint().height())
    area.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
    return area


def _same_widget(a, b) -> bool:
    """Whether output ``b`` can be shown by updating the widget of ``a`` (text, values, a progress bar)."""
    from ..shell import PROGRESS_MIME

    if a.kind in ("text", "value", "error") and b.kind == a.kind:
        return True
    return a.kind == b.kind == "rich" and PROGRESS_MIME in (a.mime or {}) and PROGRESS_MIME in (b.mime or {})


def _update_widget(w, out):
    from ..shell import PROGRESS_MIME

    if isinstance(w, ProgressOutput):
        w.set_state(out.mime[PROGRESS_MIME])
    else:
        w.setText(str(out.data).rstrip("\n"))


class ProgressOutput(QtWidgets.QWidget):
    """A progress bar under a cell (:mod:`ewald.progress`): the bar, then what, how far and how fast."""

    STEPS = 1000

    def __init__(self, state: dict, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(2)
        self.bar = QtWidgets.QProgressBar(objectName="cellProgress", textVisible=False)
        self.label = QtWidgets.QLabel(objectName="cellText")
        self.label.setFont(mono_font())
        lay.addWidget(self.bar)
        lay.addWidget(self.label)
        self.set_state(state)

    def set_state(self, state: dict):
        total, n = state.get("total"), state.get("n") or 0
        if total:
            self.bar.setRange(0, self.STEPS)
            self.bar.setValue(int(self.STEPS * min(1.0, n / total)))
        else:
            self.bar.setRange(0, 0 if not state.get("closed") else 1)   # unknown length: busy
        self.label.setText(progress_label(state))
        self.state = state


def progress_label(state: dict) -> str:
    """``XDATCAR  45%  120/260 MB  [00:27 left, 5.1 MB/s]`` from a progress bar's values."""
    from tqdm.std import tqdm

    total, n, unit = state.get("total"), state.get("n") or 0, state.get("unit") or "it"
    scale = unit == "B"
    fmt = (lambda v: tqdm.format_sizeof(v, "B", 1024)) if scale else (lambda v: f"{v:g}")
    parts = [str(state.get("desc") or "").rstrip(": ")]
    rate, elapsed = state.get("rate"), state.get("elapsed") or 0
    if total:
        parts.append(f"{100 * n / total:3.0f}%")
        parts.append(f"{fmt(n)}/{fmt(total)}" + ("" if scale else f" {unit}"))
    else:
        parts.append(fmt(n) + ("" if scale else f" {unit}"))
    if state.get("closed"):
        parts.append(f"[{tqdm.format_interval(elapsed)}]")
    elif rate:
        left = f"{tqdm.format_interval((total - n) / rate)} left, " if total and n <= total else ""
        speed = tqdm.format_sizeof(rate, "B/s", 1024) if scale else f"{rate:.3g} {unit}/s"
        parts.append(f"[{left}{speed}]")
    return "  ".join(p for p in parts if p)


def latex_png(text: str, color: str = "black") -> bytes | None:
    """``$...$`` LaTeX drawn by matplotlib's mathtext (no TeX install needed), or None."""
    import io

    body = text.strip()
    for a, b in (("$$", "$$"), ("\\[", "\\]"), ("$", "$")):
        if body.startswith(a) and body.endswith(b) and len(body) > len(a) + len(b):
            body = body[len(a):-len(b)].strip()
            break
    body = body.replace("\\displaystyle", "").strip()   # IPython's Math(); mathtext has no display style
    try:
        from matplotlib import mathtext

        buf = io.BytesIO()
        mathtext.math_to_image(f"${body}$", buf, dpi=110, format="png", color=color)
        return buf.getvalue()
    except Exception:  # noqa: BLE001 - not mathtext (a whole LaTeX document): the text is shown
        return None


def svg_png(svg: str, max_width: int = 900) -> bytes | None:
    """An SVG output as PNG bytes (Qt's SVG renderer), or None."""
    try:
        from PySide6 import QtSvg
    except ImportError:
        return None
    r = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode("utf-8")))
    if not r.isValid():
        return None
    size = r.defaultSize()
    if size.width() <= 0 or size.height() <= 0:
        return None
    scale = min(2.0, max_width / size.width())
    img = QtGui.QImage(int(size.width() * scale), int(size.height() * scale), QtGui.QImage.Format_ARGB32)
    img.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(img)
    r.render(p)
    p.end()
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


class MiniPlot(QtWidgets.QWidget):
    """A small histogram (``bars``) or line (``line``) drawn in the data color."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(150, 44)
        self.bars = None
        self.line = None
        self.label = ""

    def set_bars(self, counts, label=""):
        self.bars, self.line, self.label = np.asarray(counts, float), None, label
        self.setToolTip(label)
        self.update()

    def set_line(self, x, y, label=""):
        self.line, self.bars, self.label = (np.asarray(x, float), np.asarray(y, float)), None, label
        self.setToolTip(label)
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height() - 2
        color = QtGui.QColor(TOKENS["data"])
        if self.bars is not None and len(self.bars) and self.bars.max() > 0:
            n = len(self.bars)
            bw = min(w / n, 16.0)
            scale = np.log1p(self.bars) / np.log1p(self.bars.max())  # log: one tall bar does not flatten the rest
            for i, s in enumerate(scale):
                bh = max(1.0, s * (h - 2))
                p.fillRect(QtCore.QRectF(i * bw + 1, h - bh, max(1.0, bw - 2), bh), color)
        elif self.line is not None and len(self.line[0]) > 1:
            x, y = self.line
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if len(x) > 1 and np.ptp(x) > 0:
                yr = np.ptp(y) or 1.0
                pts = [QtCore.QPointF((xi - x.min()) / np.ptp(x) * (w - 2) + 1, h - 1 - (yi - y.min()) / yr * (h - 4))
                       for xi, yi in zip(x, y)]
                p.setPen(QtGui.QPen(color, 1.6))
                p.drawPolyline(pts)
        p.setPen(QtGui.QColor(TOKENS["border"]))
        p.drawLine(0, h, w, h)
        p.end()


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    s = repr(v)
    return s if len(s) <= 24 else s[:22] + "…"


def result_chips(out: dict | None, data) -> list[str]:
    """Short texts for what a modifier did (from ``Modifier.last_output``)."""
    if not out:
        return []
    chips = []
    if out["count"] != out["count_before"]:
        d = out["count"] - out["count_before"]
        chips.append(f"{'+' if d > 0 else '−'}{abs(d):,} particles → {out['count']:,}")
    if out.get("selected") is not None:
        chips.append(f"{out['selected']:,} selected")
    chips += [f"+ {k}" for k in out["properties"] if k != "Selection"]
    for k in out["attributes"][:3]:
        v = None if data is None else data.attributes.get(k)
        chips.append(f"{k} = {_fmt(v)}" if np.isscalar(v) else k)
    chips += [f"table {k}" for k in out["tables"]]
    return chips


def result_plot(out: dict | None, data):
    """('bars', counts, label) for a new numeric property, ('line', x, y, label) for a new table, or None."""
    if not out or data is None:
        return None
    for k in out["properties"]:
        a = data.particles.get(k)
        if a is not None and a.ndim == 1 and a.dtype.kind in "iuf" and k != "Selection" and len(a):
            a = a[np.isfinite(a)] if a.dtype.kind == "f" else a
            if not len(a):
                continue
            if a.dtype.kind in "iu" and a.max() - min(a.min(), 0) <= 40:
                lo = min(int(a.min()), 0)  # counts from 0, so a single value is one bar among its peers
                counts = np.bincount((a - lo).astype(int))
                label = f"{k}: {a.min()} … {a.max()}"
            else:
                counts, edges = np.histogram(a, bins=24)
                label = f"{k}: {edges[0]:.3g} … {edges[-1]:.3g}"
            return ("bars", counts, label)
    for k in out["tables"]:
        t = data.tables.get(k)
        cols = [v for v in (t or {}).values() if isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype.kind in "iuf"]
        if len(cols) >= 2:
            names = [n for n, v in t.items() if v is cols[0] or v is cols[1]]
            return ("line", cols[0], cols[1], f"{k}: {' vs '.join(names[::-1])}")
    return None


# ---------------------------------------------------------------- cells
CELL_MIME = "application/x-ewald-cell"
GUTTER_WIDTH = 26   # run button, counter, grip and fold chevron


class _DragHandle(QtWidgets.QLabel):
    """The dotted grip in a cell's gutter: drag it to move the cell (the notebook view drops it)."""

    def __init__(self, cell_id: str, parent=None):
        super().__init__(parent, objectName="cellDrag")
        self._paint_grip("muted")
        self.cell_id = cell_id
        self.setToolTip("Drag to move the cell")
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._start = None

    def _paint_grip(self, token: str):
        from .icons import _pixmap

        ratio = self.devicePixelRatioF() or 1.0
        pix = _pixmap("grip", TOKENS[token], round(header.ICON * ratio))
        pix.setDevicePixelRatio(ratio)
        self.setPixmap(pix)

    def enterEvent(self, event):
        self._paint_grip("accent")
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._paint_grip("muted")
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._start = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._start is None or (event.position().toPoint() - self._start).manhattanLength() < \
                QtWidgets.QApplication.startDragDistance():
            return
        self._start = None
        mime = QtCore.QMimeData()
        mime.setData(CELL_MIME, self.cell_id.encode())
        drag = QtGui.QDrag(self)
        drag.setMimeData(mime)
        cell = self.parent()
        while cell is not None and not isinstance(cell, CellWidget):
            cell = cell.parent()
        if cell is not None:
            pix = cell.grab()
            drag.setPixmap(pix.scaledToWidth(min(pix.width(), 360), QtCore.Qt.SmoothTransformation))
        drag.exec(QtCore.Qt.MoveAction)


LIVE_PLOT_HEIGHT = (220, 320)   # px of a live plot under a cell


def live_plot():
    """The live plot of a view cell or a ``show(pipe, ...)`` output."""
    from .plots import PlotsPanel

    p = PlotsPanel()
    p.setMinimumHeight(LIVE_PLOT_HEIGHT[0])
    p.setMaximumHeight(LIVE_PLOT_HEIGHT[1])
    p.compact()
    return p


def _num_frames(pipeline):
    try:
        return int(pipeline.num_frames) if pipeline is not None else None
    except Exception:  # noqa: BLE001 - a source that cannot tell: one frame
        return None


class PlotCard(QtWidgets.QFrame):
    """What stays under a cell when its plot is in the Plots panel: a thumbnail of the plot and
    a link that brings the panel's tab to the front."""

    opened = QtCore.Signal()
    THUMB_HEIGHT = 72

    def __init__(self, title: str, png: bytes | None = None, parent=None):
        super().__init__(parent, objectName="plotCard")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip("Shown in the Plots panel (it changes with the frame or is linked to the 3D view). "
                        "Click to open it there; right-click to show it under the cell")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(10)
        self.thumb = QtWidgets.QLabel(objectName="cellImage")
        self.png = None
        self.set_png(png)
        lay.addWidget(self.thumb)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(2)
        col.addStretch()
        col.addWidget(QtWidgets.QLabel(title, objectName="plotCardTitle"))
        self.link = QtWidgets.QPushButton("Open in Plots panel", objectName="plotCardLink", flat=True)
        self.link.setCursor(QtCore.Qt.PointingHandCursor)
        self.link.clicked.connect(self.opened.emit)
        col.addWidget(self.link, 0, QtCore.Qt.AlignLeft)
        col.addStretch()
        lay.addLayout(col)
        lay.addStretch()

    def set_png(self, png: bytes | None):
        """The thumbnail: the image the plot's ``show()`` drew when the cell ran (none: no thumbnail)."""
        if png is self.png:
            return
        self.png = png
        pix = QtGui.QPixmap()
        if png and pix.loadFromData(png, "PNG") and not pix.isNull():
            self.thumb.setPixmap(pix.scaledToHeight(self.THUMB_HEIGHT, QtCore.Qt.SmoothTransformation))
            self.thumb.show()
        else:
            self.thumb.hide()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.opened.emit()
        super().mouseReleaseEvent(event)


class PlotSlot(QtWidgets.QWidget):
    """The place of one live plot in a cell (a view cell's or a ``show(pipe, ...)`` output's).
    The plot sits here, under the cell, or in the Plots panel (:func:`ewald.views.in_panel`:
    plots that change with the frame or are linked to the 3D view, or ``panel=True``), and then
    a :class:`PlotCard` takes its place. :meth:`place` decides once data is there; the right-click
    menu of either asks the cell to move it (``placeRequested``, written into the code)."""

    placeRequested = QtCore.Signal(bool)   # True: to the Plots panel, False: under the cell

    def __init__(self, args: dict, png: bytes | None = None, view_id: str = "", parent=None,
                 panel=None, title: str | None = None):
        super().__init__(parent, objectName="plotSlot")
        self.args = dict(args or {})
        self.view_id = "plot:cell-" + (view_id or f"{id(self):x}")
        self.title = title or self.args.get("histogram") or self.args.get("table") or "Plot"
        self.docked = None             # None until placed: True = in the Plots panel
        self.host = None               # the main window holding it there
        self._pending = False          # belongs in the Plots panel, but no window yet: placed again later
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.panel = live_plot() if panel is None else panel   # a FigureView for a step's figure
        self.panel.extra_menu = self._menu_entries
        self.panel.hide()
        lay.addWidget(self.panel)
        self.card = PlotCard(self.title, png)
        self.card.opened.connect(self.open)
        self.card.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.card.customContextMenuRequested.connect(self._card_menu)
        self.card.hide()
        lay.addWidget(self.card)

    def _menu_entries(self):
        if self.docked:
            return [("Show under the cell", lambda: self.placeRequested.emit(False))]
        return [("Show in Plots panel", lambda: self.placeRequested.emit(True))]

    def _card_menu(self, pos):
        m = QtWidgets.QMenu(self.card)
        m.addAction("Open in Plots panel", self.open)
        m.addAction("Show under the cell", lambda: self.placeRequested.emit(False))
        m.exec(self.card.mapToGlobal(pos))

    def decide(self, data, pipeline=None, force: bool = False):
        """Place the plot by :func:`ewald.views.in_panel` (once; ``force``: again)."""
        if self.docked is not None and not force and not self._pending:
            return
        if data is None and self.args.get("panel") is None:
            return   # no frame yet (it comes next): placed then, so the plot does not jump
        from ..views import in_panel

        try:
            docked = in_panel(data, self.args, _num_frames(pipeline))
        except Exception:  # noqa: BLE001 - a plot that cannot be judged stays under the cell
            docked = False
        self.place(docked, show=self.docked is None or self._pending or force)

    def place(self, docked: bool, show: bool = True):
        """Show the plot under the cell (False) or in the Plots panel (True; ``show`` brings its tab
        to the front). Without a main window it stays under the cell."""
        from .window_plugins import plugin_host

        host = plugin_host(self)
        host = host if host is not None and hasattr(host, "dock_plot") else None
        self._pending = docked and host is None
        if self._pending:
            docked = False
        if docked == self.docked:
            return
        self.docked = docked
        if docked:
            self.layout().removeWidget(self.panel)
            self.panel.setMinimumHeight(0)          # a tab of the panel: as tall as the panel
            self.panel.setMaximumHeight(16777215)   # QWIDGETSIZE_MAX
            self.host = host
            host.dock_plot(self, show=show)
            self.card.show()
        else:
            if self.host is not None:
                self.host.undock_plot(self)
                self.host = None
            self.panel.setMinimumHeight(LIVE_PLOT_HEIGHT[0])
            self.panel.setMaximumHeight(LIVE_PLOT_HEIGHT[1])
            self.layout().insertWidget(0, self.panel)
            self.panel.show()
            self.card.hide()

    def open(self):
        """Bring the plot's tab in the Plots panel to the front (again, after it was closed)."""
        if self.docked and self.host is not None:
            self.host.dock_plot(self, show=True)

    def release(self):
        """The cell drops this plot (it ran again or is gone): out of the Plots panel too."""
        if self.docked and self.host is not None:
            self.host.undock_plot(self)
            self.host = None
            self.panel.deleteLater()


class FigureView(QtWidgets.QLabel):
    """A figure a Python step drew (:class:`ewald.pipeline.StepFigure`), scaled to fit: the
    current frame's image of an each-frame step, or the one image of an all-frames step."""

    def __init__(self, parent=None):
        super().__init__(parent, objectName="cellImage")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setMinimumSize(120, 90)
        self.png = None
        self._pix = None
        self.extra_menu = None   # () -> [(label, callable)]: where the figure is shown

    def set_png(self, png: bytes | None):
        if png is self.png:
            return   # the same frame's figure (or an all-frames step's one figure): nothing to redraw
        self.png = png
        pix = QtGui.QPixmap()
        self._pix = pix if png and pix.loadFromData(png, "PNG") and not pix.isNull() else None
        self._fit()

    def _fit(self):
        if self._pix is None:
            self.clear()
            return
        scale = min(1.5, self.width() / self._pix.width(), self.height() / self._pix.height())
        self.setPixmap(self._pix.scaled(max(1, int(self._pix.width() * scale)), max(1, int(self._pix.height() * scale)),
                                        QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def contextMenuEvent(self, event):
        entries = self.extra_menu() if self.extra_menu else []
        if entries:
            m = QtWidgets.QMenu(self)
            for label, fn in entries:
                m.addAction(label, fn)
            m.exec(event.globalPos())


def step_figures(cell: Cell, data) -> dict:
    """``{attribute key: (StepFigure, each frame?)}`` of the figures the Python steps ``cell``
    added drew at this frame (``data.attributes["Figure: <step>"]``)."""
    from ..pipeline import AllFramesModifier, PythonModifier

    if data is None:
        return {}
    found = {}
    for m in cell.added:
        if not isinstance(m, PythonModifier):
            continue
        prefix = f"Figure: {m.name}"
        for k, v in data.attributes.items():
            if (k == prefix or k.startswith(prefix + " (")) and hasattr(v, "png"):
                found[k] = (v, not isinstance(m, AllFramesModifier))
    return found


def apply_view_args(p, args: dict, data, pipeline=None, applied=None):
    """Give plot ``p`` the current frame and select what ``show(...)``'s ``args`` name; the
    range (``xlim``/``ylim``) only when it differs from ``applied``, the range applied last.
    Returns the range applied."""
    p.pipeline = pipeline
    p.set_data(data)
    if args.get("histogram"):
        p.hist.setCurrentText(args["histogram"])
    elif args.get("table"):
        p.hist.setCurrentText("")
        p.table.setCurrentText(args["table"])
    if args.get("y"):
        p.column.setCurrentText(args["y"])
    view = tuple(None if args.get(k) is None else tuple(args[k]) for k in ("xlim", "ylim"))
    if view != p.view_limits() and view != applied:
        if view == (None, None):
            p.reset_view(emit=False)
        else:
            p.set_view(*view, emit=False)
    return view


def elapsed_text(seconds: float) -> str:
    """A running cell's time in at most four characters, for the counter under its run button
    (the toolbar chip has tenths)."""
    s = max(0.0, seconds)
    if s < 600:
        return f"{int(s)}s"
    if s < 3600:
        return f"{int(s // 60)}m"
    return f"{int(s // 3600)}h"


class _RunRing(QtWidgets.QWidget):
    """A turning accent arc around a running cell's stop button (painted, so no font glyph)."""

    PERIOD = 1.2   # seconds per turn

    def __init__(self, button):
        super().__init__(button)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setGeometry(button.rect())
        self.hide()

    def paintEvent(self, event):
        import time

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = QtCore.QRectF(self.rect()).adjusted(2, 2, -2, -2)
        track = QtGui.QColor(TOKENS["accent"])
        track.setAlphaF(0.22)
        p.setPen(QtGui.QPen(track, 2))
        p.drawEllipse(rect)
        pen = QtGui.QPen(QtGui.QColor(TOKENS["accent"]), 2)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)
        turn = (time.monotonic() % self.PERIOD) / self.PERIOD
        p.drawArc(rect, int(-turn * 360 * 16), 100 * 16)   # Qt angles: 1/16 degree, counterclockwise
        p.end()


class CellWidget(QtWidgets.QFrame):
    """One cell: gutter (run button, counter) and the code, markdown or step body."""

    runRequested = QtCore.Signal(object, bool)   # cell, advance
    selected = QtCore.Signal(object)
    edited = QtCore.Signal(object)               # a step's checkbox or a view's plot rewrote the cell
    # cell, 'up' | 'down' | 'delete' | 'code' | 'below' | 'clear' | 'kind:code' | 'kind:markdown' |
    # 'above' | 'after' (insert a code cell) | 'ask' | 'frames:...'
    actionRequested = QtCore.Signal(object, str)
    plotCreated = QtCore.Signal(object)          # a live plot of a show(pipe, ...) output

    def __init__(self, notebook: Notebook, cell: Cell, parent=None):
        super().__init__(parent)
        self.notebook = notebook
        self.cell = cell
        self.editor = None
        self.plot_panel = None
        self.built_for = _shape(cell)
        self.setObjectName("cell")
        self.setProperty("kind", "step" if cell.is_step else "view" if cell.is_view else cell.kind)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)   # Esc from the editor lands here (command keys)
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(2, 5, 8, 6)
        outer.setSpacing(4)
        self.gutter = QtWidgets.QWidget(objectName="cellGutter")
        self.gutter.setFixedWidth(GUTTER_WIDTH)
        gutter = QtWidgets.QVBoxLayout(self.gutter)
        gutter.setContentsMargins(0, 0, 0, 0)
        gutter.setSpacing(1)
        self.run_button = header.button("play", native_text("Run cell (Ctrl+Enter)"), name="cellRun")
        self.run_button.setProperty("ewald_icon_color", "accent")
        self.run_button.setIcon(icon("play", TOKENS["accent"]))
        self.run_button.clicked.connect(self._run_clicked)
        self.ring = _RunRing(self.run_button)
        self.started = None   # time.monotonic() when this cell started running, while it runs
        self.counter = QtWidgets.QLabel(objectName="cellCounter")
        self.counter.setFont(mono_font())
        self.counter.setAlignment(QtCore.Qt.AlignCenter)
        self.fold_button = header.button("chevron_down", "Collapse or expand the cell", name="cellFold")
        self.fold_button.clicked.connect(lambda: self.set_collapsed(not self.cell.collapsed))
        self.drag_handle = _DragHandle(cell.id)
        # the gutter holds only what is always there, so a one-line cell is one line tall; the
        # grip and the fold chevron sit with the actions menu at the top right
        gutter.addWidget(self.run_button, 0, QtCore.Qt.AlignHCenter)
        gutter.addWidget(self.counter, 0, QtCore.Qt.AlignHCenter)
        gutter.addStretch()
        for w in (self.drag_handle, self.fold_button):   # shown on hover and on the current cell only
            policy = w.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            w.setSizePolicy(policy)
        outer.addWidget(self.gutter)
        right = QtWidgets.QVBoxLayout()
        self.content = QtWidgets.QWidget(objectName="cellContent")
        self.body = QtWidgets.QVBoxLayout(self.content)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(4)
        self.folded_label = QtWidgets.QLabel(objectName="cellFolded")
        self.folded_label.setToolTip("Collapsed: click the chevron or double-click to expand")
        right.addWidget(self.content)
        right.addWidget(self.folded_label)
        outer.addLayout(right, 1)
        self.tools = QtWidgets.QWidget(objectName="cellTools")
        tools = QtWidgets.QHBoxLayout(self.tools)
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(0)
        tools.addWidget(self.drag_handle, 0, QtCore.Qt.AlignVCenter)
        tools.addWidget(self.fold_button)
        tools.addWidget(self._menu_button())
        outer.addWidget(self.tools, 0, QtCore.Qt.AlignTop)
        if cell.kind == "markdown":
            self.run_button.setVisible(False)
            self.counter.setVisible(False)
            self._build_markdown()
        elif cell.is_step:
            self._build_step()
        elif cell.is_view:
            self._build_view()
        else:
            self._build_code()
        self.update_state()
        self.set_collapsed(cell.collapsed)
        self._hovered = False
        self._show_controls()

    def _show_controls(self):
        """Drag grip and fold chevron appear on hover and on the current cell, like VS Code's
        cell toolbar; the run button and the counter stay."""
        on = self._hovered or bool(self.property("current"))
        self.drag_handle.setVisible(on)
        self.fold_button.setVisible(on or self.cell.collapsed)

    def enterEvent(self, event):
        self._hovered = True
        self._show_controls()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._show_controls()
        super().leaveEvent(event)

    def set_collapsed(self, on: bool):
        """Fold the cell to one line (its title, step name or first line of code), or unfold it."""
        self.cell.collapsed = bool(on)
        self.content.setVisible(not on)
        self.folded_label.setVisible(on)
        if hasattr(self, "_hovered"):
            self._show_controls()
        header.set_icon(self.fold_button, "chevron_right" if on else "chevron_down")
        if on:
            first = next((ln for ln in self.cell.source.splitlines() if ln.strip()), "")
            if self.cell.kind == "markdown":
                first = first.lstrip("# ")
            text = self.cell.title or (self.cell.modifier.name if self.cell.modifier is not None else first)
            n = len(self.cell.source.splitlines())
            self.folded_label.setText(f"{text}   ·  {n} line{'s' if n != 1 else ''}" if n > 1 else text)

    # ----------------------------------------------------------- bodies
    def _menu_button(self):
        b = header.button("more", "Cell actions", name="cellMenu")
        b.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        m = QtWidgets.QMenu(b)
        m.setToolTipsVisible(True)
        act = lambda name: (lambda: self.actionRequested.emit(self.cell, name))  # noqa: E731
        code = self.cell.kind == "code"
        if code:
            m.addAction(native_text("Run\tCtrl+Enter"), lambda: self.runRequested.emit(self.cell, False))
            m.addAction("Run this and below", act("below"))
        else:
            m.addAction("Edit", self.edit_markdown)
        m.addSeparator()
        kind = m.addMenu("Cell type")
        group = QtGui.QActionGroup(kind)
        for key, label, keys in (("code", "Code", "Y"), ("markdown", "Text (Markdown)", "M")):
            a = kind.addAction(f"{label}\t{keys}", act("kind:" + key))
            a.setCheckable(True)
            a.setChecked(self.cell.kind == key)
            group.addAction(a)
        if self.cell.is_step or self.cell.is_view:
            m.addAction("Show as code cell", act("code"))
        if code and not (self.cell.is_step or self.cell.is_view):
            self._frames_menu(m.addMenu("Runs"))
        if self.cell.is_view:
            m.addAction("Save plot…", lambda: self.plot_panel.save_plot())
        if code:
            m.addAction("Clear output", act("clear"))
        m.addAction("Ask AI about this cell", act("ask")).setToolTip(
            "Open the AI chat with this cell's code in the question")
        m.addSeparator()
        m.addAction("Insert cell above\tA", act("above"))
        m.addAction("Insert cell below\tB", act("after"))
        m.addAction(native_text("Move up\tCtrl+Shift+Up"), act("up"))
        m.addAction(native_text("Move down\tCtrl+Shift+Down"), act("down"))
        m.addSeparator()
        m.addAction("Cut\tX", act("cut"))
        m.addAction("Copy\tC", act("copy"))
        m.addAction("Paste below\tV", act("paste:below"))
        m.addAction("Merge with the cell below\tShift+M", act("merge"))
        m.addSeparator()   # the destructive entry apart from the rest
        m.addAction("Delete\tD, D", act("delete"))
        self._plugin_actions = []
        m.aboutToShow.connect(lambda: self._plugin_menu_items(m))
        b.setMenu(m)
        return b

    def _plugin_menu_items(self, menu):
        """The "Notebook cell" entries of @menu_item plugins, fresh each time the menu opens."""
        for a in self._plugin_actions:
            menu.removeAction(a)
            a.deleteLater()
        from .window_plugins import plugin_host

        host = plugin_host(self)
        sep = menu.addSeparator()
        added = host.add_plugin_items(menu, "Notebook cell", target=self.cell) if host else []
        if added:
            menu.removeAction(added.pop())     # its closing separator: these end the menu
            self._plugin_actions = [sep, *added]
        else:
            menu.removeAction(sep)
            self._plugin_actions = []

    FRAMES = {"": ("Once", "Runs when you run the cell"),
              "each": ("Each frame", "The cell's code is a step: it runs at every frame with `frame` and `data`"),
              "all": ("All frames", "The cell's code is a step: it runs once with `frames` (every frame) and "
                                    "`data`, and its result is kept for every frame")}

    def _frames_menu(self, menu):
        """When the cell's code runs: once, or as a step at each frame or once over all frames."""
        group = QtGui.QActionGroup(menu)
        for key, (label, tip) in self.FRAMES.items():
            a = menu.addAction(label, lambda k=key: self.actionRequested.emit(self.cell, "frames:" + k))
            a.setCheckable(True)
            a.setChecked(self.cell.frames == key)
            a.setToolTip(tip)
            group.addAction(a)
        menu.setToolTipsVisible(True)
        return menu

    def _frames_badge(self):
        """The cell's ``[each frame]`` / ``[all frames]`` tag; a click changes it."""
        label, tip = self.FRAMES[self.cell.frames]
        b = QtWidgets.QToolButton(text=label, objectName="cellFrames", toolTip=tip)
        b.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        b.setMenu(self._frames_menu(QtWidgets.QMenu(b)))
        return b

    def _build_code(self):
        tagged = self.cell.frames in ("each", "all")
        if self.cell.title or tagged:
            row = QtWidgets.QHBoxLayout()
            if self.cell.title:
                row.addWidget(QtWidgets.QLabel(self.cell.title, objectName="cellTitle"))
            if tagged:
                self.frames_badge = self._frames_badge()
                row.addWidget(self.frames_badge)
            row.addStretch()
            self.body.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.editor = CodeEditor(self.cell.source)
        self.editor.escaped.connect(self.leave_editor)
        self.editor.textChanged.connect(self._source_edited)
        self.editor.runRequested.connect(lambda adv: self.runRequested.emit(self.cell, adv))
        self.editor.focused.connect(lambda: self.selected.emit(self.cell))
        row.addWidget(self.editor, 1)
        self.body.addLayout(row)
        self.outputs = OutputArea()
        self.outputs.key = self.cell.id
        self.outputs.plotCreated.connect(self.plotCreated.emit)
        self.outputs.placeRequested.connect(self._place_output)
        self.body.addWidget(self.outputs)
        self.outputs.set_outputs(self.cell.outputs)

    def panels(self) -> list:
        """The live plots in this cell: a view cell's, and those of its ``show(pipe, ...)`` outputs."""
        live = [p for p, _ in self.outputs.panels] if isinstance(self.outputs, OutputArea) else []
        return ([self.plot_panel] if self.plot_panel is not None else []) + live

    def set_live_data(self, data, pipeline=None, force: bool = False):
        """The current frame for the live plots of ``show(pipe, ...)`` outputs (``force``: a new
        frame; else only plots that have none yet)."""
        self._update_figures(data)
        if not isinstance(self.outputs, OutputArea):
            return
        for slot in self.outputs.slots:
            slot.decide(data, pipeline)
        for p, args in self.outputs.panels:
            if force or p.data is None:
                p._applied = apply_view_args(p, args, data, pipeline, getattr(p, "_applied", None))

    def _update_figures(self, data):
        """The figures this cell's Python steps drew at this frame: an each-frame step's in the
        Plots panel (a thumbnail here), following the frame; an all-frames step's under the cell."""
        found = step_figures(self.cell, data) if self.cell.kind == "code" else {}
        slots = self.__dict__.setdefault("_figure_slots", {})
        if not found and not slots:
            return
        for key in [k for k in slots if k not in found]:
            slot = slots.pop(key)
            slot.release()
            slot.deleteLater()
        for key, (fig, each) in found.items():
            slot = slots.get(key)
            if slot is None:
                view_id = f"{self.cell.id}-" + "".join(c if c.isalnum() else "-" for c in key)
                slot = PlotSlot({}, fig.png, view_id, panel=FigureView(), title=key[len("Figure: "):])
                slot.placeRequested.connect(slot.place)   # the code says when it runs, not where: kept here
                self.body.addWidget(slot)
                slots[key] = slot
                slot.place(each)
            elif slot._pending:
                slot.place(each)
            slot.panel.set_png(fig.png)
            if slot.docked:
                slot.card.set_png(fig.png)

    def _source_edited(self):
        text = self.editor.toPlainText()
        if text != self.cell.source:  # not when the widget only shows the source
            self.cell.source = text
            self.notebook.dirty = True

    def _build_markdown(self):
        self.view = QtWidgets.QLabel(objectName="cellMarkdown")
        self.view.setTextFormat(QtCore.Qt.MarkdownText)
        self.view.setWordWrap(True)
        self.view.setText(self.cell.source or "*Empty text cell; double-click to edit*")
        self.view.setToolTip(native_text("Double-click to edit; Ctrl+Enter shows the text"))
        self.editor = CodeEditor(self.cell.source)
        self.editor.escaped.connect(self.leave_editor)
        self.editor.highlighter.setDocument(None)
        self.editor.setVisible(False)
        self.editor.textChanged.connect(self._source_edited)
        self.editor.runRequested.connect(lambda adv: self._render_markdown())
        row = QtWidgets.QHBoxLayout()
        col = QtWidgets.QVBoxLayout()
        col.addWidget(self.view)
        col.addWidget(self.editor)
        row.addLayout(col, 1)
        self.body.addLayout(row)
        self.outputs = None

    def mouseDoubleClickEvent(self, event):
        if self.cell.collapsed:
            self.set_collapsed(False)
        elif self.cell.kind == "markdown":
            self.edit_markdown()
        super().mouseDoubleClickEvent(event)

    def edit_markdown(self):
        self.view.setVisible(False)
        self.editor.setVisible(True)
        self.editor.setFocus()

    def leave_editor(self):
        """Esc in the editor: the cell keeps the keyboard (command mode: Y, M, A, B, DD, X, C, V,
        Z, Shift+arrows, Enter), as in Jupyter."""
        if self.cell.kind == "markdown":
            self._render_markdown()
        self.selected.emit(self.cell)
        self.setFocus()

    # Jupyter's command mode (the cell has the keyboard, not its editor): key -> NotebookView action
    COMMAND_KEYS = {QtCore.Qt.Key_Y: "kind:code", QtCore.Qt.Key_M: "kind:markdown",
                    QtCore.Qt.Key_A: "above", QtCore.Qt.Key_B: "after", QtCore.Qt.Key_D: "d",
                    QtCore.Qt.Key_X: "cut", QtCore.Qt.Key_C: "copy", QtCore.Qt.Key_V: "paste:below",
                    QtCore.Qt.Key_Z: "undelete", QtCore.Qt.Key_Up: "prev", QtCore.Qt.Key_K: "prev",
                    QtCore.Qt.Key_Down: "next", QtCore.Qt.Key_J: "next"}
    SHIFT_KEYS = {QtCore.Qt.Key_Up: "extend:up", QtCore.Qt.Key_K: "extend:up",
                  QtCore.Qt.Key_Down: "extend:down", QtCore.Qt.Key_J: "extend:down",
                  QtCore.Qt.Key_V: "paste:above", QtCore.Qt.Key_M: "merge",
                  QtCore.Qt.Key_Return: "run:advance", QtCore.Qt.Key_Enter: "run:advance"}

    def keyPressEvent(self, event):
        key, mods = event.key(), event.modifiers() & ~QtCore.Qt.KeypadModifier
        table = {QtCore.Qt.NoModifier: self.COMMAND_KEYS, QtCore.Qt.ShiftModifier: self.SHIFT_KEYS}.get(mods)
        if table is not None and key in table:
            self.actionRequested.emit(self.cell, table[key])
            return
        if key == QtCore.Qt.Key_O and mods in (QtCore.Qt.NoModifier, QtCore.Qt.ShiftModifier) \
                and isinstance(self.outputs, OutputArea):
            # Jupyter's O hides the outputs, Shift+O scrolls them or shows them whole
            self.outputs.toggle_scrolled() if mods else self.outputs.toggle_hidden()
            return
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and mods == QtCore.Qt.ControlModifier:
            self.actionRequested.emit(self.cell, "run")   # runs the selected cells
            return
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and not mods and self.editor is not None:
            if self.cell.kind == "markdown":
                self.edit_markdown()
            elif self.editor.isVisible():
                self.editor.setFocus()
            return
        super().keyPressEvent(event)

    def _render_markdown(self):
        self.view.setText(self.cell.source or "*Empty text cell; double-click to edit*")
        self.editor.setVisible(False)
        self.view.setVisible(True)

    def _build_step(self):
        mod = self.cell.modifier
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(6)
        self.enabled_box = QtWidgets.QCheckBox(toolTip="Enabled")
        self.enabled_box.setChecked(mod is None or mod.enabled)
        self.enabled_box.toggled.connect(self._toggle_enabled)
        self.enabled_box.setEnabled(mod is not None)
        head.addWidget(self.enabled_box)
        name = mod.name if mod is not None else _call_name(self.cell.source)
        self.title = QtWidgets.QLabel(name, objectName="stepTitle")
        head.addWidget(self.title)
        badge = _badge(mod)
        if badge:
            head.addWidget(QtWidgets.QLabel(badge, objectName="stepBadge"))
        head.addStretch()
        self.timing = QtWidgets.QLabel(objectName="muted")
        head.addWidget(self.timing)
        self.code_button = QtWidgets.QToolButton(text="Python", checkable=True, objectName="segment")
        self.code_button.toggled.connect(self.show_code)
        head.addWidget(self.code_button)
        self.body.addLayout(head)
        self.summary = QtWidgets.QLabel(objectName="stepSummary")
        self.summary.setFont(mono_font())
        self.summary.setWordWrap(True)
        self.body.addWidget(self.summary)
        self.editor = CodeEditor(self.cell.source)
        self.editor.escaped.connect(self.leave_editor)
        self.editor.setVisible(False)
        self.editor.textChanged.connect(self._source_edited)
        self.editor.runRequested.connect(lambda adv: self.runRequested.emit(self.cell, adv))
        self.editor.focused.connect(lambda: self.selected.emit(self.cell))
        self.body.addWidget(self.editor)
        res = QtWidgets.QHBoxLayout()
        self.chips = QtWidgets.QLabel(objectName="stepResult")
        self.chips.setWordWrap(True)
        res.addWidget(self.chips, 1)
        self.plot = MiniPlot()
        res.addWidget(self.plot, 0, QtCore.Qt.AlignRight)
        self.body.addLayout(res)
        self.outputs = OutputArea()  # errors of the step's own code
        self.body.addWidget(self.outputs)
        self.outputs.set_outputs([o for o in self.cell.outputs if o.kind == "error"])
        if mod is None and self.cell.state in ("new", "stale"):
            self.code_button.setChecked(True)

    def _build_view(self):
        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel("View", objectName="stepTitle"))
        head.addWidget(QtWidgets.QLabel("linked to the 3D view", objectName="muted"))
        head.addStretch()
        self.code_button = QtWidgets.QToolButton(text="Python", checkable=True, objectName="segment")
        self.code_button.toggled.connect(lambda on: self.editor.setVisible(on))
        head.addWidget(self.code_button)
        self.body.addLayout(head)
        self.editor = CodeEditor(self.cell.source)
        self.editor.escaped.connect(self.leave_editor)
        self.editor.setVisible(False)
        self.editor.textChanged.connect(self._source_edited)
        self.editor.runRequested.connect(lambda adv: self.runRequested.emit(self.cell, adv))
        self.editor.focused.connect(lambda: self.selected.emit(self.cell))
        self.body.addWidget(self.editor)
        png = next((o.data for o in self.cell.outputs if o.kind == "image" and o.view is not None), None)
        self.plot_slot = PlotSlot(self.cell.view or {}, png, self.cell.id)
        self.plot_slot.placeRequested.connect(self._place_view)
        self.plot_panel = self.plot_slot.panel
        for combo in (self.plot_panel.table, self.plot_panel.column, self.plot_panel.hist):
            combo.currentTextChanged.connect(self._view_changed)
        self.plot_panel.viewChanged.connect(self._view_changed)
        self.body.addWidget(self.plot_slot)
        self.outputs = OutputArea(live=False)  # errors (and a saved image until run): the plot replaces it
        self.body.addWidget(self.outputs)
        self._applying = False

    def set_view_data(self, data, pipeline=None):
        """Give a view cell's plot the current frame and select what its ``show(...)`` names."""
        args = dict(self.cell.view or {})
        if args.get("panel") != self.plot_slot.args.get("panel"):   # panel= edited in the code
            self.plot_slot.args = args
            self.plot_slot.decide(data, pipeline, force=True)
        else:
            self.plot_slot.decide(data, pipeline)
        self._applying = True
        try:
            self._applied_view = apply_view_args(self.plot_panel, self.cell.view or {}, data, pipeline,
                                                 getattr(self, "_applied_view", None))
        finally:
            self._applying = False

    def _view_changed(self, *_):
        if self._applying or self.plot_panel is None:
            return
        p = self.plot_panel
        hist = p.hist.currentText() or None
        y = p.column.currentText()
        kw = {"histogram": hist, "table": None if hist else (p.table.currentText() or None),
              "y": None if hist or y in ("", "(all)") or not p.column.isEnabled() else y}
        kw["xlim"], kw["ylim"] = (None if lim is None else tuple(float(f"{v:.4g}") for v in lim)
                                  for lim in p.view_limits())
        self._applied_view = (kw["xlim"], kw["ylim"])
        old = self.cell.source
        self.notebook.update_view(self.cell, **kw)  # other arguments (frame, bins) stay
        if self.cell.source != old:
            self.editor.blockSignals(True)
            self.editor.setPlainText(self.cell.source)
            self.editor.blockSignals(False)
            self.edited.emit(self.cell)

    def _place_view(self, docked: bool):
        """The view cell's plot moves (its menu): ``panel=`` goes into its ``show(...)`` call."""
        self.notebook.update_view(self.cell, panel=docked)
        self.plot_slot.args = dict(self.cell.view or {})
        self._source_rewritten()
        self.plot_slot.place(docked)

    def _place_output(self, k: int, docked: bool):
        """The k-th live plot under a code cell moves: ``panel=`` goes into the k-th ``show(...)``
        call when the cell's calls and plots match one to one (else only this run's plot moves)."""
        from ..views import set_keyword

        slot = self.outputs.slots[k]
        slot.args["panel"] = docked
        live = [o for o in self.cell.outputs if o.kind == "image" and o.view is not None]
        if k < len(live):
            live[k].view["panel"] = docked     # kept with the outputs (.ipynb, .ewd)
        calls = _show_calls(self.cell.source)
        new = set_keyword(self.cell.source, "panel", docked, index=k) if calls == len(live) else None
        if new is not None and new != self.cell.source:
            self.cell.source = new
            self.notebook.dirty = True
            self._source_rewritten()
        slot.place(docked)

    def _source_rewritten(self):
        if self.editor is not None and self.editor.toPlainText() != self.cell.source:
            self.editor.blockSignals(True)
            self.editor.setPlainText(self.cell.source)
            self.editor.blockSignals(False)
        self.edited.emit(self.cell)

    def show_code(self, on: bool = True):
        """Show or hide a step cell's Python (+ Step and View › Edit step open it)."""
        self.code_button.blockSignals(True)
        self.code_button.setChecked(on)
        self.code_button.blockSignals(False)
        if on and not self.editor.isVisible():
            self.editor.setPlainText(self.cell.source)
        self.editor.setVisible(on)

    def _toggle_enabled(self, on):
        mod = self.cell.modifier
        if mod is not None and mod.enabled != on:
            mod.enabled = on
            self.notebook.update_step(self.cell)
            self.edited.emit(self.cell)

    # ----------------------------------------------------------- state
    def update_state(self, data=None):
        c = self.cell
        if self.started is not None or self.property("state") == "queued":
            self._set_run_button(False)
        self.counter.setText({"new": "", "stale": "·"}.get(c.state, str(c.count) if c.count else ""))
        self.counter.setToolTip({"new": "Not run yet", "stale": "Out of date: an earlier cell changed",
                                 "error": "Failed"}.get(c.state, f"Run {c.count}" if c.count else ""))
        self.setProperty("state", c.state)
        if c.kind == "markdown":
            return
        if c.is_view:  # the plot replaces show()'s image, except for results saved in the file
            self.plot_slot.card.set_png(next((o.data for o in c.outputs if o.kind == "image" and o.view is not None), None))
            self.outputs.set_outputs(c.outputs if c.state == "new" else [o for o in c.outputs if o.kind == "error"])
            self._repolish()
            return
        if not c.is_step:
            self.outputs.set_outputs(c.outputs + self._step_errors())
            self._repolish()
            return
        mod = c.modifier
        if mod is not None:
            params = ", ".join(f"{k}={_fmt(v)}" for k, v in mod.parameters().items() if k not in ("source", "params"))
            self.summary.setText(params or "no parameters")
            if mod.status:
                self.timing.setText("error")
                self.chips.setText(mod.status)
                self.chips.setObjectName("stepError")
            else:
                t = mod.last_time
                self.timing.setText("" if t is None or not mod.enabled else (f"{t * 1000:.0f} ms" if t >= 0.001 else "<1 ms"))
                self.chips.setObjectName("stepResult")
                self.chips.setText("   ".join(result_chips(mod.last_output, data)) if mod.enabled else "disabled")
            plot = result_plot(mod.last_output, data) if mod.enabled and not mod.status else None
            if plot and plot[0] == "bars":
                self.plot.set_bars(plot[1], plot[2])
            elif plot:
                self.plot.set_line(plot[1], plot[2], plot[3])
            self.plot.setVisible(plot is not None)
            if self.editor is not None and not self.editor.isVisible() and self.editor.toPlainText() != c.source:
                self.editor.blockSignals(True)
                self.editor.setPlainText(c.source)
                self.editor.blockSignals(False)
        else:
            self.summary.setText(c.source.splitlines()[-1] if c.source else "")
            self.chips.setText("")
            self.plot.setVisible(False)
        self.outputs.set_outputs([o for o in c.outputs if o.kind == "error"])
        self._repolish()

    def _step_errors(self) -> list:
        """The error of the step an ``[each frame]`` / ``[all frames]`` cell added, at this frame
        (the same Output object while the message stays, so the output is not rebuilt per frame)."""
        c = self.cell
        if not c.frames:
            return []
        status = next((m.status for m in c.added if getattr(m, "status", None)), None) or \
            next((s.get("status") for s in c.steps if s.get("status")), None)
        if not status:
            return []
        if getattr(self, "_status_output", None) is None or self._status_output.data != status:
            from ..notebook import Output

            self._status_output = Output("error", status)
        return [self._status_output]

    def show_live_outputs(self):
        """The outputs of this cell while it runs: everything for a code cell; for a step or a
        view cell what it printed, its errors and progress bars (its header or plot is the rest)."""
        c = self.cell
        outs = list(c.outputs)
        if c.is_step or c.is_view:
            outs = [o for o in outs if o.kind in ("text", "error") or (o.kind == "rich" and o.display_id)]
        self.outputs.set_outputs(outs)

    def set_running(self, phase: str = "running", started: float | None = None):
        """Show that this cell runs now (``running``: a tinted card with a 2 px accent border, the
        run button turns into Interrupt with a turning ring, the counter counts the time) or waits
        for the cells before it (``queued``: a dashed border and ``*``, like Jupyter's queue)."""
        import time

        running = phase == "running"
        self._set_run_button(running, started if started is not None else time.monotonic())
        self.counter.setText(elapsed_text(time.monotonic() - self.started) if running else "*")
        self.counter.setToolTip("Running: the stop button above interrupts it" if running
                                else "Waiting for the cells before it")
        self.setProperty("state", phase)
        self._repolish()
        self.counter.repaint()

    def tick(self):
        """Advance the ring and the time of a running cell (the notebook's run timer)."""
        if self.started is None:
            return
        import time

        text = elapsed_text(time.monotonic() - self.started)
        if self.counter.text() != text:
            self.counter.setText(text)
        self.ring.update()

    def _set_run_button(self, running: bool, started: float | None = None):
        self.started = started if running else None
        name = "stop" if running else "play"
        self.run_button.setProperty("ewald_icon", name)
        self.run_button.setIcon(icon(name, TOKENS["accent"]))
        self.run_button.setToolTip("Interrupt: stop the running cell" if running else native_text("Run cell (Ctrl+Enter)"))
        self.ring.setGeometry(self.run_button.rect())
        self.ring.setVisible(running)

    def _run_clicked(self):
        if self.started is not None:
            self.actionRequested.emit(self.cell, "interrupt")
        else:
            self.runRequested.emit(self.cell, False)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.property("state") != "running":
            return
        # the style sheet tints the running cell; its border is drawn 2 px wide here, inside the
        # 1 px one, so the cell's contents do not shift when it starts (D72)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtGui.QPen(QtGui.QColor(TOKENS["accent"]), 2))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRoundedRect(QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1), 7.5, 7.5)
        p.end()

    def set_current(self, on: bool):
        self.setProperty("current", on)
        self._show_controls()
        self._repolish()

    def _repolish(self):
        chips = self.chips if self.cell.is_step else None
        key = (self.property("state"), self.property("current"), self.property("selected"), self.property("flash"),
               chips.objectName() if chips is not None else None)
        if key == getattr(self, "_polished", None):
            return   # nothing the style sheet reads changed (e.g. a new frame during playback)
        self._polished = key
        for w in (self, chips):
            if w is not None:
                w.style().unpolish(w)
                w.style().polish(w)

    FLASH_MS = 1200

    def flash(self):
        """Light the cell up for a moment (a link in the AI chat led here)."""
        self.setProperty("flash", True)
        self._repolish()

        def off():
            self.setProperty("flash", False)
            self._repolish()
        QtCore.QTimer.singleShot(self.FLASH_MS, self, off)

    def set_selected(self, on: bool):
        """Part of a multi-cell selection (Shift+Up/Down, Shift+click)."""
        if bool(self.property("selected")) != on:
            self.setProperty("selected", on)
            self._repolish()

    def mousePressEvent(self, event):
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            self.actionRequested.emit(self.cell, "extend:to")
            self.setFocus()
        else:
            self.selected.emit(self.cell)
        super().mousePressEvent(event)


def _shape(cell: Cell):
    """What a CellWidget was built for: a change means it has to be rebuilt."""
    return cell, cell.modifier, (cell.kind, cell.is_step, cell.is_view, cell.frames)


def _show_calls(source: str) -> int:
    """How many ``show(<name>, ...)`` calls ``source`` has (0 when it does not parse)."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    return sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "show"
               and bool(n.args) and isinstance(n.args[0], ast.Name) for n in ast.walk(tree))


def _same_shape(a, b) -> bool:
    return a[0] is b[0] and a[1] is b[1] and a[2] == b[2]


def _badge(mod) -> str:
    if mod is None:
        return ""
    from .. import modifiers as mods
    from ..pipeline import PythonModifier

    if isinstance(mod, (PythonModifier, mods.PythonScript)):
        return "Python"
    if type(mod).__name__ not in mods.__all__:
        return "plugin"
    return ""


def _call_name(source: str) -> str:
    arg = step_call(source)
    if arg is not None:
        import ast

        return ast.unparse(arg).split("(")[0]
    return "Step"


def fill_view_menu(menu, data, add):
    """'+ Plot' entries: every data table and a histogram of every numeric particle property."""
    menu.clear()
    tables = list(data.tables) if data is not None else []
    for name in tables:
        menu.addAction(name, lambda n=name: add(table=n))
    if not tables:
        menu.addAction("No data tables yet: add an analysis step").setEnabled(False)
    hist = menu.addMenu("Histogram of")
    props = [k for k, v in data.particles.items() if v.ndim == 1 and v.dtype.kind in "biuf"] if data is not None else []
    for name in props:
        hist.addAction(name, lambda n=name: add(histogram=n))
    hist.setEnabled(bool(props))


# ---------------------------------------------------------------- one notebook
