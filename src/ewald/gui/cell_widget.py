"""One notebook cell in the window: :class:`CellWidget` (gutter, editor, step form, outputs)
with its :class:`OutputArea`, result chips and mini plots."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..notebook import Cell, Notebook, step_call
from .params import ParameterForm
from . import header
from .icons import icon
from .theme import TOKENS
from .code_editor import CodeEditor, mono_font


class OutputArea(QtWidgets.QWidget):
    """Printed text, values, errors and images under a code cell. An image ``show(pipe, ...)``
    drew is a live plot here (``live``): it follows the frame and is linked to the 3D view."""

    MAX_IMAGE_HEIGHT = 360   # a figure or snapshot never takes the whole panel
    plotCreated = QtCore.Signal(object)   # a live PlotsPanel: the main window links it

    def __init__(self, parent=None, live: bool = True):
        super().__init__(parent)
        self.setObjectName("cellOutput")
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.setContentsMargins(2, 2, 0, 0)
        self.lay.setSpacing(4)
        self.live = live
        self._images = []    # (label, full-size pixmap): rescaled when the cell's width changes
        self.panels = []     # (PlotsPanel, show() arguments) of the live plots

    def set_outputs(self, outputs):
        outputs = list(outputs)
        old = getattr(self, "outputs", None)
        if old is not None and len(old) == len(outputs) and all(a is b for a, b in zip(old, outputs)):
            return   # the same outputs (every frame of playback): rebuilding them made the cell flicker
        self.outputs = outputs
        while self.lay.count():
            w = self.lay.takeAt(0).widget()
            if w is not None:
                w.hide()          # gone now, not at the next event loop pass (no stale copy drawn)
                w.deleteLater()
        self._images, self.panels = [], []
        for out in outputs:
            if out.kind == "image" and out.view is not None and self.live:
                w = live_plot()
                self.panels.append((w, out.view))
                self.plotCreated.emit(w)
            elif out.kind == "image":
                pix = QtGui.QPixmap()
                pix.loadFromData(out.data, "PNG")
                w = QtWidgets.QLabel(objectName="cellImage")
                w.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
                self._images.append((w, pix))
            else:
                w = QtWidgets.QLabel(str(out.data).rstrip("\n"))
                w.setFont(mono_font())
                w.setWordWrap(True)
                w.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                w.setObjectName({"error": "cellError", "value": "cellValue"}.get(out.kind, "cellText"))
            self.lay.addWidget(w)
        self._fit_images()
        self.setVisible(bool(outputs))

    def _fit_images(self):
        width = max(120, self.width() - 4)
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


def live_plot():
    """The live plot of a view cell or a ``show(pipe, ...)`` output."""
    from .plots import PlotsPanel

    p = PlotsPanel()
    p.setMinimumHeight(220)
    p.setMaximumHeight(320)
    p.compact()
    return p


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


class CellWidget(QtWidgets.QFrame):
    """One cell: gutter (run button, counter) and the code, markdown or step body."""

    runRequested = QtCore.Signal(object, bool)   # cell, advance
    selected = QtCore.Signal(object)
    edited = QtCore.Signal(object)               # a step's form changed its modifier
    # cell, 'up' | 'down' | 'delete' | 'code' | 'below' | 'clear' | 'kind:code' | 'kind:markdown' |
    # 'above' | 'after' (insert a code cell) | 'ask' | 'frames:...'
    actionRequested = QtCore.Signal(object, str)
    plotCreated = QtCore.Signal(object)          # a live plot of a show(pipe, ...) output

    def __init__(self, notebook: Notebook, cell: Cell, parent=None):
        super().__init__(parent)
        self.notebook = notebook
        self.cell = cell
        self.editor = None
        self.form = None
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
        self.run_button = header.button("play", "Run cell (Ctrl+Enter)", name="cellRun")
        self.run_button.setProperty("ewald_icon_color", "accent")
        self.run_button.setIcon(icon("play", TOKENS["accent"]))
        self.run_button.clicked.connect(lambda: self.runRequested.emit(self.cell, False))
        self.counter = QtWidgets.QLabel(objectName="cellCounter")
        self.counter.setFont(mono_font())
        self.counter.setAlignment(QtCore.Qt.AlignCenter)
        self.fold_button = header.button("chevron_down", "Collapse or expand the cell", name="cellFold")
        self.fold_button.clicked.connect(lambda: self.set_collapsed(not self.cell.collapsed))
        self.drag_handle = _DragHandle(cell.id)
        gutter.addWidget(self.drag_handle, 0, QtCore.Qt.AlignHCenter)
        gutter.addWidget(self.run_button, 0, QtCore.Qt.AlignHCenter)
        gutter.addWidget(self.counter, 0, QtCore.Qt.AlignHCenter)
        gutter.addWidget(self.fold_button, 0, QtCore.Qt.AlignHCenter)
        gutter.addStretch()
        for w in (self.drag_handle, self.fold_button):   # shown on hover and on the current cell only
            policy = w.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            w.setSizePolicy(policy)
        outer.addWidget(self.gutter)
        right = QtWidgets.QVBoxLayout()
        self.content = QtWidgets.QWidget()
        self.body = QtWidgets.QVBoxLayout(self.content)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(4)
        self.folded_label = QtWidgets.QLabel(objectName="cellFolded")
        self.folded_label.setToolTip("Collapsed: click the chevron or double-click to expand")
        right.addWidget(self.content)
        right.addWidget(self.folded_label)
        outer.addLayout(right, 1)
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
            m.addAction("Run", lambda: self.runRequested.emit(self.cell, False))
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
        m.addAction("Move up", act("up"))
        m.addAction("Move down", act("down"))
        m.addAction("Delete", act("delete"))
        b.setMenu(m)
        return b

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
            row.addWidget(self._menu_button())
            self.body.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.editor = CodeEditor(self.cell.source)
        self.editor.escaped.connect(self.leave_editor)
        self.editor.textChanged.connect(self._source_edited)
        self.editor.runRequested.connect(lambda adv: self.runRequested.emit(self.cell, adv))
        self.editor.focused.connect(lambda: self.selected.emit(self.cell))
        row.addWidget(self.editor, 1)
        if not (self.cell.title or tagged):
            row.addWidget(self._menu_button(), 0, QtCore.Qt.AlignTop)
        self.body.addLayout(row)
        self.outputs = OutputArea()
        self.outputs.plotCreated.connect(self.plotCreated.emit)
        self.body.addWidget(self.outputs)
        self.outputs.set_outputs(self.cell.outputs)

    def panels(self) -> list:
        """The live plots in this cell: a view cell's, and those of its ``show(pipe, ...)`` outputs."""
        live = [p for p, _ in self.outputs.panels] if isinstance(self.outputs, OutputArea) else []
        return ([self.plot_panel] if self.plot_panel is not None else []) + live

    def set_live_data(self, data, pipeline=None, force: bool = False):
        """The current frame for the live plots of ``show(pipe, ...)`` outputs (``force``: a new
        frame; else only plots that have none yet)."""
        if not isinstance(self.outputs, OutputArea):
            return
        for p, args in self.outputs.panels:
            if force or p.data is None:
                p._applied = apply_view_args(p, args, data, pipeline, getattr(p, "_applied", None))

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
        self.view.setToolTip("Double-click to edit; Ctrl+Enter shows the text")
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
        row.addWidget(self._menu_button(), 0, QtCore.Qt.AlignTop)
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
        """Esc in the editor: the cell keeps the keyboard (Y, M, A, B, Enter), as in Jupyter."""
        if self.cell.kind == "markdown":
            self._render_markdown()
        self.selected.emit(self.cell)
        self.setFocus()

    COMMAND_KEYS = {QtCore.Qt.Key_Y: "kind:code", QtCore.Qt.Key_M: "kind:markdown",
                    QtCore.Qt.Key_A: "above", QtCore.Qt.Key_B: "after"}

    def keyPressEvent(self, event):
        key, mods = event.key(), event.modifiers()
        if not mods & ~QtCore.Qt.KeypadModifier and key in self.COMMAND_KEYS:
            self.actionRequested.emit(self.cell, self.COMMAND_KEYS[key])
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
        self.form_button = QtWidgets.QToolButton(text="Form", checkable=True, objectName="segment")
        self.code_button = QtWidgets.QToolButton(text="Python", checkable=True, objectName="segment")
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(False)
        self.form_button.toggled.connect(lambda on: self._show_page("form" if on else None))
        self.code_button.toggled.connect(lambda on: self._show_page("code" if on else None))
        head.addWidget(self.form_button)
        head.addWidget(self.code_button)
        head.addWidget(self._menu_button())
        self.body.addLayout(head)
        self.summary = QtWidgets.QLabel(objectName="stepSummary")
        self.summary.setFont(mono_font())
        self.summary.setWordWrap(True)
        self.body.addWidget(self.summary)
        self.form_page = QtWidgets.QWidget()
        fl = QtWidgets.QVBoxLayout(self.form_page)
        fl.setContentsMargins(0, 0, 0, 0)
        if mod is not None:
            doc = (type(mod).__doc__ or "").strip().split("\n\n")[0]
            if doc:
                fl.addWidget(QtWidgets.QLabel(" ".join(doc.replace("``", "").split()), wordWrap=True, objectName="muted"))
            self.form = ParameterForm(mod)
            fl.addWidget(self.form)
            apply = QtWidgets.QPushButton("Apply  (Ctrl+Enter)", objectName="primary")
            apply.clicked.connect(self._apply_form)
            QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self.form_page, activated=self._apply_form)
            fl.addWidget(apply, 0, QtCore.Qt.AlignLeft)
        self.form_page.setVisible(False)
        self.body.addWidget(self.form_page)
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
        head.addWidget(self._menu_button())
        self.body.addLayout(head)
        self.editor = CodeEditor(self.cell.source)
        self.editor.escaped.connect(self.leave_editor)
        self.editor.setVisible(False)
        self.editor.textChanged.connect(self._source_edited)
        self.editor.runRequested.connect(lambda adv: self.runRequested.emit(self.cell, adv))
        self.editor.focused.connect(lambda: self.selected.emit(self.cell))
        self.body.addWidget(self.editor)
        self.plot_panel = live_plot()
        for combo in (self.plot_panel.table, self.plot_panel.column, self.plot_panel.hist):
            combo.currentTextChanged.connect(self._view_changed)
        self.plot_panel.viewChanged.connect(self._view_changed)
        self.body.addWidget(self.plot_panel)
        self.outputs = OutputArea(live=False)  # errors (and a saved image until run): the plot replaces it
        self.body.addWidget(self.outputs)
        self._applying = False

    def set_view_data(self, data, pipeline=None):
        """Give a view cell's plot the current frame and select what its ``show(...)`` names."""
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

    def _show_page(self, page):
        for b, name in ((self.form_button, "form"), (self.code_button, "code")):
            b.blockSignals(True)
            b.setChecked(page == name)
            b.blockSignals(False)
        self.form_page.setVisible(page == "form")
        self.editor.setVisible(page == "code")
        if page == "code":
            self.editor.setPlainText(self.cell.source)
        self.summary.setVisible(page != "form")

    def _toggle_enabled(self, on):
        mod = self.cell.modifier
        if mod is not None and mod.enabled != on:
            mod.enabled = on
            self.notebook.update_step(self.cell)
            self.edited.emit(self.cell)

    def _apply_form(self):
        if self.form is None:
            return
        self.form.apply()
        self.notebook.update_step(self.cell)
        self.edited.emit(self.cell)

    # ----------------------------------------------------------- state
    def update_state(self, data=None):
        c = self.cell
        self.counter.setText({"new": "", "stale": "·"}.get(c.state, str(c.count) if c.count else ""))
        self.counter.setToolTip({"new": "Not run yet", "stale": "Out of date: an earlier cell changed",
                                 "error": "Failed"}.get(c.state, f"Run {c.count}" if c.count else ""))
        self.setProperty("state", c.state)
        if c.kind == "markdown":
            return
        if c.is_view:  # the plot replaces show()'s image, except for results saved in the file
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

    def set_running(self):
        self.counter.setText("*")
        self.counter.setToolTip("Running")
        self.setProperty("state", "running")
        self._repolish()
        self.counter.repaint()

    def set_current(self, on: bool):
        self.setProperty("current", on)
        self._show_controls()
        self._repolish()

    def _repolish(self):
        chips = self.chips if self.cell.is_step else None
        key = (self.property("state"), self.property("current"), chips.objectName() if chips is not None else None)
        if key == getattr(self, "_polished", None):
            return   # nothing the style sheet reads changed (e.g. a new frame during playback)
        self._polished = key
        for w in (self, chips):
            if w is not None:
                w.style().unpolish(w)
                w.style().polish(w)

    def mousePressEvent(self, event):
        self.selected.emit(self.cell)
        super().mousePressEvent(event)


def _shape(cell: Cell):
    """What a CellWidget was built for: a change means it has to be rebuilt."""
    return cell, cell.modifier, (cell.kind, cell.is_step, cell.is_view, cell.frames)


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
    """'+ View' entries: every data table and a histogram of every numeric particle property."""
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
