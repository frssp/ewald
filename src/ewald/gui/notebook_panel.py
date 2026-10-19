"""Notebook view: Python cells (like Jupyter) that build the pipeline the viewport shows.

Each tab is one :class:`ewald.notebook.Notebook` (a percent-format ``.py`` file). Code
cells run in the notebook's namespace and show printed text, values, errors and
matplotlib figures underneath. Step cells (one ``pipe.modifiers.append(...)``) show the
modifier as a parameter form with what it did to the data; their Python source follows
the form, and editing the source re-creates the modifier.
"""
from __future__ import annotations

import builtins
import keyword
import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..notebook import Cell, Notebook, new_notebook, step_call
from .params import ParameterForm
from .pipeline_panel import _discover, fill_modifier_menu
from .notebook_tabs import NotebookTabs
from .theme import TOKENS

MAX_EDITOR_LINES = 30


def mono_font() -> QtGui.QFont:
    f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
    f.setFamilies(["JetBrains Mono", "IBM Plex Mono", f.family()])
    f.setPixelSize(12)
    return f


# ---------------------------------------------------------------- editor
class PythonHighlighter(QtGui.QSyntaxHighlighter):
    """Keywords, builtins, strings, numbers, comments and decorators."""

    def __init__(self, document):
        super().__init__(document)

        def fmt(color, bold=False, italic=False):
            f = QtGui.QTextCharFormat()
            f.setForeground(QtGui.QColor(color))
            if bold:
                f.setFontWeight(QtGui.QFont.DemiBold)
            f.setFontItalic(italic)
            return f

        kw = "|".join(keyword.kwlist)
        bi = "|".join(n for n in dir(builtins) if not n.startswith("_"))
        self.rules = [(QtCore.QRegularExpression(rf"\b({kw})\b"), fmt(TOKENS["accent"], bold=True)),
                      (QtCore.QRegularExpression(rf"\b({bi})\b"), fmt(TOKENS["secondary"])),
                      (QtCore.QRegularExpression(r"\b[A-Z][A-Za-z0-9_]*(?=\()"), fmt(TOKENS["accent"])),
                      (QtCore.QRegularExpression(r"\b[0-9][0-9_]*\.?[0-9]*([eE][-+]?[0-9]+)?\b"), fmt(TOKENS["data"])),
                      (QtCore.QRegularExpression(r"@[A-Za-z_][A-Za-z0-9_.]*"), fmt(TOKENS["muted"])),
                      (QtCore.QRegularExpression(r"[rbfu]?\"[^\"\\n]*\"|[rbfu]?'[^'\\n]*'"), fmt(TOKENS["data"])),
                      (QtCore.QRegularExpression(r"#[^\n]*"), fmt(TOKENS["muted"], italic=True))]
        self.triple = fmt(TOKENS["data"])

    def highlightBlock(self, text):
        for regex, f in self.rules:
            it = regex.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), f)
        # triple-quoted strings spanning lines
        self.setCurrentBlockState(0)
        start = 0 if self.previousBlockState() == 1 else text.find('"""')
        while start >= 0:
            end = text.find('"""', start + (0 if self.previousBlockState() == 1 and start == 0 else 3))
            if end < 0:
                self.setCurrentBlockState(1)
                self.setFormat(start, len(text) - start, self.triple)
                break
            self.setFormat(start, end + 3 - start, self.triple)
            start = text.find('"""', end + 3)


class CodeEditor(QtWidgets.QPlainTextEdit):
    """Python editor that grows with its text. Ctrl+Enter runs, Shift+Enter runs and moves on."""

    runRequested = QtCore.Signal(bool)
    focused = QtCore.Signal()

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setFont(mono_font())
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)  # the page never scrolls sideways
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setObjectName("cellEditor")
        self.highlighter = PythonHighlighter(self.document())
        self.textChanged.connect(self._fit)
        self._fit()

    def _fit(self):
        doc, lines = self.document(), 0
        block = doc.firstBlock()
        while block.isValid():  # wrapped lines count too
            lines += max(1, block.layout().lineCount() if block.layout() is not None else 1)
            block = block.next()
        lines = max(1, lines)
        shown = min(lines, MAX_EDITOR_LINES)
        h = shown * self.fontMetrics().lineSpacing() + 2 * self.document().documentMargin() + 6
        self.setFixedHeight(int(h))
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded if lines > MAX_EDITOR_LINES
                                        else QtCore.Qt.ScrollBarAlwaysOff)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            QtCore.QTimer.singleShot(0, self, self._fit)  # rewrapped after the layout; dropped if deleted

    def keyPressEvent(self, event):
        key, mods = event.key(), event.modifiers()
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if mods & QtCore.Qt.ControlModifier:
                self.runRequested.emit(False)
                return
            if mods & QtCore.Qt.ShiftModifier:
                self.runRequested.emit(True)
                return
            line = self.textCursor().block().text()
            indent = line[:len(line) - len(line.lstrip(" "))]
            if line.rstrip().endswith(":"):
                indent += "    "
            super().keyPressEvent(event)
            self.insertPlainText(indent)
            return
        if key == QtCore.Qt.Key_Tab and not mods:
            self.insertPlainText("    ")
            return
        super().keyPressEvent(event)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.focused.emit()


# ---------------------------------------------------------------- outputs and results
class OutputArea(QtWidgets.QWidget):
    """Printed text, values, errors and images under a code cell."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.setContentsMargins(0, 2, 0, 0)
        self.lay.setSpacing(4)

    def set_outputs(self, outputs):
        while self.lay.count():
            w = self.lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for out in outputs:
            if out.kind == "image":
                pix = QtGui.QPixmap()
                pix.loadFromData(out.data, "PNG")
                if pix.width() > 560:
                    pix = pix.scaledToWidth(560, QtCore.Qt.SmoothTransformation)
                w = QtWidgets.QLabel()
                w.setPixmap(pix)
            else:
                w = QtWidgets.QLabel(str(out.data).rstrip("\n"))
                w.setFont(mono_font())
                w.setWordWrap(True)
                w.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                w.setObjectName({"error": "cellError", "value": "cellValue"}.get(out.kind, "cellText"))
            self.lay.addWidget(w)
        self.setVisible(bool(outputs))


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
class CellWidget(QtWidgets.QFrame):
    """One cell: gutter (run button, counter) and the code, markdown or step body."""

    runRequested = QtCore.Signal(object, bool)   # cell, advance
    selected = QtCore.Signal(object)
    edited = QtCore.Signal(object)               # a step's form changed its modifier
    actionRequested = QtCore.Signal(object, str)  # cell, 'up' | 'down' | 'delete' | 'code' | 'below'

    def __init__(self, notebook: Notebook, cell: Cell, parent=None):
        super().__init__(parent)
        self.notebook = notebook
        self.cell = cell
        self.editor = None
        self.form = None
        self.plot_panel = None
        self.built_for = _shape(cell)
        self.setObjectName("cell")
        self.setProperty("kind", cell.kind)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(4, 6, 8, 6)
        outer.setSpacing(6)
        gutter = QtWidgets.QVBoxLayout()
        gutter.setSpacing(2)
        self.run_button = QtWidgets.QToolButton(text="▶", toolTip="Run cell (Ctrl+Enter)", objectName="cellRun")
        self.run_button.clicked.connect(lambda: self.runRequested.emit(self.cell, False))
        self.counter = QtWidgets.QLabel(objectName="cellCounter")
        self.counter.setFont(mono_font())
        self.counter.setAlignment(QtCore.Qt.AlignCenter)
        gutter.addWidget(self.run_button, 0, QtCore.Qt.AlignHCenter)
        gutter.addWidget(self.counter, 0, QtCore.Qt.AlignHCenter)
        gutter.addStretch()
        outer.addLayout(gutter)
        self.body = QtWidgets.QVBoxLayout()
        self.body.setSpacing(6)
        outer.addLayout(self.body, 1)
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

    # ----------------------------------------------------------- bodies
    def _menu_button(self):
        b = QtWidgets.QToolButton(text="⋯", toolTip="Cell actions", objectName="cellMenu")
        b.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        m = QtWidgets.QMenu(b)
        if self.cell.kind == "code":
            m.addAction("Run", lambda: self.runRequested.emit(self.cell, False))
            m.addAction("Run this and below", lambda: self.actionRequested.emit(self.cell, "below"))
        if self.cell.is_step or self.cell.is_view:
            m.addAction("Show as code cell", lambda: self.actionRequested.emit(self.cell, "code"))
        if self.cell.is_view:
            m.addAction("Save plot…", lambda: self.plot_panel.save_plot())
        m.addSeparator()
        m.addAction("Move up", lambda: self.actionRequested.emit(self.cell, "up"))
        m.addAction("Move down", lambda: self.actionRequested.emit(self.cell, "down"))
        m.addAction("Delete", lambda: self.actionRequested.emit(self.cell, "delete"))
        b.setMenu(m)
        return b

    def _build_code(self):
        if self.cell.title:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(self.cell.title, objectName="cellTitle"))
            row.addStretch()
            row.addWidget(self._menu_button())
            self.body.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.editor = CodeEditor(self.cell.source)
        self.editor.textChanged.connect(self._source_edited)
        self.editor.runRequested.connect(lambda adv: self.runRequested.emit(self.cell, adv))
        self.editor.focused.connect(lambda: self.selected.emit(self.cell))
        row.addWidget(self.editor, 1)
        if not self.cell.title:
            row.addWidget(self._menu_button(), 0, QtCore.Qt.AlignTop)
        self.body.addLayout(row)
        self.outputs = OutputArea()
        self.body.addWidget(self.outputs)
        self.outputs.set_outputs(self.cell.outputs)

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
        if self.cell.kind == "markdown":
            self.view.setVisible(False)
            self.editor.setVisible(True)
            self.editor.setFocus()
        super().mouseDoubleClickEvent(event)

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
        from .plots import PlotsPanel

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
        self.editor.setVisible(False)
        self.editor.textChanged.connect(self._source_edited)
        self.editor.runRequested.connect(lambda adv: self.runRequested.emit(self.cell, adv))
        self.editor.focused.connect(lambda: self.selected.emit(self.cell))
        self.body.addWidget(self.editor)
        self.plot_panel = PlotsPanel()
        self.plot_panel.setMinimumHeight(300)
        self.plot_panel.compact()
        for combo in (self.plot_panel.table, self.plot_panel.column, self.plot_panel.hist):
            combo.currentTextChanged.connect(self._view_changed)
        self.body.addWidget(self.plot_panel)
        self.outputs = OutputArea()  # errors only: the plot replaces the image show() draws
        self.body.addWidget(self.outputs)
        self._applying = False

    def set_view_data(self, data, pipeline=None):
        """Give a view cell's plot the current frame and select what its ``show(...)`` names."""
        p = self.plot_panel
        p.pipeline = pipeline
        args = self.cell.view or {}
        self._applying = True
        try:
            p.set_data(data)
            if args.get("histogram"):
                p.hist.setCurrentText(args["histogram"])
            elif args.get("table"):
                p.hist.setCurrentText("")
                p.table.setCurrentText(args["table"])
            if args.get("y"):
                p.column.setCurrentText(args["y"])
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
        self.counter.setText({"new": "[ ]", "stale": "[·]"}.get(c.state, f"[{c.count}]" if c.count else "[ ]"))
        self.setProperty("state", c.state)
        if c.kind == "markdown":
            return
        if c.is_view:
            self.outputs.set_outputs([o for o in c.outputs if o.kind == "error"])
            self._repolish()
            return
        if not c.is_step:
            self.outputs.set_outputs(c.outputs)
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
            if self.editor is not None and not self.editor.isVisible():
                self.editor.blockSignals(True)
                self.editor.setPlainText(c.source)
                self.editor.blockSignals(False)
        else:
            self.summary.setText(c.source.splitlines()[-1] if c.source else "")
            self.chips.setText("")
            self.plot.setVisible(False)
        self.outputs.set_outputs([o for o in c.outputs if o.kind == "error"])
        self._repolish()

    def set_current(self, on: bool):
        self.setProperty("current", on)
        self._repolish()

    def _repolish(self):
        for w in (self, self.chips if self.cell.is_step else None):
            if w is not None:
                w.style().unpolish(w)
                w.style().polish(w)

    def mousePressEvent(self, event):
        self.selected.emit(self.cell)
        super().mousePressEvent(event)


def _shape(cell: Cell):
    """What a CellWidget was built for: a change means it has to be rebuilt."""
    return cell, cell.modifier, (cell.kind, cell.is_step, cell.is_view)


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


class _FitWidthScrollArea(QtWidgets.QScrollArea):
    """Scrolls vertically only: the page is always as wide as the area, like a document."""

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.widget() is not None:
            self.widget().setFixedWidth(self.viewport().width())


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
class NotebookView(QtWidgets.QWidget):
    """The cells of one notebook, a toolbar and the add-cell bar."""

    changed = QtCore.Signal()                 # pipeline modified: recompute
    pipelineReplaced = QtCore.Signal(object)  # the notebook bound 'pipe' to a new pipeline
    askRequested = QtCore.Signal()
    titleChanged = QtCore.Signal()
    plotCreated = QtCore.Signal(object)       # the live PlotsPanel of a new view cell

    def __init__(self, notebook: Notebook, parent=None):
        super().__init__(parent)
        self.notebook = notebook
        self.widgets: list[CellWidget] = []
        self.current = None
        self.data = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 6)
        run_all = QtWidgets.QPushButton("▶  Run all", objectName="primary", toolTip="Restart and run every cell")
        run_all.clicked.connect(self.run_all)
        restart = QtWidgets.QToolButton(text="Restart", toolTip="Clear the namespace (cells keep their code)")
        restart.clicked.connect(self.restart)
        self.py_button = QtWidgets.QToolButton(text=".py", checkable=True, toolTip="Show the notebook as its Python file")
        self.py_button.toggled.connect(self._show_file)
        bar.addWidget(run_all)
        bar.addWidget(restart)
        bar.addStretch()
        self.status = QtWidgets.QLabel(objectName="muted")
        bar.addWidget(self.status)
        bar.addWidget(self.py_button)
        lay.addLayout(bar)
        self.stack = QtWidgets.QStackedWidget()
        self.scroll = _FitWidthScrollArea(objectName="notebookScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # cells follow the width
        self.container = QtWidgets.QWidget(objectName="notebookPage")
        self.cells_layout = QtWidgets.QVBoxLayout(self.container)
        self.cells_layout.setContentsMargins(8, 4, 8, 12)
        self.cells_layout.setSpacing(8)
        self.scroll.setWidget(self.container)
        self.stack.addWidget(self.scroll)
        self.file_view = CodeEditor()
        self.file_view.setReadOnly(True)
        self.file_view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.file_view.textChanged.disconnect(self.file_view._fit)
        self.file_view.setMaximumHeight(16777215)
        self.file_view.setMinimumHeight(0)
        self.file_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.stack.addWidget(self.file_view)
        lay.addWidget(self.stack, 1)
        self.add_bar = self._make_add_bar()
        self.rebuild()

    def _make_add_bar(self):
        w = QtWidgets.QWidget(objectName="addBar")
        row = QtWidgets.QHBoxLayout(w)
        row.setContentsMargins(0, 4, 0, 0)
        row.setSpacing(6)
        code = QtWidgets.QPushButton("+ Code", objectName="addCell", toolTip="Python cell (Ctrl+Shift+N)")
        code.clicked.connect(lambda: self.add_cell("code"))
        step = QtWidgets.QPushButton("+ Step", objectName="addCell", toolTip="Add an analysis or modification step")
        self.step_menu = QtWidgets.QMenu(step)
        self.step_menu.aboutToShow.connect(lambda: fill_modifier_menu(self.step_menu, self.add_step))
        step.setMenu(self.step_menu)
        plugin = QtWidgets.QPushButton("+ Plugin", objectName="addCell", toolTip="Plugin steps, or load a plugin file")
        self.plugin_menu = QtWidgets.QMenu(plugin)
        self.plugin_menu.aboutToShow.connect(self._fill_plugin_menu)
        plugin.setMenu(self.plugin_menu)
        text = QtWidgets.QPushButton("+ Text", objectName="addCell", toolTip="Markdown text cell")
        text.clicked.connect(lambda: self.add_cell("markdown"))
        view = QtWidgets.QPushButton("+ View", objectName="addCell",
                                     toolTip="A plot of a data table (g(r), XRD, DOS, bands, thermo) or a histogram")
        self.view_menu = QtWidgets.QMenu(view)
        self.view_menu.aboutToShow.connect(lambda: fill_view_menu(self.view_menu, self.data, self.add_view))
        view.setMenu(self.view_menu)
        ask = QtWidgets.QPushButton("+ Ask Claude", objectName="addCell", toolTip="Ask Claude to add steps (Ctrl+J)")
        ask.clicked.connect(self.askRequested.emit)
        for b in (code, step, view, plugin, text, ask):
            row.addWidget(b)
        row.addStretch()
        return w

    def add_view(self, **kwargs) -> Cell:
        """Add a view cell (``show(pipe, **kwargs)``) after the current cell."""
        cell = self.notebook.add_view(self._index_after_current(), **kwargs)
        self.current = cell
        self.run(cell)  # rebuilds: the new cell gets its plot
        w = self.widget_for(cell)
        if w is not None:
            QtCore.QTimer.singleShot(0, w, lambda: self.scroll.ensureWidgetVisible(w))
        return cell

    def _fill_plugin_menu(self):
        from .. import plugins

        m = self.plugin_menu
        m.clear()
        found = sorted(plugins.modifiers().items())
        for name, p in found:
            act = m.addAction(name, lambda p=p: self._add_plugin(p.cls))
            act.setToolTip(p.description)
        if not found:
            m.addAction("No plugin steps installed").setEnabled(False)
        m.addSeparator()
        m.addAction("Load plugin file…", self._load_plugin_file)
        m.addAction("Open plugin folder", lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(plugins.plugin_dir()))))

    def _add_plugin(self, cls):
        try:
            mod = cls()
        except Exception as exc:  # needs arguments: give the user the code to fill in
            cell = Cell(f"from ewald.modifiers import {cls.__name__}\n"
                        f"{self.notebook.pipeline_name}.modifiers.append({cls.__name__}())  # {exc}")
            self._insert(cell)
            return
        self.add_step(mod)

    def _load_plugin_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load plugin file", "", "Python (*.py);;All files (*)")
        if path:
            cell = Cell(f"from ewald.plugins import load_plugin_file\nload_plugin_file({path!r})")
            self._insert(cell)
            self.run(cell)

    # ----------------------------------------------------------- building
    def rebuild(self):
        """Recreate every cell widget (after cells were added, removed or moved)."""
        keep = self.current
        for w in self.widgets:
            w.setParent(None)
            w.deleteLater()
        self.widgets = []
        while self.cells_layout.count():
            item = self.cells_layout.takeAt(0)
            if item.widget() is not None and item.widget() is not self.add_bar:
                item.widget().deleteLater()
        for cell in self.notebook.cells:
            w = CellWidget(self.notebook, cell)
            w.runRequested.connect(self.run)
            w.selected.connect(self.select)
            w.edited.connect(self._step_edited)
            w.actionRequested.connect(self._cell_action)
            self.widgets.append(w)
            self.cells_layout.addWidget(w)
            if w.plot_panel is not None:
                self.plotCreated.emit(w.plot_panel)
        self.cells_layout.addWidget(self.add_bar)
        self.cells_layout.addStretch()
        self.current = keep if keep in self.notebook.cells else (self.notebook.cells[-1] if self.notebook.cells else None)
        self._mark_current()
        self.update_results()

    def widget_for(self, cell):
        return next((w for w in self.widgets if w.cell is cell), None)

    def select(self, cell):
        self.current = cell
        self._mark_current()

    def _mark_current(self):
        for w in self.widgets:
            w.set_current(w.cell is self.current)

    def update_results(self, data=None):
        if data is not None:
            self.data = data
        for w in self.widgets:
            w.update_state(self.data)
            if w.plot_panel is not None and (data is not None or w.plot_panel.data is None):
                w.set_view_data(self.data, self.notebook.pipeline)
        nb = self.notebook
        pipe = nb.pipeline
        self.status.setText(("" if pipe is None else f"{len(pipe.modifiers)} steps · ") +
                            (os.path.basename(nb.path) if nb.path else "not saved") + (" ●" if nb.dirty else ""))
        if self.py_button.isChecked():
            self.file_view.setPlainText(nb.to_py())

    def _show_file(self, on):
        if on:
            self.file_view.setPlainText(self.notebook.to_py())
        self.stack.setCurrentIndex(1 if on else 0)

    # ----------------------------------------------------------- editing
    def _index_after_current(self):
        cells = self.notebook.cells
        return cells.index(self.current) + 1 if self.current in cells else len(cells)

    def _insert(self, cell, index=None):
        self.notebook.insert(self._index_after_current() if index is None else index, cell)
        self.current = cell
        self.rebuild()
        w = self.widget_for(cell)
        if w is not None and w.editor is not None:
            w.editor.setFocus()
            QtCore.QTimer.singleShot(0, w, lambda: self.scroll.ensureWidgetVisible(w))
        self.titleChanged.emit()
        return cell

    def add_cell(self, kind="code", source=""):
        return self._insert(Cell(source, kind=kind))

    def add_step(self, modifier):
        """A step cell for ``modifier`` after the current cell; the pipeline gets it right away."""
        if self.notebook.pipeline is None:
            QtWidgets.QMessageBox.information(self, "No pipeline yet",
                                              "Open a file first, or run a cell that sets pipe = import_file(...).")
            return None
        cell = self.notebook.add_step(modifier, self._index_after_current())
        self.current = cell
        self.rebuild()
        w = self.widget_for(cell)
        if w is not None:
            w.form_button.setChecked(True)
            QtCore.QTimer.singleShot(0, w, lambda: self.scroll.ensureWidgetVisible(w))
        self.changed.emit()
        self.titleChanged.emit()
        return cell

    def _step_edited(self, cell):
        self.changed.emit()
        self.titleChanged.emit()

    def _cell_action(self, cell, action):
        nb = self.notebook
        i = nb.cells.index(cell)
        if action == "delete":
            nb.delete(cell)
            self.current = nb.cells[min(i, len(nb.cells) - 1)] if nb.cells else None
            self.rebuild()
            self.changed.emit()
        elif action in ("up", "down"):
            nb.move(cell, i + (-1 if action == "up" else 1))
            self.rebuild()
            self.changed.emit()
        elif action == "code":
            w = self.widget_for(cell)
            w.code_button.setChecked(True)
        elif action == "below":
            self.run(cell, below=True)
        self.titleChanged.emit()

    # ----------------------------------------------------------- running
    def _busy(self, fn):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            return fn()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _after_run(self, before):
        nb = self.notebook
        if len(nb.cells) != len(self.widgets) or any(not _same_shape(w.built_for, _shape(c))
                                                     for w, c in zip(self.widgets, nb.cells)):
            self.rebuild()
        if nb.pipeline is not before and nb.pipeline is not None:
            self.pipelineReplaced.emit(nb.pipeline)
        else:
            self.changed.emit()
        self.update_results()
        self.titleChanged.emit()

    def run(self, cell, advance=False, below=False):
        nb = self.notebook
        before = nb.pipeline
        if cell in nb.cells and not nb.cells[nb.cells.index(cell)].kind == "markdown":
            start = nb.cells.index(cell)
            if below:
                def go():
                    for k in range(start, len(nb.cells)):
                        if not nb.run_cell(k, run_below=False):
                            break
                self._busy(go)
            else:
                self._busy(lambda: nb.run_cell(cell))
        self._after_run(before)
        if advance and cell in nb.cells:
            i = nb.cells.index(cell)
            if i + 1 < len(nb.cells):
                self.select(nb.cells[i + 1])
                w = self.widgets[i + 1]
                if w.editor is not None and w.editor.isVisible():
                    w.editor.setFocus()
            else:
                self.add_cell("code")

    def run_all(self):
        before = self.notebook.pipeline
        self._busy(self.notebook.run_all)
        self._after_run(before)

    def restart(self):
        self.notebook.reset()
        self.update_results()


# ---------------------------------------------------------------- tabs
CONFIRM_UNSAVED = True  # ask before closing notebooks with unsaved changes (tests turn it off)


class NotebookPanel(QtWidgets.QWidget):
    """Notebook tabs. Speaks the PipelinePanel interface the main window uses
    (``set_pipeline``, ``set_data``, ``add_modifier``, ``refresh``, ``changed``)."""

    changed = QtCore.Signal()
    styleChanged = QtCore.Signal()            # unused: display settings live in the Display panel
    pipelineReplaced = QtCore.Signal(object)
    askRequested = QtCore.Signal()
    plotCreated = QtCore.Signal(object)       # a view cell's live plot: the main window links it

    def __init__(self, parent=None):
        super().__init__(parent)
        _discover()
        self.data = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = NotebookTabs()
        self.tabs.newRequested.connect(lambda: self.new_notebook())
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.renameRequested.connect(self.rename)
        self.tabs.currentChanged.connect(self._tab_changed)
        lay.addWidget(self.tabs)
        self.add_menu = QtWidgets.QMenu(self)  # Pipeline menu / palette: same entries as + Step
        self.add_menu.aboutToShow.connect(lambda: fill_modifier_menu(self.add_menu, self.add_modifier))
        fill_modifier_menu(self.add_menu, self.add_modifier)
        self._switching = False
        self.new_notebook()

    # ----------------------------------------------------------- tabs
    @property
    def view(self) -> NotebookView | None:
        return self.tabs.currentWidget()

    @property
    def notebook(self) -> Notebook | None:
        return self.view.notebook if self.view is not None else None

    @property
    def pipeline(self):
        return self.notebook.pipeline if self.notebook is not None else None

    def views(self) -> list[NotebookView]:
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def _next_name(self):
        names = {v.notebook.name for v in self.views()}
        k = 1
        while f"Notebook {k}" in names:
            k += 1
        return f"Notebook {k}"

    def add_notebook(self, nb: Notebook) -> NotebookView:
        view = NotebookView(nb)
        view.changed.connect(self.changed.emit)
        view.pipelineReplaced.connect(self._replaced)
        view.askRequested.connect(self.askRequested.emit)
        view.plotCreated.connect(self.plotCreated.emit)
        for w in view.widgets:  # built before the connection
            if w.plot_panel is not None:
                self.plotCreated.emit(w.plot_panel)
        view.titleChanged.connect(lambda v=view: self._retitle(v))
        self._switching = True
        i = self.tabs.addTab(view, nb.name)
        self.tabs.setCurrentIndex(i)
        self._switching = False
        self._retitle(view)
        return view

    def new_notebook(self, name=None) -> NotebookView:
        nb = new_notebook(name or self._next_name())
        nb.run_all()  # just imports: gives the namespace the modifier names
        nb.dirty = False
        view = self.add_notebook(nb)
        view.update_results()
        return view

    def _retitle(self, view):
        i = self.tabs.indexOf(view)
        if i >= 0:
            self.tabs.setTabText(i, view.notebook.name + (" ●" if view.notebook.dirty else ""))
            self.tabs.setTabToolTip(i, view.notebook.path or "not saved")

    def _is_blank(self, view) -> bool:
        nb = view.notebook
        return nb.pipeline is None and not nb.dirty and nb.path is None

    def rename(self, index=None, name: str | None = None):
        """Rename a notebook (the tab; its file keeps its name until "Save as")."""
        view = self.tabs.widget(self.tabs.currentIndex() if index is None else index)
        if view is None:
            return
        if name is None:
            name, ok = QtWidgets.QInputDialog.getText(self, "Rename notebook", "Name:", text=view.notebook.name)
            if not ok or not name.strip():
                return
        view.notebook.name = name.strip()
        self._retitle(view)

    def confirm_discard(self, views=None, title="Close notebook") -> bool:
        """True if ``views`` (default: all) have no unsaved changes or the user lets them go."""
        dirty = [v.notebook.name for v in (self.views() if views is None else views) if v.notebook.dirty]
        if not dirty or not CONFIRM_UNSAVED:
            return True
        return QtWidgets.QMessageBox.question(
            self, title, f"Unsaved changes in {', '.join(dirty)}. Close anyway?") == QtWidgets.QMessageBox.Yes

    def close_tab(self, index):
        view = self.tabs.widget(index)
        if view is None or not self.confirm_discard([view]):
            return
        self.tabs.removeTab(index)
        view.deleteLater()
        if self.tabs.count() == 0:
            self.new_notebook()

    def _tab_changed(self, index):
        if self._switching or index < 0:
            return
        pipe = self.pipeline
        if pipe is not None:
            self.pipelineReplaced.emit(pipe)

    def _replaced(self, pipeline):
        if self.sender() is self.view:
            self.pipelineReplaced.emit(pipeline)

    # ----------------------------------------------------------- files
    def open_notebook(self, path, run: bool = False) -> NotebookView:
        nb = Notebook.load(path)
        for view in self.views():
            if view.notebook.path == nb.path:
                self.tabs.setCurrentWidget(view)
                return view
        blank = self.view if self.view is not None and self._is_blank(self.view) else None
        view = self.add_notebook(nb)
        if blank is not None:
            self.tabs.removeTab(self.tabs.indexOf(blank))
            blank.deleteLater()
        if run:
            view.run_all()
        else:
            view.update_results()
        return view

    def save(self, path=None) -> str | None:
        nb = self.notebook
        if nb is None:
            return None
        nb.save(path)
        nb.name = os.path.splitext(os.path.basename(nb.path))[0]
        self._retitle(self.view)
        self.view.update_results()
        return nb.path

    # ----------------------------------------------------------- PipelinePanel interface
    def set_pipeline(self, pipeline, source_code: str | None = None, name: str | None = None):
        """Show ``pipeline``: its tab if a notebook owns it, else a new notebook for it
        (replacing an untouched empty tab)."""
        for view in self.views():
            if view.notebook.pipeline is pipeline:
                if view is not self.view:
                    self._switching = True
                    self.tabs.setCurrentWidget(view)
                    self._switching = False
                view.update_results()
                return
        if name is None:
            src = getattr(pipeline.source, "paths", None) or [getattr(pipeline.source, "path", "")]
            name = os.path.splitext(os.path.basename(str(src[0])))[0] or self._next_name()
        nb = Notebook.from_pipeline(pipeline, source_code, name=name)
        blank = self.view if self.view is not None and self._is_blank(self.view) else None
        view = self.add_notebook(nb)
        if blank is not None:
            self._switching = True
            self.tabs.removeTab(self.tabs.indexOf(blank))
            blank.deleteLater()
            self._switching = False
        view.update_results()

    def set_data(self, data):
        self.data = data
        if self.view is not None:
            self.view.update_results(data)

    def add_modifier(self, mod):
        if self.view is not None and self.view.add_step(mod) is not None:
            pass

    def refresh(self, select=None):
        """After the pipeline changed outside the cells (console, Claude, viewport tools)."""
        view = self.view
        if view is None:
            return
        if view.notebook.sync():
            view.rebuild()
            self._retitle(view)
        if select is not None:
            mod = select[1] if isinstance(select, tuple) and len(select) == 2 else select
            cell = next((c for c in view.notebook.cells if c.modifier is mod), None)
            if cell is not None:
                view.select(cell)
                w = view.widget_for(cell)
                QtCore.QTimer.singleShot(0, w, lambda: view.scroll.ensureWidgetVisible(w))
        view.update_results()

    def update_status(self):
        if self.view is not None:
            self.view.update_results()

    def _current(self):
        return None
