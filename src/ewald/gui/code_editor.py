"""The Python code editor of notebook cells and the AI chat: :class:`CodeEditor` with
:class:`PythonHighlighter` in the theme's colors."""
from __future__ import annotations

import builtins
import keyword

from PySide6 import QtCore, QtGui, QtWidgets

from .theme import TOKENS

MAX_EDITOR_LINES = 30


def mono_font() -> QtGui.QFont:
    f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
    from .theme import STYLE

    f.setFamilies([*([STYLE["mono_font"]] if STYLE["mono_font"] else []), "JetBrains Mono", "IBM Plex Mono", f.family()])
    f.setPixelSize(12)
    return f


# ---------------------------------------------------------------- editor
class PythonHighlighter(QtGui.QSyntaxHighlighter):
    """Keywords, builtins, strings, numbers, comments and decorators."""

    def __init__(self, document):
        super().__init__(document)
        self._theme = None

    def _restyle(self):
        """Formats in the colors of the active UI theme (rebuilt when the theme changes)."""
        self._theme = dict(TOKENS)

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
        if self._theme != TOKENS:
            self._restyle()
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
    escaped = QtCore.Signal()       # Esc: leave the editor for the cell (Jupyter's command mode)

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
        try:
            doc, lines = self.document(), 0
        except RuntimeError:  # a late textChanged while the editor is being deleted
            return
        ls = self.fontMetrics().lineSpacing()
        block = doc.firstBlock()
        while block.isValid():  # wrapped lines count too; blockBoundingRect lays the block out first
            lines += max(1, round(self.blockBoundingRect(block).height() / ls))
            block = block.next()
        lines = max(1, lines)
        shown = min(lines, MAX_EDITOR_LINES)
        m = self.contentsMargins()  # frame and style sheet padding
        h = shown * ls + 2 * self.document().documentMargin() + max(6, m.top() + m.bottom())
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
        if key == QtCore.Qt.Key_Escape and not mods:
            self.escaped.emit()
            return
        super().keyPressEvent(event)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.focused.emit()


# ---------------------------------------------------------------- outputs and results
