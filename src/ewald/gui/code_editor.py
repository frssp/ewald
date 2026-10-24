"""The Python code editor of notebook cells and the AI chat: :class:`CodeEditor` with
:class:`PythonHighlighter` in the theme's colors."""
from __future__ import annotations

import builtins
import keyword
import math

from PySide6 import QtCore, QtGui, QtWidgets

from .theme import TOKENS

MAX_EDITOR_LINES = 30


def mono_font() -> QtGui.QFont:
    f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
    from .fonts import installed
    from .theme import STYLE

    f.setFamilies(installed([*([STYLE["mono_font"]] if STYLE["mono_font"] else []), "JetBrains Mono", "IBM Plex Mono",
                             f.family()]))
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


class FittedText(QtWidgets.QPlainTextEdit):
    """Plain text that is as tall as its laid-out lines, up to :attr:`max_lines` (then it
    scrolls), in pixels of the font the style sheet gave it. Rounding to whole lines of the
    font set before the style sheet, and not measuring again when the font changed, cut the last
    line off and the cell's output was drawn right over it (Windows at 200 %)."""

    max_lines = MAX_EDITOR_LINES
    grows = True                    # False: a fixed-size view (the notebook's Python file)
    pad = 6                         # at least this much around the text

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setFont(mono_font())
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.textChanged.connect(self._fit)

    def _fit(self):
        try:
            doc = self.document()
        except RuntimeError:  # a late textChanged while the widget is being deleted
            return
        if not self.grows:
            return
        ls = self.fontMetrics().lineSpacing()
        text_h, lines = 0.0, 0
        block = doc.firstBlock()
        while block.isValid():  # wrapped lines count too; blockBoundingRect lays the block out first
            h = self.blockBoundingRect(block).height()
            text_h += h
            lines += max(1, round(h / ls))
            block = block.next()
        text_h = max(text_h, ls)
        shown = text_h if lines <= self.max_lines else self.max_lines * ls
        m = self.contentsMargins()  # frame and style sheet padding
        h = math.ceil(shown + 2 * doc.documentMargin()) + max(self.pad, m.top() + m.bottom())
        if self.height() != h or self.minimumHeight() != h:
            self.setFixedHeight(h)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded if lines > self.max_lines
                                        else QtCore.Qt.ScrollBarAlwaysOff)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            QtCore.QTimer.singleShot(0, self, self._fit)  # rewrapped after the layout; dropped if deleted

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange):
            # the style sheet's font arrives after construction (and changes with the theme, the
            # font size preference and the screen's scaling): measure the text again
            self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
            QtCore.QTimer.singleShot(0, self, self._fit)


class CodeEditor(FittedText):
    """Python editor that grows with its text. Ctrl+Enter runs, Shift+Enter runs and moves on."""

    runRequested = QtCore.Signal(bool)
    focused = QtCore.Signal()
    escaped = QtCore.Signal()       # Esc: leave the editor for the cell (Jupyter's command mode)

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)  # the page never scrolls sideways
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        self.setObjectName("cellEditor")
        self.highlighter = PythonHighlighter(self.document())
        self._fit()

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


class OutputText(FittedText):
    """What a cell printed, returned or raised: selectable plain text that wraps anywhere (a
    traceback's rule or a long array line never runs past the cell) and is as tall as its lines.
    A label's height came from a width its unbreakable lines made too wide, so the last lines
    were cut off."""

    fitted = QtCore.Signal()   # measured again: the outputs' rule may toggle it now, or no more
    scroll_lines = 30          # longer output scrolls in place, as in Jupyter (python.output_lines; 0: never)
    full_lines = 10000         # even an expanded log stops growing here
    pad = 0
    expanded = False           # True: every line shown (the left rule of the outputs toggles it)
    follow_end = False         # the box stays scrolled to its last line

    def __init__(self, text="", kind="text", parent=None):
        super().__init__(text, parent)
        self.setReadOnly(True)
        self.document().setDocumentMargin(2)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.viewport().setCursor(QtCore.Qt.IBeamCursor)
        self.setObjectName({"error": "cellError", "value": "cellValue"}.get(kind, "cellText"))
        self.setProperty("ewald_kind", kind)
        # a scrolled traceback shows its end (the exception); scrolling up lets go of it
        self.follow_end = kind == "error"
        self.verticalScrollBar().actionTriggered.connect(self._scrolled_by_user)
        self._fit()

    def _scrolled_by_user(self, _action):
        bar = self.verticalScrollBar()
        QtCore.QTimer.singleShot(0, self, lambda: setattr(self, "follow_end", bar.value() >= bar.maximum()))

    @property
    def max_lines(self):
        return self.full_lines if self.expanded or self.scroll_lines <= 0 else self.scroll_lines

    def line_count(self) -> int:
        """Laid-out lines (wrapped ones count): more than ``scroll_lines`` scroll."""
        ls = max(1, self.fontMetrics().lineSpacing())
        block, n = self.document().firstBlock(), 0
        while block.isValid():
            n += max(1, round(self.blockBoundingRect(block).height() / ls))
            block = block.next()
        return n

    def overflows(self) -> bool:
        """More lines than a scrolled output shows: the rule can expand it."""
        return 0 < self.scroll_lines < self.line_count()

    def set_expanded(self, on: bool):
        if on != self.expanded:
            self.expanded = on
            self._fit()

    def _fit(self):
        super()._fit()
        try:
            if self.follow_end:
                bar = self.verticalScrollBar()
                bar.setValue(bar.maximum())
            self.fitted.emit()
        except RuntimeError:   # being deleted
            pass

    def text(self) -> str:
        return self.toPlainText()

    def setText(self, text: str):
        if text == self.toPlainText():
            return
        bar = self.verticalScrollBar()
        old = self.toPlainText()
        if old and text.startswith(old):
            # a running cell printed more: append, and a log scrolled to its end keeps following it
            self.follow_end = self.follow_end or bar.value() >= bar.maximum()
            cursor = QtGui.QTextCursor(self.document())
            cursor.movePosition(QtGui.QTextCursor.End)
            cursor.insertText(text[len(old):])
        else:
            self.setPlainText(text)
