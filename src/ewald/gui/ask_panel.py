"""Ask Claude panel: questions about the picked atoms (or the whole frame), answered in a side chat.

The conversation reads like a chat app (Chainlit-style): the user's messages in bubbles on the
right, Claude's answers on the left under its avatar, the tools it ran as small step chips,
and code blocks as framed, highlighted code with Copy and Insert buttons. In the notebook
layout this is the AI chat side bar (the Claude button in the activity bar, left of the
notebook): each question also carries the current notebook's code when it changed, and every
Python block in an answer can be inserted as a notebook cell.

Pick atoms in the viewport (Shift+click with the pick tool adds more), type a question and
press Ask. The panel sends ewald's description of the picked atoms and their surroundings
plus a close-up render (:meth:`ewald.llm.Agent.ask_about_selection`); Claude can call the
usual tools, and modifiers it adds show up in the pipeline. With nothing picked, the
question goes about the whole pipeline. Claude runs in a worker thread, so the viewer stays
responsive. Needs ``pip install ewald[llm]`` and an API key (Preferences › Claude, or
``ANTHROPIC_API_KEY``).
"""
from __future__ import annotations

import re
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from .icons import icon
from .notebook_panel import CodeEditor, _FitWidthScrollArea
from .theme import TOKENS

_SETUP_HINT = ("Claude needs the `anthropic` package (`pip install ewald[llm]`) and an API key: "
               "Edit › Preferences › Claude, or `ANTHROPIC_API_KEY`.")


NOTEBOOK_NOTE = """

You are the chat side bar of ewald's notebook window. The user's notebook is a percent-format \
Python file (`# %%` cells, one namespace) where `pipe` is the pipeline: step cells are one \
`pipe.add(Modifier(...))` (modifiers from `ewald.modifiers`, imported as `mods` or by name), \
view cells are one `show(pipe, ...)`, and `load(path)` opens a file. When the user's message \
starts with the notebook's code, that is its current state. Code you put in a ```python block \
gets an "Insert as notebook cell" button, so write each block as one complete cell that runs \
in that namespace; prefer `pipe.add(...)` steps over ad-hoc loops."""

_MAX_NOTEBOOK = 20000  # characters of notebook code sent with a question
_CODE_BLOCK = re.compile(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", re.S)
_ANY_BLOCK = re.compile(r"```([\w+.-]*)[ \t]*\n(.*?)```", re.S)
_PYTHON = ("", "python", "py", "python3")
STARTERS = ("Describe this structure", "What analysis should I run next?", "Explain the notebook's last cell")


def _default_agent(pipeline, on_event, **kwargs):
    from ..llm import Agent

    return Agent.from_pipeline(pipeline, on_event=on_event, **kwargs)


def agent_from_settings(get_settings: Callable[[], object], notebook: bool = False):
    """Agent factory using Preferences › Claude (model, effort), read when a chat starts;
    ``notebook`` tells Claude about the notebook it sits beside."""
    def factory(pipeline, on_event):
        s = get_settings()
        kwargs = {"model": s["llm.model"], "effort": s["llm.effort"]}
        if notebook:
            from ..llm.agent import SYSTEM_PROMPT

            kwargs["system"] = SYSTEM_PROMPT + NOTEBOOK_NOTE
        return _default_agent(pipeline, on_event, **kwargs)
    return factory


def code_blocks(text: str) -> list[str]:
    """The Python code blocks of a Markdown answer (fenced as python, py or untagged)."""
    return [m.group(1).rstrip() + "\n" for m in _CODE_BLOCK.finditer(text) if m.group(1).strip()]



def _step_done(chip):
    """A tool chip from "Running name…" to "✓ Used name"."""
    if chip.property("tool"):
        chip.setText(f"✓ Used {chip.property('tool')}")

class _Worker(QtCore.QThread):
    event = QtCore.Signal(str, object)
    answered = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, job: Callable[[], str], parent=None):
        super().__init__(parent)
        self.job = job

    def run(self):
        try:
            self.answered.emit(self.job())
        except Exception as exc:  # shown in the chat, never raised into Qt
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def split_answer(text: str) -> list[tuple[str, str, str]]:
    """``text`` as ``(kind, lang, body)`` parts: ``("text", "", markdown)`` and ``("code", lang, code)``."""
    parts, at = [], 0
    for m in _ANY_BLOCK.finditer(text):
        if text[at:m.start()].strip():
            parts.append(("text", "", text[at:m.start()].strip()))
        if m.group(2).strip():
            parts.append(("code", m.group(1).lower(), m.group(2).rstrip() + "\n"))
        at = m.end()
    if text[at:].strip():
        parts.append(("text", "", text[at:].strip()))
    return parts


def _markdown_label(text: str, name: str = "chatText") -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(objectName=name)
    label.setTextFormat(QtCore.Qt.MarkdownText)
    label.setText(text)
    label.setWordWrap(True)
    label.setOpenExternalLinks(True)
    label.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
    label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    return label


class CodeBlock(QtWidgets.QFrame):
    """A code block of an answer: language, Copy and (Python in a notebook) Insert as cell."""

    def __init__(self, code: str, lang: str = "", insert: Callable[[str], None] | None = None, parent=None):
        super().__init__(parent, objectName="chatCode")
        self.code = code
        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(8, 4, 4, 0)
        head.addWidget(QtWidgets.QLabel(lang or "python", objectName="chatCodeLang"), 1)
        self.copy_button = QtWidgets.QToolButton(text="Copy", objectName="chatCodeButton")
        self.copy_button.clicked.connect(self.copy)
        head.addWidget(self.copy_button)
        self.insert_button = None
        if insert is not None:
            self.insert_button = QtWidgets.QToolButton(text="+ Insert as cell", objectName="chatCodeButton",
                                                       toolTip="Add this code as a cell after the current one")
            self.insert_button.clicked.connect(lambda: insert(self.code))
            head.addWidget(self.insert_button)
        self.editor = CodeEditor(code.rstrip("\n"))
        self.editor.setReadOnly(True)
        self.editor.setObjectName("chatCodeEditor")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 4)
        lay.setSpacing(0)
        lay.addLayout(head)
        lay.addWidget(self.editor)

    def copy(self):
        QtWidgets.QApplication.clipboard().setText(self.code)
        self.copy_button.setText("Copied")
        QtCore.QTimer.singleShot(1500, self.copy_button, lambda: self.copy_button.setText("Copy"))


class Message(QtWidgets.QWidget):
    """One chat message: ``role`` is "user" (bubble on the right), "assistant" (avatar and
    answer on the left) or "note" (a centered, muted line)."""

    def __init__(self, role: str, text: str = "", meta: str = "", insert=None, parent=None):
        super().__init__(parent)
        self.role, self.text, self.meta = role, text, meta
        self.code_blocks: list[CodeBlock] = []
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        if role == "user":
            bubble = QtWidgets.QFrame(objectName="chatUserBubble")
            bl = QtWidgets.QVBoxLayout(bubble)
            bl.setContentsMargins(12, 8, 12, 8)
            bl.setSpacing(2)
            bl.addWidget(_markdown_label(text, "chatUserText"))
            if meta:
                bl.addWidget(QtWidgets.QLabel(meta, objectName="chatMeta"), 0, QtCore.Qt.AlignRight)
            bubble.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
            row.addStretch(1)
            row.addWidget(bubble, 4)
            return
        if role == "note":
            label = _markdown_label(text, "chatNote")
            label.setAlignment(QtCore.Qt.AlignCenter)
            row.addWidget(label)
            return
        avatar = QtWidgets.QLabel("C", objectName="chatAvatar", alignment=QtCore.Qt.AlignCenter)
        avatar.setFixedSize(26, 26)
        row.addWidget(avatar, 0, QtCore.Qt.AlignTop)
        self.body = QtWidgets.QVBoxLayout()
        self.body.setContentsMargins(0, 2, 0, 0)
        self.body.setSpacing(6)
        self.body.addWidget(QtWidgets.QLabel("Claude", objectName="chatAuthor"))
        self.steps = QtWidgets.QVBoxLayout()
        self.steps.setSpacing(2)
        self.body.addLayout(self.steps)
        self.status = QtWidgets.QLabel("Thinking…", objectName="chatThinking")
        self.body.addWidget(self.status)
        row.addLayout(self.body, 1)
        self.insert = insert
        if text:
            self.set_answer(text)

    def add_step(self, name: str):
        """A tool Claude runs: a chip that turns from running to done."""
        if self.steps.count():
            _step_done(self.steps.itemAt(self.steps.count() - 1).widget())
        chip = QtWidgets.QLabel(f"Running {name}…", objectName="chatStep")
        chip.setProperty("tool", name)
        self.steps.addWidget(chip, 0, QtCore.Qt.AlignLeft)
        self.status.setText("Working…")

    def set_answer(self, text: str, error: bool = False):
        self.text = text
        for i in range(self.steps.count()):
            _step_done(self.steps.itemAt(i).widget())
        self.status.hide()
        if error:
            self.body.addWidget(_markdown_label(text, "chatError"))
            return
        for kind, lang, body in split_answer(text):
            if kind == "text":
                self.body.addWidget(_markdown_label(body))
            else:
                block = CodeBlock(body, lang, self.insert if lang in _PYTHON else None)
                self.code_blocks.append(block)
                self.body.addWidget(block)


class AskPanel(QtWidgets.QWidget):
    """Chat with Claude about the current selection and notebook.

    ``context()`` returns ``(pipeline, frame, indices)`` for the moment Ask is pressed
    (``indices`` may be empty). ``agent_factory(pipeline, on_event)`` builds the agent; tests
    pass a fake. ``pipelineChanged`` fires after an answer when Claude changed the modifiers.
    ``notebook()``, when given, returns the current notebook's code (or None): it goes with a
    question whenever it changed, and Python blocks in answers offer Insert as cell
    (``insertRequested(code)``).
    """

    pipelineChanged = QtCore.Signal()
    insertRequested = QtCore.Signal(str)

    def __init__(self, context: Callable[[], tuple], agent_factory=_default_agent, parent=None,
                 notebook: Callable[[], str | None] | None = None):
        super().__init__(parent, objectName="chatPanel")
        self.context = context
        self.agent_factory = agent_factory
        self.notebook = notebook
        self.agent = None
        self._agent_pipeline = None
        self._sent_notebook: str | None = None
        self.snippets: list[str] = []  # Python code blocks of the answers, in order
        self.messages: list[Message] = []
        self.worker: _Worker | None = None
        self._pending: Message | None = None

        # --- header: what the question is about, New chat
        self.selection_label = QtWidgets.QLabel(objectName="chatContext")
        self.new_button = QtWidgets.QToolButton(text="New chat", objectName="chatHeaderButton",
                                                toolTip="Start a new conversation")
        self.new_button.clicked.connect(self.reset)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(10, 6, 6, 6)
        header.addWidget(self.selection_label, 1)
        header.addWidget(self.new_button)

        # --- messages
        self.page = QtWidgets.QWidget(objectName="chatPage")
        self.list = QtWidgets.QVBoxLayout(self.page)
        self.list.setContentsMargins(12, 12, 12, 12)
        self.list.setSpacing(14)
        self.welcome = self._make_welcome()
        self.list.addWidget(self.welcome)
        self.list.addStretch(1)
        self.scroll = _FitWidthScrollArea(objectName="chatScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll.setWidget(self.page)
        self.scroll.verticalScrollBar().rangeChanged.connect(self._follow)

        # --- composer: a rounded box with the input and a round send button
        composer = QtWidgets.QFrame(objectName="chatComposer")
        self.input = QtWidgets.QPlainTextEdit(objectName="chatInput")
        self.input.setPlaceholderText("Message Claude…")
        self.input.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.input.setFixedHeight(64)
        self.input.installEventFilter(self)
        self.ask_button = QtWidgets.QToolButton(objectName="chatSend", toolTip="Send (Enter)")
        self.ask_button.setProperty("ewald_icon", "send")
        self.ask_button.setProperty("ewald_icon_color", "on_accent")
        self.ask_button.setIcon(icon("send", TOKENS["on_accent"]))
        self.ask_button.setIconSize(QtCore.QSize(16, 16))
        self.ask_button.setFixedSize(30, 30)
        self.ask_button.clicked.connect(self.ask)
        cl = QtWidgets.QHBoxLayout(composer)
        cl.setContentsMargins(10, 6, 6, 6)
        cl.addWidget(self.input, 1)
        cl.addWidget(self.ask_button, 0, QtCore.Qt.AlignBottom)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self.input, activated=self.ask)
        hint = QtWidgets.QLabel("Claude can make mistakes; numbers come from ewald's tools.", objectName="chatHint",
                                alignment=QtCore.Qt.AlignCenter)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 8)
        lay.setSpacing(0)
        lay.addLayout(header)
        lay.addWidget(self.scroll, 1)
        bottom = QtWidgets.QVBoxLayout()
        bottom.setContentsMargins(10, 6, 10, 0)
        bottom.setSpacing(4)
        bottom.addWidget(composer)
        bottom.addWidget(hint)
        lay.addLayout(bottom)
        self.update_selection()

    def _make_welcome(self):
        w = QtWidgets.QWidget(objectName="chatWelcome")
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 24, 0, 0)
        lay.setSpacing(8)
        avatar = QtWidgets.QLabel("C", objectName="chatAvatarLarge", alignment=QtCore.Qt.AlignCenter)
        avatar.setFixedSize(44, 44)
        lay.addWidget(avatar, 0, QtCore.Qt.AlignHCenter)
        title = "Ask Claude about this notebook" if self.notebook is not None else "Ask Claude about this structure"
        lay.addWidget(QtWidgets.QLabel(title, objectName="chatWelcomeTitle", alignment=QtCore.Qt.AlignCenter))
        sub = QtWidgets.QLabel("Pick atoms in the viewport (Shift+click adds more) to ask about them; "
                               "steps Claude adds show up in the pipeline.", objectName="chatHint",
                               alignment=QtCore.Qt.AlignCenter, wordWrap=True)
        lay.addWidget(sub)
        self.starters = []
        for text in STARTERS:
            b = QtWidgets.QPushButton(text, objectName="chatStarter")
            b.clicked.connect(lambda _=False, t=text: self.ask(t))
            lay.addWidget(b, 0, QtCore.Qt.AlignHCenter)
            self.starters.append(b)
        return w

    def eventFilter(self, obj, event):
        if obj is self.input and event.type() == QtCore.QEvent.KeyPress \
                and event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) \
                and not event.modifiers() & (QtCore.Qt.ShiftModifier | QtCore.Qt.ControlModifier):
            self.ask()
            return True
        return super().eventFilter(obj, event)

    # ---------------------------------------------------------------- state
    def update_selection(self):
        """Refresh the 'N atoms selected' line (the main window calls this on every pick)."""
        try:
            _pipe, _frame, indices = self.context()
        except Exception:
            indices = []
        n = len(indices)
        self.selection_label.setText(f"{n} atom{'s' if n != 1 else ''} selected" if n
                                     else "Nothing picked: asks about the frame")
        self.selection_label.setProperty("picked", bool(n))
        self.selection_label.style().unpolish(self.selection_label)
        self.selection_label.style().polish(self.selection_label)

    def busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def text(self) -> str:
        """The conversation as plain text (user, Claude and notes, code included)."""
        return "\n\n".join(f"{m.role}: {m.text}" + (f" ({m.meta})" if m.meta else "") for m in self.messages)

    def _add(self, message: Message) -> Message:
        self.welcome.hide()
        self.messages.append(message)
        self.list.insertWidget(self.list.count() - 1, message)
        return message

    def _note(self, text):
        self._add(Message("note", text))

    def _follow(self, _lo, hi):
        if self.busy() or self._pending is not None or self.messages:
            self.scroll.verticalScrollBar().setValue(hi)

    def reset(self):
        if self.busy():
            return
        self.agent = None
        self._sent_notebook = None
        for m in self.messages:
            m.setParent(None)
            m.deleteLater()
        self.messages.clear()
        self.snippets.clear()
        self.welcome.show()

    def insert_snippet(self, index: int):
        """Insert the ``index``-th Python block of the conversation as a notebook cell."""
        self.insertRequested.emit(self.snippets[index])

    def _notebook_prompt(self, question: str) -> str:
        """``question``, preceded by the notebook's code when it changed since the last question."""
        if self.notebook is None:
            return question
        try:
            code = self.notebook()
        except Exception:
            code = None
        if not code or not code.strip() or code == self._sent_notebook:
            return question
        self._sent_notebook = code
        if len(code) > _MAX_NOTEBOOK:
            code = code[:_MAX_NOTEBOOK] + "\n# … (notebook truncated)\n"
        return f"The notebook now reads:\n```python\n{code}```\n\n{question}"

    # ---------------------------------------------------------------- asking
    def ask(self, question: str | None = None):
        """Send ``question`` (default: the input box) about the current selection."""
        if self.busy():
            return
        question = (question if isinstance(question, str) else self.input.toPlainText()).strip()
        if not question:
            return
        pipeline, frame, indices = self.context()
        if pipeline is None and self.notebook is None:
            self._note("*Open a file first.*")
            return
        if self.agent is None or self._agent_pipeline is not pipeline:
            try:
                self.agent = self.agent_factory(pipeline, self._on_event_threadsafe)
            except Exception as exc:
                self._add(Message("user", question))
                m = self._add(Message("assistant"))
                m.set_answer(f"**Claude is not available.** {_SETUP_HINT}\n\n`{type(exc).__name__}: {exc}`",
                             error=True)
                return
            self._agent_pipeline = pipeline
            self._sent_notebook = None
        indices = [int(i) for i in indices]
        what = f"{len(indices)} picked atom{'s' if len(indices) != 1 else ''} · frame {frame}" if indices \
            else f"frame {frame}" if pipeline is not None else "no structure"
        self._add(Message("user", question, what))
        self.input.clear()
        self._modifiers_before = list(pipeline.modifiers) if pipeline is not None else []
        agent = self.agent
        prompt = self._notebook_prompt(question)
        if indices and pipeline is not None:
            job = lambda: agent.ask_about_selection(prompt, indices, frame=frame)  # noqa: E731
        else:
            if pipeline is not None:
                agent.toolset.frame = frame
            agent.toolset.selection = None
            job = lambda: agent.ask(prompt)  # noqa: E731
        self._pending = self._add(Message("assistant", insert=self._insert))
        self.worker = _Worker(job, self)
        self.worker.event.connect(self._on_event)
        self.worker.answered.connect(self._on_answer)
        self.worker.failed.connect(self._on_failed)
        self.ask_button.setEnabled(False)
        self.worker.start()

    def _insert(self, code):
        self.insertRequested.emit(code)

    def _on_event_threadsafe(self, kind, payload):
        # Agent.on_event runs on the worker thread; hop to the GUI thread through the worker's signal
        if self.worker is not None:
            self.worker.event.emit(kind, payload)

    def _on_event(self, kind, payload):
        if kind == "tool_call" and self._pending is not None:
            self._pending.add_step(payload["name"])

    def _finish(self):
        self.ask_button.setEnabled(True)
        self._pending = None
        pipeline = self._agent_pipeline if self.agent else None
        if pipeline is not None and list(pipeline.modifiers) != self._modifiers_before:
            self.pipelineChanged.emit()

    def _on_answer(self, text):
        if self._pending is not None:
            self._pending.set_answer(text)
            self.snippets += [b.code for b in self._pending.code_blocks if b.insert_button is not None]
        self._finish()

    def _on_failed(self, message):
        self._sent_notebook = None  # the notebook may not have reached Claude: send it again
        low = message.lower()
        hint = f" {_SETUP_HINT}" if any(w in low for w in ("anthropic", "api_key", "authentication")) else ""
        if self._pending is not None:
            self._pending.set_answer(f"**Error.** `{message}`{hint}", error=True)
        self._finish()
