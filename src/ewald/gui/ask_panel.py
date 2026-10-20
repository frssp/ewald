"""Ask AI panel: questions about the picked atoms (or the whole frame), answered in a side chat.

The conversation reads like a chat app (Chainlit-style): the user's messages in bubbles on the
right, the model's answers on the left under its avatar, the tools it ran as small step chips,
and code blocks as framed, highlighted code with Copy and Insert buttons. In the notebook
layout this is the AI chat side bar (the AI button in the activity bar, left of the
notebook): each question also carries the current notebook's code when it changed, and every
Python block in an answer can be inserted as a notebook cell.

Pick atoms in the viewport (Shift+click with the pick tool adds more), type a question and
press Ask. The panel sends ewald's description of the picked atoms and their surroundings
plus a close-up render (:meth:`ewald.llm.Agent.ask_about_selection`); the model can call the
usual tools, and modifiers it adds show up in the pipeline. With nothing picked, the
question goes about the whole pipeline. The model runs in a worker thread, so the viewer stays
responsive. Whichever provider Preferences > AI names answers (Claude by default; OpenAI,
Gemini, a local OpenAI-compatible server or a plugin's); the provider packages (anthropic, openai)
come with ewald; it usually needs an API key.
"""
from __future__ import annotations

import inspect
import re
import threading
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from .busy import BusyLabel
from .icons import icon
from .code_editor import CodeEditor
from .pipeline_panel import FitWidthScrollArea as _FitWidthScrollArea
from . import workers
from .theme import TOKENS

_SETUP_HINT = "Choose the AI provider and enter its API key under Edit › Preferences › AI."


NOTEBOOK_NOTE = """

You are the chat side bar of ewald's notebook window. The user's notebook is a percent-format \
Python file (`# %%` cells, one namespace) where `pipe` is the pipeline: step cells are one \
`pipe.add(Modifier(...))` (modifiers from `ewald.modifiers`, imported as `mods` or by name), \
view cells are one `show(pipe, ...)`, and `load(path)` opens a file. When the user's message \
starts with the notebook's code, that is its current state. {code_cells} Write each ```python \
block as one complete cell that runs in that namespace; prefer `pipe.add(...)` steps over ad-hoc \
loops. When the user asks you to \
change the notebook itself, use the cell tools: list_cells and read_cell (sources and outputs), \
insert_cell, edit_cell (step parameters through params), move_cell and delete_cell. Cells you \
write are not run until the user runs them, unless run_cell is offered; add_modifier, set_view, \
load_structure and export_file write their own cell and run it. Say which cells you changed."""

#: How the note above describes the ```python blocks of an answer (Preferences › AI › Code in answers).
CODE_CELLS_NOTE = {
    "insert": "Every ```python block of your answer is added to the notebook as a new cell (not run), in "
              "order, so put only code meant for the notebook in ```python blocks (other code, such as "
              "a one-line illustration, in a ```text block), and do not also insert_cell the same code.",
    "button": "Every ```python block of your answer gets an \"Insert as notebook cell\" button.",
}

_MAX_NOTEBOOK = 20000  # characters of notebook code sent with a question
_CODE_BLOCK = re.compile(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", re.S)
_ANY_BLOCK = re.compile(r"```([\w+.-]*)[ \t]*\n(.*?)```", re.S)
_PYTHON = ("", "python", "py", "python3")
STARTERS = ("Describe this structure", "What analysis should I run next?", "Explain the notebook's last cell")


def _default_agent(pipeline, on_event, **kwargs):
    from ..llm import Agent

    return Agent.from_pipeline(pipeline, on_event=on_event, **kwargs)


def agent_from_settings(get_settings: Callable[[], object], notebook: bool = False):
    """Agent factory using Preferences › AI (provider, model, connection, effort), read when a
    chat starts; ``notebook`` tells the model about the notebook it sits beside. The factory's
    ``assistant()`` names the provider's assistant for the chat ("Claude", "Gemini", ...)."""
    def factory(pipeline, on_event):
        from ..llm.providers import from_settings

        kwargs = from_settings(get_settings())
        if notebook:
            from ..llm.agent import SYSTEM_PROMPT

            mode = code_cells_mode(get_settings())
            kwargs["system"] = SYSTEM_PROMPT + NOTEBOOK_NOTE.format(code_cells=CODE_CELLS_NOTE[mode])
        return _default_agent(pipeline, on_event, **kwargs)

    def assistant() -> str:
        from ..llm import providers

        try:
            return providers.get(get_settings()["llm.provider"]).assistant
        except Exception:
            return "AI"

    factory.assistant = assistant
    return factory


def code_cells_mode(settings) -> str:
    """'insert' (code blocks of answers become cells at once) or 'button' (+ Insert as cell)."""
    try:
        mode = settings["llm.code_cells"]
    except Exception:
        return "insert"
    return mode if mode in CODE_CELLS_NOTE else "insert"


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
        self.cancel = threading.Event()   # Stop: the agent ends the exchange at its next step

    def run(self):
        from ..llm.agent import Cancelled

        try:
            self.answered.emit(self.job())
        except Cancelled:
            pass
        except Exception as exc:  # shown in the chat, never raised into Qt
            self.failed.emit(f"{type(exc).__name__}: {exc}")


_STOPPED: set = set()   # stopped chat workers still waiting for their last request


def _safe(fn) -> str | None:
    """``fn()``, or None when there is no ``fn`` or it fails."""
    try:
        return fn() if fn is not None else None
    except Exception:
        return None


def _call(fn, *args, cancel, **kwargs):
    """``fn(*args, cancel=cancel, **kwargs)``, without ``cancel`` for agents that take none (plugins, tests)."""
    try:
        takes = "cancel" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        takes = False
    return fn(*args, cancel=cancel, **kwargs) if takes else fn(*args, **kwargs)


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

    def mark_inserted(self):
        """The block went into the notebook already (Code in answers: insert)."""
        if self.insert_button is not None:
            self.insert_button.setText("✓ Inserted")
            self.insert_button.setToolTip("Added to the notebook as a cell; click to add it again")

    def copy(self):
        QtWidgets.QApplication.clipboard().setText(self.code)
        self.copy_button.setText("Copied")
        QtCore.QTimer.singleShot(1500, self.copy_button, lambda: self.copy_button.setText("Copy"))


class Message(QtWidgets.QWidget):
    """One chat message: ``role`` is "user" (bubble on the right), "assistant" (avatar and
    answer on the left) or "note" (a centered, muted line)."""

    def __init__(self, role: str, text: str = "", meta: str = "", insert=None, parent=None, author: str = "AI"):
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
        avatar = QtWidgets.QLabel(author[:1].upper() or "A", objectName="chatAvatar", alignment=QtCore.Qt.AlignCenter)
        avatar.setFixedSize(26, 26)
        row.addWidget(avatar, 0, QtCore.Qt.AlignTop)
        self.body = QtWidgets.QVBoxLayout()
        self.body.setContentsMargins(0, 2, 0, 0)
        self.body.setSpacing(6)
        self.body.addWidget(QtWidgets.QLabel(author, objectName="chatAuthor"))
        self.steps = QtWidgets.QVBoxLayout()
        self.steps.setSpacing(2)
        self.body.addLayout(self.steps)
        self.status = BusyLabel("Thinking…", 14)
        self.status.label.setObjectName("chatThinking")
        self.body.addWidget(self.status)
        row.addLayout(self.body, 1)
        self.insert = insert
        if text:
            self.set_answer(text)

    def add_step(self, name: str):
        """A tool the model runs: a chip that turns from running to done."""
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
    """Chat with the AI model about the current selection and notebook.

    ``context()`` returns ``(pipeline, frame, indices)`` for the moment Ask is pressed
    (``indices`` may be empty). ``agent_factory(pipeline, on_event)`` builds the agent; tests
    pass a fake; its ``assistant()``, when it has one, names who answers.
    ``pipelineChanged`` fires after an answer when the model changed the modifiers.
    ``notebook()``, when given, returns the current notebook's code (or None): it goes with a
    question whenever it changed, and Python blocks in answers offer Insert as cell
    (``insertRequested(code)``). ``notebook_access`` (a :class:`ewald.llm.notebook_tools.NotebookAccess`)
    gives the agent's toolset the notebook cell tools. ``auto_insert()``, when it returns True, puts
    the Python blocks of each answer into the notebook at once (unless the model wrote cells with
    the cell tools in that answer). ``errors()``, when given, returns the errors of the notebook's
    failed cells (or None); they go with a question whenever they changed. While an answer is on its
    way the send button is Stop (also Esc).
    """

    pipelineChanged = QtCore.Signal()
    insertRequested = QtCore.Signal(str)

    def __init__(self, context: Callable[[], tuple], agent_factory=_default_agent, parent=None,
                 notebook: Callable[[], str | None] | None = None, notebook_access=None,
                 auto_insert: Callable[[], bool] | None = None, errors: Callable[[], str | None] | None = None):
        super().__init__(parent, objectName="chatPanel")
        self.context = context
        self.agent_factory = agent_factory
        self.notebook = notebook
        self.notebook_access = notebook_access
        self.auto_insert = auto_insert
        self.errors = errors
        self._sent_errors: str | None = None
        self._stopped: list[_Worker] = []   # stopped workers still finishing their last request
        self._tools_used: list[str] = []
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
        self.input.setPlaceholderText("Ask AI…")
        self.input.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.input.setFixedHeight(64)
        self.input.installEventFilter(self)
        self.ask_button = QtWidgets.QToolButton(objectName="chatSend", toolTip="Send (Enter)")
        self.ask_button.setProperty("ewald_icon", "send")
        self.ask_button.setProperty("ewald_icon_color", "on_accent")
        self.ask_button.setIcon(icon("send", TOKENS["on_accent"]))
        self.ask_button.setIconSize(QtCore.QSize(16, 16))
        self.ask_button.setFixedSize(30, 30)
        self.ask_button.clicked.connect(self._send_or_stop)
        cl = QtWidgets.QHBoxLayout(composer)
        cl.setContentsMargins(10, 6, 6, 6)
        cl.addWidget(self.input, 1)
        cl.addWidget(self.ask_button, 0, QtCore.Qt.AlignBottom)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self.input, activated=self.ask)
        hint = QtWidgets.QLabel("AI can make mistakes; numbers come from ewald's tools.", objectName="chatHint",
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
        avatar = QtWidgets.QLabel("AI", objectName="chatAvatarLarge", alignment=QtCore.Qt.AlignCenter)
        avatar.setFixedSize(44, 44)
        lay.addWidget(avatar, 0, QtCore.Qt.AlignHCenter)
        title = "Ask AI about this notebook" if self.notebook is not None else "Ask AI about this structure"
        lay.addWidget(QtWidgets.QLabel(title, objectName="chatWelcomeTitle", alignment=QtCore.Qt.AlignCenter))
        sub = QtWidgets.QLabel("Pick atoms in the viewport (Shift+click adds more) to ask about them; "
                               "steps the AI adds show up in the pipeline.", objectName="chatHint",
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
                and event.key() == QtCore.Qt.Key_Escape and self.busy():
            self.stop()
            return True
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
            pipe, _frame, indices = self.context()
        except Exception:
            pipe, indices = None, []
        for b in self.starters[:2]:                 # about the structure: only once there is one
            b.setEnabled(pipe is not None)
            b.setToolTip("" if pipe is not None else "Open a structure first")
        n = len(indices)
        self.selection_label.setText(f"{n} atom{'s' if n != 1 else ''} selected" if n
                                     else "Nothing picked: asks about the frame")
        self.selection_label.setProperty("picked", bool(n))
        self.selection_label.style().unpolish(self.selection_label)
        self.selection_label.style().polish(self.selection_label)

    def assistant(self) -> str:
        """Who answers, as the chat names it: the provider's assistant ("Claude", "Gemini", ...)."""
        name = getattr(self.agent_factory, "assistant", None)
        try:
            return name() if callable(name) else "AI"
        except Exception:
            return "AI"

    def busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def text(self) -> str:
        """The conversation as plain text (user, assistant and notes, code included)."""
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
        self._sent_notebook = self._sent_errors = None
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
        """``question``, preceded by the notebook's code and the errors of its failed cells, each
        when it changed since the last question."""
        parts = []
        code = _safe(self.notebook)
        if code and code.strip() and code != self._sent_notebook:
            self._sent_notebook = code
            if len(code) > _MAX_NOTEBOOK:
                code = code[:_MAX_NOTEBOOK] + "\n# … (notebook truncated)\n"
            parts.append(f"The notebook now reads:\n```python\n{code}```")
        errors = _safe(self.errors) or ""
        if errors != (self._sent_errors or ""):
            self._sent_errors = errors
            parts.append(f"These notebook cells failed when they last ran:\n```text\n{errors}```" if errors
                         else "The notebook's cells no longer show errors.")
        return "\n\n".join(parts + [question])

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
                m = self._add(Message("assistant", author=self.assistant()))
                m.set_answer(f"**The AI is not available.** {_SETUP_HINT}\n\n`{type(exc).__name__}: {exc}`",
                             error=True)
                return
            self._agent_pipeline = pipeline
            self._sent_notebook = self._sent_errors = None
            toolset = getattr(self.agent, "toolset", None)
            if self.notebook_access is not None and hasattr(toolset, "attach_notebook"):
                toolset.attach_notebook(self.notebook_access)
        indices = [int(i) for i in indices]
        what = f"{len(indices)} picked atom{'s' if len(indices) != 1 else ''} · frame {frame}" if indices \
            else f"frame {frame}" if pipeline is not None else "no structure"
        self._add(Message("user", question, what))
        self.input.clear()
        self._modifiers_before = list(pipeline.modifiers) if pipeline is not None else []
        agent = self.agent
        prompt = self._notebook_prompt(question)
        if indices and pipeline is not None:
            job = lambda: _call(agent.ask_about_selection, prompt, indices, frame=frame,  # noqa: E731
                                cancel=worker.cancel)
        else:
            if pipeline is not None:
                agent.toolset.frame = frame
            agent.toolset.selection = None
            job = lambda: _call(agent.ask, prompt, cancel=worker.cancel)  # noqa: E731
        self._tools_used = []
        self._pending = self._add(Message("assistant", insert=self._insert, author=self.assistant()))
        worker = self.worker = workers.detach(_Worker(job))  # may outlive the panel
        self.worker.event.connect(self._on_event)
        self.worker.answered.connect(self._on_answer)
        self.worker.failed.connect(self._on_failed)
        self._set_stop_button(True)
        self.worker.start()

    def _insert(self, code):
        self.insertRequested.emit(code)

    def _on_event_threadsafe(self, kind, payload):
        # Agent.on_event runs on the worker thread; hop to the GUI thread through the worker's signal
        if self.worker is not None:
            self.worker.event.emit(kind, payload)

    def _on_event(self, kind, payload):
        if kind == "tool_call" and self._pending is not None:
            self._tools_used.append(payload["name"])
            self._pending.add_step(payload["name"])

    def _send_or_stop(self):
        if self.busy():
            self.stop()
        else:
            self.ask()

    def _set_stop_button(self, on: bool):
        """The send button is Stop while an answer is on its way."""
        name = "stop" if on else "send"
        self.ask_button.setProperty("ewald_icon", name)
        self.ask_button.setIcon(icon(name, TOKENS["on_accent"]))
        self.ask_button.setToolTip("Stop (Esc)" if on else "Send (Enter)")

    def stop(self):
        """Stop the answer on its way: the chat is free at once, the model's request still on the
        wire is dropped when it returns, and the conversation forgets the stopped question."""
        worker = self.worker
        if worker is None or not worker.isRunning():
            return
        worker.cancel.set()
        for signal in (worker.event, worker.answered, worker.failed):
            signal.disconnect()
        # it may outlive the panel (closing the window): keep it alive, unparented, until it ends
        worker.setParent(None)
        self._stopped.append(worker)
        _STOPPED.add(worker)
        worker.finished.connect(lambda: (_STOPPED.discard(worker),
                                         self._stopped.remove(worker) if worker in self._stopped else None))
        self.worker = None
        self._sent_notebook = self._sent_errors = None   # the stopped question's notebook was forgotten too
        if self._pending is not None:
            self._pending.set_answer("*Stopped.*")
        self._finish()

    def _finish(self):
        self._set_stop_button(False)
        self._pending = None
        pipeline = self._agent_pipeline if self.agent else None
        now = getattr(getattr(self.agent, "toolset", None), "pipeline", pipeline)
        if now is not None and now is not pipeline:   # a load cell the AI ran made a new pipe:
            self._agent_pipeline = now                  # the conversation goes on with it
            self.pipelineChanged.emit()
        elif pipeline is not None and list(pipeline.modifiers) != self._modifiers_before:
            self.pipelineChanged.emit()

    def _on_answer(self, text):
        if self._pending is not None:
            self._pending.set_answer(text)
            blocks = [b for b in self._pending.code_blocks if b.insert_button is not None]
            self.snippets += [b.code for b in blocks]
            wrote_cells = any(t in ("insert_cell", "edit_cell") for t in self._tools_used)
            if blocks and not wrote_cells and self.auto_insert is not None and self.auto_insert():
                for b in blocks:
                    self.insertRequested.emit(b.code)
                    b.mark_inserted()
        self._finish()

    def _on_failed(self, message):
        self._sent_notebook = self._sent_errors = None  # they may not have reached the model: send again
        low = message.lower()
        hint = f" {_SETUP_HINT}" if any(w in low for w in ("anthropic", "openai", "api_key", "api key",
                                                            "authentication", "no ai provider")) else ""
        if self._pending is not None:
            self._pending.set_answer(f"**Error.** `{message}`{hint}", error=True)
        self._finish()
