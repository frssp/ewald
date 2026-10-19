"""Ask Claude panel: questions about the picked atoms (or the whole frame), answered in a side chat.

In the notebook layout this is the AI chat side bar (the Claude button in the activity bar,
beside the notebook): each question also carries the current notebook's code when it changed,
and every Python block in an answer gets an "Insert as notebook cell" link.

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


class AskPanel(QtWidgets.QWidget):
    """Chat with Claude about the current selection.

    ``context()`` returns ``(pipeline, frame, indices)`` for the moment Ask is pressed
    (``indices`` may be empty). ``agent_factory(pipeline, on_event)`` builds the agent; tests
    pass a fake. ``pipelineChanged`` fires after an answer when Claude changed the modifiers.
    ``notebook()``, when given, returns the current notebook's code (or None): it goes with a
    question whenever it changed, and answers offer their code blocks through
    ``insertRequested(code)``.
    """

    pipelineChanged = QtCore.Signal()
    insertRequested = QtCore.Signal(str)

    def __init__(self, context: Callable[[], tuple], agent_factory=_default_agent, parent=None,
                 notebook: Callable[[], str | None] | None = None):
        super().__init__(parent)
        self.context = context
        self.agent_factory = agent_factory
        self.notebook = notebook
        self.agent = None
        self._agent_pipeline = None
        self._sent_notebook: str | None = None
        self.snippets: list[str] = []  # code blocks of the answers, by link number
        self.worker: _Worker | None = None
        self.entries: list[str] = []
        self._pending_tools: list[str] = []

        self.transcript = QtWidgets.QTextBrowser()
        self.transcript.setOpenLinks(False)
        self.transcript.anchorClicked.connect(self._link)
        if notebook is not None:
            self.transcript.setPlaceholderText(
                "Chat with Claude about this notebook and structure, e.g. \"Add a coordination analysis "
                "and color by it\" or \"Why is the RDF peak split?\". Code in answers can be inserted as "
                "notebook cells. Picked atoms go with the question.")
        else:
            self.transcript.setPlaceholderText("Pick atoms in the viewport (Shift+click adds more), then ask, e.g. "
                                               "\"What are these atoms?\" or \"Why is the coordination low here?\"")
        self.selection_label = QtWidgets.QLabel()
        self.selection_label.setStyleSheet(f"color: {TOKENS['pick']};")
        self.input = QtWidgets.QPlainTextEdit()
        self.input.setPlaceholderText("Ask Claude…  (Ctrl+Enter)" if notebook is not None
                                      else "Ask about the picked atoms…  (Ctrl+Enter)")
        self.input.setMaximumHeight(90)
        self.ask_button = QtWidgets.QPushButton("Ask")
        self.ask_button.setDefault(True)
        self.ask_button.clicked.connect(self.ask)
        self.new_button = QtWidgets.QPushButton("New chat")
        self.new_button.clicked.connect(self.reset)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self.input, activated=self.ask)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.selection_label, 1)
        buttons.addWidget(self.new_button)
        buttons.addWidget(self.ask_button)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.transcript, 1)
        lay.addWidget(self.input)
        lay.addLayout(buttons)
        self.update_selection()

    # ---------------------------------------------------------------- state
    def update_selection(self):
        """Refresh the 'N atoms picked' line (the main window calls this on every pick)."""
        try:
            _pipe, _frame, indices = self.context()
        except Exception:
            indices = []
        n = len(indices)
        self.selection_label.setText(f"{n} atom{'s' if n != 1 else ''} selected" if n else "Nothing picked: asks about the frame")

    def busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def _render(self):
        self.transcript.setMarkdown("\n\n".join(self.entries + [f"*{t}…*" for t in self._pending_tools[-1:]]))
        bar = self.transcript.verticalScrollBar()
        bar.setValue(bar.maximum())

    def reset(self):
        if self.busy():
            return
        self.agent = None
        self._sent_notebook = None
        self.entries.clear()
        self.snippets.clear()
        self._render()

    def _link(self, url: QtCore.QUrl):
        if url.scheme() == "ewald-insert":
            try:
                self.insertRequested.emit(self.snippets[int(url.path())])
            except (ValueError, IndexError):
                pass
        else:
            QtGui.QDesktopServices.openUrl(url)

    def _with_insert_links(self, text: str) -> str:
        """``text`` with an "Insert as notebook cell" link after each Python block."""
        if self.notebook is None:
            return text

        def link(m):
            code = m.group(1)
            if not code.strip():
                return m.group(0)
            self.snippets.append(code.rstrip() + "\n")
            return f"{m.group(0)}\n\n[+ Insert as notebook cell](ewald-insert:{len(self.snippets) - 1})\n"
        return _CODE_BLOCK.sub(link, text)

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
            self.entries.append("*Open a file first.*")
            self._render()
            return
        if self.agent is None or self._agent_pipeline is not pipeline:
            try:
                self.agent = self.agent_factory(pipeline, self._on_event_threadsafe)
            except Exception as exc:
                self.entries.append(f"**Claude is not available.** {_SETUP_HINT}\n\n`{type(exc).__name__}: {exc}`")
                self._render()
                return
            self._agent_pipeline = pipeline
            self._sent_notebook = None
        indices = [int(i) for i in indices]
        what = f"{len(indices)} picked atom{'s' if len(indices) != 1 else ''}, frame {frame}" if indices \
            else f"frame {frame}" if pipeline is not None else "no structure"
        self.entries.append(f"**You** ({what})\n\n{question}")
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
        self.worker = _Worker(job, self)
        self.worker.event.connect(self._on_event)
        self.worker.answered.connect(self._on_answer)
        self.worker.failed.connect(self._on_failed)
        self.ask_button.setEnabled(False)
        self.ask_button.setText("Thinking…")
        self._pending_tools = ["Thinking"]
        self._render()
        self.worker.start()

    def _on_event_threadsafe(self, kind, payload):
        # Agent.on_event runs on the worker thread; hop to the GUI thread through the worker's signal
        if self.worker is not None:
            self.worker.event.emit(kind, payload)

    def _on_event(self, kind, payload):
        if kind == "tool_call":
            self._pending_tools.append(f"Running {payload['name']}")
            self._render()

    def _finish(self):
        self.ask_button.setEnabled(True)
        self.ask_button.setText("Ask")
        self._pending_tools = []
        pipeline = self._agent_pipeline if self.agent else None
        if pipeline is not None and list(pipeline.modifiers) != self._modifiers_before:
            self.pipelineChanged.emit()

    def _on_answer(self, text):
        self.entries.append(f"**Claude**\n\n{self._with_insert_links(text)}")
        self._finish()
        self._render()

    def _on_failed(self, message):
        self._sent_notebook = None  # the notebook may not have reached Claude: send it again
        low = message.lower()
        hint = f" {_SETUP_HINT}" if any(w in low for w in ("anthropic", "api_key", "authentication")) else ""
        self.entries.append(f"**Error.** `{message}`{hint}")
        self._finish()
        self._render()
