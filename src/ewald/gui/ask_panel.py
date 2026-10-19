"""Ask Claude panel: questions about the picked atoms (or the whole frame), answered in a side chat.

Pick atoms in the viewport (Shift+click with the pick tool adds more), type a question and
press Ask. The panel sends ewald's description of the picked atoms and their surroundings
plus a close-up render (:meth:`ewald.llm.Agent.ask_about_selection`); Claude can call the
usual tools, and modifiers it adds show up in the pipeline. With nothing picked, the
question goes about the whole pipeline. Claude runs in a worker thread, so the viewer stays
responsive. Needs ``pip install ewald[llm]`` and an API key (``ANTHROPIC_API_KEY``).
"""
from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from .theme import TOKENS

_SETUP_HINT = ("Claude needs the `anthropic` package (`pip install ewald[llm]`) and an API key "
               "(`ANTHROPIC_API_KEY`).")


def _default_agent(pipeline, on_event):
    from ..llm import Agent

    return Agent.from_pipeline(pipeline, on_event=on_event)


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
    """

    pipelineChanged = QtCore.Signal()

    def __init__(self, context: Callable[[], tuple], agent_factory=_default_agent, parent=None):
        super().__init__(parent)
        self.context = context
        self.agent_factory = agent_factory
        self.agent = None
        self.worker: _Worker | None = None
        self.entries: list[str] = []
        self._pending_tools: list[str] = []

        self.transcript = QtWidgets.QTextBrowser(openExternalLinks=True)
        self.transcript.setPlaceholderText("Pick atoms in the viewport (Shift+click adds more), then ask, e.g. "
                                           "\"What are these atoms?\" or \"Why is the coordination low here?\"")
        self.selection_label = QtWidgets.QLabel()
        self.selection_label.setStyleSheet(f"color: {TOKENS['pick']};")
        self.input = QtWidgets.QPlainTextEdit()
        self.input.setPlaceholderText("Ask about the picked atoms…  (Ctrl+Enter)")
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
        self.entries.clear()
        self._render()

    # ---------------------------------------------------------------- asking
    def ask(self, question: str | None = None):
        """Send ``question`` (default: the input box) about the current selection."""
        if self.busy():
            return
        question = (question if isinstance(question, str) else self.input.toPlainText()).strip()
        if not question:
            return
        pipeline, frame, indices = self.context()
        if pipeline is None:
            self.entries.append("*Open a file first.*")
            self._render()
            return
        if self.agent is None or self.agent.toolset.pipeline is not pipeline:
            try:
                self.agent = self.agent_factory(pipeline, self._on_event_threadsafe)
            except Exception as exc:
                self.entries.append(f"**Claude is not available.** {_SETUP_HINT}\n\n`{type(exc).__name__}: {exc}`")
                self._render()
                return
        indices = [int(i) for i in indices]
        what = f"{len(indices)} picked atom{'s' if len(indices) != 1 else ''}, frame {frame}" if indices \
            else f"frame {frame}"
        self.entries.append(f"**You** ({what})\n\n{question}")
        self.input.clear()
        self._modifiers_before = list(pipeline.modifiers)
        agent = self.agent
        if indices:
            job = lambda: agent.ask_about_selection(question, indices, frame=frame)  # noqa: E731
        else:
            agent.toolset.frame = frame
            agent.toolset.selection = None
            job = lambda: agent.ask(question)  # noqa: E731
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
        pipeline = self.agent.toolset.pipeline if self.agent else None
        if pipeline is not None and list(pipeline.modifiers) != self._modifiers_before:
            self.pipelineChanged.emit()

    def _on_answer(self, text):
        self.entries.append(f"**Claude**\n\n{text}")
        self._finish()
        self._render()

    def _on_failed(self, message):
        hint = f" {_SETUP_HINT}" if "anthropic" in message.lower() or "api_key" in message.lower() else ""
        self.entries.append(f"**Error.** `{message}`{hint}")
        self._finish()
        self._render()
