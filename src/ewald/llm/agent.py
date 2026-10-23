"""An AI agent that analyzes structures through a :class:`~ewald.llm.Toolset`.

>>> from ewald.llm import Agent
>>> agent = Agent.from_file("XDATCAR")                      # Claude (the default provider)
>>> agent = Agent.from_file("XDATCAR", provider="openai")   # or OpenAI, Gemini, a local server, ...
>>> print(agent.ask("What happens in this run? Show me the most interesting frame."))
>>> print(agent.toolset.to_script())     # the analysis as a reproducible script

Needs the provider's package (``pip install anthropic`` for Claude, ``openai`` for the others)
and a key (Preferences > AI, or the provider's environment variable; see
:mod:`ewald.llm.credentials` and :mod:`ewald.llm.providers`). The loop is written out by hand
over a :class:`~ewald.llm.chat.Chat` rather than an SDK tool runner, so tool results can carry
rendered images, every provider runs the same loop, and the toolset stays shared with the MCP
server and GUI.
"""
from __future__ import annotations

import threading
from typing import Callable

import numpy as np

from .tools import Toolset

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """\
You are the analysis assistant inside ewald, an atomistic structure and trajectory \
workbench built around a Python notebook. The user is a computational materials \
scientist working with VASP and LAMMPS output.

You act on a live pipeline (source -> modifiers -> rendered frame) through tools. \
Every number you report must come from a tool result in this conversation; never \
estimate lattice constants, coordination numbers, diffusion coefficients or \
energies from memory. When a question can be answered by computing something, \
compute it. You change the structure, the view and every other result only through \
the notebook: each tool that changes something writes a cell and runs it (add_modifier a step \
cell, the same one Insert > Step makes; set_view the view cell, which the Display panel edits \
too; load_structure a load cell; export_file an export cell), so the notebook alone reproduces \
the analysis. Tools are not Python functions: code you show or write for a cell uses ewald's API \
(view(pipe, ...) for set_view, pipe.add(Modifier(...)) for add_modifier), never a tool call. \
Prefer pipeline modifiers over ad-hoc code. Before you write Python for the user \
(a notebook cell or a code block), call python_api once in the conversation and follow it: \
ewald's API looks like OVITO's but differs in places, and plugins add modifiers. In the GUI the Structure tab shows \
the result, and you answer in the AI chat side bar next to the notebook.

For trajectories, start with describe_trajectory and look at its events before \
reading individual frames. Render frames when the geometry matters and look at \
the image before describing it. Use Angstrom, fs, eV and g/cm^3; name frames by \
index. Be concise and specific; say what you checked and what remains uncertain, \
and suggest the next analysis when one is clearly worth doing."""


class Cancelled(Exception):
    """An :meth:`Agent.ask` was stopped through its ``cancel`` event."""


class Agent:
    """Conversation with an AI model over one ewald session.

    ``provider`` is a name from :mod:`ewald.llm.providers` ("anthropic", the default, "openai",
    "gemini", "openai-compatible" or a plugin's) or a ``Provider``; ``model`` defaults to the
    provider's first model and ``base_url`` to its endpoint. ``client`` defaults to the
    provider's client with the saved key; pass any object with the same methods
    (``beta.messages.create`` for Claude, ``chat.completions.create`` for the others) to test
    without network access. ``fallbacks=True`` enables Claude's server-side refusal fallback
    (another model continues if the request is declined). ``effort`` applies where the provider
    has such a setting. ``stream=True`` streams the answers: their text reaches ``on_event`` as
    ``("text", piece)`` events while it arrives (a ``"tool_call"`` event ends the text of a round
    that goes on with tools), and a Stop ends the request on the wire at once.
    """

    def __init__(self, toolset: Toolset | None = None, *, client=None, model: str | None = None,
                 provider=None, base_url: str | None = None,
                 effort: str = "high", max_tokens: int = 16000, max_turns: int = 30,
                 system: str = SYSTEM_PROMPT, fallbacks: bool = True,
                 on_event: Callable[[str, object], None] | None = None, stream: bool = False):
        from .providers import get

        self.toolset = toolset or Toolset()
        self.provider = get(provider)
        self.model = model or self.provider.default_model
        if client is None:
            client = self.provider.client(base_url=base_url, model=self.model)
        self.client = client
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.system = system
        self.fallbacks = fallbacks
        self.on_event = on_event or (lambda kind, payload: None)
        self.stream = stream
        self.chat = self.provider.chat(client, self.model, system=system, tools=self.toolset.definitions,
                                       max_tokens=max_tokens, effort=effort, fallbacks=fallbacks)
        self._lock = threading.Lock()

    @property
    def messages(self) -> list:
        """The conversation in the provider's own message format."""
        return self.chat.messages

    @classmethod
    def from_file(cls, path, **kwargs) -> "Agent":
        allow_code = kwargs.pop("allow_code", False)
        ts = Toolset(allow_code=allow_code)
        res = ts.call("load_structure", {"path": str(path)})
        if res.is_error:
            raise RuntimeError(res.content)
        return cls(ts, **kwargs)

    @classmethod
    def from_pipeline(cls, pipeline, **kwargs) -> "Agent":
        return cls(Toolset(pipeline, allow_code=kwargs.pop("allow_code", False), notebook=kwargs.pop("notebook", None)),
                   **kwargs)

    # ------------------------------------------------------------------ loop
    def _sync(self):
        """Settings changed on the agent after it was made (model, system, ...) reach the chat."""
        c = self.chat
        c.model, c.system, c.max_tokens = self.model, self.system, self.max_tokens
        if self.provider.effort:
            c.effort = self.effort
        c.options["fallbacks"] = self.fallbacks

    def ask(self, prompt: str, images: list[bytes] | None = None, cancel=None) -> str:
        """Send ``prompt`` (optionally with PNG images), run tools until the model answers, return its text.

        ``cancel`` (a :class:`threading.Event`) stops the exchange when set, from another thread: the
        request on the wire still finishes (a streamed one ends at its next piece), then no further
        tool runs or request goes out,
        :class:`Cancelled` is raised and the conversation is as it was before this question.
        Questions from several threads take turns."""
        with self._lock:
            start = len(self.chat.messages)
            try:
                return self._ask(prompt, images, cancel)
            except Cancelled:
                del self.chat.messages[start:]
                raise

    def _ask(self, prompt, images, cancel) -> str:
        def check():
            if cancel is not None and cancel.is_set():
                raise Cancelled("stopped")

        def text(piece):
            check()
            if piece:
                self.on_event("text", piece)

        check()
        self._sync()
        self.chat.user(prompt, images)
        for _ in range(self.max_turns):
            reply = self.chat.send(on_text=text) if self.stream else self.chat.send()
            check()
            if reply.stop == "refusal":
                return "[request declined by the model]"
            if reply.stop != "tool_use":
                text = reply.text
                self.on_event("answer", text)
                if reply.stop == "max_tokens":
                    text += "\n[truncated: max_tokens reached]"
                return text
            results = []
            for call in reply.calls:
                check()
                self.on_event("tool_call", {"name": call.name, "input": call.input})
                result = self.toolset.call(call.name, dict(call.input or {}))
                self.on_event("tool_result", {"name": call.name, "result": result})
                results.append((call, result))
            self.chat.tool_results(results)
        return f"[stopped after {self.max_turns} tool rounds]"

    def ask_about_selection(self, question: str, indices, frame: int | None = None, radius: float = 6.0,
                            image: bool = True, cancel=None) -> str:
        """Ask about particles the user picked: sends the question with ewald's description of the
        selection and its surroundings (:mod:`ewald.llm.selection`) and a close-up render.
        ``cancel`` as in :meth:`ask`."""
        from .selection import describe_selection, selection_prompt

        ts = self.toolset
        pipe = ts._need_pipeline()
        if frame is not None:
            ts.frame = int(frame) % pipe.num_frames
        ts.selection = [int(i) for i in np.unique(np.asarray(indices, int))]
        data = pipe.compute(ts.frame)
        description = describe_selection(data, ts.selection, radius=radius, frame=ts.frame)
        png = ts.selection_image(data, ts.selection, radius) if image else None
        return self.ask(selection_prompt(question, description), images=[png] if png else None, cancel=cancel)

    def reset(self):
        self.chat.reset()
