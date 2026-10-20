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
compute it. Prefer pipeline modifiers over ad-hoc code, so the analysis stays \
reproducible: in the GUI every modifier you add becomes a step cell in the user's notebook \
(the same cell Insert > Step makes), display settings become the view cell (the Display \
panel edits the same settings), the Structure tab shows the result, and you answer in the AI \
chat side bar next to the notebook.

For trajectories, start with describe_trajectory and look at its events before \
reading individual frames. Render frames when the geometry matters and look at \
the image before describing it. Use Angstrom, fs, eV and g/cm^3; name frames by \
index. Be concise and specific; say what you checked and what remains uncertain, \
and suggest the next analysis when one is clearly worth doing."""


class Agent:
    """Conversation with an AI model over one ewald session.

    ``provider`` is a name from :mod:`ewald.llm.providers` ("anthropic", the default, "openai",
    "gemini", "openai-compatible" or a plugin's) or a ``Provider``; ``model`` defaults to the
    provider's first model and ``base_url`` to its endpoint. ``client`` defaults to the
    provider's client with the saved key; pass any object with the same methods
    (``beta.messages.create`` for Claude, ``chat.completions.create`` for the others) to test
    without network access. ``fallbacks=True`` enables Claude's server-side refusal fallback
    (another model continues if the request is declined). ``effort`` applies where the provider
    has such a setting.
    """

    def __init__(self, toolset: Toolset | None = None, *, client=None, model: str | None = None,
                 provider=None, base_url: str | None = None,
                 effort: str = "high", max_tokens: int = 16000, max_turns: int = 30,
                 system: str = SYSTEM_PROMPT, fallbacks: bool = True,
                 on_event: Callable[[str, object], None] | None = None):
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
        self.chat = self.provider.chat(client, self.model, system=system, tools=self.toolset.definitions,
                                       max_tokens=max_tokens, effort=effort, fallbacks=fallbacks)

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
        return cls(Toolset(pipeline, allow_code=kwargs.pop("allow_code", False)), **kwargs)

    # ------------------------------------------------------------------ loop
    def _sync(self):
        """Settings changed on the agent after it was made (model, system, ...) reach the chat."""
        c = self.chat
        c.model, c.system, c.max_tokens = self.model, self.system, self.max_tokens
        if self.provider.effort:
            c.effort = self.effort
        c.options["fallbacks"] = self.fallbacks

    def ask(self, prompt: str, images: list[bytes] | None = None) -> str:
        """Send ``prompt`` (optionally with PNG images), run tools until the model answers, return its text."""
        self._sync()
        self.chat.user(prompt, images)
        for _ in range(self.max_turns):
            reply = self.chat.send()
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
                self.on_event("tool_call", {"name": call.name, "input": call.input})
                result = self.toolset.call(call.name, dict(call.input or {}))
                self.on_event("tool_result", {"name": call.name, "result": result})
                results.append((call, result))
            self.chat.tool_results(results)
        return f"[stopped after {self.max_turns} tool rounds]"

    def ask_about_selection(self, question: str, indices, frame: int | None = None, radius: float = 6.0,
                            image: bool = True) -> str:
        """Ask about particles the user picked: sends the question with ewald's description of the
        selection and its surroundings (:mod:`ewald.llm.selection`) and a close-up render."""
        from .selection import describe_selection, selection_prompt

        ts = self.toolset
        pipe = ts._need_pipeline()
        if frame is not None:
            ts.frame = int(frame) % pipe.num_frames
        ts.selection = [int(i) for i in np.unique(np.asarray(indices, int))]
        data = pipe.compute(ts.frame)
        description = describe_selection(data, ts.selection, radius=radius, frame=ts.frame)
        png = ts.selection_image(data, ts.selection, radius) if image else None
        return self.ask(selection_prompt(question, description), images=[png] if png else None)

    def reset(self):
        self.chat.reset()
