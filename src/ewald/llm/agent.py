"""A Claude agent that analyzes structures through a :class:`~ewald.llm.Toolset`.

>>> from ewald.llm import Agent
>>> agent = Agent.from_file("XDATCAR")
>>> print(agent.ask("What happens in this run? Show me the most interesting frame."))
>>> print(agent.toolset.to_script())     # the analysis as a reproducible script

Requires ``pip install anthropic`` and credentials (``ANTHROPIC_API_KEY`` or
``ant auth login``). The loop is written out by hand rather than using the SDK
tool runner so tool results can carry rendered images and the toolset stays
shared with the MCP server and GUI.
"""
from __future__ import annotations

import base64
from typing import Callable

import numpy as np

from .tools import Toolset, ToolResult

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """\
You are the analysis assistant inside ewald, an atomistic structure and trajectory \
viewer (an open alternative to OVITO Pro). The user is a computational materials \
scientist working with VASP and LAMMPS output.

You act on a live pipeline (source -> modifiers -> rendered frame) through tools. \
Every number you report must come from a tool result in this conversation; never \
estimate lattice constants, coordination numbers, diffusion coefficients or \
energies from memory. When a question can be answered by computing something, \
compute it. Prefer pipeline modifiers over ad-hoc code, so the analysis stays \
reproducible and shows up in the user's pipeline panel.

For trajectories, start with describe_trajectory and look at its events before \
reading individual frames. Render frames when the geometry matters and look at \
the image before describing it. Use Angstrom, fs, eV and g/cm^3; name frames by \
index. Be concise and specific; say what you checked and what remains uncertain, \
and suggest the next analysis when one is clearly worth doing."""


def _tool_result_block(tool_use_id: str, result: ToolResult) -> dict:
    content: list[dict] = [{"type": "text", "text": result.text()}]
    for png in result.images:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                    "data": base64.standard_b64encode(png).decode()}})
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if result.is_error:
        block["is_error"] = True
    return block


class Agent:
    """Conversation with Claude over one ewald session.

    ``client`` defaults to ``anthropic.Anthropic()``; pass any object with the
    same ``beta.messages.create`` method to test without network access.
    ``fallbacks=True`` enables server-side refusal fallback (another model
    continues if the request is declined).
    """

    def __init__(self, toolset: Toolset | None = None, *, client=None, model: str = DEFAULT_MODEL,
                 effort: str = "high", max_tokens: int = 16000, max_turns: int = 30,
                 system: str = SYSTEM_PROMPT, fallbacks: bool = True,
                 on_event: Callable[[str, object], None] | None = None):
        self.toolset = toolset or Toolset()
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.system = system
        self.fallbacks = fallbacks
        self.on_event = on_event or (lambda kind, payload: None)
        self.messages: list[dict] = []

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
    def _request(self) -> dict:
        tools = self.toolset.definitions()
        if tools:  # cache the stable prefix (tools + system) across turns
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        params = {"model": self.model, "max_tokens": self.max_tokens,
                  "system": [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
                  "tools": tools, "messages": self.messages,
                  "output_config": {"effort": self.effort}}
        if self.fallbacks:
            params.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        return params

    def ask(self, prompt: str, images: list[bytes] | None = None) -> str:
        """Send ``prompt`` (optionally with PNG images), run tools until Claude answers, return its text."""
        content: list[dict] = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                            "data": base64.standard_b64encode(im).decode()}}
                               for im in images or []]
        content.append({"type": "text", "text": prompt})
        self.messages.append({"role": "user", "content": content})
        for _ in range(self.max_turns):
            response = self.client.beta.messages.create(**self._request())
            # append the full content (thinking blocks included) unchanged
            self.messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "refusal":
                return "[request declined by the model]"
            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text")
                self.on_event("answer", text)
                if response.stop_reason == "max_tokens":
                    text += "\n[truncated: max_tokens reached]"
                return text
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                self.on_event("tool_call", {"name": block.name, "input": block.input})
                result = self.toolset.call(block.name, dict(block.input or {}))
                self.on_event("tool_result", {"name": block.name, "result": result})
                results.append(_tool_result_block(block.id, result))
            self.messages.append({"role": "user", "content": results})
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
        self.messages.clear()
