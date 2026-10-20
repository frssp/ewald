"""One conversation with a model, in the provider's own message format.

The agent loop (:class:`ewald.llm.Agent`), the Modifier Forge and the trajectory narrator talk
to every AI provider through :class:`Chat`: add a user message (text and PNG images), send,
get a :class:`Reply` (text, tool calls, why it stopped), hand back tool results. Tools are
given in Anthropic's shape (``name``, ``description``, ``input_schema``; what
``Toolset.definitions()`` returns) and converted where a provider needs another one.

* :class:`AnthropicChat` - Claude (Messages API: tool_use blocks, prompt caching, effort,
  server-side refusal fallback, structured output).
* :class:`OpenAIChat` - the Chat Completions API of OpenAI and every server that copies it
  (Google Gemini's OpenAI endpoint, Ollama, LM Studio, vLLM, OpenRouter, ...).

A provider plugin (``@ai_provider``) brings its own subclass when neither fits.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Reply:
    """What the model answered: ``stop`` is "end", "tool_use", "max_tokens" or "refusal"."""

    text: str
    calls: list[ToolCall] = field(default_factory=list)
    stop: str = "end"


def _b64(png: bytes) -> str:
    return base64.standard_b64encode(png).decode()


def _result_parts(result) -> tuple[str, list[bytes], bool]:
    """(text, images, is_error) of a ``ToolResult`` or a plain string."""
    if isinstance(result, str):
        return result, [], False
    return result.text(), list(result.images), bool(result.is_error)


class Chat:
    """Base class: the conversation state and the calls the loops use.

    ``tools`` is a list of tool definitions or a function returning one (read on every send,
    so plugin tools loaded later join). ``schema`` asks for a JSON answer of that shape.
    ``effort`` is honored where the provider has such a setting.
    """

    def __init__(self, client, model: str, *, system: str = "", tools: list | Callable | None = None,
                 max_tokens: int = 16000, effort: str | None = None, schema: dict | None = None,
                 vision: bool = True, **options):
        self.client = client
        self.model = model
        self.system = system
        self.tools = tools
        self.max_tokens = max_tokens
        self.effort = effort
        self.schema = schema
        self.vision = vision
        self.options = options
        self.messages: list = []

    def tool_definitions(self) -> list[dict]:
        tools = self.tools() if callable(self.tools) else self.tools
        return list(tools or [])

    def user(self, text: str, images: list[bytes] | None = None):
        raise NotImplementedError

    def send(self, tool_choice: str | None = None) -> Reply:
        """Send the conversation; ``tool_choice`` forces a call to that tool."""
        raise NotImplementedError

    def tool_results(self, results: list[tuple[ToolCall, object]]):
        """Answer the calls of the last reply: ``(call, ToolResult or str)`` pairs, or
        ``(call, str, is_error)``."""
        raise NotImplementedError

    def reset(self):
        self.messages.clear()


class AnthropicChat(Chat):
    """Claude through the Messages API (``anthropic`` SDK, or any object with the same
    ``beta.messages.create``/``messages.create``). Options: ``fallbacks`` (server-side
    refusal fallback, beta), ``beta`` (use the beta endpoint, default True), ``cache``."""

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def user(self, text, images=None):
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(im)}}
                   for im in images or []]
        content.append({"type": "text", "text": text})
        self.messages.append({"role": "user", "content": content})

    def request(self, tool_choice=None) -> dict:
        cache = self.options.get("cache", True)
        params = {"model": self.model, "max_tokens": self.max_tokens, "messages": self.messages}
        if self.system:
            params["system"] = ([{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}]
                                if cache else self.system)
        tools = self.tool_definitions()
        if tools:
            if cache:  # cache the stable prefix (tools + system) across turns
                tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = {"type": "tool", "name": tool_choice}
        config = {}
        if self.effort:
            config["effort"] = self.effort
        if self.schema:
            config["format"] = {"type": "json_schema", "schema": self.schema}
        if config:
            params["output_config"] = config
        if self.options.get("fallbacks"):
            params.update(betas=[self.FALLBACK_BETA], fallbacks="default")
        return params

    def send(self, tool_choice=None):
        params = self.request(tool_choice)
        api = self.client.beta.messages if self.options.get("beta", True) else self.client.messages
        response = api.create(**params)
        # keep the full content (thinking blocks included) unchanged
        self.messages.append({"role": "assistant", "content": response.content})
        calls = [ToolCall(b.id, b.name, dict(b.input or {})) for b in response.content
                 if getattr(b, "type", None) == "tool_use"]
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        stop = {"tool_use": "tool_use", "max_tokens": "max_tokens", "refusal": "refusal"}.get(response.stop_reason, "end")
        if calls and stop == "end":
            stop = "tool_use"
        return Reply(text, calls, stop)

    def tool_results(self, results):
        blocks = []
        for call, result, *flag in results:
            text, images, is_error = _result_parts(result)
            is_error = is_error or bool(flag and flag[0])
            if isinstance(result, str):
                content = text
            else:
                content = [{"type": "text", "text": text}]
                content += [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(im)}}
                            for im in images]
            block = {"type": "tool_result", "tool_use_id": call.id, "content": content}
            if is_error:
                block["is_error"] = True
            blocks.append(block)
        self.messages.append({"role": "user", "content": blocks})


class OpenAIChat(Chat):
    """The Chat Completions API (``openai`` SDK with a ``base_url`` for other servers, or any
    object with the same ``chat.completions.create``). Images go as data URLs; tool results
    are ``tool`` messages, and images they carry follow in one user message. Options:
    ``max_tokens_param`` ("max_completion_tokens" for OpenAI, "max_tokens" for most others),
    ``reasoning`` (send ``reasoning_effort`` for models that think)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def _image_parts(self, images) -> list[dict]:
        return [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(im)}"}} for im in images]

    def user(self, text, images=None):
        images = list(images or [])
        if images and not self.vision:
            text += f"\n\n({len(images)} image(s) left out: this model reads text only.)"
            images = []
        if images:
            content = [{"type": "text", "text": text}] + self._image_parts(images)
        else:
            content = text
        self.messages.append({"role": "user", "content": content})

    def reset(self):
        self.messages[:] = [m for m in self.messages[:1] if m.get("role") == "system"]

    def request(self, tool_choice=None) -> dict:
        params = {"model": self.model, "messages": self.messages,
                  self.options.get("max_tokens_param", "max_tokens"): self.max_tokens}
        tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                   "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}}
                 for t in self.tool_definitions()]
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
        if self.schema:
            params["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "answer", "schema": self.schema}}
        if self.effort and self.options.get("reasoning"):
            params["reasoning_effort"] = {"xhigh": "high", "max": "high"}.get(self.effort, self.effort)
        return params

    def send(self, tool_choice=None):
        response = self.client.chat.completions.create(**self.request(tool_choice))
        choice = response.choices[0]
        msg = choice.message
        text = msg.content or ""
        calls = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (TypeError, json.JSONDecodeError):
                args = {}
            calls.append(ToolCall(tc.id, tc.function.name, args if isinstance(args, dict) else {}))
        entry = {"role": "assistant", "content": text or None}
        if calls:
            entry["tool_calls"] = [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": json.dumps(c.input)}} for c in calls]
        self.messages.append(entry)
        if getattr(msg, "refusal", None) or choice.finish_reason == "content_filter":
            return Reply(text or getattr(msg, "refusal", "") or "", [], "refusal")
        stop = "tool_use" if calls else {"length": "max_tokens"}.get(choice.finish_reason, "end")
        return Reply(text, calls, stop)

    def tool_results(self, results):
        pictures = []
        for call, result, *flag in results:
            text, images, is_error = _result_parts(result)
            is_error = is_error or bool(flag and flag[0])
            if is_error and not text.startswith("ERROR"):
                text = "ERROR: " + text
            if images:
                text += f"\n({len(images)} image(s) follow in the next message.)" if self.vision else \
                        f"\n({len(images)} image(s) left out: this model reads text only.)"
                pictures += images if self.vision else []
            self.messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
        if pictures:
            self.messages.append({"role": "user", "content": [{"type": "text", "text": "Images from the tool results:"}]
                                  + self._image_parts(pictures)})


__all__ = ["Chat", "AnthropicChat", "OpenAIChat", "Reply", "ToolCall"]
