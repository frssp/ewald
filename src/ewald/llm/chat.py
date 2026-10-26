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

``send(on_text=...)`` streams: the answer's text reaches ``on_text`` piece by piece as it arrives
(the AI chat side bar shows it growing), and the :class:`Reply` is the same as without streaming.
A client without streaming (a test fake, a server that refuses ``stream``) answers in one piece.

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

    def send(self, tool_choice: str | None = None, on_text: Callable[[str], None] | None = None) -> Reply:
        """Send the conversation; ``tool_choice`` forces a call to that tool. ``on_text`` gets the
        answer's text as it streams in (an exception it raises, such as a Stop, ends the request);
        ``options["stream"] = False`` sends it in one piece."""
        raise NotImplementedError

    def streams(self, on_text) -> bool:
        return on_text is not None and self.options.get("stream", True) is not False

    def tool_results(self, results: list[tuple[ToolCall, object]]):
        """Answer the calls of the last reply: ``(call, ToolResult or str)`` pairs, or
        ``(call, str, is_error)``."""
        raise NotImplementedError

    def reset(self):
        self.messages.clear()


def _cache_last(messages: list) -> list:
    """``messages`` with a cache breakpoint on the last block of the last user message, so each
    round of a tool loop reads the conversation so far from the prompt cache instead of paying
    for it again (with tools and system that makes three of the four breakpoints allowed).
    The stored history stays unmarked: breakpoints from earlier rounds would pile up."""
    if not messages or messages[-1].get("role") != "user":
        return messages
    content = messages[-1]["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not content or not isinstance(content[-1], dict):
        return messages
    content = [*content[:-1], {**content[-1], "cache_control": {"type": "ephemeral"}}]
    return [*messages[:-1], {**messages[-1], "content": content}]


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
        params = {"model": self.model, "max_tokens": self.max_tokens,
                  "messages": _cache_last(self.messages) if cache else self.messages}
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

    def send(self, tool_choice=None, on_text=None):
        params = self.request(tool_choice)
        api = self.client.beta.messages if self.options.get("beta", True) else self.client.messages
        if self.streams(on_text) and hasattr(api, "stream"):
            with api.stream(**params) as stream:
                for piece in stream.text_stream:
                    on_text(piece)
                response = stream.get_final_message()
        else:
            response = api.create(**params)
            if on_text is not None:
                on_text("".join(b.text for b in response.content if getattr(b, "type", None) == "text"))
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
            if is_error and images:     # the API takes only text in an error result (a cell that drew, then failed)
                text += f"\n[{len(images)} image(s) not sent with this error]"
                images = []
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
    ``reasoning`` (send ``reasoning_effort`` for models that think), ``model_prefix`` (put in front
    of the model id on the wire, e.g. "openai/" for a vendor-qualified route), ``header_templates``
    (HTTP headers sent with every request, ``{uuid}`` in a value replaced by a new id each time)."""

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
        params = {"model": f"{self.options.get('model_prefix') or ''}{self.model}", "messages": self.messages,
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
        if self.options.get("header_templates"):
            from .providers import fill_headers

            params["extra_headers"] = fill_headers(self.options["header_templates"])
        return params

    #: HTTP statuses of a server that does not take ``stream``: the request goes again unstreamed
    NO_STREAM = (400, 404, 405, 415, 422, 501)

    def send(self, tool_choice=None, on_text=None):
        params = self.request(tool_choice)
        create = self.client.chat.completions.create
        if self.streams(on_text):
            try:
                chunks = create(**params, stream=True)
            except Exception as exc:
                if getattr(exc, "status_code", None) not in self.NO_STREAM:
                    raise
                chunks = None
            if chunks is not None and not hasattr(chunks, "choices"):
                try:
                    return self._reply(*self._collect(chunks, on_text))
                finally:
                    getattr(chunks, "close", lambda: None)()
            if chunks is None:
                self.options["stream"] = False   # this server answers in one piece from now on
                params = self.request(tool_choice)   # (new {uuid} headers for the new request)
        else:
            chunks = None
        response = chunks if chunks is not None else create(**params)   # (a client that ignores stream)
        choice = response.choices[0]
        msg = choice.message
        calls = [(tc.id, tc.function.name, tc.function.arguments) for tc in getattr(msg, "tool_calls", None) or []]
        if on_text is not None and msg.content:
            on_text(msg.content)
        return self._reply(msg.content or "", calls, choice.finish_reason, getattr(msg, "refusal", None))

    @staticmethod
    def _collect(chunks, on_text):
        """(text, calls, finish_reason, refusal) of a streamed answer; tool calls come in pieces
        (``index``, then the id and name once, the arguments in parts)."""
        text, refusal, slots, finish = [], [], {}, None
        for chunk in chunks:
            if not getattr(chunk, "choices", None):
                continue   # e.g. a closing usage chunk
            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(delta, "content", None):
                text.append(delta.content)
                on_text(delta.content)
            if getattr(delta, "refusal", None):
                refusal.append(delta.refusal)
            for tc in getattr(delta, "tool_calls", None) or []:
                index = tc.index if getattr(tc, "index", None) is not None else len(slots)
                slot = slots.setdefault(index, ["", "", ""])
                slot[0] = slot[0] or (tc.id or "")
                fn = getattr(tc, "function", None)
                if fn is not None:
                    slot[1] = slot[1] or (fn.name or "")
                    slot[2] += fn.arguments or ""
            finish = choice.finish_reason or finish
        calls = [tuple(slots[i]) for i in sorted(slots)]
        return "".join(text), calls, finish, "".join(refusal) or None

    def _reply(self, text, raw_calls, finish, refusal) -> Reply:
        """Record the assistant's turn and make its :class:`Reply`; ``raw_calls`` are ``(id, name,
        arguments JSON)``."""
        calls = []
        for i, (cid, name, arguments) in enumerate(raw_calls):
            try:
                args = json.loads(arguments or "{}")
            except (TypeError, json.JSONDecodeError):
                args = {}
            calls.append(ToolCall(cid or f"call_{i}", name, args if isinstance(args, dict) else {}))
        entry = {"role": "assistant", "content": text or None}
        if calls:
            entry["tool_calls"] = [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": json.dumps(c.input)}} for c in calls]
        self.messages.append(entry)
        if refusal or finish == "content_filter":
            return Reply(text or refusal or "", [], "refusal")
        stop = "tool_use" if calls else {"length": "max_tokens"}.get(finish, "end")
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
