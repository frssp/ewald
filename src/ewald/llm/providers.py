"""AI providers: which service answers Ask AI, the AI chat side bar, ``ewald ask``, the narrator
and the Modifier Forge.

Built in (Preferences > AI > Provider, setting ``llm.provider``):

* ``anthropic`` - Claude (the default), ``pip install anthropic``.
* ``openai`` - OpenAI, ``pip install openai``.
* ``gemini`` - Google Gemini through its OpenAI-compatible endpoint, ``pip install openai``.
* ``openai-compatible`` - any server speaking the OpenAI Chat Completions API: Ollama and
  LM Studio on your machine, vLLM, OpenRouter, ... Set its address as ``llm.base_url``.

Plugins add more with :func:`ewald.plugins.ai_provider` (a :class:`Provider`, usually with
``chat_class=OpenAIChat`` and a ``base_url``, or a :class:`~ewald.llm.chat.Chat` subclass of
their own). Every provider makes a client (:meth:`Provider.client`) and a chat
(:meth:`Provider.chat`); the agent loop, tools, images and history work the same with each.

API keys: one keychain entry per provider (:mod:`ewald.llm.credentials`), then the provider's
environment variable. A key is handed to the client object, never put into ``os.environ`` or
settings.json.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable

from .chat import AnthropicChat, Chat, OpenAIChat

DEFAULT = "anthropic"


def _anthropic_client(api_key=None, base_url=None, **kwargs):
    import anthropic

    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(api_key=api_key, **kwargs) if api_key else anthropic.Anthropic(**kwargs)


def _openai_client(api_key=None, base_url=None, **kwargs):
    import openai

    if base_url:
        kwargs["base_url"] = base_url
    # a local server needs no key, but the SDK insists on one
    return openai.OpenAI(api_key=api_key or "not-needed", **kwargs)


@dataclass
class Provider:
    """An AI service. ``models`` maps suggested model ids to labels (the first is the default;
    any other id the service knows works too). ``key_name`` is its keychain entry ("" when it
    takes no key), ``env_var`` the environment variable read when none is saved."""

    name: str
    label: str
    models: dict = field(default_factory=dict)
    assistant: str = "AI"               # how the chat names the answers
    sdk: str = "openai"                 # Python package the client comes from
    make_client: Callable | None = None  # (api_key, base_url, **kwargs) -> client
    chat_class: type = OpenAIChat
    chat_options: dict = field(default_factory=dict)
    base_url: str = ""                  # default endpoint ("" = the SDK's own)
    key_name: str = ""
    env_var: str = ""
    key_required: bool = True
    key_hint: str = "API key"           # placeholder of the key field
    key_help: str = ""                  # where to get a key
    vision: bool = True                 # takes images (renders, close-ups)
    effort: bool = False                # honors llm.effort
    description: str = ""
    origin: str = "built-in"

    @property
    def builtin(self) -> bool:
        return self.origin == "built-in"

    @property
    def default_model(self) -> str:
        return next(iter(self.models), "")

    @property
    def package(self) -> str:
        return {"anthropic": "anthropic", "openai": "openai"}.get(self.sdk, self.sdk)

    def client(self, api_key: str | None = None, base_url: str | None = None, **kwargs):
        """A client for this provider: ``api_key``, else the saved key, else the environment."""
        from . import credentials

        key = api_key or credentials.get_api_key(self.name)[0]
        make = self.make_client or (_anthropic_client if self.sdk == "anthropic" else _openai_client)
        return make(key, base_url or self.base_url or None, **kwargs)

    def chat(self, client, model: str | None = None, **kwargs) -> Chat:
        options = {**self.chat_options, **kwargs}
        options.setdefault("vision", self.vision)
        if not self.effort:
            options.pop("effort", None)
        return self.chat_class(client, model or self.default_model, **options)

    def check(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None,
              _client=None) -> tuple[bool, str]:
        """Ask the service whether the key and model work (the models endpoint, no tokens used)."""
        model = model or self.default_model
        try:
            sdk = importlib.import_module(self.sdk)
        except ImportError:
            return False, f"The {self.package} package is not installed: pip install {self.package}"
        errors = {n: getattr(sdk, n, None) for n in ("AuthenticationError", "PermissionDeniedError",
                                                     "NotFoundError", "APIConnectionError", "APIStatusError")}

        def caught(exc, name):
            return errors[name] is not None and isinstance(exc, errors[name])

        try:
            c = _client or self.client(api_key, base_url, timeout=15.0, max_retries=0)
            info = c.models.retrieve(model)
        except Exception as exc:
            if caught(exc, "AuthenticationError"):
                return False, "The key was rejected (invalid or revoked)."
            if caught(exc, "PermissionDeniedError"):
                return False, f"The key has no access to the {self.label} API."
            if caught(exc, "NotFoundError"):
                return False, f"Connected, but model {model} is not available."
            if caught(exc, "APIConnectionError"):
                where = base_url or self.base_url or f"the {self.label} API"
                return False, f"Could not reach {where} (network, proxy or server not running)."
            if caught(exc, "APIStatusError"):
                return False, f"API error {getattr(exc, 'status_code', '')}.".replace(" .", ".")
            msg = str(exc)
            if "api_key" in msg or "auth" in msg.lower():
                hint = f", or set {self.env_var}" if self.env_var else ""
                return False, f"No API key: enter one{hint}."
            return False, f"{type(exc).__name__}: {msg[:200]}"
        name = getattr(info, "display_name", None) or getattr(info, "id", None) or model
        return True, f"Connected: {name}"


BUILTIN = {
    "anthropic": Provider(
        "anthropic", "Anthropic (Claude)", assistant="Claude", sdk="anthropic", chat_class=AnthropicChat,
        models={"claude-opus-5-5": "Claude Opus 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5 (faster, cheaper)",
                "claude-haiku-5-5": "Claude Haiku 5.5 (fastest)", "claude-fable-5-1": "Claude Fable 5.1 (most capable)"},
        key_name="anthropic-api-key", env_var="ANTHROPIC_API_KEY", key_hint="sk-ant-…",
        key_help="Get a key at console.anthropic.com, or run ant auth login.", effort=True,
        description="Claude models through the Anthropic API."),
    "openai": Provider(
        "openai", "OpenAI", assistant="OpenAI", models={"gpt-5": "GPT-5", "gpt-5-mini": "GPT-5 mini (faster, cheaper)",
                                                        "gpt-4.1": "GPT-4.1"},
        chat_options={"max_tokens_param": "max_completion_tokens", "reasoning": True},
        key_name="openai-api-key", env_var="OPENAI_API_KEY", key_hint="sk-…",
        key_help="Get a key at platform.openai.com.", effort=True,
        description="OpenAI models through the Chat Completions API."),
    "gemini": Provider(
        "gemini", "Google Gemini", assistant="Gemini",
        models={"gemini-2.5-pro": "Gemini 2.5 Pro", "gemini-2.5-flash": "Gemini 2.5 Flash (faster, cheaper)"},
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_name="gemini-api-key", env_var="GEMINI_API_KEY", key_hint="AIza…",
        key_help="Get a key at aistudio.google.com.",
        description="Gemini models through Google's OpenAI-compatible endpoint."),
    "openai-compatible": Provider(
        "openai-compatible", "OpenAI-compatible server", assistant="AI",
        models={"llama3.1": "Llama 3.1", "qwen2.5": "Qwen 2.5"},
        base_url="http://localhost:11434/v1", key_name="openai-compatible-api-key", key_required=False,
        key_hint="Only if the server needs one", key_help="Local servers (Ollama, LM Studio) need no key.",
        vision=False,
        description="Any server with the OpenAI Chat Completions API: Ollama (http://localhost:11434/v1), "
                    "LM Studio (http://localhost:1234/v1), vLLM, OpenRouter (https://openrouter.ai/api/v1). "
                    "Set its address in Server URL and the model it serves."),
}


def providers() -> dict[str, Provider]:
    """Every provider by name: the built-in ones, then ``@ai_provider`` plugins."""
    from .. import plugins

    return plugins.ai_providers()


def get(name: "str | Provider | None" = None) -> Provider:
    """The provider called ``name`` (default: Anthropic)."""
    if isinstance(name, Provider):
        return name
    name = name or DEFAULT
    found = BUILTIN.get(name) or providers().get(name)
    if found is None:
        raise KeyError(f"no AI provider {name!r}; known: {', '.join(providers())}")
    return found


__all__ = ["Provider", "BUILTIN", "DEFAULT", "providers", "get"]
