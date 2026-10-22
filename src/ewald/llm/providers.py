"""AI providers: which service answers Ask AI, the AI chat side bar, ``ewald ask``, the narrator
and the Modifier Forge.

Built in (Preferences > AI > Provider, setting ``llm.provider``):

* ``anthropic`` - Claude (the default).
* ``openai`` - OpenAI.
* ``gemini`` - Google Gemini through its OpenAI-compatible endpoint.
* ``openai-compatible`` - any server speaking the OpenAI Chat Completions API: Ollama and
  LM Studio on your machine, vLLM, OpenRouter, ... Set its address as ``llm.base_url``.

A *connection* is a provider plus the address, model and headers to use with it
(:meth:`Provider.connect`, :func:`from_settings`); Preferences > AI saves several as named
connections (``llm.profiles``) and ``ewald ask --profile NAME`` picks one.

Plugins add more with :func:`ewald.plugins.ai_provider` (a :class:`Provider`, usually with
``chat_class=OpenAIChat`` and a ``base_url``, or a :class:`~ewald.llm.chat.Chat` subclass of
their own). A plugin can also bring its own client (``make_client``) for a gateway with its own
authentication headers, models configured in the environment (``env_models``, ``model_url``)
and a model prefix (``env_prefix``). Every provider makes a client (:meth:`Provider.client`) and a chat
(:meth:`Provider.chat`); the agent loop, tools, images and history work the same with each.

API keys: one keychain entry per provider (:mod:`ewald.llm.credentials`), then the provider's
environment variable. A key is handed to the client object, never put into ``os.environ`` or
settings.json.
"""
from __future__ import annotations

import importlib
import uuid
from dataclasses import dataclass, field, replace
from typing import Callable

from .chat import AnthropicChat, Chat, OpenAIChat

DEFAULT = "anthropic"


def _anthropic_client(api_key=None, base_url=None, **kwargs):
    import anthropic

    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(api_key=api_key, **kwargs) if api_key else anthropic.Anthropic(**kwargs)


def _openai_client(api_key=None, base_url=None, *, key_header=None, **kwargs):
    import openai

    if base_url:
        kwargs["base_url"] = base_url
    if key_header:
        # the key goes in a header of its own (Azure's api-key, gateways' ticket headers)
        kwargs["default_headers"] = {**kwargs.get("default_headers", {}), key_header: api_key or ""}
        api_key = None
    # a local server needs no key, but the SDK insists on one
    return openai.OpenAI(api_key=api_key or "not-needed", **kwargs)


def fill_headers(templates: dict) -> dict:
    """Header values with ``{uuid}`` replaced by a new random id (one per call)."""
    return {k: str(v).replace("{uuid}", str(uuid.uuid4())) for k, v in (templates or {}).items()}


def parse_headers(headers) -> dict[str, str]:
    """``"Name: value; Other: value"`` (or a dict) as a dict of HTTP headers."""
    if isinstance(headers, dict):
        return {str(k): str(v) for k, v in headers.items()}
    out = {}
    for item in str(headers or "").replace("\n", ";").split(";"):
        name, sep, value = item.partition(":")
        if sep and name.strip():
            out[name.strip()] = value.strip()
    return out


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
    key_title: str = "API key"          # what the key is called (e.g. "credential")
    key_hint: str = "API key"           # placeholder of the key field
    key_help: str = ""                  # where to get a key
    env_fallbacks: tuple = ()           # more variables for the key, read after env_var
    vision: bool = True                 # takes images (renders, close-ups)
    effort: bool = False                # honors llm.effort
    options: tuple = ()                 # connection fields it uses besides base_url, headers and
                                        # timeout: "route_prefix", "key_header" and keywords its
                                        # make_client takes
    env_models: Callable | None = None  # () -> {model: url} configured in the environment
    model_url: Callable | None = None   # model -> endpoint when no base_url is set
    env_prefix: Callable | None = None  # () -> model prefix when none is set
    check_with: str = "models"          # "models": the models endpoint; "chat": a tiny request;
                                        # "auto": the models endpoint, else a tiny request
    client_options: dict = field(default_factory=dict)  # passed to make_client (Provider.connect)
    description: str = ""
    origin: str = "built-in"

    @property
    def builtin(self) -> bool:
        return self.origin == "built-in"

    @property
    def all_models(self) -> dict:
        """Suggested models: those configured in the environment, then ``models``."""
        env = self.env_models() if self.env_models else {}
        return {**{m: "from the environment" for m in env}, **self.models}

    @property
    def default_model(self) -> str:
        return next(iter(self.all_models), "")

    @property
    def env_vars(self) -> tuple:
        return tuple(v for v in (self.env_var, *self.env_fallbacks) if v)

    def connect(self, *, base_url: str | None = None, route_prefix: str | None = None, headers=None,
                timeout: float | None = None, key_header: str | None = None, **options) -> "Provider":
        """This provider with an address, model prefix, extra HTTP headers, a timeout (seconds;
        0 or None keeps the SDK's) and the provider's own ``options`` (keywords its ``make_client``
        takes); empty values keep the defaults, options it does not use are ignored. A header value
        with ``{uuid}`` gets a new id in every request; ``key_header`` sends the API key in that
        header instead of ``Authorization: Bearer``."""
        client_options = dict(self.client_options)
        chat_options = dict(self.chat_options)
        extra = parse_headers(headers)
        fresh = {k: v for k, v in extra.items() if "{uuid}" in v}
        if static := {k: v for k, v in extra.items() if k not in fresh}:
            client_options["default_headers"] = {**client_options.get("default_headers", {}), **static}
        if fresh:
            chat_options["header_templates"] = {**chat_options.get("header_templates", {}), **fresh}
        if key_header and "key_header" in self.options:
            client_options["key_header"] = key_header.strip()
        if timeout:
            client_options["timeout"] = float(timeout)
        client_options.update({k: v for k, v in options.items() if v and k in self.options})
        if route_prefix and "route_prefix" in self.options:
            chat_options["model_prefix"] = route_prefix
        return replace(self, base_url=base_url or self.base_url, client_options=client_options,
                       chat_options=chat_options)

    @property
    def package(self) -> str:
        return {"anthropic": "anthropic", "openai": "openai"}.get(self.sdk, self.sdk)

    def client(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None,
               **kwargs):
        """A client for this provider: ``api_key``, else the saved key, else the environment.
        ``model`` picks the endpoint of a provider whose models have their own (``model_url``)."""
        from . import credentials

        key = api_key or credentials.get_api_key(self)[0]
        base_url = base_url or self.base_url or None
        if not base_url and self.model_url:
            base_url = self.model_url(model or self.default_model) or None
        make = self.make_client or (_anthropic_client if self.sdk == "anthropic" else _openai_client)
        return make(key, base_url, **{**self.client_options, **kwargs})

    def model_prefix(self) -> str:
        """What goes in front of the model id on the wire (``route_prefix``)."""
        return self.chat_options.get("model_prefix") or (self.env_prefix() if self.env_prefix else "") or ""

    def chat(self, client, model: str | None = None, **kwargs) -> Chat:
        options = {**self.chat_options, **kwargs}
        if self.model_prefix():
            options.setdefault("model_prefix", self.model_prefix())
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
            return False, (f"The {self.package} package is not installed; ewald installs it, so reinstall ewald "
                           f"or pip install {self.package}")
        errors = {n: getattr(sdk, n, None) for n in ("AuthenticationError", "PermissionDeniedError",
                                                     "NotFoundError", "APIConnectionError", "APIStatusError")}

        def caught(exc, name):
            return errors[name] is not None and isinstance(exc, errors[name])

        def ping():
            c.chat.completions.create(model=f"{self.model_prefix()}{model}", max_tokens=16,
                                      messages=[{"role": "user", "content": "ping"}],
                                      extra_headers=fill_headers(self.chat_options.get("header_templates")))
            return True, f"Connected: {model}"

        try:
            c = _client or self.client(api_key, base_url, model, timeout=15.0, max_retries=0)
            if self.check_with == "chat":  # gateways often have no models endpoint
                return ping()
            try:
                info = c.models.retrieve(model)
            except Exception as exc:
                # no models endpoint (or not this model listed): a tiny chat request decides
                if self.check_with != "auto" or not (caught(exc, "NotFoundError") or caught(exc, "APIStatusError")) \
                        or caught(exc, "AuthenticationError") or caught(exc, "PermissionDeniedError"):
                    raise
                return ping()
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
            if isinstance(exc, ValueError) and msg.startswith("No "):
                return False, msg
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
        vision=False, options=("route_prefix", "key_header"), check_with="auto",
        description="Any server with the OpenAI Chat Completions API: Ollama (http://localhost:11434/v1), "
                    "LM Studio (http://localhost:1234/v1), vLLM, OpenRouter (https://openrouter.ai/api/v1), "
                    "Azure OpenAI or an API gateway (API key header, extra headers). "
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


CONNECTION_FIELDS = ("base_url", "route_prefix", "key_header", "headers", "timeout")
PROFILE_FIELDS = ("provider", "model", "effort") + CONNECTION_FIELDS


def profile_from_settings(settings) -> dict:
    """The connection Preferences > AI shows now, as a profile (``llm.profiles`` entry)."""
    return {f: settings[f"llm.{f}"] for f in PROFILE_FIELDS}


def from_settings(settings=None, profile: str | None = None, **overrides) -> dict:
    """Agent keyword arguments for the connection in ``settings`` (default: the saved ones),
    or for the saved connection ``profile``: ``{"provider": Provider, "model": str | None,
    "effort": str}``. ``overrides`` (provider, model, base_url, ...) win over both; None is
    ignored. Raises KeyError for an unknown profile or provider."""
    if settings is None:
        from ..settings import load

        settings = load()
    values = profile_from_settings(settings)
    if profile:
        saved = settings["llm.profiles"]
        if profile not in saved:
            raise KeyError(f"no saved AI connection {profile!r}; saved: {', '.join(saved) or 'none'}")
        values = {**{f: "" for f in CONNECTION_FIELDS}, "timeout": 0.0, "effort": values["effort"],
                  **saved[profile]}
    if overrides.get("provider") not in (None, values["provider"]):
        # the saved model and address belong to the saved provider
        values = {**{f: "" for f in CONNECTION_FIELDS}, "timeout": 0.0, "model": "", "effort": values["effort"]}
    values.update({k: v for k, v in overrides.items() if v is not None})
    p = get(values.get("provider") or None)
    conn = p.connect(**{f: values.get(f) for f in CONNECTION_FIELDS})
    return {"provider": conn, "model": values.get("model") or None, "effort": values.get("effort") or "high"}


__all__ = ["Provider", "BUILTIN", "DEFAULT", "CONNECTION_FIELDS", "PROFILE_FIELDS", "providers", "get",
           "from_settings", "profile_from_settings", "parse_headers", "fill_headers"]
