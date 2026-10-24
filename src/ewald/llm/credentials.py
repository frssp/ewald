"""AI API keys: saved in the OS keychain, never in settings.json, sessions or logs.

>>> from ewald.llm import credentials
>>> credentials.save_api_key("sk-ant-...")                  # Anthropic (the default provider)
>>> credentials.save_api_key("sk-...", provider="openai")   # one keychain entry per provider
>>> client = credentials.client()                           # anthropic.Anthropic with the saved key
>>> client = credentials.client(provider="openai")          # openai.OpenAI with the saved key
>>> credentials.check()                                     # (True, "Connected: Claude Opus 5.5")

Lookup order: the saved key, then the provider's environment variable (``ANTHROPIC_API_KEY``,
``OPENAI_API_KEY``, ``GEMINI_API_KEY``). With neither, the client falls back to its SDK's own
lookup (for Anthropic ``ANTHROPIC_AUTH_TOKEN`` or an ``ant auth login`` profile). Where no
keychain backend works (a headless Linux box, ssh without a desktop session), keys go to
``<config dir>/credentials.json`` readable by the owner only. Providers are in
:mod:`ewald.llm.providers`.

A key is passed to the client object, never put into ``os.environ``, so the Modifier Forge
test subprocess (``forge._subprocess_env``) cannot see it.
"""
from __future__ import annotations

import os
from pathlib import Path

from .. import keychain
from ..keychain import _keyring, keyring_name  # noqa: F401  (tests replace _keyring)

SERVICE = keychain.SERVICE
USERNAME = "anthropic-api-key"  # keychain entry of the default provider
ENV_VAR = "ANTHROPIC_API_KEY"

# Claude model id -> label; the first is the default (other providers: providers.BUILTIN)
MODELS = {
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5 (faster, cheaper)",
    "claude-haiku-5-5": "Claude Haiku 5.5 (fastest)",
    "claude-fable-5-1": "Claude Fable 5.1 (most capable)",
}
DEFAULT_MODEL = next(iter(MODELS))


def _provider(provider=None):
    from .providers import get

    return get(provider)


def _file() -> Path:
    return keychain.secrets_file()


def stored_api_key(provider=None) -> tuple[str | None, str | None]:
    """(key, where) for the saved key of ``provider``; where is 'keychain', 'file' or None."""
    name = _provider(provider).key_name
    return keychain.get_secret(name, _kr=_keyring()) if name else (None, None)


def get_api_key(provider=None) -> tuple[str | None, str | None]:
    """(key, source): source is 'keychain', 'file', 'env' or None (SDK defaults apply)."""
    key, where = stored_api_key(provider)
    if key:
        return key, where
    for var in _provider(provider).env_vars:
        env = os.environ.get(var, "").strip()
        if env:
            return env, "env"
    return None, None


_changes = 0   # saved keys changed this many times in this process (see changes())


def changes() -> int:
    """How often a key was saved or deleted in this process; a chat compares it to know that the
    client it made earlier holds an old key."""
    return _changes


def save_api_key(key: str, provider=None) -> str:
    """Save ``key`` for ``provider``; returns 'keychain' or 'file' (the keychain when one works)."""
    key = key.strip()
    if not key:
        raise ValueError("empty API key")
    name = _provider(provider).key_name
    if not name:
        raise ValueError(f"{_provider(provider).label} takes no API key")
    global _changes
    _changes += 1
    return keychain.set_secret(name, key, _kr=_keyring())


def delete_api_key(provider=None):
    """Forget the saved key (keychain and file); the environment variable is left alone."""
    global _changes
    name = _provider(provider).key_name
    if name:
        _changes += 1
        keychain.delete_secret(name, _kr=_keyring())


def mask(key: str | None) -> str:
    """'sk-ant-…3f9a' for display; never show more."""
    if not key:
        return ""
    return f"{key[:7]}…{key[-4:]}" if len(key) > 16 else "…" + key[-2:]


def client(api_key: str | None = None, provider=None, base_url: str | None = None, **kwargs):
    """A client of ``provider`` (default Anthropic: ``anthropic.Anthropic``) using ``api_key``,
    else the saved key, else the SDK's own lookup. ``base_url`` overrides the endpoint."""
    return _provider(provider).client(api_key, base_url, **kwargs)


def check(api_key: str | None = None, model: str | None = None, _client=None, *, provider=None,
          base_url: str | None = None) -> tuple[bool, str]:
    """Test the key and model of ``provider`` against its models endpoint (no tokens used);
    returns (ok, message)."""
    p = _provider(provider)
    return p.check(api_key, model or p.default_model, base_url, _client=_client)
