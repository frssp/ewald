"""The Anthropic API key: saved in the OS keychain, never in settings.json, sessions or logs.

>>> from ewald.llm import credentials
>>> credentials.save_api_key("sk-ant-...")      # system keychain (``keyring``); 0600 file if none
>>> client = credentials.client()               # anthropic.Anthropic with the saved key
>>> credentials.check()                         # (True, "Connected: Claude Opus 5.5")

Lookup order: the saved key, then ``ANTHROPIC_API_KEY``. With neither, ``client()`` is a bare
``anthropic.Anthropic()``, which still finds ``ANTHROPIC_AUTH_TOKEN`` or an ``ant auth login``
profile. Where no keychain backend works (a headless Linux box, ssh without a desktop session),
the key goes to ``<config dir>/credentials.json`` readable by the owner only.

The key is passed to the client object, never put into ``os.environ``, so the Modifier Forge
test subprocess (``forge._subprocess_env``) cannot see it.
"""
from __future__ import annotations

import os
from pathlib import Path

from .. import keychain
from ..keychain import _keyring, keyring_name  # noqa: F401  (tests replace _keyring)

SERVICE = keychain.SERVICE
USERNAME = "anthropic-api-key"
ENV_VAR = "ANTHROPIC_API_KEY"

# model id -> label for the Preferences picker; the first is the default
MODELS = {
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5 (faster, cheaper)",
    "claude-haiku-5-5": "Claude Haiku 5.5 (fastest)",
    "claude-fable-5-1": "Claude Fable 5.1 (most capable)",
}
DEFAULT_MODEL = next(iter(MODELS))


def _file() -> Path:
    return keychain.secrets_file()


def stored_api_key() -> tuple[str | None, str | None]:
    """(key, where) for the saved key; where is 'keychain', 'file' or None."""
    return keychain.get_secret(USERNAME, _kr=_keyring())


def get_api_key() -> tuple[str | None, str | None]:
    """(key, source): source is 'keychain', 'file', 'env' or None (SDK defaults apply)."""
    key, where = stored_api_key()
    if key:
        return key, where
    env = os.environ.get(ENV_VAR, "").strip()
    return (env, "env") if env else (None, None)


def save_api_key(key: str) -> str:
    """Save ``key``; returns 'keychain' or 'file' (the keychain when one works)."""
    key = key.strip()
    if not key:
        raise ValueError("empty API key")
    return keychain.set_secret(USERNAME, key, _kr=_keyring())


def delete_api_key():
    """Forget the saved key (keychain and file); ``ANTHROPIC_API_KEY`` is left alone."""
    keychain.delete_secret(USERNAME, _kr=_keyring())


def mask(key: str | None) -> str:
    """'sk-ant-…3f9a' for display; never show more."""
    if not key:
        return ""
    return f"{key[:7]}…{key[-4:]}" if len(key) > 16 else "…" + key[-2:]


def client(api_key: str | None = None, **kwargs):
    """``anthropic.Anthropic`` using ``api_key``, else the saved key, else the SDK's own lookup."""
    import anthropic

    key = api_key or get_api_key()[0]
    return anthropic.Anthropic(api_key=key, **kwargs) if key else anthropic.Anthropic(**kwargs)


def check(api_key: str | None = None, model: str = DEFAULT_MODEL, _client=None) -> tuple[bool, str]:
    """Test the key against the Models API (no tokens used); returns (ok, message)."""
    try:
        import anthropic
    except ImportError:
        return False, "The anthropic package is not installed: pip install 'ewald[llm]'"
    try:
        c = _client or client(api_key, timeout=15.0, max_retries=0)
        info = c.models.retrieve(model)
    except anthropic.AuthenticationError:
        return False, "The key was rejected (invalid or revoked)."
    except anthropic.PermissionDeniedError:
        return False, "The key has no access to the Claude API."
    except anthropic.NotFoundError:
        return False, f"The key works, but model {model} is not available to it."
    except anthropic.APIConnectionError:
        return False, "Could not reach api.anthropic.com (network or proxy)."
    except anthropic.APIStatusError as exc:
        return False, f"API error {exc.status_code}."
    except Exception as exc:  # e.g. no credentials at all
        msg = str(exc)
        if "api_key" in msg or "auth" in msg.lower():
            return False, "No API key: enter one, or set ANTHROPIC_API_KEY."
        return False, f"{type(exc).__name__}: {msg[:200]}"
    return True, f"Connected: {getattr(info, 'display_name', None) or model}"
