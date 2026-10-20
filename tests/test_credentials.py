"""API key storage (ewald.llm.credentials): keychain first, 0600 file fallback, never settings.json."""
import json
import os
import stat
import sys
from types import SimpleNamespace as NS

import pytest

from ewald import settings
from ewald.llm import credentials


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, service, user):
        return self.store.get((service, user))

    def set_password(self, service, user, value):
        self.store[(service, user)] = value

    def delete_password(self, service, user):
        self.store.pop((service, user))

    def get_keyring(self):
        return self


@pytest.fixture
def keychain(monkeypatch):
    kr = FakeKeyring()
    monkeypatch.setattr(credentials, "_keyring", lambda: kr)
    return kr


@pytest.fixture
def no_keychain(monkeypatch):
    monkeypatch.setattr(credentials, "_keyring", lambda: None)


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_saved_in_keychain_not_in_settings(keychain):
    assert credentials.get_api_key() == (None, None)
    assert credentials.save_api_key("  sk-ant-test-1234567890abcd ") == "keychain"
    assert keychain.store[("ewald", "anthropic-api-key")] == "sk-ant-test-1234567890abcd"
    assert credentials.get_api_key() == ("sk-ant-test-1234567890abcd", "keychain")
    assert not (settings.config_dir() / "credentials.json").exists()
    s = settings.load()
    s["llm.model"] = "claude-sonnet-5-5"
    s.save()
    assert "sk-ant" not in s.path.read_text()
    credentials.delete_api_key()
    assert credentials.get_api_key() == (None, None)


def test_file_fallback_is_private(no_keychain):
    assert credentials.save_api_key("sk-ant-file-key-000000000") == "file"
    path = settings.config_dir() / "credentials.json"
    assert json.loads(path.read_text()) == {"anthropic-api-key": "sk-ant-file-key-000000000"}
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert credentials.get_api_key() == ("sk-ant-file-key-000000000", "file")
    credentials.delete_api_key()
    assert not path.exists()


def test_saved_key_wins_over_env(keychain, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env-99999999")
    assert credentials.get_api_key() == ("sk-ant-from-env-99999999", "env")
    credentials.save_api_key("sk-ant-saved-11111111111")
    assert credentials.get_api_key()[1] == "keychain"
    with pytest.raises(ValueError):
        credentials.save_api_key("   ")


def test_mask_never_shows_the_key():
    key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz-WXYZ"
    assert credentials.mask(key) == "sk-ant-…WXYZ"
    assert credentials.mask("short") == "…rt" and credentials.mask(None) == ""


def test_client_uses_saved_key(keychain):
    pytest.importorskip("anthropic")
    credentials.save_api_key("sk-ant-saved-22222222222")
    assert credentials.client().api_key == "sk-ant-saved-22222222222"
    assert credentials.client("sk-ant-explicit-333333").api_key == "sk-ant-explicit-333333"
    assert "ANTHROPIC_API_KEY" not in os.environ  # handed to the client, not to subprocesses


def test_check_reports_each_failure():
    anthropic = pytest.importorskip("anthropic")
    import httpx2 as httpx

    req = httpx.Request("GET", "https://api.anthropic.com/v1/models/x")

    def raising(exc):
        return NS(models=NS(retrieve=lambda model: (_ for _ in ()).throw(exc)))

    ok = NS(models=NS(retrieve=lambda model: NS(id=model, display_name="Claude Opus 5.5")))
    assert credentials.check(_client=ok) == (True, "Connected: Claude Opus 5.5")
    resp = httpx.Response(401, request=req)
    bad = anthropic.AuthenticationError("bad key", response=resp, body=None)
    assert credentials.check(_client=raising(bad)) == (False, "The key was rejected (invalid or revoked).")
    missing = anthropic.NotFoundError("no model", response=httpx.Response(404, request=req), body=None)
    okflag, msg = credentials.check(model="claude-x", _client=raising(missing))
    assert not okflag and "claude-x" in msg
    offline = anthropic.APIConnectionError(request=req)
    assert "Could not reach" in credentials.check(_client=raising(offline))[1]


def test_ai_settings_and_providers():
    from ewald.llm import providers

    assert settings.OPTION["llm.provider"].default == "anthropic" and settings.OPTION["llm.model"].default == ""
    assert {settings.OPTION[k].page for k in ("llm.provider", "llm.model", "llm.base_url", "llm.effort")} == {"AI"}
    assert providers.get().models == credentials.MODELS and providers.get().default_model == credentials.DEFAULT_MODEL
    assert {"anthropic", "openai", "gemini", "openai-compatible"} <= set(providers.providers())
    with pytest.raises(KeyError):
        providers.get("no-such-ai")


def test_each_provider_keeps_its_own_key(keychain, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    credentials.save_api_key("sk-ant-anthropic-1111111111")
    assert credentials.save_api_key("sk-openai-2222222222222", provider="openai") == "keychain"
    assert keychain.store[("ewald", "openai-api-key")] == "sk-openai-2222222222222"
    assert credentials.get_api_key()[0] == "sk-ant-anthropic-1111111111"
    assert credentials.get_api_key("openai") == ("sk-openai-2222222222222", "keychain")
    assert credentials.get_api_key("gemini") == (None, None)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-from-env-333333")
    assert credentials.get_api_key("gemini") == ("AIza-from-env-333333", "env")
    credentials.delete_api_key("openai")
    assert credentials.get_api_key("openai") == (None, None) and credentials.get_api_key()[0]


def test_openai_client_gets_the_key_and_address(keychain, monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    credentials.save_api_key("sk-openai-saved-44444444", provider="openai")
    c = credentials.client(provider="openai")
    assert c.api_key == "sk-openai-saved-44444444" and "OPENAI_API_KEY" not in os.environ
    local = credentials.client(provider="openai-compatible", base_url="http://localhost:1234/v1")
    assert str(local.base_url).startswith("http://localhost:1234/v1") and local.api_key == "not-needed"
    gemini = credentials.client("AIza-explicit", provider="gemini")
    assert "generativelanguage.googleapis.com" in str(gemini.base_url) and gemini.api_key == "AIza-explicit"


def test_check_with_an_openai_compatible_server():
    openai = pytest.importorskip("openai")
    import httpx2 as httpx

    ok = NS(models=NS(retrieve=lambda model: NS(id=model)))
    assert credentials.check(model="llama3.1", _client=ok, provider="openai-compatible") == (True, "Connected: llama3.1")
    req = httpx.Request("GET", "http://localhost:11434/v1/models/x")
    offline = openai.APIConnectionError(request=req)
    failing = NS(models=NS(retrieve=lambda model: (_ for _ in ()).throw(offline)))
    msg = credentials.check(_client=failing, provider="openai-compatible")[1]
    assert "Could not reach http://localhost:11434/v1" in msg


def test_forge_subprocess_cannot_use_keyring():
    from ewald.llm import forge

    env = forge._subprocess_env("/tmp")
    assert env["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
    assert "ANTHROPIC_API_KEY" not in env
