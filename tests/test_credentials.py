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
    for var in list(os.environ):
        if var.startswith("TICKET_"):
            monkeypatch.delenv(var)


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


def _ticket_provider():
    """A gateway-style plugin provider: its own client with a ticket header, models configured in
    the environment, a model prefix and a check by chat request."""
    from ewald.llm import providers

    def make_client(api_key=None, base_url=None, *, team=None, default_headers=None, **kwargs):
        import openai

        if not base_url:
            raise ValueError("No gateway address: set Server URL.")
        headers = {"X-Ticket": api_key or "", "X-Team": team or "none", **(default_headers or {})}
        return openai.OpenAI(api_key="dummy", base_url=base_url, default_headers=headers, **kwargs)

    def env_models():
        return {m: "" for m in os.environ.get("TICKET_MODELS", "").split(",") if m}

    return providers.Provider(
        "ticket", "Ticket gateway", make_client=make_client, key_title="credential", key_name="ticket-credential",
        env_var="TICKET_CREDENTIAL", vision=False, options=("route_prefix", "team"), env_models=env_models,
        model_url=lambda model: os.environ.get("TICKET_URL", ""), env_prefix=lambda: "vendor/",
        check_with="chat", origin="python")


def test_plugin_provider_with_its_own_client(no_keychain, monkeypatch):
    """Provider fields a gateway plugin uses: make_client options, env models, model URL and prefix."""
    pytest.importorskip("openai")
    gw = _ticket_provider()
    with pytest.raises(ValueError, match="No gateway address"):
        gw.client()
    monkeypatch.setenv("TICKET_URL", "http://gw.example/v1")
    monkeypatch.setenv("TICKET_CREDENTIAL", "credential:T-1")
    monkeypatch.setenv("TICKET_MODELS", "model-a,model-b")
    assert credentials.get_api_key(gw) == ("credential:T-1", "env")
    assert gw.default_model == "model-a" and list(gw.all_models)[:2] == ["model-a", "model-b"]
    assert gw.model_prefix() == "vendor/"
    c = gw.client()
    assert str(c.base_url).startswith("http://gw.example/v1") and c.default_headers["X-Ticket"] == "credential:T-1"
    assert c.api_key == "dummy" and "X-Ticket" not in os.environ
    # Preferences win over the environment; options the provider does not list are ignored
    conn = gw.connect(base_url="http://other/v1", team="md", route_prefix="openai/", headers="X-Site: a",
                      timeout=5, user="ignored")
    h = conn.client().default_headers
    assert (h["X-Team"], h["X-Site"]) == ("md", "a") and conn.client_options.get("user") is None
    assert conn.model_prefix() == "openai/" and conn.chat(None, "gpt-oss-120b").options["model_prefix"] == "openai/"


def test_plugin_provider_round_trip_sends_the_headers(no_keychain, monkeypatch, tmp_path):
    openai = pytest.importorskip("openai")
    import httpx2 as httpx

    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((dict(request.headers), body))
        msg = {"role": "assistant", "content": "Cu, fcc."}
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                                         "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]})

    monkeypatch.setenv("TICKET_CREDENTIAL", "credential:T-rt")
    conn = _ticket_provider().connect(base_url="http://gw.example/v1", route_prefix="openai/", team="u1")
    client = conn.client(http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert isinstance(client, openai.OpenAI)
    ok, msg = conn.check(model="gpt-oss-120b", _client=client)
    assert ok and msg == "Connected: gpt-oss-120b"
    headers, body = seen[-1]
    assert body["model"] == "openai/gpt-oss-120b" and headers["x-ticket"] == "credential:T-rt"
    assert headers["x-team"] == "u1"


def test_openai_compatible_key_header_fresh_ids_and_chat_check(no_keychain, monkeypatch):
    """A gateway through the OpenAI-compatible provider: the key in its own header, a new id per
    request from {uuid}, and Test connection falling back to a chat request without /models."""
    openai = pytest.importorskip("openai")
    import httpx2 as httpx

    from ewald.llm import providers

    seen = []

    def handler(request):
        if request.url.path.endswith("/models/m1"):
            return httpx.Response(404, json={"error": {"message": "no models endpoint"}})
        body = json.loads(request.content)
        seen.append((dict(request.headers), body))
        msg = {"role": "assistant", "content": "ok"}
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                                         "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]})

    conn = providers.get("openai-compatible").connect(
        base_url="http://gw.example/v1", key_header="x-ticket", route_prefix="openai/",
        headers="X-System: ewald; X-Request-Id: {uuid}")
    client = conn.client(api_key="credential:T-1", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert isinstance(client, openai.OpenAI)
    ok, msg = conn.check(model="m1", _client=client)
    assert ok and msg == "Connected: m1"
    chat = conn.chat(client, "m1")
    chat.user("hi")
    chat.send()
    (h1, b1), (h2, b2) = seen[-2:]
    assert b1["model"] == b2["model"] == "openai/m1"
    assert h1["x-ticket"] == h2["x-ticket"] == "credential:T-1" and "T-1" not in h1.get("authorization", "")
    assert h1["x-system"] == "ewald" and h1["x-request-id"] != h2["x-request-id"] != "{uuid}"
    assert len(h2["x-request-id"]) == 36
    # without key_header the key stays a bearer token
    plain = providers.get("openai-compatible").connect(base_url="http://gw.example/v1")
    assert plain.client(api_key="k").api_key == "k"


def test_saved_connections_and_from_settings(monkeypatch):
    from ewald.llm import providers

    s = settings.Settings(path="/nonexistent/settings.json")
    ai = providers.from_settings(s)
    assert (ai["provider"].name, ai["model"], ai["effort"]) == ("anthropic", None, "high")
    s["llm.provider"], s["llm.model"], s["llm.base_url"] = "openai-compatible", "qwen3:8b", "http://box:8000/v1"
    s["llm.profiles"] = {"hub": {"provider": "openai-compatible", "model": "gpt-oss-120b",
                                 "base_url": "http://gw/v1", "headers": "X-Team: md", "route_prefix": "openai/",
                                 "user_id": "an old field"}}
    ai = providers.from_settings(s)
    assert (ai["provider"].name, ai["model"], ai["provider"].base_url) == ("openai-compatible", "qwen3:8b",
                                                                          "http://box:8000/v1")
    hub = providers.from_settings(s, profile="hub")
    assert (hub["provider"].name, hub["model"], hub["provider"].base_url) == ("openai-compatible", "gpt-oss-120b",
                                                                             "http://gw/v1")
    assert hub["provider"].client_options == {"default_headers": {"X-Team": "md"}}
    assert hub["provider"].model_prefix() == "openai/"
    assert providers.from_settings(s, profile="hub", model="qwen3:8b")["model"] == "qwen3:8b"
    other = providers.from_settings(s, provider="openai")   # the saved model and address are not OpenAI's
    assert (other["provider"].name, other["model"], other["provider"].base_url) == ("openai", None, "")
    with pytest.raises(KeyError, match="hub"):
        providers.from_settings(s, profile="nope")
    assert set(providers.profile_from_settings(s)) == set(providers.PROFILE_FIELDS)
    assert providers.parse_headers("A: 1; B:2;junk") == {"A": "1", "B": "2"}
    # settings of the gateway provider that moved to a plugin are dropped quietly
    assert "llm.user_id" not in settings.Settings({"llm.user_id": "me", "llm.system_name": "x"})._values


def test_forge_subprocess_cannot_use_keyring():
    from ewald.llm import forge

    env = forge._subprocess_env("/tmp")
    assert env["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
    assert "ANTHROPIC_API_KEY" not in env


def test_cli_profile_flag():
    from ewald import __main__ as cli

    s = settings.load()
    s["llm.profiles"] = {"local": {"provider": "openai-compatible", "model": "qwen3:8b",
                                   "base_url": "http://box:8000/v1"}}
    s.save()
    args = cli.build_parser().parse_args(["ask", "POSCAR", "hi", "--profile", "local"])
    ai = cli._ai_options(args)
    assert (ai["provider"].name, ai["model"], ai["provider"].base_url) == ("openai-compatible", "qwen3:8b",
                                                                          "http://box:8000/v1")
    args = cli.build_parser().parse_args(["forge", "x", "--profile", "local", "--model", "llama3.1"])
    assert cli._ai_options(args, effort=False) == {"provider": ai["provider"], "model": "llama3.1"}
