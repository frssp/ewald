"""ewald.llm.acp: Ask AI through an Agent Client Protocol agent (OpenCode), against a fake agent.

The fake agent (tests/helpers/acp_agent.py) is a small Python script written at test time and
started as the agent program (``AcpClient(command=sys.executable, args=[script])``).
"""
import json
import os
import sys
import threading

import pytest

from ewald.llm import acp
from ewald.llm.acp import ACP_NOTE, OPENCODE_PERMISSIONS, AcpClient, AcpError, models_of, resolve_command
from ewald.llm.agent import Agent, Cancelled
from helpers.acp_agent import write_agent


@pytest.fixture(autouse=True)
def _known_models():
    saved = dict(acp.KNOWN_MODELS)
    acp.KNOWN_MODELS.clear()
    yield
    acp.KNOWN_MODELS.clear()
    acp.KNOWN_MODELS.update(saved)


@pytest.fixture
def fake(tmp_path):
    """``make(**env)`` -> (AcpClient on the fake agent, function returning the logged messages)."""
    script = write_agent(tmp_path)
    log = tmp_path / "agent-log.jsonl"
    work = tmp_path / "work"
    work.mkdir()
    clients = []

    def received(method=None):
        if not log.exists():
            return []
        with open(log, encoding="utf-8") as f:
            messages = [json.loads(line) for line in f if line.strip()]
        return [m for m in messages if method is None or m.get("method") == method]

    def make(**env):
        client = AcpClient(sys.executable, [str(script)], env={"FAKE_LOG": str(log), **env}, timeout=20,
                           cwd=str(work))
        clients.append(client)
        return client, received

    yield make
    for client in clients:
        client.close()


def _agent(client, **kwargs):
    return Agent(provider="opencode", client=client, system=kwargs.pop("system", "SYS"), **kwargs)


def test_handshake_and_streamed_answer(fake):
    client, received = fake()
    events = []
    agent = _agent(client, stream=True, on_event=lambda kind, payload: events.append((kind, payload)))
    answer = agent.ask("hello there", images=[b"\x89PNG fake"])
    assert answer == "Echo: hello there"
    assert [p for k, p in events if k == "text"] == ["Echo: ", "hello there"]     # thoughts are not text
    assert ("answer", "Echo: hello there") in events
    assert client.agent_name == "FakeAgent 1.0" and client.images
    init = received("initialize")[0]["params"]
    assert init["protocolVersion"] == acp.PROTOCOL_VERSION and init["clientInfo"]["name"] == "ewald"
    assert init["clientCapabilities"]["terminal"] is False     # ewald offers the agent nothing to run
    new = received("session/new")[0]["params"]
    assert new["mcpServers"] == [] and os.path.samefile(new["cwd"], client.cwd)
    blocks = received("session/prompt")[0]["params"]["prompt"]
    assert blocks[1]["type"] == "image" and blocks[1]["mimeType"] == "image/png"
    assert blocks[-1] == {"type": "text", "text": "hello there"}
    assert agent.messages == [{"role": "user", "content": "hello there"},
                              {"role": "assistant", "content": "Echo: hello there"}]
    assert acp.KNOWN_MODELS == {"fake/a": "Fake A", "fake/b": "Fake B"}


def test_answer_without_streaming(fake):
    client, _ = fake()
    assert _agent(client).ask("plain") == "Echo: plain"


def test_system_prompt_goes_with_the_first_prompt_of_a_session(fake):
    client, received = fake()
    agent = _agent(client, system="You are ewald's assistant.")
    agent.ask("one")
    agent.ask("two")
    first, second = (m["params"] for m in received("session/prompt"))
    assert first["prompt"][0] == {"type": "text", "text": "You are ewald's assistant." + ACP_NOTE}
    assert "ewald-api.md" in ACP_NOTE
    assert second["prompt"] == [{"type": "text", "text": "two"}]
    assert first["sessionId"] == second["sessionId"] == "sess-1"
    agent.reset()                                    # a new conversation is a new session
    assert agent.messages == []
    agent.ask("three")
    third = received("session/prompt")[2]["params"]
    assert third["sessionId"] == "sess-2" and len(received("session/new")) == 2
    assert third["prompt"][0]["text"].endswith(ACP_NOTE)
    agent.system = "Changed."                        # a changed system prompt is told again
    agent.ask("four")
    assert received("session/prompt")[3]["params"]["prompt"][0]["text"] == "Changed." + ACP_NOTE


def test_model_is_set_as_the_sessions_config_option(fake):
    client, received = fake()
    agent = _agent(client, model="fake/b")
    agent.ask("hi")
    agent.ask("again")                               # set once per session
    sets = received("session/set_config_option")
    assert [m["params"] for m in sets] == [{"sessionId": "sess-1", "configId": "model", "value": "fake/b"}]
    assert agent.chat.models == {"fake/a": "Fake A", "fake/b": "Fake B"}


def test_agents_default_model_is_not_set_again(fake):
    client, received = fake()
    _agent(client, model="fake/a").ask("hi")         # the session's current model already
    assert received("session/set_config_option") == []
    client2, received2 = fake()
    _agent(client2).ask("hi")                        # no model: the agent's own default
    assert received2("session/set_config_option") == []


def test_model_falls_back_to_set_model_for_older_agents(fake):
    client, received = fake(FAKE_NO_CONFIG="1")
    _agent(client, model="fake/b").ask("hi")
    assert [m["params"] for m in received("session/set_model")] == [{"sessionId": "sess-1", "modelId": "fake/b"}]


def test_permission_requests_are_always_rejected(fake):
    client, received = fake()
    answer = _agent(client).ask("PERMIT me to run a command")
    assert answer.endswith("[rejected]")
    assert client.permissions_refused == [{"toolCallId": "t1", "title": "rm -rf /", "kind": "execute"}]
    reply = next(m for m in received() if m.get("id") == "perm-1")
    assert reply["result"] == {"outcome": {"outcome": "selected", "optionId": "no"}}


def test_permission_without_a_reject_option_is_cancelled(fake):
    client, _ = fake()
    client.start()
    replies = []
    client._write = replies.append                   # what the client answers, without the wire
    client._answer(7, "session/request_permission",
                   {"toolCall": {"title": "edit"}, "options": [{"optionId": "y", "kind": "allow_once"}]})
    client._answer(8, "fs/write_text_file", {"path": "x"})
    assert replies[0]["result"] == {"outcome": {"outcome": "cancelled"}}
    assert replies[1]["error"]["code"] == -32601      # nothing else is offered


def test_stop_cancels_the_prompt_and_keeps_the_conversation(fake):
    client, received = fake()
    cancel = threading.Event()

    def on_event(kind, payload):
        if kind == "text":
            cancel.set()                             # Stop pressed while the answer streams in

    agent = _agent(client, stream=True, on_event=on_event)
    agent.ask("first")
    cancel.clear()
    before = list(agent.messages)
    with pytest.raises(Cancelled):
        agent.ask("SLOW answer", cancel=cancel)
    assert agent.messages == before
    assert [m["params"] for m in received("session/cancel")] == [{"sessionId": "sess-1"}]
    cancel.clear()
    agent.on_event = lambda kind, payload: None
    assert agent.ask("after the stop") == "Echo: after the stop"   # the session still works


def test_api_reference_is_written_into_the_working_folder(fake):
    client, _ = fake()
    client.new_session()
    path = os.path.join(client.cwd, "ewald-api.md")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "pipe.add" in text or "Pipeline" in text
    assert acp.write_reference(client.cwd) == path   # written once per folder


def test_default_working_folder_is_a_fresh_temp_folder_removed_by_close(tmp_path):
    client = AcpClient(sys.executable, [write_agent(tmp_path)], timeout=20)
    try:
        client.new_session()
        folder = client.cwd
        assert os.path.basename(folder).startswith("ewald-acp-")
        assert os.path.isfile(os.path.join(folder, "ewald-api.md"))
    finally:
        client.close()
    assert not os.path.exists(folder) and client.cwd is None


def test_a_given_working_folder_is_kept(fake):
    client, _ = fake()
    client.new_session()
    folder = client.cwd
    client.close()
    assert client.cwd == folder and os.path.isfile(os.path.join(folder, "ewald-api.md"))


def test_chat_opens_a_new_session_after_the_agent_restarted(fake):
    client, received = fake()
    agent = _agent(client)
    agent.ask("one")
    client.close()
    assert agent.ask("two").startswith("Echo")
    assert len(received("session/new")) == 2


def test_models_of_flat_grouped_and_legacy():
    flat = {"configOptions": [{"id": "model", "currentValue": "a/x",
                               "options": [{"value": "a/x", "name": "X"}, {"value": "a/y"}]}]}
    assert models_of(flat) == ({"a/x": "X", "a/y": "a/y"}, "a/x")
    grouped = {"configOptions": [
        {"id": "thought", "category": "thought_level", "options": [{"value": "high"}]},
        {"id": "picker", "category": "model", "currentValue": "b/2", "options": [
            {"group": "anthropic", "name": "Anthropic", "options": [{"value": "b/1", "name": "One"}]},
            {"group": "openai", "name": "OpenAI", "options": [{"value": "b/2", "name": "Two"}]}]}]}
    assert models_of(grouped) == ({"b/1": "One", "b/2": "Two"}, "b/2")
    legacy = {"models": {"currentModelId": "c/1", "availableModels": [
        {"modelId": "c/1", "name": "C one"}, {"modelId": "c/2"}]}}
    assert models_of(legacy) == ({"c/1": "C one", "c/2": "c/2"}, "c/1")
    assert models_of({}) == ({}, "")


def test_resolve_command(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(AcpError, match="opencode was not found.*Agent command"):
        resolve_command("")
    with pytest.raises(AcpError, match="no-such-agent was not found"):
        resolve_command("no-such-agent")
    folder = tmp_path / "Program Files" / "open code"
    folder.mkdir(parents=True)
    program = folder / "opencode agent.exe"
    program.write_bytes(b"")
    assert resolve_command(str(program)) == str(program)
    assert resolve_command(f'  "{program}" ') == str(program)     # pasted with quotes


@pytest.mark.skipif(os.name == "nt", reason="a script with a #! line is a program on POSIX only")
def test_agent_program_in_a_folder_with_spaces(tmp_path):
    folder = tmp_path / "my agents" / "open code"
    folder.mkdir(parents=True)
    program = write_agent(folder, "fake agent", program=True)
    client = AcpClient(program, [], timeout=20, cwd=str(tmp_path))
    try:
        assert _agent(client).ask("spaces") == "Echo: spaces"
    finally:
        client.close()


def test_check_opencode(fake):
    from ewald.llm import providers

    p = providers.get("opencode")
    client, _ = fake()
    ok, message = p.check(_client=client)
    assert ok and "FakeAgent" in message and "2 models" in message and "fake/a" in message
    assert acp.KNOWN_MODELS == {"fake/a": "Fake A", "fake/b": "Fake B"}
    assert providers._acp_models() == acp.KNOWN_MODELS      # what Preferences offers under Model
    ok, message = p.check(model="fake/b", _client=client)
    assert ok and "model fake/b" in message
    ok, message = p.check(model="fake/zzz", _client=client)
    assert not ok and "no model fake/zzz" in message


def test_check_opencode_reports_a_missing_program(monkeypatch, tmp_path):
    from ewald.llm import providers

    monkeypatch.setenv("PATH", str(tmp_path))
    ok, message = providers.get("opencode").connect(command=str(tmp_path / "nope")).check()
    assert not ok and "was not found" in message


def test_from_settings_passes_the_command_to_the_client(monkeypatch):
    from ewald import settings
    from ewald.llm import credentials, providers

    monkeypatch.setattr(credentials, "_keyring", lambda: None)
    s = settings.load()
    kw = providers.from_settings(s, provider="opencode", command=r"C:\Tools\open code\opencode.cmd")
    assert kw["model"] is None
    client = kw["provider"].client()
    assert isinstance(client, AcpClient) and client.proc is None    # not started yet
    assert client.command == r"C:\Tools\open code\opencode.cmd" and client.args == ["acp"]
    s["llm.provider"], s["llm.command"] = "opencode", "/opt/opencode/bin/opencode"
    assert providers.from_settings(s)["provider"].client().command == "/opt/opencode/bin/opencode"
    assert providers.get("opencode").client().command == "opencode"     # empty: on PATH


def test_opencode_provider_defaults():
    from ewald.llm import providers

    p = providers.get("opencode")
    assert p.default_model == "" and p.sdk == "acp" and not p.key_required
    chat = p.chat(object())
    assert isinstance(chat, acp.AcpChat) and chat.model == "" and chat.tools is None


def test_opencode_config_denies_changes():
    client = acp._opencode_client()
    config = json.loads(client.env["OPENCODE_CONFIG_CONTENT"])
    assert config == {"permission": OPENCODE_PERMISSIONS}
    assert {OPENCODE_PERMISSIONS[k] for k in ("edit", "bash", "webfetch")} == {"deny"}


def test_close_stops_the_process_and_a_request_starts_it_again(fake):
    client, received = fake()
    client.new_session()
    proc = client.proc
    assert proc.poll() is None
    client.close()
    assert client.proc is None and proc.poll() is not None
    client.close()                                   # twice is fine
    client.new_session()                             # the next request starts a new agent
    assert client.proc is not proc and len(received("initialize")) == 2


def test_crashed_agent_error_names_its_stderr(fake):
    client, _ = fake(FAKE_CRASH="1")
    with pytest.raises(AcpError, match="no provider is configured"):
        client.new_session()


def test_agent_that_dies_while_answering(fake):
    client, _ = fake()
    agent = _agent(client)
    agent.ask("hi")
    with pytest.raises(AcpError, match="session/prompt: the agent stopped: panic: the model server went away"):
        agent.ask("DIE now")


def test_chats_share_one_agent_process_with_a_session_each(monkeypatch, tmp_path):
    from ewald.llm import providers

    monkeypatch.setattr(acp, "_SHARED", {})
    p = providers.from_settings(None, provider="opencode", command=sys.executable)["provider"]
    one, two = p.client(), p.client()
    assert one is two                                  # one process for every notebook tab's chat
    assert p.client(shared=False) is not one           # Test connection starts its own
    other = providers.from_settings(None, provider="opencode", command="elsewhere")["provider"]
    assert other.client() is not one


def test_two_chats_on_one_client_keep_their_own_sessions(fake):
    client, received = fake()
    a, b = _agent(client), _agent(client)
    a.ask("first tab")
    b.ask("second tab")
    a.ask("first tab again")
    assert a.chat.session_id != b.chat.session_id
    assert len(received("session/new")) == 2 and len(received("initialize")) == 1
