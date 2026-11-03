"""A fake Agent Client Protocol agent for tests (tests/test_acp.py, tests/test_gui.py).

:data:`FAKE_AGENT` is the agent's source: written to a file at test time and started as the agent
program. It logs every message it receives to ``FAKE_LOG`` (one JSON per line) and is steered by
the prompt text: ``PERMIT`` asks the client for a permission first, ``SLOW`` waits for
``session/cancel``, ``DIE`` exits with a message on stderr; ``FAKE_CRASH`` exits after
``initialize`` arrives, ``FAKE_NO_CONFIG`` refuses ``session/set_config_option``.
"""
import os
import sys

FAKE_AGENT = r'''
import json, os, sys

LOG = os.environ.get("FAKE_LOG")
MODELS = [{"value": "fake/a", "name": "Fake A"}, {"value": "fake/b", "name": "Fake B"}]
sessions = 0


def log(message):
    if LOG:
        with open(LOG, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(message) + "\n")


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def receive():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    message = json.loads(line)
    log(message)
    return message


def chunk(sid, text):
    send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": {
        "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}}})


def prompt(rid, params):
    sid = params["sessionId"]
    text = params["prompt"][-1]["text"]
    # something the client must ignore: another kind of update
    send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": {
        "sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "thinking"}}}})
    extra = ""
    if "DIE" in text:
        sys.stderr.write("panic: the model server went away\n")
        sys.stderr.flush()
        sys.exit(4)
    if "PERMIT" in text:
        send({"jsonrpc": "2.0", "id": "perm-1", "method": "session/request_permission", "params": {
            "sessionId": sid, "toolCall": {"toolCallId": "t1", "title": "rm -rf /", "kind": "execute"},
            "options": [{"optionId": "yes", "name": "Allow", "kind": "allow_once"},
                        {"optionId": "always", "name": "Always", "kind": "allow_always"},
                        {"optionId": "no", "name": "Reject", "kind": "reject_once"},
                        {"optionId": "never", "name": "Never", "kind": "reject_always"}]}})
        while True:
            m = receive()
            if m.get("id") == "perm-1" and "method" not in m:
                break
        outcome = (m.get("result") or {}).get("outcome") or {}
        extra = " [rejected]" if outcome.get("optionId") == "no" else f" [allowed {outcome}]"
    chunk(sid, "Echo: ")
    if "SLOW" in text:
        chunk(sid, "more")
        while True:
            m = receive()
            if m.get("method") == "session/cancel":
                send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "cancelled"}})
                return
    chunk(sid, text + extra)
    send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})


if os.environ.get("FAKE_CRASH"):
    receive()
    sys.stderr.write("fatal: no provider is configured\n")
    sys.stderr.flush()
    sys.exit(3)

while True:
    m = receive()
    method, rid, params = m.get("method"), m.get("id"), m.get("params") or {}
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": 1, "agentInfo": {"name": "FakeAgent", "version": "1.0"},
            "agentCapabilities": {"promptCapabilities": {"image": True}}}})
    elif method == "session/new":
        sessions += 1
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "sessionId": f"sess-{sessions}",
            "configOptions": [{"id": "mode", "category": "mode", "currentValue": "build", "options": []},
                              {"id": "model", "category": "model", "type": "select",
                               "currentValue": "fake/a", "options": MODELS}]}})
    elif method == "session/set_config_option":
        if os.environ.get("FAKE_NO_CONFIG"):
            send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}})
        else:
            send({"jsonrpc": "2.0", "id": rid, "result": {"configOptions": []}})
    elif method == "session/set_model":
        send({"jsonrpc": "2.0", "id": rid, "result": {}})
    elif method == "session/prompt":
        prompt(rid, params)
    elif rid is not None and method is not None:
        send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"no {method}"}})
'''


def write_agent(folder, name="fake_agent.py", program=False):
    """Write the fake agent into ``folder``; ``program=True`` makes it a program of its own (a
    ``#!`` line to this Python, executable; POSIX only). Returns the path."""
    path = os.path.join(str(folder), name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write((f"#!{sys.executable}\n" if program else "") + FAKE_AGENT)
    if program:
        os.chmod(path, 0o755)
    return path
