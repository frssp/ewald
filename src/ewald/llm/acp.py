"""Ask AI through a coding agent that speaks the Agent Client Protocol (ACP): OpenCode.

ACP (https://agentclientprotocol.com) is to coding agents what LSP is to language servers: the
app starts the agent program (``opencode acp``) and both exchange JSON-RPC 2.0 messages, one per
line, over its stdin and stdout. The agent brings its own models, sign-in and provider settings
(OpenCode's ``opencode auth login``, a company's OpenCode configuration), so the only setting
here is the command (Preferences > AI > Agent command, ``llm.command``).

What ewald uses of the protocol: ``initialize``, ``session/new`` (a working folder, no MCP
servers), ``session/prompt`` (text and PNG images), the ``session/update`` notifications
(``agent_message_chunk`` is the answer's text), ``session/cancel`` (Stop) and the model as the
session's ``model`` config option (``session/set_config_option``; ``session/set_model`` for
older agents). The agent's own tools that would change something (``edit``, ``bash``, ...) are
switched off for OpenCode (``OPENCODE_CONFIG_CONTENT``) and every ``session/request_permission``
is answered with a refusal, so the AI changes things only through the notebook cells its answers
hold (D4, D40): the ```python blocks become cells as with every other provider. The agent gets
ewald's API reference as ``ewald-api.md`` in its working folder (a fresh ``ewald-acp-*`` folder)
and reads it with its own read tool.

:class:`AcpClient` is the connection (one agent process, started on first use, stopped by
:meth:`AcpClient.close` or at exit); :class:`AcpChat` is the :class:`~ewald.llm.chat.Chat` the
agent loop talks to. An answer never holds tool calls for ewald's :class:`~ewald.llm.Toolset`.
"""
from __future__ import annotations

import atexit
import base64
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import weakref
from typing import Callable

from .chat import Chat, Reply

PROTOCOL_VERSION = 1

#: What OpenCode may do on its own: read and search its working folder, nothing that changes
#: something (merged over the user's own OpenCode configuration).
OPENCODE_PERMISSIONS = {"edit": "deny", "bash": "deny", "webfetch": "deny", "external_directory": "deny"}

#: Told to the agent with the question after one the user stopped (the agent's session still has it).
STOPPED_NOTE = "(The user stopped your previous answer before it ended; it is not part of the conversation " \
               "any more. Answer only the question below.)"

#: Told to the agent with the first question of a session (after the agent's system prompt).
ACP_NOTE = """

You are reached through the Agent Client Protocol, so you have none of the tools named above \
(python_api, search_api, insert_cell, edit_cell, run_cell, render_image, ...): you act only by \
answering. ewald's whole Python API reference is the file ewald-api.md in your working folder: \
read it before you write ewald code (search it for the step or function you need). Never edit \
files or run commands yourself; put every change in ```python blocks in your answer, one cell \
each, which ewald inserts into the user's notebook (and runs them when it says so above). Each \
question carries the notebook as it reads now and the outputs of its cells."""


class AcpError(RuntimeError):
    """The agent could not be started, answered with an error or stopped."""


def resolve_command(command: str | None, default: str = "opencode") -> str:
    """The agent program: ``command`` as an existing file, else looked up on PATH (with PATHEXT
    on Windows, so an npm ``opencode.cmd`` is found); raises :class:`AcpError` when there is none."""
    command = (command or "").strip().strip('"') or default
    expanded = os.path.expanduser(command)
    if os.path.isfile(expanded):
        return os.path.abspath(expanded)
    found = shutil.which(expanded)
    if found:
        return found
    raise AcpError(f"{command} was not found. Install OpenCode (npm install -g opencode-ai) or enter the "
                   f"full path of the opencode program in Preferences > AI > Agent command "
                   f"({'where.exe opencode' if os.name == 'nt' else 'which opencode'} shows it).")


#: Models the agents offered when last asked (Test connection, a new session): {id: label}.
KNOWN_MODELS: dict = {}

_LIVE: "weakref.WeakSet[AcpClient]" = weakref.WeakSet()


@atexit.register
def _close_all():
    for client in list(_LIVE):
        client.close()


class AcpClient:
    """One agent process spoken to over ACP. ``command`` is the program (path or name on PATH),
    ``args`` its arguments (``["acp"]`` for OpenCode); ``env`` adds environment variables;
    ``timeout`` bounds a single request (seconds, None: no limit; a whole answer is one request).
    The process starts on the first request; :meth:`close` stops it and the next request starts
    it again. Thread-safe: requests from several threads take turns per request id."""

    def __init__(self, command: str | None = None, args=("acp",), *, env: dict | None = None,
                 timeout: float | None = None, popen=None, cwd: str | None = None):
        self.command = command
        self.args = list(args)
        self.env = dict(env or {})
        self.timeout = timeout or None
        self._popen = popen or subprocess.Popen
        self.cwd = cwd
        self._own_cwd = False           # a temporary folder made here, removed by close()
        self.generation = 0             # counts agent processes: a session lives in one
        self.proc = None
        self.info: dict = {}            # the initialize result
        self._next = 0
        self._pending: dict[int, queue.Queue] = {}
        self._listeners: dict[str, Callable[[dict], None]] = {}
        self._lock = threading.Lock()
        self._starting = threading.RLock()
        self._stderr: list[str] = []
        self.permissions_refused: list[dict] = []
        _LIVE.add(self)

    # ---------------------------------------------------------------- process
    def start(self):
        with self._starting:   # two chats asking at once start one process
            if self.proc is not None and self.proc.poll() is None:
                return
            self._start()

    def _start(self):
        program = resolve_command(self.command)
        env = {**os.environ, **self.env}
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        if self.cwd is None:
            self.cwd = tempfile.mkdtemp(prefix="ewald-acp-")
            self._own_cwd = True
        try:
            self.proc = self._popen([program, *self.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=env, cwd=self.cwd, creationflags=flags)
        except OSError as exc:
            raise AcpError(f"Could not start {program}: {exc}") from exc
        self.generation += 1
        with self._lock:   # a request registering now goes to the new process's table
            old, self._pending = self._pending, {}
        for q in old.values():   # requests to the old process: they fail now, not never
            q.put({"error": {"message": "the agent was restarted"}})
        self._stderr = []
        errors = threading.Thread(target=self._read_stderr, args=(self.proc,), daemon=True,
                                  name="ewald-acp-stderr")
        errors.start()
        threading.Thread(target=self._read, args=(self.proc, errors), daemon=True, name="ewald-acp-read").start()
        self.info = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
            "clientInfo": {"name": "ewald", "version": _version()}})

    def close(self):
        """Stop the agent process (and its children on Windows) and remove the temporary working
        folder made for it."""
        proc, self.proc = self.proc, None
        try:
            self._stop(proc)
        finally:
            if self._own_cwd and self.cwd:
                shutil.rmtree(self.cwd, ignore_errors=True)
                self.cwd, self._own_cwd = None, False

    def _stop(self, proc):
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":   # opencode.cmd -> node -> opencode: stop the whole tree
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
            else:
                proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        for q in list(self._pending.values()):
            q.put({"error": {"message": "the agent was stopped"}})

    @property
    def stderr(self) -> str:
        return "".join(self._stderr)[-2000:]

    # ---------------------------------------------------------------- wire
    def _write(self, message: dict):
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        proc = self.proc
        if proc is None:
            raise AcpError("the agent is not running")
        with self._lock:
            try:
                proc.stdin.write(data)
                proc.stdin.flush()
            except (OSError, ValueError) as exc:
                raise AcpError(f"the agent stopped{self._why()}") from exc

    def _why(self) -> str:
        text = self.stderr.strip()
        return f": {text.splitlines()[-1][:300]}" if text else ""

    def _read(self, proc, errors=None):
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                continue    # a stray log line
            try:
                self._dispatch(message)
            except Exception:   # a bad message must not stop the reader
                pass
        if errors is not None:  # the agent's last words on stderr belong in the error
            errors.join(timeout=1)
        for q in list(self._pending.values()):
            q.put({"error": {"message": f"the agent stopped{self._why()}"}})

    def _read_stderr(self, proc):
        for raw in proc.stderr:
            self._stderr.append(raw.decode("utf-8", "replace"))
            if len(self._stderr) > 200:
                del self._stderr[:100]

    def _dispatch(self, message: dict):
        if "method" not in message:     # a response to one of ours
            q = self._pending.pop(message.get("id"), None)
            if q is not None:
                q.put(message)
            return
        method, params = message["method"], message.get("params") or {}
        if "id" in message:             # a request from the agent
            self._answer(message["id"], method, params)
        elif method == "session/update":
            listener = self._listeners.get(params.get("sessionId"))
            if listener is not None:
                listener(params.get("update") or {})

    def _answer(self, rid, method, params):
        if method == "session/request_permission":
            # the AI changes things only through notebook cells: refuse whatever the agent asks for
            self.permissions_refused.append(params.get("toolCall") or {})
            options = params.get("options") or []
            reject = next((o for o in options if o.get("kind") == "reject_once"), None) or \
                next((o for o in options if str(o.get("kind", "")).startswith("reject")), None)
            outcome = ({"outcome": "selected", "optionId": reject["optionId"]} if reject
                       else {"outcome": "cancelled"})
            self._write({"jsonrpc": "2.0", "id": rid, "result": {"outcome": outcome}})
        else:   # fs/*, terminal/*: not offered in initialize
            self._write({"jsonrpc": "2.0", "id": rid,
                         "error": {"code": -32601, "message": f"ewald does not offer {method}"}})

    def _request(self, method: str, params: dict, timeout: float | None = None):
        with self._lock:
            self._next += 1
            rid = self._next
            q = self._pending[rid] = queue.Queue()
        self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        try:
            reply = q.get(timeout=timeout or self.timeout)
        except queue.Empty:
            self._pending.pop(rid, None)
            raise AcpError(f"the agent did not answer {method} in time") from None
        if "error" in reply:
            err = reply["error"] or {}
            raise AcpError(f"{method}: {err.get('message', err)}")
        return reply.get("result") or {}

    def request(self, method: str, params: dict, timeout: float | None = None):
        """Send ``method`` and wait for its result (starts the agent first)."""
        self.start()
        return self._request(method, params, timeout)

    def notify(self, method: str, params: dict):
        if self.proc is not None:
            self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def listen(self, session_id: str, listener: Callable[[dict], None] | None):
        if listener is None:
            self._listeners.pop(session_id, None)
        else:
            self._listeners[session_id] = listener

    # ---------------------------------------------------------------- sessions
    def new_session(self) -> dict:
        """A new session in the client's working folder (with ``ewald-api.md``); its result holds
        ``sessionId`` and the agent's ``configOptions`` / ``models``."""
        self.start()
        write_reference(self.cwd)
        return self._request("session/new", {"cwd": os.path.abspath(self.cwd), "mcpServers": []})

    @property
    def images(self) -> bool:
        caps = (self.info.get("agentCapabilities") or {}).get("promptCapabilities") or {}
        return bool(caps.get("image"))

    @property
    def agent_name(self) -> str:
        info = self.info.get("agentInfo") or {}
        return " ".join(str(v) for v in (info.get("title") or info.get("name"), info.get("version")) if v)


def models_of(session: dict) -> tuple[dict, str]:
    """``({model id: label}, current id)`` from a ``session/new`` result: the ``model`` config
    option (grouped or flat), else the older ``models`` field."""
    for opt in session.get("configOptions") or []:
        if opt.get("category") == "model" or opt.get("id") == "model":
            out = {}
            for o in opt.get("options") or []:
                for item in (o.get("options") if "group" in o else [o]) or []:
                    out[item.get("value")] = item.get("name") or item.get("value")
            return out, opt.get("currentValue") or ""
    models = session.get("models") or {}
    out = {m.get("modelId"): m.get("name") or m.get("modelId") for m in models.get("availableModels") or []}
    return out, models.get("currentModelId") or ""


def write_reference(folder: str):
    """ewald's API reference (what ``ewald api`` prints) as ``ewald-api.md`` in ``folder``."""
    from .reference import api_reference

    path = os.path.join(folder, "ewald-api.md")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(api_reference(with_plugins=True))
    return path


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("ewald")
    except Exception:
        return "0"


class AcpChat(Chat):
    """One conversation as one ACP session. ``client`` is an :class:`AcpClient`; ``model`` is the
    agent's model id ('' or None: the agent's own default). Tools are not offered (see the module
    docstring); the system prompt goes with the first question of a session, followed by
    :data:`ACP_NOTE`. :meth:`reset` starts a new session."""

    def __init__(self, client, model, **kwargs):
        kwargs.pop("tools", None)
        super().__init__(client, model or "", tools=None, **kwargs)
        self.session_id: str | None = None
        self._generation = 0
        self.models: dict = {}
        self._model_set = None
        self._told = None
        self._stopped_turn = False      # the last prompt was stopped: the agent's session still holds it
        self._pending: list[dict] = []
        self._images: list[bytes] = []

    def user(self, text, images=None):
        self._pending = [{"type": "text", "text": text}]
        self._images = list(images or [])
        self.messages.append({"role": "user", "content": text})

    def _image_blocks(self) -> list[dict]:
        # asked after _session(): before the agent has started, its capabilities are unknown
        if not (self._images and self.vision and getattr(self.client, "images", True)):
            return []
        return [{"type": "image", "mimeType": "image/png", "data": base64.standard_b64encode(im).decode()}
                for im in self._images]

    def _session(self):
        if hasattr(self.client, "start"):
            self.client.start()         # (again, when the agent stopped)
        if self.session_id is not None and getattr(self.client, "generation", 0) != self._generation:
            self.session_id = None      # the agent was restarted: its sessions are gone
        if self.session_id is None:
            result = self.client.new_session()
            self._generation = getattr(self.client, "generation", 0)
            self.session_id = result["sessionId"]
            self.models, current = models_of(result)
            KNOWN_MODELS.update(self.models)
            self._model_set = current
            self._told = None
            self._stopped_turn = False  # a new session never saw the stopped answer
        if self.model and self.model != self._model_set:
            self._set_model(self.model)

    def _set_model(self, model):
        try:
            self.client.request("session/set_config_option",
                                {"sessionId": self.session_id, "configId": "model", "value": model})
        except AcpError:
            self.client.request("session/set_model", {"sessionId": self.session_id, "modelId": model})
        self._model_set = model

    def send(self, tool_choice=None, on_text=None):
        self._session()
        blocks = self._image_blocks() + list(self._pending)
        instructions = (self.system or "") + ACP_NOTE
        if self._stopped_turn:
            # Stop takes the question out of ewald's conversation, but the agent's own session
            # keeps it and its half answer: say so, or the two conversations differ
            blocks.insert(0, {"type": "text", "text": STOPPED_NOTE})
        if instructions != self._told:
            blocks.insert(0, {"type": "text", "text": instructions})
        pieces: list[str] = []
        failure: list[BaseException] = []
        stream = self.streams(on_text)

        def update(u):
            if u.get("sessionUpdate") != "agent_message_chunk":
                return
            content = u.get("content") or {}
            piece = content.get("text", "") if content.get("type", "text") == "text" else ""
            if not piece or failure:
                return
            pieces.append(piece)
            if stream:
                try:
                    on_text(piece)
                except BaseException as exc:    # a Stop: end the turn on the agent's side
                    failure.append(exc)
                    self.client.notify("session/cancel", {"sessionId": self.session_id})

        self.client.listen(self.session_id, update)
        try:
            result = self.client.request("session/prompt", {"sessionId": self.session_id, "prompt": blocks})
        finally:
            self.client.listen(self.session_id, None)
        if failure:
            self._stopped_turn = True
            raise failure[0]
        self._told, self._stopped_turn = instructions, False
        self._pending, self._images = [], []
        text = "".join(pieces)
        self.messages.append({"role": "assistant", "content": text})
        stop = {"max_tokens": "max_tokens", "refusal": "refusal"}.get(result.get("stopReason"), "end")
        return Reply(text, [], stop)

    def forget_stopped_turn(self):
        """The agent loop dropped a stopped question from :attr:`messages`; the agent's session
        still has it, so the next prompt says it is gone."""
        self._stopped_turn = True

    def tool_results(self, results):    # never asked for: answers hold no tool calls
        raise AcpError("an ACP agent's answers hold no ewald tool calls")

    def reset(self):
        super().reset()
        self.session_id = None
        self._stopped_turn = False
        self._pending, self._images = [], []


#: One agent process per (command, timeout), shared by every chat (each notebook tab has its own
#: chat and so its own session in that process).
_SHARED: dict = {}
_SHARED_LOCK = threading.Lock()


def _opencode_client(api_key=None, base_url=None, *, command=None, timeout=None, shared=True, **_ignored):
    env = {"OPENCODE_CONFIG_CONTENT": json.dumps({"permission": OPENCODE_PERMISSIONS})}
    if not shared:
        return AcpClient(command or "opencode", ["acp"], env=env, timeout=timeout)
    key = (command or "opencode", timeout or None)
    with _SHARED_LOCK:
        client = _SHARED.get(key)
        if client is None:
            client = _SHARED[key] = AcpClient(command or "opencode", ["acp"], env=env, timeout=timeout)
        return client


def check_opencode(provider, api_key=None, model=None, base_url=None, _client=None) -> tuple[bool, str]:
    """Start the agent, open a session and say which models it offers (no tokens used)."""
    client = _client or provider.client(shared=False)   # its own process, stopped right after
    try:
        session = client.new_session()
    except AcpError as exc:
        return False, str(exc)
    finally:
        if _client is None:
            client.close()
    models, current = models_of(session)
    KNOWN_MODELS.clear()
    KNOWN_MODELS.update(models)
    if model and models and model not in models:
        return False, f"Connected to {client.agent_name or 'the agent'}, but it has no model {model}."
    shown = model or current
    more = f" ({len(models)} models, listed under Model)" if models else ""
    return True, f"Connected: {client.agent_name or 'agent'}, model {shown or 'its default'}{more}"


__all__ = ["AcpClient", "AcpChat", "AcpError", "ACP_NOTE", "OPENCODE_PERMISSIONS", "models_of", "resolve_command",
           "check_opencode", "write_reference"]
