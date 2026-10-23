"""ewald.llm: descriptors, toolset, agent loop (mocked LLM) and MCP adapter."""
import base64
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write

from ewald import DataCollection, Pipeline, StaticSource
from ewald.llm import Agent, Toolset, describe_frame, describe_trajectory, detect_change_points, suggest_cutoff
from ewald.llm import tools as tools_mod


@pytest.mark.parametrize("structure,a,expected", [("fcc", 3.615, 12), ("diamond", 5.43, 4), ("bcc", 2.87, 8)])
def test_suggest_cutoff_gives_first_shell(structure, a, expected):
    data = DataCollection.from_ase(bulk("Si" if structure == "diamond" else "Cu", structure, a=a, cubic=True).repeat(4))
    desc = describe_frame(data, cutoff=suggest_cutoff(data))
    assert desc["coordination"]["by_type"][data.types[0]]["histogram"] == {str(expected): data.count}


def test_describe_frame(cu_fcc):
    d = describe_frame(cu_fcc)
    assert d["composition"] == {"Cu": 256}
    assert d["cell"]["a"] == pytest.approx(14.46)
    assert d["density_g_cm3"] == pytest.approx(8.93, abs=0.02)
    assert d["min_distance"]["value"] == pytest.approx(3.615 / np.sqrt(2), abs=1e-3)
    assert d["coordination"]["mean"] == 12
    assert d["warnings"] == []
    json.dumps(d)  # JSON-ready


def test_health_check_flags_overlap(cu_fcc):
    data = cu_fcc.copy()
    pos = data.positions.copy()
    pos[1] = pos[0] + 0.1
    data.positions = pos
    assert any("closer than" in w for w in describe_frame(data, cutoff=None)["warnings"])


def test_change_points():
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(0, 0.1, 50), rng.normal(2, 0.1, 50)]
    assert detect_change_points(x) == [50]
    assert detect_change_points(rng.normal(0, 0.1, 100)) == []


def _jump_trajectory(base, n=40, jump_at=20, rng=None):
    rng = rng or np.random.default_rng(0)
    frames = []
    for k in range(n):
        f = base.copy()
        shift = np.zeros((base.count, 3))
        if k >= jump_at:
            shift[:16, 2] = 1.2  # 16 atoms hop 1.2 A at once
        f.positions = base.positions + rng.normal(0, 0.03, (base.count, 3)) + shift
        frames.append(f)
    return Pipeline(StaticSource(frames))


def test_describe_frame_small_cell():
    # a unit cell narrower than 8 A: the shortest-distance search stays within half its width
    from ewald import examples

    d = describe_frame(examples.load("Si.cif").compute(0))
    assert d["num_particles"] == 8 and abs(d["min_distance"]["value"] - 2.352) < 0.01


def test_describe_trajectory_finds_event(cu_fcc):
    out = describe_trajectory(_jump_trajectory(cu_fcc))
    assert [e["frame"] for e in out["events"]] == [20]
    changes = out["events"][0]["changes"]
    assert changes["max_displacement"]["after"] > 1.0
    assert changes["mean_coordination"]["after"] < changes["mean_coordination"]["before"]
    json.dumps(out)


@pytest.fixture
def cu_file(tmp_path):
    path = tmp_path / "cu.extxyz"
    write(path, [bulk("Cu", "fcc", a=3.615, cubic=True).repeat(3)] * 3)
    return path


@pytest.fixture
def fake_render(monkeypatch):
    from PIL import Image

    calls = []

    def render_png(data, path, **kw):
        calls.append((data.count, kw))
        Image.new("RGB", (8, 6), "white").save(path)

    monkeypatch.setattr(tools_mod, "render_png", render_png)
    return calls


def test_symmetry_and_diffraction_tools(cu_file):
    pytest.importorskip("spglib")
    ts = Toolset()
    ts.call("load_structure", {"path": str(cu_file)})
    r = ts.call("analyze_symmetry", {})
    assert r.content["space_group"] == "Fm-3m" and r.content["sites"][0]["wyckoff"] == "4a"
    r = ts.call("compute_diffraction", {"frame": 0})
    top = r.content["peaks"][0]
    assert top["intensity"] == 100 and top["d"] == pytest.approx(3.615 / 3 ** 0.5, abs=1e-3)   # (111)
    assert ts.call("compute_diffraction", {"kind": "sq", "q_max": 6}).content["table"]["q"]


def test_toolset_end_to_end(cu_file, fake_render):
    ts = Toolset()
    r = ts.call("load_structure", {"path": str(cu_file)})
    assert not r.is_error and r.content["num_frames"] == 3
    assert "CoordinationAnalysis" in ts.call("list_modifiers").content

    r = ts.call("add_modifier", {"name": "CoordinationAnalysis", "params": {"cutoff": 3.0}})
    assert "Coordination" in r.content["effect"]["new_properties"]
    r = ts.call("evaluate_expression", {"expression": "Coordination == 12"})
    assert r.content["count"] == 108
    r = ts.call("add_modifier", {"name": "Slice", "params": {"normal": [0, 0, 1], "distance": 5.0}})
    assert r.content["effect"]["num_particles"][1] < 108

    r = ts.call("render_image", {"direction": "z"})
    assert r.images and r.images[0][:4] == b"\x89PNG" and fake_render[0][1]["direction"] == "z"

    r = ts.call("compute_rdf", {"cutoff": 5.0})
    assert r.content["peaks"]["g(r)"][0]["r"] == pytest.approx(2.556, abs=0.05)

    script = ts.to_script()   # the session notebook: every change above is one of its cells
    assert "pipe = load(" in script and "pipe.add(CoordinationAnalysis(cutoff=3.0))" in script and "Slice(" in script
    ns = {}
    exec(script, ns)  # the exported script runs and reproduces the result
    assert ns["data"].count == ts.pipeline.compute(0).count


def test_toolset_errors_are_results(cu_file):
    ts = Toolset()
    assert ts.call("describe_frame").is_error  # nothing loaded
    ts.call("load_structure", {"path": str(cu_file)})
    r = ts.call("add_modifier", {"name": "NoSuchModifier"})
    assert r.is_error and "available" in r.content
    r = ts.call("add_modifier", {"name": "ExpressionSelection", "params": {"expression": "Bogus > 1"}})
    assert r.is_error and ts.pipeline.modifiers == []  # failed modifier rolled back
    assert [c.source for c in ts.notebook.cells if "ExpressionSelection" in c.source] == []   # and its cell
    assert "run_python" not in ts.tools and "run_python" not in Toolset(allow_code=True).tools


def test_msd_diffusion(cu_fcc):
    frames = []
    rng = np.random.default_rng(3)
    pos = cu_fcc.positions.copy()
    for _ in range(60):  # random walk, <dr^2> = 3 * 0.05^2 per frame -> slope 0.0075, D = 0.00125
        f = cu_fcc.copy()
        f.positions = pos
        frames.append(f)
        pos = pos + rng.normal(0, 0.05, pos.shape)
    ts = Toolset(Pipeline(StaticSource(frames)))
    r = ts.call("compute_msd", {"timestep_fs": 1.0})
    fit = r.content["fits"]["MSD"]
    assert fit["slope_A2_per_frame"] == pytest.approx(0.0075, rel=0.2)
    assert fit["D_cm2_per_s"] == pytest.approx(0.00125e-1, rel=0.2)


def test_model_code_runs_only_as_notebook_cells(cu_fcc):
    """No run_python: with allow_code the model's own code is a cell it inserts and runs."""
    ts = Toolset(Pipeline(StaticSource(cu_fcc)), allow_code=True)
    code = "@pipe.add\ndef tag_top(frame, data):\n    data['Top'] = data.positions[:, 2] > 7\nprint(pipe.compute(0).count)"
    k = ts.call("insert_cell", {"source": code}).content["inserted"]["index"]
    r = ts.call("run_cell", {"index": k})
    assert not r.is_error and r.content["outputs"] == [{"kind": "text", "text": "256\n"}]
    assert "Top" in ts.pipeline.compute(0) and "def tag_top" in ts.to_script()


# ------------------------------------------------------------------ agent loop

class FakeClient:
    """Plays back scripted responses and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **params):
        self.requests.append(json.loads(json.dumps(params, default=lambda o: vars(o))))
        return self.responses.pop(0)


def _tool_use(id, name, input):
    return NS(type="tool_use", id=id, name=name, input=input)


def _text(t):
    return NS(type="text", text=t)


def test_agent_tool_loop(cu_fcc, fake_render):
    client = FakeClient([
        NS(stop_reason="tool_use", content=[
            _text("Let me look."),
            _tool_use("t1", "describe_frame", {"cutoff": 3.0}),
            _tool_use("t2", "render_image", {"direction": "iso"})]),
        NS(stop_reason="tool_use", content=[
            _tool_use("t3", "add_modifier", {"name": "NoSuchModifier"})]),
        NS(stop_reason="end_turn", content=[_text("Perfect fcc Cu, CN = 12 for all 256 atoms.")]),
    ])
    events = []
    agent = Agent.from_pipeline(Pipeline(StaticSource(cu_fcc)), client=client,
                                on_event=lambda k, p: events.append(k))
    answer = agent.ask("What is this structure?")
    assert answer.startswith("Perfect fcc Cu")

    first = client.requests[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert first["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert {t["name"] for t in first["tools"]} >= {"describe_frame", "render_image", "add_modifier"}

    # both tool results go back in ONE user message, the render as an image block
    results = client.requests[1]["messages"][-1]
    assert results["role"] == "user" and [b["tool_use_id"] for b in results["content"]] == ["t1", "t2"]
    desc = json.loads(results["content"][0]["content"][0]["text"])
    assert desc["coordination"]["mean"] == 12
    img = results["content"][1]["content"][1]
    assert img["type"] == "image" and base64.b64decode(img["source"]["data"])[:4] == b"\x89PNG"
    err = client.requests[2]["messages"][-1]["content"][0]
    assert err["is_error"] is True
    assert events.count("tool_call") == 3 and events[-1] == "answer"

    # conversation continues with history
    client.responses.append(NS(stop_reason="end_turn", content=[_text("Yes.")]))
    assert agent.ask("Sure?") == "Yes." and len(client.requests[-1]["messages"]) == 7


def test_agent_refusal_and_turn_limit(cu_fcc):
    ts = Toolset(Pipeline(StaticSource(cu_fcc)))
    agent = Agent(ts, client=FakeClient([NS(stop_reason="refusal", content=[])]), fallbacks=False)
    assert "declined" in agent.ask("hi")
    loop = [NS(stop_reason="tool_use", content=[_tool_use(f"t{i}", "show_pipeline", {})]) for i in range(3)]
    client = FakeClient(loop)
    agent = Agent(ts, client=client, max_turns=3, fallbacks=False)
    assert "stopped" in agent.ask("loop")
    assert "fallbacks" not in client.requests[0]


class FakeOpenAI:
    """Stands in for ``openai.OpenAI``: ``chat.completions.create`` answers from a script."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **params):
        self.requests.append(json.loads(json.dumps(params)))
        return self.responses.pop(0)


def _completion(text=None, calls=(), finish="stop"):
    tool_calls = [NS(id=i, type="function", function=NS(name=n, arguments=json.dumps(a))) for i, n, a in calls]
    return NS(choices=[NS(finish_reason="tool_calls" if calls else finish,
                          message=NS(content=text, tool_calls=tool_calls or None, refusal=None))])


def test_agent_tool_loop_with_an_openai_compatible_provider(cu_fcc, fake_render):
    client = FakeOpenAI([
        _completion("Let me look.", [("c1", "describe_frame", {"cutoff": 3.0}), ("c2", "render_image", {})]),
        _completion("Perfect fcc Cu, CN = 12."),
    ])
    agent = Agent.from_pipeline(Pipeline(StaticSource(cu_fcc)), client=client, provider="openai")
    assert agent.ask("What is this structure?") == "Perfect fcc Cu, CN = 12."
    first = client.requests[0]
    assert first["model"] == "gpt-5" and first["max_completion_tokens"] == 16000
    assert first["reasoning_effort"] == "high" and "fallbacks" not in first
    assert first["messages"][0]["role"] == "system" and "ewald" in first["messages"][0]["content"]
    names = {t["function"]["name"] for t in first["tools"]}
    assert {"describe_frame", "render_image", "add_modifier"} <= names
    second = client.requests[1]["messages"]
    assert second[2]["role"] == "assistant" and [c["id"] for c in second[2]["tool_calls"]] == ["c1", "c2"]
    results = [m for m in second if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["c1", "c2"]
    assert json.loads(results[0]["content"])["coordination"]["mean"] == 12
    image = second[-1]["content"][1]   # the render follows the tool messages as a user image
    assert second[-1]["role"] == "user" and image["image_url"]["url"].startswith("data:image/png;base64,")

    # a local model without vision: images become a note, the address and model are the user's
    local = FakeOpenAI([_completion(None, [("c1", "render_image", {})]), _completion("Done.")])
    agent = Agent.from_pipeline(Pipeline(StaticSource(cu_fcc)), client=local, provider="openai-compatible",
                                model="qwen2.5")
    assert agent.ask("Show it") == "Done."
    req = local.requests[1]
    assert req["model"] == "qwen2.5" and "max_tokens" in req and "reasoning_effort" not in req
    assert "image(s) left out" in req["messages"][-1]["content"]
    assert all(m["role"] != "user" or isinstance(m["content"], str) for m in req["messages"])


def test_openai_sdk_round_trip(cu_fcc, fake_render):
    """The real openai client against a mocked server: the requests the SDK sends are valid."""
    openai = pytest.importorskip("openai")
    import httpx2 as httpx

    answers = iter([
        {"tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "describe_frame", "arguments": "{\"cutoff\": 3.0}"}}]},
        {"content": "Cu, fcc."}])
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((request.url.path, request.headers.get("authorization"), body))
        msg = {"role": "assistant", "content": None, **next(answers)}
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                                         "choices": [{"index": 0, "message": msg,
                                                      "finish_reason": "tool_calls" if "tool_calls" in msg else "stop"}]})

    client = openai.OpenAI(api_key="sk-test", base_url="http://localhost:11434/v1",
                           http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    agent = Agent.from_pipeline(Pipeline(StaticSource(cu_fcc)), client=client, provider="openai-compatible")
    assert agent.ask("What is it?") == "Cu, fcc."
    assert seen[0][0] == "/v1/chat/completions" and seen[0][1] == "Bearer sk-test"
    assert seen[1][2]["messages"][-1]["role"] == "tool"


def _sse(events) -> bytes:
    """A server-sent event stream: ``(event name or None, JSON payload or "[DONE]")`` pairs."""
    out = []
    for name, data in events:
        out.append((f"event: {name}\n" if name else "") + "data: " + (data if isinstance(data, str) else json.dumps(data)))
    return ("\n\n".join(out) + "\n\n").encode()


def test_streamed_answers_with_the_real_sdks(cu_fcc, fake_render):
    """``Agent(stream=True)`` over the real anthropic and openai clients against mocked SSE servers:
    the text arrives in pieces as ``text`` events, tool calls built from pieces run, and the reply
    and conversation are the same as without streaming."""
    anthropic = pytest.importorskip("anthropic")
    openai = pytest.importorskip("openai")
    import httpx2 as httpx

    def message_events(blocks, stop):
        ev = [("message_start", {"type": "message_start", "message": {
            "id": "m", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0}}})]
        for i, block in enumerate(blocks):
            if block[0] == "text":
                ev.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                   "content_block": {"type": "text", "text": ""}}))
                ev += [("content_block_delta", {"type": "content_block_delta", "index": i,
                                                "delta": {"type": "text_delta", "text": piece}}) for piece in block[1]]
            else:
                _, tid, name, args = block
                ev.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": {
                    "type": "tool_use", "id": tid, "name": name, "input": {}}}))
                ev += [("content_block_delta", {"type": "content_block_delta", "index": i,
                                                "delta": {"type": "input_json_delta", "partial_json": part}})
                       for part in (args[:5], args[5:])]
            ev.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
        ev += [("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                  "usage": {"output_tokens": 5}}),
               ("message_stop", {"type": "message_stop"})]
        return _sse(ev)

    replies = iter([message_events([("text", ["Let me ", "look."]),
                                    ("tool", "t1", "describe_frame", json.dumps({"cutoff": 3.0}))], "tool_use"),
                    message_events([("text", ["Cu, ", "fcc", "."])], "end_turn")])
    seen = []

    def claude(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=next(replies))

    client = anthropic.Anthropic(api_key="sk-ant-test", http_client=httpx.Client(transport=httpx.MockTransport(claude)))
    events = []
    agent = Agent.from_pipeline(Pipeline(StaticSource(cu_fcc)), client=client, stream=True,
                                on_event=lambda k, p: events.append((k, p if isinstance(p, str) else p.get("name"))))
    assert agent.ask("What is it?") == "Cu, fcc."
    assert seen[0]["stream"] is True and seen[0]["fallbacks"] == "default"
    assert [p for k, p in events if k == "text"] == ["Let me ", "look.", "Cu, ", "fcc", "."]
    assert ("tool_call", "describe_frame") in events
    assert events.index(("tool_call", "describe_frame")) == 2   # the first round's text, then its tool
    result = seen[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "t1" and json.loads(result["content"][0]["text"])["coordination"]["mean"] == 12

    def chunk(delta, finish=None):
        return {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    streams = iter([
        _sse([(None, chunk({"role": "assistant", "content": "Hm"})),
              (None, chunk({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                            "function": {"name": "describe_frame", "arguments": "{\"cut"}}]})),
              (None, chunk({"tool_calls": [{"index": 0, "function": {"arguments": "off\": 3.0}"}}]})),
              (None, chunk({}, "tool_calls")), (None, "[DONE]")]),
        _sse([(None, chunk({"content": "Cu, "})), (None, chunk({"content": "fcc."})), (None, chunk({}, "stop")),
              (None, "[DONE]")])])
    sent = []

    def compatible(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=next(streams))

    client = openai.OpenAI(api_key="sk-test", base_url="http://localhost:11434/v1",
                           http_client=httpx.Client(transport=httpx.MockTransport(compatible)))
    texts = []
    agent = Agent.from_pipeline(Pipeline(StaticSource(cu_fcc)), client=client, provider="openai-compatible",
                                stream=True, on_event=lambda k, p: k == "text" and texts.append(p))
    assert agent.ask("What is it?") == "Cu, fcc."
    assert sent[0]["stream"] is True and texts == ["Hm", "Cu, ", "fcc."]
    call = sent[1]["messages"][2]["tool_calls"][0]
    assert call["id"] == "c1" and json.loads(call["function"]["arguments"]) == {"cutoff": 3.0}
    assert sent[1]["messages"][3]["role"] == "tool"


def test_streaming_falls_back_and_stops():
    """A server that refuses ``stream`` gets the request again in one piece (and no more streamed
    ones); a fake client without streaming answers in one piece; a Stop during a stream ends it."""
    import threading

    from ewald.llm.agent import Cancelled
    from ewald.llm.chat import AnthropicChat, OpenAIChat

    class Refused(Exception):
        status_code = 400

    calls = []

    def create(**params):
        calls.append(params.get("stream", False))
        if params.get("stream"):
            raise Refused("stream is not supported")
        return _completion("whole")

    chat = OpenAIChat(NS(chat=NS(completions=NS(create=create))), "m")
    chat.user("hi")
    pieces = []
    assert chat.send(on_text=pieces.append).text == "whole" and pieces == ["whole"] and calls == [True, False]
    chat.user("again")
    assert chat.send(on_text=pieces.append).text == "whole" and calls == [True, False, False]

    claude = AnthropicChat(FakeClient([NS(stop_reason="end_turn", content=[_text("one piece")])]), "m")
    claude.user("hi")
    assert claude.send(on_text=pieces.append).text == "one piece" and pieces[-1] == "one piece"

    closed = []

    class Stream:
        def __iter__(self):
            for word in ("a", "b", "c"):
                yield NS(choices=[NS(delta=NS(content=word, tool_calls=None, refusal=None), finish_reason=None)])

        def close(self):
            closed.append(True)

    cancel = threading.Event()
    seen = []

    def on_event(kind, piece):
        seen.append(piece)
        cancel.set()                               # Stop after the first piece

    client = NS(chat=NS(completions=NS(create=lambda **p: Stream())))
    agent = Agent(Toolset(), client=client, provider="openai", stream=True, on_event=on_event)
    with pytest.raises(Cancelled):
        agent.ask("hi", cancel=cancel)
    assert seen == ["a"] and closed and agent.messages[1:] == []   # (the system prompt stays)


def test_forge_and_narrate_with_another_provider(cu_fcc):
    from ewald.llm import narrate
    from ewald.llm.forge import forge

    submit = {"module_name": "m", "class_name": "M", "category": "Analysis", "module_source": "x = 1\n",
              "test_source": "def test_x(): pass\n", "notes": ""}
    client = FakeOpenAI([_completion(None, [("c1", "submit_plugin", submit)])])
    result = forge("anything", client=client, provider="gemini")
    assert result.module_name == "m" and not result.verified
    req = client.requests[0]
    assert req["model"] == "gemini-2.5-pro"
    assert req["tool_choice"] == {"type": "function", "function": {"name": "submit_plugin"}}

    pipe, _ = _vacancy_hop(cu_fcc)
    answer = {"summary": "A hop.", "events": [{"frame": 24, "title": "Hop", "description": "d", "confidence": "high"}]}
    client = FakeOpenAI([_completion("```json\n" + json.dumps(answer) + "\n```")])
    tl = narrate(pipe, client, provider="openai")
    assert tl.summary == "A hop." and tl.narrated_by == "gpt-5"
    assert client.requests[0]["response_format"]["type"] == "json_schema"


def test_ai_provider_plugin(monkeypatch):
    from ewald import plugins
    from ewald.llm import credentials, providers
    from ewald.llm.providers import Provider

    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    monkeypatch.setattr(plugins._registry, "loaded", True)

    @plugins.ai_provider
    def mistral():
        """Mistral through its OpenAI-compatible API."""
        return Provider("mistral", "Mistral", models={"mistral-large-latest": "Mistral Large"},
                        base_url="https://api.mistral.ai/v1", key_name="mistral-api-key", env_var="MISTRAL_API_KEY")

    p = providers.get("mistral")
    assert p.origin == "python" and p.default_model == "mistral-large-latest" and not p.builtin
    assert "mistral" in providers.providers() and plugins.status()["ai_providers"][-1]["name"] == "mistral"
    monkeypatch.setenv("MISTRAL_API_KEY", "m-key-from-env")
    monkeypatch.setattr(credentials, "_keyring", lambda: None)
    assert credentials.get_api_key("mistral") == ("m-key-from-env", "env")
    client = FakeOpenAI([_completion("Bonjour.")])
    from ewald.llm import Agent as A

    assert A(client=client, provider="mistral").ask("hi") == "Bonjour."
    with pytest.raises(ValueError):
        plugins.register_ai_provider("openai", Provider("openai", "Fake"))
    with pytest.raises(TypeError):
        plugins.register_ai_provider("x", object())


def test_mcp_adapter(cu_fcc, fake_render):
    pytest.importorskip("mcp")
    from ewald.llm import mcp_server

    ts = Toolset(Pipeline(StaticSource(cu_fcc)))
    names = {t.name for t in mcp_server.list_tools(ts)}
    assert {"describe_frame", "render_image", "get_script"} <= names
    out = mcp_server.call_tool(ts, "render_image", {})
    assert out[0].type == "text" and out[1].type == "image" and (getattr(out[1], "mime_type", None) or getattr(out[1], "mimeType")) == "image/png"
    assert mcp_server.call_tool(ts, "nope", {})[0].text.startswith("ERROR")
    assert mcp_server.build_server(ts).name == "ewald"


# ------------------------------------------------------------------ narrator

def _vacancy_hop(cu_fcc, n=40, hop_at=24):
    """fcc Cu with one vacancy; at ``hop_at`` a nearest neighbor jumps into it."""
    base = cu_fcc.delete(np.arange(cu_fcc.count) == 0)
    base.particles["Particle Identifier"] = np.arange(1, base.count + 1)
    vac = cu_fcc.positions[0]
    d = np.linalg.norm(base.cell.wrap(base.positions - vac + 7) - 7, axis=1)
    hopper = int(np.argmin(d))
    rng = np.random.default_rng(5)
    frames = []
    for k in range(n):
        f = base.copy()
        pos = base.positions.copy()
        if k >= hop_at:
            pos[hopper] = vac
        f.positions = pos + rng.normal(0, 0.04, pos.shape)
        frames.append(f)
    return Pipeline(StaticSource(frames)), hopper + 1


def test_no_events_for_noise_or_ramp():
    rng = np.random.default_rng(2)
    assert detect_change_points(rng.normal(0, 0.5, 100)) == []
    assert detect_change_points(0.05 * np.arange(100) + rng.normal(0, 0.01, 100)) == []


def test_narrate_vacancy_hop_rule_based(cu_fcc):
    from ewald.llm import narrate

    pipe, hopper_id = _vacancy_hop(cu_fcc)
    tl = narrate(pipe)
    assert [e.frame for e in tl.events] == [24] and tl.narrated_by == "rules"
    ev = tl.events[0].evidence
    assert ev["moved"]["count"] == 1 and ev["extent"] == "local"
    assert ev["top_movers"][0]["id"] == hopper_id
    assert ev["top_movers"][0]["displacement"] == pytest.approx(3.615 / np.sqrt(2), abs=0.2)
    assert tl.bookmarks()[0][0] == 24 and "Frame 24" in tl.to_markdown()
    json.dumps(tl.to_dict())


def test_narrate_with_llm(cu_fcc, fake_render):
    from ewald.llm import narrate

    pipe, _ = _vacancy_hop(cu_fcc)
    answer = {"summary": "One vacancy hop.", "events": [
        {"frame": 24, "title": "Vacancy hop", "description": "A Cu atom jumps 2.5 A into the vacancy.",
         "confidence": "high"},
        {"frame": 3, "title": "Invented", "description": "x", "confidence": "low"}]}
    client = FakeClient([NS(stop_reason="end_turn", content=[_text(json.dumps(answer))])])
    tl = narrate(pipe, client, render=True)
    req = client.requests[0]
    assert req["output_config"]["format"]["type"] == "json_schema"
    images = [b for b in req["messages"][0]["content"] if b["type"] == "image"]
    assert len(images) == 2  # before and after the one event
    payload = json.loads(req["messages"][0]["content"][-1]["text"])
    assert payload["events"][0]["evidence"]["moved"]["count"] == 1
    assert tl.summary == "One vacancy hop." and tl.narrated_by == "claude-opus-5-5"
    assert [(e.frame, e.title, e.confidence) for e in tl.events] == [(24, "Vacancy hop", "high")]


def test_event_details_tool(cu_fcc):
    pipe, hopper_id = _vacancy_hop(cu_fcc)
    r = Toolset(pipe).call("event_details", {"frame": 24})
    assert not r.is_error and r.content["top_movers"][0]["id"] == hopper_id


# ------------------------------------------------------------------ code-writing agents

def test_llms_txt_lists_every_modifier():
    from pathlib import Path

    import ewald.modifiers as mods

    text = (Path(__file__).parents[1] / "llms.txt").read_text()
    missing = [n for n in mods.__all__ if n not in ("Modifier", "PythonModifier") and f"`{n}(" not in text]
    assert not missing, f"llms.txt is stale (missing {missing}); regenerate with: ewald api > llms.txt"


def test_ovito_modifier_names_resolve():
    from ewald.modifiers import CommonNeighborAnalysis, CommonNeighborAnalysisModifier, WrapAtoms, WrapPeriodicImagesModifier

    assert CommonNeighborAnalysisModifier is CommonNeighborAnalysis and WrapPeriodicImagesModifier is WrapAtoms
    with pytest.raises(ImportError):
        from ewald.modifiers import NoSuchThingModifier  # noqa: F401


def test_data_summary(cu_fcc):
    s = cu_fcc.summary()
    assert "256 particles" in s and "'Position': float64 [256x3]" in s and "('Cu', 256)" in s
    assert "rows of cell.matrix are the cell vectors" in s


def test_code_modifiers_need_allow_code(cu_fcc):
    pipe = Pipeline(StaticSource(cu_fcc))
    closed = Toolset(pipe)
    assert "PythonScript" not in closed.call("list_modifiers").content
    assert closed.call("add_modifier", {"name": "PythonScript", "params": {"source": "x"}}).is_error
    assert "OvitoModifier" not in closed.call("list_modifiers").content
    opened = Toolset(pipe, allow_code=True)
    assert "PythonScript" in opened.call("list_modifiers").content
    ref = closed.call("python_api").content   # a reference for writing cells: offered to every session
    assert "Differences from OVITO" in ref and "`CommonNeighborAnalysis(" in ref


def test_python_api_cells_run_as_documented(tmp_path, monkeypatch):
    """The notebook section of the API reference the AI reads: its cell kinds run on the default
    imports and the GUI recognizes them (step and view cells)."""
    import matplotlib

    from ewald import plugins
    from ewald.llm.reference import GUIDE
    from ewald.notebook import IMPORTS, Notebook
    from ewald.pipeline import Modifier

    matplotlib.use("Agg")
    for name in ("show(pipe, table=None", "paint(data, where, color)", "isolate(data, where)", "[each frame]",
                 'data.tables["DOS"]', "view(pipe, ...)"):
        assert name in GUIDE, name
    path = tmp_path / "cu.vasp"
    write(path, bulk("Cu", "fcc", a=3.615, cubic=True).repeat(2), format="vasp")
    nb = Notebook.from_py(f"# %%\nimport numpy as np\n{IMPORTS}\n\n# %%\npipe = load({str(path)!r})\n\n"
                          "# %%\npipe.add(CoordinationAnalysis(cutoff=3.0))\n\n"
                          "# %% surface [each frame]\npaint(data, 'Coordination >= 12', 'gold')\n"
                          "fade(data, data['Position'][:, 2] > 3, 0.5)\n\n"
                          "# %%\nshow(pipe, table='coordination-rdf', xlim=(0, 3))\n\n"
                          "# %%\nview(pipe, style='Ball & stick', direction='z')\n")
    assert nb.run_all()
    assert [c.is_step for c in nb.cells] == [False, False, True, False, False, False]
    data = nb.namespace["pipe"].compute(0)
    assert np.allclose(data["Color"][0], matplotlib.colors.to_rgb("gold"))
    assert nb.namespace["pipe"].view.direction == "z"

    # python_api lists the modifier plugins installed here, which the model can add too
    tracker = type("GrainTracker", (Modifier,), {"__doc__": "Track grains."})
    monkeypatch.setattr(plugins, "modifiers",
                        lambda: {"GrainTracker": plugins.ModifierPlugin("GrainTracker", tracker, "Plugins", "", "python")})
    assert "`GrainTracker()`: Track grains." in Toolset().call("python_api").content


def test_notebook_tools_read_insert_edit_move_delete(tmp_path):
    from ewald.notebook import Notebook

    path = tmp_path / "cu.vasp"
    write(path, bulk("Cu", "fcc", a=3.615, cubic=True).repeat(2), format="vasp")
    nb = Notebook.from_py(f"# %%\nfrom ewald import load\nfrom ewald.modifiers import *\npipe = load({str(path)!r})\n\n"
                          "# %%\npipe.add(CoordinationAnalysis(cutoff=3.0))\n\n# %%\nprint('hi')\n")
    assert nb.run_all()
    ts = Toolset(nb.pipeline, notebook=nb)
    names = [d["name"] for d in ts.definitions()]
    assert {"list_cells", "read_cell", "insert_cell", "edit_cell", "move_cell", "delete_cell"} <= set(names)
    assert "run_cell" not in names and "run_cell" not in [d["name"] for d in Toolset(None).definitions()]

    cells = ts.call("list_cells").content["cells"]
    assert [c["kind"] for c in cells] == ["code", "step", "code"] and cells[1]["modifier"] == "CoordinationAnalysis"
    read = ts.call("read_cell", {"index": 2}).content
    assert read["source"] == "print('hi')" and read["outputs"] == [{"kind": "text", "text": "hi\n"}]
    assert ts.call("read_cell", {"index": 1}).content["params"]["cutoff"] == 3.0

    # step parameters: the step's call is rewritten with the new literal values and the cell runs
    old = nb.pipeline.modifiers[0]
    r = ts.call("edit_cell", {"index": 1, "params": {"cutoff": 2.8}})
    assert not r.is_error and nb.cells[1].source == "pipe.add(CoordinationAnalysis(cutoff=2.8))"
    assert nb.pipeline.modifiers[0] is not old and nb.pipeline.modifiers[0].cutoff == 2.8   # the cell ran
    assert nb.cells[1].added == nb.pipeline.modifiers and nb.cells[1].state == "ok"
    assert ts.call("edit_cell", {"index": 1, "params": {"nope": 1}}).is_error
    assert ts.call("edit_cell", {"index": 1, "params": {"cutoff": "far"}}).is_error
    assert nb.cells[1].source.endswith("cutoff=2.8))") and nb.pipeline.modifiers[0].cutoff == 2.8

    # inserted and edited cells are not run
    new = ts.call("insert_cell", {"source": "y = 2", "index": 2}).content["inserted"]
    assert new["index"] == 2 and nb.cells[2].state == "new" and "y" not in nb.namespace
    assert ts.call("edit_cell", {"id": new["id"], "old_text": "2", "new_text": "3"}).content["edited"]["first_line"] == "y = 3"
    assert ts.call("edit_cell", {"index": 2, "old_text": "zzz", "new_text": "1"}).is_error
    assert ts.call("edit_cell", {"index": 2}).is_error                     # nothing to change
    assert "y" not in nb.namespace

    # a step cell's new source takes its old step out of the pipeline until the cell runs again
    r = ts.call("edit_cell", {"index": 1, "source": "pipe.add(CoordinationAnalysis(cutoff=3.5))"}).content
    assert "note" in r and nb.pipeline.modifiers == [] and nb.cells[1].state == "stale"

    ts.call("move_cell", {"index": 3, "to": 1})
    assert nb.cells[1].source == "print('hi')"
    ts.call("delete_cell", {"index": 1})
    assert len(nb.cells) == 3 and ts.call("read_cell", {"index": 9}).is_error
    ts.call("insert_cell", {"source": "# Notes", "kind": "markdown"})
    assert nb.cells[-1].kind == "markdown" and nb.cells[-1].state == "ok"


def test_notebook_tools_run_cell_with_allow_code():
    from ewald.notebook import Notebook

    nb = Notebook.from_py("# %%\nx = 6\n\n# %%\nprint(x * 7)\n")
    ts = Toolset(None, notebook=nb, allow_code=True)
    assert "run_cell" in [d["name"] for d in ts.definitions()]
    assert ts.call("run_cell", {"index": 0}).content["ok"]
    r = ts.call("run_cell", {"index": 1})
    assert not r.is_error and r.content["outputs"] == [{"kind": "text", "text": "42\n"}]
    ts.call("insert_cell", {"source": "1 / 0"})
    r = ts.call("run_cell", {"index": -1})
    assert r.is_error and r.content["state"] == "error" and "ZeroDivisionError" in r.content["outputs"][0]["text"]
    assert "run_cell" not in [d["name"] for d in Toolset(None).definitions()]   # not without allow_code


def test_notebook_tools_run_the_cells_they_write_when_allowed():
    from ewald.llm.notebook_tools import NotebookAccess
    from ewald.notebook import Notebook

    nb = Notebook.from_py("# %%\nx = 6\n")
    allowed = [True]
    ts = Toolset(None, notebook=NotebookAccess(nb, can_run=lambda: allowed[0]))
    r = ts.call("insert_cell", {"source": "print(x * 7)"})       # x = 6 has not run: NameError
    assert r.is_error and r.content["run"]["state"] == "error" and nb.cells[-1].state == "error"
    ts.call("run_cell", {"index": 0})
    r = ts.call("edit_cell", {"index": -1, "old_text": "7", "new_text": "7 + 0"})
    assert not r.is_error and r.content["run"]["outputs"] == [{"kind": "text", "text": "42\n"}]
    r = ts.call("insert_cell", {"source": "y = 1", "run": False})
    assert "run" not in r.content and nb.cells[-1].state == "new"
    ts.call("insert_cell", {"source": "# Notes", "kind": "markdown"})   # markdown never runs
    allowed[0] = False                                             # the user unticks the preference
    r = ts.call("insert_cell", {"source": "z = 2"})
    assert "run" not in r.content and "not run" in r.content["note"] and nb.cells[-1].state == "new"
    assert "z" not in nb.namespace


def test_agent_stops_when_cancelled_and_forgets_the_question(cu_fcc):
    import threading

    from ewald.llm.agent import Cancelled

    cancel = threading.Event()
    ts = Toolset(Pipeline(StaticSource(cu_fcc)))
    ran = []
    ts.tools["show_pipeline"].function = lambda: ran.append(1) or {}

    class StopAfterFirst(FakeClient):
        def _create(self, **params):
            out = super()._create(**params)
            cancel.set()                         # Stop pressed while this request was on the wire
            return out

    client = StopAfterFirst([NS(stop_reason="tool_use", content=[_tool_use("t1", "show_pipeline", {})])])
    agent = Agent(ts, client=client, fallbacks=False)
    with pytest.raises(Cancelled):
        agent.ask("loop", cancel=cancel)
    assert ran == [] and agent.messages == []    # no tool ran; the conversation is as before
    client.responses.append(NS(stop_reason="end_turn", content=[_text("Hi.")]))
    assert agent.ask("hello", cancel=threading.Event()) == "Hi." and len(agent.messages) == 2


def test_failed_cells_lists_errors():
    from ewald.llm.notebook_tools import failed_cells
    from ewald.notebook import Notebook

    nb = Notebook.from_py("# %%\nx = 1\n\n# %%\n1 / 0\n")
    nb.run_all()
    text = failed_cells(nb)
    assert text.startswith("cell index 1 (1 / 0):") and "ZeroDivisionError" in text
    nb.cells[1].source = "1 / 1"
    nb.run_all()
    assert failed_cells(nb) == ""


# ------------------------------------------------------------------ the AI changes results only through cells

#: Every built-in tool, by what it may change. A new tool fails test_every_tool_is_classified until it
#: is listed here; one that changes the pipeline, the view or files must do it by writing and running a cell.
READ_ONLY_TOOLS = {"describe_frame", "describe_selection", "describe_trajectory", "event_details", "list_modifiers",
                   "show_pipeline", "set_frame", "evaluate_expression", "compute_rdf", "analyze_symmetry",
                   "compute_diffraction", "compute_msd", "check_run", "render_image", "get_view", "list_visuals",
                   "list_overlays", "list_styles", "get_script", "python_api", "list_cells", "read_cell"}
WORKFLOW_TOOLS = {"list_workflows", "describe_workflow"}   # read-only; offered with a workflow source (test_workflow.py)
CELL_WRITING_TOOLS = set(tools_mod.CELL_WRITERS)
CELL_EDITING_TOOLS = {"insert_cell", "edit_cell", "move_cell", "delete_cell", "run_cell"}


def test_every_tool_is_classified():
    from ewald import plugins

    names = set(Toolset(allow_code=True).tools) - set(plugins.tools())
    assert names == READ_ONLY_TOOLS | CELL_WRITING_TOOLS | CELL_EDITING_TOOLS | WORKFLOW_TOOLS


def test_tools_do_not_touch_the_pipeline_or_view_directly():
    """tools.py changes modifiers and view settings only by running cells, never by hand."""
    import inspect
    import re

    src = inspect.getsource(tools_mod)
    hits = re.findall(r"\.modifiers\.(?:append|insert|pop|remove|extend|clear)\(|\.modifiers\[[^\]]*\]\s*=(?!=)"
                      r"|\.view\.(?:update|reset|rotate|look|color|radius|surface|zoom_by)\(|\.view\.\w+\s*=(?!=)", src)
    assert hits == []


def _state(nb):
    pipe = nb.pipeline
    return [repr(m) for m in pipe.modifiers], pipe.view.changed(), nb.snapshot()


def test_read_only_tools_change_nothing(cu_file, fake_render):
    ts = Toolset()
    ts.call("load_structure", {"path": str(cu_file)})
    ts.call("add_modifier", {"name": "CoordinationAnalysis", "params": {"cutoff": 3.0}})
    nb = ts.notebook
    before = _state(nb)
    calls = {"describe_selection": {"indices": [0], "image": False}, "event_details": {"frame": 1},
             "evaluate_expression": {"expression": "Coordination > 0"}, "compute_rdf": {"cutoff": 4.0},
             "set_frame": {"frame": 2}, "render_image": {"theme": "VESTA", "direction": "x"},
             "read_cell": {"index": 0}}
    for name in sorted(READ_ONLY_TOOLS):
        r = ts.call(name, calls.get(name, {}))
        assert not r.is_error, (name, r.content)
        assert _state(nb) == before, name


def test_the_notebook_alone_reproduces_what_the_tools_did(cu_file, fake_render, tmp_path):
    from ewald.notebook import Notebook, _owned

    ts = Toolset(workdir=str(tmp_path))
    steps = [("load_structure", {"path": str(cu_file)}),
             ("add_modifier", {"name": "CoordinationAnalysis", "params": {"cutoff": 3.0}}),
             ("add_modifier", {"name": "ExpressionSelection", "params": {"expression": "Coordination == 12"}}),
             ("add_modifier", {"name": "Slice", "params": {"distance": 5.0, "width": 4.0, "select": True}, "index": 0}),
             ("remove_modifier", {"index": 2}),
             ("set_view", {"bonds": False, "colors": {"Cu": "gold"}, "direction": "z"}),
             ("add_overlay", {"name": "Coordinate tripod"}),
             ("set_view", {"reset": ["colors"], "background": "white", "rotate": [10, 0]}),
             ("remove_overlay", {"index": 0}),
             ("export_file", {"path": "out.extxyz", "frames": "all"})]
    for name, args in steps:
        r = ts.call(name, args)
        assert not r.is_error, (name, r.content)
        nb, pipe = ts.notebook, ts.pipeline
        assert [id(m) for m in pipe.modifiers] == [id(m) for m in _owned(nb.cells)], name   # every step is a cell's
        assert all(c.state == "ok" for c in nb.cells if c.kind == "code"), name           # and every cell ran
    assert [type(m).__name__ for m in ts.pipeline.modifiers] == ["Slice", "CoordinationAnalysis"]
    assert ts.pipeline.view.changed()["background"] == "white" and "colors" not in ts.pipeline.view.changed()
    assert (tmp_path / "out.extxyz").exists()

    again = Notebook.from_py(ts.notebook.to_py())    # a fresh run of the cells alone
    (tmp_path / "out.extxyz").unlink()
    assert again.run_all()
    assert _state(again)[:2] == _state(ts.notebook)[:2] and (tmp_path / "out.extxyz").exists()


def test_cell_writing_tools_refuse_a_restricted_notebook(cu_file):
    from ewald.notebook import Notebook

    nb = Notebook.from_py(f"# %%\nfrom ewald import load\npipe = load({str(cu_file)!r})\n")
    assert nb.run_all()
    nb.restricted = True                              # an .ewd file not trusted yet
    ts = Toolset(notebook=nb)
    r = ts.call("add_modifier", {"name": "CoordinationAnalysis"})
    assert r.is_error and "trust" in r.content and nb.pipeline.modifiers == [] and len(nb.cells) == 1
    assert ts.call("set_view", {"bonds": False}).is_error and nb.pipeline.view.changed() == {}


def test_touched_cells_names_the_cells_the_tools_wrote(cu_file):
    """Every tool that writes, runs or deletes a cell says which one (index and id), so the AI chat
    can link to it ("Edited cell 2") instead of relying on the model's text."""
    from ewald.llm.notebook_tools import NotebookAccess, touched_cells
    from ewald.notebook import Notebook

    nb = Notebook.from_py(f"# %%\nfrom ewald import load\npipe = load({str(cu_file)!r})\n")
    assert nb.run_all()
    ts = Toolset(None, notebook=NotebookAccess(nb, can_run=True))

    def refs(name, args):
        return touched_cells(name, ts.call(name, args))

    step = refs("add_modifier", {"name": "CoordinationAnalysis", "params": {"cutoff": 3.0}})
    assert step == [{"id": nb.cells[1].id, "index": 1, "verb": "Added step in"}]
    view = refs("set_view", {"bonds": False})
    cell = nb.view_cell()
    assert view == [{"id": cell.id, "index": nb.cells.index(cell), "verb": "Changed view in"}]
    new = refs("insert_cell", {"source": "x = 1"})
    assert new == [{"id": nb.cells[-1].id, "index": len(nb.cells) - 1, "verb": "Inserted and ran"}]
    assert refs("insert_cell", {"source": "y = 1", "run": False})[0]["verb"] == "Inserted"
    assert refs("edit_cell", {"index": 1, "params": {"cutoff": 3.2}}) == [
        {"id": nb.cells[1].id, "index": 1, "verb": "Edited"}]
    assert refs("edit_cell", {"index": -1, "source": "y = 2"})[0]["verb"] == "Edited and ran"
    assert refs("run_cell", {"index": 0}) == [{"id": nb.cells[0].id, "index": 0, "verb": "Ran"}]
    k = len(nb.cells) - 1
    assert refs("delete_cell", {"index": -1}) == [{"id": None, "index": k, "verb": "Deleted"}]
    assert refs("list_cells", {}) == [] and touched_cells("x", "an error") == []
