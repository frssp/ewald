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

    script = ts.to_script()
    assert "CoordinationAnalysis(cutoff=3.0" in script and "Slice(" in script
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
    assert "run_python" not in ts.tools


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


def test_run_python_opt_in(cu_fcc):
    ts = Toolset(Pipeline(StaticSource(cu_fcc)), allow_code=True)
    code = "def tag_top(frame, data):\n    data['Top'] = data.positions[:, 2] > 7\nprint(len(data))"
    r = ts.call("run_python", {"code": code, "add_as_modifier": "tag_top"})
    assert r.content["stdout"].strip() == "256"
    assert "Top" in ts.pipeline.compute(0)
    assert "def tag_top" in ts.to_script() and "pipe.modifiers.append(tag_top)" in ts.to_script()


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
    assert "OvitoModifier" not in closed.call("list_modifiers").content and "python_api" not in closed.tools
    opened = Toolset(pipe, allow_code=True)
    assert "PythonScript" in opened.call("list_modifiers").content
    ref = opened.call("python_api").content
    assert "Differences from OVITO" in ref and "`CommonNeighborAnalysis(" in ref
