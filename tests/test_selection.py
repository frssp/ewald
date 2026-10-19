"""Ask about selection: the selection descriptor, the tool, the agent entry point and the CLI."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from ase.build import bulk

from ewald import DataCollection, Pipeline, StaticSource
from ewald.llm import Agent, Toolset
from ewald.llm import tools as tools_mod
from ewald.llm.selection import describe_selection, selection_prompt, selection_scene, structure_names
from ewald.modifiers import CommonNeighborAnalysis

A = 3.615


@pytest.fixture
def vacancy_pipe():
    """fcc Cu 5x5x5 with the atom at the origin removed, CNA on top."""
    atoms = bulk("Cu", "fcc", a=A, cubic=True).repeat((5, 5, 5))
    del atoms[0]
    data = DataCollection.from_ase(atoms)
    data["Particle Identifier"] = np.arange(1, data.count + 1)
    return Pipeline(StaticSource(data), [CommonNeighborAnalysis()])


def _vacancy_neighbors(data):
    d = np.linalg.norm(data.positions - np.round(data.positions / (5 * A)) * 5 * A, axis=1)  # MIC to origin
    return np.nonzero(np.isclose(d, A / np.sqrt(2), atol=1e-6))[0]


@pytest.fixture
def fake_render(monkeypatch):
    from PIL import Image

    calls = []

    def render_png(data, path, **kw):
        calls.append((data, kw))
        Image.new("RGB", (8, 6), "white").save(path)

    monkeypatch.setattr(tools_mod, "render_png", render_png)
    return calls


def test_describe_vacancy_neighbors(vacancy_pipe):
    data = vacancy_pipe.compute(0)
    near = _vacancy_neighbors(data)
    assert len(near) == 12
    out = describe_selection(data, near[:3], frame=0)

    assert 2.6 < out["cutoff"] < 3.6                       # between the 1st (2.556) and 2nd (3.615) shells
    sel = out["selection"]
    assert sel["count"] == 3 and sel["types"] == {"Cu": 3}
    assert sel["cn_mean_by_type"] == {"Cu": 11}             # one neighbor missing: the vacancy
    assert sel["structures"] == {"OTHER": 3}
    assert len(sel["distances"]) == 3 and "angle_at_middle" in sel
    atom = out["atoms"][0]
    assert atom["cn"] == 11 and atom["structure"] == "OTHER" and atom["id"] == int(data["Particle Identifier"][near[0]])
    assert np.isclose(atom["nearest_distance"], A / np.sqrt(2), rtol=1e-4)
    assert len(atom["neighbors"]) <= 16 and all(n["d"] >= atom["nearest_distance"] - 1e-9 for n in atom["neighbors"])
    assert any(n.get("selected") for n in atom["neighbors"])

    assert out["frame_reference"]["count"] == data.count == 499
    assert out["frame_reference"]["structures"] == {"OTHER": 12, "FCC": 487}
    assert out["bulk_structure"] == "FCC"
    assert out["nearest_defect"]["structure"] == "OTHER"          # the other 9 vacancy neighbors
    assert out["nearest_defect"]["defect_atoms_in_environment"] == 9
    assert out["environment"]["count"] > 50 and "FCC" in out["environment"]["structures"]
    json.dumps(out)                                                # JSON-ready


def test_describe_without_structure_types_and_errors(cu_fcc):
    out = describe_selection(cu_fcc, [5], cutoff=3.0)
    assert out["atoms"][0]["cn"] == 12 and "hint" in out and "structures" not in out["selection"]
    assert structure_names(cu_fcc) is None
    with pytest.raises(ValueError, match="empty"):
        describe_selection(cu_fcc, [])
    with pytest.raises(ValueError, match="out of range"):
        describe_selection(cu_fcc, [cu_fcc.count])


def test_structure_names_follow_the_modifier(vacancy_pipe):
    assert structure_names(vacancy_pipe.compute(0)) == ["OTHER", "FCC", "HCP", "BCC", "ICO"]


def test_scene_crops_unwraps_and_highlights(vacancy_pipe):
    data = vacancy_pipe.compute(0)
    near = _vacancy_neighbors(data)
    scene = selection_scene(data, near, radius=4.0)
    assert 12 < scene.count < data.count and not scene.cell.pbc.any()
    pos = scene.positions
    assert np.linalg.norm(pos - pos.mean(axis=0), axis=1).max() < 4.0 + A    # one contiguous blob, no images
    highlighted = np.all(np.isclose(scene["Color"], (1.0, 0.847, 0.302)), axis=1)
    assert highlighted.sum() == 12
    assert not np.allclose(data["Color"][near], (1.0, 0.847, 0.302))   # the frame itself is untouched


def test_prompt_carries_question_and_numbers(vacancy_pipe):
    data = vacancy_pipe.compute(0)
    desc = describe_selection(data, [1], frame=0)
    text = selection_prompt("What is this atom?", desc)
    assert "Question: What is this atom?" in text and "<selection>" in text and "frame 0" in text
    assert json.loads(text.split("<selection>\n")[1].split("\n</selection>")[0])["selection"]["count"] == 1


def test_tool_uses_picks_ids_and_expressions(vacancy_pipe, fake_render):
    ts = Toolset(vacancy_pipe)
    assert ts.call("describe_selection", {}).is_error              # nothing picked
    ts.selection = [3, 4]
    r = ts.call("describe_selection", {})
    assert not r.is_error and r.content["selection"]["count"] == 2 and len(r.images) == 1
    scene, kw = fake_render[-1]
    assert kw["show_cell"] is False and scene.count < 499
    r = ts.call("describe_selection", {"ids": [1, 2, 3], "image": False})
    assert r.content["selection"]["count"] == 3 and not r.images
    assert ts.call("describe_selection", {"ids": [100000]}).is_error
    r = ts.call("describe_selection", {"expression": "StructureType == 0", "image": False})
    assert r.content["selection"]["count"] == 12


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **params):
        self.requests.append(json.loads(json.dumps(params, default=lambda o: vars(o))))
        return self.responses.pop(0)


def test_agent_ask_about_selection(vacancy_pipe, fake_render):
    client = FakeClient([
        NS(stop_reason="tool_use", content=[NS(type="tool_use", id="t1", name="describe_selection",
                                               input={"radius": 4.0, "image": False})]),
        NS(stop_reason="end_turn", content=[NS(type="text", text="Neighbors of a vacancy.")]),
    ])
    agent = Agent.from_pipeline(vacancy_pipe, client=client, fallbacks=False)
    data = vacancy_pipe.compute(0)
    near = _vacancy_neighbors(data)
    assert agent.ask_about_selection("What are these?", near[:2]) == "Neighbors of a vacancy."
    first = client.requests[0]["messages"][0]["content"]
    assert first[0]["type"] == "image" and first[-1]["type"] == "text"
    assert "Question: What are these?" in first[-1]["text"] and '"OTHER": 2' in first[-1]["text"]
    assert agent.toolset.selection == sorted(int(i) for i in near[:2])
    # Claude's own describe_selection call defaults to the same picks
    tool_result = json.loads(client.requests[1]["messages"][-1]["content"][0]["content"][0]["text"])
    assert tool_result["selection"]["count"] == 2 and tool_result["environment"]["radius"] == 4.0


def test_cli_ask_select(tmp_path, monkeypatch, fake_render, capsys):
    from ase.io import write

    from ewald import llm
    from ewald.__main__ import main

    atoms = bulk("Cu", "fcc", a=A, cubic=True).repeat((3, 3, 3))
    path = tmp_path / "cu.extxyz"
    write(path, atoms)
    seen = []

    def fake_agent(ts, **kw):
        agent = Agent(ts, client=FakeClient([NS(stop_reason="end_turn", content=[NS(type="text", text="bulk fcc")])]),
                      fallbacks=False)
        seen.append(agent)
        return agent

    monkeypatch.setattr(llm, "Agent", fake_agent)
    assert main(["ask", str(path), "what", "is", "this?", "--select", "Position.Z < 0.1"]) is None
    assert capsys.readouterr().out.strip().endswith("bulk fcc")
    prompt = seen[0].client.requests[0]["messages"][0]["content"][-1]["text"]
    assert "Question: what is this?" in prompt and '"count": 18' in prompt
    assert main(["ask", str(path), "x", "--select", "Position.Z < -5"]) == 1
