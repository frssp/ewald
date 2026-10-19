"""Modifier Forge with a scripted fake Claude: verify loop, feedback, saving."""
from __future__ import annotations

import textwrap
from types import SimpleNamespace

import pytest

from ewald import plugins
from ewald.llm import forge as forge_mod

GOOD = textwrap.dedent('''
    import numpy as np
    from ewald import Modifier
    from ewald.plugins import modifier


    @modifier(category="Analysis")
    class HeightAboveMin(Modifier):
        """Height of each particle above the lowest one along an axis."""

        def __init__(self, axis=2):
            self.axis = axis

        def modify(self, data, ctx):
            xp = ctx.xp
            z = xp.asarray(data.positions)[:, self.axis]
            data["Height"] = ctx.backend.to_numpy(z - z.min())
            data.attributes["HeightAboveMin.max"] = float(ctx.backend.to_numpy(z.max() - z.min()))
''')

# Same answer, but shifts the input positions in place.
MUTATES = GOOD.replace('z = xp.asarray(data.positions)[:, self.axis]',
                       'data.positions[:, self.axis] -= 0.0\n        z = xp.asarray(data.positions)[:, self.axis]'
                       '\n        data.positions[0, 0] += 1e-3')

TESTS = textwrap.dedent('''
    import numpy as np
    from ase import Atoms
    from ewald import DataCollection, Pipeline, StaticSource
    from height_above_min import HeightAboveMin


    def test_heights():
        atoms = Atoms("H3", positions=[[0, 0, 1], [0, 0, 3], [0, 0, 6]], cell=[10, 10, 10], pbc=True)
        pipe = Pipeline(StaticSource(DataCollection.from_ase(atoms)))
        pipe.modifiers.append(HeightAboveMin())
        out = pipe.compute(0)
        assert np.allclose(out["Height"], [0, 2, 5])
        assert np.isclose(out.attributes["HeightAboveMin.max"], 5)
''')

WRONG_TEST = TESTS.replace("[0, 2, 5]", "[0, 2, 4]")


class FakeClient:
    """Answers each messages.create with the next scripted submit_plugin call."""

    def __init__(self, *submissions):
        self.submissions = list(submissions)
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        module, tests = self.submissions.pop(0)
        call = SimpleNamespace(type="tool_use", id=f"toolu_{len(self.calls)}", name="submit_plugin",
                               input={"module_name": "height_above_min", "class_name": "HeightAboveMin",
                                      "category": "Analysis", "module_source": module,
                                      "test_source": tests, "notes": "Angstrom"})
        return SimpleNamespace(content=[call], stop_reason="tool_use")


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    yield


def test_verify_passes_good_plugin():
    report = forge_mod.verify("height_above_min", "HeightAboveMin", GOOD, TESTS)
    assert forge_mod._passed(report), report
    names = [c["name"] for c in report["checks"]]
    assert names == ["import", "class registered", "parameters", "no in-place writes", "GPU code path"]
    assert "Height" in report["checks"][3]["detail"]


def test_verify_reports_in_place_write_and_failing_test():
    report = forge_mod.verify("height_above_min", "HeightAboveMin", MUTATES, WRONG_TEST)
    failed = {c["name"] for c in report["checks"] + report["tests"] if not c["ok"]}
    assert "no in-place writes" in failed and "test_heights" in failed


def test_verify_rejects_contract_violations():
    unregistered = GOOD.replace('@modifier(category="Analysis")\n', "")
    report = forge_mod.verify("height_above_min", "HeightAboveMin", unregistered, TESTS)
    assert report["checks"][-1]["name"] == "class registered" and not report["checks"][-1]["ok"]

    unstored = GOOD.replace("self.axis = axis", "self._axis = axis").replace("self.axis]", "self._axis]")
    report = forge_mod.verify("height_above_min", "HeightAboveMin", unstored, TESTS)
    bad = next(c for c in report["checks"] if c["name"] == "parameters")
    assert not bad["ok"] and "axis" in bad["detail"]

    assert not forge_mod._passed(forge_mod.verify("1bad", "HeightAboveMin", GOOD, TESTS))


def test_verify_time_limit():
    slow = GOOD.replace("import numpy as np", "import time\nimport numpy as np\ntime.sleep(30)")
    report = forge_mod.verify("height_above_min", "HeightAboveMin", slow, TESTS, timeout=3)
    assert report["checks"][0]["name"] == "time limit"


def test_forge_fixes_after_feedback_and_saves():
    client = FakeClient((MUTATES, TESTS), (GOOD, TESTS))
    events = []
    result = forge_mod.forge("height of atoms above the lowest one", client=client, run_code=True,
                             on_event=lambda kind, r: events.append(kind))
    assert result.ok and result.rounds == 2, result.summary()
    assert events == ["draft", "report", "draft", "report"]

    feedback = client.calls[1]["messages"][2]["content"][0]   # user, assistant, feedback
    assert feedback["type"] == "tool_result" and feedback["is_error"] and feedback["tool_use_id"] == "toolu_1"
    assert "no in-place writes" in feedback["content"]
    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": "submit_plugin"}
    assert "HeightAboveMin" not in client.calls[0]["system"][0]["text"]   # reference has built-ins only

    assert result.path == plugins.plugin_dir() / "height_above_min.py"
    assert result.path.read_text().startswith("# Generated by ewald forge")
    assert (plugins.plugin_dir() / "_forge_tests" / "test_height_above_min.py").exists()
    from ewald.modifiers import HeightAboveMin, HeightAboveMinModifier
    assert HeightAboveMin is HeightAboveMinModifier
    assert "HeightAboveMin" in plugins.modifiers()

    with pytest.raises(FileExistsError):
        forge_mod.forge("again", client=FakeClient((GOOD, TESTS)), run_code=True)
    assert forge_mod.forge("again", client=FakeClient((GOOD, TESTS)), run_code=True, overwrite=True).ok


def test_forge_gives_up_after_max_rounds():
    client = FakeClient(*[(MUTATES, TESTS)] * 2)
    result = forge_mod.forge("x", client=client, run_code=True, max_rounds=2)
    assert not result.ok and result.verified and result.rounds == 2 and result.path is None
    assert "FAILED" in result.summary()
    assert not (plugins.plugin_dir() / "height_above_min.py").exists()


def test_forge_without_run_code_runs_nothing():
    result = forge_mod.forge("x", client=FakeClient((MUTATES, TESTS)), run_code=False)
    assert not result.verified and not result.ok and result.path is None
    assert result.module_source == MUTATES and "unverified" in result.summary()
    assert not (plugins.plugin_dir() / "height_above_min.py").exists()


def test_cli_asks_first_then_saves(monkeypatch, capsys):
    from ewald.__main__ import main

    real = forge_mod.forge
    monkeypatch.setattr(forge_mod, "forge", lambda *a, **kw: real(*a, client=FakeClient((MUTATES, TESTS), (GOOD, TESTS)), **kw))
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert main(["forge", "height", "above", "min"]) == 1
    assert not (plugins.plugin_dir() / "height_above_min.py").exists()

    assert main(["forge", "height", "--draft"]) is None
    assert "data.positions[0, 0] += 1e-3" in capsys.readouterr().out

    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert main(["forge", "height", "above", "min"]) is None
    out = capsys.readouterr().out
    assert "failed: no in-place writes" in out and "passed after 2 round(s)" in out
    assert (plugins.plugin_dir() / "height_above_min.py").exists()


def test_forge_asks_before_saving():
    seen = []
    result = forge_mod.forge("x", client=FakeClient((GOOD, TESTS)), run_code=True,
                             confirm=lambda r: seen.append(r.module_source) or False)
    assert result.ok and result.path is None and seen == [GOOD]
    assert not (plugins.plugin_dir() / "height_above_min.py").exists()
