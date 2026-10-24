"""The AI eval harness (ewald.llm.evals): tasks, graders, usage and the CLI, with scripted
clients only. Real model runs cost money and are never part of the suite."""
import json
from types import SimpleNamespace as NS

import pytest

from ewald.__main__ import main
from ewald.llm import evals
from ewald.llm import tools as tools_mod
from ewald.llm.eval_tasks import CATEGORIES, TASKS


@pytest.fixture
def fake_render(monkeypatch):
    from PIL import Image

    def render_png(data, path, **kw):
        Image.new("RGB", (8, 6), "white").save(path)

    monkeypatch.setattr(tools_mod, "render_png", render_png)


def test_tasks_are_well_formed():
    ids = [t.id for t in TASKS]
    assert len(ids) == len(set(ids)) and 30 <= len(TASKS) <= 60
    assert set(CATEGORIES) == {"answer", "step", "view", "export", "code", "notebook", "recipe", "plugin", "paper"}
    for t in TASKS:
        assert t.checks and t.solution and t.answer, t.id
        numbers = [c for c in t.checks if isinstance(c, evals.Number)]
        assert t.answer.count("{") == len(numbers), t.id   # the reference answer fills every Number
        if t.category in ("step", "view", "notebook", "recipe"):
            assert any(c.state for c in t.checks), t.id   # graded on the notebook's state, re-run too


def test_every_reference_solution_passes(fake_render):
    """Each task's reference solution, played through the real agent loop and tools, passes every
    check and reproduces from its notebook: the tasks are solvable and the graders agree."""
    tasks = [t for t in TASKS if not t.needs_download]
    report = evals.run(tasks, dry_run=True)
    failed = {r["id"]: (r["checks"], r["error"], r["reproduce_detail"]) for r in report["results"] if not r["passed"]}
    assert not failed
    s = report["summary"]
    assert s["ran"] == len(tasks) and s["accuracy"] == 1.0 and s["reproducible"] == 1.0
    assert s["tool_errors"] == 0   # the solutions call the tools correctly


def _play(task_id, calls=None, answer=None, **kw):
    task = next(t for t in TASKS if t.id == task_id)
    client = evals.SolutionClient(task.solution if calls is None else calls, answer or "")
    return evals.run_task(task, client=client, agent_options={"model": "claude-opus-5-5"}, **kw)


def test_wrong_answers_fail():
    assert _play("sto-count", answer="ANSWER: 40, 24").passed
    r = _play("sto-count", answer="ANSWER: 40, 20")
    assert not r.passed and r.checks[0][1] and not r.checks[1][1]
    r = _play("sto-count", answer="There are 40 atoms and 24 O.")      # no ANSWER line
    assert not r.passed and "no value" in r.checks[0][2]
    assert _play("sto-spacegroup", answer="Pm-3m").passed
    assert not _play("sto-spacegroup", answer="Fm-3m").passed
    assert not _play("sto-white", calls=[("set_view", {"background": "black", "bonds": False})]).passed
    assert not _play("sto-color-cn", calls=[("add_modifier", {"name": "CoordinationAnalysis"})]).passed


def test_answer_values():
    assert evals.answer_values("x\nANSWER: 3.905") == [3.905]
    assert evals.answer_values("**ANSWER:** 1.23 x 10^-5 cm^2/s, 4") == pytest.approx([1.23e-5, 4.0])
    assert evals.answer_values("ANSWER: 1.2e-5, −3") == pytest.approx([1.2e-5, -3.0])
    assert evals.answer_values("Answer: 12\nanswer: 6") == [6.0]   # the last ANSWER line
    assert evals.answer_values("no line") == []


def test_reproducibility_catches_state_the_notebook_does_not_make():
    # the model ran a cell, then edited it without running it again: the view is white now,
    # but the notebook makes it black
    code = "view(pipe, background='white', bonds=False)"
    calls = [("insert_cell", {"source": code}), ("run_cell", {"index": -1}),
             ("edit_cell", {"index": -1, "old_text": "'white'", "new_text": "'black'", "run": False})]
    r = _play("sto-white", calls=calls, answer="Done.", allow_code=True)
    assert r.passed and r.reproducible is False and "background" in r.reproduce_detail
    r = _play("sto-white", calls=calls[:2], answer="Done.", allow_code=True)
    assert r.passed and r.reproducible is True


def test_code_tasks_need_allow_code():
    task = next(t for t in TASKS if t.id == "cu-code-height")
    r = evals.run_task(task, client=evals.SolutionClient(task.solution, "Done."))
    assert r.passed is None and "allow-code" in r.skipped
    r = evals.run_task(task, client=evals.SolutionClient(task.solution, "Done."), allow_code=True)
    assert r.passed and r.reproducible


class MeteredClient:
    """Plays responses that carry Anthropic usage."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **params):
        return self.responses.pop(0)


def _usage(i, o, read=0, write=0):
    return NS(input_tokens=i, output_tokens=o, cache_read_input_tokens=read, cache_creation_input_tokens=write)


def test_usage_tokens_and_cost():
    client = MeteredClient([
        NS(stop_reason="tool_use", usage=_usage(1000, 50, write=4000),
           content=[NS(type="tool_use", id="t1", name="describe_frame", input={})]),
        NS(stop_reason="end_turn", usage=_usage(200, 30, read=4000), content=[NS(type="text", text="ANSWER: 40, 24")]),
    ])
    task = next(t for t in TASKS if t.id == "sto-count")
    r = evals.run_task(task, client=client, agent_options={"model": "claude-opus-5-5"})
    assert r.passed and r.requests == 2 and r.tool_calls == 1 and r.tools == ["describe_frame"]
    assert r.tokens == {"input": 1200, "output": 80, "cache_read": 4000, "cache_write": 4000}
    assert r.cost == pytest.approx((1200 * 4 + 80 * 20 + 4000 * 0.2 + 4000 * 5) / 1e6)
    assert evals.cost(r.tokens, "some-local-model") is None
    assert evals.cost({"input": 10 ** 6}, "m", {"m": (1.0, 2.0)}) == pytest.approx(1.0)


def test_openai_usage_shape():
    usage = NS(prompt_tokens=500, completion_tokens=20, prompt_tokens_details=NS(cached_tokens=300))
    meter = evals.UsageMeter(NS(chat=NS(completions=NS(create=lambda **kw: NS(usage=usage)))))
    meter.chat.completions.create(model="x")
    assert meter.totals == {"requests": 1, "input": 200, "output": 20, "cache_read": 300, "cache_write": 0}


def test_fatal_errors_stop_the_run():
    class AuthenticationError(Exception):
        pass

    def create(**params):
        raise AuthenticationError("invalid x-api-key")

    with pytest.raises(evals.FatalError, match="invalid x-api-key"):
        evals.run(evals.select(TASKS, "sto-count"), client=NS(beta=NS(messages=NS(create=create))))


def test_select_compare_and_report(tmp_path):
    assert {t.id for t in evals.select(TASKS, "sto-white,cu-f*")} == {"cu-frames", "cu-fmax", "sto-white"}
    assert all(t.category == "view" for t in evals.select(TASKS, category="view"))
    a = evals.run(evals.select(TASKS, "sto-count,sto-white"), dry_run=True)
    b = json.loads(json.dumps(a))
    b["results"][0]["passed"] = False
    b["summary"] = evals.summarize(b["results"])
    text = evals.compare(a, b)
    assert "now failing: sto-count" in text and "100.0% ->" in text
    assert "accuracy 100.0%: 2/2" in evals.format_summary(a)
    evals.save(a, tmp_path / "a.json")
    assert evals.load(tmp_path / "a.json")["summary"]["passed"] == 2


def test_cli(tmp_path, capsys):
    main(["eval", "--list", "--category", "view"])
    out = capsys.readouterr().out
    assert "sto-white" in out and "sto-count" not in out
    path = tmp_path / "run.json"
    assert main(["eval", "--dry-run", "--tasks", "sto-white,sto-count", "--out", str(path)]) == 0
    out = capsys.readouterr().out
    assert "PASS  sto-white" in out and "accuracy 100.0%" in out
    main(["eval", "--compare", str(path), str(path)])
    assert "accuracy" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no task matches"):
        main(["eval", "--tasks", "nope"])


class RecordingClient(evals.SolutionClient):
    """A SolutionClient that keeps the requests it got."""

    def __init__(self, calls, answer):
        super().__init__(calls, answer)
        self.requests = []
        create = self._create
        self.beta.messages.create = lambda **params: (self.requests.append(params), create(**params))[1]


def _first_text(request):
    content = request["messages"][0]["content"]
    return content if isinstance(content, str) else "".join(b.get("text", "") for b in content if isinstance(b, dict))


def test_gui_context_asks_like_the_chat_side_bar():
    """By default a task is asked as the GUI's AI chat asks: the notebook note in the system prompt,
    the notebook's code (and failing cells) before the question; 'plain' is the bare agent."""
    from ewald.llm.notebook_prompt import RUN_CELLS_NOTE

    task = next(t for t in TASKS if t.id == "nb-fix-error")
    client = RecordingClient(task.solution, "Done.")
    r = evals.run_task(task, client=client, agent_options={"model": "claude-opus-5-5"}, allow_code=True)
    assert r.passed and r.reproducible
    system = json.dumps(client.requests[0]["system"])
    assert "chat side bar of ewald's notebook window" in system and json.dumps(RUN_CELLS_NOTE[True])[1:-1] in system
    text = _first_text(client.requests[0])
    assert text.startswith("The notebook now reads:") and "CoordinationAnalysis(cutof=3.0)" in text
    assert "These notebook cells failed" in text and text.rstrip().endswith(evals.ANSWER_HINT.strip())

    client = RecordingClient(task.solution, "Done.")
    evals.run_task(task, client=client, agent_options={"model": "claude-opus-5-5"}, allow_code=True, context="plain")
    assert "notebook window" not in json.dumps(client.requests[0]["system"])
    assert _first_text(client.requests[0]).startswith(task.prompt)


def test_answer_code_blocks_become_cells():
    """As with Preferences > AI > Code in answers = insert: the answer's python blocks are added as
    cells, run when the AI may run cells, left waiting (and so not graded as done) otherwise."""
    answer = "Here you go:\n```python\nview(pipe, background='white', bonds=False)\n```\n"
    r = _play("sto-white", calls=[], answer=answer, allow_code=True)
    assert r.passed and r.reproducible and "view(pipe, background='white'" in r.notebook
    r = _play("sto-white", calls=[], answer=answer)
    assert not r.passed and "view(pipe, background='white'" in r.notebook   # inserted, not run
    r = _play("sto-white", calls=[], answer=answer, allow_code=True, context="plain")
    assert not r.passed and "background='white'" not in r.notebook


def test_plugin_tasks_load_and_unload_their_plugins():
    from ewald import plugins

    assert "bond_valence_sums" not in plugins.tools()
    r = _play("plugin-tool", answer="ANSWER: 4.138")
    assert r.passed and r.tools == ["bond_valence_sums"]
    assert "bond_valence_sums" not in plugins.tools()
    r = _play("plugin-tool", calls=[], answer="ANSWER: 4.14")       # right number, tool not used
    assert not r.passed and not dict((c[0], c[1]) for c in r.checks)["UsedTool bond_valence_sums"]
    r = _play("plugin-modifier", answer="Done.")
    assert r.checks[0][1] and "LocalComposition" not in plugins.modifiers()   # the step was added


def test_notebook_checks_need_the_cell_edited_not_added_again():
    r = _play("nb-edit-cutoff", calls=[("add_modifier", {"name": "CoordinationAnalysis", "params": {"cutoff": 2.5}})],
              answer="Done.")
    assert not r.passed and "coordination cells: 2 vs 1" in " ".join(c[2] for c in r.checks)


def test_tool_errors_are_kept():
    r = _play("sto-count", calls=[("add_modifier", {"name": "NoSuchModifier"}), ("describe_frame", {})],
              answer="ANSWER: 40, 24")
    assert r.passed and r.tool_errors == 1 and r.tool_error_messages[0][0] == "add_modifier"
    assert "NoSuchModifier" in r.tool_error_messages[0][1]
    assert "~ add_modifier error:" in evals.format_result(r)
