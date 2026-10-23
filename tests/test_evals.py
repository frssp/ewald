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
    assert set(CATEGORIES) == {"answer", "step", "view", "export", "code", "paper"}
    for t in TASKS:
        assert t.checks and t.solution and t.answer, t.id
        numbers = [c for c in t.checks if isinstance(c, evals.Number)]
        assert t.answer.count("{") == len(numbers), t.id   # the reference answer fills every Number
        if t.category in ("step", "view"):
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
