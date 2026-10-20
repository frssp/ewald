"""AI eval suite: how well a model drives ewald, scored on tasks with known answers.

Each :class:`Task` (the built-in ones are in :mod:`ewald.llm.eval_tasks`) gives the model a
question about a shipped example file (or a paper dataset), runs the normal :class:`Agent` loop
over a fresh notebook session, and grades the outcome automatically:

* numbers in the answer's final ``ANSWER:`` line, within a tolerance (:class:`Number`), and text
  patterns (:class:`Matches`);
* the state the model left behind: pipeline steps (:class:`Step`), view settings
  (:class:`View`), computed data (:class:`Data`), written files (:class:`File`) and cell outputs
  (:class:`CellNumber`);
* reproducibility: the cells that ran are run again in a fresh namespace and every state check
  must still pass, so the notebook alone reproduces what the AI did (the rule that the AI
  changes results only through cells).

Per task it records pass/fail, tool calls, tool errors, requests, tokens and cost; a run is
saved as JSON and two runs can be compared (:func:`compare`). ``ewald eval`` is the command
line. Runs cost money and need a key, so nothing here is part of the default test suite;
``ewald eval --dry-run`` plays each task's reference solution instead of a model (no key, no
network), which checks the tasks and graders on this machine.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from ..notebook import Cell, Notebook
from .tools import Toolset

#: Appended to every task prompt, so numeric answers can be graded.
ANSWER_HINT = ("\n\n(This question is part of an automated evaluation. Work in the notebook as usual. "
               "When the question asks for one or more values, end your answer with a single line "
               "'ANSWER: <value>' (several values separated by commas, in the order asked, in the units asked, "
               "numbers only).)")

#: USD per million tokens: input, output, cache read, cache write (5-minute). First-party API list
#: prices as of 2026-10; other models (and other providers) need ``prices=`` / ``--price``.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0, 0.25, 12.5),
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_TIMES = re.compile(r"\s*(?:x|×|\*)\s*10\s*\^?\s*([-+−]?\d+)")


# ------------------------------------------------------------------ run context
@dataclass
class Outcome:
    """What a task run left behind, as the checks see it."""

    answer: str
    toolset: Toolset
    workdir: str
    reference: Callable          # () -> a fresh pipeline over the task's file, for reference values

    @property
    def notebook(self) -> Notebook:
        return self.toolset.notebook

    @property
    def pipeline(self):
        return self.toolset.pipeline

    def data(self, frame=None):
        pipe = self.pipeline
        if pipe is None:
            raise RuntimeError("the notebook has no pipeline")
        if frame is None:
            frame = pipe.view.frame
        return pipe.compute(int(frame) % pipe.num_frames)


def answer_values(text: str) -> list[float]:
    """The values of the last ``ANSWER:`` line: the first number of each comma-separated field,
    so units after it are ignored (``1.2e-5`` and ``1.2 x 10^-5 cm^2/s`` both read as 1.2e-5)."""
    lines = [ln for ln in (text or "").splitlines() if re.match(r"\s*\**\s*ANSWER\s*\**\s*:", ln, re.I)]
    if not lines:
        return []
    out = []
    for part in lines[-1].split(":", 1)[1].replace("\u2212", "-").split(","):
        m = _NUMBER.search(part)
        if m is None:
            continue
        value = float(m.group())
        power = _TIMES.match(part, m.end())
        if power and not re.search(r"[eE]", m.group()):
            value *= 10 ** int(power.group(1))
        out.append(value)
    return out


def _close(value, expected, tol=None, rel=None) -> bool:
    if tol is None and rel is None:
        tol = 0.0
    bound = max(tol or 0.0, abs(expected) * (rel or 0.0))
    return abs(float(value) - float(expected)) <= bound + 1e-12


def _resolve(expected, outcome: Outcome):
    return expected(outcome.reference()) if callable(expected) else expected


# ------------------------------------------------------------------ checks
class Check:
    """One graded condition. ``state`` checks are re-run on the reproduced notebook."""

    state = False

    def grade(self, outcome: Outcome) -> tuple[bool, str]:
        raise NotImplementedError

    def label(self) -> str:
        return type(self).__name__


@dataclass
class Number(Check):
    """The ``index``-th number of the answer's ``ANSWER:`` line equals ``expected`` (a value or a
    function of a fresh reference pipeline) within ``tol`` (absolute) or ``rel`` (relative)."""

    expected: object
    tol: float | None = None
    rel: float | None = None
    index: int = 0
    name: str = ""

    def grade(self, outcome):
        expected = float(_resolve(self.expected, outcome))
        values = answer_values(outcome.answer)
        if len(values) <= self.index:
            return False, f"no value #{self.index + 1} in an ANSWER line (expected {expected:.6g})"
        got = values[self.index]
        return _close(got, expected, self.tol, self.rel), f"{got:.6g} vs {expected:.6g}"

    def label(self):
        return f"Number {self.name or self.index}"


@dataclass
class Matches(Check):
    """The answer matches ``pattern`` (a regular expression, case-insensitive)."""

    pattern: str

    def grade(self, outcome):
        ok = re.search(self.pattern, outcome.answer or "", re.I | re.S) is not None
        return ok, f"/{self.pattern}/ {'found' if ok else 'not found'}"

    def label(self):
        return f"Matches {self.pattern}"


def _class_name(m) -> str:
    return type(m).__name__


@dataclass
class Step(Check):
    """The pipeline has an enabled step of class ``name`` (OVITO's ``...Modifier`` names count)
    whose parameters match ``params`` (numbers within 1e-6, a callable gets the value)."""

    name: str
    params: dict = field(default_factory=dict)
    state = True

    def grade(self, outcome):
        pipe = outcome.pipeline
        if pipe is None:
            return False, "no pipeline"
        name = self.name.removesuffix("Modifier")
        found = [m for m in pipe.modifiers if m.enabled and _class_name(m).removesuffix("Modifier") == name]
        if not found:
            return False, f"no {self.name} step (steps: {[_class_name(m) for m in pipe.modifiers]})"
        for m in found:
            if all(_param_ok(getattr(m, k, None), v) for k, v in self.params.items()):
                return True, repr(m)
        return False, f"{self.name} steps with other parameters: {[repr(m) for m in found]}"

    def label(self):
        return f"Step {self.name}"


def _param_ok(value, expected) -> bool:
    if callable(expected):
        return bool(expected(value))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return _close(value, expected, 1e-6)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, (list, tuple)):
        try:
            return len(value) == len(expected) and all(_param_ok(a, b) for a, b in zip(value, expected))
        except TypeError:
            return False
    return value == expected


def _rgb(c):
    from matplotlib.colors import to_rgb

    return tuple(round(x, 3) for x in to_rgb(c))


def same_color(a, b) -> bool:
    try:
        return _rgb(a) == _rgb(b)
    except (ValueError, TypeError):
        return False


def _direction(d):
    import numpy as np

    axes = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    if isinstance(d, str):
        s = d.strip().lower()
        sign = -1.0 if s.startswith("-") else 1.0
        s = s.lstrip("+-")
        if s in axes:
            return sign * np.array(axes[s], float)
        return None
    v = np.asarray(d, float).ravel()
    n = np.linalg.norm(v)
    return v / n if v.shape == (3,) and n else None


def along(axis: str, both: bool = True):
    """A :class:`View` matcher: the view direction is ``axis`` ('x', 'y', 'z'), either sign
    when ``both``."""
    target = _direction(axis)

    def match(value):
        v = _direction(value)
        if v is None:
            return False
        dot = float(v @ target)
        return dot > 0.999 or (both and dot < -0.999)
    return match


@dataclass
class View(Check):
    """``pipe.view`` has these settings. Colors compare as colors; a dict setting (colors,
    radii, ...) needs only the given keys; a callable gets the value."""

    settings: dict
    state = True

    def __init__(self, **settings):
        self.settings = settings

    def grade(self, outcome):
        pipe = outcome.pipeline
        if pipe is None:
            return False, "no pipeline"
        view = pipe.view
        bad = []
        for name, expected in self.settings.items():
            value = getattr(view, name)
            if not _view_ok(name, value, expected):
                bad.append(f"{name}={value!r}")
        return not bad, ("; ".join(bad) + f" (wanted {self.settings})") if bad else f"{self.settings}"

    def label(self):
        return "View " + ",".join(self.settings)


def _view_ok(name, value, expected) -> bool:
    if callable(expected):
        return bool(expected(value))
    if isinstance(expected, dict):
        if not isinstance(value, dict):
            return False
        return all(k in value and _view_ok(name, value[k], v) for k, v in expected.items())
    if "color" in name or name in ("background", "colors"):
        return value == expected if value is None or expected is None else same_color(value, expected)
    if isinstance(expected, (int, float)) and not isinstance(expected, bool) and not isinstance(value, bool):
        try:
            return _close(value, expected, 1e-6)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, (list, tuple)):
        return isinstance(value, (list, tuple)) and len(value) == len(expected) and \
            all(_view_ok(name, a, b) for a, b in zip(value, expected))
    if isinstance(expected, str) and isinstance(value, str):
        return value.strip().lower() == expected.strip().lower()
    return value == expected


@dataclass
class Data(Check):
    """``value(data)`` of the final pipeline output (the view's frame, or ``frame``) equals
    ``expected`` (a value or a function of a fresh reference pipeline) within ``tol``/``rel``;
    ``value`` may also return a bool (then it must be True)."""

    value: Callable
    expected: object = True
    tol: float | None = None
    rel: float | None = None
    frame: int | None = None
    name: str = "data"
    state = True

    def grade(self, outcome):
        try:
            got = self.value(outcome.data(self.frame))
        except Exception as exc:   # a missing property is a failed check, not a crash
            return False, f"{type(exc).__name__}: {exc}"
        expected = _resolve(self.expected, outcome)
        if isinstance(expected, bool):
            return bool(got) is expected, f"{self.name}: {got!r}"
        return _close(got, expected, self.tol, self.rel), f"{self.name}: {float(got):.6g} vs {float(expected):.6g}"

    def label(self):
        return f"Data {self.name}"


@dataclass
class File(Check):
    """A file ``pattern`` (glob, relative to the task's working folder, where relative paths
    the model writes land) exists and ``check(path, outcome)`` holds."""

    pattern: str
    check: Callable | None = None

    def grade(self, outcome):
        hits = sorted(Path(outcome.workdir).glob(self.pattern))
        if not hits:
            return False, f"no file {self.pattern} in the working folder"
        if self.check is None:
            return True, hits[0].name
        try:
            ok = bool(self.check(hits[0], outcome))
        except Exception as exc:
            return False, f"{hits[0].name}: {type(exc).__name__}: {exc}"
        return ok, hits[0].name

    def label(self):
        return f"File {self.pattern}"


@dataclass
class CellNumber(Check):
    """A code cell printed a number equal to ``expected`` within ``tol``/``rel`` (any number in
    any text output of any cell that ran)."""

    expected: object
    tol: float | None = None
    rel: float | None = None
    state = True

    def grade(self, outcome):
        expected = float(_resolve(self.expected, outcome))
        seen = []
        for cell in outcome.notebook.cells:
            for out in cell.outputs:
                if out.kind in ("text", "value"):
                    for m in _NUMBER.finditer(str(out.data)):
                        v = float(m.group())
                        seen.append(v)
                        if _close(v, expected, self.tol, self.rel):
                            return True, f"{v:.6g} vs {expected:.6g}"
        return False, f"no cell printed {expected:.6g} (saw {seen[:8]})"

    def label(self):
        return "CellNumber"


@dataclass
class Any(Check):
    """At least one of ``checks`` passes."""

    checks: tuple

    def __init__(self, *checks):
        self.checks = checks

    @property
    def state(self):
        return any(c.state for c in self.checks)

    def grade(self, outcome):
        details = []
        for c in self.checks:
            ok, detail = c.grade(outcome)
            if ok:
                return True, detail
            details.append(detail)
        return False, " | ".join(details)

    def label(self):
        return "Any(" + ", ".join(c.label() for c in self.checks) + ")"


# ------------------------------------------------------------------ tasks
@dataclass
class Task:
    """A question with a gradable outcome.

    ``file`` is an example file name (:mod:`ewald.examples`) or ``"paper:<dataset>/<file>"``
    (downloaded once with :func:`ewald.examples.fetch`); it is loaded before the model starts,
    as when the user opens it. ``solution`` is the reference: tool calls (``(name, arguments)``)
    and a final answer, where ``{0}``, ``{1}``, ... are the expected values of the task's
    :class:`Number` checks. ``allow_code`` marks tasks that need the model to run cells it
    wrote (skipped unless the run allows it)."""

    id: str
    prompt: str
    file: str
    checks: list
    category: str = "answer"
    solution: list = field(default_factory=list)
    answer: str = ""
    allow_code: bool = False
    reproducible: bool = True     # also grade the state checks on the re-run notebook

    def resolve_file(self, fetch: bool = True) -> Path:
        from .. import examples

        if self.file.startswith("paper:"):
            name, _, filename = self.file[len("paper:"):].partition("/")
            folder = examples.fetch(name, quiet=True) if fetch else _paper_folder(name)
            return Path(folder) / filename
        return examples.path(self.file)

    @property
    def needs_download(self) -> bool:
        return self.file.startswith("paper:")


def _paper_folder(name: str) -> Path:
    from ..examples import papers

    return papers.data_dir() / papers.get(name).name


@dataclass
class TaskResult:
    id: str
    category: str
    passed: bool | None            # None: skipped or failed to run
    checks: list = field(default_factory=list)     # [label, ok, detail]
    reproducible: bool | None = None
    reproduce_detail: str = ""
    tool_calls: int = 0
    tool_errors: int = 0
    tools: list = field(default_factory=list)      # tool names in call order
    requests: int = 0
    tokens: dict = field(default_factory=dict)     # input, output, cache_read, cache_write
    cost: float | None = None
    seconds: float = 0.0
    answer: str = ""
    notebook: str = ""
    error: str = ""
    skipped: str = ""


# ------------------------------------------------------------------ usage metering
class UsageMeter:
    """Wraps a provider client and adds up the ``usage`` of every response (Anthropic Messages
    and OpenAI Chat Completions shapes)."""

    _PATHS = {"beta", "messages", "chat", "completions"}

    def __init__(self, client, totals: dict | None = None):
        self._client = client
        self._totals = totals if totals is not None else {"requests": 0, "input": 0, "output": 0,
                                                           "cache_read": 0, "cache_write": 0}

    @property
    def totals(self) -> dict:
        return self._totals

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if name in self._PATHS:
            return UsageMeter(attr, self._totals)
        if name == "create":
            def create(*args, **kwargs):
                response = attr(*args, **kwargs)
                self._add(getattr(response, "usage", None))
                return response
            return create
        return attr

    def _add(self, usage):
        t = self._totals
        t["requests"] += 1
        if usage is None:
            return

        def get(obj, name):
            v = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
            return int(v or 0) if isinstance(v, (int, float)) or v is None else 0

        if get(usage, "prompt_tokens") or get(usage, "completion_tokens"):   # OpenAI
            details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else \
                getattr(usage, "prompt_tokens_details", None)
            cached = get(details, "cached_tokens") if details is not None else 0
            t["input"] += get(usage, "prompt_tokens") - cached
            t["cache_read"] += cached
            t["output"] += get(usage, "completion_tokens")
        else:
            t["input"] += get(usage, "input_tokens")
            t["output"] += get(usage, "output_tokens")
            t["cache_read"] += get(usage, "cache_read_input_tokens")
            t["cache_write"] += get(usage, "cache_creation_input_tokens")


def cost(tokens: dict, model: str, prices: dict | None = None) -> float | None:
    """USD for ``tokens`` (as :class:`UsageMeter` counts them) at ``prices[model]`` or
    :data:`PRICES`; None for a model without a price."""
    table = {**PRICES, **(prices or {})}
    price = table.get(model)
    if price is None:
        return None
    p_in, p_out, *rest = price
    p_read = rest[0] if rest else p_in * 0.1
    p_write = rest[1] if len(rest) > 1 else p_in * 1.25
    return (tokens.get("input", 0) * p_in + tokens.get("output", 0) * p_out
            + tokens.get("cache_read", 0) * p_read + tokens.get("cache_write", 0) * p_write) / 1e6


# ------------------------------------------------------------------ reference solutions
class SolutionClient:
    """A stand-in for Claude that plays a task's reference solution: one tool call per turn,
    then the answer. ``ewald eval --dry-run`` and the tests use it; it speaks the Messages API
    shape (``beta.messages.create``) and reports no usage."""

    def __init__(self, calls: list, answer: str):
        self.script = list(calls)
        self.answer = answer
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.messages = self.beta.messages
        self.n = 0

    def _create(self, **params):
        self.n += 1
        if self.script:
            name, arguments = self.script.pop(0)
            block = SimpleNamespace(type="tool_use", id=f"s{self.n}", name=name, input=dict(arguments))
            return SimpleNamespace(stop_reason="tool_use", content=[block], usage=None)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=self.answer)],
                               usage=None)


def solution_answer(task: Task, outcome_reference: Callable) -> str:
    values = []
    for c in task.checks:
        if isinstance(c, Number):
            v = float(c.expected(outcome_reference()) if callable(c.expected) else c.expected)
            values.append(f"{v:.6g}")
    return task.answer.format(*values) if values else task.answer


# ------------------------------------------------------------------ running
def select(tasks: list[Task], patterns=None, category=None) -> list[Task]:
    """Tasks whose id matches one of ``patterns`` (globs, comma-separated string or list) and
    whose category is ``category``."""
    if isinstance(patterns, str):
        patterns = [p.strip() for p in patterns.split(",") if p.strip()]
    out = [t for t in tasks if not patterns or any(fnmatch.fnmatch(t.id, p) for p in patterns)]
    if category:
        cats = {c.strip() for c in str(category).split(",")}
        out = [t for t in out if t.category in cats]
    return out


def reproduce(nb: Notebook, workdir: str) -> Notebook:
    """The cells of ``nb`` that ran, run again from the top in a fresh namespace (``workdir``
    as the current folder, so relative paths land where they did)."""
    cells = [Cell(c.source, kind=c.kind, title=c.title, frames=c.frames)
             for c in nb.cells if c.kind != "code" or c.count is not None]
    fresh = Notebook(cells, name=nb.name, pipeline_name=nb.pipeline_name)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        fresh.run_all(stop_on_error=False)
    finally:
        os.chdir(cwd)
    return fresh


class FatalError(RuntimeError):
    """An error every later task would hit too (no key, wrong model): the run stops."""


_FATAL = ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "ImportError",
          "ModuleNotFoundError")


def run_task(task: Task, *, agent_options: dict | None = None, client=None, allow_code: bool = False,
             dry_run: bool = False, max_turns: int = 30, prices: dict | None = None,
             fetch: bool = True, on_event: Callable | None = None) -> TaskResult:
    """Run one task with the model in ``agent_options`` (``provider``, ``model``, ``effort``,
    ``base_url``: what :func:`ewald.llm.providers.from_settings` returns) or ``client``, or the
    task's reference solution with ``dry_run``; grade it."""
    from .agent import Agent

    result = TaskResult(task.id, task.category, None)
    if task.allow_code and not (allow_code or dry_run):
        result.skipped = "needs --allow-code (the model runs cells it wrote)"
        return result
    try:
        path = task.resolve_file(fetch=fetch)
    except Exception as exc:
        result.skipped = f"data not available: {type(exc).__name__}: {exc}"
        return result
    if not path.exists():
        result.skipped = f"data not available: {path} (run once with network access)"
        return result

    workdir = tempfile.mkdtemp(prefix=f"ewald-eval-{task.id}-")
    cwd = os.getcwd()
    started = time.perf_counter()
    try:
        os.chdir(workdir)
        # the reference solutions are ours, so a dry run may run their cells
        ts = Toolset(allow_code=allow_code or (dry_run and task.allow_code), workdir=workdir)
        loaded = ts.call("load_structure", {"path": str(path)})
        if loaded.is_error:
            raise RuntimeError(f"could not load {path}: {loaded.content}")

        def reference():
            from .. import import_file
            return import_file(str(path))

        options = dict(agent_options or {})
        if dry_run:
            agent_client = SolutionClient(task.solution, solution_answer(task, reference))
            options = {"provider": "anthropic", "model": "solution"}
        else:
            if client is None:
                provider = options.get("provider")
                from .providers import get
                provider = get(provider)
                options["provider"] = provider
                client = provider.client(base_url=options.pop("base_url", None),
                                         model=options.get("model") or provider.default_model)
            else:
                options.pop("base_url", None)
            agent_client = client
        meter = UsageMeter(agent_client)

        def event(kind, payload):
            if kind == "tool_call":
                result.tools.append(payload["name"])
            elif kind == "tool_result" and payload["result"].is_error:
                result.tool_errors += 1
            if on_event:
                on_event(kind, payload)

        agent = Agent(ts, client=meter, max_turns=max_turns, on_event=event,
                      fallbacks=not dry_run and options.get("fallbacks", True),
                      **{k: v for k, v in options.items() if k in ("provider", "model", "effort")})
        answer = agent.ask(task.prompt + ANSWER_HINT)
        result.answer = answer
        result.tool_calls = len(result.tools)
        result.requests = meter.totals["requests"]
        result.tokens = {k: meter.totals[k] for k in ("input", "output", "cache_read", "cache_write")}
        result.cost = None if dry_run else cost(result.tokens, agent.model, prices)
        result.notebook = ts.notebook.to_py() if ts.notebook is not None else ""

        outcome = Outcome(answer, ts, workdir, reference)
        for check in task.checks:
            ok, detail = _grade(check, outcome)
            result.checks.append([check.label(), ok, detail])
        result.passed = all(ok for _, ok, _ in result.checks)

        state = [c for c in task.checks if c.state]
        if task.reproducible and state and ts.notebook is not None:
            fresh_nb = reproduce(ts.notebook, workdir)
            fresh_ts = Toolset(workdir=workdir, notebook=fresh_nb)
            fresh = Outcome(answer, fresh_ts, workdir, reference)
            failed = []
            for check in state:
                ok, detail = _grade(check, fresh)
                if not ok:
                    failed.append(f"{check.label()}: {detail}")
            errors = [k + 1 for k, c in enumerate(fresh_nb.cells) if c.state == "error"]
            result.reproducible = not failed
            result.reproduce_detail = "; ".join(failed) + (f" (cells with errors: {errors})" if errors else "")
    except Exception as exc:
        if type(exc).__name__ in _FATAL:
            raise FatalError(f"{type(exc).__name__}: {exc}") from exc
        result.error = f"{type(exc).__name__}: {exc}"
        result.passed = False
    finally:
        os.chdir(cwd)
        result.seconds = round(time.perf_counter() - started, 2)
        shutil.rmtree(workdir, ignore_errors=True)
    return result


def _grade(check: Check, outcome: Outcome) -> tuple[bool, str]:
    try:
        return check.grade(outcome)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run(tasks: list[Task] | None = None, *, agent_options: dict | None = None, client=None,
        allow_code: bool = False, dry_run: bool = False, repeat: int = 1, max_turns: int = 30,
        prices: dict | None = None, fetch: bool = True, progress: Callable | None = None) -> dict:
    """Run ``tasks`` (default: every built-in one) ``repeat`` times; returns the report
    (:func:`summarize` plus every :class:`TaskResult`), ready for :func:`save`."""
    import ewald

    from .eval_tasks import TASKS

    tasks = list(TASKS if tasks is None else tasks)
    options = dict(agent_options or {})
    provider = options.get("provider")
    model = options.get("model")
    results = []
    for r in range(repeat):
        for task in tasks:
            res = run_task(task, agent_options=options, client=client, allow_code=allow_code, dry_run=dry_run,
                           max_turns=max_turns, prices=prices, fetch=fetch)
            res_dict = asdict(res)
            res_dict["repeat"] = r
            results.append(res_dict)
            if progress:
                progress(res)
    meta = {"provider": "solution" if dry_run else getattr(provider, "name", provider) or "anthropic",
            "model": "solution" if dry_run else (model or getattr(provider, "default_model", None)),
            "effort": options.get("effort"), "allow_code": allow_code, "repeat": repeat,
            "ewald": getattr(ewald, "__version__", "?"), "commit": _commit(),
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    return {"meta": meta, "summary": summarize(results), "results": results}


def _commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def summarize(results: list[dict]) -> dict:
    """Accuracy (over tasks that ran), per category, reproducibility, tool calls, tokens and cost."""
    ran = [r for r in results if r["passed"] is not None]
    passed = [r for r in ran if r["passed"]]
    repro = [r for r in ran if r["reproducible"] is not None]
    cats = {}
    for r in ran:
        c = cats.setdefault(r["category"], [0, 0])
        c[0] += bool(r["passed"])
        c[1] += 1
    tokens = {k: sum(r["tokens"].get(k, 0) for r in ran) for k in ("input", "output", "cache_read", "cache_write")}
    costs = [r["cost"] for r in ran if r["cost"] is not None]
    return {
        "tasks": len(results), "ran": len(ran), "passed": len(passed),
        "skipped": sum(1 for r in results if r["skipped"]), "errors": sum(1 for r in results if r["error"]),
        "accuracy": round(len(passed) / len(ran), 4) if ran else None,
        "reproducible": round(sum(r["reproducible"] for r in repro) / len(repro), 4) if repro else None,
        "by_category": {k: {"passed": v[0], "ran": v[1], "accuracy": round(v[0] / v[1], 4)} for k, v in sorted(cats.items())},
        "tool_calls": sum(r["tool_calls"] for r in ran),
        "tool_calls_per_task": round(sum(r["tool_calls"] for r in ran) / len(ran), 2) if ran else None,
        "tool_errors": sum(r["tool_errors"] for r in ran),
        "requests": sum(r["requests"] for r in ran),
        "tokens": tokens,
        "cost": round(sum(costs), 4) if costs else None,
        "seconds": round(sum(r["seconds"] for r in results), 1),
    }


def save(report: dict, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return path


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ reports
def format_result(r) -> str:
    r = asdict(r) if isinstance(r, TaskResult) else r
    if r["skipped"]:
        return f"SKIP  {r['id']:<26} {r['skipped']}"
    mark = "PASS" if r["passed"] else ("ERR " if r["error"] else "FAIL")
    repro = {True: "repro", False: "NOT-REPRO", None: "-"}[r["reproducible"]]
    cost_s = f"${r['cost']:.3f}" if r["cost"] is not None else "-"
    tok = r["tokens"].get("input", 0) + r["tokens"].get("cache_read", 0) + r["tokens"].get("cache_write", 0)
    line = (f"{mark}  {r['id']:<26} tools {r['tool_calls']:>2} (err {r['tool_errors']})  "
            f"in {tok:>7} out {r['tokens'].get('output', 0):>6}  {cost_s:>7}  {r['seconds']:>6.1f}s  {repro}")
    why = [f"{label}: {detail}" for label, ok, detail in r["checks"] if not ok]
    if r["error"]:
        why.append(r["error"])
    if r["reproducible"] is False:
        why.append("reproduce: " + r["reproduce_detail"])
    return line + "".join(f"\n      - {w}" for w in why)


def format_summary(report: dict) -> str:
    s, m = report["summary"], report["meta"]
    acc = f"{100 * s['accuracy']:.1f}%" if s["accuracy"] is not None else "-"
    rep = f"{100 * s['reproducible']:.1f}%" if s["reproducible"] is not None else "-"
    lines = [f"{m['provider']} / {m['model']}" + (f" (effort {m['effort']})" if m.get("effort") else ""),
             f"accuracy {acc}: {s['passed']}/{s['ran']} passed, {s['skipped']} skipped, {s['errors']} errors; "
             f"reproducible {rep}",
             f"tool calls {s['tool_calls']} ({s['tool_calls_per_task']} per task, {s['tool_errors']} errors), "
             f"requests {s['requests']}",
             f"tokens: input {s['tokens']['input']}, output {s['tokens']['output']}, cache read "
             f"{s['tokens']['cache_read']}, cache write {s['tokens']['cache_write']}; cost "
             + (f"${s['cost']:.2f}" if s["cost"] is not None else
                "-" if m["model"] == "solution" else "unknown (no price for this model; use --price)"),
             "by category: " + ", ".join(f"{k} {v['passed']}/{v['ran']}" for k, v in s["by_category"].items())]
    return "\n".join(lines)


def compare(a: dict, b: dict) -> str:
    """What changed from run ``a`` to run ``b``: accuracy, cost, tool calls, and the tasks that
    flipped (a task run several times counts as passed when most repeats passed)."""
    def rate(report):
        out = {}
        for r in report["results"]:
            if r["passed"] is None:
                continue
            p = out.setdefault(r["id"], [0, 0])
            p[0] += bool(r["passed"])
            p[1] += 1
        return {k: v[0] / v[1] for k, v in out.items()}

    sa, sb = a["summary"], b["summary"]
    ra, rb = rate(a), rate(b)

    def name(rep):
        return f"{rep['meta']['provider']}/{rep['meta']['model']} ({rep['meta'].get('commit') or '?'})"

    def pct(x):
        return "-" if x is None else f"{100 * x:.1f}%"

    lines = [f"A: {name(a)}", f"B: {name(b)}",
             f"accuracy      {pct(sa['accuracy']):>8} -> {pct(sb['accuracy']):>8}",
             f"reproducible  {pct(sa['reproducible']):>8} -> {pct(sb['reproducible']):>8}",
             f"tools/task    {sa['tool_calls_per_task']!s:>8} -> {sb['tool_calls_per_task']!s:>8}",
             f"tool errors   {sa['tool_errors']:>8} -> {sb['tool_errors']:>8}",
             "cost          " + (f"${sa['cost']:.2f}" if sa["cost"] is not None else "-").rjust(8) + " -> "
             + (f"${sb['cost']:.2f}" if sb["cost"] is not None else "-").rjust(8)]
    fixed = sorted(k for k in ra.keys() & rb.keys() if ra[k] < 0.5 <= rb[k])
    broke = sorted(k for k in ra.keys() & rb.keys() if rb[k] < 0.5 <= ra[k])
    if fixed:
        lines.append("now passing: " + ", ".join(fixed))
    if broke:
        lines.append("now failing: " + ", ".join(broke))
    only = sorted(rb.keys() - ra.keys())
    if only:
        lines.append("only in B: " + ", ".join(only))
    return "\n".join(lines)


__all__ = ["ANSWER_HINT", "PRICES", "Any", "CellNumber", "Check", "Data", "File", "Matches", "Number",
           "Outcome", "SolutionClient", "Step", "Task", "TaskResult", "UsageMeter", "View", "along",
           "answer_values", "compare", "cost", "format_result", "format_summary", "load", "reproduce", "run",
           "run_task", "same_color", "save", "select", "summarize"]
