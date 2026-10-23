"""Workflows: calculations that ran in steps (an AiiDA WorkGraph or WorkChain and the
calculations it called), as one tree that a side bar shows, a notebook opens and the AI reads.

A plugin that knows such calculations registers a *source* (:func:`ewald.plugins.workflow_source`)
and fills these dataclasses; ewald does the rest, the same way for every source::

    from ewald.plugin_api import Workflow, WorkflowOutput, WorkflowPage, WorkflowStep, workflow_source

    @workflow_source("mydb", title="My database", schemes=("mydb",))
    class MyDatabase:
        def account(self):            # "" means "not set up": the side bar offers its settings page
            return "alice at https://db.example.org"

        def list(self, query=None, cursor=None):   # newest first, one page
            return WorkflowPage([Workflow("relax", id="12", path="mydb:12")], next=None)

        def describe(self, ref):      # ref: an id the user typed or a path of this source
            return Workflow("relax", id="12", path="mydb:12", complete=True, steps=[
                WorkflowStep("relax", path="mydb:12/relax", state="finished",
                             outputs=[WorkflowOutput("vasprun.xml", "mydb:12/relax/vasprun.xml")])])

Every ``path`` is a string ``load()`` opens (the source's reader serves its scheme), so whatever
the side bar or the AI does with a workflow is something a notebook cell can repeat.

A source that knows how to analyse a kind of workflow (sim-agent's ``vasp_band_offset`` ->
the potential lineup) also has ``analyses(wf) -> [Analysis]``: notebook cells, written by the
plugin from :mod:`ewald.potential` and the like, that load what they need and compute the
result in the open. :func:`notebook_text` appends them and the AI adds one with
``add_analysis``; neither writes analysis code itself.
:func:`notebook_text` turns a workflow into one notebook with a variable per step; the
generic side bar is :func:`workflow_panel`, which a plugin registers as its ``@panel``.
"""
from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field

#: process states, as AiiDA names them, and the mark the side bar shows for each
STATE_MARKS = {"finished": "✓", "excepted": "✗", "killed": "✗", "running": "…", "waiting": "…", "created": "…"}


@dataclass
class WorkflowOutput:
    """Something a step made: a file, a structure, a trajectory, other data.

    ``path`` opens it with ``load()`` ("" when it cannot be opened, only listed); ``kind`` is
    "structure", "trajectory", "band" (NEB images as frames), "volumetric", "file" or "data"; ``note`` is shown after the name
    ("Si2", "on the cluster, over SSH")."""

    name: str
    path: str = ""
    kind: str = "file"
    note: str = ""

    def as_dict(self) -> dict:
        return {k: v for k, v in (("name", self.name), ("path", self.path), ("kind", self.kind),
                                  ("note", self.note)) if v}


@dataclass
class WorkflowStep:
    """One step of a workflow: a calculation, or a sub-workflow with ``steps`` of its own.

    ``path`` opens the step's own result (its trajectory or final structure; "" if none);
    ``runs`` counts the restarts joined into it (a relaxation resubmitted after a walltime);
    ``id`` is the source's own id (an AiiDA PK); ``work_dir`` where it ran, as the source knows
    it; ``scalars`` its numbers (energies, band gap, ...)."""

    name: str
    path: str = ""
    state: str = ""
    exit_status: int | None = None
    runs: int = 1
    id: str = ""
    work_dir: str = ""
    outputs: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    scalars: dict = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.state in ("excepted", "killed") or self.exit_status not in (None, 0)

    def mark(self) -> str:
        """✓ / ✗ / … for the state, plus the exit status when it is not 0."""
        text = STATE_MARKS.get(self.state, "")
        if self.exit_status not in (None, 0):
            text = f"{text} exit {self.exit_status}".strip()
        return text

    def walk(self, depth: int = 0):
        """``(depth, step)`` for every step below this one, depth first, in order."""
        for s in self.steps:
            yield depth, s
            yield from s.walk(depth + 1)

    def as_dict(self) -> dict:
        out = {"name": self.name}
        for key in ("path", "state", "id", "work_dir"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.exit_status is not None:
            out["exit_status"] = self.exit_status
        if self.runs != 1:
            out["runs"] = self.runs
        if self.scalars:
            out["scalars"] = dict(self.scalars)
        if self.outputs:
            out["outputs"] = [o.as_dict() for o in self.outputs]
        if self.steps:
            out["steps"] = [s.as_dict() for s in self.steps]
        return out


@dataclass
class Analysis:
    """An analysis a source offers for a workflow: ``cells`` are notebook code cells (each one
    self-contained after the ones before it: they ``load()`` the paths they need), ``result`` the
    variable the last cell leaves a :class:`ewald.verbs.Table` in. ``reason`` says why it
    cannot run on this workflow ("" when it can), e.g. a step it needs is missing. Paths and
    other text from the source go into the cells only as string literals."""

    name: str
    title: str = ""
    cells: list = field(default_factory=list)
    result: str = ""
    reason: str = ""

    @property
    def available(self) -> bool:
        return not self.reason and bool(self.cells)

    def as_dict(self) -> dict:
        out = {"name": self.name, "title": self.title or self.name, "available": self.available}
        if self.reason:
            out["reason"] = self.reason
        return out


@dataclass
class Workflow(WorkflowStep):
    """A whole workflow: the top step, and what a list of them shows.

    ``path`` opens its main result (what a double click on a step-less row opens); ``kind`` is
    the template or process label ("vasp_relax", "PwRelaxWorkChain"); ``source`` the name of the
    source it came from (filled in by ewald). A row of :meth:`list` may leave ``steps`` and
    ``outputs`` empty with ``complete`` False: :func:`describe` fills them when it is expanded."""

    label: str = ""
    description: str = ""
    kind: str = ""
    formula: str = ""
    ctime: str = ""
    source: str = ""
    complete: bool = False
    analyses: list = field(default_factory=list)

    @property
    def title(self) -> str:
        """What a list shows for it: kind, formula, id and state."""
        bits = [self.kind or self.name or "workflow", self.formula, f"<{self.id}>" if self.id else "", self.mark()]
        return "  ".join(b for b in bits if b)

    def as_dict(self) -> dict:
        out = super().as_dict()
        for key in ("label", "description", "kind", "formula", "ctime", "source"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.analyses:
            out["analyses"] = [a.as_dict() for a in self.analyses]
        return out

    def analysis(self, name: str) -> Analysis:
        """The analysis called ``name``; LookupError (naming the ones there are) if none."""
        for a in self.analyses:
            if a.name == name:
                return a
        raise LookupError(f"no analysis {name!r} for this workflow; it has: "
                          f"{', '.join(a.name for a in self.analyses) or 'none'}")


@dataclass
class WorkflowPage:
    """One page of a source's list, newest first; ``next`` is the cursor of the next page
    (None: no more). A page can be empty with a ``next`` (a search read a bounded number of
    rows); the side bar follows it."""

    workflows: list = field(default_factory=list)
    next: str | None = None


# ----------------------------------------------------------------------- sources
def sources() -> dict:
    """Installed workflow sources by name (:func:`ewald.plugins.workflow_sources`)."""
    from . import plugins

    return plugins.workflow_sources()


def source_for(path: str):
    """The source plugin whose scheme ``path`` has (``simagent:1234`` -> sim-agent), else None."""
    from . import plugins

    scheme = plugins.path_scheme(path)
    for p in sources().values():
        if scheme and scheme in p.schemes:
            return p
    return None


def _folder_like(path: str) -> bool:
    """A local path, or ``host:/path`` on an SSH host, rather than ``scheme:id`` of a source."""
    import os

    from . import plugins
    from .io import remote

    if plugins.path_scheme(path) is None or os.path.isdir(os.path.expanduser(path)):
        return True
    return remote.is_remote(path) and "/" in path.split(":", 1)[1] and plugins.scheme_reader(path) is None


def describe(path: str, source: str | None = None) -> Workflow:
    """The workflow at ``path`` (a path of an installed source, ``simagent:1234``, or a folder of
    calculations, local or ``host:/path``: :mod:`ewald.workflow_folder`), with its steps.
    ``source`` names the source when ``path`` is a bare id it understands (``"1234"``)."""
    plugin = sources().get(source) if source else source_for(path)
    if plugin is None and not source and _folder_like(path):
        plugin = sources().get("folders")
    if plugin is None:
        known = ", ".join(f"{s}:" for p in sources().values() for s in p.schemes) or "none installed"
        raise LookupError(f"no workflow source for {path!r} (sources: {known})")
    return plugin.describe(path)


def reference(wf: Workflow) -> str:
    """The string :func:`describe` reads ``wf`` back from: a folder workflow's folder (its
    ``path`` is the last step's result), else the source's path or id."""
    if wf.source == "folders" and wf.id:
        return wf.id
    return wf.path or wf.id


# ----------------------------------------------------------------------- search and open
#: what :func:`find` filters a workflow's state to
STATUSES = ("succeeded", "failed", "running", "any")
#: rows one :func:`find` reads at most, so a search never walks a whole database
FIND_ROWS = 4000
_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def status(step: WorkflowStep) -> str:
    """``succeeded`` (finished, exit 0), ``failed`` (an error exit, excepted, killed), ``running``
    (created, waiting, running) or "" when the source records no state (a folder)."""
    if step.failed:
        return "failed"
    if step.state == "finished":
        return "succeeded"
    return "running" if step.state in STATE_MARKS else ""


def elements(formula: str | None) -> set:
    """The element symbols of a formula: ``TiN``, Hill ``N4Ti4``, ``Ti0.5N`` -> {"Ti", "N"}."""
    return {sym for sym, _ in _ELEMENT.findall(formula or "")}


def day(text: str, now):
    """Local midnight starting the day ``text`` names: ``today``, ``yesterday``, ``3d`` (three days
    ago) or ``YYYY-MM-DD``; ``now`` is an aware datetime."""
    from datetime import datetime, time, timedelta

    t = str(text).strip().lower()
    if t == "today":
        d = now.date()
    elif t == "yesterday":
        d = now.date() - timedelta(days=1)
    elif re.fullmatch(r"\d+\s*d(ays?)?(\s*ago)?", t):
        d = now.date() - timedelta(days=int(re.match(r"\d+", t).group()))
    else:
        try:
            d = datetime.fromisoformat(t[:10]).date()
        except ValueError:
            raise ValueError(f"{text!r} is not a day: use 'today', 'yesterday', '3d' (3 days ago) "
                             "or a date such as 2026-10-22") from None
    return datetime.combine(d, time(), tzinfo=now.tzinfo)


def _when(ctime: str):
    from datetime import datetime, timezone

    try:
        value = datetime.fromisoformat(str(ctime).replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def find(source: str, *, formula: str | None = None, elements_: list | None = None, kind: str | None = None,
         status_: str = "any", since: str | None = None, until: str | None = None, search: str | None = None,
         limit: int = 20, now=None) -> dict:
    """Workflows of ``source`` (newest first) that match every filter given, decided here rather
    than by the source's text search: ``formula`` the same elements in any ratio (``TiN`` finds
    Hill ``N4Ti4``, a slab or a defect cell), ``elements_`` all of these among others, ``kind`` a
    substring of the kind (template), ``status_`` one of :data:`STATUSES`, ``since``/``until``
    days (both inclusive, see :func:`day`; the day the workflow was created), ``search`` the
    source's own text search. Pages are read until ``limit`` matches, a row older than ``since``
    or :data:`FIND_ROWS` rows. Returns ``{"today", "workflows", "more"}`` (and ``note`` when
    the search stopped early)."""
    from datetime import datetime, timedelta

    now = now or datetime.now().astimezone()
    if status_ not in STATUSES:
        raise ValueError(f"status is one of {', '.join(STATUSES)}, not {status_!r}")
    exact, wanted = elements(formula), set(elements_ or ())
    if formula and not exact:
        raise ValueError(f"{formula!r} is not a formula; write it like TiN or SrTiO3")
    bad = [e for e in wanted if not re.fullmatch(r"[A-Z][a-z]?", str(e))]
    if bad:
        raise ValueError(f"elements are symbols such as Ti or N, not {bad}")
    start = day(since, now) if since else None
    end = day(until, now) + timedelta(days=1) if until else None
    dated = start is not None or end is not None
    plugin = sources().get(source)
    if plugin is None:
        raise ValueError(f"name a source: one of {sorted(sources())}")
    matches, rows, cursor = [], 0, None
    answer = {"today": now.strftime("%Y-%m-%d (%A)"), "workflows": matches, "more": False}
    while True:
        page = plugin.list(query=search or None, cursor=cursor)
        for wf in page.workflows:
            rows += 1
            created = _when(wf.ctime) if wf.ctime else None
            if start and created and created < start:      # newest first: nothing older matches
                return answer
            if dated and created is None or end and created >= end:
                continue
            have = elements(wf.formula)
            if exact and have != exact or wanted and not wanted <= have:
                continue
            if kind and kind.lower() not in (wf.kind or "").lower():
                continue
            if status_ != "any" and status(wf) != status_:
                continue
            if len(matches) == limit:
                answer["more"] = True
                return answer
            matches.append(wf)
        cursor = page.next
        if not cursor:
            return answer
        if rows >= FIND_ROWS:
            answer["more"] = True
            answer["note"] = (f"stopped after the newest {rows} workflows; give 'since' or a narrower filter "
                              "to search older ones")
            return answer


def target(wf: Workflow, step: str | None = None, output: str | None = None) -> str:
    """The path that opens ``step`` of ``wf`` (its name, or names joined by ``/`` down the tree:
    ``bands/scf``), or its ``output`` (a file or structure listed under that step, or under the
    workflow without a step); the workflow's own path without either. Names that are not
    there raise with the ones that are, so nothing is guessed."""
    node = wf
    if step:
        names = [n for n in str(step).split("/") if n]
        found = [s for _, s in wf.walk() if s.name == names[-1]]
        if len(names) > 1:
            found = [s for s in found if _chain(wf, s)[-len(names):] == names]
        if len(found) != 1:
            every = sorted({"/".join(_chain(wf, s)) for _, s in wf.walk()})
            outs = [o.name for o in wf.outputs]
            why = "is not a step" if not found else "names several steps"
            raise LookupError(f"{step!r} {why} of {wf.kind or wf.name} {wf.id}; steps: {', '.join(every) or 'none'}"
                              + (f"; outputs: {', '.join(outs)}" if outs else ""))
        node = found[0]
    if output:
        hit = [o for o in node.outputs if o.name == output and o.path]
        if not hit:
            names = [o.name for o in node.outputs if o.path]
            where = f"step {step!r}" if step else f"{wf.kind or wf.name} {wf.id}"
            raise LookupError(f"{output!r} is not an output of {where}; outputs: {', '.join(names) or 'none'}"
                              + ("" if step else " (name a step for its files)"))
        return hit[0].path
    if not node.path:
        names = [o.name for o in node.outputs if o.path]
        raise LookupError(f"step {step!r} has no result of its own to open; name an output: "
                          f"{', '.join(names) or 'none'}" if step else f"{wf.id} has no path to open")
    return node.path


def _chain(wf: WorkflowStep, step: WorkflowStep) -> list:
    """The names from ``wf`` down to ``step``."""
    for s in wf.steps:
        if s is step:
            return [s.name]
        below = _chain(s, step)
        if below:
            return [s.name, *below]
    return []


# ----------------------------------------------------------------------- as text
def text(wf: Workflow) -> str:
    """The workflow as an indented tree, one line per step and output, with the paths."""
    lines = [f"{wf.title}" + (f"  {wf.label}" if wf.label and wf.label != wf.kind else "")
             + (f"  -> {wf.path}" if wf.path else "")]

    def outputs(step, pad):
        for o in step.outputs:
            lines.append(f"{pad}- {o.name}" + (f"  ({o.note})" if o.note else "") + (f"  -> {o.path}" if o.path else ""))

    outputs(wf, "  ")
    for depth, s in wf.walk():
        pad = "  " * (depth + 1)
        runs = f"  ({s.runs} runs)" if s.runs > 1 else ""
        lines.append(f"{pad}{s.name}{runs}  {s.mark()}".rstrip() + (f"  -> {s.path}" if s.path else ""))
        outputs(s, pad + "  ")
    return "\n".join(lines)


# ----------------------------------------------------------------------- as a notebook
_TAKEN = {"pipe", "load", "import_file", "show", "view", "paint", "hide", "isolate", "fade", "resize", "np"}


def variable_name(name: str, taken=()) -> str:
    """A Python name for a step called ``name`` ("bands/scf" -> ``bands_scf``, "02_neb" -> ``neb``), not one of
    ``taken`` nor a name the notebook imports."""
    base = re.sub(r"\W+", "_", name.strip().lower()).strip("_") or "step"
    base = re.sub(r"^\d+_+(?=[a-z])", "", base)   # "01_relax" (a folder in step order) -> relax
    if base[0].isdigit():
        base = f"step_{base}"
    if keyword.iskeyword(base):
        base += "_"
    out, n = base, 2
    while out in _TAKEN or out in taken:
        out, n = f"{base}_{n}", n + 1
    return out


def _one_line(value) -> str:
    """``value`` on one line: text from a server (a label, a step name) must not end a markdown
    cell or a cell title and start a code cell of its own."""
    return " ".join(str(value).split())


def _cell(value) -> str:
    """``value`` on one line for a markdown table cell (``|`` escaped)."""
    return _one_line(value).replace("|", "\\|")


#: steps listed in a workflow notebook's heading table at most
MAX_TABLE_STEPS = 40


def _steps_markdown(wf: Workflow) -> list:
    """The lines of the heading's step table (:func:`ewald.workflow_table.workflow_table`), or
    none when the workflow has fewer than two steps with numbers. One line per row, each value
    on one line, so nothing a source writes can leave the markdown cell."""
    from .workflow_table import step_rows, workflow_table

    rows = step_rows(wf)
    if len(rows) < 2 or not any(s.scalars for _, s in rows):
        return []
    lines = workflow_table(wf)._repr_markdown_().splitlines()[:MAX_TABLE_STEPS + 2]
    if len(rows) > MAX_TABLE_STEPS:
        lines.append(f"| … {len(rows) - MAX_TABLE_STEPS} more steps |")
    return [_one_line(x) for x in lines]


def _loads(wf: Workflow) -> list:
    """``(variable, path, note)`` for every step with a result, then every top-level structure,
    trajectory or band output, then the workflow's own path as ``result`` (when it is none of
    those), in the order :func:`notebook_text` loads them."""
    taken: set = set()
    seen: set = set()
    out = []
    for _, step in wf.walk():
        if step.path and step.path not in seen:
            name = variable_name(step.name, taken)
            taken.add(name)
            seen.add(step.path)
            out.append((name, step.path, step.mark() if step.failed else ""))
    for o in wf.outputs:
        if o.path and o.kind in ("structure", "trajectory", "band") and o.path not in seen:
            name = variable_name(o.name, taken)
            taken.add(name)
            seen.add(o.path)
            out.append((name, o.path, o.note))
    if out and wf.path and wf.path not in seen:   # the workflow's own result, so pipe can switch back to it
        out.append((variable_name("result", taken), wf.path, ""))
    return out


def step_variables(wf: Workflow) -> dict:
    """``{path: variable}``: the name :func:`notebook_text` gives each step and output it loads."""
    return {path: name for name, path, _ in _loads(wf)}


def notebook_text(wf: Workflow, pipeline_name: str = "pipe") -> str:
    """One notebook (percent format) for a whole workflow: a heading with what it is, and one
    cell that loads every step with a result and every top-level output structure
    (``name = load(path)``, a line each, named by :func:`step_variables`), and a last cell that
    sets ``pipe`` to the main result, so the 3D view starts there and any step is one
    reassignment (``pipe = relax``) or ``show(scf, ...)`` away. Names, labels and numbers from
    the source only reach comments and markdown, on one line each, and paths only string
    literals, so a workflow cannot put code into the notebook."""
    from .notebook import IMPORTS

    line = _one_line
    head = [f"# # {line(wf.label or wf.kind or wf.name)}", "#"]
    facts = [("Kind", wf.kind), ("Formula", wf.formula), ("Id", wf.id), ("State", wf.mark() or wf.state),
             ("Created", wf.ctime), ("Source", wf.source)]
    head += [f"# {k}: {line(v)}  " for k, v in facts if v]
    if wf.description:
        head += ["#", f"# {line(wf.description)}"]
    numbers = list(wf.scalars.items())[:12]
    if numbers:
        head += ["#", "# | Quantity | Value |", "# |---|---|"] + [f"# | {_cell(k)} | {_cell(v)} |" for k, v in numbers]
    steps = _steps_markdown(wf)
    if steps:
        head += ["#", "# Steps (`workflow_table(...)` gives this table to compare with other workflows):", "#"]
        head += [f"# {row}" for row in steps]
    cells = ["# %% [markdown]\n" + "\n".join(head), "# %%\n" + IMPORTS]
    loads = _loads(wf)
    by_path = {path: name for name, path, _ in loads}
    main = by_path.get(wf.path) if wf.path else None
    if main is None and wf.path:
        main = f"load({str(wf.path)!r})"
    if main is None and by_path:
        main = list(by_path.values())[-1]
    body = [f"{name} = load({str(path)!r})" + (f"  # {line(note)}" if note else "") for name, path, note in loads]
    if body:
        cells.append("# %%\n" + "\n".join(body))
    if main is not None:            # a cell of its own: the side bar's step switch rewrites only it
        cells.append(f"# %%\n{pipeline_name} = {main}")
    for a in wf.analyses:
        cells += analysis_cells(a)
    return "\n\n".join(cells) + "\n"


def analysis_cells(a: Analysis) -> list[str]:
    """Percent-format cells for analysis ``a``: a heading, then its code cells; only the heading,
    with the reason, when it cannot run."""
    title = _one_line(a.title or a.name)
    if not a.available:
        return [f"# %% [markdown]\n# ## {title}\n#\n# Not available: {_one_line(a.reason or 'no cells')}"]
    return [f"# %% [markdown]\n# ## {title}"] + [f"# %%\n{c.strip()}" for c in a.cells]


# ----------------------------------------------------------------------- the side bar
def workflow_panel(source: str, title: str = "", icon: str = "", tooltip: str = "", shortcut: str = "",
                   width: int | None = None):
    """A side bar factory for the workflow source ``source``: its account, a search box, its
    workflows newest first, their steps and outputs. Register it as the plugin's panel::

        sim_agent = workflow_panel("simagent", title="sim-agent", icon=ICON)   # ewald.panels entry point

    A double click on a workflow opens it as one notebook (:func:`notebook_text`), on a step or
    output opens that path; the Qt code is imported when the side bar first opens."""

    def factory(context):
        from .gui.workflow_panel import WorkflowPanel

        return WorkflowPanel(context, source)

    factory.__name__ = f"{source}_workflows"
    factory.__doc__ = f"Workflows of {title or source}: recent ones, their steps and outputs; double-click opens."
    factory.title = title or source
    factory.icon = icon
    factory.tooltip = tooltip or factory.title
    factory.shortcut = shortcut
    factory.width = width
    return factory
