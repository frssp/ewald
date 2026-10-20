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
        return out


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


def notebook_text(wf: Workflow, pipeline_name: str = "pipe") -> str:
    """One notebook (percent format) for a whole workflow: a heading with what it is, and one
    cell that loads every step with a result and every top-level output structure
    (``name = load(path)``, a line each) and sets ``pipe`` to the main result, so the 3D view starts there and any step is one
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
        head += ["#", "# | Quantity | Value |", "# |---|---|"] + [f"# | {line(k)} | {line(v)} |" for k, v in numbers]
    cells = ["# %% [markdown]\n" + "\n".join(head), "# %%\n" + IMPORTS]
    taken: set = set()
    by_path: dict = {}
    loads = []
    for _, step in wf.walk():
        if step.path and step.path not in by_path:
            name = variable_name(step.name, taken)
            taken.add(name)
            by_path[step.path] = name
            loads.append((f"{name} = load({str(step.path)!r})", line(step.mark()) if step.failed else ""))
    for o in wf.outputs:
        if o.path and o.kind in ("structure", "trajectory", "band") and o.path not in by_path:
            name = variable_name(o.name, taken)
            taken.add(name)
            by_path[o.path] = name
            loads.append((f"{name} = load({str(o.path)!r})", line(o.note)))
    main = by_path.get(wf.path) if wf.path else None
    if main is None and wf.path:
        main = f"load({str(wf.path)!r})"
    if main is None and by_path:
        main = list(by_path.values())[-1]
    body = [f"{code}  # {note}" if note else code for code, note in loads]
    if main is not None:
        body.append(f"{pipeline_name} = {main}")
    if body:
        cells.append("# %%\n" + "\n".join(body))
    return "\n\n".join(cells) + "\n"


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
