# Workflow sources

A workflow is a calculation that ran in steps: an AiiDA WorkGraph or WorkChain and the
calculations it called, a relaxation that was restarted after a walltime, an NEB that relaxed
its end points first. ewald's own source, `folders` (`ewald.workflow_folder`), reads plain
folders of VASP, LAMMPS, NEB and phonopy calculations ([Workflows](../user/workflows.md)); a
plugin that knows where other workflows live (a sim-agent server, an AiiDA profile) describes them with four small dataclasses from `ewald.workflow`, and ewald does
the rest the same way for every source:

- a side bar with the source's account, a search box and its workflows newest first, where
  expanding a workflow lists its steps (nested, restarts joined) and their outputs;
- **open as notebook**: a double click on a workflow opens one notebook whose load cell names
  every step, a line each (`relax = load(...)`, `scf = load(...)`), and a last cell sets `pipe` to the main result, so the 3D view
  starts there and every other step is one `pipe = scf` or `show(scf, ...)` away;
- **step switch and Plot rows**: while that notebook is on screen the side bar marks the step
  `pipe` is (●) and the other loaded ones (○); a click on ○ rewrites the `pipe = ...` cell, and
  the "Plot <table>" rows under a step add `show(step, table=..., panel=True)` view cells;
- the AI tools `list_workflows` and `describe_workflow`, offered when a source is installed.

## The model

| Class | What it holds |
|---|---|
| `Workflow` | the whole workflow: `id`, `path` (its main result), `kind` (template or process label), `formula`, `label`, `state`, `exit_status`, `ctime`, `scalars`, `steps`, `outputs`; `complete=False` for a list row whose steps are not read yet |
| `WorkflowStep` | one step: `name`, `path` (its own result, or ""), `state`, `exit_status`, `runs` (restarts joined), `id`, `work_dir`, `scalars`, `outputs` and nested `steps` |
| `WorkflowOutput` | something a step made: `name`, `path`, `kind` ("structure", "trajectory", "band" for NEB images, "volumetric", "file", "data"), `note` ("Si2", "on the cluster") |
| `WorkflowPage` | one page of a list: `workflows` and `next`, the cursor of the next page (None: the last) |
| `Analysis` | something the plugin can compute from the workflow: `name`, `title`, `cells` (notebook code), `result` (the variable the last cell leaves, usually a `Table`), `reason` (why it cannot run here; `available` is False) |

Every `path` is a string `load()` opens, served by the plugin's own reader for its scheme
(`@reader(..., schemes=("mydb",))`). So whatever the side bar or the AI does with a workflow, a
notebook cell can repeat.

A step's `scalars` are what `workflow_table(...)` shows and compares across workflows
([Workflows](../user/workflows.md)): numbers become columns, and the names ewald's own sources use
(`energy`, `energy/atom`, `barrier`, `gap`, `max force`, `fermi energy`, `atoms`, all in eV and
eV/Å) come first and get units in the heads, so a source that uses them lines up with the folder
source. Words (`code`, `kind`) show only in the table of one workflow.

## Writing a source

```python
from ewald.plugin_api import (Workflow, WorkflowOutput, WorkflowPage, WorkflowStep, workflow_panel,
                              workflow_source)

@workflow_source("mydb", title="My database", schemes=("mydb",), settings_page="mydb")
class MyDatabase:
    def account(self):                        # "" = not set up: the side bar offers its settings page
        return "alice at https://db.example.org"

    def list(self, query=None, cursor=None):  # newest first, one page
        rows = fetch_rows(query, cursor)
        return WorkflowPage([Workflow(r["label"], id=str(r["pk"]), path=f"mydb:{r['pk']}",
                                      kind=r["template"], state=r["state"]) for r in rows["rows"]],
                            next=rows["next"])

    def describe(self, ref):                  # an id the user typed, or a path of this source
        ...
        return Workflow("relax", id="12", path="mydb:12/relax", complete=True, steps=[
            WorkflowStep("relax", path="mydb:12/relax", state="finished", runs=2,
                         outputs=[WorkflowOutput("vasprun.xml", "mydb:12/relax/vasprun.xml")])])

my_database = workflow_panel("mydb", title="My database", icon="mydb.svg")   # its side bar
```

In a package, register the source and the side bar as entry points:

```toml
[project.entry-points."ewald.workflows"]
mydb = "my_plugin.source:MyDatabase"

[project.entry-points."ewald.panels"]
mydb = "my_plugin.source:my_database"
```

`list` may return an empty page with a `next` cursor (a search that reads a bounded number of
rows per page); the side bar follows up to ten of them. Errors raised by `list` or `describe`
are shown in the side bar, so raise `LookupError` or `ValueError` with a message the user can
act on. Requests run off the GUI thread.

Names, labels and numbers from a source reach the notebook only as comments and markdown, one
line each, and paths only as string literals, so a workflow cannot put code into the notebook
it opens (`tests/test_security.py`).

## Analyses

A source may also say what can be computed from a finished workflow, by adding an
`analyses(wf)` method that returns `Analysis` entries. `describe` calls it for complete
workflows; the opened notebook gets each analysis as a markdown heading and its cells, and the
AI tool `add_analysis` adds and runs them. An analysis is code the user can read and change, not a
number the plugin hands over, so it is written against EWALD's own kernels (for example
`ewald.potential` for planar-averaged potentials, band edges and lineups) and reads files through
the paths the steps already carry.

```python
from ewald.plugin_api import Analysis

def analyses(self, wf):
    if wf.kind != "band_offset":
        return []
    steps = {s.name: s for _, s in wf.walk()}
    if "interface" not in steps:                      # say why, never guess another step
        return [Analysis("lineup", "Potential lineup", reason="no interface step")]
    fetch = f"run = load({steps['interface'].path!r})\nframe = run.compute(run.num_frames - 1)"
    return [Analysis("lineup", "Potential lineup", [fetch, "lineup = ..."], result="lineup")]
```

Recipes are keyed by the workflow's `kind` (its template), and name the steps they read by the
template's own step names. When a step is missing or failed, return the analysis with a `reason`
rather than nothing, so the user and the AI see why it is not there. A recipe that raises is
shown as unavailable with the error. Values from the source go into the cells only as literals
(`repr` of strings and numbers), as for the generated `load` cells, and the cells themselves
come from the plugin's code, which is trusted because it is installed. When `result` names a
`Table`, `add_analysis` returns its `energy_axis`, `result` and `levels` attributes to the AI, so
it can quote the numbers instead of computing them.
