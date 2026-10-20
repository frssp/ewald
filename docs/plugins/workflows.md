# Workflow sources

A workflow is a calculation that ran in steps: an AiiDA WorkGraph or WorkChain and the
calculations it called, a relaxation that was restarted after a walltime, an NEB that relaxed
its end points first. A plugin that knows where such workflows live (a sim-agent server, an
AiiDA profile) describes them with four small dataclasses from `ewald.workflow`, and ewald does
the rest the same way for every source:

- a side bar with the source's account, a search box and its workflows newest first, where
  expanding a workflow lists its steps (nested, restarts joined) and their outputs;
- **open as notebook**: a double click on a workflow opens one notebook with a `load` cell per
  step (`relax = load(...)`, `scf = load(...)`) and `pipe` on the main result, so the 3D view
  starts there and every other step is one `pipe = scf` or `show(scf, ...)` away;
- the AI tools `list_workflows` and `describe_workflow`, offered when a source is installed.

## The model

| Class | What it holds |
|---|---|
| `Workflow` | the whole workflow: `id`, `path` (its main result), `kind` (template or process label), `formula`, `label`, `state`, `exit_status`, `ctime`, `scalars`, `steps`, `outputs`; `complete=False` for a list row whose steps are not read yet |
| `WorkflowStep` | one step: `name`, `path` (its own result, or ""), `state`, `exit_status`, `runs` (restarts joined), `id`, `work_dir`, `scalars`, `outputs` and nested `steps` |
| `WorkflowOutput` | something a step made: `name`, `path`, `kind` ("structure", "trajectory", "band" for NEB images, "volumetric", "file", "data"), `note` ("Si2", "on the cluster") |
| `WorkflowPage` | one page of a list: `workflows` and `next`, the cursor of the next page (None: the last) |

Every `path` is a string `load()` opens, served by the plugin's own reader for its scheme
(`@reader(..., schemes=("mydb",))`). So whatever the side bar or the AI does with a workflow, a
notebook cell can repeat.

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
