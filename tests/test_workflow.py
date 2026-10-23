"""Workflows (ewald.workflow): the model, sources, the notebook a workflow opens as, AI tools."""
import pytest
from ase.build import bulk

from ewald import plugins, workflow
from ewald.data import DataCollection
from ewald.notebook import Notebook
from ewald.workflow import Workflow, WorkflowOutput, WorkflowPage, WorkflowStep


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    monkeypatch.setattr(plugins._registry, "loaded", True)   # no installed plugins here


def demo_workflow(complete=True):
    steps = [WorkflowStep("relax", path="demo:7/relax", state="finished", runs=2, id="8",
                          outputs=[WorkflowOutput("vasprun.xml", "demo:7/relax/vasprun.xml"),
                                   WorkflowOutput("CHGCAR", "demo:7/relax/CHGCAR", note="on the cluster")]),
             WorkflowStep("bands", steps=[WorkflowStep("scf", path="demo:7/bands/scf", state="excepted",
                                                       exit_status=400)])]
    return Workflow("relax", id="7", path="demo:7/relax", kind="vasp_relax", formula="Si2", state="finished",
                    label="Si relax", scalars={"energy": -10.8}, complete=complete,
                    steps=steps if complete else [],
                    outputs=[WorkflowOutput("output_structure", "demo:7/output_structure", kind="structure")]
                    if complete else [])


class Demo:
    """Demo calculations."""

    schemes = ("demo",)
    settings_page = "demo"

    last = None

    def __init__(self):
        self.asked = []
        Demo.last = self

    def account(self):
        return "me at demo"

    def list(self, query=None, cursor=None):
        self.asked.append((query, cursor))
        if cursor is None:
            return WorkflowPage([], next="1")           # an empty page with a next one
        return WorkflowPage([demo_workflow(complete=False)], next=None)

    def describe(self, ref):
        if "7" not in ref:
            raise LookupError(f"no workflow {ref}")
        return demo_workflow()


def si(path, **kwargs):
    return [DataCollection.from_ase(bulk("Si"))]


def test_model_walks_marks_and_serializes():
    wf = demo_workflow()
    assert [(d, s.name) for d, s in wf.walk()] == [(0, "relax"), (0, "bands"), (1, "scf")]
    scf = wf.steps[1].steps[0]
    assert scf.failed and scf.mark() == "✗ exit 400" and not wf.steps[0].failed
    assert wf.title == "vasp_relax  Si2  <7>  ✓"
    d = wf.as_dict()
    assert d["steps"][0]["runs"] == 2 and d["steps"][1]["steps"][0]["exit_status"] == 400
    assert d["outputs"] == [{"name": "output_structure", "path": "demo:7/output_structure", "kind": "structure"}]
    text = workflow.text(wf)
    assert "relax  (2 runs)  ✓  -> demo:7/relax" in text and "    scf  ✗ exit 400" in text
    assert "CHGCAR  (on the cluster)  -> demo:7/relax/CHGCAR" in text


def test_source_registry_and_describe_by_scheme():
    p = plugins.workflow_source("demo", title="Demo")(Demo)
    src = plugins.workflow_sources()["demo"]
    assert (src.title, src.schemes, src.settings_page, src.description) == ("Demo", ("demo",), "demo", "Demo calculations.")
    assert p is Demo
    wf = workflow.describe("demo:7")
    assert wf.source == "demo" and wf.complete
    assert workflow.source_for("demo:1") is src and workflow.source_for("aiida:1") is None
    with pytest.raises(LookupError, match="no workflow source for 'aiida:1' \\(sources: demo:\\)"):
        workflow.describe("aiida:1")
    assert workflow.describe("7", source="demo").id == "7"
    assert src.list().workflows == [] and src.list(cursor="1").workflows[0].source == "demo"
    with pytest.raises(TypeError, match="no describe\\(\\) method"):
        plugins.register_workflow_source("bad", type("Bad", (), {"list": lambda self, **k: None}))
    assert {w["name"]: w["schemes"] for w in plugins.status()["workflows"]}["demo"] == ["demo"]


def test_notebook_has_a_variable_per_step_and_pipe_on_the_main_result():
    plugins.register_reader("demo", si, schemes=("demo",))
    text = workflow.notebook_text(demo_workflow())
    nb = Notebook.from_py(text, name="Si relax")
    sources = [c.source for c in nb.cells]
    assert sources[0].startswith("# Si relax") and "| energy | -10.8 |" in sources[0]
    assert len(nb.cells) == 4             # heading, imports, a cell that loads everything, pipe = ...
    assert sources[2].splitlines() == ["relax = load('demo:7/relax')",       # a failed step says so
                                       "scf = load('demo:7/bands/scf')  # ✗ exit 400",
                                       "output_structure = load('demo:7/output_structure')"]
    assert sources[3] == "pipe = relax"   # its own cell: the side bar's step switch rewrites only it
    assert nb.run_all()
    assert nb.pipeline is nb.namespace["relax"] and nb.namespace["scf"].num_frames == 1
    assert workflow.step_variables(demo_workflow()) == {
        "demo:7/relax": "relax", "demo:7/bands/scf": "scf", "demo:7/output_structure": "output_structure"}


def test_variable_names():
    assert workflow.variable_name("bands/scf") == "bands_scf"
    assert workflow.variable_name("1st") == "step_1st"
    assert workflow.variable_name("class") == "class_"
    assert workflow.variable_name("pipe") == "pipe_2"
    assert workflow.variable_name("relax", taken={"relax"}) == "relax_2"
    assert workflow.variable_name("") == "step"


def test_notebook_without_step_results_loads_the_main_path():
    text = workflow.notebook_text(Workflow("x", id="3", path="demo:3"))
    assert text.rstrip().endswith("pipe = load('demo:3')")


def test_workflow_panel_factory_carries_its_side_bar_attributes():
    f = workflow.workflow_panel("demo", title="Demo", icon="<svg/>")
    p = plugins.register_panel("demo", f)
    assert (p.title, p.icon, p.tooltip, p.width) == ("Demo", "<svg/>", "Demo", 320)
    assert "Workflows of Demo" in p.description


def test_ai_tools_list_and_describe_workflows():
    from ewald.llm.tools import Toolset

    ts = Toolset()
    assert ts.tools["describe_workflow"].offered()      # the built-in folders source
    plugins.workflow_source("demo")(Demo)
    assert ts.tools["list_workflows"].offered()
    rows = ts.call("list_workflows", {"search": "Si"}).content
    assert rows["source"] == "demo" and [w["id"] for w in rows["workflows"]] == ["7"]   # past the empty page
    assert Demo.last.asked == [("Si", None), ("Si", "1")]
    out = ts.call("describe_workflow", {"path": "demo:7"}).content
    assert out["steps"][0]["path"] == "demo:7/relax" and out["source"] == "demo"
    assert ts.call("describe_workflow", {"path": "demo:9"}).is_error


class Analysing(Demo):
    """Demo calculations with an analysis for relaxations."""

    def analyses(self, wf):
        if wf.kind != "vasp_relax":
            return []
        steps = {s.name for _, s in wf.walk()}
        cells = ["relaxed = load('demo:7/relax')",
                 ("from ewald.verbs import Table\n"
                  "summary = Table('Summary', {'frame': [0], 'atoms': [relaxed.compute(0).count]}, "
                  "attributes={'result': {'atoms': relaxed.compute(0).count}})")]
        return [workflow.Analysis("summary", "Atom count", cells, result="summary"),
                workflow.Analysis("phonons", "Phonons", reason="no phonon step" if "phonons" not in steps else "")]


def test_a_source_offers_analyses_that_become_notebook_cells():
    plugins.register_reader("demo", si, schemes=("demo",))
    plugins.workflow_source("demo")(Analysing)
    wf = workflow.describe("demo:7")
    assert [a.as_dict() for a in wf.analyses] == [
        {"name": "summary", "title": "Atom count", "available": True},
        {"name": "phonons", "title": "Phonons", "available": False, "reason": "no phonon step"}]
    assert wf.as_dict()["analyses"][1]["reason"] == "no phonon step"
    nb = Notebook.from_py(workflow.notebook_text(wf), name="Si relax")
    sources = [c.source for c in nb.cells]
    assert sources[4].startswith("## Atom count") and sources[5] == "relaxed = load('demo:7/relax')"
    assert "Not available: no phonon step" in sources[-1]
    assert nb.run_all()
    assert nb.namespace["summary"].attributes["result"] == {"atoms": 2}
    with pytest.raises(LookupError, match="it has: summary, phonons"):
        wf.analysis("bands")


def test_a_broken_recipe_does_not_hide_the_workflow():
    class Broken(Demo):
        def analyses(self, wf):
            raise RuntimeError("recipe bug")

    plugins.workflow_source("demo")(Broken)
    wf = workflow.describe("demo:7")
    assert wf.steps and not wf.analyses[0].available and "recipe bug" in wf.analyses[0].reason


def test_ai_adds_an_analysis_through_cells_and_quotes_its_result():
    from ewald.llm.tools import Toolset

    plugins.register_reader("demo", si, schemes=("demo",))
    plugins.workflow_source("demo")(Analysing)
    ts = Toolset()
    out = ts.call("add_analysis", {"path": "demo:7", "name": "summary"}).content
    assert out["result"] == {"atoms": 2} and out["table"] == "Summary" and out["cells"] == 2
    assert ts.notebook.cells[-1].source.startswith("from ewald.verbs import Table")
    err = ts.call("add_analysis", {"path": "demo:7", "name": "phonons"})
    assert err.is_error and "no phonon step" in err.content


def test_add_analysis_reads_only_where_the_ai_may(tmp_path):
    from ewald.llm.access import ReadAccess
    from ewald.llm.tools import Toolset

    (tmp_path / "allowed").mkdir()
    ts = Toolset()
    ts.read_access = ReadAccess([tmp_path / "allowed"])
    err = ts.call("add_analysis", {"path": str(tmp_path / "other"), "name": "summary"})
    assert err.is_error and "outside the folders" in err.content

# ------------------------------------------------------------------ search and open
from datetime import datetime, timedelta, timezone  # noqa: E402

KST = timezone(timedelta(hours=9))
NOW = datetime(2026, 10, 23, 10, 0, tzinfo=KST)


class Calcs:
    """TiN, TiO2 and Si workflows, newest first, two per page."""

    schemes = ("calcs",)

    def __init__(self):
        rows = [("6", "2026-10-22T03:00:00+00:00", "O4Ti2", "vasp_relax", "finished", 0),
                ("5", "2026-10-22T02:00:00+00:00", "N4Ti4", "vasp_relax", "finished", 508),
                ("4", "2026-10-22T01:00:00+00:00", "N31Ti32", "vasp_bands", "finished", 0),
                ("3", "2026-10-21T03:00:00+00:00", "N4Ti4", "vasp_relax", "finished", 0),
                ("2", "2026-10-20T03:00:00+00:00", "Si2", "vasp_bands", "running", None),
                ("1", "2026-10-01T03:00:00+00:00", "N4Ti4", "vasp_relax", "excepted", None)]
        self.rows = [Workflow(k, id=i, path=f"calcs:{i}", kind=k, formula=f, state=st, exit_status=ex, ctime=c)
                     for i, c, f, k, st, ex in rows]
        self.pages = 0

    def account(self):
        return "me"

    def list(self, query=None, cursor=None):
        self.pages += 1
        start = int(cursor or 0)
        return WorkflowPage(self.rows[start:start + 2], next=str(start + 2) if start + 2 < len(self.rows) else None)

    def describe(self, ref):
        return demo_workflow()


def _ids(found):
    return [w.id for w in found["workflows"]]


def test_find_filters_in_code():
    src = Calcs()
    plugins.register_workflow_source("calcs", src)
    find = workflow.find
    assert _ids(find("calcs", formula="TiN", now=NOW)) == ["5", "4", "3", "1"]     # any ratio, Hill order
    assert _ids(find("calcs", elements_=["Ti", "O"], now=NOW)) == ["6"]
    assert _ids(find("calcs", kind="bands", now=NOW)) == ["4", "2"]
    assert _ids(find("calcs", formula="TiN", status_="failed", now=NOW)) == ["5", "1"]
    assert _ids(find("calcs", status_="running", now=NOW)) == ["2"]
    # 4 was created 2026-10-22 10:00 KST, 3 on 2026-10-21 12:00 KST
    src.pages = 0
    found = find("calcs", formula="TiN", status_="succeeded", since="yesterday", until="yesterday", now=NOW)
    assert _ids(found) == ["4"] and found["today"] == "2026-10-23 (Friday)"
    assert src.pages == 2            # newest first: the first row older than 'since' ends the search
    more = find("calcs", formula="TiN", limit=2, now=NOW)
    assert _ids(more) == ["5", "4"] and more["more"]


@pytest.mark.parametrize("kwargs, message", [({"since": "last week"}, "not a day"), ({"status_": "done"}, "status is"),
                                             ({"formula": "tin"}, "not a formula"),
                                             ({"elements_": ["titanium"]}, "symbols")])
def test_find_says_what_to_write(kwargs, message):
    plugins.register_workflow_source("calcs", Calcs)
    with pytest.raises(ValueError, match=message):
        workflow.find("calcs", now=NOW, **kwargs)


def test_days_and_status():
    assert workflow.day("3d", NOW) == datetime(2026, 10, 20, tzinfo=KST)
    assert workflow.day("2 days ago", NOW) == datetime(2026, 10, 21, tzinfo=KST)
    assert workflow.day("2026-10-01", NOW) == datetime(2026, 10, 1, tzinfo=KST)
    assert workflow.elements("Ti0.5N") == {"Ti", "N"}
    wf = demo_workflow()
    assert [workflow.status(s) for s in (wf, wf.steps[1].steps[0], WorkflowStep("x", state="waiting"),
                                         WorkflowStep("folder"))] == ["succeeded", "failed", "running", ""]


def test_target_names_steps_and_outputs_or_lists_them():
    wf = demo_workflow()
    assert workflow.target(wf) == "demo:7/relax"
    assert workflow.target(wf, "scf") == workflow.target(wf, "bands/scf") == "demo:7/bands/scf"
    assert workflow.target(wf, "relax", "CHGCAR") == "demo:7/relax/CHGCAR"
    assert workflow.target(wf, output="output_structure") == "demo:7/output_structure"
    with pytest.raises(LookupError, match="'band' is not a step .* steps: bands, bands/scf, relax"):
        workflow.target(wf, "band")
    with pytest.raises(LookupError, match="'LOCPOT' is not an output of step 'relax'; outputs: vasprun.xml, CHGCAR"):
        workflow.target(wf, "relax", "LOCPOT")
    with pytest.raises(LookupError, match="no result of its own"):
        workflow.target(wf, "bands")


def test_ai_tools_find_and_open_a_workflow():
    from ewald.llm.tools import Toolset

    plugins.register_reader("demo", si, schemes=("demo",))
    plugins.register_workflow_source("calcs", Calcs)
    plugins.workflow_source("demo")(Demo)
    ts = Toolset()
    rows = ts.call("list_workflows", {"source": "calcs", "formula": "TiN", "status": "failed"}).content
    assert [(w["id"], w["status"]) for w in rows["workflows"]] == [("5", "failed"), ("1", "failed")]
    assert "today" in rows
    out = ts.call("open_workflow", {"path": "demo:7", "step": "scf"})
    assert not out.is_error, out.content
    assert out.content["path"] == "demo:7/bands/scf" and out.content["steps"] == ["relax", "bands", "bands/scf"]
    assert out.content["loaded"]["num_frames"] == 1
    assert "pipe = load('demo:7/bands/scf')" in ts.to_script()
    bad = ts.call("open_workflow", {"path": "demo:7", "step": "band"})
    assert bad.is_error and "steps: bands, bands/scf, relax" in bad.content
