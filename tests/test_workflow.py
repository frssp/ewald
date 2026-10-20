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

    def __init__(self):
        self.asked = []

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
    assert plugins.status()["workflows"][0]["schemes"] == ["demo"]


def test_notebook_has_a_variable_per_step_and_pipe_on_the_main_result():
    plugins.register_reader("demo", si, schemes=("demo",))
    text = workflow.notebook_text(demo_workflow())
    nb = Notebook.from_py(text, name="Si relax")
    sources = [c.source for c in nb.cells]
    assert sources[0].startswith("# Si relax") and "| energy | -10.8 |" in sources[0]
    assert len(nb.cells) == 3                       # heading, imports, and one cell that loads everything
    assert sources[2].splitlines() == ["relax = load('demo:7/relax')",       # a failed step says so
                                       "scf = load('demo:7/bands/scf')  # ✗ exit 400",
                                       "output_structure = load('demo:7/output_structure')",
                                       "pipe = relax"]
    assert nb.run_all()
    assert nb.pipeline is nb.namespace["relax"] and nb.namespace["scf"].num_frames == 1


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
    assert not ts.tools["describe_workflow"].offered()
    plugins.workflow_source("demo")(Demo)
    assert ts.tools["list_workflows"].offered()
    rows = ts.call("list_workflows", {"search": "Si"}).content
    assert rows["source"] == "demo" and rows["workflows"] == []      # one page, as the source gave it
    out = ts.call("describe_workflow", {"path": "demo:7"}).content
    assert out["steps"][0]["path"] == "demo:7/relax" and out["source"] == "demo"
    assert ts.call("describe_workflow", {"path": "demo:9"}).is_error
