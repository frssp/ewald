"""The numbers of a workflow's steps as one table, and several workflows side by side
(ewald.workflow_table), plus the folder source's step numbers it shows."""
import ast
import csv
import shutil

import numpy as np
import pytest

from ewald import workflow, workflow_table
from ewald.notebook import Notebook
from ewald.workflow import Workflow, WorkflowStep
from ewald.workflow_folder import _band_gap, _max_force, describe_folder
from ewald.workflow_table import step_rows


def study_a():
    return Workflow("li", label="PBE", steps=[
        WorkflowStep("01_relax", path="a/relax/OUTCAR", state="finished",
                     scalars={"code": "vasp", "kind": "relax", "energy": -16.27512, "atoms": 3, "max force": 0.012}),
        WorkflowStep("bands", steps=[
            WorkflowStep("scf", path="a/bands/scf/vasprun.xml", state="finished", scalars={"energy": -16.3, "gap": 1.1}),
            WorkflowStep("nscf", path="a/bands/nscf/vasprun.xml", state="excepted", scalars={"gap": 1.05})]),
        WorkflowStep("md", path="a/md/run*/XDATCAR", runs=2, state="running", scalars={"kind": "md"},
                     steps=[WorkflowStep("run1"), WorkflowStep("run2")]),
    ])


def study_b():
    return Workflow("li", label="r2SCAN", steps=[
        WorkflowStep("1-Relax", path="b/relax/OUTCAR", state="finished", scalars={"energy": -18.0, "atoms": 3}),
        WorkflowStep("bands", steps=[WorkflowStep("scf", path="b/scf", state="finished",
                                                   scalars={"energy": -18.1, "gap": 1.6})]),
        WorkflowStep("neb", path="b/neb", state="finished", scalars={"barrier": 0.31}),
    ])


def test_rows_are_calculations_groups_name_their_steps_and_restarts_are_one_row():
    rows = step_rows(study_a())
    assert [n for n, _ in rows] == ["01_relax", "bands / scf", "bands / nscf", "md"]
    assert step_rows(Workflow("single", scalars={"energy": 1.0}))[0][0] == "single"


def test_one_workflow_table():
    t = workflow_table(study_a())
    assert t.labels == ["PBE"] and len(t) == 4
    assert t.columns == ["energy", "gap", "max force", "atoms"] and t.words == ["code", "kind"]
    assert t.cell("bands / scf", "gap") == 1.1 and t.cell("md", "energy") is None
    assert t.state("bands / nscf") == "✗" and t.state("01_relax") == "✓"
    assert t.path("md") == "a/md/run*/XDATCAR"
    e = t["energy"]["PBE"]
    assert e[0] == pytest.approx(-16.27512) and np.isnan(e[3])
    assert list(t["kind"]["PBE"]) == ["relax", "", "", "md"]
    text = repr(t)
    assert text.splitlines()[0].split()[:5] == ["step", "state", "code", "kind", "energy"]
    assert "-16.2751" in text and text.splitlines()[1].endswith("  3")   # 4 decimals, counts as integers
    with pytest.raises(KeyError, match="energy"):
        t["nothing"]
    picked = workflow_table(study_a(), columns=["gap", "kind"])
    assert picked.columns == ["gap"] and picked.words == ["kind"]


def test_several_workflows_side_by_side_matched_by_name():
    t = workflow_table(study_a(), study_b())
    assert t.labels == ["PBE", "r2SCAN"]
    # "01_relax" and "1-Relax" are one step; neb only in the second; nscf, md only in the first
    assert t.rows == ["01_relax", "bands / scf", "bands / nscf", "md", "neb"]
    assert t.cell("01_relax", "energy", "r2SCAN") == -18.0 and t.cell("neb", "barrier", "PBE") is None
    assert t.words == []                                   # words only in a one-workflow table
    assert t.columns[:3] == ["energy", "barrier", "gap"]
    gap = t["gap"]
    assert gap["PBE"][1] == 1.1 and gap["r2SCAN"][1] == 1.6 and np.isnan(gap["r2SCAN"][2])
    heads = repr(t).splitlines()[0]
    assert "state PBE" in heads and "energy (eV) r2SCAN" in heads
    by_position = workflow_table(study_a(), study_b(), align="position")
    assert by_position.rows == ["01_relax", "bands / scf", "bands / nscf", "md"]
    assert by_position.cell("bands / nscf", "barrier", "r2SCAN") == 0.31
    d = t.difference("PBE")
    assert d.labels == ["r2SCAN − PBE"] and d.title == "Difference to PBE"
    assert d.cell("01_relax", "energy") == pytest.approx(-18.0 + 16.27512)
    assert d.cell("01_relax", "atoms") == 0 and isinstance(d.cell("01_relax", "atoms"), int)
    assert d.cell("neb", "barrier") is None and repr(d).startswith("Difference to PBE")
    assert workflow_table([study_a(), study_b()], labels=["x", "y"]).labels == ["x", "y"]
    same = workflow_table(study_a(), study_a())
    assert same.labels == ["PBE", "PBE (2)"]
    with pytest.raises(ValueError):
        workflow_table(study_a(), study_b(), labels=["x", "x"])
    with pytest.raises(ValueError):
        workflow_table()
    with pytest.raises(ValueError):
        workflow_table(study_a(), align="size")
    with pytest.raises(ValueError):
        workflow_table(study_a()).difference()


def test_outputs_html_markdown_csv_and_plot(tmp_path):
    bad = Workflow("x", label="<b>evil</b>", steps=[
        WorkflowStep("a|b <script>", state="finished", scalars={"energy": -1.0, "note": "x\ny"}),
        WorkflowStep("c", scalars={"energy": -2.0})])
    t = workflow_table(bad, columns=["energy", "note"])
    page = t._repr_html_()
    assert "<script>" not in page and "&lt;script&gt;" in page and "<table" in page
    md = t._repr_markdown_()
    assert "a\\|b" in md and len(md.splitlines()) == 4     # heads, rule, two rows: the newline is gone
    both = workflow_table(study_a(), study_b())._repr_html_()
    assert "colspan='2'" in both and "<th>r2SCAN</th>" in both
    out = tmp_path / "t.csv"
    workflow_table(study_a(), study_b()).to_csv(out)
    raw = out.read_bytes()
    assert b"\r\n" not in raw
    rows = list(csv.reader(raw.decode("utf-8").splitlines()))
    assert rows[0][:3] == ["step", "state PBE", "state r2SCAN"] and rows[1][0] == "01_relax"
    fig = workflow_table(study_a(), study_b()).figure("energy")
    assert len(fig.axes[0].lines) == 2
    with pytest.raises(KeyError):
        workflow_table(study_a()).figure("kind")
    assert workflow_table(study_a()).as_dict()["values"]["PBE"]["bands / scf"]["gap"] == 1.1


def test_workflow_notebook_heading_has_the_step_table_and_stays_markdown():
    wf = study_a()
    wf.steps[0].name = "relax\n# %%\nimport os"
    text = workflow.notebook_text(wf)
    nb = Notebook.from_py(text)
    head = nb.cells[0]
    assert head.kind == "markdown" and "| step | state |" in head.source and "workflow_table" in head.source
    assert "| relax # %% import os |" in head.source
    for cell in nb.cells:                                  # the name made no cell of its own
        if cell.kind == "code":
            assert "import os" not in cell.source.replace("import numpy", "")
    assert "| step |" not in workflow.notebook_text(Workflow("x", path="demo:1"))   # no steps: no table


def test_reference_reads_a_workflow_back():
    assert workflow.reference(Workflow("a", id="C:\\calc\\li", path="C:\\calc\\li\\04\\OUTCAR",
                                       source="folders")) == "C:\\calc\\li"
    assert workflow.reference(Workflow("a", id="12", path="aiida:12", source="aiida")) == "aiida:12"


# ----------------------------------------------------------------------- the folder source's numbers
EIGENVAL = """    2    2    1    1
  0.1E+02  0.1E+02  0.1E+02  0.1E-08  0.1E-08
  1.0E-004
  CAR
 Si
     8     2     3

  0.0  0.0  0.0  0.5
    1   -5.0000   1.0000
    2    1.0000   1.0000
    3    2.5000   0.0000

  0.5  0.0  0.0  0.5
    1   -4.0000   1.0000
    2    0.8000   1.0000
    3    2.0000   0.0000
"""

EIGENVAL_SPIN = """    2    2    1    2
  0.1E+02  0.1E+02  0.1E+02  0.1E-08  0.1E-08
  1.0E-004
  CAR
 Fe
     8     1     2

  0.0  0.0  0.0  1.0
    1   -1.0000   -0.5000   1.0000   1.0000
    2    0.3000    0.1000   0.0000   0.0000
"""


def test_band_gap_from_eigenval_lf_and_crlf(tmp_path):
    lf = tmp_path / "EIGENVAL"
    lf.write_text(EIGENVAL, encoding="utf-8", newline="\n")
    crlf = tmp_path / "EIGENVAL.crlf"
    crlf.write_text(EIGENVAL, encoding="utf-8", newline="\r\n")
    assert _band_gap(str(lf)) == pytest.approx(1.0) and _band_gap(str(crlf)) == pytest.approx(1.0)
    spin = tmp_path / "EIGENVAL.spin"
    spin.write_text(EIGENVAL_SPIN, encoding="utf-8", newline="\n")
    assert _band_gap(str(spin)) == pytest.approx(0.6)
    no_occ = tmp_path / "EIGENVAL.old"
    no_occ.write_text("\n".join(line.rsplit(None, 1)[0] if line.startswith("    ") else line
                                for line in EIGENVAL.splitlines()), encoding="utf-8", newline="\n")
    assert _band_gap(str(no_occ)) is None
    assert _band_gap(str(tmp_path / "missing")) is None


def test_max_force_from_the_last_block(tmp_path, vasp_dir):
    assert _max_force(str(vasp_dir / "OUTCAR"), 3) == pytest.approx(0.01)
    assert _max_force(str(vasp_dir / "OUTCAR"), 4) is None       # not every atom's row: not trusted
    crlf = tmp_path / "OUTCAR"
    crlf.write_bytes((vasp_dir / "OUTCAR").read_bytes().replace(b"\n", b"\r\n"))
    assert _max_force(str(crlf), 3) == pytest.approx(0.01)


def test_folder_steps_get_numbers_and_compare(tmp_path, vasp_dir):
    for name, extra in (("pbe", None), ("hse", EIGENVAL)):
        relax = tmp_path / name / "01_relax"
        relax.mkdir(parents=True)
        for f in ("OUTCAR", "CONTCAR"):
            shutil.copy(vasp_dir / f, relax / f)
        shutil.copy(vasp_dir / "md" / "OSZICAR", relax / "OSZICAR")
        static = tmp_path / name / "02_static"
        shutil.copytree(relax, static)
        if extra:
            (static / "EIGENVAL").write_text(extra, encoding="utf-8", newline="\n")
    wf = describe_folder(tmp_path / "hse")
    assert wf.steps[1].scalars["gap"] == pytest.approx(1.0) and wf.steps[0].scalars["atoms"] == 3
    t = workflow_table(str(tmp_path / "pbe"), str(tmp_path / "hse"))   # paths, read by describe
    assert t.labels == ["pbe", "hse"] and t.rows == ["01_relax", "02_static"]
    assert t.cell("02_static", "gap", "hse") == pytest.approx(1.0) and t.cell("02_static", "gap", "pbe") is None
    assert t.difference("pbe").cell("01_relax", "energy/atom") == pytest.approx(0.0)


def test_ai_reference_mentions_workflow_table():
    from ewald.llm.reference import GUIDE

    assert "workflow_table(" in GUIDE
    tree = ast.parse("from ewald import workflow_table")
    assert tree.body
