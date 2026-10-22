"""Folders of calculations as workflows (ewald.workflow_folder): the built-in "folders" source."""
import shutil

import pytest

from ewald import import_file, plugins, workflow
from ewald.notebook import Notebook
from ewald.workflow_folder import (FolderSource, add_recent_folder, describe_folder, is_workflow_folder,
                                   recent_folders)


@pytest.fixture
def study(tmp_path, vasp_dir, lammps_dir):
    """relax -> VASP NEB -> MD restarted twice -> LAMMPS NEB -> phonopy, as folders in step order."""
    root = tmp_path / "study"
    relax = root / "01_relax"
    relax.mkdir(parents=True)
    for f in ("OUTCAR", "CONTCAR"):
        shutil.copy(vasp_dir / f, relax / f)
    shutil.copy(vasp_dir / "md" / "OSZICAR", relax / "OSZICAR")
    (relax / "INCAR").write_text("IBRION = 2\nNSW = 50\n")
    shutil.copytree(vasp_dir / "neb", root / "02_neb", ignore=shutil.ignore_patterns("*.py"))
    for run in ("run1", "run2"):
        shutil.copytree(vasp_dir / "md", root / "03_md" / run)
    shutil.copytree(lammps_dir / "neb", root / "04_lmp_neb")
    phonon = root / "10_phonon"
    (phonon / "disp-001").mkdir(parents=True)
    (phonon / "disp-002").mkdir()
    (phonon / "phonopy_disp.yaml").write_text("phonopy:\n  version: 2.20\n")
    (phonon / "FORCE_SETS").write_text("1\n")
    (root / ".git").mkdir()
    (root / "notes").mkdir()                      # nothing in it: not a step
    return root


def test_folder_of_calculations_is_one_workflow(study):
    wf = describe_folder(study)
    assert [s.name for s in wf.steps] == ["01_relax", "02_neb", "03_md", "04_lmp_neb", "10_phonon"]
    relax, neb, md, lmp, phonon = wf.steps
    assert relax.path == str(study / "01_relax" / "OUTCAR") and relax.state == "finished"
    assert relax.scalars == {"code": "vasp", "kind": "relax", "energy": -16.275, "ionic steps": 2}
    assert relax.outputs[0].name == "CONTCAR" and relax.outputs[0].kind == "structure"
    assert neb.path == str(study / "02_neb") and neb.scalars["kind"] == "neb" and neb.scalars["images"] == 5
    assert neb.outputs[0].kind == "band"
    # an MD restarted in run1, run2 is one step: a wildcard joins the segments
    assert md.runs == 2 and [s.name for s in md.steps] == ["run1", "run2"]
    assert md.path == str(study / "03_md" / "run*" / "XDATCAR") and md.scalars["kind"] == "md"
    assert lmp.scalars["code"] == "lammps" and lmp.scalars["barrier"] == pytest.approx(1.83045, abs=1e-4)
    assert phonon.scalars["displacements"] == 2 and phonon.path == str(study / "10_phonon" / "phonopy_disp.yaml")
    assert wf.kind == "relax → neb → md → neb → phonon" and wf.path == phonon.path
    assert wf.id == str(study) and wf.complete and wf.label == "study"
    text = workflow.text(wf)
    assert "03_md  (2 runs)  ✓" in text and "notes" not in text and ".git" not in text


def test_a_workflow_folder_opens_as_a_notebook_that_runs(study):
    shutil.rmtree(study / "10_phonon")            # its reader is the phonopy plugin
    nb = Notebook.from_py(workflow.notebook_text(workflow.describe(str(study))))
    loads = nb.cells[-2].source.splitlines()
    assert f"relax = load({str(study / '01_relax' / 'OUTCAR')!r})" in loads
    assert f"md = load({str(study / '03_md' / 'run*' / 'XDATCAR')!r})" in loads
    assert nb.cells[-1].source == "pipe = lmp_neb"
    nb.trust()
    assert nb.run_all()
    ns = nb.namespace
    assert ns["md"].num_frames == ns["run1"].num_frames + ns["run2"].num_frames
    assert ns["neb"].num_frames == 5 and nb.pipeline is ns["lmp_neb"]


def test_one_calculation_and_restarts_alone(tmp_path, vasp_dir):
    one = describe_folder(vasp_dir / "md")
    assert [s.name for s in one.steps] == ["md"] and one.path.endswith("XDATCAR")
    runs = tmp_path / "md"
    for i in range(1, 12):                        # run10 and run11 after run9
        shutil.copytree(vasp_dir / "md", runs / f"run{i}")
    wf = describe_folder(runs)
    assert len(wf.steps) == 1 and wf.steps[0].runs == 11
    pipe = import_file(wf.steps[0].path)
    assert [p.split("/")[-2] for p in pipe.source.paths] == [f"run{i}" for i in range(1, 12)]
    assert is_workflow_folder(runs) and not is_workflow_folder(vasp_dir / "md")
    assert not is_workflow_folder(vasp_dir / "neb")        # one NEB opens as its band


def test_folder_without_calculations(tmp_path):
    (tmp_path / "empty" / "sub").mkdir(parents=True)
    with pytest.raises(LookupError, match="no VASP, LAMMPS, NEB or phonopy calculation"):
        describe_folder(tmp_path / "empty")
    with pytest.raises(LookupError, match="not a folder"):
        describe_folder(tmp_path / "missing")
    assert not is_workflow_folder(tmp_path / "empty")


def test_folders_source_is_built_in_and_lists_recent_folders(study, tmp_path):
    src = plugins.workflow_sources()["folders"]
    assert src.origin == "built-in" and src.account() and "workflows" in plugins.panels()
    assert src.list().workflows == []
    add_recent_folder(study)
    add_recent_folder(tmp_path / "gone")          # deleted since: not listed
    assert recent_folders()[0] == str(tmp_path / "gone")
    rows = src.list().workflows
    assert [w.path for w in rows] == [str(study)] and rows[0].source == "folders" and not rows[0].complete
    assert src.list(query="nothing").workflows == []
    assert src.list(query=str(study)).workflows[0].complete     # a folder typed in the search box
    assert src.describe(str(study)).steps[0].name == "01_relax"
    # a plain path goes to the folders source; scheme:id paths still need their source
    assert workflow.describe(str(study)).source == "folders"
    with pytest.raises(LookupError, match="no workflow source for 'aiida:1'"):
        workflow.describe("aiida:1")


def test_ai_tools_read_folders(study):
    from ewald.llm.tools import Toolset

    ts = Toolset()
    add_recent_folder(study)
    rows = ts.call("list_workflows", {}).content
    assert rows["source"] == "folders" and rows["workflows"][0]["path"] == str(study)
    out = ts.call("describe_workflow", {"path": str(study)}).content
    assert out["steps"][2]["runs"] == 2 and out["source"] == "folders"


def test_remote_workflow_folder(server, tmp_path, vasp_dir):
    data = tmp_path / "remote_fs" / "data" / "calc"
    shutil.copytree(vasp_dir / "neb", data / "neb", ignore=shutil.ignore_patterns("*.py"))
    for run in ("run1", "run2"):
        shutil.copytree(vasp_dir / "md", data / "md" / run)
    wf = workflow.describe("hpc:/data/calc")
    assert [s.name for s in wf.steps] == ["md", "neb"]
    assert wf.steps[0].runs == 2 and wf.steps[0].path == "hpc:/data/calc/md/run*/XDATCAR"
    assert wf.steps[1].path == "hpc:/data/calc/neb"
    assert import_file(wf.steps[0].path).num_frames == 4


def test_source_class_has_no_scheme():
    assert FolderSource.schemes == () and FolderSource.browse_folders


def test_folder_names_cannot_put_code_into_the_notebook(tmp_path, vasp_dir):
    """A folder name is text from the disk (or an SSH host): it reaches the notebook only as a
    cell title or comment on one line, and paths only as string literals."""
    evil = "x\n# %%\nimport os; os.system('touch PWNED')\n"
    root = tmp_path / "s"
    shutil.copytree(vasp_dir / "md", root / evil)
    shutil.copytree(vasp_dir / "md", root / "b")
    nb = Notebook.from_py(workflow.notebook_text(describe_folder(root)))
    assert not any(c.source.startswith("import os") for c in nb.cells)
    assert all("os.system" not in line or "load(" in line or line.startswith("#")
               for c in nb.cells for line in c.source.splitlines())
