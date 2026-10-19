import pytest

from ewald import import_file
from ewald import modifiers as mods
from ewald.notebook import Cell, Notebook, new_notebook, step_call, step_source

NOTEBOOK = '''# %% [markdown]
# # Cu test
#
# two steps

# %%
from ewald import import_file
from ewald.modifiers import *
pipe = import_file({path!r})

# %% coordination
pipe.modifiers.append(CoordinationAnalysis(cutoff=3.0))

# %%
pipe.modifiers.append(ExpressionSelection("Coordination < 12"))

# %%
data = pipe.compute(0)
print("atoms", data.count)
int(data.selection.sum())
'''


@pytest.fixture
def cu_file(tmp_path):
    from ase.build import bulk
    from ase.io import write

    atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((3, 3, 3))
    del atoms[[0]]  # a vacancy: its 12 neighbours have CN 11
    path = tmp_path / "cu.extxyz"
    write(path, atoms)
    return str(path)


@pytest.fixture
def nb(cu_file):
    return Notebook.from_py(NOTEBOOK.format(path=cu_file))


def test_parse_and_write_percent_format(nb):
    assert [c.kind for c in nb.cells] == ["markdown", "code", "code", "code", "code"]
    assert nb.cells[0].source == "# Cu test\n\ntwo steps"
    assert nb.cells[2].title == "coordination"
    assert [c.is_step for c in nb.cells] == [False, False, True, True, False]
    again = Notebook.from_py(nb.to_py())
    assert [(c.kind, c.title, c.source) for c in again.cells] == [(c.kind, c.title, c.source) for c in nb.cells]
    assert nb.to_py().startswith("# %% [markdown]\n# # Cu test\n#\n# two steps\n\n# %%\n")


def test_plain_script_is_one_cell():
    nb = Notebook.from_py("x = 1\nprint(x)\n")
    assert len(nb.cells) == 1 and nb.cells[0].source == "x = 1\nprint(x)"


def test_step_detection():
    assert step_call("pipe.modifiers.append(Slice())") is not None
    assert step_call("from ewald.modifiers import Slice\npipe.modifiers.append(Slice())") is not None
    for src in ("x = 1\npipe.modifiers.append(Slice())", "pipe.modifiers.append(a, b)", "other.modifiers.append(A())",
                "pipe.modifiers.extend([A()])", "", "def f(:"):
        assert step_call(src) is None, src


def test_run_all_builds_the_pipeline_and_shows_outputs(nb):
    assert nb.run_all()
    pipe = nb.pipeline
    assert [m.name for m in pipe.modifiers] == ["CoordinationAnalysis", "ExpressionSelection"]
    assert nb.cells[2].modifier is pipe.modifiers[0] and nb.cells[3].modifier is pipe.modifiers[1]
    last = nb.cells[4]
    assert [o.kind for o in last.outputs] == ["text", "value"]
    assert last.outputs[0].data == "atoms 107\n"
    assert last.outputs[1].data == "12"
    assert [c.count for c in nb.cells if c.kind == "code"] == [1, 2, 3, 4]
    assert all(c.state == "ok" for c in nb.cells[1:])


def test_rerunning_a_cell_replaces_its_modifier_in_place(nb):
    nb.run_all()
    first = nb.pipeline.modifiers[0]
    nb.run_cell(2)
    nb.run_cell(2)
    assert len(nb.pipeline.modifiers) == 2
    assert nb.pipeline.modifiers[0] is not first and nb.pipeline.modifiers[0] is nb.cells[2].modifier
    nb.cells[2].source = "pipe.modifiers.append(CoordinationAnalysis(cutoff=2.0))"
    nb.run_cell(2)
    assert [m.name for m in nb.pipeline.modifiers] == ["CoordinationAnalysis", "ExpressionSelection"]
    assert nb.pipeline.modifiers[0].cutoff == 2.0


def test_errors_stop_run_all_and_mark_later_cells_stale(nb):
    nb.cells[2].source = "pipe.modifiers.append(CoordinationAnalysis(cutof=3.0))"
    assert not nb.run_all()
    err = nb.cells[2].outputs[-1]
    assert err.kind == "error" and "<cell 3>" in err.data and "cutof" in err.data
    assert nb.cells[2].state == "error"
    assert [c.state for c in nb.cells[3:]] == ["stale", "stale"]


def test_rebinding_the_pipeline_reruns_the_cells_below(nb):
    nb.run_all()
    old = nb.pipeline
    nb.run_cell(1)
    assert nb.pipeline is not old
    assert [m.name for m in nb.pipeline.modifiers] == ["CoordinationAnalysis", "ExpressionSelection"]


def test_add_step_update_move_and_delete(nb):
    nb.run_all()
    pipe = nb.pipeline
    cell = nb.add_step(mods.Slice((0, 0, 1), 2.0), index=3)
    assert cell.is_step and [m.name for m in pipe.modifiers] == ["CoordinationAnalysis", "Slice", "ExpressionSelection"]
    cell.modifier.distance = 3.5
    nb.update_step(cell)
    assert "distance=3.5" in cell.source and step_call(cell.source) is not None
    nb.move(cell, 2)
    assert [m.name for m in pipe.modifiers] == ["Slice", "CoordinationAnalysis", "ExpressionSelection"]
    nb.delete(cell)
    assert [m.name for m in pipe.modifiers] == ["CoordinationAnalysis", "ExpressionSelection"]
    assert nb.dirty


def test_sync_follows_changes_made_outside_the_cells(nb):
    nb.run_all()
    pipe = nb.pipeline
    pipe.modifiers.insert(1, mods.Slice((0, 0, 1), 2.0))     # console / Claude
    assert nb.sync()
    assert [c.modifier.name for c in nb.cells if c.modifier] == ["CoordinationAnalysis", "Slice", "ExpressionSelection"]
    assert nb.cells[3].modifier is pipe.modifiers[1]
    del pipe.modifiers[0]
    pipe.modifiers[0].distance = 1.0
    assert nb.sync()
    assert [c.modifier.name for c in nb.cells if c.modifier] == ["Slice", "ExpressionSelection"]
    assert "distance=1.0" in nb.cells[2].source
    assert not nb.sync()


def test_step_source_imports_classes_outside_the_public_list():
    class LocalThing(mods.Modifier):
        def __init__(self, k: int = 1):
            self.k = k

        def modify(self, data, ctx):
            return None

    src = step_source(LocalThing(2))
    assert src.splitlines()[0] == f"from {__name__} import LocalThing"
    assert src.splitlines()[1] == "pipe.modifiers.append(LocalThing(k=2))"
    assert step_source(LocalThing(2), {"LocalThing": LocalThing}) == "pipe.modifiers.append(LocalThing(k=2))"
    m = mods.WrapAtoms()
    m.enabled = False
    assert step_source(m).endswith("pipe.modifiers[-1].enabled = False")


def test_matplotlib_figures_and_images_become_image_outputs():
    pytest.importorskip("matplotlib")
    nb = Notebook.from_py("# %%\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
                          "plt.plot([0, 1], [1, 0])\nplt.show()\n\n# %%\nimport numpy as np\n"
                          "np.zeros((4, 5, 3), np.uint8)\n")
    assert nb.run_all()
    (fig,) = nb.cells[0].outputs
    assert fig.kind == "image" and fig.data.startswith(b"\x89PNG")
    assert nb.cells[1].outputs[0].kind == "image"
    import matplotlib.pyplot as plt

    assert plt.get_fignums() == []


def test_from_pipeline_and_save_load_roundtrip(cu_file, tmp_path):
    pipe = import_file(cu_file)
    pipe.modifiers.append(mods.CoordinationAnalysis(cutoff=3.0))
    nb = Notebook.from_pipeline(pipe, f"pipe = import_file({cu_file!r})", name="cu")
    assert nb.pipeline is pipe and nb.cells[-1].modifier is pipe.modifiers[0]
    path = tmp_path / "cu.py"
    nb.save(path)
    again = Notebook.load(path)
    assert again.name == "cu"
    assert again.run_all()
    assert [m.name for m in again.pipeline.modifiers] == ["CoordinationAnalysis"]
    # the file is also a plain script
    import runpy

    ns = runpy.run_path(str(path))
    assert ns["pipe"].compute(0).count == 107


def test_new_notebook_has_imports():
    nb = new_notebook("Notebook 2")
    assert nb.cells[0].kind == "markdown" and "Notebook 2" in nb.cells[0].source
    assert nb.run_all() and "CoordinationAnalysis" in nb.namespace and nb.pipeline is None


def test_modifier_output_summary(cu_file):
    pipe = import_file(cu_file)
    pipe.modifiers.append(mods.CoordinationAnalysis(cutoff=3.0))
    pipe.modifiers.append(mods.ExpressionSelection("Coordination < 12"))
    pipe.modifiers.append(mods.DeleteSelected())
    pipe.compute(0)
    coord, sel, delete = (m.last_output for m in pipe.modifiers)
    assert coord["properties"] == ["Coordination"] and "coordination-rdf" in coord["tables"]
    assert coord["count_before"] == coord["count"] == 107
    assert sel["properties"] == ["Selection"] and sel["selected"] == 12
    assert (delete["count_before"], delete["count"]) == (107, 95) and delete["properties"] == []


def test_cell_defaults():
    c = Cell("x = 1")
    assert c.kind == "code" and c.state == "new" and c.modifier is None and not c.is_step
