import numpy as np
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
    assert step_call("pipe.add(Slice())") is not None and step_call("other.add(Slice())") is None
    assert step_call("from ewald.modifiers import Slice\npipe.modifiers.append(Slice())") is not None
    for src in ("x = 1\npipe.modifiers.append(Slice())", "pipe.modifiers.append(a, b)", "other.modifiers.append(A())",
                "pipe.modifiers.extend([A()])", "", "def f(:"):
        assert step_call(src) is None, src


def test_run_all_builds_the_pipeline_and_shows_outputs(nb):
    assert nb.run_all()
    pipe = nb.pipeline
    assert [m.name for m in pipe.modifiers] == ["CoordinationAnalysis", "ExpressionSelection"]
    assert nb.cells[2].modifier is pipe.modifiers[0] and nb.cells[3].modifier is pipe.modifiers[1]
    last = nb.cells[-1]
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
    assert src.splitlines()[1] == "pipe.add(LocalThing(k=2))"
    assert step_source(LocalThing(2), {"LocalThing": LocalThing}) == "pipe.add(LocalThing(k=2))"
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


def test_load_plugin_file_registers_its_modifier(tmp_path, cu_file):
    plugin = tmp_path / "shift_plugin.py"
    plugin.write_text(
        "from ewald.plugins import modifier\nfrom ewald.pipeline import Modifier\n\n"
        "@modifier(category='Test')\nclass ShiftZ(Modifier):\n"
        "    def __init__(self, dz: float = 1.0):\n        self.dz = dz\n\n"
        "    def modify(self, data, ctx):\n        out = data.copy()\n"
        "        out.positions = data.positions + [0, 0, self.dz]\n        return out\n")
    nb = Notebook.from_py(f"# %%\nfrom ewald import import_file\nfrom ewald.plugins import load_plugin_file\n"
                          f"names = load_plugin_file({str(plugin)!r})\npipe = import_file({cu_file!r})\n\n"
                          "# %%\nfrom ewald.modifiers import ShiftZ\npipe.modifiers.append(ShiftZ(dz=2.0))\n")
    assert nb.run_all(), nb.cells[0].outputs
    assert nb.namespace["names"] == ["ShiftZ"]
    assert nb.cells[1].modifier.dz == 2.0
    assert step_source(nb.cells[1].modifier) == "from ewald.modifiers import ShiftZ\npipe.add(ShiftZ(dz=2.0))"


def test_from_pipeline_writes_the_load_line(cu_file):
    pipe = import_file(cu_file)
    nb = Notebook.from_pipeline(pipe)
    assert nb.cells[1].source == f"pipe = load({cu_file!r})"


def test_view_cells_plot_tables_and_histograms(nb):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from ewald.views import view_args, view_source

    assert view_args("show(pipe, table='coordination-rdf', y='g(r)')") == \
        {"source": "pipe", "table": "coordination-rdf", "y": "g(r)"}
    for src in ("show(pipe, table=name)", "x = 1\nshow(pipe)", "show(pipe.compute(0))", "plot(pipe)"):
        assert view_args(src) is None, src
    assert view_source("pipe", table="DOS", y=None) == "show(pipe, table='DOS')"
    nb.cells.insert(0, Cell("from ewald.views import show"))
    table = nb.add_view(table="coordination-rdf")
    hist = nb.add_view(histogram="Coordination", bins=12)
    assert table.is_view and hist.view == {"source": "pipe", "histogram": "Coordination", "bins": 12}
    assert not nb.cells[3].is_view and nb.cells[3].is_step
    assert nb.run_all(), [o.data for c in nb.cells for o in c.outputs if o.kind == "error"]
    assert [o.kind for o in table.outputs] == ["image"] and [o.kind for o in hist.outputs] == ["image"]
    nb.update_view(table, y="g(r)")
    assert table.source.splitlines()[-1] == "show(pipe, table='coordination-rdf', y='g(r)')" and nb.dirty
    assert Notebook.from_py(nb.to_py()).cells[-1].view == hist.view


def test_load_add_and_paint(cu_file):
    from ewald import load
    from ewald.paint import fade, hide, isolate, paint, resize

    pipe = load(cu_file)
    coord = pipe.add(mods.CoordinationAnalysis(cutoff=3.0))
    assert pipe.modifiers == [coord]

    @pipe.add
    def style(frame, data):
        paint(data, data["Coordination"] < 12, "red")
        resize(data, "Coordination < 12", 2.0)
        hide(data, [0, 1])
        fade(data, [2], 0.5)

    assert callable(style) and pipe.modifiers[-1].name == "style"
    d = pipe.compute(0)
    surface = d["Coordination"] < 12
    assert surface.sum() == 12 and (d["Color"][surface] == [1, 0, 0]).all() and not (d["Color"][~surface] == [1, 0, 0]).all(1).any()
    assert np.allclose(d["Radius"][surface], 2 * d["Radius"][~surface][0])
    assert list(d["Transparency"][:3]) == [1, 1, 0.5]
    isolate(d, surface)
    assert (d["Transparency"][~surface] == 1).all()


def test_history_undo_redo_follows_the_pipeline(cu_file):
    from ewald.notebook import History

    nb = Notebook.from_py(NOTEBOOK.format(path=cu_file))
    nb.run_all()
    hist = History(nb)
    pipe = nb.pipeline
    assert not hist.can_undo and hist.undo() is None
    coord = nb.cells[2]
    # a step form edit, dragged in three moves: one entry
    for cutoff in (3.1, 3.2, 3.3):
        coord.modifier.cutoff = cutoff
        nb.update_step(coord)
        hist.commit("Edit CoordinationAnalysis", merge=("form", id(coord)))
    nb.delete(nb.cells[3])  # the ExpressionSelection step
    hist.commit("Delete cell")
    assert [m.name for m in pipe.modifiers] == ["CoordinationAnalysis"]
    nb.cells[-1].source += "\nprint('typed, not run')"  # typing counts as one entry on undo
    assert hist.undo_label == "Edit cell"
    assert hist.undo() == "Edit cell" and "typed" not in nb.cells[-1].source
    assert nb.cells[-1].state == "stale"  # plain code is not re-run on undo
    assert hist.undo() == "Delete cell"
    assert [m.name for m in pipe.modifiers] == ["CoordinationAnalysis", "ExpressionSelection"]
    assert nb.cells[3].modifier is pipe.modifiers[1]
    kept = pipe.modifiers[0]
    assert hist.undo() == "Edit CoordinationAnalysis"
    assert pipe.modifiers[0].cutoff == 3.0 and pipe.modifiers[0] is not kept  # the step ran again
    assert int(pipe.compute(0).selection.sum()) == 12
    assert hist.undo() is None and hist.can_redo
    assert hist.redo() == "Edit CoordinationAnalysis" and pipe.modifiers[0].cutoff == 3.3
    assert hist.redo() == "Delete cell" and len(pipe.modifiers) == 1
    nb.add_step(mods.ExpressionSelection("Coordination < 12"))
    assert hist.commit("Add step") and not hist.can_redo  # a new edit drops the redo entries
