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


def test_clear_outputs_drops_them_from_the_saved_file(nb, tmp_path):
    assert nb.run_all()
    last = nb.cells[-1]
    nb.clear_outputs(last)
    assert last.outputs == [] and last.count is None and nb.cells[-2].count is not None
    nb.clear_outputs()
    assert all(not c.outputs and c.count is None for c in nb.cells) and nb.pipeline is not None
    path = tmp_path / "cleared.ipynb"
    nb.save(str(path))
    assert all(not c.outputs for c in Notebook.load(str(path)).cells)


def test_set_kind_turns_code_into_text_and_back_like_jupyter(nb):
    assert nb.run_all()
    cell = nb.cells[2]
    mod = cell.added[0]
    assert mod in nb.pipeline.modifiers
    assert nb.set_kind(cell, "markdown") and nb.dirty
    assert cell.kind == "markdown" and "CoordinationAnalysis" in cell.source
    assert mod not in nb.pipeline.modifiers and not cell.added and cell.count is None
    assert "# %% [markdown]" in nb.to_py()
    assert not nb.set_kind(cell, "markdown")               # already text: nothing to do
    assert nb.set_kind(cell, "code") and cell.state == "new"
    nb.run_cell(2)
    assert any(isinstance(m, mods.CoordinationAnalysis) for m in nb.pipeline.modifiers)
    with pytest.raises(ValueError):
        nb.set_kind(cell, "raw")


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


def test_show_of_the_pipeline_is_marked_for_a_live_plot(tmp_path):
    pytest.importorskip("matplotlib")
    from ewald import examples
    from ewald.notebook_io import from_ipynb, to_ipynb

    nb = Notebook.from_py(f"# %%\nimport matplotlib\nmatplotlib.use('Agg')\nfrom ewald import load\n"
                          f"from ewald.modifiers import *\nfrom ewald.views import show\n"
                          f"pipe = load({str(examples.path('Cu_md.extxyz'))!r})\n"
                          "pipe.add(MeanSquaredDisplacement())\n\n"
                          "# %%\nprint('anywhere in a cell')\nshow(pipe, table='msd', xlim=(0, 2))\n"
                          "show(pipe.compute(0), table='msd')\n")
    assert nb.run_all()
    text, live, other = nb.cells[1].outputs
    assert text.kind == "text" and live.kind == other.kind == "image"
    assert live.view == {"table": "msd", "bins": 50, "xlim": [0.0, 2.0]}
    assert other.view is None                       # not the notebook's pipeline: an image
    again = from_ipynb(to_ipynb(nb))                # .ipynb keeps it (Jupyter shows the PNG)
    assert again.cells[1].outputs[1].view == live.view and again.cells[1].outputs[1].data == live.data


def test_plots_go_to_the_plots_panel_when_they_change_with_the_frame_or_are_linked():
    from ewald import examples, load
    from ewald.data import DataCollection
    from ewald.modifiers import CoordinationAnalysis
    from ewald.views import in_panel

    pipe = load(examples.path("POSCAR.SrTiO3"))
    pipe.add(CoordinationAnalysis(cutoff=3.0))
    data = pipe.compute(0)
    assert pipe.num_frames == 1
    assert in_panel(data, {"table": "coordination-rdf"}, 1)       # g(r): a range selects atoms
    assert in_panel(data, {"histogram": "Coordination"}, 1)        # histogram: linked to the atoms
    static = DataCollection()
    static.tables["diffraction"] = {"2theta": np.linspace(10, 90, 5), "Intensity": np.ones(5)}
    assert not in_panel(static, {"table": "diffraction"}, 1)       # one frame, nothing to link: inline
    assert not in_panel(static, {}, 1)                             # (the first table)
    assert in_panel(static, {"table": "diffraction"}, 10)          # changes with the frame
    neb = DataCollection()
    neb.tables["NEB"] = {"Reaction Coordinate": np.arange(3.0), "Energy": np.zeros(3), "Frame": np.arange(3)}
    assert in_panel(neb, {"table": "NEB"}, 1)                      # a click goes to the image's frame
    assert not in_panel(data, {"table": "coordination-rdf", "panel": False}, 10)   # panel= decides
    assert in_panel(static, {"table": "diffraction", "panel": True}, 1)
    assert not in_panel(None, {"table": "diffraction"}, 10) and in_panel(None, {"panel": True})


def test_set_keyword_rewrites_one_show_call_in_place():
    from ewald.views import set_keyword

    src = "x = 1\nshow(pipe, table='rdf')  # g(r)\nshow(pipe, histogram='Q', panel=False)\n"
    assert set_keyword(src, "panel", True) == \
        "x = 1\nshow(pipe, table='rdf', panel=True)  # g(r)\nshow(pipe, histogram='Q', panel=False)\n"
    assert set_keyword(src, "panel", True, index=1).endswith("show(pipe, histogram='Q', panel=True)\n")
    assert set_keyword(src, "panel", None, index=1).endswith("show(pipe, histogram='Q')\n")
    assert set_keyword("show(pipe, panel=True, table='é')", "panel", None) == "show(pipe, table='é')"
    assert set_keyword("show(pipe)", "panel", False) == "show(pipe, panel=False)"
    assert set_keyword("show(pipe)", "panel", None) == "show(pipe)"
    assert set_keyword(src, "panel", True, index=2) is None and set_keyword("show(", "panel", True) is None
    assert set_keyword("print(pipe)", "panel", True) is None


def test_show_panel_argument_is_kept_for_the_gui(tmp_path):
    pytest.importorskip("matplotlib")
    from ewald import examples

    nb = Notebook.from_py(f"# %%\nimport matplotlib\nmatplotlib.use('Agg')\nfrom ewald import load\n"
                          f"from ewald.modifiers import *\nfrom ewald.views import show\n"
                          f"pipe = load({str(examples.path('Cu_md.extxyz'))!r})\n"
                          "pipe.add(MeanSquaredDisplacement())\n\n"
                          "# %%\nshow(pipe, table='msd', panel=False)\n")
    assert nb.run_all()
    (live,) = nb.cells[1].outputs
    assert live.view == {"table": "msd", "bins": 50, "panel": False}
    assert nb.cells[1].view == {"source": "pipe", "table": "msd", "panel": False}


def test_figures_of_an_each_frame_cell_do_not_land_under_another_cell():
    pytest.importorskip("matplotlib")
    from ewald import examples

    nb = Notebook.from_py(f"# %%\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
                          f"from ewald import load\npipe = load({str(examples.path('Cu_md.extxyz'))!r})\n\n"
                          "# %% z histogram [each frame]\nplt.hist(data['Position'][:, 2])\n\n"
                          "# %%\nn = pipe.compute(2).count\n")
    assert nb.run_all()
    assert nb.cells[2].outputs == []                           # the step's figure stays with the step
    assert nb.pipeline.compute(2).attributes["Figure: z_histogram"].png.startswith(b"\x89PNG")


def test_each_frame_and_all_frames_cells_are_steps(tmp_path):
    from ewald import examples
    from ewald.notebook_io import from_ipynb, to_ipynb

    text = (f"# %%\nimport numpy as np\nfrom ewald import load\npipe = load({str(examples.path('Cu_md.extxyz'))!r})\n"
            "height = 'mine'\n\n"
            "# %% height [each frame]\ndata.attributes['z'] = data['Position'][:, 2].mean()\n\n"
            "# %% [all frames]\nz = [d['Position'][:, 2].mean() for d in frames]\n"
            "data.tables['height'] = {'Frame': np.arange(len(z)), 'Mean z': np.array(z)}\n\n"
            "# %% [each frame]\ndata.attributes['boom'] = 1 / 0\n")
    nb = Notebook.from_py(text)
    assert [(c.title, c.frames) for c in nb.cells] == [("", ""), ("height", "each"), ("", "all"), ("", "each")]
    assert nb.to_py() == text                                    # the tags stay in the file
    assert nb.run_all()                                          # running a cell only adds its step
    pipe = nb.pipeline
    assert [m.name for m in pipe.modifiers] == ["height", "all_frames_cell", "each_frame_cell"]
    assert nb.namespace["height"] == "mine" and "_ewald_step" not in nb.namespace   # nothing left behind
    with pytest.raises(ZeroDivisionError) as err:
        pipe.compute(0)
    assert "line 1, in each_frame_cell" in "".join(__import__("traceback").format_tb(err.tb))
    assert not nb.set_frames(nb.cells[3], "")                    # back to once: its step goes
    assert nb.cells[3].state == "stale" and [m.name for m in pipe.modifiers] == ["height", "all_frames_cell"]
    d = pipe.compute(4)
    assert d.attributes["z"] == pytest.approx(d["Position"][:, 2].mean()) and len(d.tables["height"]["Frame"]) == 10
    assert nb.run_cell(nb.cells[1], run_below=False) and len(pipe.modifiers) == 2   # again: replaced
    assert nb.set_frames(nb.cells[3], "all") and nb.cells[3].frames == "all"
    again = from_ipynb(to_ipynb(nb))                              # .ipynb keeps it in the cell metadata
    assert [c.frames for c in again.cells] == ["", "each", "all", "all"]
    snap = nb.snapshot()
    nb.cells[1].frames = ""
    nb.restore(snap)                                             # undo brings the tag back and runs the cell
    assert nb.cells[1].frames == "each" and nb.cells[1].state == "ok"
    bare = Notebook.from_py("# %% [each frame]\ndata\n")
    assert not bare.run_all() and "load a file first" in bare.cells[0].outputs[0].data


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


def test_py_header_keeps_kernel_view_state_and_data_links(cu_file, tmp_path):
    import shutil

    data = tmp_path / "cu.xyz"
    shutil.copy(cu_file, data)
    nb = Notebook.from_py(f"# %%\nfrom ewald import load\npipe = load({str(data)!r})\n")
    assert nb.run_all()
    nb.state = {"frame": 1, "camera": {"position": [1.0, 2.0, 3.0]}, "render": {"quality": "Draft"}}
    nb.metadata["ewald"] = {"kernel": {"name": None, "python": "/env/bin/python", "display_name": "Python (env)"}}
    nb.save(tmp_path / "a.py")
    text = (tmp_path / "a.py").read_text()
    assert text.startswith("# ---\n# jupyter:\n#   ewald:\n#     kernel: {") and '"path": "cu.xyz"' in text
    back = Notebook.load(tmp_path / "a.py")
    assert back.state == nb.state and back.metadata["ewald"]["kernel"]["python"] == "/env/bin/python"
    assert back.links[0]["path"] == "cu.xyz" and back.warnings == [] and len(back.cells) == 1
    back.save(tmp_path / "b.py")                                  # not run: the links stay
    assert (tmp_path / "b.py").read_text() == text
    data.write_text(data.read_text() + "\n")
    assert Notebook.load(tmp_path / "a.py").warnings == ["cu.xyz changed since the notebook was saved"]
    assert nb.to_py().startswith("# %%")                          # the shown code has no header
    import runpy

    assert runpy.run_path(str(tmp_path / "a.py"))["pipe"].num_frames == 1   # still a plain script


def test_py_header_keeps_jupytext_lines():
    text = ("# ---\n# jupyter:\n#   jupytext:\n#     formats: ipynb,py:percent\n#   kernelspec:\n"
            "#     name: python3\n# ---\n\n# %%\nx = 1\n")
    nb = Notebook.from_py(text)
    assert [c.source for c in nb.cells] == ["x = 1"] and nb.to_py(header=True) == text
    nb.state = {"frame": 2}
    again = Notebook.from_py(nb.to_py(header=True))
    assert again.state == {"frame": 2} and again.to_py(header=True) == nb.to_py(header=True)
    assert Notebook.from_py("# %%\nx = 1\n").to_py(header=True) == "# %%\nx = 1\n"


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


def test_interrupt_stops_a_cell_running_on_another_thread():
    """Notebook.interrupt raises CellInterrupted in the thread running a cell; afterwards a pending
    interrupt never leaks into later code, and interrupting with nothing running is a no-op."""
    import threading
    import time

    nb = Notebook.from_py("# %%\nimport time\nwhile True:\n    time.sleep(0.01)\n\n# %%\nafter = 1\n")
    assert not nb.interrupt()
    result = []
    t = threading.Thread(target=lambda: result.append(nb.run_all()))
    t.start()
    for _ in range(500):
        if nb.__dict__.get("_exec_thread"):
            break
        time.sleep(0.01)
    assert nb.interrupt()
    t.join(10)
    assert not t.is_alive() and result == [False]
    assert nb.cells[0].state == "error" and "CellInterrupted" in nb.cells[0].outputs[-1].data
    assert "after" not in nb.namespace and not nb.interrupt()
    assert nb.run_cell(1) and nb.namespace["after"] == 1


# ---------------------------------------------------------------- IPython cells (ewald.shell)
def test_cells_run_in_ipython(capsys):
    """Cells are IPython cells: magics, display(), rich reprs, _ / Out[n], ';' and IPython's
    tracebacks, without touching __main__, the builtins or sys.stdout of this process."""
    import builtins
    import sys

    main, stdout, hook = sys.modules["__main__"], sys.stdout, sys.displayhook

    nb = Notebook([Cell("%time x = 6 * 7\nx"), Cell("display({'a': 1})\nx;"), Cell("_ + 1"),
                   Cell("Out[1] * 2"),
                   Cell("class Table:\n    def _repr_html_(self):\n        return '<table><tr><td>1</td></tr></table>'\nTable()"),
                   Cell("def f():\n    return 1 / 0\nf()")])
    nb.run_all(stop_on_error=False)
    time, display_, last, out1, table, err = nb.cells
    assert "Wall time" in time.outputs[0].data and time.outputs[-1].data == "42"
    assert [o.data for o in display_.outputs] == ["{'a': 1}"]           # ';' hides the value
    assert last.outputs[-1].data == "43" and out1.outputs[-1].data == "84"
    assert table.outputs[0].kind == "rich" and "<table>" in table.outputs[0].mime["text/html"]
    assert err.state == "error" and "ZeroDivisionError" in err.outputs[0].data and "<cell 6>" in err.outputs[0].data
    assert "return 1 / 0" in err.outputs[0].data                          # the failing line is shown
    assert sys.modules["__main__"] is main and sys.stdout is stdout and sys.displayhook is hook
    assert not hasattr(builtins, "get_ipython") and "x" not in vars(main)
    assert capsys.readouterr().out == ""                                 # nothing leaked to the terminal


def test_update_display_replaces_its_output():
    nb = Notebook([Cell("h = display('one', display_id=True)\nprint('between')\nh.update('two')")])
    assert nb.run_all()
    assert [(o.kind, o.data) for o in nb.cells[0].outputs] == [("value", "'two'"), ("text", "between\n")]


def test_output_streams_like_a_terminal():
    """Carriage returns redraw the line (tqdm and other text progress bars), CRLF from Windows
    programs is a newline, and a line split over two writes stays one line."""
    from ewald.notebook import _CellSink

    cell = Cell("")
    sink = _CellSink(Notebook(), cell)
    for chunk in ("a\r\n", "10%\r", "\r50%", "\r100%\n", "half ", "line\r", "\nnext"):
        sink.stream("stdout", chunk)
    assert [o.kind for o in cell.outputs] == ["text"]
    assert cell.outputs[0].data == "a\n100%\nhalf line\nnext"
    sink.stream("stderr", "\rab")
    assert cell.outputs[0].data.endswith("\nabxt")                      # overwritten in place, as a terminal does


def test_two_notebooks_print_on_two_threads_at_once():
    """Two tabs (or a tab and the console) running at the same time each get their own output."""
    import threading

    code = "import time\nfor i in range(20):\n    print('{0}', i)\n    time.sleep(0.005)\n'{0}'"
    a, b = Notebook([Cell(code.format("A"))]), Notebook([Cell(code.format("B"))])
    threads = [threading.Thread(target=nb.run_all) for nb in (a, b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    for nb, name in ((a, "A"), (b, "B")):
        text, value = nb.cells[0].outputs
        assert text.data.split() == [x for i in range(20) for x in (name, str(i))]
        assert value.data == repr(name)


def test_live_outputs_reach_the_listener_while_the_cell_runs():
    import ewald.notebook as nbmod

    seen = []
    nb = Notebook([Cell("import time\nfor i in range(3):\n    print(i)\n    time.sleep(0.15)")])
    nb.on_output = lambda cell: seen.append(cell.outputs[-1].data if cell.outputs else "")
    assert nb.run_all()
    assert seen[0].startswith("0") and seen[-1] == "0\n1\n2\n" and len(seen) >= 3
    assert nbmod.OUTPUT_THROTTLE <= 0.2


def test_restart_forgets_ipython_history():
    nb = Notebook([Cell("y = 5\ny")])
    nb.run_all()
    assert nb.namespace["Out"][1] == 5
    nb.reset()
    assert "y" not in nb.namespace and nb.shell.user_ns is nb.namespace and not nb.namespace["Out"]


def test_percent_py_round_trips_through_jupytext(tmp_path):
    """Our percent .py files are jupytext's: titles, Markdown and the [each frame] tag survive both ways."""
    jupytext = pytest.importorskip("jupytext")
    nb = Notebook.from_py("# %% [markdown]\n# # Title\n#\n# text\n\n# %% setup\nx = 1\n\n"
                          "# %% speed [each frame]\ndata.attributes['v'] = frame\n")
    doc = jupytext.reads(nb.to_py(), fmt="py:percent")
    assert [c.cell_type for c in doc.cells] == ["markdown", "code", "code"]
    assert [c.source for c in doc.cells] == ["# Title\n\ntext", "x = 1", "data.attributes['v'] = frame"]
    back = Notebook.from_py(jupytext.writes(doc, fmt="py:percent"))
    assert [(c.kind, c.title, c.frames, c.source) for c in back.cells] == \
        [(c.kind, c.title, c.frames, c.source) for c in nb.cells]
