import sys

import numpy as np
import pytest

pytest.importorskip("jupyter_client")
pytest.importorskip("ipykernel")

from ewald import kernel_bridge, notebook_io  # noqa: E402
from ewald import modifiers as mods  # noqa: E402
from ewald.kernel import Kernel, KernelSource, _short  # noqa: E402
from ewald.notebook import Notebook  # noqa: E402


@pytest.fixture
def kernel():
    k = Kernel(python=sys.executable, display_name="Python (test)")
    yield k
    k.shutdown()


def _notebook(path):
    return Notebook.from_py(f"# %%\nfrom ewald import load\nfrom ewald.modifiers import *\npipe = load({str(path)!r})\n"
                            "import os; print(os.getpid())\n\n"
                            "# %%\npipe.add(CoordinationAnalysis(cutoff=3.0))\n\n"
                            "# %%\npipe.num_frames\n")


def test_cells_run_in_the_kernel_and_frames_come_back(kernel, lammps_dir):
    import os

    nb = _notebook(lammps_dir / "dump.custom.lammpstrj")
    nb.set_kernel(kernel)
    assert nb.metadata["ewald"]["kernel"]["display_name"] == "Python (test)" and not kernel.alive  # starts on use
    assert nb.run_all()
    assert kernel.alive and [c.state for c in nb.cells] == ["ok", "ok", "ok"]
    pid = int(nb.cells[0].outputs[0].data)
    assert pid != os.getpid()                                      # another process
    assert nb.cells[2].outputs[-1].data == "3"
    assert [s["name"] for s in nb.cells[1].steps] == ["CoordinationAnalysis"] and nb.cells[1].added == []
    pipe = nb.pipeline
    assert isinstance(pipe.source, KernelSource) and pipe.num_frames == 3
    data = pipe.compute(1)
    assert data.count == 32 and "Coordination" in data.particles and "coordination-rdf" in data.tables
    assert nb.run_cell(1) and nb.pipeline is pipe                  # re-run: same proxy, no second step
    nb.add_step(mods.CreateBonds(cutoff=2.9), index=2)             # runs in the kernel at once
    assert nb.cells[2].state == "ok" and pipe.compute(0).bonds is not None
    assert len(nb.cells[1].steps) == 1


def test_errors_and_switching_back(kernel, lammps_dir):
    nb = Notebook.from_py("# %%\nx = 41\n\n# %%\n1/0\n\n# %%\nx + 1\n")
    nb.set_kernel(kernel)
    assert not nb.run_all()
    assert nb.cells[1].state == "error" and "ZeroDivisionError" in nb.cells[1].outputs[-1].data
    assert nb.cells[2].state == "stale"
    assert nb.run_cell(2) and nb.cells[2].outputs[-1].data == "42"
    nb.set_kernel(None)                                            # back in this Python, fresh namespace
    assert "kernel" not in nb.metadata["ewald"] and not kernel.alive
    assert nb.run_all() is False and nb.namespace["x"] == 41


def test_a_python_without_a_kernel_reports_it():
    nb = Notebook.from_py("# %%\nx = 1\n")
    nb.set_kernel(Kernel(python="/nonexistent/bin/python"))
    assert not nb.run_cell(0)
    assert nb.cells[0].state == "error" and "could not start" in nb.cells[0].outputs[0].data


def test_the_kernel_is_saved_with_the_notebook(tmp_path, kernel):
    nb = Notebook.from_py("# %%\nx = 1\n")
    nb.set_kernel(kernel)
    back = notebook_io.from_ipynb(notebook_io.to_ipynb(nb))
    spec = back.metadata["ewald"]["kernel"]
    assert Kernel.from_spec(spec).python == sys.executable and spec["display_name"] == "Python (test)"
    assert _short("/opt/conda/envs/ml/bin/python") == "Python (ml)" and _short(sys.executable) == "This Python"


def test_frames_round_trip_through_npz(tmp_path, cu_fcc):
    from ewald.bonds import Bonds

    d = cu_fcc.copy() if hasattr(cu_fcc, "copy") else cu_fcc
    d.tables["t"] = {"x": np.arange(3.0), "y": np.ones(3)}
    d.bonds = Bonds([[0, 1]], [[0, 0, 1]])
    d.attributes.update({"Timestep": 7, "skip": object()})
    kernel_bridge.save_frame(d, tmp_path / "f.npz")
    back = kernel_bridge.load_frame(tmp_path / "f.npz")
    assert back.count == d.count and np.allclose(back.positions, d.positions)
    assert list(back.particles["Particle Type"]) == list(d.particles["Particle Type"])
    assert np.allclose(back.cell.matrix, d.cell.matrix) and back.tables["t"]["y"].tolist() == [1, 1, 1]
    assert back.bonds.topology.tolist() == [[0, 1]] and back.attributes["Timestep"] == 7 and "skip" not in back.attributes


def test_frames_carry_surfaces_grids_and_vectors(tmp_path, cu_fcc):
    from ewald.arrows import VectorStyle
    from ewald.data import VoxelGrid
    from ewald.mesh import SurfaceMesh

    d = cu_fcc.copy()
    d.surfaces["poly"] = SurfaceMesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], {"Color": [[1, 0, 0]]},
                                     edges=[[0, 1]], transparency=0.3)
    d.grids["rho"] = VoxelGrid(d.cell, {"Charge Density": np.arange(8.0).reshape(2, 2, 2)})
    d.particles["Force"] = np.zeros((d.count, 3))
    d.vectors["Force"] = VectorStyle(scaling=2.0)
    kernel_bridge.save_frame(d, tmp_path / "f.npz")
    back = kernel_bridge.load_frame(tmp_path / "f.npz")
    m = back.surfaces["poly"]
    assert m.faces.tolist() == [[0, 1, 2]] and m.edges.tolist() == [[0, 1]] and m.transparency == 0.3
    assert m.face_properties["Color"].tolist() == [[1, 0, 0]]
    assert back.grids["rho"]["Charge Density"].sum() == 28 and back.grids["rho"].shape == (2, 2, 2)
    assert back.vectors["Force"].scaling == 2.0


def test_kernel_steps_get_a_local_copy_with_their_results(kernel, lammps_dir):
    nb = Notebook.from_py(f"# %%\nfrom ewald import load\nfrom ewald.modifiers import *\n"
                          f"pipe = load({str(lammps_dir / 'dump.custom.lammpstrj')!r})\n\n"
                          "# %%\npipe.add(CoordinationAnalysis(cutoff=3.0))\n\n"
                          "# %%\npipe.add(ExpressionSelection(expression='Coordination < 99'))\n")
    nb.set_kernel(kernel)
    assert nb.run_all()
    step = nb.cells[1].modifier
    assert type(step).__name__ == "CoordinationAnalysis" and step.cutoff == 3.0 and nb.cells[1].added == []
    nb.pipeline.compute(0)
    nb.refresh_kernel_steps()
    assert "Coordination" in step.last_output["properties"] and nb.cells[2].modifier.last_output["selected"] == 32
    step.enabled = False                                           # the GUI's checkbox, then the cell runs again
    nb.update_step(nb.cells[1])
    assert nb.cells[1].is_step and nb.run_cell(1)
    nb.pipeline.compute(0)
    nb.refresh_kernel_steps()
    assert not nb.cells[1].modifier.enabled and "Coordination" in nb.cells[2].modifier.status   # skipped, reported


def test_frames_carry_the_electronic_structure(tmp_path, vasp_dir):
    """DOS and band plots draw from the 'Electronic Structure' object, so a kernel's frame must
    bring it back, or the Plots tab falls back to plain table lines."""
    from ewald import import_file

    data = import_file(vasp_dir / "electronic" / "vasprun.xml").compute(0)
    path = tmp_path / "f.npz"
    kernel_bridge.save_frame(data, str(path))
    back = kernel_bridge.load_frame(str(path))
    a, b = data.attributes["Electronic Structure"], back.attributes["Electronic Structure"]
    assert type(b) is type(a) and b.efermi == a.efermi and b.num_spins == a.num_spins
    np.testing.assert_allclose(b.eigenvalues, a.eigenvalues)
    np.testing.assert_allclose(b.pdos, a.pdos)
    assert [tuple(t) for t in b.ticks] == [tuple(t) for t in a.ticks] and list(b.breaks) == list(a.breaks)
    assert b.has_projections == a.has_projections
    if a.has_projections:
        np.testing.assert_allclose(b.band_weights([0], None), a.band_weights([0], None))


def test_plain_pyplot_in_the_kernel_gets_the_notebook_style(kernel):
    nb = Notebook.from_py("# %%\nimport matplotlib.pyplot as plt\nprint(plt.rcParams['font.size'])\n")
    nb.set_kernel(kernel)
    nb.run_all()
    assert nb.cells[0].outputs[0].data.split() == ["8.0"]
    nb.set_plot_style("default")
    nb.run_all()
    assert nb.cells[0].outputs[0].data.split() != ["8.0"]


def test_each_frame_cells_run_in_the_kernel(kernel, lammps_dir):
    nb = Notebook.from_py(f"# %%\nfrom ewald import load\npipe = load({str(lammps_dir / 'dump.custom.lammpstrj')!r})\n\n"
                          "# %% [each frame]\ndata.attributes['Twice'] = 2 * frame\n")
    nb.set_kernel(kernel)
    assert nb.run_all()
    assert [s["name"] for s in nb.cells[1].steps] == ["each_frame_cell"]
    assert nb.pipeline.compute(2).attributes["Twice"] == 4


def test_rich_outputs_and_magics_come_back_from_the_kernel(kernel):
    """A kernel cell is an IPython cell too: its HTML, progress bars and magics arrive as they do here."""
    from ewald.shell import PROGRESS_MIME

    nb = Notebook.from_py("# %%\nfrom IPython.display import HTML\nfrom ewald.progress import progress\n"
                          "for _ in progress(range(3), desc='k'):\n    pass\n%time y = 2\nHTML('<b>x</b>')\n")
    nb.set_kernel(kernel)
    assert nb.run_all(), nb.cells[0].outputs
    kinds = [o.kind for o in nb.cells[0].outputs]
    assert kinds == ["rich", "text", "rich"]
    bar, text, html = nb.cells[0].outputs
    assert bar.mime[PROGRESS_MIME]["n"] == 3 and "Wall time" in text.data and html.mime["text/html"] == "<b>x</b>"


def test_view_tabs_come_back_from_the_kernel(kernel, lammps_dir):
    """view(pipe, tab=...) run in the kernel makes the same view tab on the window's side."""
    nb = Notebook.from_py(f"# %%\nfrom ewald import load, view\npipe = load({str(lammps_dir / 'dump.custom.lammpstrj')!r})\n"
                          "\n# %%\nview(pipe, tab='Top', direction='z')\n")
    nb.set_kernel(kernel)
    assert nb.run_all()
    assert nb.pipeline.views["Top"].changed() == {"direction": "z"}
    nb.cells[1].source = "pipe.views.remove('Top')"
    assert nb.run_cell(1)
    assert "Top" not in nb.pipeline.views
