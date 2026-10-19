import json
import shutil
import zipfile

import pytest

from ewald import notebook_io
from ewald.notebook import Notebook

NOTEBOOK = '''# %% [markdown]
# # Cu test

# %%
from ewald import load
from ewald.modifiers import *
from ewald.views import show
pipe = load({path!r})

# %% coordination
pipe.add(CoordinationAnalysis(cutoff=3.0))

# %% CN histogram
show(pipe, histogram="Coordination")

# %%
print("atoms", pipe.compute(0).count)
'''


@pytest.fixture
def cu_file(tmp_path):
    from ase.build import bulk
    from ase.io import write

    atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((3, 3, 3))
    del atoms[[0]]
    path = tmp_path / "data" / "cu.extxyz"
    path.parent.mkdir()
    write(path, atoms)
    return str(path)


@pytest.fixture
def nb(cu_file, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    nb = Notebook.from_py(NOTEBOOK.format(path="data/cu.extxyz"), path=str(tmp_path / "cu.py"))
    assert nb.run_all(), [c.outputs for c in nb.cells]
    return nb


def test_ipynb_round_trip_keeps_code_titles_outputs_and_metadata(nb, tmp_path):
    path = tmp_path / "cu.ipynb"
    nb.metadata["custom"] = {"kept": True}       # e.g. a Jupyter extension's metadata
    nb.state = {"frame": 0, "camera": {"position": (1.0, 2.0, 3.0)}}
    nb.save(str(path))
    doc = json.loads(path.read_text())
    nbformat = pytest.importorskip("nbformat")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no missing cell ids or other nbformat complaints
        nbformat.validate(nbformat.reads(path.read_text(), as_version=4))
    assert doc["metadata"]["ewald"]["state"]["camera"]["position"] == [1.0, 2.0, 3.0]
    back = Notebook.load(str(path))
    assert [c.source for c in back.cells] == [c.source for c in nb.cells]
    assert [c.title for c in back.cells] == [c.title for c in nb.cells]
    assert [c.id for c in back.cells] == [c.id for c in nb.cells]
    assert [[o.kind for o in c.outputs] for c in back.cells] == [[o.kind for o in c.outputs] for c in nb.cells]
    assert any(o.kind == "image" for o in back.cells[3].outputs)        # the view's figure
    assert back.cells[4].outputs[0].data == "atoms 107\n"
    assert back.metadata["custom"] == {"kept": True} and back.state["frame"] == 0
    assert back.to_py() == nb.to_py()                                      # .py <-> .ipynb


def test_ipynb_from_jupyter_shows_steps_and_runs(cu_file, tmp_path):
    """A notebook written by Jupyter (list sources, no ewald metadata) opens with step cells."""
    doc = {"nbformat": 4, "nbformat_minor": 5, "metadata": {},
           "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": ["from ewald import load\n", "from ewald.modifiers import *\n",
                                 f"pipe = load({cu_file!r})"]},
                     {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": ["pipe.add(CoordinationAnalysis(cutoff=3.0))"]},
                     {"cell_type": "raw", "metadata": {}, "source": ["ignored"]}]}
    nb = notebook_io.from_ipynb(json.dumps(doc))
    assert len(nb.cells) == 2 and nb.cells[1].is_step
    assert nb.run_all() and "Coordination" in nb.pipeline.compute(0).particles


def test_ewd_bundles_figures_data_and_opens_restricted(nb, cu_file, tmp_path):
    path = tmp_path / "cu.ewd"
    nb.state = {"frame": 0}
    nb.save(str(path))
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        manifest = json.loads(z.read("manifest.json"))
    assert {"notebook.ipynb", "manifest.json", "data/data/cu.extxyz",
            "figures/cn-histogram.svg", "figures/cn-histogram.png", "figures/cn-histogram.csv"} <= names
    assert manifest["data"][0]["path"] == "data/cu.extxyz" and manifest["data"][0]["bundled"]
    assert manifest["figures"][0]["cell"] == 3 and manifest["state"] == {"frame": 0}

    opened = Notebook.load(str(path))
    assert opened.restricted and opened.warnings == [] and opened.name == "cu"
    assert any(o.kind == "image" for o in opened.cells[3].outputs)          # readable without running
    with pytest.raises(PermissionError):
        opened.run_all()
    opened.trust()
    assert opened.run_all() and opened.pipeline.compute(0).count == 107


def test_ewd_finds_its_bundled_data_after_moving(nb, tmp_path):
    path = tmp_path / "cu.ewd"
    nb.save(str(path))
    elsewhere = tmp_path / "sent"
    elsewhere.mkdir()
    shutil.copy(path, elsewhere / "cu.ewd")      # sent without the data folder
    opened = Notebook.load(str(elsewhere / "cu.ewd"))
    assert opened.warnings == ["data/cu.extxyz not found next to the notebook: using the copy saved in it"]
    opened.trust()
    assert opened.run_all() and opened.pipeline.compute(0).count == 107


def test_ewd_links_large_files_and_warns_when_they_change(nb, cu_file, tmp_path):
    path = tmp_path / "cu.ewd"
    nb.save(str(path), data_limit_mb=0, figures=False)
    with zipfile.ZipFile(path) as z:
        assert not any(n.startswith(("data/", "figures/")) for n in z.namelist())
    with open(cu_file, "a") as fh:
        fh.write("\n")
    assert Notebook.load(str(path)).warnings == ["data/cu.extxyz changed since the notebook was saved"]
    shutil.rmtree(tmp_path / "data")
    assert "not saved in the notebook" in Notebook.load(str(path)).warnings[0]


def test_fingerprint_reads_only_the_ends(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"x" * (3 << 20))
    b.write_bytes(b"x" * (1 << 20) + b"y" + b"x" * ((2 << 20) - 1))   # differs only in the middle
    assert notebook_io.fingerprint(a) == notebook_io.fingerprint(b)
    b.write_bytes(b"x" * (3 << 20) + b"z")
    assert notebook_io.fingerprint(a) != notebook_io.fingerprint(b)


def test_ewd_carries_the_plugins_its_steps_use(cu_file, tmp_path):
    from ewald import plugins

    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ewd_shift.py").write_text(
        "from ewald.pipeline import Modifier\nfrom ewald.plugins import modifier\n\n"
        "__ewald__ = {'version': '0.3'}\n\n"
        "@modifier(category='Test')\nclass EwdShift(Modifier):\n    def __init__(self, dz: float = 1.0):\n"
        "        self.dz = dz\n\n    def modify(self, data, ctx):\n        return None\n")
    plugins.load_plugin_file(folder / "ewd_shift.py")
    nb = Notebook.from_py(f"# %%\nfrom ewald import load\npipe = load({cu_file!r})\n\n"
                          "# %%\nfrom ewald.modifiers import EwdShift\npipe.add(EwdShift(dz=2.0))\n")
    assert nb.run_all()
    manifest = notebook_io.save_ewd(nb, tmp_path / "p.ewd", figures=False)
    assert manifest["plugins"] == [{"id": "ewd_shift", "name": "ewd_shift", "version": "0.3", "kind": "folder",
                                    "modifiers": ["EwdShift"], "file": "plugins/ewd_shift.py"}]
    with zipfile.ZipFile(tmp_path / "p.ewd") as z:
        assert "class EwdShift" in z.read("plugins/ewd_shift.py").decode()
