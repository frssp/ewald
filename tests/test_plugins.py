import gzip
import shutil
import textwrap
from pathlib import Path

import numpy as np
import pytest

from ewald import DataCollection, import_file, plugins
from ewald.io.base import FrameReader

EXAMPLE = Path(__file__).parents[1] / "examples" / "plugins" / "npz_trajectory.py"


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    yield
    monkeypatch.setattr(plugins, "_registry", plugins._Registry())


def _write_plugin(name, source):
    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(textwrap.dedent(source))
    return folder / name


def _npz(path, frames=4, atoms=5):
    rng = np.random.default_rng(1)
    np.savez(path, positions=rng.random((frames, atoms, 3)) * 10, types=np.array(["Li"] * 2 + ["O"] * 3),
             cell=np.eye(3) * 10, forces=rng.random((frames, atoms, 3)), energies=np.arange(frames, dtype=float))
    return path


def test_function_reader_from_plugin_folder(tmp_path):
    _write_plugin("simple.py", '''
        from ewald import DataCollection
        from ewald.plugins import reader

        @reader("pts", patterns=("*.pts",), description="one point per line")
        def read_pts(path, scale=1.0):
            frames = []
            for block in open(path).read().strip().split("\\n\\n"):
                xyz = [[float(v) * scale for v in line.split()] for line in block.splitlines()]
                frames.append(DataCollection(xyz, ["Ar"] * len(xyz)))
            return frames
    ''')
    f = tmp_path / "run.pts"
    f.write_text("0 0 0\n1 0 0\n\n0 0 1\n2 0 0\n")
    pipe = import_file(f, scale=2.0)
    assert pipe.num_frames == 2
    data = pipe.compute(1)
    assert data.count == 2 and data.positions[1, 0] == 4.0
    assert list(data.types) == ["Ar", "Ar"]
    assert plugins.status()["readers"][0]["origin"].endswith("simple.py")


def test_example_npz_plugin_is_lazy_framereader(tmp_path):
    folder = plugins.plugin_dir()
    folder.mkdir(parents=True)
    shutil.copy(EXAMPLE, folder)
    path = _npz(tmp_path / "md.npz")
    pipe = import_file(path, type_map={"Li": "Na"})
    assert isinstance(pipe.source.readers[0], FrameReader)
    assert pipe.num_frames == 4
    d = pipe.compute(2)
    assert d.count == 5 and set(d.types) == {"Na", "O"}
    assert d["Force"].shape == (5, 3)
    assert d.attributes["Energy"] == 2.0
    assert np.isclose(d.cell.volume, 1000.0)


def test_compressed_name_still_matches(tmp_path):
    shutil.copy(EXAMPLE, plugins.plugin_dir().mkdir(parents=True) or plugins.plugin_dir())
    plain = _npz(tmp_path / "md.npz")
    assert plugins.find_reader(str(plain) + ".gz").name == "npz-trajectory"


def test_explicit_format_and_sniff(tmp_path):
    class Magic(FrameReader):
        def __init__(self, path):
            self.n = int(Path(path).read_text().split()[1])

        def __len__(self):
            return self.n

        def read(self, i):
            return DataCollection([[i, 0, 0]], ["H"])

    plugins.register_reader("magic", Magic, sniff=lambda p: Path(p).read_text().startswith("MAGIC"))
    f = tmp_path / "anything.txt"
    f.write_text("MAGIC 3")
    assert import_file(f).num_frames == 3                  # detected by sniff
    g = tmp_path / "other.txt"
    g.write_text("x 2")
    assert import_file(g, format="MAGIC").num_frames == 2  # format= wins, case-insensitive


def test_native_readers_take_precedence(vasp_dir):
    plugins.register_reader("greedy", lambda p: DataCollection([[0, 0, 0]]), patterns=("*",))
    pipe = import_file(vasp_dir / "XDATCAR")
    assert pipe.num_frames > 1            # still the native XDATCAR reader, not the 1-frame plugin


def test_broken_plugin_is_reported_not_raised(tmp_path, capsys):
    _write_plugin("broken.py", "import does_not_exist\n")
    _write_plugin("ok.py", '''
        from ewald.plugins import reader

        @reader("okfmt", patterns=("*.ok",))
        def read_ok(path):
            raise ValueError("bad file")
    ''')
    st = plugins.status()
    assert [r["name"] for r in st["readers"]] == ["okfmt"]
    (err,) = [v for k, v in st["errors"].items() if k.endswith("broken.py")]
    assert "ModuleNotFoundError" in err

    from ewald.__main__ import main

    assert main(["plugins"]) == 1
    out = capsys.readouterr().out
    assert "okfmt" in out and "FAILED" in out and "broken.py" in out

    f = tmp_path / "x.ok"
    f.write_text("")
    with pytest.raises(ValueError, match="bad file"):  # reader errors surface to the caller as usual
        import_file(f)


def test_entry_point_readers(monkeypatch, tmp_path):
    import importlib.metadata as md

    class EP:
        name, value = "ep-format", "fake_pkg:reader"

        def load(self):
            def read(path):
                return [DataCollection([[0, 0, 0], [1, 1, 1]], ["Cu", "Cu"])]

            read.patterns = ("*.epf",)
            return read

    real = md.entry_points
    monkeypatch.setattr(md, "entry_points", lambda **kw: [EP()] if kw.get("group") == "ewald.readers" else real(**kw))
    f = tmp_path / "a.epf"
    f.write_text("")
    assert import_file(f).compute(0).count == 2
    assert plugins.readers()["ep-format"].origin == "entry point fake_pkg:reader"


def test_disable_with_env(monkeypatch):
    _write_plugin("simple.py", "from ewald.plugins import reader\nreader('zzz', patterns=('*.zzz',))(lambda p: [])\n")
    monkeypatch.setenv("EWALD_NO_PLUGINS", "1")
    assert plugins.readers() == {}


def test_llm_tool_sees_plugin_formats(tmp_path):
    shutil.copy(EXAMPLE, plugins.plugin_dir().mkdir(parents=True) or plugins.plugin_dir())
    from ewald.llm import Toolset

    ts = Toolset()
    (load,) = [d for d in ts.definitions() if d["name"] == "load_structure"]
    assert "npz-trajectory" in load["input_schema"]["properties"]["format"]["description"]
    res = ts.call("load_structure", {"path": str(_npz(tmp_path / "md.npz"))})
    assert not res.is_error and '"num_frames": 4' in res.text()
    assert "import_file(" in ts.to_script()


def test_gui_open_dialog_lists_plugin_patterns():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from ewald.gui.app import file_filter

    plugins.register_reader("pts", lambda p: [], patterns=("*.pts",))
    assert "*.pts" in file_filter() and "XDATCAR*" in file_filter()
