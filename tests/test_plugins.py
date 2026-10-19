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
    assert plugins.readers()["pts"].origin.endswith("simple.py")


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
    assert [r["name"] for r in st["readers"] if r["origin"] != "built-in"] == ["okfmt"]
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
    assert "zzz" not in plugins.readers() and "xdatcar" in plugins.readers()   # built-ins stay


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


def test_builtin_formats_live_in_the_registry(vasp_dir, lammps_dir):
    st = {r["name"]: r for r in plugins.status()["readers"]}
    assert {"lammps-dump", "xdatcar", "lammps-data", "ase"} <= set(st)
    assert all(st[n]["origin"] == "built-in" for n in ("lammps-dump", "xdatcar", "ase"))
    order = [r["name"] for r in plugins.status()["readers"]]
    assert order.index("lammps-dump") < order.index("ase")
    assert plugins.find_reader(vasp_dir / "XDATCAR").name == "xdatcar"
    assert plugins.find_reader(vasp_dir / "POSCAR").name == "ase"
    assert plugins.find_reader("x", "lammps/dump").name == "lammps-dump"   # alias
    assert plugins.find_reader("x", "espresso-in").name == "ase"           # unknown names go to ASE
    assert import_file(vasp_dir / "POSCAR", format="vasp").compute(0).count > 0
    structure = plugins.file_patterns("structure")
    assert structure.index("*.lammpstrj") < structure.index("POSCAR*")   # priority order


def test_plugin_can_replace_a_builtin(vasp_dir):
    plugins.register_reader("xdatcar", lambda p: [DataCollection([[0, 0, 0]], ["Si"])],
                            sniff=lambda p: "XDATCAR" in str(p), priority=11)
    assert import_file(vasp_dir / "XDATCAR").num_frames == 1


def test_volumetric_readers_are_builtin_plugins(tmp_path):
    names = plugins.readers()
    assert names["cube"].category == names["vasp-volumetric"].category == "volumetric"
    assert plugins.find_reader(tmp_path / "LOCPOT").name == "vasp-volumetric"
    assert plugins.find_reader(tmp_path / "x.cube.gz").name == "cube"
    assert plugins.find_reader("x", "locpot").name == "vasp-volumetric"   # alias, kind from the name
    assert "CHGCAR*" in plugins.file_patterns("volumetric")
    assert "CHGCAR*" not in plugins.file_patterns("structure")


LOCAL_COMPOSITION = EXAMPLE.parent / "local_composition.py"


def test_modifier_plugin_from_folder_is_everywhere(cu_fcc):
    folder = plugins.plugin_dir()
    folder.mkdir(parents=True)
    shutil.copy(LOCAL_COMPOSITION, folder)
    from ewald import StaticSource, Pipeline
    from ewald.modifiers import LocalComposition          # imports like a built-in

    assert plugins.modifiers()["LocalComposition"].category == "Analysis"
    pipe = Pipeline(StaticSource(cu_fcc))
    pipe.modifiers.append(LocalComposition(cutoff=3.0, element="Cu"))
    assert np.allclose(pipe.compute(0)["Local Cu Fraction"], 1.0)

    from ewald.llm import Toolset

    ts = Toolset(pipe)
    listed = ts.call("list_modifiers", {}).text()
    assert "LocalComposition" in listed
    res = ts.call("add_modifier", {"name": "LocalComposition", "params": {"cutoff": 2.9}})
    assert not res.is_error, res.text()
    script = ts.to_script()
    assert "from ewald.modifiers import" in script and "LocalComposition" in script

    from ewald.llm.reference import api_reference

    assert "LocalComposition" not in api_reference()
    assert "`LocalComposition(cutoff: float = 3.0" in api_reference(with_plugins=True)


def test_modifier_name_clash_and_type_checks(capsys):
    from ewald import Modifier

    with pytest.raises(ValueError, match="built-in"):
        plugins.register_modifier(type("CoordinationAnalysis", (Modifier,), {}))
    with pytest.raises(TypeError):
        plugins.register_modifier(object)
    _write_plugin("clash.py", '''
        from ewald import Modifier
        from ewald.plugins import modifier

        @modifier
        class Slice(Modifier):
            pass
    ''')
    st = plugins.status()
    assert any(k.endswith("clash.py") and "built-in" in v for k, v in st["errors"].items())


def test_modifier_entry_point_and_cli(monkeypatch, capsys):
    import importlib.metadata as md

    from ewald import Modifier

    class Shift(Modifier):
        """Move every particle by ``dz`` along z."""

        category = "Transform"

        def __init__(self, dz: float = 1.0):
            self.dz = dz

        def modify(self, data, ctx):
            data.positions = data.positions + [0, 0, self.dz]

    class EP:
        name, value = "ShiftZ", "pkg:Shift"

        def load(self):
            return Shift

    real = md.entry_points
    monkeypatch.setattr(md, "entry_points", lambda **kw: [EP()] if kw.get("group") == "ewald.modifiers" else real(**kw))
    import ewald.modifiers as mods

    assert mods.ShiftZ is Shift and mods.ShiftZModifier is Shift   # OVITO-style suffix too
    assert plugins.modifiers()["ShiftZ"].category == "Transform"
    from ewald.__main__ import main

    assert main(["plugins"]) is None
    out = capsys.readouterr().out
    assert "ShiftZ" in out and "Move every particle" in out
    with pytest.raises(AttributeError):
        mods.NoSuchModifier


def test_gui_menu_lists_modifier_plugins():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from ewald import Modifier
    from ewald.gui import pipeline_panel

    @plugins.modifier(category="My lab")
    class Tagger(Modifier):
        def modify(self, data, ctx):
            data["Tag"] = np.ones(data.count)

    pipeline_panel._discover()
    try:
        assert pipeline_panel.MODIFIER_MENU["My lab"]["Tagger"] is Tagger
    finally:
        pipeline_panel.MODIFIER_MENU.pop("My lab", None)


# --- plugin manifests, enable/disable and installed() ---------------------------------------------

def test_file_manifest_is_read_without_running_the_file(tmp_path):
    _write_plugin("labtools.py", '''
        """Lab tools."""
        from __future__ import annotations
        from ewald import Modifier
        from ewald.plugins import modifier

        @modifier(category="Lab")
        class LabCount(Modifier):
            """Counts atoms."""
            def modify(self, data, ctx):
                data.attributes["LabCount.n"] = data.count

        __ewald__ = {"id": "lab-tools", "display-name": "Lab tools", "version": "1.2.0",
                     "publisher": "frssp", "requires-ewald": ">=0.1"}
    ''')
    _write_plugin("future.py", '''
        raise RuntimeError("must not be imported")
        __ewald__ = {"requires-ewald": ">=99"}
    ''')
    found = plugins.installed()
    lab = found["lab-tools"]
    assert (lab.name, lab.version, lab.publisher, lab.kind, lab.state) == ("Lab tools", "1.2.0", "frssp", "folder", "loaded")
    assert lab.contributions == {"modifiers": ["LabCount"]}
    future = found["future"]
    assert future.state == "incompatible" and "needs ewald >=99" in future.errors[0]
    st = plugins.status()
    assert any("needs ewald" in e for e in st["errors"].values())
    assert not any("must not be imported" in e for e in st["errors"].values())
    assert {p["id"] for p in st["plugins"]} == {"lab-tools", "future"}


def test_plugin_without_manifest_gets_its_file_name():
    _write_plugin("plain.py", '''
        from ewald.plugins import colormap

        @colormap
        def stripes():
            return ["black", "white"]
    ''')
    p = plugins.installed()["plain"]
    assert p.name == "plain" and p.version == "" and p.contributions == {"colormaps": ["stripes"]}


def test_disable_and_enable_a_plugin():
    _write_plugin("tool.py", '''
        from ewald import Modifier
        from ewald.plugins import modifier

        @modifier
        class ToolMod(Modifier):
            def modify(self, data, ctx):
                pass
    ''')
    assert "ToolMod" in plugins.modifiers()
    plugins.set_enabled("tool", False)
    assert "ToolMod" not in plugins.modifiers()
    assert plugins.installed()["tool"].state == "disabled"
    assert plugins.disabled() == {"tool"}
    plugins._registry.loaded = False      # a new session reads the saved state
    assert "ToolMod" not in plugins.modifiers()
    plugins.set_enabled("tool", True)
    assert "ToolMod" in plugins.modifiers() and plugins.installed()["tool"].state == "loaded"


def test_package_manifest_and_entry_points(tmp_path, monkeypatch):
    site = tmp_path / "site"
    pkg = site / "ewald_fakefmt"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(textwrap.dedent('''
        from ewald import DataCollection

        def read(path):
            """Fake format."""
            return [DataCollection([[0, 0, 0]], ["H"])]
    '''))
    (pkg / "ewald-plugin.toml").write_text(textwrap.dedent('''
        id = "fakefmt"
        display-name = "Fake format"
        version = "0.3.0"
        requires-ewald = ">=0.1, <99"
    '''))
    info = site / "ewald_fakefmt-0.2.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: ewald-fakefmt\nVersion: 0.2\nSummary: Fake\n")
    (info / "entry_points.txt").write_text("[ewald.readers]\nfake = ewald_fakefmt:read\n\n"
                                           "[ewald.plugins]\nfakefmt = ewald_fakefmt\n")
    monkeypatch.syspath_prepend(str(site))
    p = plugins.installed()["fakefmt"]
    assert (p.kind, p.name, p.version, p.origin, p.state) == ("package", "Fake format", "0.3.0", "ewald-fakefmt", "loaded")
    assert p.contributions == {"readers": ["fake"]}
    assert plugins.readers()["fake"].origin == "entry point ewald_fakefmt:read"
    plugins.set_enabled("fakefmt", False)
    assert "fake" not in plugins.readers()


@pytest.mark.parametrize("version, requirement, ok", [
    ("0.1.0", ">=0.1", True), ("0.1.0", ">0.1", False), ("0.1.0", ">=0.1, <1", True),
    ("1.0", "<1", False), ("0.3.2", "~=0.3.0", True), ("0.4", "~=0.3.0", False),
    ("0.1.0", "==0.1", True), ("0.1.0", "!=0.1.0", False), ("0.2.0.dev1", ">=0.2", True), ("0.1", "", True)])
def test_version_satisfies(version, requirement, ok):
    assert plugins.version_satisfies(version, requirement) is ok


def test_bad_version_requirement_is_reported():
    with pytest.raises(ValueError, match="version requirement"):
        plugins.version_satisfies("0.1", "latest")
    _write_plugin("odd.py", '__ewald__ = {"requires-ewald": "latest"}\n')
    assert plugins.installed()["odd"].state == "incompatible"


def test_forge_saves_a_manifest(tmp_path):
    from ewald.llm.forge import ForgeResult, save_plugin

    result = ForgeResult("Count the atoms", module_name="atom_count", class_name="AtomCount",
                         module_source='"""Doc."""\nfrom __future__ import annotations\nX = 1\n', test_source="")
    path = save_plugin(result)
    manifest = plugins.file_manifest(path)
    assert manifest["source"] == "forge" and manifest["id"] == "atom_count"
    assert plugins.installed()["atom_count"].kind == "forge"
