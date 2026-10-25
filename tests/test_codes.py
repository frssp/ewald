"""Simulation codes (ewald.codes): VASP and LAMMPS as built-in entries, plugin codes the same way."""
import importlib.metadata
import json
import ntpath
from pathlib import Path

import pytest

from ewald import codes, plugins, settings
from ewald.codes import SimulationCode
from ewald.runinfo import check_run, run_info
from ewald.settings import Option

EXAMPLE = Path(__file__).parents[1] / "examples" / "plugins" / "qe_code.py"

_INSTALLED = {(ep.name, ep.value) for ep in importlib.metadata.entry_points(group="ewald.codes")}


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    real = plugins._entry_point_scan

    def scan():
        eps = real()
        return lambda group: [ep for ep in eps(group) if (group != "ewald.codes" or (ep.name, ep.value) not in _INSTALLED)]

    monkeypatch.setattr(plugins, "_entry_point_scan", scan)
    monkeypatch.setattr(plugins, "_registry", plugins._Registry())


@pytest.fixture
def qe_plugin(tmp_path, monkeypatch):
    folder = tmp_path / "config" / "plugins"
    folder.mkdir(parents=True)
    (folder / EXAMPLE.name).write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("EWALD_CONFIG_DIR", str(tmp_path / "config"))
    plugins.load_plugins(force=True)
    return folder


def test_vasp_and_lammps_are_builtin_entries():
    found = codes.codes()
    assert list(found)[:2] == ["vasp", "lammps"]
    assert all(c.origin == "built-in" for c in found.values())
    assert codes.get("VASP") is found["vasp"]
    assert plugins.installed()["lammps"].contributions["codes"] == ["lammps"]
    assert {c["name"] for c in plugins.status()["codes"]} >= {"vasp", "lammps"}
    with pytest.raises(KeyError, match="known: vasp, lammps"):
        codes.get("nwchem")


@pytest.mark.parametrize("path, name", [
    ("run/XDATCAR", "vasp"), ("run/vasprun.xml", "vasp"), ("POSCAR_relaxed", "vasp"), ("outcar", "vasp"),
    ("md/dump.lammpstrj", "lammps"), ("md/DUMP.LAMMPSTRJ", "lammps"), ("log.lammps", "lammps"),
    (r"C:\runs\md\XDATCAR", "vasp"), ("cluster:/scratch/me/md/traj.dump", "lammps"), ("cu.xyz", None)])
def test_code_of_by_file_name_on_every_platform(path, name):
    code = codes.code_of(path)
    assert (code.name if code else None) == name


def test_code_of_reads_a_windows_path_on_any_os():
    assert codes.code_of(ntpath.join("D:\\", "data", "run1", "OSZICAR")).name == "vasp"


def test_lammps_dump_with_any_name_is_found_by_content(tmp_path):
    dump = tmp_path / "trajectory.txt"
    dump.write_text("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\n"
                    "ITEM: ATOMS id type x y z\n1 1 0 0 0\n", encoding="utf-8", newline="\r\n")
    assert codes.code_of(dump).name == "lammps"


def test_lammps_settings_live_on_the_codes_page_and_old_keys_move(tmp_path):
    assert settings.option("codes.lammps.type_prompt").page == "Codes"
    assert "general.lammps_type_prompt" not in settings.OPTION and settings.option("codes.lammps.type_map") is None
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"general.lammps_type_map": "1=Cu,2=Zr", "general.lammps_type_prompt": "never"}),
                    encoding="utf-8")
    s = settings.load(path)          # the old default map is dropped: it fits almost no run
    assert s["codes.lammps.type_prompt"] == "never"
    s.save()
    assert json.loads(path.read_text(encoding="utf-8")) == {"codes.lammps.type_prompt": "never"}
    with pytest.raises(ValueError):
        s["codes.lammps.type_prompt"] = "sometimes"


def test_lammps_open_options_ask_for_a_type_map(tmp_path):
    dump = tmp_path / "dump.lammpstrj"
    dump.write_text("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n0\n", encoding="utf-8")
    s = settings.Settings({}, tmp_path / "s.json")
    hook = codes.get("lammps").hook("open_options")
    asked = []
    assert hook(dump, s, lambda *a: asked.append(a) or "1=Cu, 2=Zr") == {"type_map": {"1": "Cu", "2": "Zr"}}
    assert asked[0][2] == ""                            # the prompt starts empty
    assert hook(dump, s, lambda *a: None) == {}         # cancelled
    assert hook(dump, s, lambda *a: "  ") == {}         # left empty: numeric types
    s["codes.lammps.type_prompt"] = "never"
    assert hook(dump, s, lambda *a: pytest.fail("asked")) == {}
    assert hook(tmp_path / "POSCAR", settings.Settings({}, tmp_path / "t.json"), lambda *a: pytest.fail("asked")) == {}
    with pytest.raises(ValueError, match="type=element"):
        codes.parse_type_map("1Cu")


def test_disabling_the_builtin_group_turns_the_code_off():
    plugins.set_enabled("lammps", False)
    try:
        assert "lammps" not in codes.codes() and "vasp" in codes.codes()
        assert codes.code_of("dump.lammpstrj") is None
    finally:
        plugins.set_enabled("lammps", True)
    assert "lammps" in codes.codes()


def test_a_plugin_cannot_take_a_builtin_name_or_misname_options():
    with pytest.raises(ValueError, match="built-in"):
        plugins.register_simulation_code(SimulationCode("vasp"))
    with pytest.raises(ValueError, match="codes.cp2k."):
        SimulationCode("cp2k", options=[Option("codes.other.x", "", "X")])
    with pytest.raises(ValueError, match="one word"):
        SimulationCode("my code")
    with pytest.raises(TypeError):
        plugins.register_simulation_code(lambda: 3)


def test_plugin_code_options_survive_before_and_after_loading(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"codes.cp2k.project": "water", "codes.gone.x": 1}), encoding="utf-8")
    s = settings.load(path)       # cp2k not registered yet: kept, not dropped
    assert s.changed() == {"codes.cp2k.project": "water", "codes.gone.x": 1}
    s.save()
    assert json.loads(path.read_text(encoding="utf-8"))["codes.cp2k.project"] == "water"

    @plugins.simulation_code
    def cp2k():
        return SimulationCode("cp2k", "CP2K", files=("*.ener",),
                              options=[Option("codes.cp2k.project", "", "Project name")])

    assert s["codes.cp2k.project"] == "water"
    assert settings.option("codes.cp2k.project").page == "Codes"
    assert "codes.cp2k.project" in [o.key for o in settings.all_options()]
    assert codes.code_of("run-1.ener").name == "cp2k"
    with pytest.raises(KeyError):
        s["codes.gone.x"]


def test_runinfo_and_checks_dispatch_through_the_registry(tmp_path):
    md = tmp_path / "md"
    md.mkdir()
    (md / "INCAR").write_text("IBRION = 0\nNSW = 100\nPOTIM = 2\nTEBEG = 600\nSMASS = 0\n", encoding="utf-8")
    info = run_info(str(md / "XDATCAR"))
    assert info.code == "vasp" and info.kind == "md" and info.timestep == 2.0
    folder = tmp_path / "lmp"
    folder.mkdir()
    (folder / "log.lammps").write_text("units metal\ntimestep 0.002\nfix 1 all nvt temp 300 300 0.1\nrun 1000\n"
                                       "ERROR: Lost atoms\n", encoding="utf-8")
    info = run_info(str(folder / "whatever.out"))      # known by the log.lammps marker next to it
    assert info.code == "lammps" and info.ensemble == "NVT" and info.timestep == pytest.approx(2.0)


def test_check_run_uses_a_plugin_codes_check(tmp_path, cu_fcc):
    from ewald import load
    from ewald.io import export_file
    from ewald.runinfo import Finding

    def check(chk, info, path):
        chk.findings.append(Finding("warning", "Toy code says no", path))

    plugins.register_simulation_code(SimulationCode("toy", files=("*.toy.xyz",), check=check))
    path = tmp_path / "run.toy.xyz"
    export_file(cu_fcc, str(path), format="extxyz")
    result = check_run(load(str(path), format="extxyz"))
    assert result.info.code == "toy"
    assert any(f.title == "Toy code says no" for f in result.findings)


def test_thermo_siblings_come_from_the_codes(tmp_path):
    from ewald.io.thermo import sibling_files

    (tmp_path / "OSZICAR").write_text("", encoding="utf-8")
    (tmp_path / "log.lammps").write_text("", encoding="utf-8")
    assert sibling_files(tmp_path / "XDATCAR", "xdatcar") == [str(tmp_path / "OSZICAR")]
    assert sibling_files(tmp_path / "dump.lammpstrj", "lammps-dump") == [str(tmp_path / "log.lammps")]
    plugins.register_simulation_code(SimulationCode("toy", thermo={"extxyz": "toy.log"}))
    (tmp_path / "toy.log").write_text("", encoding="utf-8")
    assert sibling_files(tmp_path / "traj.xyz", "extxyz") == [str(tmp_path / "toy.log")]


def test_example_qe_plugin(qe_plugin, tmp_path):
    from ewald.workflow_folder import describe_folder

    qe = codes.get("qe")
    assert qe.title == "Quantum ESPRESSO" and qe.origin.endswith("qe_code.py")
    assert plugins.installed()["qe_code"].contributions == {"codes": ["qe"]}
    run = tmp_path / "study" / "01_md"
    run.mkdir(parents=True)
    (run / "pw.in").write_text("&CONTROL\n  calculation = 'md'\n  dt = 40, nstep = 500\n/\n"
                               "&IONS\n  ion_temperature = 'rescaling', tempw = 800.d0\n/\n", encoding="utf-8")
    (run / "pw.out").write_text("Program PWSCF\n", encoding="utf-8")
    info = run_info(str(run / "pw.out"))
    assert (info.code, info.kind, info.ensemble, info.steps) == ("qe", "md", "NVT", 500)
    assert info.timestep == pytest.approx(40 * 4.8378e-2) and info.temperature == (800.0, 800.0)
    wf = describe_folder(tmp_path / "study")
    assert [s.scalars.get("code") for s in wf.steps] == ["qe"]
