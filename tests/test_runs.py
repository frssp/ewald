"""Run information from input files, run health checks, and several runs in one notebook."""
import numpy as np
import pytest

from ewald import arrhenius, check_run, load, load_runs, msd, rdf, run_info
from ewald.runinfo import detect_equilibration, read_incar, read_lammps_input, statistical_inefficiency
from ewald.runs import RunTables

KB = 8.617333262e-5


def write_vasp_md(folder, temperature, frames=200, nblock=5, potim=2.0, ea=0.3, d0=5.0, natoms=16,
                  seed=0, nelm_hit=(), relax_steps=0, incar=True):
    """A VASP MD run: INCAR, OSZICAR (SCF lines, T=, E=) and XDATCAR of Li atoms on random walks
    whose diffusion coefficient follows D0 exp(-Ea/kT) (A^2/ps)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if incar:
        (folder / "INCAR").write_text(
            "SYSTEM = Li walk  # comment\n"
            f"IBRION = 0 ; NSW = {frames * nblock}\n"
            f"POTIM = {potim}\nNBLOCK = {nblock}\n"
            f"TEBEG = {temperature} ; TEEND = {temperature}\n"
            "MDALGO = 2 ! Nose-Hoover\nSMASS = 1.0\nNELM = 40\n")
    (folder / "KPOINTS").write_text("Gamma only\n0\nGamma\n1 1 1\n")
    nsteps = frames * nblock
    lines = []
    for step in range(1, nsteps + 1):
        n = 40 if step in nelm_hit else 12
        lines += [f"DAV: {k:3d}    -0.1E+02   -0.1E-02   -0.1E-01   48   0.1E+00" for k in range(1, n + 1)]
        t = temperature + rng.normal(0, 0.05 * temperature) + 400 * np.exp(-step / 20) * (relax_steps > 0)
        e0 = -50.0 + rng.normal(0, 0.01) + 2.0 * np.exp(-step / 30) * (relax_steps > 0)
        lines.append(f"{step:5d} T= {t:8.1f} E= {-49.0 + rng.normal(0, 1e-4):.8E} F= {e0:.8E} E0= {e0:.8E}  "
                     f"EK= 0.10E+01 SP= 0.0E+00 SK= 0.0E+00")
    (folder / "OSZICAR").write_text("       N       E                     dE\n" + "\n".join(lines) + "\n")
    cell = 16.0
    d = d0 * np.exp(-ea / (KB * temperature))
    dt = nblock * potim / 1000.0
    steps = rng.normal(0.0, np.sqrt(2 * d * dt), size=(frames, natoms, 3))
    steps[0] = 0
    grid = np.array([(i, j, k) for i in range(4) for j in range(2) for k in range(2)], float)[:natoms]
    pos = grid * [4.0, 8.0, 8.0] + 1.0 + np.cumsum(steps, axis=0)
    out = ["Li walk", "           1", f"  {cell:.6f} 0 0", f"  0 {cell:.6f} 0", f"  0 0 {cell:.6f}",
           "   Li", f"   {natoms}"]
    for k in range(frames):
        out.append(f"Direct configuration= {k + 1:5d}")
        out += [" ".join(f"{x:.8f}" for x in row) for row in (pos[k] / cell) % 1.0]
    (folder / "XDATCAR").write_text("\n".join(out) + "\n")
    return folder / "XDATCAR"


def write_lammps_run(folder, target=600.0, mean=600.0, frames=60, every=100, error=False, warnings=0, units="metal",
                     fix="fix 1 all nvt temp {T} {T} 0.1"):
    """A LAMMPS run: log.lammps echoing its input plus thermo, and a dump of 4 atoms."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    log = ["LAMMPS (22 Jul 2025)", f"units {units}", "atom_style atomic", "timestep 0.002",
           fix.format(T=target), f"dump d1 all custom {every} dump.lammpstrj id type x y z", "thermo 100",
           "WARNING: something odd" if warnings else "", f"run {frames * every}",
           "   Step          Temp          PotEng         TotEng"]
    log = [x for x in log if x]
    for k in range(frames + 1):
        log.append(f"{k * every:8d} {mean + rng.normal(0, 5):12.4f} {-100 + rng.normal(0, 0.05):14.6f} "
                   f"{-95 + rng.normal(0, 1e-4):14.6f}")
    log.append("Loop time of 1.0 on 1 procs")
    log += ["WARNING: more"] * max(warnings - 1, 0)
    if error:
        log.append("ERROR: Lost atoms: original 4 current 3")
    (folder / "log.lammps").write_text("\n".join(log) + "\n")
    dump = []
    for k in range(frames + 1):
        dump += ["ITEM: TIMESTEP", str(k * every), "ITEM: NUMBER OF ATOMS", "4",
                 "ITEM: BOX BOUNDS pp pp pp", "0 8", "0 8", "0 8", "ITEM: ATOMS id type x y z"]
        dump += [f"{i + 1} 1 {2 * i + 0.01 * k} 1.0 1.0" for i in range(4)]
    (folder / "dump.lammpstrj").write_text("\n".join(dump) + "\n")
    return folder / "dump.lammpstrj"


# ----------------------------------------------------------------------- input files
def test_read_incar_comments_and_semicolons(tmp_path):
    f = tmp_path / "INCAR"
    f.write_text("ENCUT = 520 # cutoff\nISPIN=2; MAGMOM = 2*1.0 ! moments\nLWAVE = .FALSE.\n# NSW = 5\nPOTIM=0.5\n")
    p = read_incar(f)
    assert p == {"ENCUT": 520, "ISPIN": 2, "MAGMOM": "2*1.0", "LWAVE": False, "POTIM": 0.5}


def test_vasp_md_run_info(tmp_path):
    xdatcar = write_vasp_md(tmp_path / "T600", 600)
    info = run_info(load(xdatcar))
    assert (info.code, info.kind, info.ensemble) == ("vasp", "md", "NVT")
    assert info.timestep == 2.0 and info.temperature == (600.0, 600.0)
    assert info.time_per_frame == pytest.approx(0.01)  # NBLOCK 5 x POTIM 2 fs
    assert info.steps == 1000 and info.kpoints == "Gamma 1 1 1"
    assert any(f.endswith("INCAR") for f in info.files)
    assert "VASP md NVT" in repr(info)


def test_vasp_parameters_echoed_in_outcar(vasp_dir):
    info = run_info(load(vasp_dir / "md" / "XDATCAR"))
    assert (info.kind, info.timestep, info.temperature) == ("md", 1.0, (300.0, 300.0))
    assert info.ensemble == "NVE"  # SMASS not echoed: VASP's default -3
    assert any("OUTCAR" in n for n in info.notes)


def test_lammps_log_run_info(lammps_dir):
    p = read_lammps_input(lammps_dir / "thermo" / "log.lammps")
    assert p["units"] == "metal" and p["timestep"] == 0.002 and p["dump_every"] == 10 and p["runs"] == [20, 20]
    info = run_info(load(lammps_dir / "thermo" / "dump.lammpstrj"))
    assert (info.code, info.kind, info.ensemble, info.units) == ("lammps", "md", "NVE", "metal")
    assert info.timestep == pytest.approx(2.0) and info.time_per_frame == pytest.approx(0.02)


def test_lammps_thermostat_and_units(tmp_path):
    info = run_info(load(write_lammps_run(tmp_path / "a", target=450)))
    assert info.ensemble == "NVT" and info.temperature == (450.0, 450.0)
    assert info.time_per_frame == pytest.approx(0.2)  # 100 steps x 2 fs
    info = run_info(load(write_lammps_run(tmp_path / "b", units="real", fix="fix 1 all nve\nfix 2 all langevin {T} {T} 100 4")))
    assert info.ensemble == "NVT" and info.timestep == pytest.approx(0.002)  # real: fs
    info = run_info(load(write_lammps_run(tmp_path / "c", units="lj")))
    assert info.time_per_frame is None and info.parameters["time_per_frame_tau"] == pytest.approx(0.2)


def test_no_input_files_is_unknown_not_wrong(examples_dir=None):
    from ewald import examples

    info = run_info(load(examples.path("Cu_void.lammpstrj")))
    assert info.code == "lammps" and info.kind is None and info.time_per_frame is None
    assert info.notes


# ----------------------------------------------------------------------- statistics
def test_equilibration_detection_finds_the_transient():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 2000)
    x[:300] += np.linspace(20, 0, 300)
    t0, g, neff = detect_equilibration(x)
    assert 200 <= t0 <= 450 and neff > 1000
    assert statistical_inefficiency(rng.normal(size=5000)) == pytest.approx(1.0, abs=0.3)
    ar = np.zeros(5000)
    for k in range(1, 5000):
        ar[k] = 0.9 * ar[k - 1] + rng.normal()
    assert statistical_inefficiency(ar) == pytest.approx(19, rel=0.35)  # (1 + 0.9) / (1 - 0.9)


# ----------------------------------------------------------------------- check_run
def test_check_run_on_a_healthy_vasp_md(tmp_path):
    check = check_run(load(write_vasp_md(tmp_path / "T600", 600)))
    assert check.ok, str(check)
    titles = " | ".join(f.title for f in check.findings)
    assert "SCF converged" in titles and "Equilibrated" in titles and "Temperature at the target" in titles
    assert "Energy drift small" in titles
    assert check.equilibrated is not None and check.to_dict()["ok"]
    assert "Run check" in check._repr_markdown_()


def test_check_run_flags_nelm_and_a_slow_start(tmp_path):
    xd = write_vasp_md(tmp_path / "bad", 600, nelm_hit=(3, 7), relax_steps=1)
    check = check_run(load(xd))
    assert not check.ok
    nelm = [f for f in check.findings if "hit NELM" in f.title]
    assert nelm and nelm[0].level == "warning" and "3, 7" in nelm[0].detail
    eq = [f for f in check.findings if "Equilibrated" in f.title]
    assert eq and check.equilibrated > 0  # the hot start is cut off


def test_check_run_lammps_temperature_errors_and_warnings(tmp_path):
    check = check_run(load(write_lammps_run(tmp_path / "a", target=600, mean=700, error=True, warnings=2)))
    levels = {f.title: f.level for f in check.findings}
    assert levels["LAMMPS stopped with an error"] == "error"
    assert levels["2 LAMMPS warning(s) in the log"] == "warning"
    assert levels["Temperature off the target"] == "warning"


def test_check_run_close_contacts(tmp_path):
    f = tmp_path / "close.xyz"
    f.write_text('3\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
                 "O 1 1 1\nO 1.3 1 1\nO 5 5 5\n")
    check = check_run(load(f))
    bad = [x for x in check.findings if x.level == "error"]
    assert bad and "atoms 0 and 1" in bad[0].detail and "0.30" in bad[0].detail


def test_check_run_relaxation(tmp_path):
    d = tmp_path / "relax"
    d.mkdir()
    (d / "INCAR").write_text("IBRION = 2\nNSW = 50\nEDIFFG = -0.02\n")
    import shutil

    from ewald import examples

    shutil.copy(examples.path("POSCAR.SrTiO3"), d / "POSCAR")
    (d / "OUTCAR").write_text("stuff\n reached required accuracy - stopping structural energy minimisation\n")
    check = check_run(load(d / "POSCAR"))
    assert check.info.kind == "relax"
    assert any(f.title == "Relaxation reached EDIFFG" for f in check.findings)
    (d / "OUTCAR").write_text("stuff\n")
    check = check_run(load(d / "POSCAR"))
    assert any(f.title == "Relaxation did not reach EDIFFG" and f.level == "warning" for f in check.findings)


def test_run_info_over_ssh(server, tmp_path):
    folder = tmp_path / "remote_fs" / "data" / "md"
    write_vasp_md(folder, 900, frames=20)
    pipe = load("hpc:/data/md/XDATCAR")
    info = run_info(pipe)
    assert info.time_per_frame == pytest.approx(0.01) and info.temperature == (900.0, 900.0)
    assert any("INCAR" in f and f.startswith("hpc:") for f in info.files)


# ----------------------------------------------------------------------- several runs
@pytest.fixture
def series(tmp_path):
    for k, t in enumerate((900, 300, 600)):
        write_vasp_md(tmp_path / f"T{t}", t, seed=k)
    return tmp_path


def test_load_runs_labels_and_order(series):
    runs = load_runs(str(series / "T*" / "XDATCAR"))
    assert runs.labels == ["T300", "T600", "T900"]
    assert list(runs.values) == [300, 600, 900]
    assert runs["T600"].pipe is runs[1].pipe
    runs2 = load_runs(str(series / "T*" / "XDATCAR"), label=r"T(\d+)")
    assert runs2.labels == ["300", "600", "900"] and runs2[600.0].label == "600"
    assert runs.pipe is runs[0].pipe and runs.select("T900") is runs[2].pipe and runs.pipe is runs[2].pipe
    with pytest.raises(KeyError):
        runs["T1200"]
    with pytest.raises(FileNotFoundError):
        load_runs(str(series / "nothing*"))


def test_verbs_on_runs_and_arrhenius(series):
    runs = load_runs(str(series / "T*" / "XDATCAR"), label=r"T(\d+)")
    d = msd(runs)
    assert isinstance(d, RunTables) and list(d) == ["300", "600", "900"]
    assert d["600"]["Time"][1] == pytest.approx(0.01)  # ps, from INCAR
    dif = d.attribute("MeanSquaredDisplacement.diffusion_coefficient")
    assert np.all(np.diff(dif) > 0)
    fit = arrhenius(d)
    assert fit.Ea == pytest.approx(0.3, abs=0.05) and fit.r2 > 0.98
    assert "Ea" in repr(fit) and fit._repr_png_()[:4] == b"\x89PNG"
    assert d._repr_png_()[:4] == b"\x89PNG"
    # a second call replaces the steps, one per run, never shared between runs
    msd(runs, fit_range=(0.1, 0.9))
    mods = [r.pipe.modifiers[0] for r in runs]
    assert all(len(r.pipe.modifiers) == 1 for r in runs) and len({id(m) for m in mods}) == 3
    t = runs.attribute("MeanSquaredDisplacement.diffusion_coefficient")
    assert list(t["Run"]) == [300, 600, 900]
    assert np.all(runs.temperatures() == [300, 600, 900])
    assert arrhenius([300, 600], [1e-3, 1e-1]).Ea > 0


def test_runs_and_tables_find_runs_the_ways_the_ai_asked(series, tmp_path):
    """The AI eval's study tasks (2026-10-26): models indexed runs by the number of a "T500" label,
    by the Run object itself and like a dict, and called fit.plot(); each was an error."""
    runs = load_runs(str(series / "T*" / "XDATCAR"))
    assert runs["600"] is runs[1] and runs[600.0] is runs[1] and runs[runs[2]] is runs[2]
    assert runs.keys() == ["T300", "T600", "T900"] and runs.items()[0] == ("T300", runs[0])
    d = msd(runs)
    assert d[runs[1]] is d["T600"] and d["600"] is d["T600"]
    with pytest.raises(KeyError, match="diffusion_coefficient"):
        d.attribute("MeanSquaredDisplacement.D")            # lists the attributes there are
    fit = arrhenius(dict(zip(runs.temperatures(), d.attribute("MeanSquaredDisplacement.diffusion_coefficient"))))
    assert fit.Ea == pytest.approx(arrhenius(d).Ea)
    fit.savefig(tmp_path / "a.png")
    assert (tmp_path / "a.png").read_bytes()[:4] == b"\x89PNG"
    import matplotlib
    matplotlib.use("Agg")
    fit.plot()
    from ewald.notebook import Cell, Notebook

    nb = Notebook([Cell("from ewald import arrhenius\nfit = arrhenius([500, 600, 700], [1, 2, 4])\nfit.plot()\n"
                        "print(round(fit.Ea, 3))")])
    nb.run_all()
    assert [o.kind for o in nb.cells[0].outputs] == ["image", "text"]   # the plot shows under the cell
    with pytest.raises(ValueError, match="different temperatures.*600 K"):
        arrhenius([600, 600, 600], [0.1, 0.2, 0.15])
    with pytest.raises(ValueError, match="2 temperatures for 3"):
        arrhenius([600, 700], [0.1, 0.2, 0.15])


def test_runs_add_copies_steps(series):
    from ewald.modifiers import CoordinationAnalysis

    runs = load_runs(str(series / "T*" / "XDATCAR"))
    mod = CoordinationAnalysis(cutoff=3.0)
    assert runs.add(mod) is mod
    assert all(type(r.pipe.modifiers[0]) is CoordinationAnalysis for r in runs)
    assert len({id(r.pipe.modifiers[0]) for r in runs}) == 3

    @runs.add
    def tag(frame, data):
        data.attributes["Tagged"] = 1

    assert runs.attribute("Tagged")["Tagged"].tolist() == [1, 1, 1]
    g = rdf(runs, cutoff=5.0)
    assert set(g) == set(runs.labels) and g.column() == "g(r)"
    checks = runs.check()
    assert all(c.info.kind == "md" for c in checks.values())
