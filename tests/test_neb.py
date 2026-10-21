"""NEB bands as trajectories (ewald.io.neb) and the NebEnergyProfile modifier.

Fixtures: tests/data/vasp/neb (an EMT vacancy-hop NEB written in VASP layout by make_neb.py)
and tests/data/lammps/neb (a real LAMMPS 2025 ``neb`` run on 7 partitions, in.neb).
"""
import numpy as np
import pytest

from ewald import Pipeline, StaticSource, import_file
from ewald import modifiers as mods
from ewald.data import DataCollection
from ewald.io import NebReader, read_lammps_neb_log
from ewald.io.neb import is_neb


def _outcar_energy(path):
    for line in open(path):
        if "energy  without entropy" in line:
            e = float(line.split()[-1])
    return e


def test_vasp_neb_folder(vasp_dir):
    folder = vasp_dir / "neb"
    assert is_neb(folder) and not is_neb(vasp_dir)
    pipe = import_file(folder)
    assert pipe.num_frames == 5 and type(pipe.modifiers[1]) is mods.NebEnergyProfile
    reader = pipe.source.readers[0]
    assert isinstance(reader, NebReader)
    assert [p.split("/")[-2:] for p in reader.files] == [["00", "OUTCAR"], ["01", "OUTCAR"], ["02", "OUTCAR"],
                                                        ["03", "OUTCAR"], ["04", "POSCAR"]]
    d = pipe.compute(2)
    assert d.attributes["Timestep"] == 2 and d.attributes["NEB.Image"] == 2
    e = np.array([_outcar_energy(folder / f"{i:02d}" / "OUTCAR") for i in range(4)])
    table = d.tables["NEB"]
    np.testing.assert_allclose(table["Energy"][:4], e - e[0])
    assert np.isnan(table["Energy"][4])  # endpoint with only a POSCAR
    assert d.attributes["NEB.Barrier"] == pytest.approx(e.max() - e[0])
    assert d.attributes["NEB.Saddle Image"] == 2
    assert "NEB.Reverse Barrier" not in d.attributes and "Energy" not in pipe.compute(4).attributes
    # reaction coordinate: summed minimum-image distances between images, as ASE computes them
    from ase.geometry import find_mic

    images = [pipe.source.compute(i) for i in range(5)]
    steps = [np.sqrt((find_mic(b.positions - a.positions, a.cell.matrix, True)[0] ** 2).sum())
             for a, b in zip(images, images[1:])]
    np.testing.assert_allclose(table["Reaction Coordinate"], np.concatenate([[0], np.cumsum(steps)]), atol=1e-5)
    # forces along the tangent: zero at the symmetric saddle, opposite on the two flanks
    f = table["Tangent Force"]
    assert abs(f[2]) < 1e-6 and f[1] < -0.5 and f[1] == pytest.approx(-f[3], rel=0.05)
    spline = d.tables["NEB spline"]
    assert len(spline["Energy"]) == 200 and spline["Reaction Coordinate"][-1] == pytest.approx(table["Reaction Coordinate"][3])
    assert d.attributes["NEB.Spline Barrier"] == pytest.approx(d.attributes["NEB.Barrier"], abs=1e-3)
    # the missing endpoint energy can be given
    full = import_file(folder, energies=[None] * 4 + [e[0]]).compute(0)  # None: the OUTCAR's
    assert full.attributes["NEB.Reverse Barrier"] == pytest.approx(full.attributes["NEB.Barrier"])
    assert full.attributes["NEB.Reaction Energy"] == pytest.approx(0)


def test_lammps_neb_run(lammps_dir):
    folder = lammps_dir / "neb"
    log = read_lammps_neb_log(folder / "log.lammps")
    for pipe in (import_file(folder), import_file(folder / "dump.neb.*", format="neb")):
        assert pipe.num_frames == 7
        d = pipe.compute(3)
        table = d.tables["NEB"]
        np.testing.assert_allclose(table["Energy"], log["Energy"] - log["Energy"][0])
        assert d.attributes["Energy"] == log["Energy"][3]
        # LAMMPS prints the normalized reaction coordinate RDi and the path length RDT
        rdt = _last_row(folder)["RDT"]
        np.testing.assert_allclose(table["Reaction Coordinate"], log["Reaction Coordinate"] * rdt, atol=1e-4)
        row = _last_row(folder)
        assert d.attributes["NEB.Barrier"] == pytest.approx(row["EBF"], abs=1e-5)
        assert d.attributes["NEB.Reverse Barrier"] == pytest.approx(row["EBR"], abs=1e-5)
        assert d.attributes["NEB.Saddle Image"] == 3 and not np.isnan(table["Tangent Force"]).any()
    files = pipe.source.readers[0].files
    assert [f.rsplit(".", 1)[1] for f in files] == [str(i) for i in range(1, 8)]
    # a wildcard without format="neb" stays one trajectory over all files
    assert import_file(folder / "dump.neb.*").num_frames == 7 * 3


def _last_row(folder):
    cols = last = None
    for line in open(folder / "log.lammps"):
        tok = line.split()
        if tok[:2] == ["Step", "MaxReplicaForce"]:
            cols = tok
        elif cols and len(tok) == len(cols) and tok[0].isdigit():
            last = dict(zip(cols, map(float, tok)))
    return last


def test_profile_of_ase_neb_trajectory_matches_ase():
    """An optimizer history (several bands in a row): images= takes the final band; path,
    energies and barrier agree with ASE's NEBTools."""
    from ase.build import fcc100, add_adsorbate
    from ase.calculators.emt import EMT
    from ase.constraints import FixAtoms
    from ase.mep import NEB, NEBTools
    from ase.optimize import BFGS
    from ase.utils.forcecurve import fit_images

    slab = fcc100("Al", size=(2, 2, 3), vacuum=4.0)
    add_adsorbate(slab, "Au", 1.7, "hollow")
    slab.set_constraint(FixAtoms(mask=[a.tag > 1 for a in slab]))
    initial, final = slab.copy(), slab.copy()
    final[-1].x += slab.get_cell()[0, 0] / 2
    for end in (initial, final):
        end.calc = EMT()
        BFGS(end, logfile=None).run(fmax=0.05)
    images = [initial] + [initial.copy() for _ in range(3)] + [final]
    for im in images[1:-1]:
        im.calc = EMT()
    band = NEB(images, method="improvedtangent")
    band.interpolate()
    history = []

    def snapshot():
        history.extend(_frame(im) for im in images)

    snapshot()
    BFGS(band, logfile=None).run(fmax=0.05, steps=200)
    snapshot()
    pipe = Pipeline(StaticSource(history))
    pipe.modifiers.append(mods.NebEnergyProfile(images=5))
    d = pipe.compute(len(history) - 1)
    table = d.tables["NEB"]
    np.testing.assert_array_equal(table["Frame"], np.arange(5, 10))
    assert d.attributes["NEB.Image"] == 4 and "NEB.Image" not in pipe.compute(0).attributes
    fit = fit_images(images)
    np.testing.assert_allclose(table["Reaction Coordinate"], fit.path, atol=1e-8)
    np.testing.assert_allclose(table["Energy"], fit.energies, atol=1e-8)
    barrier, delta = NEBTools(images).get_barrier(fit=False)
    assert d.attributes["NEB.Barrier"] == pytest.approx(barrier)
    assert d.attributes["NEB.Reaction Energy"] == pytest.approx(delta)
    assert d.attributes["NEB.Spline Barrier"] == pytest.approx(NEBTools(images).get_barrier()[0], abs=5e-3)
    with pytest.raises(ValueError, match="images=11"):
        Pipeline(StaticSource(history), [mods.NebEnergyProfile(images=11)]).compute(0, errors="raise")


def _frame(atoms):
    d = DataCollection.from_ase(atoms)
    d.attributes["Energy"] = float(atoms.get_potential_energy())
    d.particles["Force"] = atoms.get_forces()
    return d


def test_profile_without_forces_or_energies():
    cell_frames = []
    for i, x in enumerate(np.linspace(0, 2, 5)):
        d = DataCollection(np.array([[x, 0, 0], [5, 5, 5]]), np.array(["H", "H"]))
        d.attributes["PotEng"] = float(np.sin(np.pi * i / 4))  # LAMMPS-style name, picked up
        cell_frames.append(d)
    d = Pipeline(StaticSource(cell_frames), [mods.NebEnergyProfile()]).compute(1)
    np.testing.assert_allclose(d.tables["NEB"]["Reaction Coordinate"], [0, 0.5, 1, 1.5, 2])
    assert np.isnan(d.tables["NEB"]["Tangent Force"]).all()
    assert d.attributes["NEB.Barrier"] == pytest.approx(1) and d.attributes["NEB.Saddle Image"] == 2
    assert d.attributes["NEB.Spline Barrier"] == pytest.approx(1, abs=0.02)  # natural cubic spline
    for f in cell_frames:
        f.attributes.clear()
    with pytest.raises(ValueError, match="no image has an energy"):
        Pipeline(StaticSource(cell_frames), [mods.NebEnergyProfile()]).compute(0, errors="raise")


def test_overlay_frames_ghosts_moving_atoms(vasp_dir, lammps_dir):
    pipe = import_file(vasp_dir / "neb")
    assert [type(m) for m in pipe.modifiers] == [mods.UnwrapTrajectories, mods.NebEnergyProfile,
                                                 mods.OverlayFrames]
    plain = Pipeline(pipe.source, pipe.modifiers[:2]).compute(2)
    d = pipe.compute(2)
    n = plain.count
    frame = d["Overlay Frame"]
    assert (frame[:n] == -1).all() and (d["Transparency"][:n] == 0).all()
    np.testing.assert_array_equal(d.positions[:n], plain.positions)
    ghosts = np.flatnonzero(frame >= 0)
    assert len(ghosts) == d.attributes["OverlayFrames.count"] > 0 and 2 not in frame
    assert np.allclose(d["Transparency"][ghosts], 0.7)
    # each copy sits where its atom is in that image (minimum image next to the current position)

    ids = d["Particle Identifier"]
    for g in ghosts:
        other = pipe.source.compute(int(frame[g]))
        k = int(np.flatnonzero(other["Particle Identifier"] == ids[g])[0]) if "Particle Identifier" in other \
            else int(ids[g]) - 1
        here = d.positions[int(ids[g]) - 1]
        np.testing.assert_allclose(d.positions[g], here + d.cell.minimum_image(other.positions[[k]] - here)[0], atol=1e-9)
        assert np.linalg.norm(d.positions[g] - here) > 0.1
    # the hopping atom of the LAMMPS band leaves a copy at every other image
    lp = import_file(lammps_dir / "neb")
    d = lp.compute(0)
    hop = d["Overlay Frame"][d["Particle Identifier"] == 2]
    assert set(hop) == {-1, 1, 2, 3, 4, 5, 6}
    lp.modifiers[-1].min_displacement = 100.0
    assert lp.compute(0).count == lp.source.compute(0).count  # nothing moved that far: no copies


def _wrapped_band(n=5, hop=2.0):
    """A band whose files wrap atoms into the cell differently, as VASP CONTCARs do: atom 0
    sits on the x = 0 face (at 1e-3 in some images, at L - 1e-3 in others), atom 1 hops."""
    cell = np.diag([6.0, 6.0, 6.0])
    frames = []
    for i in range(n):
        x0 = 1e-3 if i % 2 else 6.0 - 1e-3
        x1 = 3.0 + hop * i / (n - 1)
        d = DataCollection(np.array([[x0, 1, 1], [x1, 3, 3], [1, 4, 4]]), np.array(["Fe"] * 3))
        d.cell.matrix = cell
        d.cell.pbc = np.array([True] * 3)
        d.attributes["Energy"] = float(np.sin(np.pi * i / (n - 1)))
        frames.append(d)
    return frames


def test_reaction_coordinate_uses_minimum_image_like_ase():
    from ase.utils.forcecurve import fit_raw  # ASE NEBTools' path: find_mic between neighbouring images

    frames = _wrapped_band()
    d = Pipeline(StaticSource(frames), [mods.NebEnergyProfile()]).compute(0)
    s = d.tables["NEB"]["Reaction Coordinate"]
    np.testing.assert_allclose(np.diff(s), np.hypot(0.5, 2e-3), rtol=1e-9)  # not the 6 A wrap jump
    pos = [f.positions for f in frames]
    path = fit_raw([f.attributes["Energy"] for f in frames], [np.zeros((3, 3))] * len(frames), pos,
                   cell=frames[0].cell.matrix, pbc=[True] * 3).path
    np.testing.assert_allclose(s, path, atol=1e-12)


def test_neb_band_is_unwrapped_for_display(tmp_path):
    import ase.io

    from ewald.render import scene_bonds

    frames = _wrapped_band()
    for i, f in enumerate(frames):
        (tmp_path / f"{i:02d}").mkdir()
        ase.io.write(tmp_path / f"{i:02d}" / "POSCAR", f.to_ase(), format="vasp")
    pipe = import_file(tmp_path, energies=[f.attributes["Energy"] for f in frames])
    assert type(pipe.modifiers[0]) is mods.UnwrapTrajectories
    pipe.modifiers[-1].enabled = False
    x = np.array([pipe.compute(i).positions[0, 0] for i in range(5)])
    np.testing.assert_allclose(x, [5.999, 6.001, 5.999, 6.001, 5.999], atol=1e-6)  # no 6 A jumps across the face
    pipe.modifiers[-1].enabled = True
    d = pipe.compute(2)
    ghosts = np.flatnonzero(d["Overlay Frame"] >= 0)
    assert len(ghosts) == 4  # only the hopping atom moved: one copy per other image, none for atom 0
    b = scene_bonds(d, cutoff=3.5)
    assert b.count and (b.topology < 3).all()  # the see-through copies get no bonds


def test_neb_run_of_finds_the_run_of_one_image(vasp_dir, lammps_dir, tmp_path):
    from ewald.io import neb_run_of

    run = str((vasp_dir / "neb").resolve())
    assert neb_run_of(vasp_dir / "neb" / "02" / "OUTCAR") == run
    assert neb_run_of(vasp_dir / "neb" / "00" / "POSCAR") == run
    assert neb_run_of(lammps_dir / "neb" / "dump.neb.3") == str((lammps_dir / "neb").resolve())
    assert neb_run_of(lammps_dir / "neb" / "log.lammps") is None
    assert neb_run_of(vasp_dir / "neb") is None  # the folder itself opens as NEB already
    (tmp_path / "00").mkdir()
    (tmp_path / "00" / "POSCAR").write_text("x")
    assert neb_run_of(tmp_path / "00" / "POSCAR") is None  # one numbered folder is not a band
