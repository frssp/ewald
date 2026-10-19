"""Regenerate tests/data/crystal_reference.npz with the official `ovito` Python module.

    pip install ovito ase
    python tests/data/make_crystal_reference.py

Inputs and OVITO results for the crystal-analysis modifiers: Elastic strain calculation,
Grain segmentation and Identify fcc planar faults. The test suite compares against these
arrays without needing OVITO installed.
"""
import os

import numpy as np
from ase.build import bulk
from ovito.data import DataCollection, Particles, SimulationCell
from ovito.modifiers import ElasticStrainModifier as ES
from ovito.modifiers import GrainSegmentationModifier as GS
from ovito.modifiers import IdentifyFCCPlanarFaultsModifier as PF
from ovito.modifiers import PolyhedralTemplateMatchingModifier as PTM
from ovito.pipeline import Pipeline, StaticSource

rng = np.random.default_rng(7)
systems = {}


def add(name, pos, cell, noise, types=None):
    pos = pos + rng.normal(0, noise, pos.shape)
    systems[name] = dict(pos=pos, cell=np.asarray(cell, float),
                         types=np.ones(len(pos), np.int32) if types is None else types)


def random_rotation():
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def polycrystal(unit, length, ngrains, mindist):
    """Periodic Voronoi polycrystal: each grain is a randomly rotated copy of `unit` (an ase Atoms)."""
    L = length
    seeds = rng.uniform(0, L, (ngrains, 3))
    reach = int(np.ceil(L * np.sqrt(3) / min(unit.cell.lengths()))) + 1
    base = unit.repeat((2 * reach,) * 3)
    lattice = base.get_positions() - unit.cell.sum(0) * reach
    pos = []
    for g, s in enumerate(seeds):
        p = lattice @ random_rotation().T + s
        p = p[np.all(np.abs(p - s) < L / 2, 1)]
        f = p % L
        d = f[:, None, :] - seeds[None]
        d -= L * np.round(d / L)
        own = np.argmin((d ** 2).sum(-1), 1) == g
        pos.append(f[own])
    pos = np.concatenate(pos)
    # Drop one atom of each pair closer than mindist at the grain boundaries.
    from scipy.spatial import cKDTree
    tree = cKDTree(pos, boxsize=L)
    drop = set()
    for i, j in sorted(tree.query_pairs(mindist)):
        if i not in drop and j not in drop:
            drop.add(j)
    keep = np.setdiff1d(np.arange(len(pos)), sorted(drop))
    return pos[keep], np.eye(3) * L


def stacking(seq, a_nn, nx, ny):
    """fcc/hcp close-packed layers stacked along z in the order given by `seq` ('A', 'B', 'C')."""
    h = a_nn * np.sqrt(2.0 / 3.0)
    shift = {"A": (0.0, 0.0), "B": (a_nn / 2, a_nn * np.sqrt(3) / 6), "C": (0.0, a_nn * np.sqrt(3) / 3)}
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    layer = np.concatenate([np.stack([ix * a_nn, iy * a_nn * np.sqrt(3)], -1).reshape(-1, 2),
                            np.stack([ix * a_nn + a_nn / 2, (iy + 0.5) * a_nn * np.sqrt(3)], -1).reshape(-1, 2)])
    pos = []
    for k, s in enumerate(seq):
        xy = (layer + shift[s]) % [nx * a_nn, ny * a_nn * np.sqrt(3)]
        pos.append(np.column_stack([xy, np.full(len(xy), k * h)]))
    cell = np.diag([nx * a_nn, ny * a_nn * np.sqrt(3), len(seq) * h])
    return np.concatenate(pos), cell


def deform(atoms_pos, cell, F):
    return atoms_pos @ F.T, cell @ F.T


F = np.array([[1.012, 0.004, -0.002], [0.0, 0.993, 0.003], [0.001, -0.002, 1.006]])
for name, at, noise in [
        ("fcc", bulk("Cu", "fcc", a=3.615, cubic=True).repeat((5, 5, 5)), 0.04),
        ("bcc", bulk("Fe", "bcc", a=2.87, cubic=True).repeat((6, 6, 6)), 0.04),
        ("hcp", bulk("Mg", "hcp", a=3.21, c=5.21).repeat((6, 6, 4)), 0.04),
        ("diamond", bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3)), 0.03),
        ("hexdiamond", bulk("SiSi", "wurtzite", a=3.84, c=6.27, u=0.375).repeat((4, 4, 3)), 0.03)]:
    p, c = deform(at.get_positions(), np.array(at.cell), F)
    add(name, p, c, noise)

# Stacking faults in fcc, rotated off the axes: an intrinsic fault (2 hcp layers), a twin boundary
# (1 layer) and a 3-layer hcp slab (the periodic wrap adds no fault).
seq = "ABCABCAB" + "ABCABCABC" + "BACBACBACB" + "ABABCABC"
p, c = stacking(seq, 2.556, 8, 5)
R = random_rotation()
add("faults", p @ R.T, c @ R.T, 0.04)
add("poly_fcc", *polycrystal(bulk("Cu", "fcc", a=3.615, cubic=True), 46.0, 3, 2.2), 0.05)
add("poly_bcc", *polycrystal(bulk("Fe", "bcc", a=2.87, cubic=True), 40.0, 3, 2.0), 0.04)


def embedded_grain(unit, reps, radius, center, mindist):
    """A randomly rotated spherical grain inside a single-crystal matrix, crossing the periodic boundary."""
    matrix = unit.repeat(reps)
    L = np.diag(matrix.cell)
    pos = matrix.get_positions()
    d = pos - center
    d -= L * np.round(d / L)
    pos = pos[(d ** 2).sum(1) > radius ** 2]
    reach = int(np.ceil(radius / min(unit.cell.lengths()))) + 1
    ball = unit.repeat((2 * reach,) * 3).get_positions() - unit.cell.sum(0) * reach
    ball = ball @ random_rotation().T
    ball = ball[(ball ** 2).sum(1) <= radius ** 2] + center
    pos = np.concatenate([pos, ball % L])
    from scipy.spatial import cKDTree
    drop = set()
    for i, j in sorted(cKDTree(pos, boxsize=L).query_pairs(mindist)):
        if i not in drop and j not in drop:
            drop.add(j)
    return pos[np.setdiff1d(np.arange(len(pos)), sorted(drop))], np.diag(L)


add("sphere_fcc", *embedded_grain(bulk("Cu", "fcc", a=3.615, cubic=True), (10, 10, 10), 11.0,
                                  np.array([2.0, 3.0, 34.0]), 2.2), 0.04)
add("poly_diamond", *polycrystal(bulk("Si", "diamond", a=5.43, cubic=True), 44.0, 3, 2.0), 0.03)


def make(s):
    dc = DataCollection()
    m = np.zeros((3, 4))
    m[:, :3] = s["cell"].T
    sc = SimulationCell()
    sc[...] = m
    sc.pbc = (True, True, True)
    dc.objects.append(sc)
    p = Particles()
    p.create_property("Position", data=s["pos"])
    p.create_property("Particle Type", data=s["types"])
    dc.objects.append(p)
    return dc


def run(s, *mods):
    pipe = Pipeline(source=StaticSource(data=make(s)))
    pipe.modifiers.extend(mods)
    return pipe.compute()


def ptm(**kw):
    m = PTM(output_orientation=True, output_interatomic_distance=True, **kw)
    m.structures[PTM.Type.CUBIC_DIAMOND].enabled = True
    m.structures[PTM.Type.HEX_DIAMOND].enabled = True
    return m


out = {}
for name, s in systems.items():
    for key, val in s.items():
        out[f"{name}/{key}"] = val

# Elastic strain.
lattice = {"fcc": (ES.Lattice.FCC, 3.615, 1.0), "faults": (ES.Lattice.FCC, 2.556 * np.sqrt(2), 1.0),
           "bcc": (ES.Lattice.BCC, 2.87, 1.0), "hcp": (ES.Lattice.HCP, 3.21, 5.21 / 3.21),
           "diamond": (ES.Lattice.CubicDiamond, 5.43, 1.0),
           "hexdiamond": (ES.Lattice.HexagonalDiamond, 3.84, 6.27 / 3.84),
           "poly_fcc": (ES.Lattice.FCC, 3.615, 1.0)}
for name, (lat, a0, ca) in lattice.items():
    small = len(systems[name]["pos"]) < 1000
    for tag, kw in [("es", {}), ("esmat", dict(push_strain_tensors_forward=False))] if small else [("es", {})]:
        d = run(systems[name], ES(input_crystal_structure=lat, lattice_constant=a0, axial_ratio=ca,
                                  calculate_deformation_gradients=True, **kw))
        props = ["Structure Type", "Cluster", "Volumetric Strain"]
        if name != "poly_fcc":
            props.append("Elastic Strain")
        if small:
            props.append("Elastic Deformation Gradient")
        for prop in props:
            out[f"{name}/{tag}/{prop}"] = np.array(d.particles[prop])

# Grain segmentation (after PTM with orientations).
variants = [("gs", {}), ("gsmst", dict(algorithm=GS.Algorithm.MinimumSpanningTree, merging_threshold=3.0)),
            ("gsman", dict(algorithm=GS.Algorithm.GraphClusteringManual, merging_threshold=8.5,
                           min_grain_size=50, orphan_adoption=False)),
            ("gsnosf", dict(handle_stacking_faults=False))]
for name in ("poly_fcc", "poly_bcc", "poly_diamond", "faults", "hcp", "sphere_fcc"):
    for tag, kw in variants:
        d = run(systems[name], ptm(), GS(**kw))
        out[f"{name}/{tag}/Grain"] = np.array(d.particles["Grain"])
        for key in d.tables["grains"].keys():
            out[f"{name}/{tag}/grains/{key}"] = np.array(d.tables["grains"][key])
        for tab in ("grains-merge", "grains-log") if tag == "gs" else ("grains-merge",):
            if tab in d.tables:
                for key in d.tables[tab].keys():
                    out[f"{name}/{tag}/{tab}/{key}"] = np.array(d.tables[tab][key])
        out[f"{name}/{tag}/count"] = np.array(d.attributes["GrainSegmentation.grain_count"])
        if "GrainSegmentation.auto_merge_threshold" in d.attributes:
            out[f"{name}/{tag}/threshold"] = np.array(d.attributes["GrainSegmentation.auto_merge_threshold"])

# Planar faults.
d = run(systems["faults"], ptm(), PF())
out["faults/pf/Planar Fault Type"] = np.array(d.particles["Planar Fault Type"])
for key in d.tables["planar_faults"].keys():
    out[f"faults/pf/table/{key}"] = np.array(d.tables["planar_faults"][key])

np.savez_compressed(os.path.join(os.path.dirname(__file__), "crystal_reference.npz"), **out)
print(f"wrote {len(out)} arrays")
for name in systems:
    print(name, len(systems[name]["pos"]),
          {k.split("/")[1]: int(v) for k, v in out.items() if k.startswith(name) and k.endswith("/count")},
          {k.split("/")[1]: float(v) for k, v in out.items() if k.startswith(name) and k.endswith("/threshold")})
