"""Regenerate tests/data/ovito_reference.npz with the official `ovito` Python module.

    pip install ovito ase
    python tests/data/make_ovito_reference.py

The file stores the input systems together with OVITO's results, so the test
suite compares against OVITO without needing OVITO installed.
"""
import os

import numpy as np
from ase.build import bulk
from ase.cluster import Icosahedron
from ovito.data import DataCollection, Particles, SimulationCell
from ovito.modifiers import AcklandJonesModifier as AJ
from ovito.modifiers import AtomicStrainModifier as STRAIN
from ovito.modifiers import CentroSymmetryModifier as CSP
from ovito.modifiers import ClusterAnalysisModifier as CLU
from ovito.modifiers import ChillPlusModifier as CHILL
from ovito.modifiers import CommonNeighborAnalysisModifier as CNA
from ovito.modifiers import IdentifyDiamondModifier as IDD
from ovito.modifiers import PolyhedralTemplateMatchingModifier as PTM
from ovito.modifiers import VoronoiAnalysisModifier as VOR
from ovito.modifiers import WignerSeitzAnalysisModifier as WS
from ovito.pipeline import Pipeline, StaticSource

rng = np.random.default_rng(1)
systems = {}


def add(name, atoms, noise=0.0, pbc=True):
    systems[name] = dict(pos=atoms.get_positions() + rng.normal(0, noise, (len(atoms), 3)),
                         cell=np.array(atoms.cell), pbc=np.array([pbc] * 3),
                         types=atoms.get_atomic_numbers().astype(np.int32))


add("fcc", bulk("Cu", "fcc", a=3.615, cubic=True).repeat((5, 5, 5)), 0.08)
add("hcp", bulk("Mg", "hcp", a=3.21, c=5.21).repeat((6, 6, 4)), 0.08)
add("bcc", bulk("Fe", "bcc", a=2.87, cubic=True).repeat((6, 6, 6)), 0.06)
add("fcc_hot", bulk("Cu", "fcc", a=3.615, cubic=True).repeat((5, 5, 5)), 0.22)
add("fcc_triclinic", bulk("Cu", "fcc", a=3.615).repeat((7, 7, 7)), 0.1)
add("diamond", bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3)), 0.05)
l12 = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((4, 4, 4))
f = l12.get_scaled_positions() * 4
z = l12.get_atomic_numbers()
z[np.all(np.abs(f - np.round(f)) < 1e-6, axis=1)] = 79
l12.set_atomic_numbers(z)
add("l12", l12, 0.03)
ico = Icosahedron("Cu", 4)
ico.center(vacuum=2.0)
add("ico_cluster", ico, 0.02, pbc=False)
add("disordered", bulk("Cu", "fcc", a=3.615, cubic=True).repeat((4, 4, 4)), 0.5)
add("hexdiamond", bulk("SiSi", "wurtzite", a=3.84, c=6.27, u=0.375).repeat((4, 4, 3)), 0.04)
vac = bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3))
del vac[[0, 50, 101]]
add("diamond_vac", vac, 0.03)


def make(pos, cell, pbc, types):
    dc = DataCollection()
    m = np.zeros((3, 4))
    m[:, :3] = cell.T
    sc = SimulationCell()
    sc[...] = m
    sc.pbc = tuple(bool(b) for b in pbc)
    dc.objects.append(sc)
    p = Particles()
    p.create_property("Position", data=pos)
    p.create_property("Particle Type", data=types)
    dc.objects.append(p)
    return dc


def run(dc, mod):
    pipe = Pipeline(source=StaticSource(data=dc))
    pipe.modifiers.append(mod)
    return pipe.compute()


out = {}
for name, s in systems.items():
    for key, val in s.items():
        out[f"{name}/{key}"] = val
    dc = make(s["pos"], s["cell"], s["pbc"], s["types"])
    ptm = PTM(output_orientation=True, output_interatomic_distance=True, output_deformation_gradient=True,
              output_ordering=True)
    for st in ptm.structures:
        st.enabled = True
    for tag, mod in [("cna", CNA()), ("cnafix", CNA(mode=CNA.Mode.FixedCutoff, cutoff=3.05)),
                     ("cnaint", CNA(mode=CNA.Mode.IntervalCutoff)), ("ptm", ptm), ("aj", AJ()),
                     ("csp", CSP()), ("cspmwm", CSP(mode=CSP.Mode.Matching, num_neighbors=8)),
                     ("vor", VOR(compute_indices=True)), ("diamondid", IDD())]:
        d = run(dc, mod)
        for prop in d.particles.keys():
            if prop not in ("Position", "Particle Type", "Color"):
                out[f"{name}/{tag}/{prop}"] = np.array(d.particles[prop])

# Wigner-Seitz: thermal noise, one Frenkel pair, one atom shifted by a cell vector.
a = bulk("Fe", "bcc", a=2.87).repeat((6, 6, 6))
ref = a.get_positions()
cur = ref + rng.normal(0, 0.15, ref.shape)
cur[5] = ref[17] + np.array([0.9, 0.9, 0.0])
cur[40] += a.cell[0]
types = np.where(np.arange(len(a)) % 3 == 0, 1, 2).astype(np.int32)
out.update({"ws/ref": ref, "ws/cur": cur, "ws/cell": np.array(a.cell), "ws/types": types})
for tag, kw in [("site", {}), ("cur", dict(output_displaced=True)), ("pertype", dict(per_type_occupancies=True))]:
    mod = WS(**kw)
    mod.reference = StaticSource(data=make(ref, np.array(a.cell), [True] * 3, types))
    d = run(make(cur, np.array(a.cell), [True] * 3, types), mod)
    out[f"ws/{tag}/Occupancy"] = np.array(d.particles["Occupancy"])
    out[f"ws/{tag}/counts"] = np.array([d.attributes["WignerSeitz.vacancy_count"],
                                        d.attributes["WignerSeitz.interstitial_count"]])
    if tag == "cur":
        out["ws/cur/Site Index"] = np.array(d.particles["Site Index"])

# Cluster analysis: compact random blobs in a periodic box, several crossing the boundary.
L = 40.0
centers = rng.uniform(0, L, (60, 3))
blobs = [c + rng.normal(0, 1.2, (rng.integers(1, 30), 3)) for c in centers]
cpos = np.concatenate(blobs) % L
cmass = rng.uniform(1, 3, len(cpos))
ccell = np.diag([L, L, L])
out.update({"cluster/pos": cpos, "cluster/cell": ccell, "cluster/mass": cmass})
for tag, kw in [("plain", {}), ("sorted", dict(sort_by_size=True, compute_com=True, compute_gyration=True,
                                                 unwrap_particles=True))]:
    dc = make(cpos, ccell, [True] * 3, np.ones(len(cpos), np.int32))
    dc.particles_.create_property("Mass", data=cmass)
    d = run(dc, CLU(cutoff=2.0, **kw))
    out[f"cluster/{tag}/Cluster"] = np.array(d.particles["Cluster"])
    out[f"cluster/{tag}/Position"] = np.array(d.particles["Position"])
    t = d.tables["clusters"]
    for key in t.keys():
        out[f"cluster/{tag}/table/{key}"] = np.array(t[key])
    out[f"cluster/{tag}/count"] = np.array(d.attributes["ClusterAnalysis.cluster_count"])

# Chill+: oxygen sublattices of ice Ih, ice Ic and sI hydrate; noisy copies and an ice/disordered slab.
from ase.spacegroup import crystal  # noqa: E402

doo = 2.76
ice_h = bulk("OO", "wurtzite", a=doo * np.sqrt(8 / 3), c=doo * 8 / 3, u=0.375).repeat((4, 4, 3))
ice_c = bulk("O", "diamond", a=doo * 4 / np.sqrt(3), cubic=True).repeat((3, 3, 3))
hyd = crystal(["O", "O", "O"], [(0.25, 0, 0.5), (0.1837, 0.1837, 0.1837), (0, 0.3077, 0.1172)],
              spacegroup=223, cellpar=[11.88, 11.88, 11.88, 90, 90, 90]).repeat((2, 2, 2))
slab = ice_c.copy()
f = slab.get_scaled_positions()
melt = f[:, 2] > 0.5
p_ = slab.get_positions()
p_[melt] += rng.normal(0, 0.45, (melt.sum(), 3))
slab.set_positions(p_)
chill = {"iceh": (ice_h, 0.0), "iceh_hot": (ice_h, 0.15), "icec": (ice_c, 0.1), "hydrate": (hyd, 0.05),
         "slab": (slab, 0.0)}
for tag, (at, noise) in chill.items():
    cpos_ = at.get_positions() + rng.normal(0, noise, (len(at), 3))
    out[f"chill/{tag}/pos"] = cpos_
    out[f"chill/{tag}/cell"] = np.array(at.cell)
    d = run(make(cpos_, np.array(at.cell), [True] * 3, np.ones(len(at), np.int32)), CHILL())
    out[f"chill/{tag}/Structure Type"] = np.array(d.particles["Structure Type"])

# Atomic strain: noisy fcc reference, current = homogeneous deformation of cell and atoms + local noise.
sref = systems["fcc_hot"]
Fh = np.array([[1.03, 0.02, 0.0], [0.0, 0.98, 0.01], [0.01, 0.0, 1.01]])
scur_cell = sref["cell"] @ Fh.T
scur = sref["pos"] @ Fh.T + rng.normal(0, 0.05, sref["pos"].shape)
out.update({"strain/cur": scur, "strain/cell": scur_cell})
for tag, kw in [("plain", {}), ("toref", dict(affine_mapping=STRAIN.AffineMapping.ToReference)),
                ("small", dict(cutoff=2.2))]:
    mod = STRAIN(cutoff=kw.pop("cutoff", 3.0), output_deformation_gradients=True, output_strain_tensors=True,
                 output_nonaffine_squared_displacements=True, output_rotations=True,
                 output_stretch_tensors=True, **kw)
    mod.reference = StaticSource(data=make(sref["pos"], sref["cell"], [True] * 3, sref["types"]))
    d = run(make(scur, scur_cell, [True] * 3, sref["types"]), mod)
    for prop in d.particles.keys():
        if prop not in ("Position", "Particle Type", "Color"):
            out[f"strain/{tag}/{prop}"] = np.array(d.particles[prop])
    out[f"strain/{tag}/invalid"] = np.array(d.attributes["AtomicStrain.invalid_particle_count"])

np.savez_compressed(os.path.join(os.path.dirname(__file__), "ovito_reference.npz"), **out)
print(f"wrote {len(out)} arrays")
