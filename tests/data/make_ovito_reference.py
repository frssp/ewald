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
from ovito.modifiers import CommonNeighborAnalysisModifier as CNA
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
                     ("cnaint", CNA(mode=CNA.Mode.IntervalCutoff)), ("ptm", ptm),
                     ("vor", VOR(compute_indices=True))]:
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

np.savez_compressed(os.path.join(os.path.dirname(__file__), "ovito_reference.npz"), **out)
print(f"wrote {len(out)} arrays")
