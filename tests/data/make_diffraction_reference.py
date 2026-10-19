"""Regenerate tests/data/diffraction_reference.npz: pymatgen powder patterns (XRDCalculator,
NDCalculator) and freud static structure factors (StaticStructureFactorDirect / Debye).

    pip install pymatgen freud-analysis ase
    python tests/data/make_diffraction_reference.py

Inputs are stored next to the results, so the test suite needs neither pymatgen nor freud.
"""
import os

import freud
import numpy as np
from ase.build import bulk
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure

rng = np.random.default_rng(11)
out = {}


def store(name, atoms):
    out[f"{name}.cell"] = np.asarray(atoms.cell[:], float)
    out[f"{name}.positions"] = atoms.positions.copy()
    out[f"{name}.types"] = np.asarray(atoms.get_chemical_symbols())


def structure(name):
    return Structure(Lattice(out[f"{name}.cell"]), list(out[f"{name}.types"]), out[f"{name}.positions"],
                     coords_are_cartesian=True)


crystals = {
    "cu": bulk("Cu", "fcc", a=3.615, cubic=True),
    "nacl": bulk("NaCl", "rocksalt", a=5.64),
    "zno": bulk("ZnO", "wurtzite", a=3.25, c=5.21),
}
# a triclinic cell with random positions: exercises skewed reciprocal lattices
tri = bulk("Si", "diamond", a=5.43).repeat((1, 1, 2))
tri.set_cell(tri.cell[:] + rng.normal(0, 0.3, (3, 3)), scale_atoms=True)
tri.positions += rng.normal(0, 0.15, tri.positions.shape)
tri.set_chemical_symbols(["Si", "O"] * (len(tri) // 2))
crystals["tri"] = tri
for name, atoms in crystals.items():
    store(name, atoms)

cases = {"cu_xrd": ("cu", "xray", "CuKa", None), "cu_dw": ("cu", "xray", "MoKa", {"Cu": 0.55}),
         "nacl_xrd": ("nacl", "xray", "CuKa", None), "nacl_nd": ("nacl", "neutron", 1.54, None),
         "zno_xrd": ("zno", "xray", "CuKa", None), "zno_nd": ("zno", "neutron", 1.8, None),
         "tri_xrd": ("tri", "xray", "CuKa", None)}
for case, (name, radiation, lam, dw) in cases.items():
    calc = (XRDCalculator if radiation == "xray" else NDCalculator)(wavelength=lam, debye_waller_factors=dw)
    pat = calc.get_pattern(structure(name), two_theta_range=(5, 120))
    out[f"{case}.2theta"] = np.asarray(pat.x)
    out[f"{case}.intensity"] = np.asarray(pat.y)
    out[f"{case}.d"] = np.asarray(pat.d_hkls)

# a liquid-like configuration (random sequential addition) for S(q)
L, n = 14.0, 220
pts = []
while len(pts) < n:
    p = rng.uniform(0, L, 3)
    if all(np.linalg.norm((p - q + L / 2) % L - L / 2) > 1.9 for q in pts):
        pts.append(p)
liq = np.asarray(pts)
out["liquid.positions"] = liq
out["liquid.cell"] = np.eye(3) * L
box = freud.box.Box.cube(L)
centered = box.wrap(liq - L / 2)
direct = freud.diffraction.StaticStructureFactorDirect(bins=60, k_max=8.0, k_min=0.5, num_sampled_k_points=10**9)
direct.compute((box, centered), reset=True)
out["direct.k"] = direct.bin_centers
out["direct.k_points"] = np.asarray(direct.k_points)   # freud samples a subset of the lattice
out["direct.S"] = direct.S_k
cluster = liq[np.linalg.norm(liq - L / 2, axis=1) < 6.0]
out["cluster.positions"] = cluster
big = freud.box.Box.cube(200.0)
debye = freud.diffraction.StaticStructureFactorDebye(num_k_values=80, k_max=10.0, k_min=0.5)
debye.compute((big, cluster - cluster.mean(0)), reset=True)
out["debye.k"] = debye.k_values
out["debye.S"] = debye.S_k
np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffraction_reference.npz"), **out)
print("wrote", len(out), "arrays")
