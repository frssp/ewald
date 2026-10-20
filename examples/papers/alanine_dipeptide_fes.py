"""Ramachandran free energy surface of alanine dipeptide from metadynamics hills (PLUMED-NEST 20.023).

Needs the ewald-plumed plugin (Plugins panel, or pip install
'git+https://github.com/frssp/ewald-plugins#subdirectory=packages/ewald-plumed'); the PLUMED library
itself is not needed.

Data: https://github.com/spiwokv/metadynminerdata (no license stated; downloaded, not
redistributed), 2D well-tempered metadynamics of alanine dipeptide in water on phi and psi
(300 K, bias factor 10, 30,000 hills, kJ/mol).
Paper: D. Trapl, V. Spiwok, "Analysis of the Results of Metadynamics Simulations by metadynminer
and metadynminer3d", The R Journal 14(3) (2022), doi:10.32614/RJ-2022-057.

The surface shows the familiar basins: beta/C5 and C7eq on the left (phi < 0, psi > 0),
alpha_R below them, and the higher left-handed alpha_L/C7ax basins at phi > 0.

    python examples/papers/alanine_dipeptide_fes.py [outdir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ewald import examples
from ewald_plumed.plumed import fes_from_hills

out = Path(sys.argv[1] if len(sys.argv) > 1 else "papers_out")
out.mkdir(exist_ok=True)
folder = examples.fetch("alanine-dipeptide-metad")   # 4.9 MB

# %% Sum the hills on a periodic 100 x 100 grid (phi, psi in -pi..pi from the HILLS header)
result = fes_from_hills(folder / "HILLS", bins=100)
phi, psi = (np.degrees(g) for g in result["grid"])
fes = result["fes"]


def basin(phi_range, psi_range):
    """Lowest free energy and its position inside a (phi, psi) window in degrees."""
    m = np.ix_((phi >= phi_range[0]) & (phi <= phi_range[1]), (psi >= psi_range[0]) & (psi <= psi_range[1]))
    sub = fes[m]
    k = np.unravel_index(np.argmin(sub), sub.shape)
    return sub[k], phi[m[0].ravel()][k[0]], psi[m[1].ravel()][k[1]]


for name, window in (("beta / C5", ((-180, -100), (100, 180))), ("C7eq / PII", ((-100, -50), (50, 180))),
                     ("alpha_R", ((-120, -30), (-80, 0))), ("alpha_L / C7ax", ((30, 90), (-90, 90)))):
    f, p, q = basin(*window)
    print(f"{name:>15}: {f:5.1f} kJ/mol at phi = {p:6.1f}, psi = {q:6.1f}")

fig, ax = plt.subplots(figsize=(4.6, 3.8))
cs = ax.contourf(phi, psi, fes.T, levels=np.arange(0, 60, 4), cmap="viridis")
ax.set(xlabel=r"$\phi$ (deg)", ylabel=r"$\psi$ (deg)", title="alanine dipeptide")
fig.colorbar(cs, label="free energy (kJ/mol)")
fig.tight_layout()
fig.savefig(out / "alanine_dipeptide_fes.png", dpi=150)
print("wrote", out / "alanine_dipeptide_fes.png")
