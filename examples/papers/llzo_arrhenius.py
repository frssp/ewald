"""Arrhenius plot of Li diffusion in LLZO (Li7La3Zr2O12) from MD at 500-800 K (kinisi's example data).

Data: GROMACS XTC + GRO from https://github.com/bjmorgan/kinisi (MIT), 1536 atoms, 1000 frames of
50.8 fs each (50.8 ps) per temperature; reading XTC needs MDAnalysis (pip install "ewald[formats]").
Paper: A. R. McCluskey, S. W. Coles, B. J. Morgan, J. Chem. Theory Comput. 21, 79 (2025).

kinisi 1.x (diffusion from 10 ps, Bayesian fit of the covariance-weighted MSD) gives D = 2.17, 2.84,
4.26 and 6.49 x 10^-6 cm^2/s at 500, 600, 700 and 800 K, i.e. Ea = 0.12 eV. Its MSD agrees with
ewald's to about 2 % at every lag, yet its D values are 0.6x the slope of an ordinary straight-line
fit from 10 ps to the end (what this script does); we have not traced that difference inside
kinisi's covariance-weighted fit. The activation energies agree.

    python examples/papers/llzo_arrhenius.py [outdir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ewald import examples, import_file
from ewald.modifiers import IonicConductivity, MeanSquaredDisplacement, SelectType

out = Path(sys.argv[1] if len(sys.argv) > 1 else "papers_out")
out.mkdir(exist_ok=True)
folder = examples.fetch("llzo-arrhenius")  # 29 MB, cached after the first run

T = np.array([500, 600, 700, 800])
KINISI_D = np.array([2.17, 2.84, 4.26, 6.49]) * 1e-6
KB = 8.617333e-5                           # eV/K
t_frame = 100 * 5.079648e-4                # ps between stored frames
start = 10.0                               # ps, start of the diffusive regime

# %% D(T) from the Li MSD, and the Nernst-Einstein conductivity of Li
D, sigma = [], []
for temp in T:
    pipe = import_file(folder / f"traj_{temp}.xtc", topology=folder / f"traj_{temp}.gro")
    fit = (start / (pipe.num_frames * t_frame), 1.0)
    pipe.modifiers.append(MeanSquaredDisplacement(time_per_frame=t_frame, fit_range=fit))
    pipe.modifiers.append(SelectType(types={"Li"}))
    pipe.modifiers.append(IonicConductivity(temperature=float(temp), time_per_frame=t_frame, fit_range=fit,
                                            only_selected=True))
    data = pipe.compute(0)
    D.append(data.attributes["MeanSquaredDisplacement.diffusion_coefficient.Li"] * 1e-4)
    sigma.append(data.attributes["IonicConductivity.nernst_einstein_conductivity"])
    print(f"{temp} K: D(Li) = {D[-1]:.3g} cm^2/s (kinisi {KINISI_D[T == temp][0]:.3g}), "
          f"Nernst-Einstein sigma(Li) = {sigma[-1]:.3g} S/cm")
D = np.array(D)

# %% Activation energy from ln D = ln D0 - Ea / kB T
def arrhenius(d):
    slope, icpt = np.polyfit(1.0 / T, np.log(d), 1)
    return -slope * KB, np.exp(icpt)

ea, d0 = arrhenius(D)
ea_k, _ = arrhenius(KINISI_D)
print(f"Ea = {ea:.3f} eV (kinisi's D values: {ea_k:.3f} eV), D0 = {d0:.2e} cm^2/s")

fig, ax = plt.subplots(figsize=(4.5, 3.4))
ax.semilogy(1000 / T, D, "o", label=f"ewald, $E_a$ = {ea:.3f} eV")
ax.semilogy(1000 / T, KINISI_D, "s", mfc="none", label=f"kinisi, $E_a$ = {ea_k:.3f} eV")
x = np.linspace(1000 / T.max(), 1000 / T.min(), 50)
ax.semilogy(x, d0 * np.exp(-ea / (KB * 1000 / x)), "k--", lw=1)
ax.set(xlabel=r"1000/T (K$^{-1}$)", ylabel=r"D(Li) (cm$^2$/s)", title="LLZO")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out / "llzo_arrhenius.png", dpi=150)
print("wrote", out / "llzo_arrhenius.png")
