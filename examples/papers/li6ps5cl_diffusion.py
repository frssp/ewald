"""Li diffusion in the Li6PS5Cl argyrodite from AIMD (kinisi's example data).

Data: VASP XDATCAR from https://github.com/bjmorgan/kinisi (MIT), 416 atoms, 140 frames stored
every 50 steps of 2 fs. Papers: B. J. Morgan, Chem. Mater. 33, 2004 (2021);
A. R. McCluskey, S. W. Coles, B. J. Morgan, J. Chem. Theory Comput. 21, 79 (2025).

kinisi 1.x on the same file (dimension 'xyz', diffusive regime from 3 ps, random_state 42) gives
D* = 1.23(10) x 10^-5 cm^2/s and an MSD of 0.443, 0.707, 0.892 A^2 at 0.1, 0.2, 0.3 ps.

    python examples/papers/li6ps5cl_diffusion.py [outdir]
"""
# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ewald import examples, import_file
from ewald.modifiers import MeanSquaredDisplacement

out = Path(sys.argv[1] if len(sys.argv) > 1 else "papers_out")
out.mkdir(exist_ok=True)
folder = examples.fetch("li6ps5cl-aimd")   # 0.8 MB, cached after the first run

# %% Mean squared displacement, every frame a time origin
pipe = import_file(folder / "XDATCAR.gz")
t_frame = 50 * 2.0e-3                      # ps between stored frames
start, total = 3.0, pipe.num_frames * t_frame
msd = MeanSquaredDisplacement(time_per_frame=t_frame, fit_range=(start / total, 1.0))
pipe.modifiers.append(msd)
data = pipe.compute(0)
table = data.tables["msd"]
D = data.attributes["MeanSquaredDisplacement.diffusion_coefficient.Li"] * 1e-4   # A^2/ps -> cm^2/s

KINISI_D, KINISI_SIGMA = 1.23e-5, 0.10e-5
print(f"MSD(Li) at 0.1/0.2/0.3 ps: {table['MSD Li'][1:4].round(3)} A^2 (kinisi: 0.443 0.707 0.892)")
print(f"D*(Li) = {D:.3g} cm^2/s, kinisi {KINISI_D:.3g} +- {KINISI_SIGMA:.1g} "
      f"({(D - KINISI_D) / KINISI_SIGMA:+.1f} sigma)")

# %% Figure: MSD per species and the Einstein fit over the diffusive regime
fig, ax = plt.subplots(figsize=(4.5, 3.4))
for name in ("Li", "S", "P", "Cl"):
    ax.plot(table["Time"], table[f"MSD {name}"], label=name)
fit = table["Time"] >= start
ax.plot(table["Time"][fit], 6 * D * 1e4 * table["Time"][fit]
        + (table["MSD Li"][fit] - 6 * D * 1e4 * table["Time"][fit]).mean(), "k--", lw=1, label="fit")
ax.axvline(start, color="g", lw=0.8)
ax.set(xlabel="time (ps)", ylabel=r"MSD ($\AA^2$)", title=f"Li$_6$PS$_5$Cl, D* = {D:.2e} cm$^2$/s")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out / "li6ps5cl_msd.png", dpi=150)
print("wrote", out / "li6ps5cl_msd.png")
