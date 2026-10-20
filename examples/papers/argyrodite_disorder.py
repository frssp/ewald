"""Li diffusion against anion site disorder in Li6PS5I and Li6PS5Cl (AIMD at 500 K).

Data: https://github.com/bjmorgan/data_argyrodite_disorder (MIT), 37 VASP XDATCARs of 416 atoms:
for each composition and 0/50/100 % S/X site exchange, 700 frames 100 fs apart stored as 5-7
consecutive segments. A wildcard opens the segments as one trajectory.
Paper: B. J. Morgan, Chem. Mater. 33, 2004 (2021), figure msd.pdf of the repository.

The paper's figure shows MSD(t) up to 70 ps: about 8 A^2 (I) and 11 A^2 (Cl) at 60 ps for the
ordered structures, and 100 % disorder only a little higher. At 50 % it reaches about 25 A^2 (I)
and 43 A^2 (Cl) by then and still rises. Partial disorder gives the fast diffusion; full exchange
orders the anions again.

    python examples/papers/argyrodite_disorder.py [outdir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ewald import examples, import_file
from ewald.modifiers import MeanSquaredDisplacement

out = Path(sys.argv[1] if len(sys.argv) > 1 else "papers_out")
out.mkdir(exist_ok=True)
folder = examples.fetch("argyrodite-disorder")   # 25 MB, cached after the first run

# %% One MSD per composition and disorder, every frame a time origin
fig, axes = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)
for col, system in enumerate(("Li6PS5I", "Li6PS5Cl")):
    for row, disorder in enumerate(("0p", "50p", "100p")):
        pipe = import_file(folder / f"{system}_{disorder}_run*.XDATCAR.gz")   # segments in order
        pipe.modifiers.append(MeanSquaredDisplacement(time_per_frame=0.1, fit_range=(0.1, 0.5)))
        data = pipe.compute(0)
        t, msd = data.tables["msd"]["Time"], data.tables["msd"]["MSD Li"]
        D = data.attributes["MeanSquaredDisplacement.diffusion_coefficient.Li"] * 1e-4
        print(f"{system} {disorder:>4} disorder: {pipe.num_frames} frames, MSD(Li) at 20/40/60 ps "
              f"{msd[200]:.1f} / {msd[400]:.1f} / {msd[600]:.1f} A^2, D = {D:.2e} cm^2/s")
        ax = axes[row, col]
        ax.plot(t, msd)
        ax.set(xlim=(0, 70), ylim=(0, 55))
        if row == 0:
            ax.set_title(system)
        if col == 1:
            ax.text(1.04, 0.5, disorder.replace("p", " %"), transform=ax.transAxes, va="center")
for ax in axes[:, 0]:
    ax.set_ylabel(r"MSD ($\AA^2$)")
for ax in axes[-1]:
    ax.set_xlabel("time (ps)")
fig.tight_layout()
fig.savefig(out / "argyrodite_disorder_msd.png", dpi=150)
print("wrote", out / "argyrodite_disorder_msd.png")
