"""Free energy surface of ammonia borane dehydrogenation from metadynamics hills (PLUMED-NEST 19.082).

Needs the ewald-plumed plugin (Plugins panel, or pip install
'git+https://github.com/frssp/ewald-plugins#subdirectory=packages/ewald-plumed'); the PLUMED library
itself is not needed.

Data: https://github.com/gilgalad33/Ammonia-Borane-Dehydrogenation (no license stated; downloaded,
not redistributed), step 1 (AB + AB -> AaDB): HILLS of four walkers of a well-tempered
metadynamics at 400 K (bias factor 16, kJ/mol) on two coordination CVs, and the authors' fes.dat
written by `plumed sum_hills`.
Paper: V. Rizzi, D. Polino, E. Sicilia, N. Russo, M. Parrinello, Angew. Chem. Int. Ed. (2019),
doi:10.1002/anie.201900134.

ewald_plumed.plumed.fes_from_hills sums the same 9,003 hills on the same 201 x 201 grid without PLUMED
and agrees with fes.dat to a few hundredths of a kJ/mol.

    python examples/papers/ammonia_borane_fes.py [outdir]
"""
# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ewald import examples
from ewald_plumed.plumed import Hills, fes_from_hills, read_fes, read_hills

out = Path(sys.argv[1] if len(sys.argv) > 1 else "papers_out")
out.mkdir(exist_ok=True)
folder = examples.fetch("ammonia-borane-metad")   # 4.6 MB

# %% The walkers share one bias: put their hills together
walkers = [read_hills(folder / f"HILLS.{i}") for i in range(4)]
hills = Hills(walkers[0].cvs, np.vstack([w.centers for w in walkers]), np.vstack([w.sigmas for w in walkers]),
              np.concatenate([w.heights for w in walkers]), None, walkers[0].periods, walkers[0].kernel)
reference = read_fes(folder / "fes.dat")
grid = reference["grid"]
fes = fes_from_hills(hills, bins=[len(g) - 1 for g in grid], ranges=[(g[0], g[-1]) for g in grid])["fes"]
ref = reference["fes"] - reference["fes"].min()
diff = fes - ref
print(f"{len(hills)} hills from 4 walkers on {hills.cvs}")
print(f"ewald vs plumed sum_hills: RMS {np.sqrt((diff ** 2).mean()):.3f} kJ/mol, "
      f"max {np.abs(diff).max():.3f} kJ/mol over {fes.size} grid points")
i = np.unravel_index(np.argmin(fes), fes.shape)
print(f"global minimum at {hills.cvs[0]} = {grid[0][i[0]]:.3f}, {hills.cvs[1]} = {grid[1][i[1]]:.3f}")

# %% Figure: the two surfaces side by side
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
levels = np.arange(0, 200, 10)
for ax, f, title in zip(axes, (fes, ref), ("ewald fes_from_hills", "plumed sum_hills (fes.dat)")):
    cs = ax.contourf(grid[0], grid[1], f.T, levels=levels, cmap="viridis")
    ax.set(xlabel=hills.cvs[0], title=title)
axes[0].set_ylabel(hills.cvs[1])
fig.colorbar(cs, ax=axes, label="free energy (kJ/mol)")
fig.savefig(out / "ammonia_borane_fes.png", dpi=150)
print("wrote", out / "ammonia_borane_fes.png")
