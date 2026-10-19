"""Amorphous carbon: local energy by coordination and sp3 fraction against density (C-SYNTH-23).

Data: 6 of the 546 C-GAP-17 melt-quench-anneal trajectories of https://github.com/jla-gardner/carbon-data
(200 atoms, 211 frames 1 ps apart, annealed at 3000 K, densities 1.0-3.5 g/cm3; MIT as listed by
load-atoms). Paper: J. L. A. Gardner, Z. F. Beaulieu, V. L. Deringer, Digital Discovery (2023).

Fig. 2a of the paper shows the C-GAP-17 local energies of sp (2-coordinated), sp2 (3) and sp3 (4)
atoms, counted with a 1.85 A cutoff: sp2 and sp3 distributions are similar and sp atoms sit
notably higher. Here the same histogram comes from the annealed and quenched frames (t >= 10 ps) of six
trajectories instead of the whole dataset, plus the sp3 fraction of the final structures.

    python examples/papers/carbon_coordination.py [outdir]
"""
# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ewald import examples, import_file
from ewald.modifiers import CoordinationAnalysis

out = Path(sys.argv[1] if len(sys.argv) > 1 else "papers_out")
out.mkdir(exist_ok=True)
folder = examples.fetch("carbon-synth")   # 31 MB, cached after the first run

densities = ["1.0", "1.5", "2.0", "2.5", "3.0", "3.5"]
energy = {2: [], 3: [], 4: []}           # C-GAP-17 local energy (eV) by coordination
final = {}                               # density -> fractions of 2-, 3-, 4-coordinated atoms

# %% Coordination with the paper's 1.85 A cutoff, frame by frame
for rho in densities:
    pipe = import_file(folder / f"density-{rho}-T-3000.extxyz")
    pipe.modifiers.append(CoordinationAnalysis(cutoff=1.85))
    for frame in range(10, pipe.num_frames):          # skip the 9000 K melt
        data = pipe.compute(frame)
        cn, e = data.particles["Coordination"], data.particles["gap17_energy"]
        for k in energy:
            energy[k].append(e[cn == k])
    counts = np.bincount(cn, minlength=5)[2:5] / len(cn)          # last frame, 300 K
    final[rho] = counts
    print(f"{rho} g/cm3: sp {counts[0]:.2f}  sp2 {counts[1]:.2f}  sp3 {counts[2]:.2f}")

energy = {k: np.concatenate(v) for k, v in energy.items()}
for k, name in zip(energy, ("sp", "sp2", "sp3")):
    print(f"{name}: {len(energy[k])} environments, mean local energy {energy[k].mean():.2f} eV")
print(f"sp - sp2 = {energy[2].mean() - energy[3].mean():.2f} eV, "
      f"sp3 - sp2 = {energy[4].mean() - energy[3].mean():.2f} eV")

# %% Figure: Fig. 2a-style energy distributions, and sp3 fraction vs density
fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.2))
bins = np.linspace(-159.5, -153.5, 120)
for k, name in zip(energy, ("sp (N=2)", "sp$^2$ (N=3)", "sp$^3$ (N=4)")):
    a.hist(energy[k], bins, density=True, histtype="step", label=name)
a.set(xlabel="C-GAP-17 local energy (eV)", ylabel="density")
a.legend(frameon=False)
rho = np.array(densities, float)
for i, name in enumerate(("sp", "sp$^2$", "sp$^3$")):
    b.plot(rho, [final[r][i] for r in densities], "o-", label=name)
b.set(xlabel=r"density (g/cm$^3$)", ylabel="fraction (final structure)")
b.legend(frameon=False)
fig.tight_layout()
fig.savefig(out / "carbon_coordination.png", dpi=150)
print("wrote", out / "carbon_coordination.png")
