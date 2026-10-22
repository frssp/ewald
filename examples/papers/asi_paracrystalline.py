"""Paracrystalline vs polycrystalline amorphous silicon, 100,000 atoms each.

Data: https://github.com/lamr18/aSi-data (no license stated; downloaded, not redistributed),
structures quenched at 1e11 K/s (paracrystalline) and 1e10 K/s (polycrystalline) with an MTP
potential, plus the paper's computed S(q) and the experiment of Laaziri et al. (Fig. 5).
Paper: L. A. M. Rosset, D. A. Drabold, V. L. Deringer, Nat. Commun. 16, 2360 (2025).

The paper computes S(q) with the Debye equation (DebyeCalculator, X-ray); this script uses
ewald's StructureFactor (Fourier transform of g(r) up to 20 A with a Lorch window), which
matches the paracrystalline curve closely and smooths the Bragg peaks of the polycrystalline one.
IdentifyDiamond gives the crystalline fraction: small and scattered in the paracrystalline
model, most of the atoms in the polycrystalline one.

    python examples/papers/asi_paracrystalline.py [outdir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ewald import color_rule, examples, import_file, view
from ewald.modifiers import CoordinationAnalysis, IdentifyDiamond, StructureFactor

out = Path(sys.argv[1] if len(sys.argv) > 1 else "papers_out")
out.mkdir(exist_ok=True)
folder = examples.fetch("asi-paracrystalline")   # 11 MB, cached after the first run

experiment = np.loadtxt(folder / "Sq_Laaziri.csv", delimiter=",")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

for model, ax in zip(("para", "poly"), axes):
    # %% S(q), g(r), coordination and diamond-like environments of one model
    pipe = import_file(folder / f"100k-{model}.xyz")
    pipe.modifiers.extend([
        StructureFactor(method="rdf", q_min=1.0, q_max=12.0, number_of_bins=220, cutoff=20.0),
        CoordinationAnalysis(cutoff=2.85),                        # first shell, Si-Si ~2.35 A
        # Structure Type per atom, and (color_by_type) the atom colors of the 3D view, OVITO's:
        # white OTHER (amorphous), blue cubic diamond, cyan/green its 1st/2nd neighbors,
        # orange/yellow/lime hexagonal diamond and its 1st/2nd neighbors
        IdentifyDiamond(color_by_type=True),
    ])
    data = pipe.compute(0)
    print(model, color_rule.describe(data))                     # the color rule, with each color and count
    sq = data.tables["structure-factor"]
    pipe.modifiers.append(CoordinationAnalysis(cutoff=8.0, number_of_bins=200))   # g(r) to 8 A
    gr = pipe.compute(0).tables["coordination-rdf"]
    cn = np.bincount(data.particles["Coordination"], minlength=6)[3:6] / data.count
    crystalline = np.count_nonzero(data.particles["Structure Type"]) / data.count

    # the paper's CSV holds (S, q) with S shifted and scaled for plotting: plotted = (S + 0.5) * 2
    paper = np.loadtxt(folder / f"Sq-{model}.csv", delimiter=",")
    q_paper, s_paper = paper[:, 1], (paper[:, 0] + 0.5) * 2
    inside = (q_paper > 1.5) & (q_paper < 12)
    r = np.corrcoef(np.interp(q_paper[inside], sq["q"], sq["S(q)"]), s_paper[inside])[0, 1]
    print(f"{model}: first peak of g(r) at {gr['r'][np.argmax(gr['g(r)'])]:.2f} A, "
          f"3/4/5-coordinated {cn.round(3)}, diamond-like {crystalline:.1%}, "
          f"S(q) vs paper: correlation {r:.3f}")

    ax.plot(sq["q"], sq["S(q)"], label="ewald")
    ax.plot(q_paper, s_paper, lw=1, label="paper (Debye)")
    ax.plot(experiment[:, 0], experiment[:, 1], "k--", lw=1, label="Laaziri (exp.)")
    ax.set(xlim=(0, 12), xlabel=r"q ($\AA^{-1}$)", ylabel="S(q)", title=f"{model}, {crystalline:.0%} diamond-like")
    ax.legend(frameon=False)
    axes[2].plot(gr["r"], gr["g(r)"], label=model)

axes[2].set(xlabel=r"r ($\AA$)", ylabel="g(r)", xlim=(0, 8))
axes[2].legend(frameon=False)
fig.tight_layout()
fig.savefig(out / "asi_paracrystalline.png", dpi=150)
print("wrote", out / "asi_paracrystalline.png")

# %% The 3D view of the last model: colors from IdentifyDiamond (the steps' colors), with their legend.
# color_by="Coordination" (colormap, color_range) or "type" would recolor it for display only.
view(pipe, color_by=None, legend=True)
