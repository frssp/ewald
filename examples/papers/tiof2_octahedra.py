"""O/F order in cubic TiOF2: bond lengths, TiO(6-n)Fn octahedra and partial RDFs.

Data: https://github.com/bjmorgan/data_TiOF2 (data CC BY 4.0), four CP2K-relaxed 4x4x4 cells
(CIF, 256 atoms) and the 6x6x6 anion arrangement optimized by a genetic algorithm on a cluster
expansion (VASP POSCAR, 864 atoms, ideal positions).
Paper: B. J. Morgan et al., J. Am. Chem. Soc. (2024), doi:10.1021/jacs.4c06304.

The repository's notebooks (octahedral_distortions.ipynb, 6x6x6_GA_analysis.ipynb) report, for
structures 1-4, Ti-F bonds of 1.864-2.171 A with mean 1.984-1.989 A and Ti-O bonds of
1.725-1.934 A with mean 1.810-1.812 A, and 194 TiO2F4, 11 TiOF5 and 11 TiO3F3 octahedra in the
6x6x6 structure (no TiO4F2 or less), far from both random O/F (binomial) and the NbO2F-type model.

    python examples/papers/tiof2_octahedra.py [outdir]
"""
# %%
import sys
from collections import Counter
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ewald import examples, import_file
from ewald.modifiers import CoordinationAnalysis, CreateBonds

out = Path(sys.argv[1] if len(sys.argv) > 1 else "papers_out")
out.mkdir(exist_ok=True)
folder = examples.fetch("tiof2")   # 0.1 MB

PAPER = {   # structure: (Ti-F max, min, mean, mean abs. deviation, Ti-O max, min, mean, MAD)
    1: (2.162, 1.879, 1.989, 0.056, 1.909, 1.740, 1.810, 0.033),
    2: (2.171, 1.864, 1.985, 0.057, 1.934, 1.725, 1.812, 0.045),
    3: (2.136, 1.858, 1.988, 0.050, 1.890, 1.750, 1.810, 0.025),
    4: (2.158, 1.878, 1.984, 0.053, 1.917, 1.733, 1.811, 0.034),
}


def ti_anion_bonds(data):
    """Bond lengths and Ti/anion indices of every Ti-O and Ti-F bond."""
    i, j = data.bonds.topology.T
    types = np.asarray(data.types).astype(str)
    ti, anion = np.where(types[i] == "Ti", i, j), np.where(types[i] == "Ti", j, i)
    return data.bonds.lengths(data), ti, types[anion]


def stats(d):
    return d.max(), d.min(), d.mean(), np.abs(d - d.mean()).mean()


bonds = CreateBonds(mode="pairwise", pairs=[("Ti", "O", 3.0), ("Ti", "F", 3.0)])   # the notebooks' 3.0 A cutoff

# %% Bond-length table of the relaxed 4x4x4 structures
lengths = {"F": [], "O": []}
rdfs = []
print("        Ti-F max   min  mean   MAD | Ti-O max   min  mean   MAD")
for k, ref in PAPER.items():
    pipe = import_file(folder / f"config_{k}.cif")
    pipe.modifiers += [bonds, CoordinationAnalysis(cutoff=6.0, number_of_bins=300, partial=True)]
    data = pipe.compute(0)
    rdfs.append(data.tables["coordination-rdf"])
    d, _, anion = ti_anion_bonds(data)
    ours = (*stats(d[anion == "F"]), *stats(d[anion == "O"]))
    worst = max(abs(a - b) for a, b in zip(ours, ref))
    print(f"  {k}     " + "  ".join(f"{v:.3f}" for v in ours) + f"   (largest difference from the notebook: {worst:.3f} A)")
    for x in lengths:
        lengths[x].append(d[anion == x])

# %% Octahedra in the 6x6x6 GA structure, against random O/F and the NbO2F-type (Brink) model
pipe = import_file(folder / "6x6x6_opt.vasp")
pipe.modifiers.append(bonds)
data = pipe.compute(0)
_, ti, anion = ti_anion_bonds(data)
n_f = np.bincount(ti[anion == "F"], minlength=data.count)[np.asarray(data.types).astype(str) == "Ti"]
counts = Counter(n_f.tolist())
print("6x6x6 octahedra by number of F:", dict(sorted(counts.items())), "(paper: {3: 11, 4: 194, 5: 11})")
n = np.arange(7)
random = np.array([comb(6, k) * (2 / 3) ** k * (1 / 3) ** (6 - k) for k in n])
brink = np.array([0, 0, 0, 8, 12, 6, 1]) / 27
ga = np.array([counts.get(k, 0) for k in n]) / len(n_f)

fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.4))
bins = np.linspace(1.65, 2.25, 61)
for x, color in (("O", "tab:red"), ("F", "tab:green")):
    a.hist(np.concatenate(lengths[x]), bins, color=color, alpha=0.7, label=f"Ti-{x}")
a.set(xlabel=r"bond length ($\AA$)", ylabel="count", title="relaxed 4x4x4 (4 structures)")
a.legend(frameon=False)
w = 0.27
b.bar(n - w, random, w, label="random O/F")
b.bar(n, brink, w, label=r"NbO$_2$F-type (Brink)")
b.bar(n + w, ga, w, label=r"6$\times$6$\times$6 GA")
b.set(xlabel=r"F per TiO$_{6-n}$F$_n$ octahedron", ylabel="fraction")
b.legend(frameon=False)
for pair in ("g(r) O-Ti", "g(r) F-Ti", "g(r) F-F", "g(r) F-O", "g(r) O-O"):   # mean of the 4 structures
    c.plot(rdfs[0]["r"], np.mean([t[pair] for t in rdfs], axis=0), lw=1, label=pair[5:])
c.set(xlabel=r"r ($\AA$)", ylabel="g(r)", title="partial RDFs, relaxed 4x4x4", xlim=(1.5, 6))
c.legend(frameon=False)
fig.tight_layout()
fig.savefig(out / "tiof2_octahedra.png", dpi=150)
print("wrote", out / "tiof2_octahedra.png")
