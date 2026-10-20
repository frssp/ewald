"""Which O/F ordering does NbO2F prefer? 92 VASP relaxations opened as one series of runs.

Data: https://github.com/bjmorgan/data_nbo2f_chirality (data CC BY 4.0, code MIT), the 92 OUTCARs
of the 3x3x3 supercell (Nb27 O54 F27, PBEsol, 600 eV, full relaxation with ISIF=3): random anion
placements, period-three chain orderings with random inter-chain phase, and the twelve
symmetry-distinct orderings that are cis-NbO4F2 everywhere with period three along every
Nb-X-Nb chain ("cis + OOF").
Paper: B. J. Morgan, "Structural Chirality from Short-Range Order in Heteroanionic Materials",
arXiv:2608.04841 (2026), doi:10.48550/arXiv.2608.04841.

The paper's result: the ground state is orbit 11, the helical P3_121 ordering, whose two
enantiomers (P3_121 and P3_221) have the same energy; every octahedron is cis-NbO4F2. This
notebook finds that from the files alone: ``load_runs`` opens the 92 runs, ``check_run`` confirms
each relaxation reached EDIFFG, the energies rank them, ``SymmetryAnalysis`` names the space
group of the relaxed cells, and the F-Nb-F angles count cis and trans octahedra.

    python examples/papers/nbo2f_anion_order.py [outdir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ewald import examples, load_runs
from ewald.modifiers import CoordinationPolyhedra, CreateBonds, SymmetryAnalysis
from ewald.views import view

out = Path(sys.argv[1] if len(sys.argv) > 1 else "papers_out")
out.mkdir(exist_ok=True)
folder = examples.fetch("nbo2f-anion-order")    # 19 MB, 92 OUTCARs in config_000 .. config_091

# %% Open all 92 relaxations as one series; each run is an ordinary pipeline (last frame = relaxed)
runs = load_runs(folder / "config_*/OUTCAR")
problems = {label: [f.title for f in c.findings if f.level not in ("ok", "info")] for label, c in runs.check().items()}
print(f"{len(runs)} runs; relaxations with problems: {sum(map(bool, problems.values()))}")

# %% Energy per atom of every run, relative to the lowest
energy = np.array([r.pipe.compute(r.pipe.num_frames - 1).attributes["Energy"] / r.pipe.compute(0).count for r in runs])
de = (energy - energy.min()) * 1000                       # meV/atom
order = np.argsort(de)
ground = runs.labels[order[0]]
pipe = runs[ground].pipe                                  # the run the 3D view shows

# %% Space group of the relaxed cells (0.1 A tolerance) and cis/trans octahedra from the F-Nb-F angle
bonds = CreateBonds(mode="pairwise", pairs=[("Nb", "O", 2.5), ("Nb", "F", 2.5)])


def describe(run):
    p = run.pipe
    p.add(SymmetryAnalysis(symprec=0.1), bonds)
    data = p.compute(p.num_frames - 1)
    types = np.asarray(data.types).astype(str)
    i, j = data.bonds.topology.T
    vec = data.bonds.vectors(data)
    nb, other = np.where(types[i] == "Nb", i, j), np.where(types[i] == "Nb", j, i)
    vec = np.where((types[i] == "Nb")[:, None], vec, -vec)    # from Nb to the anion
    cis = 0
    for n in np.flatnonzero(types == "Nb"):
        f = vec[(nb == n) & (types[other] == "F")]
        if len(f) == 2:
            cos = f[0] @ f[1] / np.linalg.norm(f[0]) / np.linalg.norm(f[1])
            cis += cos > np.cos(np.radians(135))
    a = data.attributes
    return a["SymmetryAnalysis.space_group"], a["SymmetryAnalysis.space_group_number"], int(cis)


print("  run          dE (meV/atom)  space group   cis-NbO4F2 octahedra")
rows = {}
for k in order:
    run = runs[runs.labels[k]]
    rows[run.label] = describe(run)
for k in order[:8]:
    label = runs.labels[k]
    sg, num, cis = rows[label]
    print(f"  {label}   {de[k]:8.2f}       {sg:>8} ({num:3d})   {cis}/27")
first, second = (runs.labels[k] for k in order[:2])
print(f"ground state: {rows[first][0]} and {rows[second][0]}, {abs(de[order[1]] - de[order[0]]) * 1000:.0f} ueV/atom apart"
      f" (paper: the P3_121 / P3_221 enantiomer pair, orbit 11)")

# %% The ground state with its NbO4F2 octahedra
pipe.add(CoordinationPolyhedra(center_types=["Nb"], transparency=0.2))

# %%
view(pipe, colors={"F": "#3fbf3f", "O": "#d93f3f", "Nb": "#5c9ec7"}, direction=[1, 1, 1], type_legend=True,
     text="[SymmetryAnalysis.space_group]")

# %% Energy against the number of cis octahedra
cis = np.array([rows[label][2] for label in runs.labels])
chiral = np.array([rows[label][1] in (152, 154) for label in runs.labels])
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.scatter(cis[~chiral], de[~chiral], s=14, color="0.45", label="other orderings")
ax.scatter(cis[chiral], de[chiral], s=40, color="tab:red", marker="*", label=r"P3$_1$21 / P3$_2$21")
ax.set(xlabel=r"cis-NbO$_4$F$_2$ octahedra (of 27)", ylabel=r"$\Delta E$ (meV/atom)", yscale="symlog")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out / "nbo2f_anion_order.png", dpi=150)
print("wrote", out / "nbo2f_anion_order.png")
