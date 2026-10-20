"""Vacancy migration in fcc Fe: energy profile and barrier of a VASP climbing-image NEB folder.

Data: the NEB folder of pymatgen's NEBAnalysis tests, https://github.com/materialsproject/pymatgen
(MIT), test-files/io/vasp/fixtures/neb_analysis/neb1/neb: Fe31 (a 32-site fcc cell with one
vacancy), images 01-03 with OUTCAR and CONTCAR, endpoints 00 and 04 with OUTCAR and POSCAR.
Paper: S. P. Ong et al., "Python Materials Genomics (pymatgen)", Comput. Mater. Sci. 68, 314 (2013),
doi:10.1016/j.commatsci.2012.10.028.

pymatgen's test (tests/analysis/test_transition_state.py) asserts a barrier of 325.2 meV with the
spline maximum at 0.500 of the path. ``load`` reads the folder as a band (one frame per image) and
adds ``NebEnergyProfile`` and ``OverlayFrames``; the same line opens the folder on a cluster
(``load("cluster:/scratch/neb")``, see the remote files page of the manual).

    python examples/papers/fe_vacancy_neb.py [outdir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ewald import examples, load
from ewald.modifiers import CommonNeighborAnalysis
from ewald.views import view

out = Path(sys.argv[1] if len(sys.argv) > 1 else "papers_out")
out.mkdir(exist_ok=True)
folder = examples.fetch("fe-vacancy-neb")    # 34 MB: 00/ .. 04/

pipe = load(folder)                          # a VTST NEB folder: one frame per image
print(pipe.num_frames, "images:", [m.__class__.__name__ for m in pipe.modifiers])

# %% The barrier, against pymatgen's NEBAnalysis
a = pipe.compute(0).attributes
saddle = a["NEB.Saddle Coordinate"] / a["NEB.Path Length"]
print(f"barrier {a['NEB.Barrier'] * 1000:.1f} meV (spline {a['NEB.Spline Barrier'] * 1000:.1f} meV), "
      f"reverse {a['NEB.Reverse Barrier'] * 1000:.1f} meV, saddle at image {a['NEB.Saddle Image']} = "
      f"{saddle:.3f} of a {a['NEB.Path Length']:.3f} A path")
print("pymatgen NEBAnalysis: 325.2 meV at 0.500 of the path")

# %% Which atom hops, how far, and into what: the endpoints without the overlay copies
bare = load(folder)
bare.modifiers.clear()
bare.add(CommonNeighborAnalysis())
first, last = bare.compute(0), bare.compute(bare.num_frames - 1)
step = first.cell.minimum_image(last.positions - first.positions)
hop = int(np.argmax(np.linalg.norm(step, axis=1)))
d = first.cell.minimum_image(first.positions - first.positions[hop])
nn = np.sort(np.linalg.norm(d, axis=1))[1]
structures = np.bincount(np.asarray(first["Structure Type"]), minlength=3)
print(f"atom {hop} hops {np.linalg.norm(step[hop]):.2f} A (nearest-neighbour distance {nn:.2f} A); "
      f"CNA at image 00: {structures[1]} FCC of {first.count}, the rest next to the vacancy")

# %% The saddle image, with the hopping atom's trail from the other images
view(pipe, frame=int(a["NEB.Saddle Image"]), direction=[1, 1, 0], colors={"Fe": "#b07a4f"}, bonds=False,
     text="image [NEB.Image]: [NEB.Relative Energy] eV")

# %% Energy profile
neb, spline = pipe.compute(0).tables["NEB"], pipe.compute(0).tables["NEB spline"]
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.plot(spline["Reaction Coordinate"], spline["Energy"] * 1000, color="tab:blue", lw=1.5, label="ewald spline")
ax.plot(neb["Reaction Coordinate"], neb["Energy"] * 1000, "o", color="tab:blue", label="images")
ax.axhline(325.2, color="0.5", lw=0.8, ls="--", label="pymatgen barrier")
ax.set(xlabel=r"reaction coordinate ($\AA$)", ylabel="energy (meV)", title="vacancy hop in fcc Fe")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out / "fe_vacancy_neb.png", dpi=150)
print("wrote", out / "fe_vacancy_neb.png")
