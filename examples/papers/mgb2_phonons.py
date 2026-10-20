"""MgB2 phonons from VASP finite displacements: band structure and the E2g boron mode.

Needs the ewald-phonopy plugin (Plugins panel, or pip install
'git+https://github.com/frssp/ewald-plugins#subdirectory=packages/ewald-phonopy'), which brings phonopy.

Data: phonopy's MgB2 example, https://github.com/phonopy/phonopy (BSD-3-Clause), example/MgB2:
phonopy_disp.yaml and FORCE_SETS of a 3x3x2 supercell, forces from VASP (500 eV, 0.2 eV
smearing, 18x18x18 k-points for the unit cell).
Paper: A. Togo, "First-principles phonon calculations with phonopy and phono3py",
J. Phys. Soc. Jpn. 92, 012001 (2023), doi:10.7566/JPSJ.92.012001; the band path is that of
K.-P. Bohnen, R. Heid, B. Renker, Phys. Rev. Lett. 86, 5771 (2001).

At Gamma the four optical modes are E1u (Mg against B in plane), A2u (Mg against B along c),
E2g (the two B atoms against each other in plane, the bond-stretching mode that couples to the
sigma bands and drives superconductivity in MgB2) and B1g (B against B along c). The notebook
labels them from their eigenvectors and plays the E2g mode in the 3D view.

    python examples/papers/mgb2_phonons.py [outdir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import phonopy

from ewald import examples
from ewald.views import view
from ewald_phonopy import PhononMode, from_phonopy

out = Path(sys.argv[1] if len(sys.argv) > 1 else "papers_out")
out.mkdir(exist_ok=True)
folder = examples.fetch("mgb2-phonons")      # 15 kB

# %% Force constants from the VASP forces, band structure along Bohnen's path (G K M G A H L A)
phonon = phonopy.load(folder / "phonopy_disp.yaml", force_sets_filename=folder / "FORCE_SETS", log_level=0)
path = [[[0, 0, 0], [1 / 3, 1 / 3, 0], [1 / 2, 0, 0], [0, 0, 0], [0, 0, 1 / 2], [1 / 3, 1 / 3, 1 / 2],
         [1 / 2, 0, 1 / 2], [0, 0, 1 / 2]]]
pipe = from_phonopy(phonon, band_path=path, labels=["Γ", "K", "M", "Γ", "A", "H", "L", "A"], mode=False)
modes = pipe.compute(0).attributes["Phonon Modes"]

# %% The Gamma modes, named from their eigenvectors
THZ_TO_CM = 33.35641
types = np.asarray(modes.types).astype(str)


def symmetry(band):
    e = modes.eigenvectors[0, band].real                 # (atoms, 3), Gamma: real up to a phase
    mg, b = e[types == "Mg"], e[types == "B"]
    in_plane = np.abs(e[:, :2]).sum() > np.abs(e[:, 2]).sum()
    if np.abs(mg).sum() > 0.1:                           # Mg moves: polar (IR) modes
        return "E1u" if in_plane else "A2u"
    return "E2g" if in_plane else "B1g"                  # only B moves, the two B against each other


names = {}
for band in range(3, modes.num_bands):
    names.setdefault(symmetry(band), []).append(band)
for name, bands in names.items():
    f = modes.frequencies[0, bands[0]]
    print(f"{name}: {f:5.2f} THz = {f * THZ_TO_CM:5.0f} cm^-1 (bands {', '.join(str(b) for b in bands)})")
e2g = names["E2g"][0]

# %% Play the E2g mode: one period over the frames, with the mode pattern as arrows
pipe.add(PhononMode(qpoint=0, band=e2g, amplitude=0.15, supercell=(3, 3, 1)))

# %%
view(pipe, colors={"Mg": "#f2a33a", "B": "#3a6fd9"}, direction=[0, 0, 1], bonds=True,
     text="E2g [Phonon.Wavenumber] cm^-1")

# %% Band structure (bands sorted by frequency, so the red pair follows E2g only near Gamma and along Gamma-A)
table = pipe.compute(0).tables["Phonon Bands"]
fig, ax = plt.subplots(figsize=(5, 3.4))
for band in range(modes.num_bands):
    ax.plot(table["Distance"], table[f"Band {band + 1}"] * THZ_TO_CM, color="tab:red" if band in names["E2g"] else "0.2", lw=1)
ax.set_xticks([d for d, _ in modes.ticks], [label for _, label in modes.ticks])
for d, _ in modes.ticks:
    ax.axvline(d, color="0.8", lw=0.6)
ax.set(xlim=(table["Distance"][0], table["Distance"][-1]), ylim=(0, None), ylabel=r"frequency (cm$^{-1}$)",
       title=r"MgB$_2$ (red: the two bands that are E$_{2g}$ at $\Gamma$)")
fig.tight_layout()
fig.savefig(out / "mgb2_phonons.png", dpi=150)
print("wrote", out / "mgb2_phonons.png")
