# Phonon modes

ewald reads phonon eigenvectors from phonopy and VASP and animates a mode: the atoms vibrate
along it over one period, the mode pattern is drawn as arrows, and in the desktop GUI the
Plots tab shows the band structure, where clicking a band plays that mode.

VASP OUTCARs and phonopy's band.yaml, mesh.yaml and qpoints.yaml are read by ewald itself
(the built-in plugin group "Phonons", which can be switched off in the Plugins side bar).
Computing the bands from phonopy's input with forces (`phonopy.yaml` with `FORCE_SETS` or
`FORCE_CONSTANTS`) needs phonopy, and that is the **ewald-phonopy plugin** (id `phonopy`):
install it from the Plugins side bar or with

```bash
pip install 'git+https://github.com/frssp/ewald-plugins#subdirectory=packages/ewald-phonopy'
```

It brings phonopy along, adds the `phonopy-fc` format below, `from_phonopy(phonon)` for a
`phonopy.Phonopy` object (from aiida-phonopy, say) and the example file `Cu3Au_band.yaml`
(`ewald_phonopy.EXAMPLES`).

```python
from ewald import import_file
from ewald.modifiers import PhononMode

pipe = import_file("band.yaml", frames=24)           # 24 frames = one period
pipe.modifiers.append(PhononMode(qpoint=10, band=3, amplitude=0.3))
data = pipe.compute(0)
print(data.attributes["Phonon.Frequency"], "THz", data.attributes["Phonon.Supercell"])

from ewald.render import render_animation
render_animation(pipe, "mode.gif", fps=24)           # loops seamlessly
```

## Files

| File | Notes |
|---|---|
| phonopy `band.yaml` | band structure along a path; needs eigenvectors (`phonopy --eigvecs`, `EIGENVECTORS = .TRUE.`). High-symmetry labels come from `labels` or the per-point `label` keys |
| phonopy `mesh.yaml`, `qpoints.yaml` | modes on a mesh or a list of q-points, with eigenvectors. `qpoints.yaml` has no cell: pass the primitive cell as `structure="POSCAR"` |
| phonopy `phonopy.yaml`, `phonopy_params.yaml`, `phonopy_disp.yaml` with `FORCE_SETS` or `FORCE_CONSTANTS` | ewald-phonopy plugin, format `phonopy-fc`: phonopy computes the band structure with eigenvectors along the seekpath path, so no `band.yaml` is needed |
| VASP OUTCAR, IBRION = 5, 6, 7 or 8 | Gamma-point modes from "Eigenvectors and eigenvalues of the dynamical matrix"; the structure is the first ionic step and the masses come from POMASS. Imaginary modes (`f/i=`) become negative frequencies, as in phonopy |

A phonon OUTCAR is recognized by `IBRION` in its header; any other OUTCAR still opens as ionic
steps. `format="phonopy-yaml"` (or `"phonopy"`) or `format="vasp-phonon"` forces the phonon reader.

Each frame of the file is the primitive cell with the modes in
`data.attributes["Phonon Modes"]` (a `PhononModes` object with `qpoints`, `frequencies` in THz,
`eigenvectors` of shape `(nq, nbands, natoms, 3)`, `distances` and `ticks` for a path) and
the frequencies as the table "Phonon Bands": `Distance` and one column per band for a path,
`Q-point` for a list or mesh, `Mode` and `Frequency` for a single q-point. Reader options:
`frames` (frames per period, default 24) and `structure`.

## PhononMode

`PhononMode(qpoint, band, amplitude, supercell, ...)` displaces the atoms along one mode,
the way `phonopy --modulation` does:

$$
\mathbf u_{lj}(f) = \mathrm{Re}\left[ \frac{A}{\sqrt{m_j}}\, \mathbf e_j(\mathbf q, \nu)\,
e^{2\pi i\, \mathbf q \cdot \mathbf x_{lj}}\, e^{-2\pi i f / N} \right]
$$

where `x_lj` is the position of atom `j` in cell `l` in fractional coordinates of the primitive
cell, frame `f` of `N` sets the phase, and the global phase makes the largest component real,
so frame 0 shows the largest displacement. The tests compare the pattern with phonopy's own
modulations.

| Option | Meaning |
|---|---|
| `qpoint`, `band` | the mode, 0-based; `band=None` takes the first optical branch (band 3) |
| `amplitude` | the largest atomic displacement, Å |
| `supercell` | `(na, nb, nc)` repeats of the primitive cell; `None` takes the smallest one in which the mode is periodic (X: 2×1×1), up to `max_supercell` per axis |
| `arrows`, `arrow_length`, `arrow_width`, `arrow_color` | draw one arrow per atom; at full swing the longest is `arrow_length` Å, the others in proportion |
| `arrow_mode` | what the arrows show: `"displacement"` (default) this frame's displacement, property "Phonon Displacement", so an arrow shrinks, vanishes at the rest position and flips as the atom swings back; `"velocity"` this frame's velocity, "Phonon Velocity" in Å/ps, longest as an atom passes its rest position; `"fixed"` the mode pattern, "Phonon Mode" (the displacement at frame 0), the same in every frame, as in a still figure |

It sets the attributes `Phonon.Frequency` (THz), `Phonon.Wavenumber` (cm⁻¹), `Phonon.QPoint`,
`Phonon.Band`, `Phonon.q` and `Phonon.Supercell`, and the particle properties "Phonon Mode",
"Phonon Displacement" and "Phonon Velocity" in every frame. The arrows are an ordinary
`VectorVis` style, so every renderer draws them; for a still figure of a mode, use
`arrow_mode="fixed"` (or frame 0, where both agree) with the ray tracer or as SVG/PDF.

## In the GUI

Opening a phonon file adds a `PhononMode` to the pipeline, so the first optical mode at the
first q-point plays at once. The Plots tab draws the band structure with its high-symmetry
labels (or one tick per mode for a Gamma-only OUTCAR) and circles the mode on screen. Click a
band to switch to the nearest mode and start playback; the status bar shows its q-point and
frequency. The modifier's parameter form changes amplitude, supercell and arrows. Phonon mode
animation is also under Insert › Step (Phonons).
