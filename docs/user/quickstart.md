# Quick start

## Example files

ewald ships small example files, one per common format, so there is something to open right
away: in the GUI under File › Open example, from a shell with `ewald examples` (prints their
paths; `ewald examples --copy DIR` copies them), and from Python:

```python
from ewald import examples, import_file

pipe = import_file(examples.path("POSCAR.SrTiO3"))   # or examples.load("Cu_md.extxyz")
```

| File | Format | Contents |
|---|---|---|
| `POSCAR.SrTiO3` | VASP POSCAR | cubic perovskite SrTiO3, 2x2x2 (40 atoms); try `CoordinationPolyhedra` on Ti |
| `Si.cif` | CIF | diamond Si, conventional cell (8 atoms) |
| `Pt_nanoparticle.xyz` | XYZ | Pt truncated-octahedron nanoparticle (201 atoms), no cell |
| `Cu_md.extxyz` | extended XYZ | fcc Cu (108 atoms), 10 frames with forces and energy |
| `XDATCAR.Si` | VASP XDATCAR | diamond Si (64 atoms), 10-frame trajectory |
| `CHGCAR.Si` | VASP CHGCAR | diamond Si with a valence density on a 16³ grid; try `CreateIsosurface(isolevel=0.3)` |
| `Cu_void.lammpstrj` | LAMMPS dump | Cu block with a spherical void (462 atoms), 5 frames, `element` column |
| `CuZr.data` | LAMMPS data | B2 CuZr (128 atoms), types named in `Masses` |
| `Cu3Au_band.yaml` | phonopy band.yaml | Cu3Au phonon bands with eigenvectors; opens playing a mode |

The structures are textbook crystals; the trajectories, forces and the charge density are
synthetic (seeded random displacements, harmonic forces, Gaussians on atoms and bonds), made by
`src/ewald/examples/make_examples.py`. For real data, [Reproducing papers](papers.md) downloads
published trajectories and structures (AIMD of Li conductors, amorphous carbon and silicon,
TiOF2, PLUMED-NEST metadynamics) and reruns the papers' analyses.

## The pipeline

Everything in ewald follows one pattern, the same as OVITO's Python API: a file becomes a
`Pipeline`, you append modifiers, and `compute(frame)` returns the result for that frame as a
`DataCollection`.

```python
from ewald import import_file
from ewald.modifiers import CoordinationAnalysis, ColorCoding, Slice

pipe = import_file("dump.lammpstrj", type_map={1: "Cu", 2: "Zr"})
print(pipe.num_frames)

pipe.modifiers.append(CoordinationAnalysis(cutoff=3.1))
pipe.modifiers.append(ColorCoding("Coordination", cmap="coolwarm_r"))
pipe.modifiers.append(Slice(normal=(0, 1, 0), distance=20.0, width=10.0))

data = pipe.compute(10)
print(data.summary())                       # properties, ranges, attributes, tables
print(data["Coordination"].mean())
pipe.render("frame10.png", frame=10, size=(2000, 1500), direction="iso")
```

`type_map` is only needed for LAMMPS files, which store numeric types; VASP and most other
formats carry element names already.

Modifiers are evaluated lazily, per frame, and the unmodified input frames are cached, so
changing a modifier and calling `compute` again is cheap. Any function `f(frame, data)` can be
appended as a modifier:

```python
def undercoordinated(frame, data):
    data["Undercoordinated"] = data["Coordination"] < 12

pipe.modifiers.append(undercoordinated)
```

To look at the same pipeline in the desktop viewer:

```python
from ewald.gui import show
show(pipe)
```

or, from a shell, `ewald gui dump.lammpstrj --type-map 1=Cu,2=Zr`.

`python examples/demo.py` writes sample LAMMPS and VASP trajectories and a set of example
figures into `demo_out/`.

Next: [Desktop GUI](gui.md), [Modifiers](modifiers.md), [Python scripting](scripting.md).
