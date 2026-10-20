# Quick start

EWALD works the way a Jupyter notebook does, with a 3D view of your structure (the Structure tab) next to it. You open a
file, and the analysis grows as a list of cells: steps that add an analysis or change the atoms,
views that plot what the steps computed, and any Python in between. The notebook is saved as a
file that also runs as a script, so whatever you do in the window can be rerun without it.

## Open a file

```bash
pip install -e ".[gui]"
ewald gui                                # the welcome page: New notebook, Open, Open example, recent files
ewald gui XDATCAR                        # or your own file
ewald gui dump.lammpstrj --type-map 1=Cu,2=Zr
```

Opening a file makes a notebook for it with a `load` cell and the Structure view of the first frame. A
LAMMPS dump stores numbered types, so the GUI asks which element each number is (or pass
`--type-map`). Files on a cluster open by path, `cluster:/scratch/me/run1/dump.lammpstrj`
([Files on another machine](formats.md#files-on-another-machine-ssh)).

![A notebook for the Cu void example: steps and a linked histogram on the left, the Structure view in the middle](../images/notebook.png)

## Add steps and views

**+ Step** under the cells (or Insert › Step) adds an analysis. Pick one, for example Coordination
analysis, and the notebook writes the cell for you:

```python
pipe.add(CoordinationAnalysis(cutoff=3.0))
```

The cell shows as a step: the parameters, *Form* to edit them, how long it took and what it produced
(here a `Coordination` property and a `coordination-rdf` table, with a small histogram). Edit the
cutoff and run the cell again (Ctrl+Enter) and the step is replaced, not added twice.

**+ View** adds a plot of anything a step produced:

```python
show(pipe, histogram="Coordination")
show(pipe, table="coordination-rdf")
```

A view is linked to the Structure view. Drag a range on the histogram and those atoms are selected; pick
atoms in the Structure view and they are highlighted on the plot. On a per-frame plot, such as an MSD or
the `Thermo` table of a LAMMPS or VASP run, dragging selects frames instead.

Any other Python goes in a code cell, and a function added with `@pipe.add` runs on every frame.
`paint`, `hide`, `isolate`, `fade` and `resize` style atoms by any condition:

```python
@pipe.add
def style(frame, data):
    paint(data, data["Coordination"] < 12, "#C8551F")   # surface atoms
    hide(data, data["Position"][:, 0] > 8)              # cut the block open
```

Run all (Ctrl+Shift+Enter) reruns the notebook from the top. Steps can also come from the Python
console, Ask AI and the plugin side bar; they all end up as cells.

## Save and share

File › Save notebook (Ctrl+S) writes one of three formats:

| Format | Use it to |
|---|---|
| `.py` | keep the analysis in git and run it with `python analysis.py` |
| `.ipynb` | open it in Jupyter or VS Code, with outputs |
| `.ewd` | send it to someone: code, outputs, figures, data files up to 50 MB and the plugins it uses, in one file |

A notebook someone else sent opens without running until you agree to run it. Details:
[Notebooks](notebooks.md).

## The same thing as a script

The saved `.py` is an ordinary script. Run as a script, `show` draws its plot with matplotlib, and
the rest is the same pipeline API:

```python
from ewald import examples, load
from ewald.modifiers import CoordinationAnalysis

pipe = load(examples.path("Cu_void.lammpstrj"))
pipe.add(CoordinationAnalysis(cutoff=3.0))
data = pipe.compute(0)                      # the result for frame 0, as a DataCollection
print(data.summary())                       # properties, ranges, attributes, tables
print((data["Coordination"] < 12).sum(), "surface atoms")
pipe.render("void.png", renderer="raytrace")
```

`load` is another name for `import_file` and `pipe.add(m)` for `pipe.modifiers.append(m)`, the
names OVITO's Python API uses; both work. Modifiers run lazily, frame by frame, and the input frames
are cached, so changing a modifier and computing again is cheap. `compute(frame, device="cuda")`
runs the analysis on an NVIDIA GPU ([GPU compute](gpu.md)). To look at a pipeline built in a script,
`from ewald.gui import show; show(pipe)` opens the window on it.

## Example files

EWALD ships small example files, one per common format, so there is something to open right away:
in the GUI under File › Open example, from a shell with `ewald examples` (prints their paths;
`ewald examples --copy DIR` copies them), and from Python with `examples.path(name)` or
`examples.load(name)`.

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

The structures are textbook crystals; the trajectories, forces and the charge density are synthetic
(seeded random displacements, harmonic forces, Gaussians on atoms and bonds), made by
`src/ewald/examples/make_examples.py`. For real data, [Reproducing papers](papers.md) downloads
published trajectories and structures and reruns the papers' analyses.

Next: the tutorials, starting with [Your first notebook](tutorials/first-notebook.md).
