<img src="src/ewald/assets/icon/ewald-128.png" alt="EWALD icon" width="96" align="right">

# EWALD: Extensible Workbench for Atomic Lattices & Dynamics

[![CI](https://github.com/frssp/ewald/actions/workflows/ci.yml/badge.svg)](https://github.com/frssp/ewald/actions/workflows/ci.yml)

EWALD (Python package `ewald`) is a notebook for atomistic structures and trajectories. You open a
VASP, LAMMPS, extxyz or any other file, and the analysis is written as Python cells next to a 3D
view: each step adds a modifier, each view is a live plot linked to the atoms, and the whole thing
is saved as one file that also runs as a plain Python script. Claude can read the structure and add
steps for you, and plugins add file formats, analyses and drawing without changing EWALD itself.

![The EWALD window: a notebook on the left, the 3D view in the middle, display settings on the right](docs/images/notebook.png)

Under the notebook is an OVITO-style pipeline (`load` → `pipe.add(...)` → `pipe.compute(frame)` →
`pipe.render(...)`) that runs the same in a script, on the CPU or an NVIDIA GPU, with a built-in
ray tracer and vector output for publication figures.

## Install

```bash
pip install -e ".[gui]"     # the desktop app (PySide6, VTK); leave out [gui] for scripts only
pip install -e ".[gpu]"     # + CuPy for device="cuda" (NVIDIA)
pip install -e ".[llm]"     # + Claude (Ask Claude, ewald ask)
pip install -e ".[remote]"  # + files on SSH hosts (paramiko)
```

A C++17 compiler builds the native analyses (PTM, Voronoi, CSP matching, interval CNA); without one
the install still works, minus those. On macOS run `xcode-select --install` first. Releases ship
prebuilt wheels for Linux, macOS and Windows plus a Windows installer and a macOS app
([install from a release](docs/user/install.md#from-a-release)); changes are in [CHANGELOG.md](CHANGELOG.md).
Linux system libraries, `ssh -X`, headless machines, macOS and Windows: [Install](docs/user/install.md).

## Five minutes with a notebook

```bash
ewald gui                       # opens with an empty notebook; File › Open example › Cu_void.lammpstrj
ewald gui dump.lammpstrj        # or open your own file
```

Opening a file makes a notebook for it. Click **+ Step** to add an analysis, **+ View** to plot what
it produced, and write any Python in between. This is the notebook from the
[first tutorial](docs/user/tutorials/first-notebook.md), as it is saved:

```python
# %%
from ewald import examples, load
from ewald.modifiers import *
from ewald.paint import paint, hide
from ewald.views import show

pipe = load(examples.path("Cu_void.lammpstrj"))

# %% coordination
pipe.add(CoordinationAnalysis(cutoff=3.0))

# %% style: surface atoms, every frame
@pipe.add
def style(frame, data):
    paint(data, data["Coordination"] < 12, "#C8551F")
    hide(data, data["Position"][:, 0] > 8)           # cut the block open

# %%
show(pipe, histogram="Coordination")
```

In the GUI the `pipe.add` cell is a step with a parameter form and the time it took, `show(...)` is a
live histogram (drag a range to select those atoms in the 3D view), and the style follows every
frame of the trajectory. `python cu_void.py` runs the same file without the GUI and draws the plots
with matplotlib. File › Save notebook writes it as a `.py`, a Jupyter `.ipynb`, or an `.ewd` that
also carries the outputs, figures, small data files and the plugins it uses, so a colleague can open
it with everything in place ([Notebooks](docs/user/notebooks.md)).

The same pipeline from a script, with GPU compute and a ray-traced figure:

```python
from ewald import load
from ewald.modifiers import CommonNeighborAnalysis

pipe = load("dump.lammpstrj", type_map={1: "Cu"})
pipe.add(CommonNeighborAnalysis())
data = pipe.compute(10, device="cuda")
print(data.summary())
pipe.render("frame10.png", frame=10, renderer="raytrace")     # also "opengl" and "vector" (SVG/PDF)
```

## Tutorials

1. [Your first notebook](docs/user/tutorials/first-notebook.md): find the surface atoms of a void,
   style them, plot and save. (Cu void, LAMMPS dump)
2. [A crystal structure figure](docs/user/tutorials/structure-figure.md): space group, octahedra,
   bond lengths and a publication render. (SrTiO3, POSCAR)
3. [A trajectory](docs/user/tutorials/trajectory.md): displacements, MSD, force arrows and frames
   picked on a plot. (Cu MD, extxyz)
4. [Reproducing papers](docs/user/papers.md): published AIMD, MD and metadynamics data downloaded
   and reanalysed (Li conductors, amorphous C and Si, TiOF2).

The tutorial notebooks are in [examples/notebooks](examples/notebooks); every example file they use
ships with the package (File › Open example, `ewald examples`).

## What is in it

**Notebook GUI.** Notebook tabs, step and view cells, Markdown, matplotlib inline, a Python console
on the same namespace, a command palette (Ctrl+K) and a minimal mode (Ctrl+Shift+M) that leaves only
the notebook and the view. A VESTA-like crystal view (view along a, b, c, a*, ..., boundary images,
polyhedra, isosurfaces), hover readout, picking and measuring, and an OVITO-style classic layout as
an option. [Desktop GUI](docs/user/gui.md)

**Analysis.** About 70 modifiers: OVITO's structure identification (CNA, PTM, CSP, Voronoi,
Wigner-Seitz, DXA through the `ovito` package), clusters, rings, Steinhardt order, statistics and
binning, symmetry, powder XRD and S(q), trajectories (MSD, displacements, unwrapping, time series),
ion transport (van Hove, conductivity, site hops), NEB profiles, isosurfaces and grid slices,
DOS and band structures from VASP. Ported analyses are tested against OVITO and freud.
[Modifiers](docs/user/modifiers.md)

**Files.** Native lazy readers for big text trajectories (LAMMPS dump, XDATCAR, extxyz, OUTCAR,
vasprun.xml), H5MD and vaspout.h5, MDAnalysis for XTC/TRR/DCD and ASE for everything else, with
matching writers. Files on an SSH host open by path (`cluster:/scratch/run/dump.lammpstrj`); remote
computers can be saved as profiles like in an SSH terminal app (the Remotes side bar, Ctrl+Alt+R, or
`ewald remote add`), with passwords kept in the OS keychain. Runs that are still writing grow on the
timeline. [File formats](docs/user/formats.md)

**Figures.** A numba ray tracer with ambient occlusion and shadows, vector SVG/PDF/EPS, OpenGL,
journal size presets, overlays (legends, color bars, scale bars, text) and movies.
[Rendering figures](docs/user/rendering.md)

**Claude.** Ask Claude about the atoms you picked, or ask for an analysis in words: Claude works
through the same tools and its steps land in the notebook as ordinary code. `ewald ask`, `ewald
describe` and an MCP server (`ewald mcp`) do the same from a shell or Claude Code, and `llms.txt`
is an API reference for agents that write EWALD code. [Claude integration](docs/user/claude.md)

**Plugins.** Readers, writers, modifiers, visual elements, overlays, color maps and themes are all
plugins, built-in ones included. The Plugins side bar (Ctrl+Shift+X) installs them from catalogs,
and Modifier Forge has Claude write and test a new one. PLUMED and phonopy support are moving into
the `ewald-plumed` and `ewald-phonopy` plugins ([frssp/ewald-plugins](https://github.com/frssp/ewald-plugins)).
[Plugins](docs/plugins/index.md)

**Speed.** Analysis kernels run on NumPy or CuPy from one source; neighbor lists are numba or CUDA C;
the viewport draws GPU impostors and coarsens while you rotate multi-million-atom systems.
[GPU compute](docs/user/gpu.md)

## Documentation

The [manual](docs/index.md) has the [user manual](docs/user/install.md), the
[plugin guide](docs/plugins/index.md) and the [developer guide](docs/developer/architecture.md);
the GUI's Help menu opens it. What exists, generated from the code: [feature inventory](docs/features.md).
Open work: [TODO](docs/todo.md). Milestones: [roadmap](docs/ROADMAP.md). To build it as a site:
`pip install -r docs/requirements.txt && mkdocs serve`.
