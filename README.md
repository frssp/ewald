<img src="src/ewald/assets/icon/ewald-128.png" alt="EWALD icon" width="96" align="right">

# EWALD: Extensible Workbench for Atomic Lattices & Dynamics

EWALD (Python package `ewald`) is a notebook for atomistic structures and trajectories, with Claude
built in. You open a VASP, LAMMPS, extxyz or any other file and ask for what you want to see, or
write it yourself: the analysis becomes Python cells next to a 3D view, each step adds a modifier,
each view is a live plot linked to the atoms, and the whole thing is saved as one file that also
runs as a plain Python script. Plugins add file formats, analyses and drawing without changing
EWALD itself.

![The EWALD window: a notebook on the left, the Structure view in the middle, the View tab with display settings below](docs/images/notebook.png)

Under the notebook is an OVITO-style pipeline (`load` → `pipe.add(...)` → `pipe.compute(frame)` →
`pipe.render(...)`) that runs the same in a script, on the CPU or an NVIDIA GPU, with a built-in
ray tracer and vector output for publication figures.

## The idea

EWALD is for computational materials researchers who run VASP and LAMMPS, often on a remote
cluster, and need to go from raw output to analysis to a figure for a paper. Today that usually
means one program to look, scripts to analyse and another tool to draw, and what was clicked is
never written down. EWALD aims at OVITO Pro's level of analysis and figures, but its working unit
is a notebook, not a list of modifiers, so looking, analysing and drawing live in one reproducible
file.

- **AI first, one API.** The GUI, a script and Claude all drive the same notebook API (`load`,
  `pipe.add`, `pipe.compute`, `show`, `pipe.render`). Claude reads the structure, adds steps and
  changes the Structure view by writing ordinary cells you can read and edit; a click in the GUI is
  recorded the same way.
- **Quality for files, speed for the screen.** Rendered images, vector figures and movies put
  quality first (Draft, Standard and Publication presets; Publication by default). The interactive
  view puts speed first (level of detail, target frame rate, cache budget). Both are settings.
- **A small core, everything else a plugin.** VASP, LAMMPS and the common formats are built in but
  registered as plugins that can be switched off; phonopy, PLUMED and AiiDA live in
  [frssp/ewald-plugins](https://github.com/frssp/ewald-plugins). Plugins declare themselves in an
  `ewald-plugin.toml` manifest, as in VS Code.
- **Reuse before writing.** Analyses are ported from OVITO's MIT-licensed source and tested against
  OVITO and freud; names follow OVITO, so OVITO users find what they know.
- **Running jobs and remote files are normal.** `cluster:/scratch/run/dump.lammpstrj` opens like a
  local file, and a trajectory a job is still writing grows on the timeline.
- **One compute code for CPU and GPU.** Kernels are written once against NumPy/CuPy (NVIDIA CUDA);
  the backend is pluggable.
- **Nothing runs until you trust it.** A notebook, `.ewd` bundle or text inside a structure file
  cannot run code (or steer Claude into running it) until you trust it.

What EWALD is not: a clone of OVITO's interface, a workflow engine that submits calculations (that
is AiiDA's job, reachable through a plugin), a biomolecular viewer, or a web service; your data
stays on your machine and your cluster. Structure editing (supercells, slabs, defects) is not
decided yet and comes later, if at all.

## Install

```bash
git clone https://github.com/frssp/ewald && cd ewald
pip install -e ".[all]"     # desktop: GUI, MDAnalysis formats, OVITO's DXA
pip install -e ".[all,gpu]" # + CuPy for device="cuda" (NVIDIA)
pip install -e .            # scripts only (clusters without a display)
# every install has symmetry, MP4 export, HDF5 formats, Ask AI (Claude, OpenAI, Gemini, local
# models), kernels and SSH files; extras: gui, gpu, formats (MDAnalysis), ovito (docs/user/install.md)
```

A C++17 compiler builds the native analyses (PTM, Voronoi, CSP matching, interval CNA); without one
the install still works, minus those. On macOS run `xcode-select --install` first. Releases ship
prebuilt wheels for Linux, macOS and Windows plus a Windows installer and a macOS app
([install from a release](docs/user/install.md#from-a-release)); changes are in [CHANGELOG.md](CHANGELOG.md).
Linux system libraries, `ssh -X`, headless machines, macOS and Windows: [Install](docs/user/install.md).

## Five minutes with a notebook

```bash
ewald gui                       # the welcome page; Open example › Cu_void.lammpstrj
ewald gui dump.lammpstrj        # or open your own file
```

Opening a file makes a notebook for it. Click **+ Step** to add an analysis, **+ Plot** to plot what
it produced, and write any Python in between. This is the notebook from the
[first tutorial](docs/user/tutorials/first-notebook.md), as it is saved:

```python
# %%
from ewald import each_frame, examples, load
from ewald.modifiers import *
from ewald.paint import paint, hide
from ewald.views import show

pipe = load(examples.path("Cu_void.lammpstrj"))

# %% coordination
pipe.add(CoordinationAnalysis(cutoff=3.0))

# %% style: surface atoms, every frame
@pipe.add
@each_frame
def style(frame, data):
    paint(data, "Coordination < 12", "#C8551F")      # the legend names the condition
    hide(data, "Position.X > 8")                     # cut the block open

# %%
show(pipe, histogram="Coordination")
```

In the GUI the `pipe.add` cell is a step with a parameter form and the time it took, `show(...)` is a
live histogram (drag a range to select those atoms in the Structure view), and the style follows every
frame of the trajectory. `python cu_void.py` runs the same file without the GUI and draws the plots
with matplotlib. Every display setting of the Structure view (style, colors, radii, bonds, cell, background,
camera, frame) is code as well, one `view(pipe, style="Ball & stick", colors={"O": "red"}, direction="z")`
cell that the viewport rewrites when you change it there and that Claude's chat sets through the same
API. Cells are IPython cells (magics, `display()`, HTML/LaTeX/Markdown outputs, live output and
progress bars while a cell runs). File › Save notebook writes it as a `.py`, a Jupyter `.ipynb`, or an `.ewd` that
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
the notebook and the view. A VESTA-like crystal view (view along a, b, c, a*, ..., boundary images, replicated cells,
polyhedra, isosurfaces), hover readout, picking and measuring. [Desktop GUI](docs/user/gui.md)

**Analysis.** About 70 modifiers: OVITO's structure identification (CNA, PTM, CSP, Voronoi,
Wigner-Seitz, DXA through the `ovito` package), clusters, rings, Steinhardt order, statistics and
binning, symmetry, powder XRD and S(q), trajectories (MSD, displacements, unwrapping, time series),
ion transport (van Hove, conductivity, site hops), NEB profiles, isosurfaces and grid slices,
DOS and band structures from VASP. Ported analyses are tested against OVITO and freud.
[Modifiers](docs/user/modifiers.md)

**Runs.** EWALD reads INCAR, KPOINTS and log.lammps next to a trajectory, so it knows the run's
ensemble, temperature and time per frame; `check_run(pipe)` says whether the SCF and the relaxation
converged, from which frame the MD is equilibrated, and whether the temperature and energy behave.
`runs = load_runs("T*/XDATCAR")` puts a temperature or composition series in one notebook: every
analysis runs on all of them, plots overlay, and `arrhenius(msd(runs))` gives the activation
energy. [Runs](docs/user/runs.md)

**Workflows.** A folder of calculations (relaxed end points, a NEB, an MD restarted in `run1/ run2/`,
phonopy) opens as one notebook with a `load` line per step: File › Open from › Workflow folder…, a folder
dropped on the window, the Workflows side bar, or `workflow.describe("hpc:/scratch/study")`.
`workflow_table("pbe/", "r2scan/")` lines up the numbers of every step (energy per atom, gap, max
force, barrier) and of several workflows side by side.
AiiDA and other databases plug in as workflow sources. [Workflows](docs/user/workflows.md)

**Files.** Native lazy readers for big text trajectories (LAMMPS dump, XDATCAR, extxyz, OUTCAR,
vasprun.xml), H5MD and vaspout.h5, MDAnalysis for XTC/TRR/DCD and ASE for everything else, with
matching writers. Files on an SSH host open by path (`cluster:/scratch/run/dump.lammpstrj`); remote
computers can be saved as profiles like in an SSH terminal app (the Remotes side bar, Ctrl+Alt+R, or
`ewald remote add`), with passwords kept in the OS keychain. Runs that are still writing grow on the
timeline. [File formats](docs/user/formats.md)

**Figures.** A numba ray tracer with ambient occlusion and shadows, vector SVG/PDF/EPS, OpenGL,
journal size presets, overlays (legends, color bars, scale bars, text) and movies.
[Rendering figures](docs/user/rendering.md)

**AI.** Ask AI about the atoms you picked, or ask for an analysis in words: Claude (the default),
OpenAI, Gemini or a model on your own machine (Ollama, LM Studio) works through the same tools and its steps land in the notebook as ordinary code. `ewald ask`, `ewald
describe` and an MCP server (`ewald mcp`) do the same from a shell or Claude Code, and `llms.txt`
is an API reference for agents that write EWALD code. [AI integration](docs/user/claude.md)

**Plugins.** Readers, writers, modifiers, visual elements, overlays, color maps, themes, AI
tools and AI providers are all plugins, built-in ones included (`ewald plugins --disable vasp` switches a core
format off). The Plugins side bar (Ctrl+Shift+X) installs them from catalogs, and Modifier Forge has
the AI write and test a new one. PLUMED and phonopy force constants are the
first-party `ewald-plumed` and `ewald-phonopy` plugins in
[frssp/ewald-plugins](https://github.com/frssp/ewald-plugins).
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

## Contributing and citing

Issues and pull requests are welcome; [CONTRIBUTING.md](CONTRIBUTING.md) says how to set up, test
and where a change belongs (often a plugin), and [SECURITY.md](SECURITY.md) how to report a
vulnerability. If EWALD helps your research, please cite it with the metadata in
[CITATION.cff](CITATION.cff) (GitHub's *Cite this repository* button gives BibTeX). You can
support the project through [GitHub Sponsors](https://github.com/sponsors/frssp).

## License

EWALD is MIT licensed ([LICENSE](LICENSE)). Analyses ported from OVITO are used under the MIT
option of its GPLv3-or-MIT license, and the vendored PTM, voro++ and mwm_csp sources keep their
own licenses; [src/ewald/_csrc/README.md](src/ewald/_csrc/README.md) lists each origin.
