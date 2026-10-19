# EWALD: Extensible Workbench for Atomic Lattices & Dynamics

EWALD (Python package `ewald`) is a notebook for atomistic structures and trajectories. A file
opens as a notebook next to a 3D view: steps add analyses, views plot their results linked to the
atoms, and the notebook is saved as a file that also runs as a plain Python script. Claude can read
the structure and add steps, and plugins add formats, analyses and drawing.

![The EWALD window](images/notebook.png)

Under the notebook is an OVITO-style pipeline in Python. A file becomes a pipeline, modifiers
analyse or change each frame, and the result is computed on the CPU or an NVIDIA GPU and rendered
to a figure:

```python
from ewald import load
from ewald.modifiers import CommonNeighborAnalysis

pipe = load("dump.lammpstrj", type_map={1: "Cu"})
pipe.add(CommonNeighborAnalysis())
data = pipe.compute(10, device="cuda")
pipe.render("frame10.png", frame=10, renderer="raytrace")
```

New here? Start with the [Quick start](user/quickstart.md) and the three tutorials:
[your first notebook](user/tutorials/first-notebook.md),
[a crystal structure figure](user/tutorials/structure-figure.md) and
[a trajectory](user/tutorials/trajectory.md).

The manual has three parts.

**[User manual](user/install.md)** is for people analysing simulations: installing, the
tutorials, notebooks, the desktop GUI, which files can be read, every modifier, Python scripting, GPU compute,
publication rendering, the Claude tools and the command line.

**[Plugins](plugins/index.md)** is for people extending ewald without changing it: reader
plugins for new file formats, your own modifiers, how to test them, and how to share them as
a pip package or a single file.

**[Developer guide](developer/architecture.md)** is for people working on ewald itself:
how the code is organised, the data model and its rules, the NumPy/CuPy backend contract,
the C++ extension, and how tests and CI run.

The [Python API reference](reference/api.md) is generated from the docstrings, and
`llms.txt` in the repository root is a compact reference written for LLMs that write ewald
code (`ewald api` prints the current version).

The docs are written in English so that one text serves the GUI Help menu, plugin authors
and the LLM reference alike.
