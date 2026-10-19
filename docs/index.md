# EWALD: Extensible Workbench for Atomic Lattices & Dynamics

EWALD (Python package `ewald`) is an atomistic structure and trajectory viewer and analysis library with an
OVITO-style pipeline in Python. A file becomes a pipeline, modifiers analyse or change each
frame, and the result is computed on the CPU or an NVIDIA GPU and rendered to a figure or
shown in the desktop GUI:

```python
from ewald import import_file
from ewald.modifiers import CommonNeighborAnalysis

pipe = import_file("dump.lammpstrj", type_map={1: "Cu"})
pipe.modifiers.append(CommonNeighborAnalysis())
data = pipe.compute(10, device="cuda")
pipe.render("frame10.png", frame=10, renderer="raytrace")
```

The manual has three parts.

**[User manual](user/install.md)** is for people analysing simulations: installing, the
desktop GUI, which files can be read, every modifier, Python scripting, GPU compute,
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
