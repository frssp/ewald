# Quick start

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
