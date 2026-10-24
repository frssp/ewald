# Python scripting

The GUI and the Python API are the same pipeline: anything done in one can be done in the
other, and the Render workspace prints the script for the figure you set up.

## The data of a frame

`pipe.compute(i)` returns a `DataCollection`:

```python
data = pipe.compute(0)
data.count                     # number of particles
data.positions                 # (N, 3) array, Å
data.types                     # element names as strings
data["Coordination"]           # any per-particle property by its OVITO name
"Force" in data
data.particles                 # dict: property name -> array
data.attributes                # dict of global values ("Timestep", "Energy", analysis results)
data.tables                    # dict of tables, each a dict of columns (g(r), MSD, ...)
data.bonds                     # bonds (None until CreateBonds or a LAMMPS data file adds them)
data.surfaces                  # meshes such as coordination polyhedra
data.cell.matrix               # 3x3, ROWS are the cell vectors a, b, c
data.cell.origin, data.cell.pbc, data.cell.volume
print(data.summary())          # everything above with dtypes and ranges
```

`data.take(mask)` and `data.delete(mask)` return new collections. `data.to_ase()` and
`DataCollection.from_ase(atoms)` convert to and from ASE, `data.to_pymatgen()` and
`DataCollection.from_pymatgen(structure)` to and from pymatgen (when it is installed).
`data.apply(modifier)` runs one step on this frame alone and returns the result, as OVITO's
`data.apply` does.

## Frames, and structures already in memory

A pipeline indexes like a list of frames, each computed when asked for:

```python
pipe = load("XDATCAR")
pipe[0], pipe[-1]              # pipe.compute(0), the last frame; out of range raises IndexError
pipe[::10]                     # a list of every tenth frame
for data in pipe: ...          # every frame, one at a time
images = pipe.to_ase()         # list of ase.Atoms, e.g. for ase.io.write
export_file(pipe, "last.vasp", format="vasp", frames="last")   # or "all", an int, a slice, a list
```

`load` also takes structures you already have: an ASE `Atoms`, a pymatgen `Structure` or
`Molecule`, a `DataCollection`, or a list of them (one frame each), so
`load(ase.io.read("OUTCAR", ":"))` or `load(bulk("Cu", cubic=True) * 4)` start a pipeline too.

## Rule: replace arrays, never write into them

Arrays in a `DataCollection` are shared with the cached input frames, so the pipeline makes them
read-only: writing into one in place (`data.positions[0] += 1`, `data["Color"][sel] = red`) raises
`ValueError: output array is read-only` with a note on how to replace it. Build a new array and
assign it:

```python
pos = data.positions.copy()
pos[:, 2] += 1.0
data.positions = pos
data["Height"] = data.positions[:, 2] - data.cell.origin[2]   # new property: plain assignment
```

Computing a frame again (a second `pipe.compute(i)`, a view cell, the GUI) reuses the result of
the built-in modifiers as long as their settings did not change, and runs only the function
modifiers and scripts after them; `pipe.cache_outputs = False` turns this off.

## Function modifiers

Any function with OVITO's signature `f(frame, data)` is a modifier. It can change `data` in
place (by assigning properties as above) or return a new `DataCollection`, for example
`return data.delete(mask)`. A third parameter receives the compute context:

```python
def displacement_from_first(frame, data, ctx):
    ref = ctx.source_frame(0)              # unmodified input of another frame
    xp = ctx.xp                            # numpy, or cupy when device="cuda"
    data["dz"] = data.positions[:, 2] - ref.positions[:, 2]

pipe.modifiers.append(displacement_from_first)
```

`ctx.input_frame(k)` gives another frame as it enters this modifier, with every modifier
above it applied. Keyword arguments become parameters:
`pipe.modifiers.append(PythonModifier(f, threshold=2.5))` calls `f(frame, data, threshold=2.5)`.

A function marked `@all_frames` reads the whole trajectory once instead, `f(frames, data)`:
`frames` (a `Frames`) gives every frame as it enters the step, and what the function sets on
`data` is kept and reused at every frame until an upstream step, the source or its parameters
change (an `AllFramesModifier`). `@each_frame` marks the default, a function run at every frame.
See [Notebooks › What runs during playback](notebooks.md#cells).

## Looping over a trajectory

```python
import numpy as np

counts = [d.attributes["CommonNeighborAnalysis.counts.FCC"] for d in pipe.frames(0, None, 10)]
```

For quantities over the whole trajectory, prefer the trajectory modifiers or `ewald.kernels`:

```python
from ewald.kernels import trajectory_msd, trajectory_rdf, diffusion_fit

msd = trajectory_msd(pipe, device="cuda")              # table: lag, MSD, MSD <type>
gr = trajectory_rdf(pipe, cutoff=6.0, nbins=300)       # time-averaged g(r)
```

## In Jupyter

A pipeline, a frame or a verb table as the last line of a Jupyter cell shows a picture: the
structure (frame 0 of a pipeline) drawn by the built-in ray tracer, which needs no OpenGL or
display, so it works on a remote Jupyter server too, or the table's curve.

```python
from ewald import load, rdf
pipe = load("cu_void.lammpstrj")
pipe                     # picture of frame 0
pipe.compute(40)         # another frame
rdf(pipe, cutoff=6)      # the g(r) curve
```

`ewald.display.options` changes the picture: `size` (480x360), `renderer` (`"raytrace"` or
`"opengl"`), `quality` (`None` picks `"standard"`, or `"draft"` above 100 000 atoms) and any
other `render()` keyword such as `direction="z"` or `theme="VESTA"`.
`ewald.display.options["enabled"] = False` goes back to the text repr, which consoles and
`print()` always show. A picture that fails also falls back to the text.

## Errors

`pipe.compute(i)` raises the first error a modifier hits. `compute(i, errors="skip")` keeps
going and stores the message in `modifier.status`; the GUI works this way.

## Coming from OVITO

| OVITO | ewald |
|---|---|
| `from ovito.io import import_file` | `from ewald import import_file` |
| `pipeline.source.num_frames` | `pipe.num_frames` |
| `data.particles.count` | `data.count` |
| `data.particles.positions` | `data.positions` |
| `data.particles_.create_property('X', data=v)` | `data['X'] = v` |
| `data.particles.particle_types` (ids, type objects) | `data.types` (element strings) |
| `data.cell[:, :3]` (columns are cell vectors) | `data.cell.matrix` (rows are cell vectors) |
| `ColorCodingModifier(property=..., gradient=...)` | `ColorCoding(property, cmap="viridis")` |
| `PythonScriptModifier(function=f)` | `pipe.modifiers.append(f)` |
| `Viewport(...).render_image(...)` | `pipe.render(filename, frame=, size=, direction=)` |
