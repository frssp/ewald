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
data.cell.matrix               # 3x3, ROWS are the cell vectors a, b, c
data.cell.origin, data.cell.pbc, data.cell.volume
print(data.summary())          # everything above with dtypes and ranges
```

`data.take(mask)` and `data.delete(mask)` return new collections. `data.to_ase()` and
`DataCollection.from_ase(atoms)` convert to and from ASE.

## Rule: replace arrays, never write into them

Arrays in a `DataCollection` are shared with the cached input frames. Writing into one in
place (`data.positions[0] += 1`, `data["Color"][sel] = red`) silently changes the cached
frame and every later result. Build a new array and assign it:

```python
pos = data.positions.copy()
pos[:, 2] += 1.0
data.positions = pos
data["Height"] = data.positions[:, 2] - data.cell.origin[2]   # new property: plain assignment
```

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
