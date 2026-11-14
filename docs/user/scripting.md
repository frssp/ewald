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

A step that changes the structure with ASE or pymatgen returns `data.with_structure(...)`.
`DataCollection.from_ase(atoms)` starts an empty frame, so the attributes and tables of the
steps above would be gone; `with_structure` keeps them:

```python
def strain(frame, data):
    atoms = data.to_ase()
    atoms.set_cell(atoms.cell * 1.02, scale_atoms=True)
    return data.with_structure(atoms)      # attributes, tables, per-atom properties stay

pipe.add(strain)
```

Per-atom properties and bonds stay while the number of atoms is the same. When it changes
(`del atoms[i]`, `atoms.repeat(...)`), the numeric properties come back through ASE, which
carries them along; what the new structure does not carry (text properties, bonds) is dropped
with a warning that names it. `with_structure` also takes a pymatgen `Structure` or
`Molecule` and another `DataCollection`.
`data.apply(modifier)` runs one step on this frame alone and returns the result, as OVITO's
`data.apply` does.

## Frames, and structures already in memory

A pipeline indexes like a list of frames, each computed when asked for:

```python
pipe = load("XDATCAR")
pipe[0], pipe[-1]              # pipe.compute(0), the last frame; out of range raises IndexError,
                               # in compute() too
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
modifiers and scripts after them; `pipe.cache_outputs = False` turns this off. Steps that take
longer than `pipe.checkpoint_seconds` (0.02 s) also keep their own result, so changing a later
step (its cell) starts from there instead of running the whole chain
again.

These results are also kept by what they were computed from, for every pipeline in the
process: the source's files with their size and modification time, the frame, and each step's
class and settings (with the size and modification time of files a setting names). A new
pipeline that loads the same files and adds the same steps, as a notebook's Run all does,
gets them back without computing; so does an all-frames step such as `msd` over a long
trajectory, or an `@all_frames` Python step (keyed by its code and the values it reads). A
changed file or setting computes again. The per-frame results of a Python step and of the
steps after it are not kept this way, and neither are results of sources without files (a
structure built in memory).
At most 512 MiB are kept (`ewald.result_store.shared.max_bytes`); `pipe.share_results = False`
or the environment variable `EWALD_SHARED_RESULTS=0` turns it off, and `pipe.clear_caches()`
(View › Redraw from scratch) forgets what was kept for that pipeline's files.

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

`ctx.source_frame(k)` and `pipe.source_frame(k)` take the frame numbers `pipe.compute(k)` takes:
negative ones count from the end, and one past the end raises IndexError.

`ctx.input_frame(k)` gives another frame as it enters this modifier, with every modifier
above it applied (`ctx.input_frames(range(...))` for a loop over many frames). Keyword arguments become parameters:
`pipe.modifiers.append(PythonModifier(f, threshold=2.5))` calls `f(frame, data, threshold=2.5)`.

A function marked `@all_frames` reads the whole trajectory once instead, `f(frames, data)`:
`frames` (a `Frames`) gives every frame as it enters the step, and what the function sets on
`data` is kept and reused at every frame until an upstream step, the source or its parameters
change (an `AllFramesModifier`). `@each_frame` marks the default, a function run at every frame.
See [Notebooks › What runs during playback](notebooks.md#cells).

The same holds for the built-in steps that read every frame (`MeanSquaredDisplacement`,
`VelocityAutocorrelation`, ...). A function step above them counts as changed when what it
computes from changes: its code, its defaults and keyword parameters, and the variables it and
the functions it calls read, compared by value (numbers, strings, lists, dicts, arrays, objects
by their attributes). Setting `scale = 2.0` in another cell therefore gives a new MSD. Two
things to know:

- What Python cannot read counts by identity: a random generator, an open file, a pipeline, a
  class. A change inside one of those is not seen; replace the object, or add the step again.
- Arrays are compared by content, which means reading them at every `compute`. A large array
  that is read-only (`a.flags.writeable = False`, a frame's own arrays, `np.load(...,
  mmap_mode="r")`) is taken by identity instead. A step whose check takes more than 50 ms says
  so once in a warning.

A step that changes what it reads (it appends to a list at every frame) would never look
unchanged; after two computes it counts by identity as well.

## Looping over a trajectory

```python
import numpy as np

counts = [d.attributes["CommonNeighborAnalysis.counts.FCC"] for d in pipe.frames(0, None, 10)]
```

`pipe.frames(start, stop, step)` counts as a slice does: `pipe.frames(-2)` gives the last two
frames, and a `stop` past the end ends at the last frame. `trajectory_msd` and the other
`ewald.kernels` functions below read their `start`, `stop` and `step` the same way.

For quantities over the whole trajectory, prefer the trajectory modifiers or `ewald.kernels`:

```python
from ewald.kernels import trajectory_msd, trajectory_rdf, diffusion_fit

msd = trajectory_msd(pipe, device="cuda")              # table: lag, MSD, MSD <type>
gr = trajectory_rdf(pipe, cutoff=6.0, bins=300)       # time-averaged g(r)
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
`print()` always show. A picture that fails also falls back to the text. A key that is neither
one of these options nor a `render()` keyword raises when it is set (`options["qualty"]`).

## Errors

`pipe.compute(i)` raises the first error a modifier hits. `compute(i, errors="skip")` keeps
going and stores the message in `modifier.status`; the GUI works this way.

Mistakes are reported where they are made, with what was meant:

- A name the frame does not have (`data["Coord"]`, `data.tables["rdf"]`, `data.attributes[...]`,
  `table["g"]`) lists the names it has and the closest one.
- `pipe.add(CoordinationAnalysis)` (the class instead of `CoordinationAnalysis(3.0)`), a function
  step that returns something other than a `DataCollection` or nothing, and a misspelt setting
  (`m.cutof = 3.5`) raise at once.
- `render()`, `snapshot()` and `load()` name a keyword they do not know and the one meant.
- A verb that fails (`rdf(pipe, cutoff=...)` out of memory, a cancelled run) leaves the pipeline
  as it was.
- A neighbor list that cannot fit into memory raises MemoryError before it is allocated and names
  the cutoff that would fit. `ewald.backend.MEMORY_BUDGET` (bytes) sets the limit by hand, for a
  job with less memory than the machine reports.

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
