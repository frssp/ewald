# Data model and pipeline

## DataCollection

One frame. `particles` maps property names to arrays whose first axis is the particle count;
`attributes` holds global values; `tables` holds tables as dicts of equal-length columns;
`cell` is a `Cell` with `matrix` (rows are the cell vectors a, b, c), `origin` and `pbc`.
Assigning `data["Name"] = value` checks the length and broadcasts scalars.

Besides particles, a frame carries things the renderers draw: `bonds` (`ewald.bonds.Bonds`),
`surfaces` (name to `ewald.mesh.SurfaceMesh`), `grids` (volumetric `VoxelGrid`s) and `vectors`
(particle property name to `ewald.arrows.VectorStyle`, set by `VectorVis`). `vectors` holds only
a style; `arrows.arrow_geometry` builds the arrows from the final particles when a renderer draws
the frame, so modifiers after `VectorVis` that delete or replicate particles need not know about it.

## Copy-on-write, and why arrays are never written in place

`DataCollection.copy()` is shallow: the dicts are new, the arrays are shared. The pipeline caches
source frames (`cache_size=8` by default) and hands each evaluation a shallow copy of the cached
frame, so copying a frame of a million atoms costs nothing. The price is one rule for all code:
**replace arrays, never write into them**. `data.positions[0] += 1` would change the cached
source frame and every later result; `pos = data.positions.copy(); pos[0] += 1;
data.positions = pos` is correct. Tests for modifiers check that the source frame is unchanged
after `compute`.

## Pipeline evaluation

`Pipeline.compute(frame, device=None, errors="raise", upto=None)`:

1. resolves the backend (`device` argument, else `pipe.device`, else CPU);
2. takes a copy of the source frame from the LRU cache, reading it on a miss;
3. runs each enabled modifier with a fresh `ComputeContext(frame, pipeline, backend,
   modifier_index=i)`, timing it into `modifier.last_time`;
4. replaces the data when a modifier returns a new collection;
5. on an exception, stores the message in `modifier.status` and either re-raises
   (`errors="raise"`) or continues (`errors="skip"`, what the GUI uses).

`cancel` takes a `threading.Event`: when another thread sets it, `compute` raises
`ComputeCancelled` before the next modifier, in `ctx.source_frame`/`ctx.input_frame`, or where a
modifier calls `ctx.check_cancel()`; that exception is never stored as a modifier error. The GUI
computes on a worker thread (`gui/compute.py`, `ComputeRunner`) and sets it when a newer frame or
parameter change supersedes a computation. The source-frame cache is guarded by a lock, so a
pipeline can be computed from two threads at once.

`upto` stops before a modifier index. `ComputeContext.input_frame(k)` uses it to evaluate
another frame through the modifiers above the running one, which is how trajectory modifiers
see upstream selections. `ComputeContext.upstream_key()` fingerprints the source, frame count,
backend and the `repr` of every enabled upstream modifier; trajectory modifiers cache their
result under it together with their own `repr`. This is one reason every constructor argument
must be stored as an attribute: `repr` is built from `parameters()`.

## Modifiers and parameters

`Modifier.parameters()` inspects the constructor signature and returns the attributes of the
same names. The GUI's parameter form (`gui/params.py`), `repr`, session code and the generated
scripts are all built from it. `ModifierList` wraps plain callables in `PythonModifier`, so
`pipe.modifiers.append(f)` works with OVITO's `f(frame, data)` signature.

## Sources

A source has `num_frames` and `compute(frame) -> DataCollection`. `FileSource` indexes one or
more files (wildcards) through `open_reader`; `StaticSource` wraps in-memory frames or ASE
`Atoms`. Readers implement `FrameReader` (`__len__`, `read(i)`); lazy readers index byte offsets
once and parse a frame on demand. Every source sets the `SourceFrame` attribute.

## Settings

`settings.py` declares every preference as an `Option(key, default, label, help, choices,
minimum, maximum, kind)` in `OPTIONS`. The Preferences dialog is generated from the registry
and the JSON file stores only values that differ from the defaults, so changing a default
reaches every user who did not override it. `EWALD_CONFIG_DIR` relocates the config folder;
tests set it per test. Only the GUI applies settings; scripts read them explicitly with
`settings.load()`.
