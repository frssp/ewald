"""A compact, always-current ewald API reference for LLMs writing ewald code.

``api_reference()`` returns Markdown: the data model and its rules, expression
syntax, the differences from OVITO's Python API (where models trained on OVITO
code go wrong), and every modifier with its exact signature, read from the
code. ``ewald api`` prints it; ``llms.txt`` in the repository root is a snapshot.
"""
from __future__ import annotations

import inspect

GUIDE = '''\
# ewald Python API (for LLMs and people writing ewald code)

ewald is an OVITO-style atomistic pipeline: `import_file` -> modifiers -> `compute(frame)` -> render.
The API mirrors OVITO's, so OVITO knowledge transfers; the differences below are where it doesn't.

```python
from ewald import import_file
from ewald.modifiers import CoordinationAnalysis, ExpressionSelection, DeleteSelected, ColorCoding

pipe = import_file("XDATCAR")                        # VASP POSCAR/CONTCAR/XDATCAR/OUTCAR, LAMMPS dump/data, extxyz, cif, ...
# pipe = import_file("dump.*.lammpstrj", type_map={1: "Cu", 2: "Zr"})   # LAMMPS numeric types -> elements
pipe.modifiers.append(CoordinationAnalysis(cutoff=3.0))
pipe.modifiers.append(ExpressionSelection("Coordination < 12"))

def mark(frame, data):                               # any f(frame, data) is a modifier (OVITO signature)
    data["Undercoordinated"] = data["Coordination"] < 12

pipe.modifiers.append(mark)
data = pipe.compute(10)                              # device="cuda" runs supported kernels on the GPU
print(data.summary())                                # look before you code: properties, ranges, attributes
pipe.render("frame10.png", frame=10, direction="iso", bonds=2.8)
```

## Data model

- `pipe.num_frames`, `pipe.compute(i)`, `pipe.frames(start, stop, step)`, `pipe.source_frame(i)` (unmodified input).
- `data = pipe.compute(i)` is a `DataCollection`: `data.count`, `data.positions` (N x 3, Angstrom),
  `data.types` (element strings), `data["Name"]` for any per-particle property, `"Name" in data`,
  `data.particles` (dict name -> array), `data.attributes` (dict of globals), `data.tables` (dict of column dicts),
  `data.cell.matrix` (3 x 3, ROWS are the cell vectors a, b, c), `data.cell.origin`, `data.cell.pbc`,
  `data.cell.volume`, `data.cell.wrap(pos)`, `data.cell.to_fractional(pos)`.
- Property names are OVITO's: "Position", "Particle Type", "Particle Identifier", "Color", "Radius",
  "Selection", "Coordination", "Structure Type", "Displacement", "Force", "Velocity".
- `data.take(mask_or_index)`, `data.delete(mask)` return new collections; `data.to_ase()`, `DataCollection.from_ase(atoms)`.
- **Never write into an array in place** (`data.positions[0] = ...`, `data["Color"][sel] = ...`): arrays are shared
  with cached frames. Build a new array and assign it: `pos = data.positions.copy(); pos[0] += 1; data.positions = pos`.
- `data.bonds` (None until `CreateBonds`, `EditBonds` or a LAMMPS data file adds them) stores each bond once:
  `data.bonds["Topology"]` (M x 2 particle indices), `data.bonds["Periodic Image"]` (M x 3), optional "Bond Type",
  "Color"; `data.bonds.vectors(data)`, `.lengths(data)`, `.neighbor_counts(N)`. `take`/`delete`/`Replicate`/`WrapAtoms`
  keep bonds consistent. `data.surfaces` maps names to `SurfaceMesh` (`vertices`, `faces`, `face_properties`,
  `edges`), e.g. `data.surfaces["coordination-polyhedra"]` from `CoordinationPolyhedra`; renderers draw both.
- A modifier function may take a third argument `ctx`: `ctx.source_frame(k)` (reference frame), `ctx.frame`,
  `ctx.xp` (numpy or cupy). Return nothing to keep `data`, or return a new DataCollection (e.g. `data.delete(mask)`).

## Expressions (ExpressionSelection, ComputeProperty, ColorCoding)

OVITO syntax evaluated with numpy: `Position.Z`, `ParticleType`, `ParticleIdentifier`, `Coordination`, `Velocity.X`,
globals `N`, `Frame`, `CellSize.X`. Combine with `&`, `|`, `~` and parentheses, not `and`/`or`/`not`:
`(Position.Z > 10) & (ParticleType == "Cu")`. Functions: `abs sqrt exp log sin cos min max`, `np.*`.

## Differences from OVITO

| OVITO | ewald |
| --- | --- |
| `from ovito.io import import_file` | `from ewald import import_file` |
| `from ovito.modifiers import CommonNeighborAnalysisModifier` | `from ewald.modifiers import CommonNeighborAnalysis` (the OVITO name also works) |
| `pipeline.source.num_frames` | `pipe.num_frames` |
| `data.particles.count` | `data.count` |
| `data.particles.positions`, `data.particles['Position']` | `data.positions`, `data['Position']` |
| `data.particles_['Position_'][...] = v` (in place) | `data.positions = v` (assign a new array) |
| `data.particles_.create_property('X', data=v)` | `data['X'] = v` |
| `data.particles.particle_types` (type objects, ids) | `data.types` (element strings) |
| `data.cell[:, :3]`: COLUMNS are the cell vectors, `data.cell[:, 3]` origin | `data.cell.matrix`: ROWS are the cell vectors, `data.cell.origin` |
| `ColorCodingModifier(property=..., gradient=...)` | `ColorCoding(property, cmap="viridis")` (matplotlib colormap name) |
| `WrapPeriodicImagesModifier()` | `WrapAtoms()` |
| `data.particles.bonds.topology`, `bonds.pbc_vectors` | `data.bonds["Topology"]`, `data.bonds["Periodic Image"]` |
| `CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)` | `CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.4)])` (`set_pairwise_cutoff` also works) |
| `PythonScriptModifier(function=f)` | `pipe.modifiers.append(f)` |
| `Viewport(...).render_image(filename=...)`, `pipeline.add_to_scene()` | `pipe.render(filename, frame=, size=, direction=)`; GUI: `ewald gui FILE` |
| `vp.render_anim(filename="movie.mp4", fps=10)` | `render_animation(pipe, "movie.mp4", fps=10, text="t = [Time] ps")` (`from ewald.render import render_animation`; .gif/.avi/.mov/PNG sequence too) |
| `TextLabelOverlay`, `ColorLegendOverlay` | `render(..., text="frame [Frame]", legend=True, type_legend=True, scale_bar=True)` |
| vector export (`.pdf`/`.svg`) | `render(data, "fig.svg", renderer="vector")` (also .pdf/.eps) |
| `TachyonRenderer(antialiasing_samples=, ambient_occlusion_samples=)` | `render(..., renderer="raytrace", quality="publication")` (default; `"draft"`/`"standard"`, or explicit `samples=`, `ao_samples=`) |
| session state (`.ovito`) | `ewald.session.save_session(path, pipe, camera=..., render=...)`, `load_session(path).pipeline` |
| `export_file(pipeline, file, format, columns=[...])` | `export_file(pipe, file, format=...)` (`"lammps/dump"` native, else ASE) |

## Other entry points

- `ewald.kernels.rdf(positions, cell, cutoff, nbins=200, types=None)` -> table with `r`, `g(r)`, partial `g(r) A-B`.
- `ewald.kernels.trajectory_msd(pipe, start=0, stop=None, step=1)` -> `lag`, `MSD`, `MSD <type>` (unwrapped, by id).
- `ewald.kernels.trajectory_vacf(pipe, time_per_frame=1.0, velocities="auto")`, `trajectory_rdf(pipe, cutoff, nbins)`
  (time-averaged g(r)), `diffusion_fit(time, msd)` -> `(D, intercept, fit window)`, `green_kubo(time, vacf)`.
- Trajectory modifiers (`MeanSquaredDisplacement`, `VelocityAutocorrelation`, `TimeAveraging`, `TimeSeries`,
  `GenerateTrajectoryLines`) read every frame of `start:stop:step` through the modifiers above them and cache the
  result, so e.g. `CoordinationAnalysis(3.0)` then `TimeAveraging("table:coordination-rdf")` gives the
  time-averaged g(r) in `tables["coordination-rdf[average]"]`.
- `ewald.llm.describe_frame(data)` / `describe_trajectory(pipe)`: compact JSON summaries and detected events.
- `ewald.llm.narrate(pipe)`: event timeline; `ewald.llm.Agent.from_file(path).ask("...")`: Claude over these tools.
'''


def _signature(cls) -> str:
    if "__init__" not in vars(cls) and not any("__init__" in vars(b) for b in cls.__mro__[1:-1]):
        return "()"
    params = []
    for p in list(inspect.signature(cls.__init__).parameters.values())[1:]:
        if p.kind is p.VAR_POSITIONAL:
            params.append(f"*{p.name}")
            continue
        if p.kind is p.VAR_KEYWORD:
            params.append(f"**{p.name}")
            continue
        text = p.name
        if p.annotation is not p.empty:
            text += f": {p.annotation if isinstance(p.annotation, str) else getattr(p.annotation, '__name__', p.annotation)}"
        if p.default is not p.empty:
            eq = " = " if p.annotation is not p.empty else "="
            text += eq + ("..." if len(repr(p.default)) > 40 else repr(p.default))  # long defaults: script templates
        params.append(text)
    return "(" + ", ".join(params) + ")"


def _doc(cls) -> str:
    text = inspect.cleandoc(cls.__doc__ or "")  # the class's own docstring, not an inherited one
    return " ".join(text.split("\n\n")[0].split())


def modifier_reference() -> str:
    """One entry per modifier exported by ``ewald.modifiers``, signature and first doc paragraph."""
    from .. import modifiers
    from ..pipeline import Modifier, PythonModifier

    lines = ["## Modifiers", ""]
    for name in modifiers.__all__:
        cls = getattr(modifiers, name)
        if not (isinstance(cls, type) and issubclass(cls, Modifier)) or cls in (Modifier, PythonModifier):
            continue
        doc = _doc(cls)
        lines.append(f"- `{name}{_signature(cls)}`" + (f": {doc}" if doc else ""))
    return "\n".join(lines) + "\n"


def api_reference() -> str:
    """The full reference: guide + live modifier list."""
    return GUIDE + "\n" + modifier_reference()
