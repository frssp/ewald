# Architecture

ewald is a Python package (`src/ewald/`) with one C++ extension. Everything a user does,
in a script, the GUI or through Claude, goes through the same objects:

```text
file ──reader──▶ source ──▶ Pipeline ──modifiers──▶ DataCollection ──▶ render / GUI / tools
                  (lazy)    (frame cache)  (per frame, on a Backend)
```

`import_file` picks a reader and wraps it in a `FileSource`. A `Pipeline` holds the source and
a list of modifiers; `pipe.compute(frame, device=...)` reads (or takes from the cache) the
source frame, runs each enabled modifier on a `ComputeContext` that carries the array backend,
and returns a `DataCollection`. Renderers, the GUI and the Claude tools only consume that
result, which is why a figure made in the GUI can be printed as a script and an analysis asked
for in words ends up as an ordinary pipeline.

## Layout

| Path | Contents |
|---|---|
| `data.py` | `DataCollection` (particles, attributes, tables, cell) and `Cell` |
| `pipeline.py` | `Pipeline`, `Modifier`, `PythonModifier`, `ComputeContext`, the frame cache |
| `__init__.py`, `__main__.py` | `import_file` and the public names; the `ewald` CLI |
| `io/` | native lazy readers (`lammps_dump.py`, `lammps_data.py`, `xdatcar.py`), everything else through `ase_reader.py`; `base.py` has `FrameReader` and file helpers |
| `backend.py` | `Backend` and `get_backend`: NumPy or CuPy as `xp` |
| `neighbors.py` | `find_pairs`: numba CPU cell list, CUDA cell list, vectorised fallback |
| `cuda_kernels.py` | CUDA C sources launched through CuPy `RawKernel` |
| `kernels.py` | RDF, unwrapping, MSD/VACF FFTs, clustering, trajectory helpers |
| `modifiers/` | one module per family; `__init__.py`'s `__all__` is the public list and resolves OVITO names |
| `modifiers/ovito_bridge.py` | DXA and `OvitoModifier` through the optional `ovito` package |
| `_csrc/` | C++ ports of OVITO's CNA, PTM, Voronoi and CSP plus vendored ptm, voro++ and mwm_csp |
| `scene.py` | the renderer-neutral scene: `VisualElement`s (built-in Particles, Bonds, Surfaces, Vectors, Trajectory lines, Simulation cell, and plugins) turn a `DataCollection` into primitives (`Spheres`, `Cylinders`, `Lines`, `SurfaceMesh`, `Arrows`) |
| `boundary.py` | VESTA's boundary range for drawing: periodic images inside a fractional range, bonds and polyhedra completed across the cell; `scene.build(boundary=...)` runs the elements on that display copy and maps every primitive back to its real atom |
| `overlays.py` | image-space overlays for every renderer: text label, legends, scale bar, and `OverlayElement`s (Coordinate tripod, Attributes, plugins) returning 2D items |
| `colormaps.py`, `themes.py` | color maps by name (matplotlib plus `@colormap` plugins, with a grayscale check) and themes (element colors, background, cell color, radius preset; VESTA colors in `tables/colors.json`) |
| `render.py`, `raytrace.py`, `vector.py` | draw a scene: VTK/PyVista (impostors above 150k atoms), the numba ray tracer, SVG/PDF via matplotlib; each draws every primitive kind |
| `settings.py` | user preferences as a registry of `Option`s, stored as JSON with only non-default values |
| `gui/` | PySide6 main window, viewport, pipeline panel, generated parameter forms, render page, preferences, console, inspector, plots |
| `llm/` | Claude agent, tools, descriptors, narration, MCP server, and `reference.py`, the source of `llms.txt` |

### One feature, three layers

A feature that has science code, a file format and a pipeline step keeps the same module name in
each layer, so `phonon` lives in three places on purpose:

| Layer | Holds | Examples |
|---|---|---|
| `ewald/<name>.py` | the domain library: plain functions and types with no pipeline or GUI | `phonon.py`, `diffraction.py`, `electronic.py`, `plumed.py`, `bonds.py` |
| `ewald/io/<name>.py` | readers and writers for that domain's files | `io/phonon.py`, `io/neb.py`, `io/volumetric.py` |
| `ewald/modifiers/<name>.py` | `Modifier` classes that call the library inside a pipeline | `modifiers/phonon.py`, `modifiers/diffraction.py`, `modifiers/neb.py` |

Dependencies point one way: `modifiers/` imports the root module and `io/`; the root module may use
`io.base` helpers to parse files but never imports `modifiers/`, `gui/` or `llm/`. New features follow the same split; a feature
that is only a pipeline step needs only the `modifiers/` module.

## Principles

**One API.** The GUI, the CLI and the Claude tools are clients of the Python API; none of them
has analysis code of its own. A feature lands in the API first.

**Follow OVITO.** Property names, attribute names, structure type ids, colors and modifier names
match OVITO, so OVITO users and OVITO-trained LLMs are productive at once, and ported analyses
can be tested against OVITO's numbers.

**Compute once, on any device.** Compute code is written against an array module, so NumPy in
CI exercises the CUDA path. See [Compute backends](backends.md).

**Python first.** numba or C++ only for a measured hot spot, with the timing in
`benchmarks/`.

**Optional dependencies stay optional.** `gui`, `gpu`, `ovito`, `llm` and `movie` are imported
lazily, inside the functions that need them, and tests skip with `pytest.importorskip`.
Headless `compute()` and `render()` never read user settings implicitly.

## Where to start reading

`pipeline.py` and `data.py` are short and define the contracts everything else follows.
`modifiers/coloring.py` is a small complete modifier, `modifiers/trajectory.py` shows modifiers
that read other frames, and `io/xdatcar.py` a small lazy reader.
