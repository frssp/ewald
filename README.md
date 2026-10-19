# ewald

[![CI](https://github.com/frssp/ewald/actions/workflows/ci.yml/badge.svg)](https://github.com/frssp/ewald/actions/workflows/ci.yml)

Atomistic structure and trajectory visualization and analysis with an OVITO-style
pipeline API in Python. Goal: an open alternative to OVITO Pro (GUI + Python
scripting + GPU), built on open libraries (ASE, VTK/PyVista, NumPy/CuPy).

Status: core, readers, pipeline, first modifiers, offscreen rendering and a first
desktop GUI are in. See [docs/ROADMAP.md](docs/ROADMAP.md).

## Install

```bash
pip install -e .            # CPU
pip install -e ".[gpu]"     # + CuPy for device="cuda" (NVIDIA)
pip install -e ".[gui]"     # + desktop viewer (PySide6, pyvistaqt, qtconsole)
```

Headless servers need an offscreen OpenGL (e.g. `apt install libosmesa6` or EGL).

## Quick start

```python
from ewald import import_file
from ewald.modifiers import CoordinationAnalysis, ColorCoding, Slice

pipe = import_file("dump.lammpstrj", type_map={1: "Cu", 2: "Zr"})   # or "XDATCAR", "OUTCAR", "POSCAR", ...
pipe.modifiers.append(CoordinationAnalysis(cutoff=3.1))
pipe.modifiers.append(ColorCoding("Coordination", cmap="coolwarm_r"))
pipe.modifiers.append(Slice(normal=(0, 1, 0), distance=20.0, inverse=True))

def my_modifier(frame, data):            # OVITO-compatible Python modifier
    data["Undercoordinated"] = data["Coordination"] < 12
pipe.modifiers.append(my_modifier)

data = pipe.compute(10, device="cuda")   # analysis on the GPU (falls back with a clear error if no CuPy)
pipe.render("frame10.png", frame=10, size=(2000, 1500), direction="iso", bonds=None)
pipe.render("frame10_rt.png", frame=10, renderer="raytrace", ao_samples=32)   # ray traced, AO + shadows
```

Desktop viewer:

```bash
ewald gui dump.lammpstrj --type-map 1=Cu      # or: ewald-gui XDATCAR
```

```python
from ewald.gui import show
show(pipe)        # open the viewer on a pipeline built in a script (in Jupyter: %gui qt first)
```

The window has a 3D viewport with a timeline underneath, a pipeline panel
(add/reorder/enable modifiers, edit parameters, edit Python script modifiers
in place), a Render workspace (ray traced, OpenGL or vector output, journal size presets,
overlays, movies), session save/load,
a data inspector (particle table with expression filter, attributes, tables)
and an IPython console where `pipeline`, `data` and `viewer` are live; changes
made in the console redraw the view.

Command line:

```bash
ewald info XDATCAR
ewald render dump.lammpstrj out.png --frame 5 --type-map 1=Cu,2=Zr --color-by "Position.Z"
```

`python examples/demo.py` builds sample LAMMPS and VASP trajectories and renders
example figures.

### Figures, movies and sessions

```python
from ewald.render import render, render_animation

data = pipe.compute(10)
render(data, "fig.png", renderer="raytrace", size=(2100, 1575), dpi=600,
       text="t = [Time] ps", type_legend=True, scale_bar=True)      # PNG with dpi metadata
render(data, "fig.svg", renderer="vector")                         # SVG/PDF/EPS: real circles, editable text
render_animation(pipe, "movie.mp4", fps=15, text="frame [Frame]")  # also .gif, .avi, .mov, frames_%04d.png
```

Output puts quality first: `quality="publication"` is the default (16 rays/pixel,
32 AO rays, shadows; SSAO and fine sphere meshes in OpenGL). `quality="standard"`
is balanced and `quality="draft"` renders half-size frames without AO or shadows
for quick previews and long test movies; any explicit `samples=`, `ao_samples=`,
`shadows=`, `antialiasing=` or `scale=` overrides the preset. The Render workspace
has the same presets (plus Custom) and a separate movie quality, with defaults in
Preferences › Rendering. The interactive viewport keeps its own speed settings.

Renderers: `opengl` (VTK, fast), `raytrace` (built-in ray tracer with ambient
occlusion and shadows) and `vector` (shaded discs and bond sticks painted far to
near, overlays stay text). Raster renderers can also write .pdf/.svg/.eps with
the image embedded. Overlays work with every renderer: `text` (expands `[Frame]`,
`[Timestep]`, `[Time]` and any attribute name), `legend` (ColorCoding color bar),
`type_legend` and `scale_bar` (a round length, or `scale_bar=10` for 10 Å).
GIF and image sequences need nothing extra; MP4/AVI/MOV use ffmpeg from
`pip install 'ewald[movie]'` or the system PATH.

Sessions (`.ewald`, JSON) hold the input file, modifiers, visual style, camera and
render settings. The GUI saves them with Ctrl+S and opens them like any file;
scripts use `ewald.session.save_session(path, pipe, camera=..., render=...)` and
`load_session(path)`. Python modifiers are stored as source and only run when
loaded with `allow_code=True` (the GUI asks first).

```bash
ewald render figure.ewald figure.pdf --renderer vector     # re-render a saved session
ewald movie dump.lammpstrj movie.gif --every 5 --fps 12 --text "t = [Time] ps" --time-per-frame 0.5
```

## Design

* `DataCollection`: per-particle arrays with OVITO property names
  (`Position`, `Particle Type`, `Particle Identifier`, `Color`, `Radius`, ...),
  plus `cell`, `attributes`, `tables`, `grids` (volumetric `VoxelGrid`s) and
  `surfaces` (display meshes such as isosurfaces).
* `Pipeline`: source + modifier list, evaluated lazily per frame; source frames are
  cached and never mutated by modifiers.
* Readers: native readers for LAMMPS dumps, LAMMPS data files and VASP
  XDATCAR; everything else through ASE (see [File formats](#file-formats)).
* Compute backends: kernels are written against an array module, so the same
  code runs on NumPy (CPU) and CuPy (CUDA). The neighbor list has a compiled,
  multithreaded numba kernel on CPU (256k-atom fcc, 3 Å cutoff: ~0.3–0.6 s vs ~5 s
  pure NumPy) and the same algorithm as a CUDA C kernel on GPU (CuPy `RawKernel`,
  with the vectorized array code as fallback, `method="xp"`). Timings:
  `python benchmarks/neighbors.py`; `ewald.kernels` adds total and
  partial RDF, trajectory unwrapping and FFT time-averaged MSD
  (`trajectory_msd(pipe, device="cuda")`).
* Rendering: VTK via PyVista, offscreen; sphere meshes for figures and GPU
  impostors above 150k atoms.

## File formats

The format is detected from the file name and content; pass `format=` to force one.
Compressed files (`.gz`, `.bz2`, `.xz`) work for the native readers.

| Format | Reader | Notes |
|---|---|---|
| LAMMPS text dump (`dump atom/custom`) | native, lazy | orthogonal, `xy xz yz` and general (`abc origin`) triclinic boxes; `x`/`xu`/`xs`/`xsu`; `c_x[1] c_x[2]` become one vector property; `dump_modify units/time yes`; partial last frame skipped with a warning; `type_map={1: "Cu"}` |
| LAMMPS data file | native | styles atomic, charge, bond/angle/molecular, full, sphere, dipole (from `Atoms # style`, inferred, or `atom_style=`); type names from `Atom Type Labels` or `Masses` comments; Velocities, image flags, Bonds (`data.bonds`); general triclinic `avec bvec cvec` |
| VASP XDATCAR | native, lazy | fixed and variable cell (NpT), VASP 6.4 `Si/hash` species labels, partial last frame skipped |
| VASP POSCAR/CONTCAR | ASE | VASP 4 and 5, selective dynamics (`Selective Dynamics`, True = free), CONTCAR velocities (`Velocity` in Å/fs) |
| VASP OUTCAR, vasprun.xml | ASE | all ionic steps with `Energy`, `Force`, `Stress` |
| extxyz, CIF, ... | ASE | anything `ase.io.read` understands |

LAMMPS values stay in the file's own unit style (`Units` attribute when the dump has it).
The test fixtures in `tests/data/` come from real LAMMPS runs (`tests/data/lammps/generate.in`)
and VASP-format files written to match VASP 6 output.

## Claude integration (`ewald.llm`)

`ewald.llm` exposes the pipeline as tools for Claude, so an analysis can be asked for in
words and still ends up as an ordinary, reproducible pipeline. Install with
`pip install -e ".[llm]"` and set `ANTHROPIC_API_KEY` (or `ant auth login`).

```python
from ewald.llm import Agent

agent = Agent.from_file("XDATCAR")            # default model: claude-opus-5-5
print(agent.ask("What happens in this run? Render the frame where it changes."))
print(agent.toolset.to_script())             # the analysis as a standalone script
```

The model never reads raw coordinates. Tools return compact descriptors instead
(composition, cell, density, minimum distance, coordination histograms, g(r) peaks,
sanity warnings) and `describe_trajectory` flags the frames where a descriptor jumps.
Renders come back as images the model can look at. These parts need no LLM:

```bash
ewald describe dump.lammpstrj --trajectory --type-map 1=Cu   # JSON summary + events
ewald narrate dump.lammpstrj --type-map 1=Cu                 # event timeline (Claude-written if available)
ewald ask XDATCAR "Is this cell reasonable? What is the Si coordination?"
ewald mcp                                                     # same tools over MCP (stdio)
claude mcp add ewald -- ewald mcp                             # use them from Claude Code
```

For agents that write ewald code instead (Claude Code, Cursor, ...), `llms.txt` in the repository
root is a compact API reference with the differences from OVITO's API (`ewald api` prints the
current version), `data.summary()` shows a frame's properties and ranges, and OVITO modifier
names such as `CommonNeighborAnalysisModifier` resolve to the ewald class.

`run_python` (model-written code executed in your process) is off unless you pass
`allow_code=True` / `--allow-code`.

## Modifiers available now

`ExpressionSelection`, `SelectType`, `InvertSelection`, `ClearSelection`,
`DeleteSelected`, `Slice`, `AssignColor`, `ColorCoding`, `ComputeProperty`,
`CoordinationAnalysis` (+ total/partial RDF), `ClusterAnalysis` (OVITO-compatible: size, COM, gyration tensor, unwrap),
`CalculateDisplacements` (+ MSD),
`Replicate`, `WrapAtoms`, `AffineTransformation`, and any Python function
`f(frame, data)`.

### Bonds and coordination polyhedra

`data.bonds` stores each bond once (OVITO's `Topology` + `Periodic Image`), so bonds
stay correct across periodic boundaries and in primitive cells smaller than twice the
bond length; `DeleteSelected`, `Slice`, `Replicate` and `WrapAtoms` keep them consistent.
`CreateBonds` finds them with a uniform cutoff, a VESTA-style pair table
(`pairs=[("Ti", "O", 2.4)]`, `*` = any type, or OVITO's `set_pairwise_cutoff`) or
covalent radii (`mode="covalent"`, VESTA's default). `EditBonds(add=[(a, b)], remove=[...])`
adds or deletes single bonds by particle identifier; in the GUI the bond tool (⌇) does it by
clicking two atoms. `CoordinationPolyhedra` puts a convex polyhedron (or a flat polygon for
planar coordination) around each center particle, colored like the center, into
`data.surfaces["coordination-polyhedra"]`. The OpenGL viewport and the ray tracer draw
bonds, polyhedra, their edges and transparency. LAMMPS data files fill `data.bonds` too.

```python
from ewald.modifiers import CreateBonds, CoordinationPolyhedra

pipe = import_file("CONTCAR")
pipe.modifiers.append(CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.4)]))
pipe.modifiers.append(CoordinationPolyhedra(center_types=["Ti"], transparency=0.3))
pipe.render("octahedra.png", renderer="raytrace", radius_scale=0.4)
```

### Trajectory analysis

These modifiers read every frame of `start:stop:step` through the modifiers above
them (one frame at a time, so lazy LAMMPS dumps and XDATCARs stay lazy), cache the
result and show it at every frame:
`MeanSquaredDisplacement` (FFT time-origin-averaged MSD, per type, unwrapped and
matched by id, Einstein diffusion coefficient), `VelocityAutocorrelation` (VACF from
`vx vy vz` or differentiated positions, Green-Kubo D), `TimeAveraging` (OVITO's
time averaging of a table, attribute or property, e.g. the time-averaged g(r)),
`TimeSeries` (attributes vs. time) and `GenerateTrajectoryLines` (particle paths
drawn as lines, plus 'Path Length', 'Net Displacement', 'Max Displacement').

```python
from ewald.modifiers import CoordinationAnalysis, MeanSquaredDisplacement, TimeAveraging

pipe = import_file("XDATCAR")
pipe.modifiers.append(MeanSquaredDisplacement(time_per_frame=0.002))   # ps between frames
pipe.modifiers.append(CoordinationAnalysis(cutoff=3.0))
pipe.modifiers.append(TimeAveraging("table:coordination-rdf"))
data = pipe.compute(0, device="cuda")
data.attributes["MeanSquaredDisplacement.diffusion_coefficient"]       # A^2/ps
data.tables["coordination-rdf[average]"]                               # time-averaged g(r)
```

### Volumetric data (CHGCAR, LOCPOT, ELFCAR, cube)

`import_file("CHGCAR")` gives the atoms plus a `VoxelGrid` in `data.grids`
(VASP CHGCAR/CHG/PARCHG/AECCAR as "Charge Density" in e/A^3 with "Magnetization",
"Spin Up", "Spin Down" for spin-polarised runs; LOCPOT "Potential"; ELFCAR "ELF";
Gaussian cube "Value", `pbc=True` for plane-wave output). `CreateIsosurface`
draws VESTA-style isosurfaces (`signed=True` adds the negative level in a second
colour, for density differences and spin densities), `GridSlice` a colour-mapped
(hkl) plane, `GridDifference` subtracts other files' grids, and `export_file`
writes `.cube` or CHGCAR. Both the OpenGL renderer and the ray tracer draw the
surfaces, translucent ones included.

```python
from ewald.modifiers import CreateIsosurface, GridDifference, GridSlice

pipe = import_file("AB/CHGCAR")
pipe.modifiers.append(GridDifference(["A/CHGCAR", "B/CHGCAR"]))       # rho_AB - rho_A - rho_B
pipe.modifiers.append(CreateIsosurface(0.002, signed=True, transparency=0.3))
pipe.modifiers.append(GridSlice(miller=(0, 0, 1), distance=0.5, cmap="RdBu_r", range=(-0.01, 0.01)))
pipe.render("ccd.png", renderer="raytrace")
spin = import_file("CHGCAR").compute().grids["charge-density"]["Spin Up"]  # numpy array
```

### Structure analysis (ported from OVITO)

`CommonNeighborAnalysis` (adaptive, interval and fixed-cutoff modes),
`PolyhedralTemplateMatching` (FCC/HCP/BCC/ICO/SC/diamond/graphene, RMSD,
orientation, elastic deformation gradient, chemical ordering),
`VoronoiAnalysis` (atomic volume, coordination, Voronoi indices, radical
tessellation, face list) and `WignerSeitzAnalysis` (vacancies/interstitials,
per-type occupancies), `IdentifyDiamond` (cubic/hexagonal diamond and their
first/second neighbors), `ChillPlus` (hexagonal/cubic/interfacial ice, hydrate),
`CentroSymmetry` (conventional and minimum-weight
matching), `AcklandJones` and `AtomicStrain` (von Mises/volumetric strain,
F, Green-Lagrange tensor, D2min, rotation, stretch). Property names, structure type ids, colors and
attribute names follow OVITO, and the test suite checks the results against
OVITO 3.16 on eleven test structures plus ice/hydrate, defect and strain cases (`tests/data/make_ovito_reference.py`).

```python
from ewald.modifiers import PolyhedralTemplateMatching, VoronoiAnalysis

pipe.modifiers.append(PolyhedralTemplateMatching(output_orientation=True))
pipe.modifiers.append(VoronoiAnalysis(compute_indices=True))
data = pipe.compute(0)
print(data.attributes["PolyhedralTemplateMatching.counts.FCC"])
ico = (data["Voronoi Index"][:, 4] == 12) & (data["Coordination"] == 12)   # <0,0,0,0,12> cells
```

With the optional `ovito` package (`pip install ovito` or `pip install -e ".[ovito]"`),
`DislocationAnalysis` runs OVITO's DXA (Burgers vectors, line lengths, segment
polylines in `tables["dislocations"]`), and `OvitoModifier(any_ovito_modifier)`
runs any other OVITO modifier inside an ewald pipeline.

CNA, PTM, Voronoi and matching-mode CSP use a C++ extension (`ewald._native`,
compiled by `pip install -e .` with pybind11; sources and licenses in
`src/ewald/_csrc/`). Installing needs a C++17 compiler (gcc/clang, Xcode command
line tools on macOS, MSVC Build Tools on Windows). For 108k atoms on 4 cores:
CNA ~1 s, PTM ~1.6 s, Voronoi ~1.7 s. With `device="cuda"` the CNA signatures
run as CuPy kernels.
