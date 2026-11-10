"""A compact, always-current ewald API reference for LLMs writing ewald code.

``api_reference()`` returns Markdown: the data model and its rules, expression
syntax, the differences from OVITO's Python API (where models trained on OVITO
code go wrong), and every modifier with its exact signature, read from the
code. ``ewald api`` prints it; ``llms.txt`` in the repository root is a snapshot.
"""
from __future__ import annotations

import inspect
import re

GUIDE = '''\
# EWALD (Extensible Workbench for Atomic Lattices & Dynamics): `ewald` Python API (for LLMs and people writing ewald code)

ewald is an OVITO-style atomistic pipeline (source file -> modifiers -> frame -> 3D view, plots, pictures) driven
from a Python notebook. The API mirrors OVITO's, so OVITO knowledge transfers; "Differences from OVITO" below lists
where it doesn't. Write notebook cells (percent format, `# %%`), one change per cell:

```python
# %%
pipe = load("XDATCAR")                              # VASP POSCAR/CONTCAR/XDATCAR/OUTCAR/vasprun.xml/vaspout.h5/CHGCAR, LAMMPS dump/data, extxyz, cif, ...
# %%
pipe.add(CoordinationAnalysis(cutoff=3.0))          # a step cell: exactly one pipe.add(...) with literal arguments
# %%
pipe.add(ExpressionSelection("Coordination < 12"))
# %% undercoordinated [each frame]
paint(data, "Coordination < 12", "red")             # a Python step: the body sees `frame` and `data`
# %%
view(pipe, style="Ball & stick", direction="z")     # a view cell: the 3D view's settings
# %%
show(pipe, table="coordination-rdf")                # a plot cell
# %%
data = pipe.compute(0)                              # a plain code cell may read results freely
print(data.summary())                               # look before you code: properties, ranges, attributes
print(np.bincount(data["Coordination"]))            # coordination histogram as numbers
```

Rules: change the structure, the view or a result only through such cells (never by mutating `pipe` from a plain
cell or a tool), so the notebook alone reproduces the analysis. Read the output names a modifier writes (properties,
`data.tables[...]`, `data.attributes[...]`) from its entry under "Modifiers" below instead of guessing them. Units:
Angstrom, eV, ps for `time_per_frame` and trajectory times, fs for MD time steps, e/A^3 for charge densities.
Outside a notebook the same API works as a script: `pipe = import_file(path)` (= `load`), `pipe.add(m)`
(= `pipe.modifiers.append(m)`), any `def f(frame, data)` is a modifier, `pipe.render("fig.png", frame=10)`,
`data = pipe.compute(10, device="cuda")` (GPU kernels where supported). Loading variants:

```python
pipe = load("dump.*.lammpstrj", type_map={1: "Cu", 2: "Zr"})     # LAMMPS numeric types -> elements; wildcards join files
pipe = load("cluster:/scratch/run1/dump.lammpstrj")             # over SSH: host:/path, user@host:path, ssh://host/path
pipe = load("dump.lammpstrj", thermo="log.lammps")              # thermo columns -> data.attributes["Temp"], data.tables["Steps"]
pipe = load("run.npz", format="npz-trajectory")                 # format= also names reader plugins (`ewald plugins`)
```

## Data model

- `pipe.num_frames`, `pipe.compute(i)` (also `pipe[i]`, `pipe[a:b]`, `for data in pipe`; negative from the end, out of
  range raises IndexError), `pipe.frames(start, stop, step)`, `pipe.source_frame(i)` (unmodified input).
- `pipe.refresh()` picks up frames a running job appended since the file was opened (dump, XDATCAR, OUTCAR,
  extxyz; local or SSH) and returns how many were added; then `pipe.compute(-1)` is the newest frame.
- `data = pipe.compute(i)` is a `DataCollection`: `data.count`, `data.positions` (N x 3, Angstrom),
  `data.types` (element strings), `data["Name"]` for any per-particle property, `"Name" in data`,
  `data.particles` (dict name -> array), `data.attributes` (dict of globals), `data.tables` (dict of column dicts),
  `data.cell.matrix` (3 x 3, ROWS are the cell vectors a, b, c), `data.cell.origin`, `data.cell.pbc`,
  `data.cell.volume`, `data.cell.wrap(pos)`, `data.cell.to_fractional(pos)`.
- Property names are OVITO's: "Position", "Particle Type", "Particle Identifier", "Color", "Radius",
  "Selection", "Coordination", "Structure Type", "Displacement", "Force", "Velocity". "Structure Type" ids are OVITO's:
  CNA, PTM and AcklandJones use 0 OTHER, 1 FCC, 2 HCP, 3 BCC, 4 ICO, 5 SC, 6 CUBIC_DIAMOND, 7 HEX_DIAMOND, 8 GRAPHENE
  (expressions: `StructureType == 1`); IdentifyDiamond and ChillPlus number their own types (see their entries), and
  every one of them also writes `<Modifier>.counts.<TYPE>` attributes.
- `data.take(mask_or_index)`, `data.delete(mask)` return new collections; `data.to_ase()`, `DataCollection.from_ase(atoms)`.
- **Never write into an array in place** (`data.positions[0] = ...`, `data["Color"][sel] = ...`): arrays are shared
  with cached frames. Build a new array and assign it: `pos = data.positions.copy(); pos[0] += 1; data.positions = pos`.
- `data.bonds` (None until `CreateBonds`, `EditBonds` or a LAMMPS data file adds them) stores each bond once:
  `data.bonds["Topology"]` (M x 2 particle indices), `data.bonds["Periodic Image"]` (M x 3), optional "Bond Type",
  "Color"; `data.bonds.vectors(data)`, `.lengths(data)`, `.neighbor_counts(N)`. `take`/`delete`/`Replicate`/`WrapAtoms`
  keep bonds consistent. `data.surfaces` maps names to `SurfaceMesh` (`vertices`, `faces`, `face_properties`, `vertex_properties`,
  `edges`), e.g. `data.surfaces["coordination-polyhedra"]` from `CoordinationPolyhedra`; renderers draw both.
- Arrows for a vector property (forces, displacements, velocities, magnetic moments): add `VectorVis("Force", scaling=2)`
  (it fills `data.vectors`), or pass `vectors="Force"` / `{"Force": {"scaling": 2, "color": "red"}}` to `render`.
- A modifier function may take a third argument `ctx`: `ctx.source_frame(k)` (reference frame), `ctx.frame`,
  `ctx.xp` (numpy or cupy). Return nothing to keep `data`, or return a new DataCollection (e.g. `data.delete(mask)`).

## Expressions (ExpressionSelection, ComputeProperty, ColorCoding)

OVITO syntax evaluated with numpy: `Position.Z`, `ParticleType`, `ParticleIdentifier`, `Coordination`, `Velocity.X`,
globals `N`, `Frame`, `CellSize.X`. Combine with `&`, `|`, `~` and parentheses, not `and`/`or`/`not`:
`(Position.Z > 10) & (ParticleType == "Cu")`. Powers: `**` or OVITO's `^` (`Force.X^2`). OVITO's conditional
`Coordination < 12 ? Position.Z : 0` works (np.where). Functions: `abs sqrt exp log sin cos min max`, numpy ufuncs and array functions such as `np.where`, `np.isin`,
`np.linalg.norm` (no attribute access beyond those, no file or module access). A property's name drops the spaces and
other characters a name cannot hold: "Particle Type" is `ParticleType`, "SqDisp (Average)" is `SqDispAverage`; the full
name in backticks also works.
Ranking: `ExpressionSelection("ParticleType == 'Li'", top=5, by="MSD")` selects the 5 Li with the largest `MSD`
(`top=-5`: the smallest; `by` is a property or an expression), e.g. the atoms whose paths `GenerateTrajectoryLines` draws.

## Differences from OVITO

| OVITO | ewald |
| --- | --- |
| `from ovito.io import import_file` | `from ewald import import_file` |
| `from ovito.modifiers import CommonNeighborAnalysisModifier` | `from ewald.modifiers import CommonNeighborAnalysis` (the OVITO name also works) |
| `pipeline.source.num_frames` | `pipe.num_frames` |
| `data.particles.count` | `data.count` (the OVITO form also works) |
| `data.particles.positions`, `data.particles['Position']` | `data.positions`, `data['Position']` (the OVITO forms also work, read-only) |
| `data.particles_['Position_'][...] = v` (in place) | `data.positions = v` (assign a new array) |
| `data.particles_.create_property('X', data=v)` | `data['X'] = v` |
| `data.particles.particle_types` (type objects, ids) | `data.types` (element strings) |
| `ParticleType == 2` in an expression (type number) | `ParticleType == "Zr"` (names; numbers only when the names are numbers) |
| `data.cell[:, :3]`: COLUMNS are the cell vectors, `data.cell[:, 3]` origin | `data.cell.matrix`: ROWS are the cell vectors, `data.cell.origin` |
| `ColorCodingModifier(property=..., gradient=...)` | `ColorCoding(property, cmap="viridis")` (matplotlib name or `@colormap` plugin) |
| `WrapPeriodicImagesModifier()` | `WrapAtoms()` |
| `data.particles.bonds.topology`, `bonds.pbc_vectors` | `data.bonds["Topology"]`, `data.bonds["Periodic Image"]` |
| `CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)` | `CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.4)])` (`set_pairwise_cutoff` also works) |
| `PythonScriptModifier(function=f)` | `pipe.modifiers.append(f)` |
| `Viewport(...).render_image(filename=...)`, `pipeline.add_to_scene()` | `pipe.render(filename, frame=, size=, direction=)`; GUI: `ewald gui FILE` |
| `vp.render_anim(filename="movie.mp4", fps=10)` | `render_animation(pipe, "movie.mp4", fps=10, text="t = [Time] ps")` (`from ewald.render import render_animation`; .gif/.avi/.mov/PNG sequence too) |
| `TextLabelOverlay`, `ColorLegendOverlay` | `render(..., text="frame [Frame]", legend=True, type_legend=True, scale_bar=True)` |
| vector export (`.pdf`/`.svg`) | `render(data, "fig.svg", renderer="vector")` (also .pdf/.eps; a few thousand bonded atoms already make a slow, 10+ MB file and a `LargeVectorWarning`, so prefer PNG for big systems) |
| `TachyonRenderer(antialiasing_samples=, ambient_occlusion_samples=)` | `render(..., renderer="raytrace", quality="publication")` (default; `"draft"`/`"standard"`, or explicit `samples=`, `ao_samples=`, `noise_threshold=` for adaptive sampling) |
| visual elements (`data.particles.vis`) | `ewald.scene` primitives; plugin elements with `@visual`, drawn by every renderer via `render(..., visuals=["Cell axes", MyElement(...)])` |
| `CoordinateTripodOverlay`, `PythonViewportOverlay` | `render(..., overlays=["Coordinate tripod", "Attributes"])`; custom ones subclass `ewald.overlays.OverlayElement` (`@overlay`) and return 2D items |
| element color presets | `render(..., theme="VESTA")` (`"CPK"`, `"Journal"`, `"ASE"`, `"Dark slide"`, `@theme` plugins) |
| ASE-GUI look (flat outlined discs) | `view(pipe, style="Simple (ASE)")` or `render(..., atom_style="flat", theme="ASE", radius_scale=0.7)` |
| session state (`.ovito`) | `ewald.session.save_session(path, pipe, camera=..., render=...)`, `load_session(path).pipeline` |
| `export_file(pipeline, file, format, columns=[...])` | `export_file(pipe, file, format=None, frames=None)`: writer from the file name (`POSCAR`, `*.data`, `*.extxyz`, `*.xsf`, `*.h5md`, `*.xtc`, `*.cif`, ...) or `format=`; one-frame formats need `frames=[i]` |

## Notebook verbs

`from ewald import load, rdf, msd, vacf, sq, xrd, bond_angles, cna, ptm, voronoi, clusters, check_run, load_runs`
(exact signatures under "Functions" below): `pipe = load(path)` is `import_file`
(a list of files or a wildcard is one trajectory: restart segments are joined on Timestep, the later file winning,
also inside one LAMMPS dump a restarted run appended to (`pipe.source.restarts`); `join="restart"` treats every step going
back as a restart, `join="files"` joins only between files, `join="concat"` keeps every frame);
`pipe.add(m)` appends a modifier (and works as `@pipe.add` on `def f(frame, data)`); `@all_frames` on
`def f(frames, data)` makes a step that reads every frame once (`frames[k]`, `len(frames)`, slices) and whose
result (tables, attributes, properties) is reused at every frame; `@each_frame` marks the per-frame default. In a
notebook a cell headed `# %% [each frame]` (or `# %% name [all frames]`) is such a step itself: its code is the body,
with `frame`/`frames` and `data` given. A verb (`rdf`, `msd`, `cna`, ...)
adds its analysis step and returns its table (which one: "Functions" below). The `Table` is a dict of columns (`pandas.DataFrame(table)` works) with `.x`, `.plot()` and, for
counting tables, `.counts` (`cna(pipe).counts == {"OTHER": 0, "FCC": 500, ...}`); calling the verb again replaces its step.
In Jupyter a Pipeline, DataCollection or Table as a cell's last line shows a picture (ray-traced frame 0, or the
curve); `ewald.display.options` sets size/renderer/quality, `options["enabled"] = False` turns it off.

## Notebook cells (the GUI notebook, `ewald notebook`, `.ewd`)

A notebook is a percent-format `.py` file (`# %%` cells, one namespace). New notebooks import
`numpy as np`, `all_frames, each_frame, import_file, load, load_table` (from `ewald`), every modifier (`from ewald.modifiers import *`), `show, view, snapshot`
(from `ewald.views`), `paint, hide, isolate, fade, resize` (from `ewald.paint`) and `rdf, msd, vacf, sq, xrd, load_runs,
check_run, run_info, arrhenius, sensitivity, export_file` (from `ewald`); import anything else in the cell.
`pipe` is the pipeline. Write each change as its own cell, in the kinds the GUI recognizes:

- load cell: `pipe = load("XDATCAR")`.
- a file without a structure (`log.lammps`, OSZICAR, a CV or energy log, a text table of runs): `t = load_table(path)`,
  a Table to plot with `show(t, y="Temp")` or read as `t["Temp"]`; `load()` refuses it and names `load_table`.
- step cell: exactly one `pipe.add(CoordinationAnalysis(cutoff=3.0))` with literal arguments (the GUI shows a form for
  it and edits the call); several steps are several cells. A Python step is a cell headed `# %% name [each frame]`
  (body sees `frame`, `data`) or `[all frames]` (`frames`, `data`), or a function under
  `@pipe.add` and `@each_frame` / `@all_frames` (also runs as a plain script).
- view cell: exactly one `view(pipe, ...)` call (the 3D view, see below) or one `show(...)` call (a plot).
- `show(pipe, table=None, y=None, histogram=None, frame=0, bins=50, xlim=None, ylim=None, layout=None, panel=None, save=None)` plots any
  `data.tables` key (`"coordination-rdf"`, `"structure-factor"`, `"diffraction"`, `"msd"`, `"DOS"`, `"Bands"`, `"Steps"`
  (thermo), ...; `y=` one column) or a histogram of a particle property (`histogram="Coordination"`). In the GUI it is a
  live plot linked to the 3D view; `panel=True`/`False` puts it in the Plots panel or under the cell. `xlim`/`ylim` set
  the range (`xlim=(-3, 3)` for a DOS window); `save="rdf.pdf"` writes it. `show(pipe)` with no table or histogram is a
  picture of the structure, not a plot. `ewald.views.figure(...)` returns the matplotlib Figure instead.
- Comparisons go through `show` too, not hand-written matplotlib: `frame=[0, 5, -1]` draws a table (one column, pick it
  with `y=`) or a histogram of each frame (shared bins), `y=["Cu-Cu", "Cu-Zr"]` several columns of one frame (column
  names or their legend labels); `layout="overlay"` (default, one axis), `"offset"` (waterfall) or `"stack"` (panels
  sharing x). A comparison is a fixed figure in the GUI (not linked, does not follow the frame).
- `snapshot(pipe, frame=None, **render_kwargs)` (`from ewald.views import snapshot`) as a cell's last line shows a
  picture of the 3D view.
- `paint(data, where, color)`, `hide(data, where)`, `isolate(data, where)`, `fade(data, where, transparency=0.7)`,
  `resize(data, where, factor)` change "Color", "Transparency" or "Radius" inside a step (`where`: boolean mask, index
  array or expression string such as `"Coordination < 12"`); they need `data`, so use them in an `[each frame]` cell or
  `@pipe.add` function, not on `pipe`.
- Plain code cells (print, numpy, matplotlib) may read `pipe.compute(i)` freely; anything that should change the
  structure, the view or a result goes through the cells above, so the notebook alone reproduces it.
- VASP DOS/bands (vasprun.xml, vaspout.h5): `data.tables["DOS"]`, `data.tables["Bands"]` (band-path runs only: a
  relaxation or SCF on a k-mesh has none; in a workflow notebook plot the bands step, `show(bands, table="Bands")`), the attributes
  `"Fermi Energy"`, `"Band Gap"` and `data.attributes["Electronic Structure"]` (`ewald.electronic.ElectronicStructure`:
  `.pdos_of(atoms, orbitals)`, `.efermi`, `.eigenvalues`).

## Runs: input files, health check, series

`run_info(pipe)` reads the run's INCAR (else the OUTCAR/vasprun echo), KPOINTS or log.lammps: `.code`, `.kind`
(md/relax/static/neb/phonon), `.ensemble`, `.temperature` (start, end), `.timestep` (fs), `.time_per_frame` (ps),
`.notes` (what was assumed). `msd(pipe)`/`vacf(pipe)` use that `time_per_frame` unless given. `check_run(pipe)` lists
findings (SCF hitting NELM, relaxation not converged, `.equilibrated` first equilibrated frame, temperature vs target,
energy drift, LAMMPS warnings/errors, close contacts); `.ok`, `.to_dict()`. Several runs:
`runs = load_runs("T*/XDATCAR", label=r"T(\\d+)")` (sorted by the label's number); `runs.pipe`, `runs["600"].pipe`,
`runs.add(m)` (a copy per run); every verb takes `runs` and returns `RunTables` `{label: Table}` (`.plot()`,
`.attribute(name)`); `msd(pipe, by_axis=True)` adds the MSD along x, y, z (`"MSD Li z"`) and 1D coefficients
(`...diffusion_coefficient.Li.z`), which show diffusion confined to a plane or a channel; `runs["500"]` also finds the run labelled "T500"; `arrhenius(msd(runs))` fits Ea (eV) to D(T) with T
from each run's thermostat (`attribute="MeanSquaredDisplacement.diffusion_coefficient.Li"` for one species), or
`arrhenius(temperatures, values)` / `arrhenius({T: D})`: `.Ea`, `.Ea_error`, `.A`, `.r2`, `.plot()`, `.savefig(path)`.
A study over several runs (D(T), Ea, a comparison): `check_run` every run first and trim (`msd(pipe, start=check.equilibrated)`)
or leave out the runs it flags, saying which and why; runs of different length fit the same time window only with
`fit_time=(t0, t1)` (`fit_range` is a fraction of each run's lags; `.fit_start`/`.fit_end` say what was fitted);
`sensitivity(pipe, step, "fit_time", [(1, 5), (2, 8)], attribute)` shows how much a result moves with a choice; take temperatures from the input files and say where a folder
name disagrees; finish with a cell that prints every number the answer quotes.
Workflows (a folder of calculations, `aiida:...`): `workflow_table(path, ...)` (`from ewald import workflow_table`)
is a row per step and a column per number (energy, energy/atom, barrier, gap, max force, atoms, ...); several paths
compare side by side (steps matched by name, `align="position"` by order; `labels=[...]`); `t.difference("PBE")`,
`t["energy"]` (`{workflow: array}`), `t.plot("gap")`, `t.to_csv(path)`, `t.as_dict()`.

## The 3D view (`pipe.view`)

Every display setting of the structure view belongs to the pipeline: `view(pipe, **settings)` (`from ewald import view`)
sets some and returns `pipe.view`; `pipe.render()`, `snapshot(pipe)`, the Jupyter picture and the GUI viewport draw with
it, and the GUI writes viewport edits back into the notebook's `view(pipe, ...)` cell. Settings not named keep their value;
`None` resets one. `View.help()` lists them all. Every color here and in steps (`AssignColor`, `Vectors`, `paint`, ...)
takes any form of `ewald.colors`: `'red'`, `'tab:blue'`, `'#ff8800'`, `'#f80'`, `'rgb(255, 128, 0)'`, `(1, 0.5, 0)` or
`(255, 128, 0)`; a bad one fails at once:

```python
view(pipe, style="Ball & stick",              # 'Space-filling', 'Simple (ASE)', 'Points' (set radius_scale, bonds, atom_style)
     theme="VESTA", colors={"O": "red"}, radii={"Ti": 0.6}, radius_preset="ionic", oxidation_states={"Ti": 4},
     bonds=True, bond_pairs=[["Ti", "O", 0, 2.3]], bond_radius=0.12,   # bonds=False, or a cutoff (A)
     # bond_pairs rules: '*' = any type, A-B beats A-* beats *-*, dmax 0 = off, 5th item False = not through PBC, 'complete' = draw partner atoms beyond the cell
     cell=False, background="white", boundary=[-0.1, 1.1],             # VESTA boundary: periodic images
     replicate=[2, 2, 2], replicate_cell="super",                      # draw 2x2x2 cells (display only)
     direction="z", zoom=1.3, orthographic=True, frame=5,              # or camera={...} (exact, from the GUI)
     renderer="raytrace", quality="publication", text="t = [Time] ps", type_legend=True, scale_bar=True)
pipe.view.bonds = False; pipe.view.color("Sr", "green"); pipe.view.look("x").rotate(azimuth=30, elevation=15)
pipe.render("fig.png")                                    # draws with pipe.view; keywords override it for one call
pipe.view.elements["Bonds"].radius = 0.12                 # per visual element: Particles, Bonds, Surfaces, Vectors,
pipe.view.elements["Particles"].enabled = False           #   Trajectory lines, Simulation cell, plugin visuals
pipe.view.surface("coordination-polyhedra", color="teal", transparency=0.5)   # polyhedra/isosurface look by name
view(pipe, transparency={"O": 0.6}, selection=True)       # fade types; selected atoms red (False: off, or a color)
view(pipe, labels="{type}{index}")                        # atom labels: True, a template ({id}, {Charge:.2f}) or a dict
view(pipe, light={"azimuth": 60, "elevation": 20, "intensity": 1.2})   # key light relative to the viewer
pipe.view.measure(0, 5); pipe.view.measure(4, 0, 7)       # distance / angle drawn with its value; returns it
view(pipe, color_by="Coordination", colormap="plasma", color_range=[2, 5])   # display-only color map; 'type': type colors
pipe.add(GenerateTrajectoryLines())                       # atom paths are a step (the selected atoms, else all) ...
view(pipe, trajectory_trail=True, trajectory_color_by="Time", trajectory_width=3)   # ... the trajectory_* settings
                                                          # only style its lines (without the step: a warning, no lines)
view(pipe, tab="Top", direction="z")    # a view tab: another 3D viewport tab, the main view plus its own settings
pipe.render("top.png", tab="Top")       # draws that tab (pipe.views["Top"]; snapshot(pipe, tab="Top"))
```

A NEB folder (`load("neb/")`, images 00..NN as frames) comes with `UnwrapTrajectories`, `NebEnergyProfile` (tables
"NEB", "NEB spline"; attributes `NEB.Barrier`, `NEB.Saddle Image`, ...) and `OverlayFrames` (the other images as
see-through atoms). For lines along the band add `GenerateTrajectoryLines()` as above; `trajectory_trail=True` then
grows them image by image with the timeline.

What colors the atoms: structure identification steps (`CommonNeighborAnalysis`, `PolyhedralTemplateMatching`,
`IdentifyDiamond`, ...) write a "Color" per structure type unless `color_by_type=False`, `ColorCoding` maps a
property, `paint` colors a condition; each records its rule, which `ewald.color_rule.describe(data)` (and the
`describe_frame`/`get_view` tools' `atom_colors`) spells out with every color. Pictures draw it as a legend.

Per-atom colors, sizes and hiding by condition are steps (`ewald.paint`: `paint`, `hide`, `fade`, `resize`), and
polyhedra, isosurfaces and vector arrows come from modifiers (`CoordinationPolyhedra`, `VectorVis`, ...).
The AI's `set_view` / `get_view` tools take the same settings.

## Other entry points

- `ewald.kernels.rdf(positions, cell, cutoff, bins=200, types=None)` -> table with `r`, `g(r)`, partial `g(r) A-B`.
- `ewald.diffraction.powder_peaks(positions, types, cell, wavelength="CuKa", radiation="xray", two_theta_range=(0, 90))`
  -> `2theta`, `Intensity` (max 100), `d`, `h`, `k`, `l`, `Multiplicity` (pymatgen XRDCalculator/NDCalculator formalism);
  `powder_profile(peaks, two_theta, fwhm, eta)`; `structure_factor(positions, cell, q_max, bins, q_min, method,
  types, weighting)` -> `q`, `S(q)`; `load_pattern(path)` reads a measured two-column pattern.
- `ewald.kernels.trajectory_msd(pipe, start=0, stop=None, step=1)` -> `lag`, `MSD`, `MSD <type>` (unwrapped, by id).
- `ewald.kernels.trajectory_vacf(pipe, time_per_frame=1.0, velocities="auto")`, `trajectory_rdf(pipe, cutoff, bins)`
  (time-averaged g(r)), `diffusion_fit(time, msd)` -> `(D, intercept, fit window)`, `green_kubo(time, vacf)`.
- Trajectory modifiers (`MeanSquaredDisplacement`, `VelocityAutocorrelation`, `TimeAveraging`, `TimeSeries`,
  `GenerateTrajectoryLines`) read every frame of `start:stop:step` through the modifiers above them and cache the
  result, so e.g. `CoordinationAnalysis(3.0)` then `TimeAveraging("table:coordination-rdf")` gives the
  time-averaged g(r) in `tables["coordination-rdf[average]"]`. Per atom: `MeanSquaredDisplacement` writes the
  property 'MSD' (each atom's own time-averaged MSD over the fit window), `GenerateTrajectoryLines` 'Path Length',
  'Net Displacement', 'Max Displacement'; rank them with `ExpressionSelection(..., top=n, by=...)`, never by hand-written ids.
- Ion transport modifiers work the same way: `IonicConductivity(temperature, time_per_frame, charges)` (collective
  charge MSD -> `IonicConductivity.conductivity` in S/cm, `.nernst_einstein_conductivity`, `.haven_ratio`; charges
  from the dict, the 'Charge' property or guessed oxidation states), `VanHoveCorrelation(times, part)`,
  `SelfIntermediateScattering(q)` (F_s(q,t), `.relaxation_time`), `NonGaussianParameter()` (alpha_2) and
  `SiteHopping(sites, site_radius)` (hops, residence times, jump D). Put `SelectType("Li")` above them with
  `only_selected=True` to analyze one species; `time_per_frame` is in ps.
- Volumetric files (`CHGCAR`, `CHG`, `PARCHG`, `AECCAR*`, `LOCPOT`, `ELFCAR`, `*.cube`): `import_file` gives the atoms
  plus `data.grids[name]` (a `VoxelGrid`: `.shape`, `.domain` (a Cell), `grid["Charge Density"]` (e/A^3, already divided
  by the cell volume), `"Magnetization"`, `"Spin Up"`/`"Spin Down"`, `"Potential"`, `"ELF"`, cube `"Value"`,
  components `"Magnetization.Z"`, `.integrate(name)`, `.interpolate(name, points)`). Grid names: "charge-density",
  "potential", "elf", "cube". `CreateIsosurface`/`GridSlice` put `SurfaceMesh`es into `data.surfaces`, which
  `render()` draws. Isolevels are absolute values in the property's units (VESTA's default style: e/A^3).
- Atom sizes: `render(..., radius_preset=)` picks the default radii ("covalent" Cordero 2008, "atomic" Slater 1964,
  "atomic (calc.)" Clementi 1967, "vdW" Alvarez 2013, "vdW (Bondi)", "ionic"/"ionic (crystal)" Shannon 1976), times
  0.8 x `radius_scale`. Ionic radii need oxidation states and CN: pass `oxidation_states={"Ti": 4, "O": -2}` and
  `coordination={"Ti": 6}`, else they are guessed (type names like "Fe3+", charges, charge balance; CN from the
  structure); `ewald.radii.resolve(data, "ionic").summary()` shows what was used. A "Radius" property or
  `type_radii` wins.
- `ewald.llm.describe_frame(data)` / `describe_trajectory(pipe)`: compact JSON summaries and detected events.
- `ewald.llm.narrate(pipe)`: event timeline; `ewald.llm.Agent.from_file(path, provider="anthropic").ask("...")`: an AI model (Claude, OpenAI, Gemini, a local server) over these tools.
'''


#: Notebook recipes for common workflows: (title, file the recipe loads, cells). ``tests/test_llm.py``
#: runs every one on a shipped example file, so they stay correct.
RECIPES = [
    ("Crystal structure: bonds, TiO6 octahedra, space group", "POSCAR", """\
# %%
pipe = load("POSCAR")
# %%
pipe.add(CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.4)]))
# %%
pipe.add(CoordinationPolyhedra(center_types=["Ti"]))
# %%
pipe.add(SymmetryAnalysis())
# %%
view(pipe, style="Ball & stick", direction="iso", type_legend=True)
# %%
data = pipe.compute(0)
print(data.attributes["SymmetryAnalysis.space_group"], data.attributes["CreateBonds.num_bonds"])
"""),
    ("MD trajectory: diffusion coefficient", "XDATCAR", """\
# %%
pipe = load("XDATCAR")
# %%
pipe.add(MeanSquaredDisplacement(time_per_frame=0.01))   # ps between stored frames (run_info(pipe) reads it from INCAR)
# %%
show(pipe, table="msd")
# %%
data = pipe.compute(0)
D = data.attributes["MeanSquaredDisplacement.diffusion_coefficient"]   # A^2/ps; per type: .diffusion_coefficient.<type>
print(f"D = {D:.3g} A^2/ps = {D * 1e-4:.3g} cm^2/s")
# %%
pipe.add(ExpressionSelection("ParticleType == 'Si'", top=5, by="MSD"))   # the 5 Si with the largest per-atom 'MSD'
# %%
pipe.add(GenerateTrajectoryLines())                       # paths of the selected atoms
# %%
view(pipe, trajectory_color_by="Time", selection=False)
"""),
    ("Defects in a crystal: structure types, show only non-FCC atoms", "dump.lammpstrj", """\
# %%
pipe = load("dump.lammpstrj")
# %%
pipe.add(CommonNeighborAnalysis())
# %% defects [each frame]
isolate(data, "StructureType != 1")
# %%
data = pipe.compute(0)
print({k: v for k, v in data.attributes.items() if k.startswith("CommonNeighborAnalysis.counts.")})
"""),
    ("NEB band: barrier and energy profile", "neb/", """\
# %%
pipe = load("neb/")          # images 00..NN as frames; UnwrapTrajectories, NebEnergyProfile, OverlayFrames come with it
# %%
show(pipe, table="NEB")
# %%
data = pipe.compute(0)
print(data.attributes["NEB.Barrier"], data.attributes["NEB.Saddle Image"])   # eV, image index
"""),
    ("Charge density: isosurface and a lattice-plane slice", "CHGCAR", """\
# %%
pipe = load("CHGCAR")
# %%
pipe.add(CreateIsosurface(isolevel=0.3))       # e/A^3, absolute
# %%
pipe.add(GridSlice(miller=(1, 1, 0), distance=0.5))
# %%
view(pipe, cell=True, direction="x")
# %%
grid = pipe.compute(0).grids["charge-density"]
print(grid.shape, grid.integrate("Charge Density"))
# %% planar average along c [each frame]
from ewald.potential import planar_average
p = planar_average(data, grid="charge-density", property="Charge Density")   # LOCPOT: the defaults
data.tables["planar-average"] = {"z": p.z, "Charge Density": p.values}      # a table, so show() plots it
# %%
show(pipe, table="planar-average")
"""),
    ("Atoms picked at one frame, followed through the trajectory", "XDATCAR", """\
# %%
pipe = load("XDATCAR")
# %%
pipe.add(CalculateDisplacements(reference_frame=0))
# %%
pipe.add(FreezeProperty("Displacement Magnitude", "Moved", freeze_at=-1))   # the last frame's value, on every frame
# %% moved atoms [each frame]
isolate(data, "Moved >= 0.3")                     # the same atoms on every frame
paint(data, "Moved >= 0.3", "crimson", label="moved > 0.3 A")   # label: its name in the color legend
# %%
data = pipe.compute(0)
print(int((data["Moved"] >= 0.3).sum()), "atoms moved more than 0.3 A by the last frame")
"""),
]

#: Headings of the modifier list, by the module that defines a modifier (``ewald.modifiers.<module>``).
FAMILIES = [
    ("Selection, coloring and editing", ("selection", "coloring", "transform", "script", "datasets")),
    ("Neighbors, bonds and topology", ("analysis", "cluster", "bonds", "topology")),
    ("Structure identification and local order",
     ("structure", "local_order", "crystal", "steinhardt", "voronoi", "wigner_seitz", "symmetry", "ovito_bridge")),
    ("Trajectories, transport, NEB and phonons", ("trajectory", "transport", "neb", "phonon")),
    ("Statistics and diffraction", ("statistics", "diffraction")),
    ("Grids (CHGCAR, LOCPOT, cube) and vector arrows", ("volume", "vectors")),
]

#: Notebook functions whose exact signature the reference lists: (module, names).
FUNCTIONS = [
    ("ewald", ("load", "import_file", "export_file", "each_frame", "all_frames", "view", "rdf", "msd", "vacf", "sq",
               "xrd", "bond_angles", "cna", "ptm", "voronoi", "clusters", "run_info", "check_run", "load_runs",
               "arrhenius", "sensitivity", "workflow_table")),
    ("ewald.views", ("show", "snapshot", "figure")),
    ("ewald.paint", ("paint", "hide", "isolate", "fade", "resize")),
]

# sentences of a docstring worth keeping beyond its first paragraph: what a modifier writes, its ids, its units
_OUTPUTS = re.compile(r"\b(Writes|writes|Sets|sets|Adds|adds|Stores|stores|Fills|fills|Outputs?|ids)\b|tables\[|attributes?\b")
_UNITS = re.compile(r"\b(Angstrom|A\^2|A\^3|1/A|ps|fs|eV|THz|S/cm|e/A\^3|degrees?|cm\^-1)\b|Å")


def _plain(text: str) -> str:
    """reST to Markdown: ``x`` -> `x`, :func:`~a.b.c` -> `c`."""
    text = re.sub(r":\w+:`~?(?:[\w.]+\.)?([^`]+)`", r"`\1`", text)
    return text.replace("``", "`")


def _signature(obj) -> str:
    if isinstance(obj, type):
        if "__init__" not in vars(obj) and not any("__init__" in vars(b) for b in obj.__mro__[1:-1]):
            return "()"
        params = list(inspect.signature(obj.__init__).parameters.values())[1:]
        ret = ""
    else:
        sig = inspect.signature(obj)
        params = list(sig.parameters.values())
        ret = sig.return_annotation
        ret = "" if ret is sig.empty else " -> " + (ret if isinstance(ret, str) else getattr(ret, "__name__", str(ret)))
    out = []
    star = False
    for p in params:
        if p.kind is p.VAR_POSITIONAL:
            out.append(f"*{p.name}")
            star = True
            continue
        if p.kind is p.VAR_KEYWORD:
            out.append(f"**{p.name}")
            continue
        if p.kind is p.KEYWORD_ONLY and not star:
            out.append("*")
            star = True
        text = p.name
        if p.annotation is not p.empty:
            text += f": {p.annotation if isinstance(p.annotation, str) else getattr(p.annotation, '__name__', p.annotation)}"
        if p.default is not p.empty:
            eq = " = " if p.annotation is not p.empty else "="
            text += eq + ("..." if len(repr(p.default)) > 40 else repr(p.default))  # long defaults: script templates
        out.append(text)
    return "(" + ", ".join(out) + ")" + ret


def _doc(obj) -> str:
    text = inspect.cleandoc(obj.__doc__ or "")  # the object's own docstring, not an inherited one
    return _plain(" ".join(text.split("\n\n")[0].split()))


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=\.)\s+(?=[A-Z`])", text.rstrip(":"), maxsplit=1)[0]


def _details(cls) -> str:
    """The sentences after the first paragraph that name outputs (properties, tables, attributes, type
    ids) or give a parameter's units: what a model needs to chain steps without guessing names."""
    text = " ".join(" ".join(inspect.cleandoc(cls.__doc__ or "").split("\n\n")[1:]).split())
    params = set(inspect.signature(cls.__init__).parameters) - {"self"} if "__init__" in dir(cls) else set()
    keep = []
    for sentence in re.split(r"(?<=[.])\s+(?=[A-Z`'(])", text):
        names = set(re.findall(r"``(\w+)``", sentence))
        if _OUTPUTS.search(sentence) or (_UNITS.search(sentence) and names & params):
            keep.append(sentence)
    return _plain(" ".join(keep))


def recipes_reference() -> str:
    """The workflow recipes as notebook cells."""
    lines = ["## Recipes (notebook cells)", ""]
    for title, _, cells in RECIPES:
        lines += [f"{title}:", "", "```python", cells.rstrip("\n"), "```", ""]
    return "\n".join(lines)


def function_reference() -> str:
    """Exact signatures of the notebook functions, read from the code."""
    import ewald
    from .. import paint, views

    modules = {"ewald": ewald, "ewald.views": views, "ewald.paint": paint}   # no import by name
    lines = ["## Functions", ""]
    for module, names in FUNCTIONS:
        mod = modules[module]
        lines.append(f"`from {module} import {', '.join(names)}`:")
        lines.append("")
        for name in names:
            obj = getattr(mod, name)
            doc = _first_sentence(_doc(obj))
            lines.append(f"- `{name}{_signature(obj)}`" + (f": {doc}" if doc else ""))
        lines.append("")
    return "\n".join(lines)


def modifier_reference() -> str:
    """One entry per modifier exported by ``ewald.modifiers``, grouped by family: signature, first doc
    paragraph and what it writes."""
    from .. import modifiers
    from ..pipeline import Modifier, PythonModifier

    groups: dict[str, list[str]] = {title: [] for title, _ in FAMILIES}
    family = {module: title for title, mods in FAMILIES for module in mods}
    for name in modifiers.__all__:
        cls = getattr(modifiers, name)
        if not (isinstance(cls, type) and issubclass(cls, Modifier)) or cls in (Modifier, PythonModifier):
            continue
        title = family.get(cls.__module__.rsplit(".", 1)[-1], "Other")
        entry = f"- `{name}{_signature(cls)}`"
        doc, details = _doc(cls), _details(cls)
        entry += (f": {doc}" if doc else "") + (f" {details}" if details else "")
        groups.setdefault(title, []).append(entry)
    lines = ["## Modifiers", "",
             "`from ewald.modifiers import *` (new notebooks do it); OVITO names (`XxxModifier`) resolve too. "
             "Add one with `pipe.add(Name(...))` in its own cell.", ""]
    for title, entries in groups.items():
        if entries:
            lines += [f"### {title}", ""] + entries + [""]
    moved = ", ".join(f"`{name}`" for name in modifiers.MOVED)
    if moved:
        packages = ", ".join(f"`{pkg}`" for pkg in modifiers.PLUGIN_PACKAGES.values())
        lines += ["### In plugins", "",
                  f"{moved} moved to first-party plugins ({packages}, frssp/ewald-plugins): they import from "
                  "`ewald.modifiers` once the plugin is installed (Plugins panel); installed plugin modifiers are "
                  "listed at the end of this reference when it is read inside ewald.", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def plugin_reference() -> str:
    """Installed modifier plugins and plugin functions (:func:`ewald.plugins.api_function`), in the
    same format (not part of llms.txt)."""
    from .. import plugins

    found = plugins.modifiers()
    lines = ["## Modifier plugins installed here", ""] if found else []
    for name, plugin in found.items():
        doc = _doc(plugin.cls)
        lines.append(f"- `{name}{_signature(plugin.cls)}`" + (f": {doc}" if doc else "") + f" [{plugin.origin}]")
    by_module: dict[str, list] = {}
    for plugin in plugins.functions().values():
        by_module.setdefault(plugin.module, []).append(plugin)
    if by_module:
        lines += ([""] if lines else []) + ["## Functions from plugins installed here", ""]
    for module, entries in by_module.items():
        lines += [f"`from {module} import {', '.join(p.name for p in entries)}`:", ""]
        for p in entries:
            doc = _doc(p.function)
            lines.append(f"- `{p.name}{_signature(p.function)}`" + (f": {doc}" if doc else ""))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n" if lines else ""


def api_reference(with_plugins: bool = False) -> str:
    """The full reference: guide, recipes, function signatures, the live modifier list (+ installed
    modifier plugins)."""
    text = "\n".join([GUIDE, recipes_reference(), function_reference(), modifier_reference()])
    plugins_text = plugin_reference() if with_plugins else ""
    return text + ("\n" + plugins_text if plugins_text else "")


# ------------------------------------------------------------------------------------------- search
_WORD = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")

#: Words people search an entry by that its own reference text lacks: ``{entry: phrases}``, an entry
#: being a modifier, a notebook function or a recipe title. A phrase counts when the query holds all
#: of its words, and then weighs like a word of the entry's name. They go into the search index
#: only, never into the printed reference (``llms.txt`` stays as it is). Be accurate, not generous:
#: a wrong phrase sends the model to the wrong step (``tests/test_api_review_search.py`` checks the
#: keys and a list of queries). Write them in lower case ("x-ray": a lone lower-case letter is no word).
SYNONYMS: dict[str, tuple[str, ...]] = {
    # --- selection, coloring and editing
    "AssignColor": ("uniform color", "single color", "one color", "same color"),
    "ColorCoding": ("colormap", "color map", "color scale", "color by property", "color by value", "heatmap",
                    "color gradient", "color atoms by"),
    "ClearSelection": ("deselect", "unselect", "select nothing", "reset selection"),
    "DeleteSelected": ("remove atoms", "remove particles", "delete atoms", "remove selected"),
    "ExpressionSelection": ("select atoms", "select by condition", "select by property", "select by value",
                            "select by threshold", "select where", "filter"),
    "InvertSelection": ("complement", "opposite selection", "select the rest"),
    "SelectType": ("select element", "select species", "select atoms", "by element", "by species"),
    "Slice": ("cut", "clip", "crop", "cross section", "cutting plane", "half space", "cut plane"),
    "AffineTransformation": ("rotate atoms", "rotate cell", "rotate structure", "rotation matrix", "translate atoms",
                             "translate structure", "shift atoms", "shift origin", "scale cell", "rescale cell",
                             "strain cell", "deform cell", "shear cell", "mirror", "transformation matrix"),
    "ComputeProperty": ("new property", "derived property", "custom property", "calculate property",
                        "define property", "property from expression"),
    "Replicate": ("supercell", "repeat cell", "repeat structure", "repeat box", "repeat lattice", "repetition",
                  "periodic images", "periodic copies", "duplicate cell", "multiply cell", "expand cell",
                  "enlarge cell", "bigger cell", "larger cell"),
    "WrapAtoms": ("fold", "pbc", "periodic boundary", "wrap coordinates", "inside the box", "into the box"),
    # with a selection word each: "first shell" alone is a coordination number question
    "ExpandSelection": ("select neighbors", "select neighbor shell", "select coordination shell",
                        "select first shell", "select surroundings", "extend selection", "select atoms around"),
    "LoadTrajectory": ("topology and trajectory", "separate trajectory file", "data file and dump",
                       "trajectory onto topology"),
    "CombineDatasets": ("merge", "combine structures", "combine files", "join structures", "add another structure",
                        "second file", "insert molecule"),
    # --- neighbors, bonds and topology
    "CalculateDisplacements": ("displacement vector", "moved atoms", "distance moved", "how far", "atomic displacement",
                               "compare with initial", "relaxation displacement"),
    "CoordinationAnalysis": ("rdf", "pair distribution", "pair correlation", "pdf", "coordination number", "cn",
                             "number of neighbors", "neighbor count", "radial distribution"),
    "ClusterAnalysis": ("connected components", "fragments", "aggregates", "molecule identification", "largest cluster",
                        "cluster size", "cluster size distribution", "group atoms"),
    "CreateBonds": ("connectivity", "find bonds", "draw bonds", "show bonds", "bond cutoff", "bond search",
                    "nearest neighbor bonds", "make bonds"),
    "EditBonds": ("add bond", "remove bond", "delete bond", "break bond", "manual bond"),
    "CoordinationPolyhedra": ("polyhedron", "polyhedral", "octahedra", "octahedron", "tetrahedra", "tetrahedron",
                              "coordination environment"),
    "BondAngleDistribution": ("adf", "angular distribution", "angle histogram", "three body"),
    "BondLengthDistribution": ("bond distance", "bond distance distribution", "interatomic distance",
                               "length histogram"),
    "FindRings": ("ring statistics", "ring size distribution", "ring distribution", "membered rings",
                  "network topology", "shortest path rings"),
    # --- structure identification and local order
    "CommonNeighborAnalysis": ("cna", "structure identification", "identify crystal structure", "structure type",
                               "local structure", "local crystal structure", "crystalline atoms", "crystallinity",
                               "phase identification", "identify fcc", "identify hcp", "identify bcc"),
    "PolyhedralTemplateMatching": ("ptm", "structure identification", "identify crystal structure", "structure type",
                                   "local structure", "local crystal structure", "crystalline atoms", "crystallinity",
                                   "phase identification", "lattice orientation", "crystal orientation",
                                   "chemical ordering", "identify fcc", "identify hcp", "identify bcc"),
    "VoronoiAnalysis": ("atomic volume", "volume per atom", "tessellation", "voronoi polyhedron", "voronoi polyhedra",
                        "voronoi index", "voronoi cell", "voronoi coordination"),
    "WignerSeitzAnalysis": ("vacancy", "interstitial", "point defect", "frenkel pair", "antisite", "defect count",
                            "site occupancy"),
    "DislocationAnalysis": ("dxa", "burgers vector", "dislocation line", "dislocation density",
                            "dislocation length", "line defect"),
    "AcklandJones": ("bond angle analysis", "structure identification", "identify crystal structure", "structure type"),
    "AtomicStrain": ("local strain", "shear strain", "von mises", "volumetric strain", "strain tensor",
                     "deformation gradient", "nonaffine", "non-affine", "plastic deformation", "shear band",
                     "shear transformation"),
    "CentroSymmetry": ("csp", "defect detection", "inversion symmetry"),
    "ChillPlus": ("ice", "cubic ice", "hexagonal ice", "clathrate", "hydrate", "water structure", "ice nucleation"),
    "IdentifyDiamond": ("diamond structure", "diamond lattice", "zincblende", "wurtzite", "lonsdaleite",
                        "structure identification"),
    "SteinhardtOrderParameters": ("bond orientational order", "bond order parameter", "boo", "local order"),
    "SymmetryAnalysis": ("space group", "point group", "crystal system", "symmetry operations", "asymmetric unit",
                         "inequivalent atoms", "unique atoms", "hermann mauguin", "international symbol",
                         "find symmetry", "crystallographic"),
    "ElasticStrain": ("lattice strain", "lattice distortion", "strain tensor", "strain field", "elastic deformation"),
    "GrainSegmentation": ("grain size", "grain boundary", "grain orientation", "grain count",
                          "polycrystalline", "nanocrystalline", "misorientation"),
    "IdentifyFCCPlanarFaults": ("stacking fault", "twin", "twin boundary", "twinning", "isf", "planar defect"),
    # --- trajectories, transport, NEB and phonons
    "MeanSquaredDisplacement": ("msd", "diffusion", "diffusivity", "diffusion coefficient", "self diffusion",
                                "tracer diffusion", "einstein relation"),
    "VelocityAutocorrelation": ("vacf", "velocity correlation", "green kubo", "velocity auto correlation"),
    "TimeAveraging": ("time average", "time averaged", "average over frames", "average over trajectory",
                      "average over time", "ensemble average", "trajectory average"),
    "TimeSeries": ("time evolution", "versus time", "vs time", "over time", "as a function of frame", "per frame value",
                   "evolution"),
    "GenerateTrajectoryLines": ("atom path", "trace", "trail", "track atoms", "diffusion path", "migration path",
                                "draw trajectory"),
    "NebEnergyProfile": ("nudged elastic band", "migration barrier", "energy barrier", "activation barrier",
                         "minimum energy path", "mep", "saddle point", "transition state", "reaction path"),
    "OverlayFrames": ("superimpose", "ghost atoms", "all images at once", "several frames at once",
                      "all frames at once"),
    "PhononMode": ("vibration", "vibrational mode", "normal mode", "eigenvector", "eigenmode", "lattice vibration",
                   "animate phonon"),
    "FreezeProperty": ("keep property", "initial value", "persistent selection", "same atoms every frame",
                       "fixed selection"),
    "UnwrapTrajectories": ("unwrap coordinates", "periodic jumps", "pbc", "continuous trajectory"),
    "IonicConductivity": ("ion conductivity", "ion transport", "charge transport", "solid electrolyte",
                          "superionic", "einstein helfand"),
    "VanHoveCorrelation": ("self correlation", "distinct correlation", "displacement distribution",
                           "jump distance distribution", "hop distance"),
    "SelfIntermediateScattering": ("sisf", "isf", "incoherent scattering", "relaxation time", "structural relaxation",
                                   "alpha relaxation", "glassy dynamics"),
    "NonGaussianParameter": ("alpha2", "dynamic heterogeneity", "dynamical heterogeneity", "heterogeneous dynamics",
                             "caging"),
    "SiteHopping": ("jump", "jump rate", "jump frequency", "hopping rate", "hop rate", "ion migration",
                    "jump diffusion", "migration events", "residence time", "dwell time"),
    # --- statistics and diffraction
    "Histogram": ("property distribution", "value distribution", "distribution of values", "distribution of property",
                  "frequency distribution", "probability density"),
    "SpatialBinning": ("density profile", "concentration profile", "number density", "profile along",
                       "temperature profile", "layer average", "spatial average", "slab average", "density map",
                       "average per bin"),
    "ReduceProperty": ("mean value", "average value", "mean of property", "average of property", "sum", "median",
                       "standard deviation", "variance", "minimum value", "maximum value", "property statistics"),
    "PowderDiffraction": ("xrd", "x-ray diffraction", "xray", "neutron diffraction", "bragg peaks", "diffractogram",
                          "two theta", "2theta", "powder pattern"),
    "StructureFactor": ("scattering", "total scattering", "neutron scattering", "x-ray scattering",
                        "scattering intensity", "faber ziman", "reciprocal space"),
    # --- grids and vector arrows
    "CreateIsosurface": ("elf", "electron localization function", "electron density", "spin density",
                         "partial charge density", "orbital", "contour surface", "level surface",
                         "isovalue"),
    "GridSlice": ("contour map", "contour plot", "heat map", "density map", "plane cut", "cross section", "2d slice",
                  "elf slice", "potential slice", "charge density slice"),
    "GridDifference": ("differential charge density", "charge transfer", "charge redistribution", "deformation density",
                       "delta rho", "subtract grids", "density difference"),
    "VectorVis": ("arrow", "force vectors", "show forces", "velocity vectors", "magnetic moments", "spin arrows",
                  "quiver", "displacement arrows", "draw vectors"),
    # --- notebook functions
    "load": ("open", "read", "import"),
    "import_file": ("open file", "read file", "several files", "file sequence", "restarted run", "join dumps"),
    # a format with its verb: "poscar" alone would put this first for "read a POSCAR"
    "export_file": ("save", "write", "convert", "output file", "save structure", "write structure",
                    "write trajectory",
                    *(f"{verb} {fmt}" for verb in ("save", "write", "convert", "export")
                      for fmt in ("poscar", "cif", "xyz", "lammps data", "lammps dump"))),
    "each_frame": ("custom step", "custom modifier", "python modifier", "own analysis", "user defined", "per frame"),
    "all_frames": ("custom step", "whole trajectory", "across frames", "over all frames"),
    "view": ("camera", "viewing direction", "view direction", "look along", "zoom", "background", "orthographic",
             "perspective", "transparent background", "display style", "ball and stick", "legend", "scale bar",
             "atom labels", "element colors", "visualize", "visualization"),
    "run_info": ("incar", "input parameters", "input settings", "simulation settings", "calculation settings",
                 "timestep", "time step", "thermostat", "ensemble"),
    "check_run": ("convergence", "converged", "equilibration", "equilibrated", "sanity check", "health check",
                  "validate run", "diagnose", "warnings", "run finished", "energy drift"),
    "load_runs": ("multiple runs", "many runs", "several trajectories", "batch", "series of runs",
                  "temperature series", "compare runs", "parameter sweep"),
    "arrhenius": ("arrhenius plot", "temperature dependence", "activation barrier", "prefactor"),
    "sensitivity": ("parameter scan", "parameter sweep", "vary parameter", "parameter dependence", "cutoff dependence",
                    "robustness", "convergence test"),
    "workflow_table": ("workflow steps", "summary table", "compare workflows", "compare calculations"),
    "show": ("plot", "graph", "chart", "curve", "plot table", "draw", "save plot", "save figure", "save pdf",
             "save png", "side by side", "compare frames"),
    "snapshot": ("render", "screenshot", "render image", "save image", "save picture", "save png", "ray tracing",
                 "publication image"),
    "figure": ("matplotlib", "savefig", "save figure", "save pdf", "customize plot", "annotate plot",
               "publication plot", "figure size"),
    "paint": ("color atoms", "highlight", "mark atoms", "color by condition", "color selected atoms", "recolor"),
    "hide": ("invisible", "hide atoms", "remove from view", "do not show"),
    "isolate": ("show only", "only show", "display only", "focus on", "keep only visible"),
    "fade": ("transparent", "transparency", "translucent", "semi transparent", "opacity"),
    "resize": ("atom size", "sphere size", "bigger atoms", "larger atoms", "smaller atoms", "enlarge atoms",
               "shrink atoms"),
    # --- recipes, by title
    "Crystal structure: bonds, TiO6 octahedra, space group": ("visualize crystal", "polyhedra", "octahedron",
                                                             "perovskite"),
    "MD trajectory: diffusion coefficient": ("diffusivity", "molecular dynamics", "aimd", "self diffusion",
                                             "most mobile atoms", "fastest atoms"),
    "Defects in a crystal: structure types, show only non-FCC atoms": ("defect atoms", "show defects",
                                                                      "non crystalline atoms", "disordered atoms",
                                                                      "hide bulk", "hide crystalline"),
    "NEB band: barrier and energy profile": ("nudged elastic band", "migration barrier", "energy barrier",
                                             "activation barrier", "minimum energy path", "saddle point",
                                             "transition state"),
    "Charge density: isosurface and a lattice-plane slice": ("planar average", "work function", "vacuum level",
                                                            "macroscopic average", "potential profile",
                                                            "electrostatic potential", "electron density"),
    "Atoms picked at one frame, followed through the trajectory": ("track atoms", "follow atoms", "tag atoms",
                                                                  "same atoms over time", "fixed selection",
                                                                  "persistent selection"),
}
# a verb adds its modifier, so the same words find both
SYNONYMS.update({verb: SYNONYMS[step] for verb, step in (
    ("rdf", "CoordinationAnalysis"), ("msd", "MeanSquaredDisplacement"), ("vacf", "VelocityAutocorrelation"),
    ("sq", "StructureFactor"), ("xrd", "PowderDiffraction"), ("bond_angles", "BondAngleDistribution"),
    ("cna", "CommonNeighborAnalysis"), ("ptm", "PolyhedralTemplateMatching"), ("voronoi", "VoronoiAnalysis"),
    ("clusters", "ClusterAnalysis"))})

#: Query words the tokenizer cannot read, with the reference's English words for them: Korean terms
#: (the owner and many users write Korean, and models pass their words through), notation
#: (``g(r)``: single letters are no words), British spellings, and names the reference only
#: abbreviates. Matched anywhere in the query, the longest first, so particles may be attached
#: (배위수의, 확산계수를); spaces inside a term of four or more characters are optional (평균 제곱 변위).
#: An empty value only keeps a shorter term from matching inside the word (검색 is not 색). A term
#: with two meanings takes the one this library has a step for (전위: dislocation; 정전위: potential).
GLOSSARY: dict[str, str] = {
    # structure and neighbors
    "배위수": "coordination number", "배위": "coordination", "이웃": "neighbor", "최근접": "nearest neighbor",
    "동경분포함수": "radial distribution function", "동경분포": "radial distribution function",
    "방사분포함수": "radial distribution function", "지름분포함수": "radial distribution function",
    "반경분포함수": "radial distribution function", "쌍분포함수": "pair distribution function",
    "쌍상관함수": "pair correlation function", "분포": "distribution", "히스토그램": "histogram",
    "결합각": "bond angle", "결합길이": "bond length", "결합거리": "bond length", "결합": "bond", "각도": "angle",
    "다면체": "polyhedra", "팔면체": "octahedra", "사면체": "tetrahedra", "고리": "ring",
    "클러스터": "cluster", "군집": "cluster", "보로노이": "voronoi", "원자부피": "atomic volume", "부피": "volume",
    "슈퍼셀": "supercell", "수퍼셀": "supercell", "초격자": "supercell", "복제": "replicate", "반복": "repeat",
    "주기경계": "periodic boundary", "언랩": "unwrap", "원자": "atom", "입자": "particle",
    # crystals and defects
    "결정구조": "crystal structure", "구조식별": "structure identification",
    "대칭": "symmetry", "공간군": "space group", "점군": "point group", "와이코프": "wyckoff",
    "중심대칭": "centrosymmetry", "공공": "vacancy", "빈자리": "vacancy", "침입형": "interstitial",
    "격자간": "interstitial", "점결함": "point defect", "결함": "defect", "전위": "dislocation", "버거스": "burgers",
    "결정립계": "grain boundary", "입계": "grain boundary", "결정립": "grain", "다결정": "polycrystalline",
    "적층결함": "stacking fault", "쌍정": "twin", "변형률": "strain", "변형": "strain deformation", "전단": "shear",
    "다이아몬드": "diamond", "얼음": "ice",
    # dynamics and transport
    "확산계수": "diffusion coefficient", "확산": "diffusion", "평균제곱변위": "mean squared displacement",
    "평균자승변위": "mean squared displacement", "변위": "displacement", "궤적": "trajectory", "경로": "path",
    "속도자기상관": "velocity autocorrelation", "자기상관": "autocorrelation", "속도": "velocity",
    "이온전도도": "ionic conductivity", "전도도": "conductivity", "호핑": "hopping hop", "도약": "jump", "점프": "jump",
    "체류시간": "residence time", "점유": "occupancy", "자리": "site", "시계열": "time series", "시간평균": "time average",
    "평균": "average", "활성화에너지": "activation energy", "에너지장벽": "energy barrier", "장벽": "barrier",
    "아레니우스": "arrhenius", "포논": "phonon", "진동": "vibration", "반호브": "van hove", "비가우시안": "non-gaussian",
    "중간산란함수": "intermediate scattering function", "수렴": "convergence", "평형": "equilibration",
    "온도": "temperature", "에너지": "energy",
    # diffraction, electronic structure and grids
    "구조인자": "structure factor", "회절": "diffraction", "엑스선": "x-ray", "x선": "x-ray", "분말": "powder",
    "중성자": "neutron", "산란": "scattering", "상태밀도": "DOS", "밴드구조": "bands", "밴드": "bands", "띠구조": "bands",
    "전하밀도": "charge density", "전자밀도": "electron density", "스핀밀도": "spin density", "전하": "charge",
    "밀도": "density", "등위면": "isosurface", "등치면": "isosurface", "등고면": "isosurface", "단면": "slice cross section",
    "슬라이스": "slice", "평면평균": "planar average", "일함수": "work function", "정전위": "electrostatic potential",
    "퍼텐셜": "potential", "포텐셜": "potential", "프로파일": "profile", "차이": "difference",
    # view, colors, selection and files
    "색": "color", "컬러": "color", "검색": "", "탐색": "", "색인": "", "선택": "select selection", "삭제": "delete",
    "제거": "remove", "숨기": "hide", "숨김": "hide", "투명": "transparent", "크기": "size", "반지름": "radius",
    "화살표": "arrow", "벡터": "vector", "힘": "force", "자기모멘트": "magnetic moments", "렌더": "render",
    "그림": "picture", "이미지": "image", "스냅샷": "snapshot", "그래프": "plot", "플롯": "plot",
    "시각화": "visualize", "그리": "draw", "그려": "draw", "그리고": "", "그리드": "grid", "카메라": "camera",
    "내보내": "export", "저장": "save", "변환": "convert", "불러오": "load open file", "읽어": "read file",
    # notation, British spellings and spelled-out abbreviations
    "g(r)": "radial distribution function", "s(q)": "structure factor", "g(r,t)": "van Hove correlation",
    "fs(q,t)": "self intermediate scattering", "f_s(q,t)": "self intermediate scattering",
    "f(q,t)": "intermediate scattering", "q4": "Steinhardt order parameters", "q6": "Steinhardt order parameters",
    "w4": "Steinhardt order parameters", "w6": "Steinhardt order parameters", "d2min": "nonaffine squared displacement",
    "density of states": "DOS", "colour": "color", "neighbour": "neighbor", "centre": "center",
}


def _glossary() -> tuple[dict[str, str], "re.Pattern"]:
    """GLOSSARY keyed without spaces and in lower case, and the pattern that finds its terms."""
    words = {"".join(key.split()).lower(): value for key, value in GLOSSARY.items()}
    pieces = [("(?<![a-z])" if key[0].isascii() else "")           # "q6" is not the end of an English word
              + (r"\s*".join(map(re.escape, key)) if len(key) > 3 else re.escape(key))
              for key in sorted(words, key=len, reverse=True)]     # the longest first: 확산계수 before 확산
    return words, re.compile("|".join(pieces), re.IGNORECASE)


def _expand(query: str) -> str:
    """``query`` with every GLOSSARY term replaced by its English search words."""
    words, pattern = _glossary()
    return pattern.sub(lambda m: f" {words[''.join(m.group().split()).lower()]} ", query)


def _terms(text: str) -> list[str]:
    """Lower-case search terms: CamelCase and snake_case split (SiteHopping -> site, hopping), a
    plural or -ing ending dropped, so "hops" finds "hop" and "Hopping", and "trajectory" finds
    "UnwrapTrajectories"."""
    out = []
    for word in _WORD.findall(text):
        w = word.lower()
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"                                  # trajectories -> trajectory
        elif len(w) > 4 and w.endswith(("sses", "xes", "ches", "shes")):
            w = w[:-2]                                        # classes, boxes, meshes
        elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]                                        # sites -> site (not "sit"); analysis, radius stay
        if len(w) > 5 and w.endswith("ing"):
            w = w[:-3]
            if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz":
                w = w[:-1]                                    # hopping -> hop, but filling -> fill
        out.append(w)
    return out


def _entry(text: str) -> str:
    """The name a piece of the reference has in SYNONYMS: the modifier or function of an entry
    line, the title of a recipe, '' for other text."""
    m = re.match(r"- `(\w+)\(", text)
    if m:
        return m.group(1)
    title, cells, _ = text.partition(":\n```")
    return title if cells else ""


def reference_chunks(with_plugins: bool = False) -> list[tuple[str, str]]:
    """The reference cut into ``(section, text)`` pieces: one per modifier or function entry, one
    per recipe (its title and cells), one per paragraph of the guide."""
    chunks, section = [], ""
    lines = api_reference(with_plugins).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            section = line.lstrip("# ").strip()
            i += 1
        elif line.startswith("```"):
            end = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("```")), len(lines) - 1)
            block = "\n".join(lines[i:end + 1])
            if chunks and chunks[-1][0] == section and not chunks[-1][1].startswith("- `"):
                chunks[-1] = (section, chunks[-1][1] + "\n" + block)   # a recipe's cells join its title
            else:
                chunks.append((section, block))
            i = end + 1
        elif line.strip():
            k = i + 1
            if line.startswith("- "):    # an entry runs until the next entry, heading or blank line
                while k < len(lines) and lines[k].strip() and not lines[k].startswith(("- ", "#", "```")):
                    k += 1
            else:
                while k < len(lines) and lines[k].strip() and not lines[k].startswith(("#", "```")):
                    k += 1
            chunks.append((section, "\n".join(lines[i:k])))
            i = k
        else:
            i += 1
    return chunks


def search_reference(query: str, limit: int = 6, with_plugins: bool = False) -> list[dict]:
    """The pieces of the reference that best match ``query`` (BM25 over words; an entry whose name
    or SYNONYMS contain a query word ranks first; GLOSSARY terms of the query, such as Korean ones,
    are read as their English words): ``[{"section", "text", "score"}]``, best first."""
    import math

    chunks = reference_chunks(with_plugins)
    docs = [_terms(text) for _, text in chunks]
    names = [_terms(m.group(1)) if (m := re.match(r"- `(\w+)", text)) else [] for _, text in chunks]
    want = list(dict.fromkeys(_terms(_expand(query))))
    if not want or not docs:
        return []
    asked = set(want)
    # the words of an entry's synonym phrases this query holds whole: "pair distribution" must not
    # make CoordinationAnalysis a match for every "distribution"
    extras = [{t for phrase in SYNONYMS.get(_entry(text), ()) if asked.issuperset(words := _terms(phrase))
               for t in words} for _, text in chunks]
    avg = sum(map(len, docs)) / len(docs)
    df = {t: sum(t in extra or t in doc for doc, extra in zip(docs, extras)) for t in want}
    scored = []
    for (section, text), doc, name, extra in zip(chunks, docs, names, extras):
        score = 0.0
        for t in want:
            f = doc.count(t) or int(t in extra)    # a synonym counts as the word written once, and like a name
            if f:
                idf = math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(doc) / avg))
                if t in name or t in extra:
                    score += 2 * idf
        if score > 0:
            scored.append((score, section, text))
    scored.sort(key=lambda s: -s[0])
    return [{"section": s, "text": _full_entry(t), "score": round(sc, 2)} for sc, s, t in scored[:limit]]


def _documented(name: str):
    """The modifier (core or plugin) or notebook function called ``name``, or None."""
    from .. import modifiers, paint, plugins, views
    import ewald

    for module, names in FUNCTIONS:
        if name in names:
            return getattr({"ewald": ewald, "ewald.views": views, "ewald.paint": paint}[module], name)
    if name in modifiers.__all__:
        return getattr(modifiers, name)
    plugin = plugins.modifiers().get(name)
    if plugin is not None:
        return plugin.cls
    function = plugins.functions().get(name)
    return function.function if function is not None else None


def _full_entry(text: str) -> str:
    """A modifier or function entry with its whole docstring (the reference keeps only the first
    paragraph and the sentences naming outputs; options such as ``site_radius`` are explained
    further down), other pieces unchanged."""
    m = re.match(r"- `(\w+)\(", text)
    obj = _documented(m.group(1)) if m else None
    if obj is None or not obj.__doc__:
        return text
    return f"- `{m.group(1)}{_signature(obj)}`:\n" + _plain(inspect.cleandoc(obj.__doc__))
