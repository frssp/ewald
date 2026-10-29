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
| `data.particles.count` | `data.count` |
| `data.particles.positions`, `data.particles['Position']` | `data.positions`, `data['Position']` |
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
`numpy as np`, `all_frames, each_frame, import_file, load` (from `ewald`), every modifier (`from ewald.modifiers import *`), `show, view`
(from `ewald.views`), `paint, hide, isolate, fade, resize` (from `ewald.paint`) and `rdf, msd, vacf, sq, xrd, load_runs,
check_run, run_info, arrhenius` (from `ewald`); import anything else in the cell.
`pipe` is the pipeline. Write each change as its own cell, in the kinds the GUI recognizes:

- load cell: `pipe = load("XDATCAR")`.
- step cell: exactly one `pipe.add(CoordinationAnalysis(cutoff=3.0))` with literal arguments (the GUI shows a form for
  it and edits the call); several steps are several cells. A Python step is a cell headed `# %% name [each frame]`
  (body sees `frame`, `data`) or `[all frames]` (`frames`, `data`), or a function under
  `@pipe.add` and `@each_frame` / `@all_frames` (also runs as a plain script).
- view cell: exactly one `view(pipe, ...)` call (the 3D view, see below) or one `show(...)` call (a plot).
- `show(pipe, table=None, y=None, histogram=None, frame=0, bins=50, xlim=None, ylim=None, panel=None, save=None)` plots any
  `data.tables` key (`"coordination-rdf"`, `"structure-factor"`, `"diffraction"`, `"msd"`, `"DOS"`, `"Bands"`, `"Steps"`
  (thermo), ...; `y=` one column) or a histogram of a particle property (`histogram="Coordination"`). In the GUI it is a
  live plot linked to the 3D view; `panel=True`/`False` puts it in the Plots panel or under the cell. `xlim`/`ylim` set
  the range (`xlim=(-3, 3)` for a DOS window); `save="rdf.pdf"` writes it. `show(pipe)` with no table or histogram is a
  picture of the structure, not a plot. `ewald.views.figure(...)` returns the matplotlib Figure instead.
- `snapshot(pipe, frame=None, **render_kwargs)` (`from ewald.views import snapshot`) as a cell's last line shows a
  picture of the 3D view.
- `paint(data, where, color)`, `hide(data, where)`, `isolate(data, where)`, `fade(data, where, transparency=0.7)`,
  `resize(data, where, factor)` change "Color", "Transparency" or "Radius" inside a step (`where`: boolean mask, index
  array or expression string such as `"Coordination < 12"`); they need `data`, so use them in an `[each frame]` cell or
  `@pipe.add` function, not on `pipe`.
- Plain code cells (print, numpy, matplotlib) may read `pipe.compute(i)` freely; anything that should change the
  structure, the view or a result goes through the cells above, so the notebook alone reproduces it.
- VASP DOS/bands (vasprun.xml, vaspout.h5): `data.tables["DOS"]`, `data.tables["Bands"]`, the attributes
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
`fit_time=(t0, t1)` (`fit_range` is a fraction of each run's lags; `.fit_start`/`.fit_end` say what was fitted); take temperatures from the input files and say where a folder
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
               "arrhenius", "workflow_table")),
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
    """Installed modifier plugins, in the same format (not part of llms.txt)."""
    from .. import plugins

    found = plugins.modifiers()
    lines = ["## Modifier plugins installed here", ""]
    for name, plugin in found.items():
        doc = _doc(plugin.cls)
        lines.append(f"- `{name}{_signature(plugin.cls)}`" + (f": {doc}" if doc else "") + f" [{plugin.origin}]")
    return "\n".join(lines) + "\n" if found else ""


def api_reference(with_plugins: bool = False) -> str:
    """The full reference: guide, recipes, function signatures, the live modifier list (+ installed
    modifier plugins)."""
    text = "\n".join([GUIDE, recipes_reference(), function_reference(), modifier_reference()])
    plugins_text = plugin_reference() if with_plugins else ""
    return text + ("\n" + plugins_text if plugins_text else "")
