# Feature inventory

Everything ewald has on `main`, generated from the code by
`python -m ewald inventory -o docs/features.md --gui`. Do not edit it by hand: `tests/test_docs.py`
(and `tests/test_gui.py` for the GUI section) fail when it is out of date. Open work and ideas are
in the [TODO list](todo.md); how each feature works is in the rest of this manual.

## Python API

`import ewald`: `Cell`, `DataCollection`, `Pipeline`, `Modifier`, `PythonModifier`, `ComputeContext`, `StaticSource`, `import_file`, `export_file`, `get_backend`, `gpu_available`. Details: [Python scripting](user/scripting.md) and the [API reference](reference/api.md).

## Modifiers

41 built-in modifiers in `ewald.modifiers` (OVITO spellings such as `CommonNeighborAnalysisModifier` also import). Details: [Modifiers](user/modifiers.md).

### Analysis

| Modifier | What it does |
|---|---|
| `CalculateDisplacements` | Displacement from a reference frame (minimum-image convention). |
| `CoordinationAnalysis` | Coordination numbers within ``cutoff`` and the radial distribution function. |

### Structure identification

| Modifier | What it does |
|---|---|
| `CommonNeighborAnalysis` | Common neighbor analysis (Honeycutt-Andersen; adaptive a-CNA of Stukowski 2012). |
| `PolyhedralTemplateMatching` | Polyhedral template matching (Larsen, Schmidt, Schiotz 2016), using the PTM library. |
| `VoronoiAnalysis` | Per-particle Voronoi cells: volume, coordination, Voronoi indices. |
| `AcklandJones` | Bond-angle structure identification of Ackland and Jones (2006). |
| `AtomicStrain` | Atomic-level strain from a reference frame (Shimizu, Ogata, Li 2007; Falk-Langer D2min). |
| `CentroSymmetry` | Centrosymmetry parameter (Kelchner et al. 1998) from ``num_neighbors`` neighbors. |
| `ChillPlus` | Chill+ identification of water structures (Nguyen and Molinero 2015). |
| `IdentifyDiamond` | Cubic and hexagonal diamond lattice identification (Maras et al. 2016). |

### Defects and clusters

| Modifier | What it does |
|---|---|
| `ClusterAnalysis` | Group particles into clusters connected by distances below ``cutoff``. |
| `WignerSeitzAnalysis` | Count atoms in the Wigner-Seitz cell of every site of a reference frame. |

### OVITO bridge (needs the optional `ovito` package)

| Modifier | What it does |
|---|---|
| `DislocationAnalysis` | Dislocation extraction algorithm (DXA) via the `ovito` package. |
| `OvitoModifier` | Run an OVITO modifier (``ovito.modifiers.*``) on the current frame. |

### Trajectory

| Modifier | What it does |
|---|---|
| `MeanSquaredDisplacement` | Time-averaged mean squared displacement over a frame range and the diffusion coefficient. |
| `VelocityAutocorrelation` | Velocity autocorrelation function and the Green-Kubo diffusion coefficient. |
| `TimeAveraging` | Average upstream results over a frame range (OVITO's TimeAveragingModifier). |
| `TimeSeries` | Global attributes as a function of time (OVITO's TimeSeriesModifier). |
| `GenerateTrajectoryLines` | Particle paths over a frame range, drawn as lines, plus path metrics. |

### Bonds and polyhedra

| Modifier | What it does |
|---|---|
| `CreateBonds` | Create bonds between nearby particles (OVITO's CreateBondsModifier, plus VESTA's bond search). |
| `EditBonds` | Add or delete individual bonds by hand, as in VESTA's bond editing. |
| `CoordinationPolyhedra` | Polyhedra spanned by the bonded neighbors of each center particle (OVITO's CoordinationPolyhedraModifier, VESTA's polyhedral style). |

### Volumetric data

| Modifier | What it does |
|---|---|
| `CreateIsosurface` | VESTA-style isosurface of a grid property (OVITO's CreateIsosurfaceModifier). |
| `GridSlice` | Colour-mapped cross-section of a grid property on a lattice plane (VESTA's 2D slice). |
| `GridDifference` | Charge-density difference: subtract the grids of other files, e.g. rho(AB) - rho(A) - rho(B) with ``GridDifference(["A/CHGCAR", "B/CHGCAR"])``. |

### Coloring

| Modifier | What it does |
|---|---|
| `AssignColor` | Give the selected particles (all, if nothing selected) one color. |
| `ColorCoding` | Map a scalar property or expression to colors through a matplotlib colormap. |

### Selection

| Modifier | What it does |
|---|---|
| `ClearSelection` | Remove the 'Selection' property. |
| `DeleteSelected` | Delete the selected particles. |
| `ExpressionSelection` | Select particles where ``expression`` is true (sets 'Selection'). |
| `InvertSelection` | Select the unselected particles and unselect the selected ones. |
| `SelectType` | Select particles whose 'Particle Type' (or another property) is in ``types``. |
| `Slice` | Keep (or select) particles in a slab ``\|n.r - distance\| <= width/2``. |

### Modification

| Modifier | What it does |
|---|---|
| `AffineTransformation` | Apply a 3x3 (or 3x4 with translation) matrix to particles and, optionally, the cell. |
| `ComputeProperty` | Set ``output`` to an expression, e.g. ``ComputeProperty("Height", "Position.Z - 5")``. |
| `Replicate` | Tile the periodic cell ``n = (na, nb, nc)`` times. |
| `WrapAtoms` | Map particles back into the periodic cell (bond periodic images follow). |

### Python

| Modifier | What it does |
|---|---|
| `PythonScript` | Run ``modify(frame, data)`` defined in ``source``. |

### plumed

| Modifier | What it does |
|---|---|
| `Plumed` | Collective variables computed by PLUMED for every frame, from a ``plumed.dat``. |
| `LoadColvar` | Values from a PLUMED COLVAR file (or any ``#! FIELDS`` file) attached to the frames. |
| `FreeEnergySurface` | Free energy surface from a metadynamics HILLS file, F(s) = -sum of the hills. |

## File formats

### Readers

Tried in this order when no `format=` is given; reader plugins slot in at priority 50. Details: [File formats](user/formats.md).

| Format | File names | Reads | `format=` aliases |
|---|---|---|---|
| `cube` | `*.cube` `*.cub` | Gaussian cube volumetric data (atoms + grid) | `gaussian-cube` |
| `vasp-volumetric` | `CHGCAR*` `CHG*` `PARCHG*` `AECCAR*` `LOCPOT*` `ELFCAR*` | VASP CHGCAR/CHG/PARCHG/AECCAR/LOCPOT/ELFCAR (atoms + grid) | `vasp-chgcar`, `chgcar`, `vasp-locpot`, `locpot`, `vasp-elfcar`, `elfcar` |
| `h5md` | `*.h5md` `*.h5` | H5MD trajectory (LAMMPS dump h5md, MDAnalysis, ...), lazy; needs h5py |  |
| `xsf` | `*.xsf` `*.axsf` | XCrySDen XSF/AXSF: structure, ANIMSTEPS trajectory, DATAGRID_3D grids |  |
| `lammps-dump` | `*.lammpstrj` `*.dump` | LAMMPS text dump (atom/custom), lazy | `lammps/dump`, `lammps-dump-text` |
| `xdatcar` | `XDATCAR*` | VASP XDATCAR, fixed or variable cell, lazy | `vasp-xdatcar` |
| `lammps-data` | `*.data` | LAMMPS data file | `lammps/data`, `lammps-data-native` |
| `ase-db` | `*.db` | ASE database (SQLite .db or .json), one row per frame, lazy; selection= query | `db`, `ase.db` |
| `ase-traj` | `*.traj` | ASE binary trajectory (.traj), lazy | `traj` |
| `md-trajectory` | `*.xtc` `*.trr` `*.dcd` `*.nc` `*.ncdf` | GROMACS XTC/TRR, DCD, AMBER NetCDF via MDAnalysis, lazy; topology= (.gro/.pdb/.tpr/...) for atom types | `xtc`, `trr`, `dcd`, `netcdf`, `ncdf`, `mdanalysis` |
| `extxyz` | `*.xyz` `*.extxyz` | XYZ / extended XYZ (Lattice=, Properties=), lazy | `xyz`, `ext-xyz` |
| `cif` | `*.cif` | Crystallographic Information File via ASE |  |
| `ase` | `POSCAR*` `CONTCAR*` `OUTCAR*` `vasprun*.xml` `*.xyz` `*.extxyz` `*.cif` `*.vasp` | anything ase.io.read understands (POSCAR, OUTCAR, vasprun.xml, extxyz, cif, ...); other format= names are passed to ASE |  |
| `lammps-data-ase` | (by `format=` only) | LAMMPS data file via ASE |  |

Remote paths (`host:/path`, `ssh://...`) work with every reader; LAMMPS dumps and XDATCARs are streamed, other formats are downloaded to a local cache.

### Writers

`export_file(source, path, format=...)` and File › Export data pick a writer by `format=` or by file name; *one frame* writers need e.g. `frames=[-1]`.

| Format | File names | Writes | `format=` aliases |
|---|---|---|---|
| `cube` *one frame* | `*.cube` | Gaussian cube (one grid property + atoms) | `gaussian-cube` |
| `vasp-volumetric` *one frame* | `CHGCAR*` `LOCPOT*` `ELFCAR*` `PARCHG*` | VASP CHGCAR-style grid (VESTA reads it) | `vasp-chgcar`, `chgcar` |
| `h5md` | `*.h5md` `*.h5` | H5MD trajectory, streamed; needs h5py |  |
| `xsf` | `*.xsf` `*.axsf` | XCrySDen XSF (structure, ANIMSTEPS, grid) |  |
| `lammps-dump` | `*.lammpstrj` `*.dump` | LAMMPS text dump | `lammps/dump` |
| `poscar` *one frame* | `POSCAR*` `CONTCAR*` `*.vasp` `*.poscar` | VASP POSCAR (species grouped, Direct) | `vasp` |
| `xdatcar` | `XDATCAR*` | VASP XDATCAR | `vasp-xdatcar` |
| `lammps-data` *one frame* | `*.data` `*.lmp` | LAMMPS data file (atomic/charge/molecular/full) | `lammps/data` |
| `ase-db` | `*.db` | ASE database, one row per frame | `db`, `ase.db` |
| `ase-traj` | `*.traj` | ASE binary trajectory | `traj` |
| `md-trajectory` | `*.xtc` `*.trr` `*.dcd` | GROMACS XTC/TRR or DCD via MDAnalysis, streamed | `xtc`, `trr`, `dcd` |
| `extxyz` | `*.extxyz` `*.xyz` | extended XYZ, streamed | `xyz` |
| `cif` | `*.cif` | CIF via ASE |  |
| `ase` | (by `format=` only) | any format ase.io.write knows (format= name) |  |

## Rendering and output

Details: [Rendering figures](user/rendering.md).

### Renderers

| `renderer=` | What it is |
|---|---|
| `opengl` | VTK, fast; SSAO and anti-aliasing; GPU impostors for large systems |
| `raytrace` | built-in numba ray tracer: ambient occlusion, shadows, anti-aliasing (publication figures) |
| `vector` | SVG/PDF/EPS with editable shapes and text (matplotlib) |

### Output formats

- Images: `.png` `.jpg` `.jpeg` `.tif` `.tiff`; vector pages: `.svg` `.pdf` `.eps` `.ps` (true vectors with `renderer="vector"`, an embedded image otherwise)
- Movies (`render_animation`, `ewald movie`): `.gif` `.mp4` `.mov` `.m4v` `.avi` `.webm` `.png` `.jpg` `.jpeg` `.tif` `.tiff` (an image suffix writes a numbered sequence; video needs ffmpeg)
- Sessions: `.ewald` (source, modifiers, style, camera, render settings)

### Quality presets

| `quality=` | `samples` | `ao_samples` | `shadows` | `antialiasing` | `ambient_occlusion` | `sphere_resolution` | `scale` |
|---|---|---|---|---|---|---|---|
| `draft` | 1 | 0 | False | False | False | 8 | 0.5 |
| `standard` | 4 | 8 | True | True | False | 16 | 1.0 |
| `publication` | 16 | 32 | True | True | True | 28 | 1.0 |

### `render()` options

`size`, `direction`, `orthographic`, `zoom`, `background`, `transparent_background`, `ambient_occlusion`, `legend`, `show_cell`, `bonds`, `radius_scale`, `antialiasing`, `text`, `camera`, `renderer`, `samples`, `ao_samples`, `shadows`, `type_legend`, `scale_bar`, `dpi`, `quality`, `scale`.

## Command line

Details: [Command line](user/cli.md).

| Command | What it does | Options |
|---|---|---|
| `ewald render INPUT OUTPUT` | render a frame to an image (.png/.jpg/.tif/.svg/.pdf/.eps) | `--frame` `--size` `--direction` `--bonds` `--color-by` `--cmap` `--type-map` `--ao` `--renderer` `--allow-code` `--quality` `--text` `--type-legend` `--scale-bar` `--dpi` |
| `ewald movie INPUT OUTPUT` | render frames to .mp4/.gif/.avi/.mov or a PNG sequence | `--fps` `--start` `--stop` `--every` `--size` `--direction` `--bonds` `--color-by` `--cmap` `--type-map` `--time-per-frame` `--renderer` `--allow-code` `--quality` `--text` `--type-legend` `--scale-bar` |
| `ewald gui [INPUT]` | open the desktop viewer | `--type-map` `--device` |
| `ewald info INPUT` | print frames and properties | `--type-map` |
| `ewald describe INPUT` | JSON summary of a frame or, with --trajectory, of the run | `--frame` `--trajectory` `--type-map` |
| `ewald ask INPUT [QUESTION...]` | ask Claude about a structure (needs anthropic + credentials) | `--type-map` `--model` `--allow-code` `--script` |
| `ewald narrate INPUT` | timeline of events in a trajectory (Claude writes it if available) | `--type-map` `--no-llm` `--render` `--json` |
| `ewald api` | print the Python API reference for LLMs (llms.txt) | `--with-plugins` |
| `ewald plugins` | list installed plugins and any that failed to load | `--json` |
| `ewald forge [REQUEST...]` | have Claude write a modifier plugin, test it and save it | `--yes` `--draft` `--overwrite` `--rounds` `--model` |
| `ewald mcp [INPUT]` | serve the analysis tools over MCP (stdio) | `--allow-code` |
| `ewald inventory` | print or update the feature inventory (docs/features.md) | `-o` `--gui` |

## Claude integration

The same tools serve `ewald ask`, the Python `ewald.llm.Agent` and `ewald mcp` (MCP server). Tools marked *code* exist only with `allow_code=True`. Details: [Claude integration](user/claude.md).

| Tool | What it does |
|---|---|
| `load_structure` | Load a structure or trajectory file (VASP POSCAR/CONTCAR/XDATCAR/OUTCAR, LAMMPS dump/data, extxyz, cif, ... ; wildcards allowed) as the session pipeline, replacing any previous one. |
| `describe_frame` | Compact description of one frame after the current modifiers: composition, cell, density, minimum distance, coordination histogram per type, g(r) peaks, property ranges, sanity warnings. |
| `describe_trajectory` | Scan the trajectory (source frames, subsampled to max_frames) and return per-frame descriptor statistics (volume, min distance, MSD and max displacement vs the first frame, coordination) and 'events': frames where a descriptor shifts abruptly. |
| `event_details` | Measure what changes around a frame (e.g. an event from describe_trajectory): which atoms moved and how far (minimum image, by identifier), by type, local vs system-wide, coordination transitions, and the top movers with positions before/after. |
| `list_modifiers` | Available modifiers with their parameters and defaults. |
| `add_modifier` | Append (or insert at 'index') a modifier to the pipeline. |
| `remove_modifier` | Remove the modifier at 'index' (see show_pipeline). |
| `show_pipeline` | Current source, modifier list and frame. |
| `set_frame` | Set the current frame used by tools that take no frame argument. |
| `evaluate_expression` | Evaluate an OVITO-style per-particle expression on a frame and return statistics (count of true values for conditions; min/max/mean/percentiles for numbers). |
| `compute_rdf` | Radial distribution function g(r) of a frame (total, plus partial pairs). |
| `compute_msd` | Time-averaged mean squared displacement over source frames (unwrapped, matched by identifier), per type, with a linear-regime fit D = slope/6. |
| `compute_plumed_cvs` | Collective variables computed by PLUMED (needs the PLUMED library) over the trajectory from a plumed.dat (text or path): e.g. 'd: DISTANCE ATOMS=1,2' (atoms numbered from 1, by Particle Identifier when present), TORSION, COORDINATION, RMSD, METAD with temperature. |
| `read_plumed_output` | Read a PLUMED output file. |
| `render_image` | Render a frame of the pipeline to PNG and return the image so you can inspect it. |
| `export_file` | Write frames of the pipeline (after its modifiers) to a file: POSCAR, LAMMPS data/dump, extxyz, CIF, XSF, H5MD, XTC, cube/CHGCAR, ASE db, ... The format follows the file name (POSCAR, *.data, *.extxyz, *.h5md) unless given. |
| `get_script` | Standalone Python script that reproduces the current pipeline. |
| `python_api` *code* | Reference for the ewald Python API (data model, expression syntax, differences from OVITO, every modifier signature). |
| `run_python` *code* | Execute Python in the session (variables persist). |

Also: `ewald describe` / `ewald narrate` (rule-based descriptors and a trajectory timeline, Claude-written when a client is available), Modifier Forge (`ewald forge`) and the `llms.txt` API reference (`ewald api`).

## Plugins

Entry point groups `ewald.readers`, `ewald.writers` and `ewald.modifiers`, or a file in `<config dir>/plugins/` with `@reader` / `@writer` / `@modifier`; `ewald plugins` lists what loaded. Details: [Plugins](plugins/index.md).

- `examples/plugins/local_composition.py`: Example modifier plugin: local chemical composition around every particle.
- `examples/plugins/npz_trajectory.py`: Example reader plugin: trajectories stored as NumPy ``.npz`` archives.

## Preferences

Edit › Preferences, one page per group; stored as JSON with only changed values. Scripts read them with `ewald.settings.load()`.

### General

| Key | Label | Default |
|---|---|---|
| `general.startup_workspace` | Startup workspace | `'Analyze'` |
| `general.font_size` | UI font size | `12` |
| `general.restore_layout` | Restore window layout | `True` |
| `general.lammps_type_prompt` | LAMMPS type names | `'ask'` |
| `general.lammps_type_map` | Default LAMMPS type map | `''` |
| `general.docs_url` | Documentation URL | `''` |

### Viewports

| Key | Label | Default |
|---|---|---|
| `viewport.background` | Background | `'#14161A'` |
| `viewport.projection` | Projection | `'ortho'` |
| `viewport.initial_view` | Initial view direction | `'iso'` |
| `viewport.style_preset` | Default style | `'Space-filling'` |
| `viewport.rotation_step` | Step rotation | `15.0` |
| `viewport.show_axes` | Show axes tripod | `True` |
| `viewport.show_overlay` | Show frame / timestep text | `True` |
| `viewport.sphere_resolution` | Sphere resolution | `10` |
| `viewport.mesh_limit` | Impostor threshold | `150000` |
| `viewport.interactive_detail` | Detail while moving | `'auto'` |
| `viewport.target_fps` | Target frame rate | `30` |
| `viewport.interactive_atoms` | Atoms while moving | `1000000` |
| `viewport.refine_delay` | Refine after | `0.3` |

### Elements

| Key | Label | Default |
|---|---|---|
| `elements.color_scheme` | Color scheme | `'Jmol'` |
| `elements.radius_source` | Radius preset | `'covalent'` |
| `elements.radius_scale` | Radius scale | `0.8` |
| `elements.overrides` | Per-element overrides | `{}` |
| `bonds.length_factor` | Bond length factor | `1.15` |
| `bonds.radius` | Bond radius | `0.15` |

### Rendering

| Key | Label | Default |
|---|---|---|
| `render.renderer` | Default renderer | `'raytrace'` |
| `render.quality` | Image quality | `'Publication'` |
| `render.movie_quality` | Movie quality | `'Same as image'` |
| `render.samples` | Anti-aliasing rays/pixel | `16` |
| `render.ao_samples` | Ambient occlusion rays | `32` |
| `render.shadows` | Shadows | `True` |
| `render.size_preset` | Size preset | `'1-column (3.5 in, 4:3)'` |
| `render.dpi` | Resolution | `600` |
| `render.background` | Background | `'white'` |
| `render.output_dir` | Output folder | `''` |
| `render.type_legend` | Particle type legend | `False` |
| `render.scale_bar` | Scale bar | `False` |
| `render.movie_format` | Movie format | `'MP4 (H.264)'` |

### Compute

| Key | Label | Default |
|---|---|---|
| `compute.device` | Default device | `'cpu'` |
| `performance.cache_memory` | Frame cache memory | `2048` |
| `compute.threads` | CPU threads | `0` |

### Python

| Key | Label | Default |
|---|---|---|
| `python.startup_script` | Console startup script | `''` |
| `python.extra_paths` | Extra module paths | `''` |

<!-- gui:start -->
## Desktop GUI

`ewald gui [FILE]`. Details: [Desktop GUI](user/gui.md).

### Menus

**File**

- Load file… (`Ctrl+O`)
- Load remote file (SSH)… (`Ctrl+Alt+O`)
- Load session… (`Ctrl+Shift+O`)
- Save session (`Ctrl+S`)
- Save session as… (`Ctrl+Shift+S`)
- Render image… (`Ctrl+Shift+R`)
- Export data… (`Ctrl+E`)
- Export movie…
- Quit (`Ctrl+Q`)

**Edit**

- Preferences… (`Ctrl+,`)

**View**

- View along a
- View along b
- View along c
- View along a*
- View along b*
- View along c*
- Top (+z)
- Front (-y)
- Right (+x)
- Isometric
- Zoom to fit (`Ctrl+0`)
- Dark background
- White background

**Pipeline**

- Analysis
  - Coordination analysis / RDF
  - Displacement vectors
  - ClusterAnalysis
  - CommonNeighborAnalysis
  - VoronoiAnalysis
  - WignerSeitzAnalysis
  - DislocationAnalysis
- Trajectory
  - Mean squared displacement / diffusion
  - Velocity autocorrelation
  - Time averaging (g(r), attributes)
  - Time series of attributes
  - Trajectory lines / paths
- Bonds & polyhedra
  - Create bonds
  - Edit bonds (add / delete)
  - Coordination polyhedra
- Volume (CHGCAR, LOCPOT, cube)
  - Isosurface
  - Isosurface, signed (density difference / spin)
  - Slice plane (2D map)
  - Charge density difference
- Coloring
  - Color coding
  - Assign color
- Modification
  - Slice
  - Replicate
  - Wrap at periodic boundaries
  - Affine transformation
  - Compute property
  - Delete selected
- Selection
  - Expression selection
  - Select type
  - Invert selection
  - Clear selection
- Collective variables (PLUMED)
  - PLUMED input (plumed.dat)
  - Load COLVAR file
  - Free energy surface from HILLS
- Python
  - Python script
- Other
  - PolyhedralTemplateMatching
  - AcklandJones
  - AtomicStrain
  - CentroSymmetry
  - ChillPlus
  - IdentifyDiamond

**Tools**

- Orbit / pan / zoom (mouse)
- Pick atom
- Measure distance
- Measure angle
- Edit bonds

**Window**

- Analyze workspace (`Ctrl+1`)
- Render workspace (`Ctrl+2`)
- Python console
- Data inspector
- Plots
- Command palette

**Help**

- User manual
- Desktop GUI
- Modifiers
- Python scripting
- Writing plugins
- Python API reference
- Open plugin folder
- About ewald

### Toolbar

Open, Open remote, Analyze, Render, Search commands, modifiers… Ctrl K, Compute device switch (CPU / CUDA / auto).

### Panels

- Workspaces: Analyze, Render
- Docks: Data, Pipeline
- Bottom tabs: Python console, Data inspector, Plots, Log
- Viewport tools: Orbit / pan / zoom (mouse); Pick atom: shows id, type, position and properties; Measure distance: click two atoms; Measure angle: click three atoms (vertex in the middle); Edit bonds: click two atoms to add or delete the bond between them

<!-- gui:end -->
