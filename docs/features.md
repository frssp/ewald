# Feature inventory

Everything ewald has on `main`, generated from the code by
`python -m ewald inventory -o docs/features.md --gui`. Do not edit it by hand: `tests/test_docs.py`
(and `tests/test_gui.py` for the GUI section) fail when it is out of date. Open work and ideas are
in the [TODO list](todo.md); how each feature works is in the rest of this manual.

## Python API

`import ewald`: `Cell`, `DataCollection`, `Pipeline`, `Modifier`, `PythonModifier`, `ComputeContext`, `ComputeCancelled`, `StaticSource`, `import_file`, `load`, `View`, `view`, `export_file`, `get_backend`, `gpu_available`, `rdf`, `msd`, `vacf`, `sq`, `xrd`, `bond_angles`, `cna`, `ptm`, `voronoi`, `clusters`. Details: [Python scripting](user/scripting.md) and the [API reference](reference/api.md).

## Modifiers

64 built-in modifiers in `ewald.modifiers` (OVITO spellings such as `CommonNeighborAnalysisModifier` also import). Details: [Modifiers](user/modifiers.md).

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
| `OverlayFrames` | Every frame of ``start:stop:step`` drawn at once: the current frame opaque, the others as see-through copies of the atoms (``transparency`` 0..1), e.g. all images of a NEB band over the image shown on the timeline. |
| `FreezeProperty` | Copy a particle property as it was at frame ``freeze_at`` to every frame (OVITO's FreezePropertyModifier). |
| `UnwrapTrajectories` | Undo the wrapping of particles at periodic boundaries, so trajectories are continuous (OVITO's UnwrapTrajectoriesModifier). |

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
| `ColorCoding` | Map a scalar property or expression to colors through a color map. |

### Selection

| Modifier | What it does |
|---|---|
| `ClearSelection` | Remove the 'Selection' property. |
| `DeleteSelected` | Delete the selected particles. |
| `ExpressionSelection` | Select particles where ``expression`` is true (sets 'Selection'). |
| `InvertSelection` | Select the unselected particles and unselect the selected ones. |
| `SelectType` | Select particles whose 'Particle Type' (or another property) is in ``types``. |
| `Slice` | Keep (or select) particles in a slab ``\|n.r - distance\| <= width/2``. |
| `ExpandSelection` | Grow the selection to the neighbors of selected particles (OVITO's ExpandSelectionModifier). |

### Modification

| Modifier | What it does |
|---|---|
| `AffineTransformation` | Apply a 3x3 (or 3x4 with translation) matrix to particles and, optionally, the cell. |
| `ComputeProperty` | Set ``output`` to an expression, e.g. ``ComputeProperty("Height", "Position.Z - 5")``. |
| `Replicate` | Tile the periodic cell ``n = (na, nb, nc)`` times. |
| `WrapAtoms` | Map particles back into the periodic cell (bond periodic images follow). |

### Combining files

| Modifier | What it does |
|---|---|
| `LoadTrajectory` | Positions (and other per-particle properties) of every frame from a trajectory file, on top of a topology from the pipeline source (OVITO's LoadTrajectoryModifier). |
| `CombineDatasets` | Add the particles (and bonds) of another file to the current frame (OVITO's CombineDatasetsModifier). |

### Python

| Modifier | What it does |
|---|---|
| `PythonScript` | Run ``modify(frame, data)`` defined in ``source``. |

### crystal

| Modifier | What it does |
|---|---|
| `ElasticStrain` | Atomic-level elastic strain in a crystal (OVITO's Elastic strain calculation). |
| `GrainSegmentation` | Decompose a polycrystal into grains from PTM orientations (OVITO's Grain segmentation). |
| `IdentifyFCCPlanarFaults` | Classify hcp-like atoms in fcc crystals by planar fault (OVITO's Identify fcc planar faults). |

### diffraction

| Modifier | What it does |
|---|---|
| `PowderDiffraction` | Powder X-ray or neutron diffraction pattern of the (periodic) cell. |
| `StructureFactor` | Static structure factor S(q) (q = 4 pi sin(theta) / lambda in 1/Angstrom), for liquids, glasses and MD. |

### neb

| Modifier | What it does |
|---|---|
| `NebEnergyProfile` | Energy along a NEB band, with the barrier, from frames that are the band's images. |

### statistics

| Modifier | What it does |
|---|---|
| `Histogram` | Histogram of a particle or bond property ('Mass', 'Position.Z', ...). |
| `SpatialBinning` | Coarse-grain a particle property on a 1D, 2D or 3D grid of bins along the cell vectors. |
| `ReduceProperty` | Reduce a particle or bond property to one number per component. |

### steinhardt

| Modifier | What it does |
|---|---|
| `SteinhardtOrderParameters` | Per-particle q_l (and w_l) for every degree in ``degrees``. |

### symmetry

| Modifier | What it does |
|---|---|
| `SymmetryAnalysis` | Space group, Wyckoff positions and symmetry-equivalent atoms (spglib). |

### topology

| Modifier | What it does |
|---|---|
| `BondAngleDistribution` | Histogram of the angles between pairs of bonds at each particle. |
| `BondLengthDistribution` | Histogram of bond lengths from 0 to ``length_cutoff`` in ``bins`` bins (longer bonds are skipped). |
| `FindRings` | Rings of ``minimum_ring_size``..``maximum_ring_size`` bonds in the bond network. |

### transport

| Modifier | What it does |
|---|---|
| `IonicConductivity` | Ionic conductivity from the collective charge displacement, its Nernst-Einstein estimate and the Haven ratio. |
| `VanHoveCorrelation` | Van Hove correlation function at a few times: the self part (how far one ion gets, with peaks at the hop distances) and the distinct part (where the other ions are). |
| `SelfIntermediateScattering` | Self intermediate scattering function F_s(q, t) and its relaxation time. |
| `NonGaussianParameter` | Non-Gaussian parameter alpha_2(t) = 3 <dr^4> / (5 <dr^2>^2) - 1 of the displacements. |
| `SiteHopping` | Site occupancy, hops between sites and residence times of mobile ions. |

### vectors

| Modifier | What it does |
|---|---|
| `VectorVis` | Draw an N x 3 particle property as arrows: 'Force', 'Displacement', 'Velocity', 'Magnetic Moment' (a scalar moment points along z), or any other vector property. |

## File formats

### Readers

Tried in this order when no `format=` is given; reader plugins slot in at priority 50. Details: [File formats](user/formats.md).

| Format | File names | Reads | `format=` aliases |
|---|---|---|---|
| `neb` | (by content) | NEB band, one frame per image: VASP NEB folder (00..NN) or LAMMPS neb run folder (detected), or format='neb' with one file per image (dump.neb.*) | `vasp-neb`, `lammps-neb` |
| `cube` | `*.cube` `*.cub` | Gaussian cube volumetric data (atoms + grid) | `gaussian-cube` |
| `vasp-volumetric` | `CHGCAR*` `CHG*` `PARCHG*` `AECCAR*` `LOCPOT*` `ELFCAR*` | VASP CHGCAR/CHG/PARCHG/AECCAR/LOCPOT/ELFCAR (atoms + grid) | `vasp-chgcar`, `chgcar`, `vasp-locpot`, `locpot`, `vasp-elfcar`, `elfcar` |
| `h5md` | `*.h5md` `*.h5` | H5MD trajectory (LAMMPS dump h5md, MDAnalysis, ...), lazy; needs h5py |  |
| `mdtraj-h5` | `*.h5` | MDTraj/OpenMM HDF5 trajectory (HDF5Reporter), topology and bonds, lazy; needs h5py | `mdtraj`, `openmm-h5` |
| `vaspout` | `vaspout*.h5` | VASP 6 vaspout.h5: ionic steps with forces, energies, stress, velocities, moments, local potential grid, DOS/PDOS and bands, lazy; a file VASP has not closed yet is read from the vasprun.xml/OUTCAR next to it; needs h5py | `vasp-h5`, `vaspout.h5` |
| `vaspwave` | `vaspwave*.h5` | VASP 6 vaspwave.h5 charge density (structure from vaspout.h5 next to it); needs h5py | `vaspwave.h5` |
| `xsf` | `*.xsf` `*.axsf` | XCrySDen XSF/AXSF: structure, ANIMSTEPS trajectory, DATAGRID_3D grids |  |
| `lammps-dump` | `*.lammpstrj` `*.dump` | LAMMPS text dump (atom/custom), lazy | `lammps/dump`, `lammps-dump-text` |
| `xdatcar` | `XDATCAR*` | VASP XDATCAR, fixed or variable cell, lazy | `vasp-xdatcar` |
| `lammps-data` | `*.data` | LAMMPS data file | `lammps/data`, `lammps-data-native` |
| `ase-db` | `*.db` | ASE database (SQLite .db or .json), one row per frame, lazy; selection= query | `db`, `ase.db` |
| `ase-traj` | `*.traj` | ASE binary trajectory (.traj), lazy | `traj` |
| `md-trajectory` | `*.xtc` `*.trr` `*.dcd` `*.nc` `*.ncdf` | GROMACS XTC/TRR, DCD, AMBER NetCDF via MDAnalysis, lazy; topology= (.gro/.pdb/.tpr/...) for atom types | `xtc`, `trr`, `dcd`, `netcdf`, `ncdf`, `mdanalysis` |
| `outcar` | `OUTCAR*` | VASP OUTCAR: ionic steps with forces, energies, stress, magnetic moments, lazy | `vasp-outcar` |
| `vasprun` | `vasprun*.xml` | VASP vasprun.xml: ionic steps with cell, forces, stress, energies, DOS/PDOS and bands, lazy | `vasp-vasprun` |
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

### Visual elements

What gets drawn; every renderer draws every element. Plugins add more ([Visual element plugins](plugins/visuals.md)).

| Element | What it draws |
|---|---|
| `Particles` | Atoms as spheres: the 'Color' and 'Radius' properties, else per-type colors (``type_colors`` first) and radii (``type_radii``, else ``radius_preset``, see :mod:`ewald.radii`), times ``radius_scale``. Particles with Transparency 1 are left out. |
| `Bonds` | Half-bonds as cylinders in their atoms' colors (or the bonds' 'Color'): ``data.bonds`` when present, else a search by ``cutoff`` or VESTA-style ``pairs`` [(A, B, dmin, dmax)]. |
| `Surfaces` | Every ``data.surfaces`` mesh (polyhedra, isosurfaces, grid slices). |
| `Vectors` | Per-particle vectors as arrows: ``data.vectors`` (VectorVis) plus ``vectors`` (a property name, a list, or ``{name: {VectorStyle fields}}``, see :class:`ewald.arrows.VectorStyle`). |
| `Trajectory lines` | ``tables['trajectory-lines']`` (GenerateTrajectoryLines) as one polyline per particle in its atom color; ``width`` in pixels, 0 hides them. |
| `Simulation cell` | The cell edges as lines, ``width`` pixels wide; ``color`` None uses the renderer's ink (black, or light grey on dark backgrounds). ``radius`` (Angstrom) sizes the ray-traced tubes, None = 0.15 % of the cell diagonal. |

### Overlays

`text`, `legend` (color bar), `type_legend` and `scale_bar` are `render()` options; overlay elements are added with `overlays=[...]` or the Render page's "+ Overlay". Plugins add more ([Overlay plugins](plugins/overlays.md)).

| Overlay | What it draws |
|---|---|
| `Coordinate tripod` | Axis directions as arrows in a corner that turn with the camera: Cartesian x, y, z or the cell vectors a, b, c. |
| `Attributes` | Global attributes (temperature, energy, timestep, ...) as "name = value" lines in a corner. |

### Themes and color maps

`theme=` (View › Theme in the GUI) sets element colors, background, cell color and radius preset; every `cmap` takes a matplotlib name or a color map plugin ([Color maps and themes](plugins/styles.md)).

| Theme | What it sets |
|---|---|
| `Default` | Jmol element colors and covalent radii (ewald's defaults) |
| `VESTA` | VESTA's element colors on white |
| `CPK` | CPK element colors |
| `Journal` | Jmol colors muted 35% toward gray, white background, dark gray cell |
| `ASE` | ASE GUI: Jmol colors and covalent radii on white (pair with the 'Simple (ASE)' style) |
| `Dark slide` | Jmol colors on black, light cell lines, for slides |

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

`size`, `direction`, `orthographic`, `zoom`, `background`, `transparent_background`, `ambient_occlusion`, `legend`, `show_cell`, `bonds`, `radius_scale`, `antialiasing`, `text`, `camera`, `renderer`, `samples`, `ao_samples`, `shadows`, `type_legend`, `scale_bar`, `dpi`, `quality`, `scale`, `theme`, `overlays`.

Scene keywords `render()` also takes: `cell_color`, `cell_width`, `bond_radius`, `atom_style`, `mesh_limit`, `sphere_resolution`, `type_colors`, `type_radii`, `bond_pairs`, `trajectory_width`, `radius_preset`, `oxidation_states`, `coordination`, `vectors`, `arrow_resolution`, `visuals`, `boundary`, `boundary_bonds`.

## Command line

Details: [Command line](user/cli.md).

| Command | What it does | Options |
|---|---|---|
| `ewald render INPUT OUTPUT` | render a frame to an image (.png/.jpg/.tif/.svg/.pdf/.eps) | `--frame` `--size` `--direction` `--bonds` `--color-by` `--cmap` `--type-map` `--ao` `--renderer` `--allow-code` `--quality` `--text` `--type-legend` `--scale-bar` `--theme` `--overlay` `--dpi` |
| `ewald movie INPUT OUTPUT` | render frames to .mp4/.gif/.avi/.mov or a PNG sequence | `--fps` `--start` `--stop` `--every` `--size` `--direction` `--bonds` `--color-by` `--cmap` `--type-map` `--time-per-frame` `--renderer` `--allow-code` `--quality` `--text` `--type-legend` `--scale-bar` `--theme` `--overlay` |
| `ewald gui [INPUT]` | open the desktop viewer | `--type-map` `--device` |
| `ewald examples` | list the shipped example files (POSCAR, CIF, extxyz, LAMMPS, ...) | `--copy` `--papers` `--fetch` |
| `ewald info INPUT` | print frames and properties | `--type-map` |
| `ewald describe INPUT` | JSON summary of a frame or, with --trajectory, of the run | `--frame` `--trajectory` `--type-map` |
| `ewald ask INPUT [QUESTION...]` | ask Claude about a structure (needs anthropic + credentials) | `--type-map` `--model` `--allow-code` `--script` `--select` `--ids` `--frame` |
| `ewald narrate INPUT` | timeline of events in a trajectory (Claude writes it if available) | `--type-map` `--no-llm` `--render` `--json` |
| `ewald api` | print the Python API reference for LLMs (llms.txt) | `--with-plugins` |
| `ewald plugins` | list installed plugins and any that failed to load | `--json` `--enable` `--disable` |
| `ewald forge [REQUEST...]` | have Claude write a modifier plugin, test it and save it | `--yes` `--draft` `--overwrite` `--rounds` `--model` |
| `ewald mcp [INPUT]` | serve the analysis tools over MCP (stdio) | `--allow-code` |
| `ewald remote [REMOTE_CMD]` | saved SSH profiles: list, add, remove, test, ls, get |  |
| `ewald inventory` | print or update the feature inventory (docs/features.md) | `-o` `--gui` |

## Claude integration

The same tools serve `ewald ask`, the Python `ewald.llm.Agent` and `ewald mcp` (MCP server). Tools marked *code* exist only with `allow_code=True`. Details: [Claude integration](user/claude.md).

| Tool | What it does |
|---|---|
| `load_structure` | Load a structure or trajectory file (VASP POSCAR/CONTCAR/XDATCAR/OUTCAR/vaspout.h5, LAMMPS dump/data, extxyz, cif, ... ; wildcards allowed) as the session pipeline, replacing any previous one. |
| `describe_frame` | Compact description of one frame after the current modifiers: composition, cell, density, minimum distance, coordination histogram per type, g(r) peaks, property ranges, sanity warnings. |
| `describe_selection` | Describe particles and their surroundings: per-atom neighbor lists, coordination numbers, structure types (if CNA/PTM ran), property values against the environment and frame averages, the nearest defect, and a close-up render with the particles highlighted. |
| `describe_trajectory` | Scan the trajectory (source frames, subsampled to max_frames) and return per-frame descriptor statistics (volume, min distance, MSD and max displacement vs the first frame, coordination) and 'events': frames where a descriptor shifts abruptly. |
| `event_details` | Measure what changes around a frame (e.g. an event from describe_trajectory): which atoms moved and how far (minimum image, by identifier), by type, local vs system-wide, coordination transitions, and the top movers with positions before/after. |
| `list_modifiers` | Available modifiers with their parameters and defaults. |
| `add_modifier` | Append (or insert at 'index') a modifier to the pipeline. |
| `remove_modifier` | Remove the modifier at 'index' (see show_pipeline). |
| `show_pipeline` | Current source, modifier list and frame. |
| `set_frame` | Set the current frame used by tools that take no frame argument. |
| `evaluate_expression` | Evaluate an OVITO-style per-particle expression on a frame and return statistics (count of true values for conditions; min/max/mean/percentiles for numbers). |
| `compute_rdf` | Radial distribution function g(r) of a frame (total, plus partial pairs). |
| `analyze_symmetry` | Space group (Hermann-Mauguin and number), point group, crystal system and the Wyckoff site of every symmetry-unique atom of a frame (spglib). |
| `compute_diffraction` | Powder X-ray or neutron diffraction peaks of a frame (2-theta, d, hkl, relative intensity; strongest first), or the structure factor S(q) when kind='sq'. |
| `compute_msd` | Time-averaged mean squared displacement over source frames (unwrapped, matched by identifier), per type, with a linear-regime fit D = slope/6. |
| `render_image` | Render a frame of the pipeline to PNG and return the image so you can inspect it. |
| `get_view` | The 3D structure view's settings (the same ones the user's notebook sets with view(pipe, ...) and the GUI viewport shows): 'changed' lists what differs from the defaults, 'settings' every current value, 'help' what each means. |
| `set_view` | Change how the structure is drawn in the GUI viewport, notebook pictures and render_image: style preset, theme, element colors and radii, bonds on/off, bond table, cell, background, VESTA boundary, plugin visuals, vectors, camera (direction, zoom, rotate), frame, and picture options (renderer, quality, legends, scale bar, text). |
| `list_visuals` | Visual elements: the built-in ones (Particles, Bonds, Surfaces, Vectors, Trajectory lines, Simulation cell; configured through render_image and modifiers) and installed plugins, which add_visual draws on top in every renderer. |
| `add_visual` | Draw a plugin visual element (see list_visuals) in every later render_image, with optional constructor parameters. |
| `remove_visual` | Stop drawing the visual element at 'index' of the added list. |
| `list_overlays` | Overlay elements drawn on top of rendered images (built-in: Coordinate tripod, Attributes; plus plugins), with their parameters, and the ones currently added. |
| `add_overlay` | Draw an overlay element (see list_overlays) on every later render_image, with optional constructor parameters. |
| `remove_overlay` | Stop drawing the overlay at 'index' of the added list. |
| `list_styles` | Themes (element colors, background, cell color, radius preset; pass one to render_image) and the color maps usable as 'cmap' in ColorCoding, GridSlice and VectorVis: common matplotlib names and plugins, each marked whether it reads in grayscale. |
| `export_file` | Write frames of the pipeline (after its modifiers) to a file: POSCAR, LAMMPS data/dump, extxyz, CIF, XSF, H5MD, XTC, cube/CHGCAR, ASE db, ... The format follows the file name (POSCAR, *.data, *.extxyz, *.h5md) unless given. |
| `get_script` | Standalone Python script that reproduces the current pipeline. |
| `python_api` *code* | Reference for the ewald Python API (data model, expression syntax, differences from OVITO, every modifier signature). |
| `run_python` *code* | Execute Python in the session (variables persist). |

Also: `ewald describe` / `ewald narrate` (rule-based descriptors and a trajectory timeline, Claude-written when a client is available), Modifier Forge (`ewald forge`) and the `llms.txt` API reference (`ewald api`).

## Plugins

Entry point groups `ewald.readers`, `ewald.writers`, `ewald.modifiers`, `ewald.visuals`, `ewald.overlays`, `ewald.colormaps`, `ewald.themes`, `ewald.tools` and `ewald.ui_themes`, or a file in `<config dir>/plugins/` with `@reader` / `@writer` / `@modifier` / `@visual` / `@overlay` / `@colormap` / `@theme` / `@claude_tool` / `@ui_theme`; `ewald plugins` lists what loaded. Details: [Plugins](plugins/index.md).

- `examples/plugins/cell_axes.py`: Example visual element plugins: cell axes and a halo around selected atoms.
- `examples/plugins/figure_extras.py`: Example figure-style plugins: a color map, a theme and an overlay.
- `examples/plugins/local_composition.py`: Example modifier plugin: local chemical composition around every particle.
- `examples/plugins/npz_trajectory.py`: Example reader plugin: trajectories stored as NumPy ``.npz`` archives.
- `examples/plugins/ui_themes.py`: Example UI color themes for the desktop app: "Nord" (dark, frost-blue accents) and "Sepia" (light, warm paper).

## Preferences

Edit › Preferences, one page per group; stored as JSON with only changed values. Scripts read them with `ewald.settings.load()`.

### General

| Key | Label | Default |
|---|---|---|
| `general.layout` | Window layout | `'notebook'` |
| `general.ui_theme` | UI theme | `'graphite'` |
| `general.startup_workspace` | Startup workspace | `'Analyze'` |
| `general.toolbar_style` | Toolbar buttons | `'Icons and text'` |
| `general.font_size` | UI font size | `12` |
| `general.restore_layout` | Restore window layout | `True` |
| `general.lammps_type_prompt` | LAMMPS type names | `'ask'` |
| `general.lammps_type_map` | Default LAMMPS type map | `''` |
| `general.live_reload` | Live reload | `True` |
| `general.live_reload_interval` | Live reload interval | `2.0` |
| `general.docs_url` | Documentation URL | `''` |

### Viewports

| Key | Label | Default |
|---|---|---|
| `viewport.follow_theme` | Background follows the UI theme | `True` |
| `viewport.background` | Background | `'#ECEDE9'` |
| `viewport.projection` | Projection | `'ortho'` |
| `viewport.initial_view` | Initial view direction | `'iso'` |
| `viewport.style_preset` | Default style | `'Ball & stick'` |
| `viewport.rotation_step` | Step rotation | `15.0` |
| `viewport.show_axes` | Show axes tripod | `True` |
| `viewport.show_overlay` | Show frame / timestep text | `True` |
| `viewport.backend` | 3D view | `'auto'` |
| `viewport.hover_info` | Atom info on hover | `True` |
| `viewport.hover_delay` | Hover delay | `0.25` |
| `viewport.sphere_drawing` | Sphere drawing | `'impostor'` |
| `viewport.sphere_resolution` | Sphere resolution | `16` |
| `viewport.mesh_limit` | Impostor threshold | `150000` |
| `viewport.anti_aliasing` | Smooth edges | `True` |
| `viewport.arrow_resolution` | Arrow resolution | `6` |
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
| `bonds.auto_limit` | Automatic bonds up to | `100000` |

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
| `plots.size` | Saved plot size | `'1-column (3.5 in)'` |
| `plots.font_size` | Saved plot font size | `8.0` |
| `render.movie_format` | Movie format | `'MP4 (H.264)'` |

### Compute

| Key | Label | Default |
|---|---|---|
| `compute.device` | Default device | `'auto'` |
| `performance.cache_memory` | Frame cache memory | `2048` |
| `performance.background_compute` | Compute in the background | `True` |
| `compute.threads` | CPU threads | `0` |

### Python

| Key | Label | Default |
|---|---|---|
| `python.run_notebooks` | Run notebooks when opened | `'ask'` |
| `python.save_figures` | Save figures with notebooks | `True` |
| `python.bundle_data_mb` | Copy data files up to (MB) | `50.0` |
| `python.plugin_catalogs` | Plugin catalogs | `''` |
| `python.startup_script` | Console startup script | `''` |
| `python.extra_paths` | Extra module paths | `''` |

### Claude

| Key | Label | Default |
|---|---|---|
| `llm.model` | Model | `'claude-opus-5-5'` |
| `llm.effort` | Effort | `'high'` |

<!-- gui:start -->
## Desktop GUI

`ewald gui [FILE]`. Details: [Desktop GUI](user/gui.md).

### Menus

**File**

- New notebook (`Ctrl+T`)
- Open… (`Ctrl+O`)
- Load remote file (SSH)… (`Ctrl+Alt+O`)
- Remote computers (SSH profiles)… (`Ctrl+Alt+R`)
- Load NEB folder…
- Open example
  - POSCAR.SrTiO3 (VASP POSCAR)
  - Si.cif (CIF)
  - Pt_nanoparticle.xyz (XYZ)
  - Cu_md.extxyz (extended XYZ)
  - XDATCAR.Si (VASP XDATCAR)
  - CHGCAR.Si (VASP CHGCAR)
  - Cu_void.lammpstrj (LAMMPS dump)
  - CuZr.data (LAMMPS data)
- Save notebook (`Ctrl+S`)
- Save notebook as… (`Ctrl+Shift+S`)
- Close notebook (`Ctrl+W`)
- Load session (.ewald)… (`Ctrl+Shift+O`)
- Save session (.ewald) as…
- Render image… (`Ctrl+Shift+R`)
- Export data… (`Ctrl+E`)
- Export movie…
- Live reload
- Quit (`Ctrl+Q`)

**Edit**

- Undo (`Ctrl+Z`)
- Redo (`Ctrl+Shift+Z`)
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
- Theme
  - Default
  - VESTA
  - CPK
  - Journal
  - ASE
  - Dark slide
- Layout
  - Split right (`Ctrl+Alt+\`)
  - Split down
  - Move view left (`Ctrl+Alt+Left`)
  - Move view right (`Ctrl+Alt+Right`)
  - Move view up (`Ctrl+Alt+Up`)
  - Move view down (`Ctrl+Alt+Down`)
  - Move view to editor area
  - Move view to panel
  - Return view home (`Ctrl+Alt+W`)
  - Maximize group (`Ctrl+Alt+M`)
  - Fold or unfold panel (`Ctrl+Alt+J`)
  - Reset layout
- Focus on the viewport (`Ctrl+\`)
- Minimal mode (menus hidden) (`Ctrl+Shift+M`)

**Notebook**

- Run all (`Ctrl+Shift+Return`)
- Run current cell and below
- Restart
- Restart and run all
- Kernel
- Trust notebook…
- Move cell up (`Ctrl+Shift+Up`)
- Move cell down (`Ctrl+Shift+Down`)
- Delete cell
- Show as Python file

**Insert**

- Code cell (`Ctrl+Shift+N`)
- Text cell
- Step
  - Analysis
    - Coordination analysis / RDF
    - Displacement vectors
    - Histogram
    - Spatial binning (profiles, density maps)
    - Reduce property (mean, sum, ...)
    - Steinhardt order parameters (q4, q6)
    - ClusterAnalysis
    - VoronoiAnalysis
    - WignerSeitzAnalysis
    - DislocationAnalysis
  - Structure identification
    - Common neighbor analysis
    - Polyhedral template matching
    - Ackland-Jones bond angles
    - Identify diamond structure
    - Chill+ (ice, hydrate)
    - Centrosymmetry
    - Atomic strain
    - Elastic strain (crystal)
    - Grain segmentation (after PTM)
    - fcc planar faults (after PTM)
  - Symmetry & diffraction
    - Space group / Wyckoff positions (spglib)
    - Powder XRD pattern
    - Powder neutron diffraction
    - Structure factor S(q)
  - Trajectory
    - Mean squared displacement / diffusion
    - Velocity autocorrelation
    - Time averaging (g(r), attributes)
    - Time series of attributes
    - Trajectory lines / paths
    - NEB energy profile / barrier
    - Overlay frames (see-through images)
    - Unwrap trajectories
    - Freeze property (values of frame 0)
    - Load trajectory (topology + dump)
  - Ion transport
    - Ionic conductivity / Haven ratio
    - Van Hove correlation G(r, t)
    - Self intermediate scattering Fs(q, t)
    - Non-Gaussian parameter
    - Site hopping / residence time
  - Bonds & polyhedra
    - Create bonds
    - Edit bonds (add / delete)
    - Coordination polyhedra
    - Bond angle distribution
    - Bond length distribution
    - Ring statistics
  - Volume (CHGCAR, LOCPOT, cube)
    - Isosurface
    - Isosurface, signed (density difference / spin)
    - Slice plane (2D map)
    - Charge density difference
  - Vectors (arrows)
    - Force / displacement / velocity arrows
    - Magnetic moment arrows
    - Arrows colored by length
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
    - Combine datasets (add particles from a file)
  - Selection
    - Expression selection
    - Select type
    - Expand selection (neighbors)
    - Invert selection
    - Clear selection
  - Python
    - Python script
- View
- 3D snapshot of the view
- View settings with the camera
- Analysis
  - Radial distribution g(r)
  - Structure factor S(q)
  - Powder diffraction (XRD)
  - Bond angles
  - Common neighbor analysis
  - Polyhedral template matching
  - Voronoi coordination
  - Clusters
  - Mean squared displacement
  - Velocity autocorrelation
- Plugin step
- Load plugin file…
- Ask Claude to add steps…

**Tools**

- Orbit / pan / zoom (mouse)
- Pick atom
- Measure distance
- Measure angle
- Edit bonds
- Ask Claude about selection… (`Ctrl+J`)

**Window**

- Notebook workspace (`Ctrl+1`)
- Render workspace (`Ctrl+2`)
- Notebook
- Plugins (`Ctrl+Shift+X`)
- Display settings
- 3D view
- Python console
- Data inspector
- Plots
- New plot view
- Log
- AI chat
- Command palette (`Ctrl+K`)

**Help**

- User manual
- Desktop GUI
- Modifiers
- Python scripting
- Writing plugins
- Python API reference
- Open plugin folder
- About EWALD

**Viewport (right click)**, entries shown only when they apply (picked atoms, phonon modes, NEB band, volumetric grid, trajectory):

- N atoms picked
- Center view on picked
- Select picked atoms
- Delete picked atoms
- Clear picks
- Phonon band structure
- NEB energy profile
- Add isosurface
- Play (`Space`)
- Overlay other frames (see-through)
- Overlay transparency
  - 50 %
  - 70 %
  - 85 %
  - 95 %
- Overlay only atoms that move
- Show
  - Bonds
  - Simulation cell
- View along
  - a
  - b
  - c
  - a*
  - b*
  - c*
- Orthographic
- Zoom to fit
- Ask Claude about selection…

### Toolbar

Open, Save, Notebook, Render, Search commands, steps, cells… Ctrl K, Compute device switch (CPU / CUDA / auto).

### Panels

- Workspaces: Notebook, Render
- Docks: AI chat, Display, Notebook, Plugins, Remotes
- Editor area tabs: 3D view
- Panel tabs: Python console, Data inspector, Plots, Log (tabs drag between groups: centre = tab, edge = split; View › Layout)
- Viewport tools: Orbit / pan / zoom (mouse); Pick atom: shows id, type, position and properties; Shift+click picks several; Measure distance: click two atoms; Measure angle: click three atoms (vertex in the middle); Edit bonds: click two atoms to add or delete the bond between them

<!-- gui:end -->
