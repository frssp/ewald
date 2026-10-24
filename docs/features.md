# Feature inventory

Everything ewald has on `main`, generated from the code by
`python -m ewald inventory -o docs/features.md --gui`. Do not edit it by hand: `tests/test_docs.py`
(and `tests/test_gui.py` for the GUI section) fail when it is out of date. Open work and ideas are
in the [TODO list](todo.md); how each feature works is in the rest of this manual.

## Python API

`import ewald`: `Cell`, `DataCollection`, `Pipeline`, `Modifier`, `PythonModifier`, `AllFramesModifier`, `Frames`, `each_frame`, `all_frames`, `ComputeContext`, `ComputeCancelled`, `StaticSource`, `import_file`, `load`, `View`, `view`, `export_file`, `get_backend`, `gpu_available`, `rdf`, `msd`, `vacf`, `sq`, `xrd`, `bond_angles`, `cna`, `ptm`, `voronoi`, `clusters`, `run_info`, `check_run`, `load_runs`, `arrhenius`, `workflow_table`. Details: [Python scripting](user/scripting.md) and the [API reference](reference/api.md).

## Modifiers

65 built-in modifiers in `ewald.modifiers` (OVITO spellings such as `CommonNeighborAnalysisModifier` also import). Details: [Modifiers](user/modifiers.md).

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

### phonon

| Modifier | What it does |
|---|---|
| `PhononMode` | Animate one phonon mode read from phonopy band.yaml/mesh.yaml/qpoints.yaml or a VASP IBRION=5-8 OUTCAR: frame ``f`` of ``N`` shows phase ``2 pi f / N`` of the vibration, so the trajectory loops over one period. |

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
| `phonopy-yaml` | `band.yaml` `mesh.yaml` `qpoints.yaml` `*.yaml` | phonopy band.yaml/mesh.yaml/qpoints.yaml with eigenvectors: phonon modes (PhononMode animates one); frames= per period, structure= for qpoints.yaml | `phonopy`, `band.yaml`, `phonon` |
| `lammps-dump` | `*.lammpstrj` `*.dump` | LAMMPS text dump (atom/custom), lazy | `lammps/dump`, `lammps-dump-text` |
| `xdatcar` | `XDATCAR*` | VASP XDATCAR, fixed or variable cell, lazy | `vasp-xdatcar` |
| `lammps-data` | `*.data` | LAMMPS data file | `lammps/data`, `lammps-data-native` |
| `ase-db` | `*.db` | ASE database (SQLite .db or .json), one row per frame, lazy; selection= query | `db`, `ase.db` |
| `ase-traj` | `*.traj` | ASE binary trajectory (.traj), lazy | `traj` |
| `md-trajectory` | `*.xtc` `*.trr` `*.dcd` `*.nc` `*.ncdf` | GROMACS XTC/TRR, DCD, AMBER NetCDF via MDAnalysis, lazy; topology= (.gro/.pdb/.tpr/...) for atom types | `xtc`, `trr`, `dcd`, `netcdf`, `ncdf`, `mdanalysis` |
| `vasp-phonon` | `OUTCAR*` | VASP OUTCAR of an IBRION=5-8 run: Gamma phonon modes (PhononMode animates one) | `outcar-phonon` |
| `outcar` | `OUTCAR*` | VASP OUTCAR: ionic steps with forces, energies, stress, magnetic moments, lazy | `vasp-outcar` |
| `vasprun` | `vasprun*.xml` | VASP vasprun.xml: ionic steps with cell, forces, stress, energies, DOS/PDOS and bands, lazy | `vasp-vasprun` |
| `extxyz` | `*.xyz` `*.extxyz` | XYZ / extended XYZ (Lattice=, Properties=), lazy | `xyz`, `ext-xyz` |
| `cif` | `*.cif` | Crystallographic Information File via ASE |  |
| `ase` | `*POSCAR*` `*CONTCAR*` `*.vasp` `*.poscar` `OUTCAR*` `vasprun*.xml` `*.xyz` `*.extxyz` `*.cif` `*.pdb` `*.gro` `*.gen` `*.res` `*.cell` `*.castep` `*.pwi` `*.pwo` `geometry.in` | anything ase.io.read understands (POSCAR, OUTCAR, vasprun.xml, extxyz, cif, ...); other format= names are passed to ASE |  |
| `lammps-data-ase` | (by `format=` only) | LAMMPS data file via ASE |  |

Remote paths (`host:/path`, `ssh://...`) work with every reader; LAMMPS dumps and XDATCARs are streamed, other formats are downloaded to a local cache.

### Writers

`export_file(source, path, format=...)` and File › Export › Data pick a writer by `format=` or by file name; *one frame* writers need e.g. `frames=[-1]`.

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
| `Particles` | Atoms as spheres: the 'Color' and 'Radius' properties, else per-type colors (``type_colors`` first) and radii (``type_radii``, else ``radius_preset``, see :mod:`ewald.radii`), times ``radius_scale``. Particles with Transparency 1 are left out. ``selection_color`` (e.g. red, as in OVITO) draws the particles of a 'Selection' property in that color; None leaves them as they are. |
| `Bonds` | Half-bonds as cylinders in their atoms' colors (or the bonds' 'Color'): ``data.bonds`` when present, else a search by ``cutoff`` or VESTA-style ``pairs`` [(A, B, dmin, dmax)]. |
| `Surfaces` | Every ``data.surfaces`` mesh (polyhedra, isosurfaces, grid slices). ``styles`` restyles them by name on top of what their step made: ``{name: {"visible", "color", "transparency", "edges"}}``. |
| `Vectors` | Per-particle vectors as arrows: ``data.vectors`` (VectorVis) plus ``vectors`` (a property name, a list, or ``{name: {VectorStyle fields}}``, see :class:`ewald.arrows.VectorStyle`). |
| `Trajectory lines` | ``tables['trajectory-lines']`` (GenerateTrajectoryLines) as one polyline per particle, in its atom color or a color map by Time, Frame, step length or a property, over a Time window or a trail. |
| `Simulation cell` | The cell edges as lines, ``width`` pixels wide; ``color`` None uses the renderer's ink (black, or light grey on dark backgrounds). ``radius`` (Angstrom) sizes the ray-traced tubes, None = 0.15 % of the cell diagonal. |
| `Measurements` | The view's ``measurements`` (:mod:`ewald.measure`): lines through 2, 3 or 4 atoms (nearest periodic images), ``width`` pixels wide in ``color``; the values are the Measurements overlay (:class:`ewald.overlays.MeasurementLabels`). Indices are those of the real atoms, also in a boundary or replicated view. |

### Overlays

`text`, `legend` (color bar), `type_legend` and `scale_bar` are `render()` options; overlay elements are added with `overlays=[...]` or the Render page's "+ Overlay". Plugins add more ([Overlay plugins](plugins/overlays.md)).

| Overlay | What it draws |
|---|---|
| `Coordinate tripod` | Axis directions as arrows in a corner that turn with the camera: Cartesian x, y, z or the cell vectors a, b, c. |
| `Attributes` | Global attributes (temperature, energy, timestep, ...) as "name = value" lines in a corner. |
| `Atom labels` | A text label on every atom, e.g. its type and index. ``text`` is a format string with ``{type}``, ``{index}`` (0-based), ``{id}`` (the Particle Identifier, else index + 1) and any per-atom property whose name has no spaces (``{Charge:.2f}``); a bare 'type', 'index', 'id' or property name means that field. ``types`` limits the labels to those particle types; ``size`` is the font size in px (None: the figure's small size); ``color`` None is the ink color. Hidden atoms (Transparency 1) get no label, and at most ``max_labels`` atoms are labelled (the nearest to the camera). |
| `Measurements` | The values of the view's ``measurements`` (:mod:`ewald.measure`) next to their lines: a distance in Angstrom at its middle, an angle at its vertex, a dihedral at its middle bond. ``size`` in px (None: the figure's small size); ``color`` None is the measurement line color. |

### Themes and color maps

`theme=` (View › Structure theme in the GUI) sets element colors, background, cell color and radius preset; every `cmap` takes a matplotlib name or a color map plugin ([Color maps and themes](plugins/styles.md)).

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

| `quality=` | `samples` | `ao_samples` | `noise_threshold` | `shadows` | `antialiasing` | `ambient_occlusion` | `sphere_resolution` | `scale` |
|---|---|---|---|---|---|---|---|---|
| `draft` | 1 | 0 | 0.0 | False | False | False | 8 | 0.5 |
| `standard` | 4 | 4 | 0.0 | True | True | False | 16 | 1.0 |
| `publication` | 16 | 8 | 0.01 | True | True | True | 28 | 1.0 |

### `render()` options

`size`, `direction`, `orthographic`, `zoom`, `background`, `transparent_background`, `ambient_occlusion`, `legend`, `show_cell`, `bonds`, `radius_scale`, `antialiasing`, `text`, `camera`, `renderer`, `samples`, `ao_samples`, `noise_threshold`, `shadows`, `type_legend`, `scale_bar`, `dpi`, `quality`, `scale`, `theme`, `overlays`, `region`, `labels`, `light`, `progress`.

Scene keywords `render()` also takes: `cell_color`, `cell_width`, `bond_radius`, `atom_style`, `mesh_limit`, `sphere_resolution`, `type_colors`, `type_radii`, `bond_pairs`, `trajectory_width`, `trajectory_color_by`, `trajectory_colormap`, `trajectory_color_range`, `trajectory_time_range`, `trajectory_trail`, `trajectory_radius`, `radius_preset`, `oxidation_states`, `coordination`, `vectors`, `arrow_resolution`, `visuals`, `boundary`, `boundary_bonds`, `replicate`, `replicate_cell`, `show_atoms`, `surfaces`, `cell_radius`, `type_transparency`, `selection_color`, `measurements`.

## Command line

Details: [Command line](user/cli.md).

| Command | What it does | Options |
|---|---|---|
| `ewald render INPUT OUTPUT` | render a frame to an image (.png/.jpg/.tif/.svg/.pdf/.eps) | `--frame` `--size` `--direction` `--bonds` `--color-by` `--cmap` `--type-map` `--ao` `--renderer` `--allow-code` `--quality` `--text` `--type-legend` `--scale-bar` `--theme` `--overlay` `--dpi` |
| `ewald movie INPUT OUTPUT` | render frames to .mp4/.gif/.avi/.mov or a PNG sequence | `--fps` `--start` `--stop` `--every` `--size` `--direction` `--bonds` `--color-by` `--cmap` `--type-map` `--time-per-frame` `--renderer` `--allow-code` `--quality` `--text` `--type-legend` `--scale-bar` `--theme` `--overlay` |
| `ewald gui [INPUT...]` | open the desktop viewer | `--type-map` `--device` |
| `ewald examples` | list the shipped example files (POSCAR, CIF, extxyz, LAMMPS, ...) | `--copy` `--papers` `--fetch` |
| `ewald info INPUT` | print frames and properties | `--type-map` |
| `ewald describe INPUT` | JSON summary of a frame or, with --trajectory, of the run | `--frame` `--trajectory` `--type-map` |
| `ewald check [INPUT...]` | what a run is (from INCAR / log.lammps) and whether it converged and equilibrated | `--json` `--type-map` |
| `ewald ask INPUT [QUESTION...]` | ask AI about a structure (Claude by default; needs the provider's package + key) | `--type-map` `--model` `--provider` `--profile` `--base-url` `--allow-code` `--script` `--select` `--ids` `--frame` |
| `ewald narrate INPUT` | timeline of events in a trajectory (an AI writes it if available) | `--type-map` `--no-llm` `--render` `--model` `--provider` `--profile` `--base-url` `--json` |
| `ewald api` | print the Python API reference for LLMs (llms.txt) | `--with-plugins` |
| `ewald plugins` | list installed plugins and any that failed to load | `--json` `--enable` `--disable` |
| `ewald forge [REQUEST...]` | have an AI write a modifier plugin, test it and save it | `--yes` `--draft` `--overwrite` `--rounds` `--model` `--provider` `--profile` `--base-url` |
| `ewald eval` | score an AI model on tasks with known answers (costs API tokens; see --dry-run) | `--model` `--provider` `--profile` `--base-url` `--effort` `--tasks` `--category` `--list` `--allow-code` `--repeat` `--max-turns` `--out` `--compare` `--dry-run` `--price` `--no-download` |
| `ewald mcp [INPUT]` | serve the analysis tools over MCP (stdio) | `--allow-code` |
| `ewald remote [REMOTE_CMD]` | saved SSH profiles: list, add, remove, test, ls, get |  |
| `ewald inventory` | print or update the feature inventory (docs/features.md) | `-o` `--gui` |

## AI integration

The same tools serve Ask AI and the AI chat, `ewald ask`, the Python `ewald.llm.Agent` and `ewald mcp` (MCP server), whichever provider answers. They work on a notebook (the AI chat's current tab, or a session notebook of their own) and change the structure and every other result only through its cells: tools marked *cell* write a cell and run it (a name and JSON values, no model-written code), *notebook* tools read and edit cells, and `run_cell` (*code*) runs a cell the AI wrote, only with `allow_code=True` or Preferences > AI > Let the AI run cells. The others only read. Details: [AI integration](user/claude.md).

Providers (Preferences > AI, `--provider`; `@ai_provider` plugins add more):

| Provider | Name | Default model | Key |
|---|---|---|---|
| Anthropic (Claude) | `anthropic` | `claude-opus-5-5` | `ANTHROPIC_API_KEY` or keychain |
| OpenAI | `openai` | `gpt-5` | `OPENAI_API_KEY` or keychain |
| Google Gemini | `gemini` | `gemini-2.5-pro` | `GEMINI_API_KEY` or keychain |
| OpenAI-compatible server | `openai-compatible` | `llama3.1` | optional |

Tools:

| Tool | What it does |
|---|---|
| `load_structure` *cell* | Load a structure or trajectory file (VASP POSCAR/CONTCAR/XDATCAR/OUTCAR/vaspout.h5, LAMMPS dump/data, extxyz, cif, ... ; wildcards allowed) as the pipeline, replacing any previous one: a pipe = load(...) cell at the end of the notebook, run at once. |
| `describe_frame` | Compact description of one frame after the current modifiers: composition, cell, density, minimum distance, coordination histogram per type, g(r) peaks, property ranges, sanity warnings, and how the 3D view colors the atoms ('atom_colors': which step or view setting, which property, and the color of each category or the color map and range). |
| `describe_selection` | Describe particles and their surroundings: per-atom neighbor lists, coordination numbers, structure types (if CNA/PTM ran), property values against the environment and frame averages, the nearest defect, and a close-up render with the particles highlighted. |
| `describe_trajectory` | Scan the trajectory (source frames, subsampled to max_frames) and return per-frame descriptor statistics (volume, min distance, MSD and max displacement vs the first frame, coordination) and 'events': frames where a descriptor shifts abruptly. |
| `event_details` | Measure what changes around a frame (e.g. an event from describe_trajectory): which atoms moved and how far (minimum image, by identifier), by type, local vs system-wide, coordination transitions, and the top movers with positions before/after. |
| `list_modifiers` | Available modifiers with their parameters and defaults. |
| `add_modifier` *cell* | Add a modifier to the pipeline (at the end, or before the one at pipeline 'index') as a step cell pipe.add(Name(...)) in the notebook, run at once. |
| `remove_modifier` *cell* | Remove the modifier at pipeline 'index' (see show_pipeline) by deleting its step cell from the notebook. |
| `show_pipeline` | Current source, modifier list and frame. |
| `set_frame` | Set the current frame used by tools that take no frame argument (only what you look at; the view's frame is set_view frame). |
| `evaluate_expression` | Evaluate an OVITO-style per-particle expression on a frame and return statistics (count of true values for conditions; min/max/mean/percentiles for numbers). |
| `compute_rdf` | Radial distribution function g(r) of a frame (total, plus partial pairs). |
| `analyze_symmetry` | Space group (Hermann-Mauguin and number), point group, crystal system and the Wyckoff site of every symmetry-unique atom of a frame (spglib). |
| `compute_diffraction` | Powder X-ray or neutron diffraction peaks of a frame (2-theta, d, hkl, relative intensity; strongest first), or the structure factor S(q) when kind='sq'. |
| `compute_msd` | Time-averaged mean squared displacement over source frames (unwrapped, matched by identifier), per type, with a linear-regime fit D = slope/6. |
| `check_run` | What the loaded run is and whether it can be trusted, from its input and log files (INCAR, OSZICAR, OUTCAR, log.lammps): code, kind (md/relax/static/neb), ensemble, target temperature, MD time step and time per stored frame, then findings: SCF steps that hit NELM, relaxation not converged, the first equilibrated frame of MD, temperature off target, energy drift, LAMMPS warnings/errors, atoms too close. |
| `render_image` | Render a frame of the pipeline to PNG and return the image so you can inspect it. |
| `get_view` | The 3D structure view's settings (the same ones the user's notebook sets with view(pipe, ...) and the GUI viewport shows): 'changed' lists what differs from the defaults, 'settings' every current value, 'elements' the same values by visual element (Particles, Bonds, Surfaces, Simulation cell, ...), 'atom_colors' what colors the atoms now (a step's structure-type colors, a color map, paint conditions or the type colors, with each color), 'measured' the values of its distance/angle measurements, 'help' what each means. |
| `set_view` *cell* | Change how the structure is drawn in the GUI viewport, notebook pictures and render_image: style preset, theme, element colors and radii, atoms on/off, bonds on/off, bond table, cell, background, VESTA boundary, polyhedra/isosurface look by surface name (surfaces), distance/angle measurements, key light, atom labels, plugin visuals, vectors, the look of trajectory lines (trajectory_*: the lines themselves come from a GenerateTrajectoryLines step, so add_modifier it first), camera (direction, zoom, rotate), frame, and picture options (renderer, quality, legends, scale bar, text). |
| `list_visuals` | Visual elements: the built-in ones (Particles, Bonds, Surfaces, Vectors, Trajectory lines, Simulation cell; configured through set_view and modifiers) and installed plugins, which add_visual draws on top in every renderer. |
| `add_visual` *cell* | Draw a plugin visual element (see list_visuals) in the view, with optional constructor parameters: the view setting 'visuals', written to the view cell like set_view. |
| `remove_visual` *cell* | Stop drawing the visual element at 'index' of the added list (the view cell changes). |
| `list_overlays` | Overlay elements drawn on top of rendered images (built-in: Coordinate tripod, Attributes; plus plugins), with their parameters, and the ones currently added. |
| `add_overlay` *cell* | Draw an overlay element (see list_overlays) on the view's pictures, with optional constructor parameters: the view setting 'overlays', written to the view cell like set_view. |
| `remove_overlay` *cell* | Stop drawing the overlay at 'index' of the added list (the view cell changes). |
| `list_styles` | Themes (element colors, background, cell color, radius preset; set_view theme, or render_image for one picture) and the color maps usable as 'cmap' in ColorCoding, GridSlice and VectorVis: common matplotlib names and plugins, each marked whether it reads in grayscale. |
| `export_file` *cell* | Write frames of the pipeline (after its modifiers) to a file: POSCAR, LAMMPS data/dump, extxyz, CIF, XSF, H5MD, XTC, cube/CHGCAR, ASE db, ... as an export_file(...) cell in the notebook, run at once. |
| `get_script` | The notebook as a standalone Python script (percent format): it reproduces everything the tools did, since they work only through its cells. |
| `python_api` | Reference for the ewald Python API (data model, expression syntax, differences from OVITO, every modifier signature, the notebook's cell kinds, show/view/paint, and the modifier plugins installed here). |
| `list_workflows` | Recent workflows (calculations that ran in steps, e.g. AiiDA WorkGraphs, or folders of VASP/LAMMPS/NEB/phonopy calculations: source 'folders') of a workflow source, newest first: id, kind, formula, state and the path that opens each, and today's date. |
| `open_workflow` *cell* | Open a workflow, one of its steps or one of its outputs as the pipeline (like load_structure: a pipe = load(...) cell at the end of the notebook, run at once). |
| `describe_workflow` | Steps of a workflow (nested; restarts joined), their states, numbers (energies, gaps) and outputs, each with the path load_structure opens, and the analyses its plugin offers (add_analysis). |
| `add_analysis` *cell* | Add one of a workflow's analyses (listed under 'analyses' by describe_workflow, e.g. the potential lineup of a band-offset workflow: potential, VBM, CBM and Fermi level on one axis) to the notebook and run it. |
| `list_cells` *notebook* | The cells of the user's notebook: index, id, kind (code, step = one pipe.add(...), view = one show(pipe, ...), markdown), state (new = not run, ok, error, stale = out of date), first line and output kinds. |
| `read_cell` *notebook* | One notebook cell: its full source and outputs (printed text, errors, values; plots come back as images). |
| `insert_cell` *notebook* | Insert a new cell into the user's notebook (default: at the end; 'index' puts it before the cell at that index). |
| `edit_cell` *notebook* | Change a notebook cell. |
| `delete_cell` *notebook* | Delete a notebook cell (and the steps it added to the pipeline). |
| `move_cell` *notebook* | Move a notebook cell to position 'to' (0-based); the pipeline follows the new order. |
| `run_cell` *notebook* *code* | Run a notebook cell, like Shift+Enter, and return its outputs. |

Also: `ewald describe` / `ewald narrate` (rule-based descriptors and a trajectory timeline, AI-written when a client is available), Modifier Forge (`ewald forge`) and the `llms.txt` API reference (`ewald api`).

## Plugins

Entry point groups `ewald.readers`, `ewald.writers`, `ewald.modifiers`, `ewald.visuals`, `ewald.overlays`, `ewald.colormaps`, `ewald.themes`, `ewald.tools`, `ewald.ui_themes`, `ewald.ai_providers`, `ewald.panels`, `ewald.menu_items`, `ewald.settings_pages` and `ewald.workflows`, or a file in `<config dir>/plugins/` with `@reader` / `@writer` / `@modifier` / `@visual` / `@overlay` / `@colormap` / `@theme` / `@ai_tool` (or `@claude_tool`) / `@ui_theme` / `@ai_provider` / `@panel` (an activity-bar side bar or a tab under the 3D view) / `@menu_item` (an entry in a menu or the 3D view's right-click menu) / `@settings_page` (a Preferences page) / `@workflow_source` (workflows a side bar lists and a notebook opens); `ewald plugins` lists what loaded. Details: [Plugins](plugins/index.md).

- `examples/plugins/cell_axes.py`: Example visual element plugins: cell axes and a halo around selected atoms.
- `examples/plugins/figure_extras.py`: Example figure-style plugins: a color map, a theme and an overlay.
- `examples/plugins/local_composition.py`: Example modifier plugin: local chemical composition around every particle.
- `examples/plugins/npz_trajectory.py`: Example reader plugin: trajectories stored as NumPy ``.npz`` archives.
- `examples/plugins/sim_agent_panel.py`: Example GUI plugin: a "Sim agent" side bar with its own activity-bar button, an "Agent jobs" tab under the 3D view, menu entries and a page in Preferences.
- `examples/plugins/ui_themes.py`: Example UI themes for the desktop app: "Nord" (dark, frost-blue accents), "Sepia" (light, warm paper) and "Studio" (dark, compact, square corners, its own fonts and extra style sheet rules).

## Preferences

Edit › Preferences, one page per group; stored as JSON with only changed values. Scripts read them with `ewald.settings.load()`.

### General

| Key | Label | Default |
|---|---|---|
| `general.ui_theme` | UI theme | `'graphite'` |
| `general.startup_workspace` | Startup workspace | `'Notebook'` |
| `general.toolbar_style` | Toolbar buttons | `'Icons and text'` |
| `general.activity_hidden` | Hidden activity bar buttons | `''` |
| `general.font_size` | UI font size | `12` |
| `general.restore_layout` | Restore window layout | `True` |
| `general.lammps_type_prompt` | LAMMPS type names | `'ask'` |
| `general.lammps_type_map` | Default LAMMPS type map | `''` |
| `general.live_reload` | Live reload | `True` |
| `general.live_reload_interval` | Live reload interval | `2.0` |
| `general.docs_url` | Documentation URL | `''` |

### 3D view

| Key | Label | Default |
|---|---|---|
| `viewport.follow_theme` | Background follows the UI theme | `True` |
| `viewport.background` | Background | `'#ECEDE9'` |
| `viewport.projection` | Projection | `'ortho'` |
| `viewport.initial_view` | Initial view direction | `'iso'` |
| `viewport.style_preset` | Default style | `'Ball & stick'` |
| `viewport.rotation_step` | Step rotation | `15.0` |
| `viewport.show_axes` | Show axes tripod | `True` |
| `viewport.render_frame` | Render frame in the 3D view | `False` |
| `viewport.show_overlay` | Show frame / timestep text | `True` |
| `viewport.backend` | Viewport renderer | `'auto'` |
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

### Atoms & bonds

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
| `render.noise_threshold` | Noise threshold | `0.01` |
| `render.shadows` | Shadows | `True` |
| `render.size_preset` | Size preset | `'1-column (3.5 in, 4:3)'` |
| `render.dpi` | Resolution | `600` |
| `render.background` | Background | `'white'` |
| `render.output_dir` | Output folder | `''` |
| `render.type_legend` | Particle type legend | `False` |
| `render.scale_bar` | Scale bar | `False` |
| `render.movie_format` | Movie format | `'MP4 (H.264)'` |

### Plots

| Key | Label | Default |
|---|---|---|
| `plots.size` | Saved plot size | `'1-column (3.5 in)'` |
| `plots.font_size` | Saved plot font size | `8.0` |
| `python.plot_style` | Style of your own plots | `'Publication'` |

### Notebooks

| Key | Label | Default |
|---|---|---|
| `python.run_notebooks` | Run notebooks when opened | `'never'` |
| `python.save_figures` | Save figures with notebooks | `True` |
| `python.bundle_data_mb` | Copy data files up to (MB) | `50.0` |

### Compute

| Key | Label | Default |
|---|---|---|
| `compute.device` | Device | `'cpu'` |
| `performance.cache_memory` | Frame cache memory | `2048` |
| `performance.background_compute` | Compute in the background | `True` |
| `compute.threads` | CPU threads | `0` |

### Python

| Key | Label | Default |
|---|---|---|
| `python.startup_script` | Console startup script | `''` |
| `python.extra_paths` | Extra module paths | `''` |

### AI

| Key | Label | Default |
|---|---|---|
| `llm.provider` | Provider | `'anthropic'` |
| `llm.model` | Model | `''` |
| `llm.profiles` | Saved connections | `{}` |
| `llm.base_url` | Server URL | `''` |
| `llm.route_prefix` | Model prefix | `''` |
| `llm.key_header` | API key header | `''` |
| `llm.headers` | Extra headers | `''` |
| `llm.timeout` | Timeout | `0.0` |
| `llm.effort` | Effort | `'high'` |
| `llm.stream` | Stream answers | `True` |
| `llm.code_cells` | Code in answers | `'insert'` |
| `llm.run_cells` | Let the AI run cells | `False` |
| `llm.follow_cells` | Follow the AI's cells | `'follow'` |
| `llm.read_folders` | Folders the AI may read | `''` |
| `llm.confirm_hosts` | Ask before the AI connects to a host | `True` |

### Plugins

| Key | Label | Default |
|---|---|---|
| `python.plugin_catalogs` | Plugin catalogs | `''` |

<!-- gui:start -->
## Desktop GUI

`ewald gui [FILE]`. Details: [Desktop GUI](user/gui.md).

### Menus

**File**

- New notebook (`Ctrl+N`)
- Open… (`Ctrl+O`)
- Open recent
- Open example
  - POSCAR.SrTiO3 (VASP POSCAR)
  - Si.cif (CIF)
  - Pt_nanoparticle.xyz (XYZ)
  - Cu_md.extxyz (extended XYZ)
  - XDATCAR.Si (VASP XDATCAR)
  - CHGCAR.Si (VASP CHGCAR)
  - Cu_void.lammpstrj (LAMMPS dump)
  - CuZr.data (LAMMPS data)
- Open from
  - Remote file (SSH)… (`Ctrl+Alt+O`)
  - Location… (`Ctrl+L`)
  - NEB folder…
  - Workflow folder…
- Save notebook (`Ctrl+S`)
- Save notebook as… (`Ctrl+Shift+S`)
- Close notebook (`Ctrl+W`)
- Export
  - Image… (`Ctrl+Shift+R`)
  - Data… (`Ctrl+E`)
  - Movie…
  - Session (.ewald)…
- Import session (.ewald)…
- Live reload
- Quit (`Ctrl+Q`)

**Edit**

- Undo (`Ctrl+Z`)
- Redo (`Ctrl+Shift+Z`)
- Cut cells
- Copy cells
- Paste cells below
- Paste cells above
- Delete cells
- Undo cell deletion
- Merge cells
- Preferences… (`Ctrl+,`)

**View**

- Look along
  - a
  - b
  - c
  - a*
  - b*
  - c*
  - Top (+z)
  - Front (-y)
  - Right (+x)
  - Isometric
- Orthographic
- Zoom to fit (`Ctrl+0`)
- Show
  - Bonds
  - Simulation cell
  - Render frame
- Replicate cell
  - Off (1 × 1 × 1)
  - 2 × 2 × 2
  - 3 × 3 × 3
  - 2 × 2 × 1
  - 3 × 3 × 1
  - 4 × 4 × 1
  - Custom…
  - Cell lines
    - Outline of the block
    - Original cell
    - Every cell
- Structure theme
  - Default
  - VESTA
  - CPK
  - Journal
  - ASE
  - Dark slide
  - Dark background
  - White background
- Layout
  - Split right (`Ctrl+Alt+\`)
  - Split down
  - Move tab left (`Ctrl+Alt+Left`)
  - Move tab right (`Ctrl+Alt+Right`)
  - Move tab up (`Ctrl+Alt+Up`)
  - Move tab down (`Ctrl+Alt+Down`)
  - Move tab to editor area
  - Move tab to panel
  - Return tab home (`Ctrl+Alt+W`)
  - Maximize group (`Ctrl+Alt+M`)
  - Fold or unfold panel (`Ctrl+Alt+J`)
  - Reset layout
  - Fold all View cards
  - Unfold all View cards
- Focus on the structure view (`Ctrl+\`)
- Minimal mode (menus hidden) (`Ctrl+Shift+M`)
- Full screen (`F11`)

**Notebook**

- Run current cell
- Run current cell and below
- Run all (`Ctrl+Shift+Return`)
- Interrupt
- Restart
- Restart & clear outputs
- Kernel
- Move cell up (`Ctrl+Shift+Up`)
- Move cell down (`Ctrl+Shift+Down`)
- Change cell to code (`Ctrl+M, Y`)
- Change cell to text (`Ctrl+M, M`)
- Clear cell output (`Ctrl+Shift+Backspace`)
- Clear all outputs (`Ctrl+Alt+Shift+Backspace`)
- Trust notebook…
- Show as Python file

**Insert**

- Code cell (`Ctrl+Shift+N`)
- Text cell
- Cell above
- Cell below
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
  - Phonons
    - Phonon mode animation
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
- Plot
- Quick analysis
  - Run check (convergence, equilibration)
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
- Snapshot of the structure view
- Write view settings to notebook

**Tools**

- Orbit, pan and zoom
- Pick atom
- Measure distance
- Measure angle
- Edit bonds
- Clear picks and measurements
- Command palette (`Ctrl+K`)

**Window**

- Notebook workspace (`Ctrl+1`)
- Render workspace (`Ctrl+2`)
- Notebook
- AI chat (`Ctrl+J`)
- Workflows
- Remotes (`Ctrl+Alt+R`)
- Plugins (`Ctrl+Shift+X`)
- Structure
- View settings
- Python console
- Data inspector
- Plots
- New plot tab
- Log

**Help**

- Tutorials
- User manual
- Desktop GUI
- Modifiers
- Python scripting
- Writing plugins
- Python API reference
- Keyboard shortcuts
- About EWALD

**Viewport (right click)**, entries shown only when they apply (picked atoms, phonon modes, NEB band, volumetric grid, trajectory):

- N atoms picked
- Center view on picked
- Select picked atoms
- Delete picked atoms
- Clear picks
- Keep measurement in the view
- Clear measurements
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
  - Render frame
- Replicate cell
  - Off (1 × 1 × 1)
  - 2 × 2 × 2
  - 3 × 3 × 3
  - 2 × 2 × 1
  - 3 × 3 × 1
  - 4 × 4 × 1
  - Custom…
  - Cell lines
    - Outline of the block
    - Original cell
    - Every cell
- Look along
  - a
  - b
  - c
  - a*
  - b*
  - c*
- Orthographic
- Zoom to fit
- Ask AI about selection…

### Toolbar

Open, Save, Notebook, Render, Search commands and steps… Ctrl+K, Compute device switch (CPU / CUDA / auto).

### Panels

- Workspaces: Notebook, Render
- Docks: AI chat, Notebook, Plugins, Remotes
- Editor area tabs: Structure
- Panel tabs: Python console, View, Data inspector, Plots, Log (tabs drag between groups: centre = tab, edge = split; View › Layout)
- Viewport tools: Orbit / pan / zoom (mouse); Pick atom: shows id, type, position and properties; Shift+click picks several; Measure distance: click two atoms; Measure angle: click three atoms (vertex in the middle); Edit bonds: click two atoms to add or delete the bond between them

<!-- gui:end -->
