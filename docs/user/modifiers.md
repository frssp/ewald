# Modifiers

A modifier takes the data of one frame and adds or changes properties, attributes or tables,
or returns a smaller or larger set of particles. Modifiers run in the order they were
appended. Their names, output property names, structure type ids and colors follow OVITO, and
OVITO's class names also work: `CommonNeighborAnalysisModifier` resolves to
`CommonNeighborAnalysis`, `WrapPeriodicImagesModifier` to `WrapAtoms`.

```python
from ewald.modifiers import CommonNeighborAnalysis, ExpressionSelection, DeleteSelected

pipe.modifiers.append(CommonNeighborAnalysis())
pipe.modifiers.append(ExpressionSelection("StructureType == 1"))   # select FCC atoms
pipe.modifiers.append(DeleteSelected())
```

Every constructor argument is listed with its default in the
[API reference](../reference/api.md) and in `llms.txt`; `help(CommonNeighborAnalysis)` shows
the full docstring.

## Expressions

`ExpressionSelection`, `ComputeProperty` and `ColorCoding` take OVITO-style expressions,
evaluated with NumPy. Property names lose their spaces (`ParticleType`,
`ParticleIdentifier`), vector components get a suffix (`Position.Z`, `Velocity.X`,
`Color.R`), and the globals `N`, `Frame`, `CellSize.X`, `CellOrigin.X` and every scalar
attribute are available. Combine conditions with `&`, `|`, `~` and parentheses, not
`and`/`or`/`not`:

```python
ExpressionSelection('(Position.Z > 10) & (ParticleType == "Cu")')
ComputeProperty("Speed", "sqrt(Velocity.X**2 + Velocity.Y**2 + Velocity.Z**2)")
ColorCoding("Position.Z - CellOrigin.Z", cmap="viridis")
```

## Selection and deletion

`ExpressionSelection(expression)` selects particles where the expression is true.
`SelectType(types)` selects by type (or another property). `InvertSelection()` and
`ClearSelection()` do what they say. `DeleteSelected()` removes the selected particles.
`Slice(normal, distance, width, inverse, select, miller)` keeps, deletes or selects a slab;
`miller=True` reads `normal` as Miller indices of the cell.

## Coloring

`ColorCoding(property, cmap, start, end)` maps a property or expression to a matplotlib
colormap and stores the range, so the renderer can draw a legend. `AssignColor(color)` colors
the selected particles.

## Vector arrows

`VectorVis(property, scaling, width, alignment, ...)` draws a per-particle vector property as
arrows: "Force", "Displacement", "Velocity", "Magnetic Moment" or any N x 3 property (an empty
`property` takes the first of these that exists). A scalar property, such as a collinear
magnetic moment, points along z, so spin up and down read as up and down arrows. The options
follow OVITO's vector visual element:

| Option | Meaning |
|---|---|
| `scaling` | Å per unit of the property |
| `width` | shaft diameter in Å; the head is 2.5× as wide and 4.5× the shaft radius long, shrunk for short arrows |
| `alignment` | `"base"` (arrow starts at the atom), `"center"` or `"head"` (arrow ends at the atom) |
| `reverse`, `offset` | flip the arrows; shift them by a vector in Å |
| `color` | one color for every arrow |
| `color_mapping_property`, `cmap`, `color_mapping_interval` | color by `"Magnitude"` (the arrow's length), another property or an expression; the range is shown as the color legend unless a `ColorCoding` already owns it |
| `shading`, `transparency` | `"normal"` (lit 3D) or `"flat"` (solid color); 0 to 1 |
| `only_selected`, `min_length` | arrows of selected particles only; skip arrows shorter than this (Å) |

The modifier stores a style in `data.vectors[property]`; arrows are built from the final
particles when the frame is drawn, so `Slice`, `DeleteSelected` or `Replicate` after it keep
every arrow on its atom. Use one `VectorVis` per property. All three renderers draw them, and
the vector renderer splits each arrow where it passes its atom, so arrows run through atoms in
SVG/PDF output too.

```python
pipe.modifiers.append(VectorVis("Force", scaling=2.0, color_mapping_property="Magnitude"))
pipe.modifiers.append(VectorVis("Magnetic Moment", scaling=0.5, alignment="center", color="crimson"))
```

## Phonon modes

`PhononMode(qpoint, band, amplitude, supercell, arrows, ...)` animates one phonon mode of a
phonopy `band.yaml`/`mesh.yaml`/`qpoints.yaml` or a VASP IBRION=5-8 OUTCAR over one period (the
file's frames), in a commensurate supercell, with the mode pattern as arrows. See
[Phonon modes](phonons.md).

```python
pipe = import_file("band.yaml")
pipe.modifiers.append(PhononMode(qpoint=50, band=3, amplitude=0.3))
```

## Transformations

`AffineTransformation(matrix)` applies a 3×3 or 3×4 matrix to particles and optionally the
cell. `Replicate(n)` tiles the periodic cell. `WrapAtoms()` maps particles back into the cell.
`ComputeProperty(output, expressions)` writes an expression into a new or existing property.

## Neighbors, clusters and displacements

`CoordinationAnalysis(cutoff, number_of_bins, partial)` gives the coordination number of
every particle and the radial distribution function as `tables["coordination-rdf"]`, with
partial g(r) per type pair when `partial=True`. `ClusterAnalysis(cutoff, ...)` groups
particles connected by distances below the cutoff, with size, center of mass, gyration tensor
and unwrapped coordinates as OVITO computes them. `CalculateDisplacements(reference_frame)`
gives each particle's displacement from a reference frame, matched by id.

## Bonds and coordination polyhedra

`CreateBonds(cutoff, mode, pairs)` finds bonds with a uniform cutoff, a VESTA-style pair table
(`mode="pairwise", pairs=[("Ti", "O", 2.4)]`, `*` matches any type) or covalent radii
(`mode="covalent"`, VESTA's default). Bonds are stored once each in `data.bonds`, with periodic
images, so they stay correct across boundaries and through `DeleteSelected`, `Slice`,
`Replicate` and `WrapAtoms`. `EditBonds(add=[(a, b)], remove=[...])` adds or deletes single
bonds by particle identifier; the GUI's bond tool does the same by clicking two atoms.
`CoordinationPolyhedra(center_types)` puts a polyhedron around each center particle, colored
like the center, into `data.surfaces["coordination-polyhedra"]`; it needs bonds upstream. LAMMPS
data files with a Bonds section fill `data.bonds` directly.

```python
pipe.modifiers.append(CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.4)]))
pipe.modifiers.append(CoordinationPolyhedra(center_types=["Ti"]))
```

## Histograms, binning and reductions

These match OVITO 3.16's Histogram, Spatial Binning and Reduce Property modifiers value for
value. A property is named as in expressions' dotted form: `"Mass"`, `"Position.Z"`,
`"Color.R"`, or a bond property such as `"Topology.A"` with `operate_on="bonds"`.

- `Histogram(property, bin_count, ...)`: `tables["histogram[<property>]"]`, as counts, relative
  frequency or probability density (`normalization`), over the value range or a fixed one;
  `select_elements=True` selects the elements in a value window.
- `SpatialBinning(property, reduction_operation, direction, bin_count)`: mean, sum, sum per
  volume, min or max of a property in bins along one, two or three cell vectors, for density
  and temperature profiles or 2D maps. 1D results go to `tables["binning"]` (with
  `first_derivative=True` for gradients), 2D and 3D results to `data.grids["binning"]`, where
  `GridSlice` and `CreateIsosurface` can show them. `property=""` counts particles; with
  `"sumvol"` that is the number density.
- `ReduceProperty(property, reduction_operation)`: min, max, sum, nonzero count, mean, median,
  variance, standard deviation, argmin or argmax as the attribute
  `ReduceProperty.particles.Mean.Position.Z` (one per component of a vector property).

```python
pipe.modifiers.append(SpatialBinning("Velocity.X", "mean", direction="Z", bin_count=40))
pipe.modifiers.append(SpatialBinning("", "sumvol", direction="XY", bin_count=(64, 64)))
pipe.modifiers.append(ReduceProperty("Potential Energy", "sum"))
```

## Bond networks and local order

- `BondAngleDistribution(bins, cosine_mode, partition)` and
  `BondLengthDistribution(bins, length_cutoff, partition)`: histograms of the angles between
  bonds at each atom and of bond lengths, as `tables["bond-angle-distr"]` and
  `tables["bond-length-distr"]`, identical to OVITO's. `partition` splits them by particle type
  ("Si-O-Si", the middle type at the vertex), particle selection, bond type or bond selection.
- `FindRings(minimum_ring_size, maximum_ring_size)`: shortest-path ring statistics (Franzblau)
  of the bond network, for amorphous networks, glasses and zeolites:
  `tables["ring-size-histogram"]`, the atoms of every ring in `tables["6-rings"]` and so on,
  and the per-atom 'Ring Count'. Every ring OVITO's Find Rings modifier reports is among
  them; OVITO drops some of the largest rings that differ from another ring of the same size
  only by a smaller ring.
- `SteinhardtOrderParameters(degrees, num_neighbors, cutoff, average, wl)`: bond-orientational
  order q_l and w_l per atom ('Steinhardt Q6', ...), with the neighbor-averaged form of Lechner
  and Dellago for solid/liquid separation. It runs on the pipeline device and matches freud.

The bond and ring modifiers need bonds upstream (`CreateBonds`, or a file with bonds).

```python
pipe.modifiers.append(CreateBonds(cutoff=2.85))
pipe.modifiers.append(BondAngleDistribution(partition="particle_type"))
pipe.modifiers.append(FindRings(3, 10))
pipe.modifiers.append(SteinhardtOrderParameters((4, 6), average=True))
```

## Symmetry and diffraction

- `SymmetryAnalysis(symprec, angle_tolerance, select_unique)`: space group (Hermann-Mauguin
  symbol and number), Hall symbol, point group and crystal system through spglib
  (`pip install "ewald[symmetry]"`), as `SymmetryAnalysis.*` attributes in the Attributes tab,
  and per atom the 'Wyckoff Position' ('4a'), 'Site Symmetry' and 'Equivalent Atom'.
  `select_unique=True` selects one atom per orbit (the asymmetric unit). `symprec` is in
  Angstrom; 0.01 finds the symmetry of relaxed DFT cells.
- `PowderDiffraction(radiation, wavelength, two_theta_min, two_theta_max, step, fwhm, eta,
  debye_waller, elements, experimental)`: the powder X-ray (atomic form factors,
  Lorentz-polarization factor) or neutron (coherent scattering lengths) pattern of the cell,
  identical to pymatgen's XRDCalculator and NDCalculator. `tables["diffraction"]` is the
  pattern broadened by a pseudo-Voigt (`fwhm` in degrees, Lorentzian fraction `eta`, maximum
  100) and `tables["diffraction-peaks"]` the reflections with d, h k l and multiplicity, drawn
  as sticks in the Plots tab. `wavelength` takes Angstrom or a name (`"CuKa"`, `"CuKa1"`,
  `"MoKa"`, `"CoKa"`, `"AgKa"`, ...).
- `StructureFactor(method, q_max, q_min, number_of_bins, weighting, cutoff, lorch, elements,
  experimental)`: S(q) for liquids, glasses and MD in `tables["structure-factor"]`, by the
  direct sum over the reciprocal lattice of the cell (`"direct"`, exact for the periodic
  configuration), the Fourier transform of g(r) (`"rdf"`, smooth, with the Lorch window) or the
  Debye equation over all pairs (`"debye"`, for clusters and nanoparticles). `weighting="xray"`
  or `"neutron"` gives the Faber-Ziman total S(q) to compare with total-scattering data.
  The direct and Debye results match freud.

`experimental` is a measured pattern, any two-column text file (`.xy`, `.xye`, `.dat`, `.csv`;
headers and comments are skipped). It becomes the 'Experimental' column of the table, drawn as
points under the calculated curve, and the attribute `PowderDiffraction.Rwp` (or
`StructureFactor.Rwp`) is the weighted profile R factor between the two. For numbered LAMMPS
types, `elements={"1": "Si", "2": "O"}` (or `"1=Si, 2=O"` in the GUI) names the elements. A
supercell or MD frame gives the pattern of that configuration, thermal diffuse scattering
included, and `TimeAveraging("table:diffraction")` averages it over a trajectory. The cost grows
with atoms times reflections, about a second for 7000 atoms at Cu K-alpha.

```python
from ewald.modifiers import PowderDiffraction, StructureFactor, SymmetryAnalysis, TimeAveraging

pipe = import_file("CONTCAR")
pipe.modifiers.append(SymmetryAnalysis())
pipe.modifiers.append(PowderDiffraction(wavelength="CuKa", two_theta_max=80, experimental="measured.xy"))
data = pipe.compute()
print(data.attributes["SymmetryAnalysis.space_group"], data.attributes["PowderDiffraction.Rwp"])

md = import_file("dump.melt")
md.modifiers.append(StructureFactor(q_max=12, weighting="neutron"))
md.modifiers.append(TimeAveraging("table:structure-factor"))
```

## Volumetric data

Charge densities and potentials (VASP CHGCAR/CHG/PARCHG/AECCAR, LOCPOT, ELFCAR, Gaussian cube)
load into `data.grids`. `CreateIsosurface(isolevel, property)` draws a surface at `isolevel`
(the mean of |values| when None); `signed=True` adds the negative surface in `negative_color`,
the usual view of spin densities and density differences. `GridSlice(miller, distance)` cuts
an (hkl) plane at a fractional `distance` and maps the values through `cmap`
(`scale="log"` for valence densities), with its own color legend. `GridDifference(["A/CHGCAR",
"B/CHGCAR"])` subtracts other files' grids, e.g. rho(AB) − rho(A) − rho(B). Surfaces go into
`data.surfaces` and are drawn by every renderer, including vector output.

```python
pipe = import_file("CHGCAR")
pipe.modifiers.append(CreateIsosurface(0.05, transparency=0.3))
pipe.modifiers.append(GridSlice((0, 0, 1), 0.5, scale="log"))
```

## Structure identification (ported from OVITO)

These run in the C++ extension and are tested against OVITO 3.16 results.

- `CommonNeighborAnalysis(mode)`: adaptive, interval or fixed-cutoff CNA (FCC, HCP, BCC, ICO).
- `PolyhedralTemplateMatching(...)`: FCC, HCP, BCC, ICO, SC, cubic and hexagonal diamond,
  graphene, with RMSD, orientation, elastic deformation gradient and chemical ordering.
- `IdentifyDiamond()`: cubic and hexagonal diamond and their first and second neighbors.
- `AcklandJones()`: bond-angle analysis.
- `ChillPlus(cutoff)`: hexagonal, cubic and interfacial ice and clathrate hydrate.
- `CentroSymmetry(num_neighbors, mode)`: conventional or minimum-weight-matching CSP.
- `VoronoiAnalysis(...)`: atomic volumes, coordination, Voronoi indices, radical
  tessellation with `use_radii=True`, and the face list.
- `WignerSeitzAnalysis(reference_frame)`: vacancies and interstitials, with per-type
  occupancies.
- `AtomicStrain(cutoff, reference_frame)`: von Mises and volumetric strain, deformation
  gradient, Green-Lagrange tensor, D²min, rotation and stretch.

With the optional `ovito` package, `DislocationAnalysis(...)` runs OVITO's dislocation
extraction algorithm (Burgers vectors, line lengths, polylines in `tables["dislocations"]`)
and `OvitoModifier(any_ovito_modifier)` runs any other OVITO modifier inside an ewald pipeline.

## Trajectory analysis

These modifiers read every frame of `start:stop:step` through the modifiers above them, one
frame at a time, cache the result, and show it at every frame. Changing an upstream modifier
recomputes them.

- `MeanSquaredDisplacement(time_per_frame, ...)`: FFT time-origin-averaged MSD, per type,
  unwrapped and matched by id, with the Einstein diffusion coefficient.
- `VelocityAutocorrelation(time_per_frame, ...)`: VACF from velocities or differentiated
  positions, with the Green-Kubo diffusion coefficient.
- `TimeAveraging(operate_on)`: average of a table, attribute or property over frames, for
  example the time-averaged g(r) with `"table:coordination-rdf"`.
- `TimeSeries(operate_on)`: attributes as a function of time.
- `GenerateTrajectoryLines(...)`: particle paths drawn as lines, plus path length, net and
  maximum displacement.

```python
pipe.modifiers.append(MeanSquaredDisplacement(time_per_frame=0.002))   # ps per frame
pipe.modifiers.append(CoordinationAnalysis(cutoff=3.0))
pipe.modifiers.append(TimeAveraging("table:coordination-rdf"))
data = pipe.compute(0)
data.attributes["MeanSquaredDisplacement.diffusion_coefficient"]       # Å²/ps
data.tables["coordination-rdf[average]"]
```

## Volumetric data

These work on the `VoxelGrid`s in `data.grids` that the CHGCAR, LOCPOT, ELFCAR and cube
readers produce (see the README's volumetric section).

- `CreateIsosurface(isolevel, property="Charge Density", signed=False, ...)`: VESTA-style
  isosurface, with a second surface at `-isolevel` when `signed=True`.
- `GridSlice(miller=(0, 0, 1), distance=0.5, ...)`: colour-mapped cross-section on a lattice
  plane, linear or log scale.
- `GridDifference(subtract=[...])`: subtract the grids of other files, for example the
  charge-density difference rho_AB - rho_A - rho_B.

```python
pipe = import_file("AB/CHGCAR")
pipe.modifiers.append(GridDifference(["A/CHGCAR", "B/CHGCAR"]))
pipe.modifiers.append(CreateIsosurface(0.002, signed=True))
```

## PLUMED collective variables

With the optional PLUMED library ([PLUMED collective variables](plumed.md)):

- `Plumed(input, start, stop, step, time_per_frame, temperature, ...)`: runs a `plumed.dat`
  (text or path) over the trajectory through PLUMED's MD-engine interface; every value becomes
  the attribute `Plumed.<name>`, per-atom vectors the property `Plumed <label>`, the series
  `tables["plumed"]`.
- `LoadColvar(file, align="index"|"time", stride, offset, ...)`: an existing COLVAR as
  attributes `Colvar.<field>` aligned to the frames.
- `FreeEnergySurface(hills, bins, ranges, up_to_frame=False)`: the metadynamics FES from a
  HILLS file, as `plumed sum_hills` computes it, in `tables["fes"]`.

```python
pipe.modifiers.append(Plumed("d: DISTANCE ATOMS=1,2\nphi: TORSION ATOMS=5,7,9,15", time_per_frame=0.1))
pipe.modifiers.append(FreeEnergySurface("HILLS", bins=(60, 60)))
```

## Python

`PythonScript(source)` runs `modify(frame, data)` from code kept as text, which the GUI edits
in place and which can be saved with a session. Any plain function `f(frame, data)` appended
to `pipe.modifiers` becomes a `PythonModifier`. To write a reusable modifier class, see
[Modifier plugins](../plugins/modifiers.md).
