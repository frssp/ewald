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
