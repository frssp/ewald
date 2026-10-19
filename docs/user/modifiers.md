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

## Python

`PythonScript(source)` runs `modify(frame, data)` from code kept as text, which the GUI edits
in place and which can be saved with a session. Any plain function `f(frame, data)` appended
to `pipe.modifiers` becomes a `PythonModifier`. To write a reusable modifier class, see
[Modifier plugins](../plugins/modifiers.md).
