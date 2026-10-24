# Rendering figures

`pipe.render(filename, frame=..., **options)` renders one frame offscreen to PNG, JPG, TIFF,
SVG, PDF or EPS; without a file name it returns the image as an RGB array. `ewald.render.render(data, ...)`
does the same for a `DataCollection`.

```python
pipe.render("fig.png", frame=10, size=(2000, 1500), direction="iso")
pipe.render("fig_rt.png", frame=10, renderer="raytrace", samples=16, ao_samples=64, shadows=True)
pipe.render("slab.png", direction="y", bonds=2.8, radius_scale=0.6, background="white",
            type_colors={"Cu": "#c8794a"}, text="300 K")
pipe.render("fig.svg", frame=10, renderer="vector", type_legend=True, scale_bar=True)
```

## The view of a pipeline

`pipe.view` holds how the pipeline's structure is drawn, and `pipe.render()`, `snapshot(pipe)`,
the picture Jupyter shows and the GUI viewport all use it. Set it with `view(pipe, ...)` (the
notebook's view cell) or one attribute at a time; keywords given to `pipe.render(...)` override it
for that call only (`bonds=2.8` replaces the bond table, `direction=` the saved camera).

```python
from ewald import load, view

pipe = load("POSCAR")
view(pipe, theme="VESTA", colors={"O": "red"}, cell=False, direction="z", zoom=1.2)
pipe.view.radius("Ti", 0.6)                   # per-type radius in Angstrom
pipe.view.look("x").rotate(azimuth=30, elevation=15)
pipe.render("fig.png", renderer="raytrace")
print(pipe.view.changed())                    # what differs from the defaults
print(pipe.view.source())                     # the same as a view(pipe, ...) line
```

| Group | Settings |
|---|---|
| style | `style` (Ball & stick, Space-filling, Simple (ASE), Points), `theme`, `atom_style` (auto, mesh, impostor, flat), `background`, `light` (key light) |
| atoms | `atoms` (False: a stick model), `color_by`, `colormap`, `color_range` (what colors the atoms, below), `radius_scale`, `colors` {type: color}, `radii` {type: Angstrom}, `transparency` {type: 0..1}, `selection` (True: selected atoms red, False, or a color), `labels` (atom labels), `radius_preset`, `oxidation_states` |
| bonds | `bonds` (True, False or a cutoff), `bond_pairs` [[A, B, dmin, dmax]], `bond_factor`, `bond_radius` |
| cell | `cell`, `cell_width`, `cell_color`, `cell_radius` (ray-traced tube, Angstrom) |
| boundary | `boundary` (VESTA range), `boundary_bonds` (half, complete, none), `replicate` [na, nb, nc], `replicate_cell` (super, unit, all) |
| elements | `surfaces` (polyhedra, isosurface and slice look by name), `visuals` (plugin visual elements), `vectors` (arrows of a vector property), `measurements` (distances, angles, dihedrals), `trajectory_width`, `trajectory_color_by`, `trajectory_colormap`, `trajectory_color_range`, `trajectory_time_range`, `trajectory_trail`, `trajectory_radius` (trajectory lines, below) |
| camera | `frame`, `direction`, `zoom`, `orthographic`, `camera` (an exact camera, as the GUI saves it) |
| picture | `renderer`, `quality`, `size`, `ambient_occlusion`, `shadows`, `transparent_background`, `text`, `legend`, `type_legend`, `scale_bar`, `overlays` |

`View.help()` prints every setting with its default. A style preset sets `radius_scale`, `bonds`
and `atom_style` unless you give them too, and Simple (ASE) also the ASE theme. The defaults are
the GUI's: Ball & stick with VESTA-style bonds (off above 250,000 atoms). Per-atom colors, sizes
and hiding by condition are steps (`paint`, `hide`, `fade`, `resize` in
[Notebooks](notebooks.md)), and polyhedra, isosurfaces and arrows come from modifiers.

### Visual elements by name

The same settings are also reachable per visual element, under the element's own names:
`Particles`, `Bonds`, `Surfaces`, `Vectors`, `Trajectory lines`, `Simulation cell` and every
plugin visual in `visuals`. `pipe.view.elements` lists them with their current values, and
setting an attribute sets the view setting behind it, so the view cell and the GUI follow.

```python
pipe.view.elements                                   # Particles(enabled=True, ...) / Bonds(...) / ...
pipe.view.elements["Bonds"].radius = 0.12            # = pipe.view.bond_radius = 0.12
pipe.view.elements["Particles"].enabled = False      # = pipe.view.atoms = False (a stick model)
pipe.view.elements["Simulation cell"].set(color="grey", width=1.5)
pipe.view.elements["Bonds"].settings                 # {'enabled': 'bonds', 'radius': 'bond_radius', ...}
```

### Trajectory lines with a color map

A `GenerateTrajectoryLines` step traces the selected particles (all of them when nothing is
selected) over its frame range; the condition of a selection step is evaluated at the range's
first frame. The view then colors the lines point by point with a color map and draws its color
bar in the viewport and in every picture (OpenGL, ray tracer, SVG/PDF):

```python
pipe.add(ExpressionSelection("ParticleType == 'Li'"))          # or "ParticleIdentifier == 17", "Position.Z > 8"
pipe.add(GenerateTrajectoryLines(start=200, stop=600))          # the frames traced
view(pipe, trajectory_color_by="Time",                          # 'Time' (timestep), 'Frame', 'Step Length',
     trajectory_colormap="viridis",                             #   a table column or a particle property
     trajectory_time_range=[20000, 40000],                      # draw only this Time window
     trajectory_color_range=[20000, 40000])                     # ends of the map (None: what is drawn)
pipe.view.elements["Trajectory lines"].set(trail=5000)          # follow the frame: the last 5000 steps
pipe.view.elements["Trajectory lines"].trail = True             # ... or everything up to the frame
```

The `trajectory_*` settings only style the lines a `GenerateTrajectoryLines` step made; without
the step there is nothing to draw, and `view(...)` (and the AI's `set_view`) warns about it. A NEB
band works the same way: its images are frames, so after `load("neb/")` add
`GenerateTrajectoryLines()` and `trajectory_trail=True` grows the paths image by image (the
`OverlayFrames` copies of the other images are skipped).

`trajectory_color_by="Step Length"` shows how far a particle moved per traced step (hops stand
out); a particle property such as `"Path Length"` or `"Net Displacement"` gives each line one
color. `trajectory_radius` sets the tube radius of ray-traced lines (default 0.04 x the width).
In the GUI the View tab's Elements card has the same settings (Trajectories, Color map, Time
window, Trail). Several color bars (a `ColorCoding` step and the trajectory map) stand side by side.
In the viewport a click on a color bar card folds the cards to one-line chips (a short ramp and the
title) and a click on a chip unfolds them; the view cell records it as `colorbar_collapsed=True`,
which code can set too. Pictures always draw the whole bars (`legend=False` leaves them out).

### What colors the atoms

Atoms take their particle type colors (`colors`, the theme) unless a step wrote a `Color`
property: structure identification (`CommonNeighborAnalysis`, `PolyhedralTemplateMatching`,
`IdentifyDiamond`, `AcklandJones`, ... with `color_by_type=True`, the default) gives each structure
type OVITO's color, `ColorCoding` maps a property through a color map, `paint` and `AssignColor`
color where a condition holds, and `ClusterAnalysis`/`GrainSegmentation` give each cluster or
grain a color. Each of them records its rule, which `ewald.color_rule` reads back:

```python
from ewald import color_rule
print(color_rule.describe(pipe.compute()))
# Atoms are colored by 'Structure Type' (from the IdentifyDiamond step (color_by_type=True)):
# 0 OTHER = #f2f2f2 (88213 atoms); 1 CUBIC_DIAMOND = #13a0fe (5117 atoms); ...
```

Pictures draw it with `legend=True` (the default): category swatches for structure types and
painted conditions, a color bar for a color map. The view's `color_by` recolors the atoms for
display only, whatever the steps wrote:

```python
view(pipe, color_by="type")                                   # the type colors
view(pipe, color_by="Coordination", colormap="plasma", color_range=[2, 5])   # a property or expression
view(pipe, color_by=None)                                     # back to the steps' colors
```

In the GUI the View tab's **Colors** card has the same settings (Color by, Color map, Range) and
lists the rule that is drawn, one row per category with its count; the AI chat gets the same
description with each question and from its `describe_frame`/`get_view` tools, so it can say what
each color means.

`transparency={"O": 0.6}` fades whole particle types with their half-bonds (1 hides the type);
a fade or hide step wins where its value is larger. Particles a selection step selected (a
`Selection` property) are drawn red, as in OVITO; `selection="orange"` picks another color and
`selection=False` turns the highlight off. Coloring steps (`AssignColor`) clear the selection,
so their atoms keep the color they set.

`labels` puts a label on each atom, in the GUI viewport and in every picture (OpenGL, ray
tracer, SVG/PDF). `True` shows the type; a template uses `{type}`, `{index}` (0-based), `{id}`
(the Particle Identifier) and per-atom properties without spaces; a dict also limits the types
and sets the font size and color. At most 2,000 atoms get a label (the nearest to the camera in
pictures). Boundary and replicated images are not labelled.

```python
view(pipe, labels="{type}{index}")                                # Sr0, Ti1, O2, ...
view(pipe, labels={"text": "{Charge:+.2f}", "types": ["O"], "size": 14, "color": "navy"})
pipe.render("fig.png", labels="id")                               # one picture only
```

`measurements` draws distances (2 atoms), angles (3, at the middle atom) and dihedrals (4) with
their values, in the viewport and every picture, using the nearest periodic image of each next
atom. `pipe.view.measure(...)` adds one and returns its value; `ewald.measure.measure(data,
[i, j])` only computes it.

```python
pipe.view.measure(0, 5)                  # 2.761 (Angstrom), and drawn from now on
pipe.view.measure(4, 0, 7)               # 90.0 (degrees, at atom 0)
pipe.view.measured()                     # [{'atoms': [0, 5], 'kind': 'distance', 'value': 2.761}, ...]
view(pipe, measurements=[[0, 5], [4, 0, 7]])
```

`light` sets the key light for the viewport and every renderer (OpenGL, ray tracer, SVG/PDF
highlights): a direction `[x, y, z]` relative to the viewer (x right, y up, z towards you) or a
dict with `azimuth` and `elevation` in degrees (azimuth turns it right, elevation up) and
`intensity` (1 is the default strength). Left at None, each renderer uses its default light
from the upper left.

```python
view(pipe, light={"azimuth": 60, "elevation": 20})                # from the right
view(pipe, light={"elevation": -45, "intensity": 0.6})            # dim, from below
pipe.render("fig.png", renderer="raytrace", light=[0, 0, 1])     # head-on, one picture only
```

Polyhedra, isosurfaces and grid slices are `data.surfaces` meshes made by a step
(`CoordinationPolyhedra` makes `coordination-polyhedra`, `CreateIsosurface` its `identifier` and
`identifier + "-"`). The view restyles them by that name without editing the step:

```python
pipe.view.surface("coordination-polyhedra", color="teal", transparency=0.5, edges=False)
pipe.view.surface("isosurface-", visible=False)      # hide the negative lobe
pipe.view.surface("coordination-polyhedra")          # back to the step's look
```

## Renderers

`renderer="opengl"` (default) uses VTK and is fast; `ambient_occlusion=True` adds VTK's
screen-space occlusion. `renderer="raytrace"` is ewald's built-in ray tracer, the renderer
for publication figures: spheres and bond cylinders with ambient occlusion (`ao_samples` rays
per hit), soft shadows (`shadows`) and anti-aliasing (`samples` rays per pixel). It is
written in numba and runs on all CPU cores; colors, radii, bonds and the camera match the
OpenGL view. The AO rays of all samples of a pixel follow a low-discrepancy sequence, so
16 samples x 8 AO rays (128 per pixel) are as smooth as 512 random rays. The simulation cell
(and any other line drawn as a tube) casts no shadow and no ambient occlusion, so its edges
never leave dark arcs on the atoms.

Atoms, bonds, arrows and polyhedra are found through a uniform grid; large, fine surface
meshes (isosurfaces, grid slices) through a bounding volume hierarchy, which traces a
270k-triangle isosurface about twice as fast as the grid did. With `noise_threshold` above 0
(adaptive sampling) every pixel first takes `max(8, samples // 4)` rays, and only pixels whose
estimated error is still above the threshold, and their neighbours, go on to `samples`: the
background and evenly lit surfaces stop early, while edges, shadows and occlusion keep the full
count. `noise_threshold=0` gives every pixel all `samples` rays.

`device="cuda"` runs the ray tracer on an NVIDIA GPU through CuPy (`pip install 'ewald[gpu]'`),
`device="opencl"` on any GPU with an OpenCL driver, Intel Arc and integrated graphics, AMD,
NVIDIA and Apple, through pyopencl (`pip install 'ewald[opencl]'`). `"auto"` takes CUDA when
there is an NVIDIA GPU, else an OpenCL GPU, else the CPU; `None`/`"cpu"` (default) the CPU. Both
GPU kernels are one C source (`ewald.raytrace_gpu`), the same algorithm as the CPU tracer, one
thread per pixel, with the same random rays, in single precision: their pictures differ from
the CPU's only at a few edge pixels (in the tests 0.7 % of pixels by more than 1/255). A GPU
without double precision, such as Intel Arc, also keeps the per-pixel sums in single precision,
which changes nothing visible. The first OpenCL picture of a session waits a few seconds while
the driver compiles the kernel. `pipe.render(...)` follows `pipe.device`, and the GUI's
Render page and movies follow Preferences › Compute › Device ([GPU compute](gpu.md)).

`renderer="vector"` writes SVG, PDF or EPS (format from the extension) with real circles for
atoms, bond sticks, polyhedra, isosurfaces and slices painted far to near, and overlays as
editable text, so a figure can be finished in Illustrator or Inkscape. It needs no OpenGL.
The raster renderers can also be given a `.pdf`, `.svg` or `.eps` name; the image is then
embedded in a page of the same size.

## Quality

File output puts quality first. `quality=` picks a preset for every sampling setting you do
not pass yourself:

| `quality` | Ray tracer | OpenGL | Size |
|---|---|---|---|
| `"draft"` | 1 ray/pixel, no AO, no shadows | no SSAA/SSAO, coarse spheres | half |
| `"standard"` | 4 rays/pixel, 4 AO rays per hit, shadows | SSAA, medium spheres | full |
| `"publication"` (default) | up to 16 rays/pixel (adaptive, `noise_threshold=0.01`), 8 AO rays per hit, shadows | SSAA + SSAO, fine spheres | full |

Explicit `samples=`, `ao_samples=`, `noise_threshold=`, `shadows=`, `antialiasing=`, `ambient_occlusion=` and
`scale=` (a factor on `size`) always win. The interactive viewport keeps its own speed settings.

## Overlays

Every renderer draws the same overlays, sized relative to the image:

- `text="t = [Time] ps"`: a label; `[Frame]`, `[Timestep]`, `[Time]` and any attribute name in
  brackets are filled in (in animations, per frame);
- `legend=True`: the atoms' color legend (category swatches of structure types and painted
  conditions, or the color bar of `ColorCoding`/`color_by`), the color bars of grid slices and of
  color-mapped trajectory lines, each on a small translucent card in the top right corner, side by
  side. Ticks are round numbers (whole ones for frames and timesteps). The 3D view draws the same
  cards, so what you see is what the picture gets;
- `type_legend=True`: one color swatch per particle type;
- `scale_bar=True`: a round length (Å, or nm from 10 Å); `scale_bar=20` for 20 Å. Exact in
  orthographic views, at the focal plane in perspective.

`dpi=600` is stored in PNG/TIFF files and sets the page size of vector output.

## Options

| Option | Meaning |
|---|---|
| `size=(w, h)` | pixels |
| `direction` | view vector from the scene to the camera, or `"x"`, `"-y"`, `"z"`, `"iso"`, ... |
| `orthographic`, `zoom` | projection and zoom factor |
| `camera` | an exact camera from the GUI (`ewald.render.camera_state`); the Render workspace fills it in |
| `region=(x, y, height)` | only that part of the view: the Render frame's center offset from the view center and its height, as fractions of the view height (`ewald.framing`); the width follows `size`. Zooming the view and moving the frame both work on it |
| `bonds=2.8` | draw bonds shorter than 2.8 Å; `bond_pairs=[("Si", "O", 0, 2.0)]` sets VESTA-style pair rules |
| `radius_scale`, `type_colors`, `type_radii` | atom size and per-type overrides |
| `radius_preset`, `oxidation_states`, `coordination` | which tabulated radii to use (see below) |
| `background`, `transparent_background` | `"white"`, any color, or a transparent PNG |
| `renderer`, `quality` | `"opengl"`, `"raytrace"` or `"vector"`; `"draft"`, `"standard"` or `"publication"` |
| `legend`, `type_legend`, `scale_bar`, `text` | overlays (see above) |
| `overlays` | overlay elements drawn on the image: `"Coordinate tripod"` (axis arrows that turn with the camera), `"Attributes"` (a temperature/energy readout), [plugins](../plugins/overlays.md); names, instances or `{"name": ..., "params": {...}}` |
| `theme` | `"VESTA"`, `"CPK"`, `"Journal"`, `"ASE"`, `"Dark slide"` or a [theme plugin](../plugins/styles.md#themes): element colors, background, cell color and radius preset; explicit arguments win |
| `show_cell`, `dpi` | cell outline; resolution stored in the file |
| `boundary`, `boundary_bonds` | VESTA's boundary: `boundary=(-0.1, 1.1)` (fractional; or three `(lo, hi)` pairs) also draws the periodic images inside the range and completes bonds and polyhedra cut by the cell (see [below](#boundary-vesta)) |
| `visuals` | extra [visual elements](../plugins/visuals.md) drawn on top in every renderer: registered names, instances, or `{"name": ..., "params": {...}}` |
| `vectors` | particle properties drawn as arrows: `"Force"`, `["Force", "Velocity"]` or `{"Force": {"scaling": 2, "color": "red"}}` (options of [`VectorVis`](modifiers.md#vector-arrows)); added to, and overriding, the arrows `VectorVis` set |

Particle colors come from the `Color` property when a modifier set one (`ColorCoding`,
`AssignColor`, structure analyses with `color_by_type`), otherwise from the element.

## Atom radii

Atoms are drawn at 0.8 × a tabulated radius × `radius_scale`, unless a `Radius` property or
`type_radii` sets them. `radius_preset` picks the table:

| Preset | Source |
|---|---|
| `"covalent"` (default) | Cordero et al., Dalton Trans. 2832 (2008) |
| `"atomic"` | Slater, J. Chem. Phys. 41, 3199 (1964), empirical |
| `"atomic (calc.)"` | Clementi, Raimondi, Reinhardt, J. Chem. Phys. 47, 1300 (1967), calculated |
| `"vdW"` | Alvarez, Dalton Trans. 42, 8617 (2013) |
| `"vdW (Bondi)"` | Bondi, J. Phys. Chem. 68, 441 (1964), completed from WebElements |
| `"ionic"` | Shannon, Acta Cryst. A32, 751 (1976), effective ionic radii |
| `"ionic (crystal)"` | Shannon (1976) crystal radii |

```python
pipe.render("sto.png", radius_preset="ionic", radius_scale=1.25)       # Shannon radii at full size
pipe.render("sto_ti3.png", radius_preset="ionic", oxidation_states={"Ti": 3}, coordination={"Ti": 6})
```

Shannon radii depend on the oxidation state and the coordination number. For each particle
type the oxidation state comes from `oxidation_states`, else from the type name (`"Fe3+"`,
`"O2-"`), an integer `Oxidation State` or `Charge` property, a pymatgen bond-valence analysis
when pymatgen is installed (cells up to 500 atoms), a charge-neutral combination of the
oxidation states most common in the ICSD, or finally the most common state alone. The
coordination number is counted from the structure: counter-ions closer than 1.25 × the median
nearest cation–anion distance of that pair of types, so O in SrTiO3 gets 2 Ti + 4 Sr = 6. The
nearest coordination number in Shannon's table is used. Pure elements and ions missing from
the table get the covalent radius and a logged warning that names them.
`ewald.radii.resolve(data, "ionic").summary()` prints what was chosen:

```text
ionic radii: Sr 1.440 (+2, CN 12, charge balance), Ti 0.605 (+4, CN 6, charge balance), O 1.400 (-2, CN 6, charge balance)
```

The Slater, Clementi and Shannon tables ship with ewald in `ewald/tables/radii.json`
(extracted from pymatgen, MIT license); covalent and van der Waals radii come from ASE.

## Large systems

Above 150 000 atoms the OpenGL renderer draws GPU impostors (shaded point sprites with
per-pixel sphere depth) instead of sphere meshes, which keeps million-atom frames interactive;
`atom_style="impostor"` uses them at any size. Bonds and other cylinders follow: with
impostor atoms, or with more cylinders than `mesh_limit`, each is one point ray-cast on the GPU
(VTK's stick mapper, flat ends hidden inside the atoms) instead of a 12-sided tube; the flat
(ASE) style keeps outlined tubes. The GUI viewport draws impostors by default. Pass `mesh_limit=` to move the
threshold; above it, vector arrows are drawn as colored lines. `arrow_resolution=` sets the
sides per arrow (the viewport uses Preferences › 3D view › Arrow resolution while you
interact, file output follows the quality preset).

## Simple (ASE-style) drawing

`atom_style="flat"` draws atoms the way ASE's GUI does: unlit discs in the element color with a
dark outline, and unshaded bonds with dark edges. The Simple (ASE) style
(`view(pipe, style="Simple (ASE)")`) is that plus `theme="ASE"` (Jmol colors and covalent radii on
white), `radius_scale=0.7` and bonds on, so the sticks show between the discs:

```python
render(data, "simple.png", atom_style="flat", theme="ASE", radius_scale=0.7, bonds=2.6)
```

It uses impostors, so it is as fast as `"impostor"` at any size, and screen-space ambient
occlusion is off unless you pass `ambient_occlusion=True`. The vector renderer draws it with
`shading="flat"` (one color plus outline); the ray tracer draws the same atoms and bonds with its
own shading.

## Boundary (VESTA)

A crystal drawn with only its own atoms looks cut open: the atom at the origin is not repeated on
the other corners, and bonds and polyhedra stop at the cell faces. `boundary` draws it the way
VESTA does. Every periodic image whose fractional coordinates fall inside the range is drawn as
well, bonds between drawn atoms become whole, and coordination polyhedra are repeated on the
images of their centers, with the corner atoms added so each polyhedron is complete:

```python
pipe = Pipeline(src, [CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.2)]), CoordinationPolyhedra(["Ti"])])
pipe.render("srtio3.png", boundary=(0, 1))                  # Ti on all eight corners, eight octahedra
pipe.render("srtio3_more.png", boundary=[(-0.1, 1.1), (-0.1, 1.1), (0, 1)], boundary_bonds="complete")
```

`boundary_bonds` decides what a bond that leaves the range looks like: `"half"` (default) draws
the half at the drawn atom, as without a boundary; `"complete"` adds the partner atom, VESTA's
"search additional atoms"; `"none"` leaves it out. Only periodic directions get images.

It is display only. `pipe.compute()` and every modifier still see the real atoms, the images are
made by the renderers (OpenGL, ray tracer and vector alike), and an image picks, hovers and
selects as its real atom in the GUI. In the GUI it is the Boundary box of the Cell card in
the View tab; the range is saved with the session.

## Replicated cell

`replicate=(na, nb, nc)` draws that many periodic copies of the cell side by side, the picture
OVITO makes with a Replicate modifier, but as a display setting: bonds join across the copies,
coordination polyhedra and isosurfaces are repeated, and the data, every analysis and every export
keep the original cell (the `Replicate` modifier is the step that really enlarges the data).

```python
view(pipe, replicate=(3, 3, 1))                          # 3 x 3 x 1 cells, every picture and the GUI
pipe.render("srtio3_3x3.png", replicate=2, replicate_cell="all")   # one picture, every cell outlined
view(pipe, replicate=None)                               # back to one cell
```

`replicate_cell` picks the cell lines: `"super"` (default) outlines the whole block, `"unit"` keeps
only the original cell, `"all"` draws every copy as a lattice grid. The boundary range applies to
the replicated block, so `replicate=(2, 2, 2), boundary=(0, 1)` adds VESTA's face, edge and corner
images around the 2 x 2 x 2 block. Copies pick, hover and select as their real atom in the GUI,
where the controls are the Replicate cell box of the Cell card in the View tab,
View › Replicate cell (also in Ctrl+K) and the 3D view's right-click menu; they write
`view(pipe, replicate=...)` into the notebook. VESTA has no single switch for this: it draws a
boundary range over several cells, which is `boundary=(0, n)` here.

## Animations

```python
from ewald.render import render_animation

render_animation(pipe, "run.mp4", frames=range(0, pipe.num_frames, 5), fps=20, size=(1280, 720),
                 text="t = [Time] ps", time_per_frame=0.5, quality="standard")
```

The extension picks the format: `.gif`, `.mp4` (H.264), `.avi` (Motion JPEG, plays in
PowerPoint), `.mov`, `.webm`, or `.png`/`.jpg`/`.tif` for a numbered image sequence
(`frames.png` writes `frames_0000.png`, ...; `frame_%03d.png` also works). GIF and image
sequences need nothing extra; video formats use the ffmpeg binary that
imageio-ffmpeg (installed with ewald) brings, or one on your PATH. `progress=lambda i, n: ...` is called after
each frame and stops early when it returns `False`. From the shell:

```bash
ewald movie dump.lammpstrj run.gif --every 5 --fps 12 --text "t = [Time] ps" --time-per-frame 0.5
```

## Sessions

A session file (`.ewald`, JSON) stores the input file, the modifiers, the visual style, the
camera and the Render settings, like OVITO's `.ovito` files:

```python
from ewald.session import load_session, save_session

save_session("figure.ewald", pipe, camera=cam, render={"renderer": "raytrace", "size": (2100, 1575)})
s = load_session("figure.ewald")
s.pipeline.render("figure.png", camera=s.camera, **s.render_kwargs())
```

The input is found by its absolute path or relative to the session file, so a folder holding
both can be moved. Python modifiers are saved as source and run only with
`load_session(path, allow_code=True)`; otherwise they load disabled, and so does a plugin
modifier that reports `untrusted_problems()`, such as the ewald-plumed plugin's `Plumed` whose input
uses `LOAD`, a `kernel` library or file paths outside its folder. Entries
that name anything but a modifier class are skipped. `ewald render figure.ewald
figure.pdf` re-renders a saved session from the shell.

## From the GUI

The Render workspace sets the same options with journal size presets, a Quality menu
(Draft / Standard / Publication / Custom) and a separate movie quality, overlay switches and
a movie format menu, and shows "Python for this figure", the `render(...)` or
`render_animation(...)` call that reproduces the output. Paste it into a script to render
the same view for every frame or every sample. File › Export › Session (.ewald) writes a
`.ewald` file; opening one restores the whole figure. Defaults live in Preferences ›
Rendering.
