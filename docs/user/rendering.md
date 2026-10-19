# Rendering figures

`pipe.render(filename, frame=..., **options)` renders one frame offscreen to PNG, JPG or
TIFF; without a file name it returns the image as an RGB array. `ewald.render.render(data, ...)`
does the same for a `DataCollection`.

```python
pipe.render("fig.png", frame=10, size=(2000, 1500), direction="iso")
pipe.render("fig_rt.png", frame=10, renderer="raytrace", samples=16, ao_samples=64, shadows=True)
pipe.render("slab.png", direction="y", bonds=2.8, radius_scale=0.6, background="white",
            type_colors={"Cu": "#c8794a"}, text="300 K")
```

## Renderers

`renderer="opengl"` (default) uses VTK and is fast; `ambient_occlusion=True` adds VTK's
screen-space occlusion. `renderer="raytrace"` is ewald's built-in ray tracer, the renderer
for publication figures: spheres and bond cylinders with ambient occlusion (`ao_samples` rays
per hit), soft shadows (`shadows`) and anti-aliasing (`samples` rays per pixel). It is
written in numba and runs on all CPU cores; colors, radii, bonds and the camera match the
OpenGL view.

## Options

| Option | Meaning |
|---|---|
| `size=(w, h)` | pixels |
| `direction` | view vector from the scene to the camera, or `"x"`, `"-y"`, `"z"`, `"iso"`, ... |
| `orthographic`, `zoom` | projection and zoom factor |
| `camera` | an exact camera from the GUI (`ewald.render.camera_state`); the Render workspace fills it in |
| `bonds=2.8` | draw bonds shorter than 2.8 Å; `bond_pairs=[("Si", "O", 0, 2.0)]` sets VESTA-style pair rules |
| `radius_scale`, `type_colors`, `type_radii` | atom size and per-type overrides |
| `radius_preset`, `oxidation_states`, `coordination` | which tabulated radii to use (see below) |
| `background`, `transparent_background` | `"white"`, any color, or a transparent PNG |
| `legend`, `show_cell`, `text` | color legend for `ColorCoding`, cell outline, overlay text |

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

Above 150 000 atoms the OpenGL renderer draws GPU impostors (shaded point sprites) instead of
sphere meshes, which keeps million-atom frames interactive. Pass `mesh_limit=` to move the
threshold.

## Animations

```python
from ewald.render import render_animation

render_animation(pipe, "run.mp4", frames=range(0, pipe.num_frames, 5), fps=20, size=(1280, 720))
```

needs the `movie` extra (imageio with ffmpeg); a `.gif` name writes a GIF.

## From the GUI

The Render workspace sets the same options with journal size presets and shows "Python for
this figure", the `pipe.render(...)` call that reproduces the image. Paste it into a script to
render the same view for every frame or every sample.
