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

## Renderers

`renderer="opengl"` (default) uses VTK and is fast; `ambient_occlusion=True` adds VTK's
screen-space occlusion. `renderer="raytrace"` is ewald's built-in ray tracer, the renderer
for publication figures: spheres and bond cylinders with ambient occlusion (`ao_samples` rays
per hit), soft shadows (`shadows`) and anti-aliasing (`samples` rays per pixel). It is
written in numba and runs on all CPU cores; colors, radii, bonds and the camera match the
OpenGL view.

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
| `"standard"` | 4 rays/pixel, 8 AO rays, shadows | SSAA, medium spheres | full |
| `"publication"` (default) | 16 rays/pixel, 32 AO rays, shadows | SSAA + SSAO, fine spheres | full |

Explicit `samples=`, `ao_samples=`, `shadows=`, `antialiasing=`, `ambient_occlusion=` and
`scale=` (a factor on `size`) always win. The interactive viewport keeps its own speed settings.

## Overlays

Every renderer draws the same overlays, sized relative to the image:

- `text="t = [Time] ps"`: a label; `[Frame]`, `[Timestep]`, `[Time]` and any attribute name in
  brackets are filled in (in animations, per frame);
- `legend=True`: the color bar of a `ColorCoding` (or grid slice) modifier;
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
| `bonds=2.8` | draw bonds shorter than 2.8 Å; `bond_pairs=[("Si", "O", 0, 2.0)]` sets VESTA-style pair rules |
| `radius_scale`, `type_colors`, `type_radii` | atom size and per-type overrides |
| `radius_preset`, `oxidation_states`, `coordination` | which tabulated radii to use (see below) |
| `background`, `transparent_background` | `"white"`, any color, or a transparent PNG |
| `renderer`, `quality` | `"opengl"`, `"raytrace"` or `"vector"`; `"draft"`, `"standard"` or `"publication"` |
| `legend`, `type_legend`, `scale_bar`, `text` | overlays (see above) |
| `show_cell`, `dpi` | cell outline; resolution stored in the file |
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

Above 150 000 atoms the OpenGL renderer draws GPU impostors (shaded point sprites) instead of
sphere meshes, which keeps million-atom frames interactive. Pass `mesh_limit=` to move the
threshold; above it, vector arrows are drawn as colored lines. `arrow_resolution=` sets the
sides per arrow (the viewport uses Preferences › Viewports › Arrow resolution while you
interact, file output follows the quality preset).

## Animations

```python
from ewald.render import render_animation

render_animation(pipe, "run.mp4", frames=range(0, pipe.num_frames, 5), fps=20, size=(1280, 720),
                 text="t = [Time] ps", time_per_frame=0.5, quality="standard")
```

The extension picks the format: `.gif`, `.mp4` (H.264), `.avi` (Motion JPEG, plays in
PowerPoint), `.mov`, `.webm`, or `.png`/`.jpg`/`.tif` for a numbered image sequence
(`frames.png` writes `frames_0000.png`, ...; `frame_%03d.png` also works). GIF and image
sequences need nothing extra; video formats use ffmpeg from the `movie` extra
(`pip install 'ewald[movie]'`) or from your PATH. `progress=lambda i, n: ...` is called after
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
`load_session(path, allow_code=True)`; otherwise they load disabled. `ewald render figure.ewald
figure.pdf` re-renders a saved session from the shell.

## From the GUI

The Render workspace sets the same options with journal size presets, a Quality menu
(Draft / Standard / Publication / Custom) and a separate movie quality, overlay switches and
a movie format menu, and shows "Python for this figure", the `render(...)` or
`render_animation(...)` call that reproduces the output. Paste it into a script to render
the same view for every frame or every sample. File › Save session (Ctrl+S) writes a
`.ewald` file; opening one restores the whole figure. Defaults live in Preferences ›
Rendering.
