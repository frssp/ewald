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
| `background`, `transparent_background` | `"white"`, any color, or a transparent PNG |
| `renderer`, `quality` | `"opengl"`, `"raytrace"` or `"vector"`; `"draft"`, `"standard"` or `"publication"` |
| `legend`, `type_legend`, `scale_bar`, `text` | overlays (see above) |
| `show_cell`, `dpi` | cell outline; resolution stored in the file |

Particle colors come from the `Color` property when a modifier set one (`ColorCoding`,
`AssignColor`, structure analyses with `color_by_type`), otherwise from the element.

## Large systems

Above 150 000 atoms the OpenGL renderer draws GPU impostors (shaded point sprites) instead of
sphere meshes, which keeps million-atom frames interactive. Pass `mesh_limit=` to move the
threshold.

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
