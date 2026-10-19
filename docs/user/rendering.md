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
| `background`, `transparent_background` | `"white"`, any color, or a transparent PNG |
| `legend`, `show_cell`, `text` | color legend for `ColorCoding`, cell outline, overlay text |

Particle colors come from the `Color` property when a modifier set one (`ColorCoding`,
`AssignColor`, structure analyses with `color_by_type`), otherwise from the element.

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
