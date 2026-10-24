# Visual element plugins

A visual element decides *what* is drawn for a structure: atoms, bonds, cell lines, arrows, or
anything you add, such as cell axes, a halo around selected atoms, or markers for defects.
Each element turns a `DataCollection` into a few renderer-neutral primitives, and every
renderer draws every primitive kind. So an element you write appears in the GUI viewport,
in OpenGL images, in the ray tracer and in SVG/PDF output alike, without changes to any renderer.

ewald's own elements (Particles, Bonds, Surfaces, Vectors, Trajectory lines, Simulation cell)
are registered the same way, so `ewald plugins` lists them next to yours.

## A first element

```python
# <config dir>/plugins/cell_axes.py
import numpy as np

from ewald.plugin_api import VisualElement, make_arrows, visual


@visual(category="Annotations")
class CellAxes(VisualElement):
    """The cell vectors a, b, c as red, green and blue arrows, ``length`` Angstrom long."""

    name = "Cell axes"

    def __init__(self, length: float = 2.0, width: float = 0.25):
        self.length = length          # store every argument under its own name
        self.width = width

    def applies(self, data):
        return data.cell.is_valid     # nothing to draw without a cell

    def build(self, data, ctx):
        m = data.cell.matrix
        unit = m / np.linalg.norm(m, axis=1, keepdims=True)
        base = np.tile(data.cell.origin, (3, 1))
        rgb = [(0.85, 0.15, 0.15), (0.15, 0.65, 0.2), (0.15, 0.3, 0.9)]
        return make_arrows(base, base + self.length * unit, self.width, rgb)
```

Draw it from a script by name, as an instance, or in the dictionary form sessions use:

```python
from ewald.render import render

render(data, "figure.png", visuals=["Cell axes"])
render(data, "figure.svg", renderer="vector", visuals=[CellAxes(length=3.0)])
render(data, "figure.png", renderer="raytrace",
       visuals=[{"name": "Cell axes", "params": {"length": 3.0}}])
```

`examples/plugins/cell_axes.py` holds this element and a second one, `SelectionHalo`, which
draws a see-through sphere around every selected atom.

## The contract

A visual element is a subclass of `ewald.scene.VisualElement`:

| Member | What it does |
|---|---|
| `name` | The name shown in the GUI, sessions, `ewald plugins` and the AI tools (default: the class name). |
| `__init__(...)` | Every argument stored as an attribute of the same name, with a default. The GUI parameter form and sessions are built from the signature, as for modifiers. |
| `applies(data)` | Optional; return False when there is nothing to draw. Keep it cheap. |
| `build(data, ctx)` | Return one primitive, a list of them, `(name, primitive)` pairs, or `None`. Never modify `data`. |
| `legend(data, ctx)` | Optional; a color bar for what `build` drew, `{"title", "cmap", "range": (lo, hi)}`, or None. Every renderer draws it next to a ColorCoding bar. |

`ctx` is a `SceneContext`: `ctx.frame`, `ctx.view_direction` (toward the camera, may be None)
and `ctx.particle_colors`, the per-atom colors the Particles element chose, so markers can
match their atoms.

Each layer is named `visual:<element>` (or `visual:<element>:<name>` for named primitives), so a
plugin never replaces a built-in actor.

## Primitives

All positions are world coordinates in Angstrom; primitives know nothing about the camera.

| Primitive | Fields | OpenGL | Ray tracer | Vector (SVG/PDF) |
|---|---|---|---|---|
| `Spheres` | `centers`, `radii`, `rgb`, `alpha`, `index` | sphere meshes or GPU impostors; `alpha` < 1 is see-through | spheres; `alpha` < 1 becomes a see-through triangle sphere | shaded discs with opacity |
| `Cylinders` | `start`, `end`, `radius`, `rgb`, `alpha`, `owner`, `start_cap`, `end_cap` ("none", "flat" (default) or "round", one or one per cylinder; `cap_end=False` = open end) | tubes with disk or hemisphere caps | cylinders with disk caps; round caps are spheres | sticks with a highlight, painted just behind `owner`'s atom, flat caps as ellipses, round ones as discs; `alpha` 0 hides one |
| `Lines` | `points`, `strips`, `rgb` (or None for the ink color), `point_rgb` (a color per point, for color maps), `width` (px at 1600 px height), `radius` | polylines, `point_rgb` blended along each piece | tubes with round joints when `radius` is set, else not drawn | strips scaled with the image height |
| `SurfaceMesh` | `vertices`, `faces`, colors, normals, `transparency`, `edges` | lit faces + edges | triangles + edge cylinders | lit polygons + edge strips |
| `Arrows` | from `make_arrows(base, tip, width, rgb, alpha, shading)` | cylinder + cone meshes | cylinders + cones | true silhouettes |

The ray tracer and OpenGL ignore cylinder `alpha` (except that the vector renderer skips
`alpha` 0); use `Spheres` or `SurfaceMesh` for see-through geometry.

## Where an element shows up

| Place | How |
|---|---|
| Scripts | `render(..., visuals=[...])`, `ewald.render.build_scene`, `ewald.raytrace.scene_from_data`, `ewald.vector.render_vector` |
| GUI | Pipeline panel › "+ Visual" adds it under Visual elements, with a checkbox and a parameter form; Del removes it |
| Sessions | stored in the view style (`style.visuals`) and restored by `load_session(...).render_kwargs()` |
| AI tools | `list_visuals`, `add_visual`, `remove_visual` (through the view cell); `render_image` draws the added ones |
| `ewald plugins` | listed as `visual <name>` with its category and origin |

## Registering

In a plugin-folder file, use `@visual` (bare, or with `name=`, `category=`, `description=`).
In an installed package, declare an entry point:

```toml
[project.entry-points."ewald.visuals"]
"Cell axes" = "my_package.visuals:CellAxes"
```

A plugin cannot take the name of a built-in element. `ewald plugins` reports a plugin that
fails to import instead of stopping ewald.

## Testing an element

`ewald.scene.build(data, visuals=[MyElement()])` returns the `Scene` without rendering
anything, so the geometry can be checked in a plain test:

```python
from ewald.scene import build

scene = build(data, visuals=[CellAxes(length=3.0)])
arrows = scene["visual:cell-axes"].primitive
assert arrows.count == 3
```

Rendering with `renderer="raytrace"` or `renderer="vector"` and a fixed `camera=` needs no
display; OpenGL needs one (`xvfb-run`). See [Testing a plugin](testing.md).
