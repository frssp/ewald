# Overlay plugins

An overlay draws in image space on top of a rendered figure: an inset plot, a readout of the
current temperature, a compass, a logo. Each overlay element returns a few 2D items in pixel
coordinates, and every renderer draws them, so an overlay you write appears in OpenGL images,
ray-traced images and SVG/PDF output alike (in vector output as real text and shapes).

ewald ships two overlay elements, registered the same way plugins are:

| Overlay | What it draws |
|---|---|
| `Coordinate tripod` | x, y, z (or the cell vectors a, b, c with `axes="abc"`) as arrows in a corner, turning with the camera |
| `Attributes` | global attributes as `name = value` lines (`attributes="Temperature, PotentialEnergy"`; empty shows every numeric one) |

The text label, color legend, type legend and scale bar stay `render()` options (`text=`,
`legend=`, `type_legend=`, `scale_bar=`).

## Using overlays

```python
from ewald.render import render

render(data, "figure.png", overlays=["Coordinate tripod"])
render(data, "figure.pdf", renderer="vector",
       overlays=[{"name": "Attributes", "params": {"attributes": "Temperature", "corner": "top-left"}}])
```

On the command line: `ewald render run.xyz fig.png --overlay "Coordinate tripod" --overlay Attributes`.
In the GUI, the Render page's "+ Overlay" button adds one under Overlays; the checkbox turns it
off, the form below edits its parameters, and the session keeps the list.

## A first overlay

```python
# <config dir>/plugins/stamp.py
from ewald.plugin_api import OverlayElement, Rect, Text, overlay


@overlay(category="Annotations")
class Stamp(OverlayElement):
    """A boxed label such as 'DRAFT' in a corner."""

    name = "Stamp"

    def __init__(self, text: str = "DRAFT", corner: str = "top-right"):
        self.text = text              # store every argument under its own name
        self.corner = corner

    def layout(self, data, ctx):
        fs = ctx.font_size
        w, h = 0.65 * fs * len(self.text) + fs, 1.6 * fs
        x, y = ctx.corner(self.corner, w, h)
        return [Rect(x, y, x + w, y + h, outline="red", width=2),
                Text(x + w / 2, y + h / 2, self.text, fs, "red", anchor="mm", bold=True)]
```

`examples/plugins/figure_extras.py` has a larger one, `Inset g(r)`, which plots the radial
distribution function of the frame with matplotlib and places it as an image.

## The contract

| Member | What it does |
|---|---|
| `name` | The name shown in the GUI, sessions, `ewald plugins` and the AI tools (default: the class name). |
| `__init__(...)` | Every argument stored as an attribute of the same name, with a default; the GUI form and sessions are built from the signature. `corner` and `axes` get a drop-down in the GUI. |
| `applies(data)` | Optional; return False when there is nothing to draw. |
| `layout(data, ctx)` | Return a list of items. Never modify `data`. |

`ctx` is an `OverlayContext`:

| Field | Meaning |
|---|---|
| `size` | image (width, height) in pixels |
| `camera` | the camera dict (`position`, `focal_point`, `up`, ...), None when unknown; `ewald.overlays.camera_axes(camera)` gives the screen's right, up and forward vectors |
| `frame` | the frame number |
| `font_size`, `small` | the figure's text sizes in pixels (they grow with the image) |
| `ink` | RGB that reads on the background (dark on white, light on dark) |
| `corner(name, w, h)` | top-left of a `w` x `h` box in 'top-left', 'top-right', 'bottom-left' or 'bottom-right' |

## Items

Coordinates are pixels with the origin at the top left. A color is anything matplotlib accepts;
`None` (or `"ink"`) is the ink color.

| Item | Fields |
|---|---|
| `Text` | `x`, `y`, `text`, `size` (px, None for the figure's), `color`, `anchor` (horizontal `l`/`m`/`r` + vertical `t`/`m`/`b`), `bold` |
| `Rect` | `x0`, `y0`, `x1`, `y1`, `fill`, `outline`, `width` |
| `Circle` | `x`, `y`, `r`, `fill`, `outline`, `width` |
| `Polyline` | `points` (n x 2), `color`, `width`, `closed`, `fill` (a closed polyline with a fill is a polygon) |
| `Image` | `x`, `y` (top left), `rgba` (H x W x 3 or 4 uint8), `width`, `height` |

## Registering

`@overlay` (bare, or with `name=`, `category=`, `description=`) in a plugin-folder file, or an
entry point in an installed package:

```toml
[project.entry-points."ewald.overlays"]
"Stamp" = "my_package.overlays:Stamp"
```

A plugin cannot take the name of a built-in overlay. The AI tools `list_overlays`,
`add_overlay` and `remove_overlay` see every registered overlay.

## Testing an overlay

`layout` needs no renderer, so items can be checked directly:

```python
from ewald.plugin_api import OverlayContext, Text

items = Stamp("TEST").layout(data, OverlayContext((400, 300), font_size=12))
assert [it.text for it in items if isinstance(it, Text)] == ["TEST"]
```

`ewald.overlays.draw_overlays(image, data, overlays=[Stamp()])` draws onto an RGB array
without rendering a structure, and `render(..., renderer="vector")` runs headless. See
[Testing a plugin](testing.md).
