# Color maps and themes

Two small kinds of plugin change how a figure looks without drawing anything new: a color map
(numbers to colors) and a theme (element colors, background, cell color and radius preset under
one name).

## Color maps

Every place ewald maps numbers to colors takes a color map name: `ColorCoding(cmap=...)`,
`GridSlice(cmap=...)`, `VectorVis(cmap=...)`, the color legend of every renderer and the GUI's
`cmap` drop-down. Any matplotlib name works, and so does a registered plugin:

```python
# <config dir>/plugins/maps.py
from ewald.plugins import colormap


@colormap(kind="diverging", description="blue-white-red for charge differences")
def charge_difference():
    return ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"]
```

The function returns a list of colors (names, hex or RGB(A) in 0..1, spaced evenly from 0 to 1),
an (n, 3|4) array, or a matplotlib `Colormap`; it runs once, the first time the map is used.
`register_colormap("name", colors)` registers data directly, and an installed package uses the
`ewald.colormaps` entry point group. `kind` is `sequential`, `diverging`, `cyclic` or
`categorical`. Appending `_r` to a plugin name reverses it, as for matplotlib maps.

```python
from ewald.modifiers import ColorCoding

pipe.modifiers.append(ColorCoding("Charge", cmap="charge_difference"))
```

`ewald.colormaps.get_colormap(name)` resolves a name to a matplotlib `Colormap`, so plugin
code that colors things itself should call it instead of `matplotlib.colormaps[...]`.

### Grayscale check

`ewald.colormaps.is_perceptual(name, kind)` checks that CIELAB lightness changes monotonically
along the map (on each side of the middle for diverging maps), which is what keeps a figure
readable in grayscale print and for most color-blind readers; `lightness(name)` returns the L*
curve. The GUI's color map drop-down shows a preview strip and this verdict as a tooltip, and the
Claude tool `list_styles` reports it as `grayscale_safe`.

## Themes

A theme sets element colors (a base scheme plus overrides, optionally muted toward gray),
background, cell color and radius preset. Built-in themes:

| Theme | What it sets |
|---|---|
| `Default` | Jmol element colors and covalent radii (ewald's defaults) |
| `VESTA` | VESTA's element colors on white |
| `CPK` | CPK element colors |
| `Journal` | Jmol colors muted 35% toward gray, white background, dark gray cell |
| `ASE` | ASE's GUI: Jmol colors and covalent radii on white, black cell (pair with `atom_style="flat"`) |
| `Dark slide` | Jmol colors on black, light cell lines |

```python
render(data, "figure.png", theme="VESTA")
render(data, "slide.png", theme="Dark slide", type_colors={"Li": "#c080ff"})   # explicit settings win
```

On the command line it is `--theme VESTA`; in the GUI, View › Theme. The GUI copies the theme's
values into the view style, so a session keeps them even where the theme plugin is not installed.

A theme plugin returns a `Theme`:

```python
from ewald.plugins import theme
from ewald.themes import Theme


@theme(name="Group style")
def group_style():
    """VESTA colors, Li purple, on white with a gray cell."""
    return Theme(base="VESTA", colors={"Li": "#7f3fbf"}, background="white", cell_color="#555555")
```

| `Theme` field | Meaning |
|---|---|
| `base` | element color scheme underneath: `Jmol`, `CPK` or `VESTA` |
| `colors` | per element or per particle type, on top of `base` |
| `mute` | 0..1, how far colors move toward the gray of the same lightness |
| `background`, `cell_color`, `radius_preset` | None leaves the setting alone |

Particle types map to elements the way the rest of ewald maps them (`Fe3+` uses Fe); types the
theme does not know keep their default colors. An installed package registers themes under the
`ewald.themes` entry point group, and a plugin cannot take a built-in theme's name.

`examples/plugins/figure_extras.py` holds both examples above and an overlay.
