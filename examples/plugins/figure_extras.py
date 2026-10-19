"""Example figure-style plugins: a color map, a theme and an overlay.

``charge_difference`` is a diverging blue-white-red color map for ``ColorCoding``/``GridSlice``;
"Group style" is a theme (VESTA colors with a few overrides on white); ``InsetRDF`` draws the
g(r) of the current frame as a small plot in a corner of every rendered figure. See
docs/plugins/styles.md and docs/plugins/overlays.md.

Copy this file into ewald's plugin folder (``ewald plugins`` prints where it is), then::

    from ewald.modifiers import ColorCoding
    from ewald.render import render

    pipe.modifiers.append(ColorCoding("Charge", cmap="charge_difference"))
    render(pipe.compute(), "figure.png", theme="Group style", overlays=["Inset g(r)"])
"""
import numpy as np

from ewald.overlays import Image, OverlayElement, Rect
from ewald.plugins import colormap, overlay, theme
from ewald.themes import Theme


@colormap(kind="diverging", description="blue-white-red for charge or density differences")
def charge_difference():
    return ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"]


@theme(name="Group style")
def group_style():
    """VESTA colors, Li purple, on white with a gray cell."""
    return Theme(base="VESTA", colors={"Li": "#7f3fbf"}, background="white", cell_color="#555555")


@overlay(category="Plots")
class InsetRDF(OverlayElement):
    """The radial distribution function g(r) of the frame as an inset plot in a corner."""

    name = "Inset g(r)"

    def __init__(self, cutoff: float = 6.0, corner: str = "top-left", size: float = 0.3):
        # every constructor argument is stored under its own name: parameters(), the GUI form,
        # sessions and the Claude tools read them back from here
        self.cutoff = cutoff
        self.corner = corner
        self.size = size

    def applies(self, data):
        return data is not None and data.count > 1 and data.cell.is_valid

    def layout(self, data, ctx):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from ewald.kernels import rdf

        m = np.asarray(data.cell.matrix, float)
        widths = abs(np.linalg.det(m)) / np.linalg.norm(np.cross(m[[1, 2, 0]], m[[2, 0, 1]]), axis=1)
        cutoff = min(self.cutoff, 0.499 * widths.min())   # the neighbor search needs < half the cell width
        table = rdf(data.positions, data.cell, cutoff, 120)
        w = h = int(self.size * ctx.size[1])
        dpi = 100
        fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0.22, 0.25, 0.74, 0.69])
        ax.patch.set_alpha(0.0)
        ax.plot(table["r"], table["g(r)"], color=ctx.ink, linewidth=1.0)
        ax.set_xlabel("r (Å)", fontsize=ctx.small * 0.7, color=ctx.ink)
        ax.set_ylabel("g(r)", fontsize=ctx.small * 0.7, color=ctx.ink)
        ax.tick_params(labelsize=ctx.small * 0.6, colors=ctx.ink)
        for spine in ax.spines.values():
            spine.set_color(ctx.ink)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba()).copy()
        x, y = ctx.corner(self.corner, w, h)
        return [Image(x, y, rgba, w, h), Rect(x, y, x + w, y + h, outline="ink", width=1)]
