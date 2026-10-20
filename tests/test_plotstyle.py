
import numpy as np
import pytest

pytest.importorskip("matplotlib")

from ewald import import_file, plotstyle  # noqa: E402
from ewald.verbs import Table  # noqa: E402
from ewald.views import figure  # noqa: E402


def test_style_is_journal_like():
    rc = plotstyle.rc()
    assert rc["font.size"] == 8 and rc["xtick.direction"] == "in" and rc["xtick.top"] and rc["ytick.minor.visible"]
    assert rc["pdf.fonttype"] == 42 and rc["svg.fonttype"] == "none" and rc["savefig.dpi"] == 600
    assert rc["axes.prop_cycle"].by_key()["color"][0] == plotstyle.LIGHT[0]
    assert plotstyle.rc({"bg": "#14161A"})["axes.prop_cycle"].by_key()["color"] == list(plotstyle.DARK)
    assert plotstyle.figsize("2-column") == (7.2, 3.8)
    with pytest.raises(ValueError):
        plotstyle.figsize("poster")


def test_cycle_starts_with_the_ui_accent():
    from ewald.gui.theme import PALETTES

    for name, tokens in PALETTES.items():
        cycle = plotstyle.colors(tokens)
        assert cycle[0] == tokens["accent"] and len(set(cycle)) == len(cycle) >= 5, name
    blue = plotstyle.colors({"bg": "#FFFFFF", "accent": "#1A66CC"})
    assert "#3A64C8" not in blue and "#2C8DD6" not in blue    # no second blue next to it
    assert plotstyle.colors({"bg": "#FFFFFF", "accent": "#22272E"}) == plotstyle.LIGHT   # gray accent


def test_labels_carry_units_and_legends_drop_the_shared_quantity():
    assert plotstyle.axis_label("r") == "$r$ (Å)"
    assert plotstyle.axis_label("g(r) Cu-Zr") == "$g(r)$"
    assert plotstyle.axis_label("Count Cu") == "Count"
    assert plotstyle.axis_label("Temp") == "Temp"
    assert plotstyle.legend_labels(["g(r)", "g(r) Cu-Cu", "g(r) Cu-Zr"]) == ["Total", "Cu–Cu", "Cu–Zr"]
    assert plotstyle.legend_labels(["MSD", "MSD fit"]) == ["MSD", "Fit"]
    assert plotstyle.legend_labels(["PotEng", "Temp"]) == ["PotEng", "Temp"]



def test_view_figure_saves_editable_text(tmp_path, vasp_dir):
    from ewald.modifiers import CoordinationAnalysis

    pipe = import_file(vasp_dir / "POSCAR")
    pipe.modifiers.append(CoordinationAnalysis(cutoff=2.6, partial=True))
    fig = figure(pipe, table="coordination-rdf")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "$r$ (Å)" and ax.get_ylabel() == "$g(r)$" and ax.get_xlim()[0] == 0
    assert tuple(fig.get_size_inches()) == plotstyle.SIZES[plotstyle.DEFAULT_SIZE]
    assert ax.get_legend().get_texts()[0].get_text() == "Total"
    plotstyle.save(fig, tmp_path / "g.pdf")
    plotstyle.save(fig, tmp_path / "g.svg")
    pdf = (tmp_path / "g.pdf").read_bytes()
    assert b"/CIDFontType2" in pdf and b"/Type3" not in pdf   # Type 42 (TrueType): text stays text
    assert "<text" in (tmp_path / "g.svg").read_text()
    zoomed = figure(pipe, table="coordination-rdf", xlim=(1.5, 2.5), ylim=(0, 3))   # what a zoomed view cell writes
    assert zoomed.axes[0].get_xlim() == (1.5, 2.5) and zoomed.axes[0].get_ylim() == (0, 3)


def test_table_picture_uses_the_style():
    from PIL import Image
    import io

    t = Table("msd", {"Time": np.arange(10.0), "MSD": np.arange(10.0) ** 1.1})
    png = t._repr_png_()
    w, h = Image.open(io.BytesIO(png)).size
    assert 400 < w < 700 and h < w


def test_session_swaps_rcparams_and_keeps_what_cells_set():
    import matplotlib as mpl

    before = mpl.rcParams["font.size"]
    s = plotstyle.Session()
    with s.active():
        assert mpl.rcParams["font.size"] == 8 and list(mpl.rcParams["figure.figsize"]) == [3.5, 2.6]
        mpl.rcParams["lines.linewidth"] = 3.0          # a cell's own choice
    assert mpl.rcParams["font.size"] == before and mpl.rcParams["lines.linewidth"] != 3.0
    s.configure("publication", {"bg": "#14161A", "text": "#EEEEEE", "secondary": "#CCCCCC", "accent": "#3FB8A6"})
    with s.active():
        assert mpl.rcParams["lines.linewidth"] == 3.0 and mpl.rcParams["axes.facecolor"] == "#14161A"
        plotstyle.use("default")                       # opt out from a cell
    with s.active():
        assert mpl.rcParams["font.size"] == mpl.rcParamsDefault["font.size"]
    s2 = plotstyle.Session("default")
    with s2.active():
        assert mpl.rcParams["font.size"] == before
    with pytest.raises(ValueError):
        plotstyle.Session("seaborn")


def test_notebook_cells_draw_plain_pyplot_in_the_style():
    import matplotlib as mpl

    from ewald.notebook import Cell, Notebook
    from ewald.settings import plot_style

    src = ("import matplotlib.pyplot as plt\nprint(plt.rcParams['font.size'], plt.rcParams['lines.linewidth'])\n"
           "plt.plot([0, 1], [0, 1])\nplt.show()")
    nb = Notebook([Cell("import matplotlib.pyplot as plt\nplt.rcParams['lines.linewidth'] = 4"), Cell(src)])
    nb.run_all()
    out = nb.cells[1].outputs
    assert out[0].data.split() == ["8.0", "4.0"] and out[1].kind == "image"
    assert mpl.rcParams["font.size"] != 8.0 or mpl.rcParamsDefault["font.size"] == 8.0   # process untouched
    nb.set_plot_style(*plot_style("Matplotlib default"))
    nb.run_all()
    assert nb.cells[1].outputs[0].data.split() == [str(float(mpl.rcParams["font.size"])), "4.0"]
    assert plot_style("Publication in UI theme colors", {"bg": "#000000"}) == ("publication", {"bg": "#000000"})
    assert plot_style("Publication", {"bg": "#000000"}) == ("publication", None)
