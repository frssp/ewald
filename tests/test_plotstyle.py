
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


def test_table_picture_uses_the_style():
    from PIL import Image
    import io

    t = Table("msd", {"Time": np.arange(10.0), "MSD": np.arange(10.0) ** 1.1})
    png = t._repr_png_()
    w, h = Image.open(io.BytesIO(png)).size
    assert 400 < w < 700 and h < w
