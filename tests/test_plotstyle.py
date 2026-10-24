
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


def test_style_figures_draw_math_in_the_style_outside_its_context(monkeypatch):
    """A GUI canvas paints from the event loop and savefig may run after ``with context()``:
    math must still come from the style's sans, never from matplotlib's default
    ``mathtext.cal`` = "cursive" (Apple Chancery on macOS, with a findfont weight warning)."""
    import io

    import matplotlib as mpl
    from matplotlib import _mathtext

    asked = []
    real = _mathtext.findfont
    monkeypatch.setattr(_mathtext, "findfont", lambda prop, *a, **k: asked.append(str(prop)) or real(prop, *a, **k))
    with mpl.rc_context(mpl.rcParamsDefault):   # process rcParams: matplotlib's, as in the GUI
        fig = plotstyle.figure()
        with plotstyle.context(*fig._ewald_style):
            ax = fig.add_subplot()
            ax.plot([0, 1], [0, 2e5])
            ax.set_xlabel(plotstyle.axis_label("r"))
        fig.canvas.draw()                                       # outside the context
        fig.savefig(io.BytesIO(), format="png")
        assert mpl.rcParams["mathtext.fontset"] == "dejavusans"   # the process rc is left alone
    assert asked and not any("cursive" in a for a in asked)
    assert any(plotstyle._sans() in a for a in asked)


def test_bound_style_follows_a_changing_theme():
    import matplotlib as mpl
    from matplotlib.artist import Artist
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    seen = []

    class Probe(Artist):
        def draw(self, renderer):
            seen.append((mpl.rcParams["font.size"], mpl.rcParams["axes.facecolor"]))

    tokens = {"bg": "#FFFFFF"}
    fig = plotstyle.bind(Figure(), lambda: (tokens, 9.0))
    fig.add_artist(Probe())
    FigureCanvasAgg(fig).draw()
    tokens = {"bg": "#14161A"}
    fig._ewald_style = lambda: (tokens, 11.0)
    fig.canvas.draw()
    assert seen == [(9.0, "#FFFFFF"), (11.0, "#14161A")]


# ---------------------------------------------------------------- style files (user.mplstyle)

def test_style_file_reads_crlf_bom_comments_and_reports_bad_lines(tmp_path):
    f = tmp_path / "s.mplstyle"
    f.write_bytes("\ufefffont.size: 10  # comment\r\n# lines.linewidth: 3\r\n"
                  "axes.prop_cycle: cycler(color=['D55E00', '#008C7E'])\r\nfoo.bar: 1\r\nfont.size 4\r\n"
                  "lines.markersize: big\r\nbackend: agg\r\n".encode("utf-8"))
    rc, problems = plotstyle.read_style(f)
    assert rc == {"font.size": "10", "axes.prop_cycle": "cycler(color=['D55E00', '#008C7E'])"}
    assert [p.split(":")[0] for p in problems] == ["line 4", "line 5", "line 6", "line 7"]
    assert plotstyle.read_style(tmp_path / "missing.mplstyle") == ({}, [])


def test_style_template_groups_keys_and_reads_back_uncommented():
    text = plotstyle.template()
    assert plotstyle.parse_style(text) == ({}, [])            # all commented out
    assert "## Fonts" in text and "## Ticks" in text and plotstyle.preset_of_text(text) == "publication"
    rc, problems = plotstyle.parse_style("\n".join(line[2:] for line in text.splitlines()
                                                   if line.startswith("# ") and plotstyle._key_of(line, commented=True)))
    assert not problems and set(rc) == {k for _t, _h, keys in plotstyle.TEMPLATE for k in keys}
    pub = plotstyle._style_rc("publication")
    for key, value in plotstyle.validated(rc).items():
        if key in pub:
            assert value == plotstyle.validated({key: pub[key]})[key], key
    for name in plotstyle.PRESETS:                           # a preset's file sets its keys, and only those
        assert plotstyle.parse_style(plotstyle.template(name)) == (plotstyle.preset_rc(name), [])


def test_presets_switch_without_touching_other_lines(tmp_path):
    f = tmp_path / "user.mplstyle"
    plotstyle.apply_preset("compact", f)
    assert plotstyle.preset_of(f) == "compact" and plotstyle.read_style(f)[0]["font.size"] == "7"
    with open(f, "a", encoding="utf-8") as fh:
        fh.write("axes.grid: True\n")
    plotstyle.apply_preset("Serif", f)          # by label too
    rc, problems = plotstyle.read_style(f)
    assert not problems and plotstyle.preset_of(f) == "serif"
    assert rc["font.family"] == "serif" and rc["font.size"] == "8" and rc["axes.grid"] == "True"
    assert "lines.linewidth" not in rc and "font.sans-serif" not in rc   # Compact's own keys went
    plotstyle.apply_preset("publication", f)
    assert plotstyle.read_style(f)[0] == {"axes.grid": "True"}
    with pytest.raises(ValueError, match="unknown preset"):
        plotstyle.preset_rc("nope")
    import matplotlib as mpl

    with mpl.rc_context():
        plotstyle.use("slides")
        assert mpl.rcParams["font.size"] == 14 and mpl.rcParams["lines.linewidth"] == 2


def test_write_style_keeps_other_lines_and_uncomments_template_lines(tmp_path):
    f = tmp_path / "plots" / "user.mplstyle"
    plotstyle.write_style({"font.size": "10"}, f)               # a new file starts from the template
    text = f.read_bytes().decode("utf-8")
    assert "\r" not in text and "\nfont.size:" in text and "# EWALD plot style" in text
    assert text.count("font.size:") == 1                      # the template line took the value
    f.write_text(text + "axes.grid: True   # mine\nfont.size: 12\n", encoding="utf-8")
    plotstyle.write_style({"lines.linewidth": "2", "font.size": "9"}, f)
    rc, _ = plotstyle.read_style(f)
    assert rc == {"font.size": "9", "lines.linewidth": "2", "axes.grid": "True"}
    assert "axes.grid: True   # mine" in f.read_text(encoding="utf-8")
    plotstyle.write_style({"font.size": None}, f)
    assert "font.size" not in plotstyle.read_style(f)[0] and "# font.size:" in f.read_text(encoding="utf-8")


def test_form_fields_round_trip_through_the_file(tmp_path):
    f = tmp_path / "user.mplstyle"
    fields = {x.label: x for x in plotstyle.FORM}
    updates = {}
    for label, value in [("Font", "Liberation Sans, Arial"), ("Font size", 9.5), ("Color cycle", "d55e00, #008C7E, red")]:
        updates.update(fields[label].updates(value))
    two = plotstyle.FormField("Ticks", ("xtick.direction", "ytick.direction"), "choice", choices=("in", "out"))
    updates.update(two.updates("out"))
    off = plotstyle.FormField("Minor", ("xtick.minor.visible", "ytick.minor.visible"), "bool")
    updates.update(off.updates(False))
    assert len(plotstyle.FORM) <= 4
    plotstyle.write_style(updates, f)
    rc, problems = plotstyle.read_style(f)
    assert not problems
    assert fields["Font"].value(rc) == "Liberation Sans, Arial" and fields["Font size"].value(rc) == 9.5
    assert two.value(rc) == "out" and rc["ytick.direction"] == "out" and off.value(rc) is False
    assert fields["Color cycle"].value(rc) == "#D55E00, #008C7E, #FF0000"
    assert fields["Line width"].value(rc) is None and fields["Line width"].default() == 1.25
    assert fields["Font size"].updates(None) == {"font.size": None}
    with pytest.raises(ValueError, match="Color cycle"):
        fields["Color cycle"].updates("#12, nocolor")


def test_style_file_goes_over_the_publication_style():
    user = {"font.size": "10", "font.sans-serif": "Liberation Sans, Arial", "axes.facecolor": "EEEEEE",
            "figure.figsize": "4, 3", "lines.linewidth": "2"}
    rc = plotstyle.rc(None, 8, user)
    assert rc["font.size"] == 10 and rc["xtick.labelsize"] == 9 and rc["lines.linewidth"] == 2
    plotstyle._sans()
    if "liberation sans" in plotstyle._FONTS:
        assert rc["mathtext.rm"] == "Liberation Sans"
    pub = plotstyle._style_rc("publication", None, user=user)
    assert pub["axes.facecolor"] == "#EEEEEE" and pub["figure.figsize"] == [4.0, 3.0]
    screen = plotstyle._style_rc("publication", {"bg": "#14161A"}, user=user)   # a UI theme keeps its colors
    assert screen["axes.facecolor"] == "#14161A" and screen["lines.linewidth"] == 2
    assert plotstyle._style_rc("default", None, user=user)["lines.linewidth"] == 2


def test_use_reads_style_files_only_when_asked(tmp_path):
    import matplotlib as mpl

    plotstyle.write_style({"lines.linewidth": "3"})            # <EWALD_CONFIG_DIR>/plots/user.mplstyle
    assert plotstyle.user_style_path().parent.name == "plots"
    (plotstyle.user_style_path().parent / "Thesis.mplstyle").write_text("lines.linewidth: 0.5\n", encoding="utf-8")
    other = tmp_path / "Mine.MPLSTYLE"
    other.write_text("lines.linewidth: 5\r\n", encoding="utf-8")
    with mpl.rc_context():
        plotstyle.use()
        assert mpl.rcParams["lines.linewidth"] == 1.25          # never implicitly
        plotstyle.use("user")
        assert mpl.rcParams["lines.linewidth"] == 3 and mpl.rcParams["font.size"] == 8
        plotstyle.use("THESIS")                                 # names ignore case
        assert mpl.rcParams["lines.linewidth"] == 0.5
        plotstyle.use(str(other))
        assert mpl.rcParams["lines.linewidth"] == 5
        assert set(plotstyle.named_styles()) == {"user", "thesis"}
        with pytest.raises(ValueError, match="unknown plot style"):
            plotstyle.use("nope")
        with pytest.warns(UserWarning, match="does not exist"):
            plotstyle.use(str(tmp_path / "gone.mplstyle"))


def test_session_and_notebook_take_the_style_file_and_cells_still_win():
    from ewald.notebook import Cell, Notebook

    src = "import matplotlib.pyplot as plt\nprint(plt.rcParams['lines.linewidth'], plt.rcParams['font.size'])"
    nb = Notebook([Cell(src)])
    nb.set_plot_style("publication", None, {"lines.linewidth": "2"})
    nb.run_all()
    assert nb.cells[0].outputs[0].data.split() == ["2.0", "8.0"]
    nb.cells.insert(0, Cell("import matplotlib.pyplot as plt\nplt.rcParams['font.size'] = 11"))
    nb.run_all()
    nb.set_plot_style("publication", None, {"lines.linewidth": "3", "font.size": "9"})   # the file changed
    nb.run_cell(1)
    assert nb.cells[1].outputs[0].data.split() == ["3.0", "11.0"]        # the cell's font size stays


def test_editor_commands_per_platform():
    from ewald.fileops import editor_commands

    assert editor_commands("C:\\Users\\me\\user.mplstyle", "win32") == ["url", ["notepad.exe", "C:\\Users\\me\\user.mplstyle"]]
    assert editor_commands("/Users/me/user.mplstyle", "darwin") == [["open", "-t", "/Users/me/user.mplstyle"], "url"]
    assert editor_commands("/home/me/user.mplstyle", "linux") == ["url"]
