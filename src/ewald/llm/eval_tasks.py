"""The built-in AI eval tasks (see :mod:`ewald.llm.evals`): questions about the shipped example
files and one paper dataset, each with an automatic grade and a reference solution.

Categories: ``answer`` (a value or fact in the reply), ``step`` (analysis steps in the
pipeline), ``view`` (how the structure is drawn), ``export`` (files written), ``code`` (cells the
model writes and runs; needs ``--allow-code``), ``notebook`` (editing a notebook the user already
has: change a step, remove one, fix a failing cell), ``recipe`` (a plot or per-frame step written
the way python_api's recipes do it), ``plugin`` (a plugin modifier or ``@ai_tool`` from
``examples/plugins``) and ``paper`` (published data, downloaded once).

Reference values are computed from the files with ewald itself when a task runs (the
``ref_*`` functions), so they follow the example files if those are regenerated; physical
facts the file was built from (lattice constants, compositions) are written out.
"""
from __future__ import annotations

import numpy as np

from .evals import (Any, CellNumber, Cells, Data, File, Matches, Number, Step, Task, UsedTool, View, along,
                    error_cells, same_color, step_cells)


# ------------------------------------------------------------------ reference values
def _distances(data):
    x = data.positions
    dx = x[:, None] - x[None]
    if data.cell is not None and np.any(data.cell.pbc):
        h = data.cell.matrix
        inv = np.linalg.inv(h)
        frac = dx @ inv
        frac -= np.round(frac) * np.asarray(data.cell.pbc, float)
        dx = frac @ h
    return np.linalg.norm(dx, axis=-1)


def _cn(data, cutoff):
    d = _distances(data)
    return ((d < cutoff) & (d > 1e-8)).sum(1)


def ref_pt_diameter(pipe):
    return float(_distances(pipe.compute(0)).max())


def ref_pt_low_cn(pipe):
    return int((_cn(pipe.compute(0), 3.2) < 12).sum())


def _cna_types(pipe, frame=0):
    from ..modifiers import CommonNeighborAnalysis

    pipe.modifiers.append(CommonNeighborAnalysis())
    return np.asarray(pipe.compute(frame).particles["Structure Type"])


def ref_pt_fcc_percent(pipe):
    return 100.0 * float(np.mean(_cna_types(pipe) == 1))


def ref_void_non_fcc(pipe):
    return int(np.sum(_cna_types(pipe) != 1))


def ref_void_low_cn(pipe):
    return int((_cn(pipe.compute(0), 3.0) < 12).sum())


def ref_cu_energy5(pipe):
    return float(pipe.compute(5).attributes["Energy"])


def ref_cu_fmax(pipe):
    f = pipe.compute(pipe.num_frames - 1).particles["Force"]
    return float(np.linalg.norm(f, axis=1).max())


def ref_cu_rdf_peak(pipe):
    d = _distances(pipe.compute(0))
    first = d[(d > 1e-8) & (d < 3.0)]
    return float(np.median(first))


def _si_msd(pipe):
    x0 = pipe.compute(0).positions
    h = pipe.compute(0).cell.matrix
    inv = np.linalg.inv(h)
    out = []
    for f in range(pipe.num_frames):
        frac = (pipe.compute(f).positions - x0) @ inv
        dx = (frac - np.round(frac)) @ h
        out.append(float((dx ** 2).sum(1).mean()))
    return out


def ref_si_msd_last(pipe):
    return _si_msd(pipe)[-1]


def ref_si_max_frame(pipe):
    return int(np.argmax(_si_msd(pipe)))


def ref_cuzr_nn(pipe):
    data = pipe.compute(0)
    t = np.asarray(data.particles["Particle Type"]).astype(str)
    d = _distances(data)
    return float(d[np.ix_(t == "Cu", t == "Zr")].min())


def ref_si_density(pipe):
    data = pipe.compute(0)
    return 28.0855 * data.count / abs(np.linalg.det(data.cell.matrix)) * 1.66053907


def ref_pt_below_zero(pipe):
    return int((pipe.compute(0).positions[:, 2] < 0).sum())


def ref_sto_mean_tio(pipe):
    data = pipe.compute(0)
    t = np.asarray(data.particles["Particle Type"]).astype(str)
    d = _distances(data)[np.ix_(t == "Ti", t == "O")]
    return float(d[d < 2.5].mean())


def ref_void_fcc_last(pipe):
    return int(np.sum(_cna_types(pipe, pipe.num_frames - 1) == 1))


def ref_cuzr_zr_fraction(pipe):
    data = pipe.compute(0)
    t = np.asarray(data.particles["Particle Type"]).astype(str)
    near = (_distances(data) < 3.2) & (_distances(data) > 1e-8)
    total = near.sum(1)
    return float(np.mean(np.where(total > 0, (near & (t == "Zr")[None]).sum(1) / np.maximum(total, 1), 0.0)))


def ref_sto_ti_bvs(pipe):
    data = pipe.compute(0)
    t = np.asarray(data.particles["Particle Type"]).astype(str)
    d = _distances(data)[np.ix_(t == "Ti", t == "O")]
    return float(np.exp((1.815 - d[d < 3.5]) / 0.37).sum() / np.sum(t == "Ti"))


# ------------------------------------------------------------------ file checks
def _png(path, outcome):
    return path.read_bytes()[:4] == b"\x89PNG"


def _frame3_xyz(path, outcome):
    from .. import import_file

    written = import_file(str(path))
    ref = outcome.reference().compute(3)
    got = written.compute(0)
    return written.num_frames == 1 and got.count == ref.count and np.allclose(got.positions, ref.positions, atol=1e-4)


def _types(name):
    return lambda data: int(np.sum(np.asarray(data.particles["Particle Type"]).astype(str) == name))


def _red(c):
    return same_color(c, "red")


def _hidden(data):
    return int(np.sum(np.asarray(data["Transparency"]) > 0.99)) if "Transparency" in data else 0


def _shows(table):
    """A view cell plots ``table`` (``show(pipe, table="...")``)."""
    return lambda nb: any(c.kind == "code" and "show(" in c.source and table in c.source for c in nb.cells)


def _cell_count(nb) -> int:
    return len(nb.cells)


def _plot_cells(table):
    """How many cells plot ``table`` (``show(pipe, table=...)``, for :class:`Cells`)."""
    return lambda nb: sum(1 for c in nb.cells if c.kind == "code" and (c.view or {}).get("table") == table)


def _plot_arg(table, name, expected):
    """The plot cell of ``table`` shows ``name`` = ``expected`` (numbers within 1e-6)."""
    def check(nb):
        args = [c.view for c in nb.cells if c.kind == "code" and (c.view or {}).get("table") == table]
        got = args[0].get(name) if len(args) == 1 else None
        return got is not None and np.allclose(np.asarray(got, float), np.asarray(expected, float), atol=1e-6)
    return check


#: A notebook with an RDF step and its plot (cells 3 and 4 after the import and load cells).
_RDF_CELLS = ["pipe.add(CoordinationAnalysis(cutoff=5.0, bins=50))", "show(pipe, table='coordination-rdf')"]
_RED_BLUE = {"coolwarm", "bwr", "rdbu", "seismic", "rdylbu"}


# ------------------------------------------------------------------ tasks
STO, SI, PT, CU, SIMD, CHG, VOID, CUZR = ("POSCAR.SrTiO3", "Si.cif", "Pt_nanoparticle.xyz", "Cu_md.extxyz",
                                          "XDATCAR.Si", "CHGCAR.Si", "Cu_void.lammpstrj", "CuZr.data")

TASKS: list[Task] = [
    # ---------------------------------------------------------- answers
    Task("sto-count", "How many atoms are in this structure, and how many of them are oxygen?", STO,
         [Number(40, name="atoms"), Number(24, index=1, name="O")],
         solution=[("describe_frame", {})], answer="40 atoms, 24 O.\nANSWER: {0}, {1}"),
    Task("sto-lattice", "This is a 2x2x2 supercell of cubic SrTiO3. What is the lattice constant of the "
         "primitive perovskite cell, in Angstrom?", STO, [Number(3.905, tol=0.01)],
         solution=[("describe_frame", {})], answer="a = 7.81/2.\nANSWER: {0}"),
    Task("sto-spacegroup", "What is the space group of this structure?", STO, [Matches(r"Pm-?3m|\b221\b")],
         solution=[("analyze_symmetry", {})], answer="Pm-3m (221)."),
    Task("sto-tio-bond", "What is the Ti-O bond length in Angstrom?", STO, [Number(1.9525, tol=0.02)],
         solution=[("compute_rdf", {"cutoff": 3.5})], answer="ANSWER: {0}"),
    Task("sto-ti-cn", "How many oxygen atoms surround each Ti atom (its coordination number)?", STO,
         [Number(6)], solution=[("add_modifier", {"name": "CoordinationAnalysis", "params": {"cutoff": 2.5}})],
         answer="ANSWER: {0}"),
    Task("si-density", "What is the density of this crystal in g/cm^3?", SI, [Number(ref_si_density, tol=0.02)],
         solution=[("describe_frame", {})], answer="ANSWER: {0}"),
    Task("si-spacegroup", "Which space group does this crystal have?", SI, [Matches(r"Fd-?3m|\b227\b")],
         solution=[("analyze_symmetry", {})], answer="Fd-3m (227)."),
    Task("si-nn", "What is the nearest-neighbour Si-Si distance in Angstrom?", SI, [Number(2.3517, tol=0.02)],
         solution=[("compute_rdf", {"cutoff": 2.7})], answer="ANSWER: {0}"),
    Task("pt-count", "How many atoms does this nanoparticle have?", PT, [Number(201)],
         solution=[("describe_frame", {})], answer="ANSWER: {0}"),
    Task("pt-diameter", "What is the largest distance between two atoms of the particle (its diameter) in "
         "Angstrom?", PT, [Number(ref_pt_diameter, rel=0.02)],
         solution=[("evaluate_expression", {"expression": "Position.X"})], answer="ANSWER: {0}"),
    Task("pt-low-cn", "How many atoms have fewer than 12 neighbours within 3.2 Angstrom?", PT,
         [Number(ref_pt_low_cn)],
         solution=[("add_modifier", {"name": "CoordinationAnalysis", "params": {"cutoff": 3.2}}),
                   ("evaluate_expression", {"expression": "Coordination < 12"})], answer="ANSWER: {0}"),
    Task("pt-fcc", "What percentage of the atoms does adaptive common neighbor analysis classify as FCC?", PT,
         [Number(ref_pt_fcc_percent, tol=1.0)],
         solution=[("add_modifier", {"name": "CommonNeighborAnalysis"})], answer="ANSWER: {0}"),
    Task("cu-frames", "How many frames does this trajectory have, and how many atoms are in each frame?", CU,
         [Number(10, name="frames"), Number(108, index=1, name="atoms")],
         solution=[("describe_trajectory", {})], answer="ANSWER: {0}, {1}"),
    Task("cu-energy", "What is the total energy of frame 5 (frames counted from 0), in eV?", CU,
         [Number(ref_cu_energy5, tol=1e-3)], solution=[("describe_frame", {"frame": 5})], answer="ANSWER: {0}"),
    Task("cu-fmax", "What is the largest force magnitude on any atom in the last frame, in eV/Angstrom?", CU,
         [Number(ref_cu_fmax, rel=0.01)],
         solution=[("evaluate_expression", {"expression": "sqrt(Force.X^2 + Force.Y^2 + Force.Z^2)", "frame": -1})],
         answer="ANSWER: {0}"),
    Task("cu-rdf-peak", "Where is the first peak of the radial distribution function in frame 0, in Angstrom?", CU,
         [Number(ref_cu_rdf_peak, tol=0.08)], solution=[("compute_rdf", {"frame": 0, "cutoff": 5.0})],
         answer="ANSWER: {0}"),
    Task("cu-lattice", "This is a 3x3x3 supercell of fcc Cu. What is the cubic lattice constant in Angstrom?", CU,
         [Number(3.615, tol=0.01)], solution=[("describe_frame", {})], answer="ANSWER: {0}"),
    Task("simd-msd", "What is the mean squared displacement of the Si atoms between the first and the last "
         "frame, in Angstrom^2?", SIMD, [Number(ref_si_msd_last, rel=0.1)],
         solution=[("compute_msd", {})], answer="ANSWER: {0}"),
    Task("simd-max-frame", "Which frame (counted from 0) deviates most from frame 0, measured by the mean "
         "squared displacement of the atoms?", SIMD, [Number(ref_si_max_frame)],
         solution=[("describe_trajectory", {})], answer="ANSWER: {0}"),
    Task("chg-grid", "What are the dimensions of the charge density grid?", CHG,
         [Matches(r"16\s*(x|×|,|by)\s*16\s*(x|×|,|by)\s*16")],
         solution=[("describe_frame", {})], answer="16 x 16 x 16"),
    Task("void-missing", "This block was cut from a perfect fcc Cu crystal filling the whole cell (lattice "
         "constant 3.615 Angstrom). How many atoms are missing?", VOID, [Number(38)],
         solution=[("describe_frame", {})], answer="ANSWER: {0}"),
    Task("void-low-cn", "In frame 0, how many atoms have fewer than 12 neighbours within 3.0 Angstrom?", VOID,
         [Number(ref_void_low_cn)],
         solution=[("add_modifier", {"name": "CoordinationAnalysis", "params": {"cutoff": 3.0}}),
                   ("evaluate_expression", {"expression": "Coordination < 12", "frame": 0})], answer="ANSWER: {0}"),
    Task("cuzr-composition", "How many Cu and how many Zr atoms are there?", CUZR,
         [Number(64, name="Cu"), Number(64, index=1, name="Zr")],
         solution=[("describe_frame", {})], answer="ANSWER: {0}, {1}"),
    Task("cuzr-spacegroup", "What is the space group of this CuZr crystal?", CUZR, [Matches(r"Pm-?3m|\b221\b")],
         solution=[("analyze_symmetry", {})], answer="Pm-3m (221), B2."),
    Task("cuzr-nn", "What is the shortest Cu-Zr distance in Angstrom?", CUZR, [Number(ref_cuzr_nn, tol=0.02)],
         solution=[("compute_rdf", {"cutoff": 3.5})], answer="ANSWER: {0}"),

    # ---------------------------------------------------------- pipeline steps
    Task("sto-color-cn", "Color the atoms by their coordination number.", STO,
         [Step("CoordinationAnalysis"),
          Any(Step("ColorCoding", {"property": lambda p: "Coordination" in str(p)}),
              View(color_by=lambda p: "Coordination" in str(p)))],
         category="step",
         solution=[("add_modifier", {"name": "CoordinationAnalysis", "params": {"cutoff": 2.5}}),
                   ("add_modifier", {"name": "ColorCoding", "params": {"property": "Coordination"}})],
         answer="Done."),
    Task("sto-octahedra", "Show the TiO6 octahedra as polyhedra.", STO,
         [Step("CoordinationPolyhedra"), Data(lambda d: len(d.surfaces) > 0, name="polyhedra drawn")],
         category="step",
         solution=[("add_modifier", {"name": "CoordinationPolyhedra", "params": {"center_types": ["Ti"]}})],
         answer="Done."),
    Task("sto-delete-o", "Remove all oxygen atoms from the structure.", STO,
         [Data(lambda d: d.count, 16, name="atoms left"), Data(_types("O"), 0, name="O left")], category="step",
         solution=[("add_modifier", {"name": "ExpressionSelection", "params": {"expression": "ParticleType == 'O'"}}),
                   ("add_modifier", {"name": "DeleteSelected"})], answer="Done."),
    Task("pt-half", "Keep only the atoms with a negative z coordinate (delete the rest).", PT,
         [Data(lambda d: d.count, ref_pt_below_zero, name="atoms left")], category="step",
         solution=[("add_modifier", {"name": "ExpressionSelection", "params": {"expression": "Position.Z >= 0"}}),
                   ("add_modifier", {"name": "DeleteSelected"})], answer="Done."),
    Task("void-cna", "Find the crystal structure of every atom with common neighbor analysis and select the "
         "atoms that are not FCC.", VOID,
         [Step("CommonNeighborAnalysis"), Data(lambda d: int(d.selection.sum()), ref_void_non_fcc, name="selected")],
         category="step",
         solution=[("add_modifier", {"name": "CommonNeighborAnalysis"}),
                   ("add_modifier", {"name": "ExpressionSelection", "params": {"expression": "StructureType != 1"}})],
         answer="Done."),
    Task("chg-isosurface", "Show an isosurface of the charge density at 0.3.", CHG,
         [Step("CreateIsosurface", {"isolevel": 0.3})], category="step",
         solution=[("add_modifier", {"name": "CreateIsosurface", "params": {"isolevel": 0.3}})], answer="Done."),
    Task("sto-replicate", "Show 3x3x1 periodic copies of the cell.", STO,
         [Any(View(replicate=[3, 3, 1]), Step("Replicate", {"n": [3, 3, 1]}))], category="step",
         solution=[("set_view", {"replicate": [3, 3, 1]})], answer="Done."),
    Task("cuzr-select-zr", "Select all Zr atoms.", CUZR,
         [Data(lambda d: int(d.selection.sum()), 64, name="selected"),
          Data(lambda d: bool(np.all(np.asarray(d.particles["Particle Type"]).astype(str)[d.selection.astype(bool)] == "Zr")),
               name="only Zr")], category="step",
         solution=[("add_modifier", {"name": "ExpressionSelection", "params": {"expression": "ParticleType == 'Zr'"}})],
         answer="Done."),

    # ---------------------------------------------------------- view
    Task("sto-white", "Make the background white and hide the bonds.", STO,
         [View(background="white", bonds=False)], category="view",
         solution=[("set_view", {"background": "white", "bonds": False})], answer="Done."),
    Task("sto-o-red", "Draw the O atoms in red with a radius of 0.6 Angstrom.", STO,
         [View(colors={"O": _red}, radii={"O": 0.6})], category="view",
         solution=[("set_view", {"colors": {"O": "red"}, "radii": {"O": 0.6}})], answer="Done."),
    Task("sto-top", "Look straight down the c axis.", STO, [View(direction=along("z"))], category="view",
         solution=[("set_view", {"direction": "z"})], answer="Done."),
    Task("cu-last-frame", "Show the last frame of the trajectory.", CU, [View(frame=lambda f: f is not None and int(f) % 10 == 9)], category="view",
         solution=[("set_view", {"frame": 9})], answer="Done."),
    Task("void-cell-scale", "Hide the simulation cell and add a scale bar.", VOID,
         [View(cell=False, scale_bar=lambda v: bool(v))], category="view",
         solution=[("set_view", {"cell": False, "scale_bar": True})], answer="Done."),
    Task("sto-type-legend", "Add a legend that shows which color is which element.", STO,
         [View(type_legend=True)], category="view", solution=[("set_view", {"type_legend": True})], answer="Done."),
    Task("si-spacefill", "Switch to a space-filling style.", SI, [View(style="Space-filling")], category="view",
         solution=[("set_view", {"style": "Space-filling"})], answer="Done."),

    # ---------------------------------------------------------- export
    Task("cu-export", "Save frame 3 (counted from 0) as an extended XYZ file named frame3.xyz.", CU,
         [File("frame3.xyz", _frame3_xyz)], category="export",
         solution=[("export_file", {"path": "frame3.xyz", "format": "extxyz", "frames": [3]})], answer="Saved."),
    Task("sto-render", "Render a picture of the structure and save it as sto.png.", STO,
         [File("sto.png", _png)], category="export",
         solution=[("render_image", {"filename": "sto.png"})], answer="Saved."),

    # ---------------------------------------------------------- code cells
    Task("sto-code-tio", "Add a notebook cell that computes and prints the mean Ti-O distance over all Ti-O "
         "pairs shorter than 2.5 Angstrom, and run it.", STO, [CellNumber(ref_sto_mean_tio, tol=0.01)],
         category="code", allow_code=True,
         solution=[("insert_cell", {"source": (
             "import numpy as np\nd = pipe.compute(0)\nt = np.asarray(d.particles['Particle Type']).astype(str)\n"
             "x = d.positions; L = d.cell.matrix.diagonal()\n"
             "v = x[t == 'Ti'][:, None] - x[t == 'O'][None]\nv -= L * np.round(v / L)\n"
             "r = np.linalg.norm(v, axis=-1)\nprint(round(float(r[r < 2.5].mean()), 4))")}),
             ("run_cell", {"index": -1})], answer="Done."),
    Task("cu-code-height", "Add a per-atom property called 'Height' equal to each atom's z coordinate, as a "
         "Python step in the notebook, and run it.", CU,
         [Data(lambda d: bool(np.allclose(d.particles["Height"], d.positions[:, 2])), name="Height == z")],
         category="code", allow_code=True,
         solution=[("insert_cell", {"source": (
             "@pipe.add\ndef height(frame, data):\n    data['Height'] = data.positions[:, 2].copy()")}),
             ("run_cell", {"index": -1})], answer="Done."),

    # ---------------------------------------------------------- the user's notebook
    Task("nb-edit-cutoff", "The coordination step in my notebook uses the wrong cutoff. Change it to 2.5 Angstrom.",
         STO, [Step("CoordinationAnalysis", {"cutoff": 2.5}), Cells(lambda nb: step_cells(nb, "CoordinationAnalysis"), 1,
                                                                    name="coordination cells")],
         category="notebook",
         setup=["pipe.add(CoordinationAnalysis(cutoff=2.0))", "pipe.add(ColorCoding(property='Coordination'))"],
         solution=[("edit_cell", {"index": 2, "params": {"cutoff": 2.5}})], answer="Done."),
    Task("nb-remove-step", "Take the color coding out of my notebook; I want the element colors back.", STO,
         [Cells(lambda nb: step_cells(nb, "ColorCoding"), 0, name="color coding cells"),
          Cells(lambda nb: step_cells(nb, "CoordinationAnalysis"), 1, name="coordination cells")],
         category="notebook",
         setup=["pipe.add(CoordinationAnalysis(cutoff=2.5))", "pipe.add(ColorCoding(property='Coordination'))"],
         solution=[("delete_cell", {"index": 3})], answer="Done."),
    Task("nb-fix-error", "One of my notebook cells fails. Fix it so the analysis runs.", CU,
         [Cells(error_cells, 0, name="cells with errors"), Step("CoordinationAnalysis", {"cutoff": 3.0})],
         category="notebook", allow_code=True,
         setup=["pipe.add(CoordinationAnalysis(cutof=3.0))"],
         solution=[("edit_cell", {"index": 2, "source": "pipe.add(CoordinationAnalysis(cutoff=3.0))"})],
         answer="Done."),

    # options changed on cells the notebook already has: the cell is edited, no second one added
    Task("nb-plot-range", "Make the RDF plot show only r = 2 to 4 Angstrom.", CU,
         [Cells(_plot_cells("coordination-rdf"), 1, name="RDF plot cells"),
          Cells(_plot_arg("coordination-rdf", "xlim", (2, 4)), name="xlim (2, 4)"), Cells(_cell_count, 4, name="cells")],
         category="notebook", allow_code=True, setup=_RDF_CELLS,
         solution=[("edit_cell", {"index": 3, "source": "show(pipe, table='coordination-rdf', xlim=(2, 4))"})],
         answer="Done."),
    Task("nb-rdf-bins", "The g(r) curve looks coarse. Make it finer, with 200 bins.", CU,
         [Step("CoordinationAnalysis", {"bins": 200, "cutoff": 5.0}),
          Cells(lambda nb: step_cells(nb, "CoordinationAnalysis"), 1, name="coordination cells"),
          Cells(_plot_cells("coordination-rdf"), 1, name="RDF plot cells"), Cells(_cell_count, 4, name="cells")],
         category="notebook", setup=_RDF_CELLS,
         solution=[("edit_cell", {"index": 2, "params": {"bins": 200}})], answer="Done."),
    Task("nb-redo-cutoff", "Redo it with a 7 Angstrom cutoff.", CU,
         [Step("CoordinationAnalysis", {"cutoff": 7.0}),
          Cells(lambda nb: step_cells(nb, "CoordinationAnalysis"), 1, name="coordination cells"),
          Cells(_plot_cells("coordination-rdf"), 1, name="RDF plot cells"), Cells(_cell_count, 4, name="cells")],
         category="notebook", setup=_RDF_CELLS,
         solution=[("edit_cell", {"index": 2, "params": {"cutoff": 7.0}})], answer="Done."),
    Task("nb-recolor", "I'd rather see the atoms in a red-to-blue color map.", CU,
         [Step("ColorCoding", {"cmap": lambda v: str(v).lower().removesuffix("_r") in _RED_BLUE}),
          Cells(lambda nb: step_cells(nb, "ColorCoding"), 1, name="color coding cells"), Cells(_cell_count, 4, name="cells")],
         category="notebook",
         setup=["pipe.add(CoordinationAnalysis(cutoff=3.0))", "pipe.add(ColorCoding(property='Coordination', "
                "cmap='viridis'))"],
         solution=[("edit_cell", {"index": 3, "params": {"cmap": "coolwarm"}})], answer="Done."),

    # ---------------------------------------------------------- recipes (cells as python_api writes them)
    Task("recipe-rdf-plot", "Plot the radial distribution function in the notebook.", CU,
         [Cells(_shows("rdf"), name="show cell"),
          Data(lambda d: any("rdf" in k for k in d.tables), name="rdf table")],
         category="recipe", allow_code=True,
         solution=[("add_modifier", {"name": "CoordinationAnalysis", "params": {"cutoff": 5.0}}),
                   ("insert_cell", {"source": "show(pipe, table='coordination-rdf')"})], answer="Done."),
    Task("recipe-msd-plot", "Compute the mean squared displacement over the trajectory and plot it in the notebook.",
         SIMD, [Step("MeanSquaredDisplacement"), Cells(_shows("msd"), name="show cell")],
         category="recipe", allow_code=True,
         solution=[("add_modifier", {"name": "MeanSquaredDisplacement"}),
                   ("insert_cell", {"source": "show(pipe, table='msd')"})], answer="Done."),
    Task("recipe-hide-fcc", "Hide the FCC atoms in the 3D view, in every frame, so only the defects around the "
         "void stay visible.", VOID,
         [Step("CommonNeighborAnalysis"), Data(_hidden, ref_void_fcc_last, frame=-1, name="hidden atoms")],
         category="recipe", allow_code=True,
         solution=[("add_modifier", {"name": "CommonNeighborAnalysis"}),
                   ("insert_cell", {"source": "# %% hide fcc [each frame]\nhide(data, 'StructureType == 1')"})],
         answer="Done."),

    # ---------------------------------------------------------- plugins (examples/plugins)
    Task("plugin-modifier", "A local composition plugin is installed. Use it to find the average fraction of Zr "
         "among each atom's neighbours within 3.2 Angstrom, over all atoms.", CUZR,
         [Step("LocalComposition", {"cutoff": 3.2, "element": "Zr"}), Number(ref_cuzr_zr_fraction, tol=0.005)],
         category="plugin", plugins=["local_composition.py"],
         solution=[("add_modifier", {"name": "LocalComposition", "params": {"cutoff": 3.2, "element": "Zr"}})],
         answer="ANSWER: {0}"),
    Task("plugin-tool", "What is the bond valence sum of the Ti atoms? A bond valence tool is installed; use it.",
         STO, [UsedTool("bond_valence_sums"), Number(ref_sto_ti_bvs, tol=0.02)],
         category="plugin", plugins=["bond_valence.py"],
         solution=[("bond_valence_sums", {})], answer="ANSWER: {0}"),

    # ---------------------------------------------------------- paper data (downloaded once)
    Task("li-diffusion", "This is an AIMD run of the Li6PS5Cl argyrodite; frames are 0.1 ps apart. What is the "
         "Li tracer diffusion coefficient in cm^2/s, fitting the mean squared displacement from 3 ps on?",
         "paper:li6ps5cl-aimd/XDATCAR.gz", [Number(1.23e-5, rel=0.25)], category="paper",
         solution=[("compute_msd", {"timestep_fs": 100.0, "fit_range": [0.215, 1.0]})], answer="ANSWER: {0}"),
]

#: Task ids by category.
CATEGORIES = sorted({t.category for t in TASKS})

__all__ = ["CATEGORIES", "TASKS"]
