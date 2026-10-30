"""The built-in AI eval tasks (see :mod:`ewald.llm.evals`): questions about the shipped example
files and one paper dataset, each with an automatic grade and a reference solution.

Categories: ``answer`` (a value or fact in the reply), ``step`` (analysis steps in the
pipeline), ``view`` (how the structure is drawn), ``export`` (files written), ``code`` (cells the
model writes and runs; needs ``--allow-code``), ``notebook`` (editing a notebook the user already
has: change a step, remove one, fix a failing cell), ``recipe`` (a plot or per-frame step written
the way python_api's recipes do it), ``plugin`` (a plugin modifier or ``@ai_tool`` from
``examples/plugins``), ``paper`` (published data, downloaded once) and ``study`` (a whole study
from one sentence: a series of MD runs written by :mod:`ewald.llm.eval_data`, compared, fitted,
plotted and written up, some with a trap a careful analyst catches or a premise that is wrong) and
``research`` (a long request in Korean with several subtasks, as a researcher writes it, graded
per subtask: values, the analysis choices reported, the figure, no failed cell, the language and
how much code the model wrote).

Reference values are computed from the files with ewald itself when a task runs (the
``ref_*`` functions), so they follow the example files if those are regenerated; physical
facts the file was built from (lattice constants, compositions) are written out.
"""
from __future__ import annotations

import numpy as np

from .evals import (ANSWER_TURN, Any, CellNumber, Cells, Data, File, Matches, Mentions, NoAnswer, Not, Number, Prose, Step, Task,
                    UsedTool, View, along, code_lines, error_cells, figure_cells, same_color, step_cells)


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


_TOP_MSD = {}


def _top_msd_ids(n=5, species="Li"):
    """The ids of the ``n`` ``species`` atoms of the Li6PS5Cl run with the largest MSD, under every
    reading of "an atom's MSD" a careful analyst may take: its squared displacement first to
    last frame, its time-averaged MSD over all lags, or over the lags of a diffusion fit window
    (the default one and the paper's, from 3 ps on). A list of sets; cached."""
    if (n, species) not in _TOP_MSD:
        from .. import examples, import_file
        from ..kernels import collect_trajectory, fit_lags, particle_msd, unwrap

        pipe = import_file(str(examples.fetch("li6ps5cl-aimd", quiet=True) / "XDATCAR.gz"))
        traj = collect_trajectory(pipe.source_frame(k) for k in range(pipe.num_frames))
        x = np.asarray(unwrap(traj["Position"], traj["cells"], traj["pbc"]))
        ids, mine = np.asarray(traj["Particle Identifier"]), np.asarray(traj["types"]).astype(str) == species
        rankings = [((x[-1] - x[0]) ** 2).sum(-1), particle_msd(x)] + [
            particle_msd(x, fit_lags(len(x), window)) for window in ((0.2, 0.8), (0.215, 1.0))]
        _TOP_MSD[n, species] = [set(ids[mine][np.argsort(-r[mine], kind="stable")[:n]].tolist()) for r in rankings]
    return _TOP_MSD[n, species]


def _traces_top_msd(data):
    """The trajectory lines are those of the five Li atoms with the largest MSD (any reading)."""
    traced = set(np.unique(np.asarray(data.tables["trajectory-lines"]["Particle Identifier"])).tolist())
    return traced in _top_msd_ids()


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



def _show_kwargs(nb) -> list[dict]:
    """The literal keyword arguments of every ``show(...)`` call in ``nb``'s code cells."""
    import ast

    out = []
    for c in nb.cells:
        if c.kind != "code":
            continue
        try:
            tree = ast.parse(c.source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "show":
                kw = {}
                for k in node.keywords:
                    try:
                        kw[k.arg] = ast.literal_eval(k.value)
                    except ValueError:
                        kw[k.arg] = None
                out.append(kw)
    return out


def _compares(frames=None, columns=None, layout=None):
    """A ``show(...)`` call compares ``frames`` (each a set of accepted indices, e.g. {9, -1}) or
    ``columns`` (count) with ``layout``, through show() rather than hand-written matplotlib."""
    def check(nb):
        if any("plt." in c.source or "matplotlib.pyplot" in c.source for c in nb.cells if c.kind == "code"):
            return False
        for kw in _show_kwargs(nb):
            got = kw.get("frame")
            if frames is not None and not (isinstance(got, (list, tuple)) and len(got) == len(frames)
                                           and all(g in ok for g, ok in zip(got, frames))):
                continue
            y = kw.get("y")
            if columns is not None and not (isinstance(y, (list, tuple)) and len(y) == columns):
                continue
            if layout is not None and kw.get("layout") != layout:
                continue
            return True
        return False
    return check

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


# ------------------------------------------------------------------ study tasks (eval_data series)
_LI_D = "MeanSquaredDisplacement.diffusion_coefficient.Li"
_STUDY = {}


def _series_fit(skip_unequilibrated=False, exclude=()):
    """Li diffusion coefficients (A^2/ps) and their Arrhenius fit for the series in the current
    folder (the task's working folder), the way a careful analyst does it: Li only, temperatures
    from each run's INCAR, the unequilibrated start cut where check_run finds one, runs in
    ``exclude`` left out. Cached per folder."""
    import os

    key = (os.getcwd(), skip_unequilibrated, tuple(exclude))
    if key not in _STUDY:
        from ..runinfo import check_run
        from ..runs import arrhenius, load_runs
        from ..verbs import msd

        runs = load_runs("*/XDATCAR")
        temps, ds = [], []
        for run, t in zip(runs, runs.temperatures()):
            if run.label in exclude:
                continue
            start = (check_run(run.pipe).equilibrated or 0) if skip_unequilibrated else 0
            ds.append(float(msd(run.pipe, start=start).attributes[_LI_D]))
            temps.append(float(t))
        _STUDY[key] = (dict(zip(temps, ds)), arrhenius(temps, ds))
    return _STUDY[key]


def _ref_ea(**kw):
    return lambda _pipe: _series_fit(**kw)[1].Ea


def _ref_d_cm2(temperature, **kw):
    return lambda _pipe: _series_fit(**kw)[0][float(temperature)] * 1e-4


#: Activation energies a model may claim: "Ea = 0.3 eV", "activation energy of 0.28 eV".
_EA_CLAIM = r"(?i)(E_?\{?a\}?|activation energy)[^.\n]{0,40}?[=≈~:]?\s*(of\s*)?\$?\d+\.\d+\s*\$?\s*eV"

#: The reference solution of the study tasks: one cell from the runs to Ea (Li only, INCAR temperatures).
_STUDY_CELL = """from ewald import load_runs, msd, arrhenius, check_run
runs = load_runs("*/XDATCAR")
{prep}D = {{}}
for run in runs:
    {skip}D[run.label] = msd(run.pipe{start}).attributes["MeanSquaredDisplacement.diffusion_coefficient.Li"]
T = [t for r, t in zip(runs, runs.temperatures()) if r.label in D]
fit = arrhenius(T, list(D.values()))
print(f"Ea = {{fit.Ea:.4f}} eV")
fit"""


def _study_cell(prep="", skip="", start=""):
    return _STUDY_CELL.format(prep=prep, skip=skip, start=start)


def _study_checks(ea, tol=0.03, figure=True):
    """What every Arrhenius study must leave: the value, printed by a cell (so the notebook makes
    it), an Arrhenius figure under a cell, and no failed cell."""
    checks = [Number(ea, tol=tol, name="Ea"), CellNumber(ea, tol=tol), Cells(error_cells, 0, name="cells with errors")]
    return checks + ([Cells(figure_cells, True, name="figure")] if figure else [])


def _fit_cells(nb) -> int:
    """How many code cells fit an Arrhenius law (call ``arrhenius(...)``)."""
    import re

    return sum(1 for c in nb.cells if c.kind == "code" and re.search(r"\barrhenius\s*\(", c.source))


#: The Li of the eval series hop in the ab plane only: "ab plane", "2D", "along x and y", ...
_IN_PLANE = (r"(?i)\b(ab|a-b|xy|x-y)[- ]plane|two-dimensional|\b2-?D\b|in-plane|layer|"
             r"\b(along|in|on)\s+(the\s+)?(a|x)(\s*-?\s*axis)?\s*(and|&|/)\s*(the\s+)?(b|y)\b")


_SERIES_INTRO = ("This VASP MD run of Li in a sulfide host is one of {n} runs at different temperatures; "
                 "their folders ({folders}) are in the current folder. ")


# ------------------------------------------------------------------ tasks
STO, SI, PT, CU, SIMD, CHG, VOID, CUZR = ("POSCAR.SrTiO3", "Si.cif", "Pt_nanoparticle.xyz", "Cu_md.extxyz",
                                          "XDATCAR.Si", "CHGCAR.Si", "Cu_void.lammpstrj", "CuZr.data")

# ------------------------------------------------------------------ research: long requests, graded by subtask
def ref_li_axis(axis):
    """Li tracer D along ``axis`` (cm^2/s) of the Li6PS5Cl run, default fit window."""
    def ref(pipe):
        from ..verbs import msd

        a = msd(pipe, time_per_frame=0.1, by_axis=True).attributes
        return a[f"MeanSquaredDisplacement.diffusion_coefficient.Li.{axis}"] * 1e-4
    return ref


def ref_carbon_sp3(pipe, density="3.0"):
    """Percent of 4-coordinated C in the last frame of the 3.0 g/cm^3 quench (cutoff 1.85 A; the
    fraction is the same from 1.8 to 2.0 A)."""
    from .. import examples, load
    from ..modifiers import CoordinationAnalysis

    p = load(str(examples.fetch("carbon-synth", quiet=True) / f"density-{density}-T-3000.extxyz"))
    p.add(CoordinationAnalysis(cutoff=1.85))
    cn = np.asarray(p.compute(p.num_frames - 1).particles["Coordination"])
    return 100.0 * float(np.mean(cn == 4))


def ref_neb(pipe, which="barrier"):
    e = [pipe.compute(i).attributes["NEB.Relative Energy"] for i in range(pipe.num_frames)]
    return max(e) if which == "barrier" else e[-1]


#: A Korean answer: five Korean words in a row (the research prompts are in Korean, as the owner writes).
_KOREAN = r"(?:[가-힣]+[^가-힣\n]{1,12}){5}"


def _research_checks(figure=True, max_lines=80):
    """What every research request must leave, besides its own subtasks: no failed cell, the
    figure asked for, the answer in the user's language and no more code than the analysis needs
    (lines the model wrote; a pile of hand-written code is the cost a reviewer pays)."""
    checks = [Cells(error_cells, 0, name="cells with errors"), Matches(_KOREAN, name="answers in Korean"),
              Cells(code_lines, lambda n: n <= max_lines, name=f"code lines <= {max_lines}")]
    return checks + ([Cells(figure_cells, True, name="figure")] if figure else [])


_LI_AXIS_CELL = """\
from ewald import msd, sensitivity
t = msd(pipe, time_per_frame=0.1, by_axis=True)
a = t.attributes
print(f"fit window {a['MeanSquaredDisplacement.fit_start']:.1f}-{a['MeanSquaredDisplacement.fit_end']:.1f} ps")
for ax in "xyz":
    print(f"D_{ax} = {a[f'MeanSquaredDisplacement.diffusion_coefficient.Li.{ax}'] * 1e-4:.3g} cm^2/s")
print(sensitivity(pipe, "MeanSquaredDisplacement", "fit_range", [(0.1, 0.7), (0.2, 0.8), (0.3, 0.9)]))
t"""

_CARBON_CELL = """\
import numpy as np
import matplotlib.pyplot as plt
from ewald import examples, load
from ewald.modifiers import CoordinationAnalysis
cutoff = 1.85   # A, first minimum of the C-C g(r); the fractions are the same from 1.8 to 2.0 A
rows = {}
for f in sorted(examples.fetch("carbon-synth", quiet=True).glob("density-*-T-3000.extxyz")):
    p = load(str(f))
    p.add(CoordinationAnalysis(cutoff=cutoff))
    cn = np.asarray(p.compute(p.num_frames - 1).particles["Coordination"])
    rows[float(f.name.split("-")[1])] = [100 * np.mean(cn == k) for k in (2, 3, 4)]
rho = sorted(rows)
fig, ax = plt.subplots()
for i, label in enumerate(["sp", "sp2", "sp3"]):
    ax.plot(rho, [rows[r][i] for r in rho], "o-", label=label)
ax.set(xlabel="density (g/cm$^3$)", ylabel="fraction (%)")
ax.legend()
print(f"sp3 at 3.0 g/cm^3: {rows[3.0][2]:.1f} %")
plt.show()"""

_NEB_CELL = """\
E = [pipe.compute(i).attributes["NEB.Relative Energy"] for i in range(pipe.num_frames)]
print(f"barrier = {max(E):.3f} eV, reaction energy = {E[-1]:.4f} eV (energy(sigma->0) of each image's OUTCAR)")
show(pipe, table="NEB")"""

_LI_SITES_CELL = """\
pipe.add(ExpressionSelection("ParticleType == 'Li'"))
# sites: maxima of the time-averaged Li density, so empty sites count; a hop needs a 0.3 ps stay
pipe.add(SiteHopping(sites="density", only_selected=True, time_per_frame=0.1, site_radius=0.8, min_stay=0.3))
a = pipe.compute(0).attributes
print(f"{a['SiteHopping.sites']} sites, {a['SiteHopping.hops']} hops, "
      f"mean residence {a['SiteHopping.mean_residence_time']:.2f} ps")
show(pipe, table="residence-times")"""

#: Research requests: several subtasks in one Korean message, as a researcher writes them; graded
#: per subtask (TaskResult.score) besides pass/fail, with the analysis choices reported.
RESEARCH = [
    Task("research-li-axes", "Li6PS5Cl argyrodite AIMD 결과야 (frame 간격 0.1 ps). Li 확산이 방향에 따라 다른지 보고 "
         "싶어. x, y, z 방향 Li tracer 확산계수를 cm^2/s로 구하고 MSD 그래프를 보여줘. fit 구간을 어떻게 잡았는지, "
         "그 구간을 바꾸면 값이 얼마나 달라지는지도 알려줘.", "paper:li6ps5cl-aimd/XDATCAR.gz",
         [Step("MeanSquaredDisplacement", {"by_axis": True}),
          Number(ref_li_axis("x"), rel=0.35, index=0, name="D_x"),
          Number(ref_li_axis("y"), rel=0.35, index=1, name="D_y"),
          Number(ref_li_axis("z"), rel=0.35, index=2, name="D_z"),
          Matches(r"\d+(\.\d+)?\s*(ps)?\s*(-|–|~|to|부터|에서)\s*\d+(\.\d+)?\s*ps", name="reports the fit window"),
          Matches(r"\d+(\.\d+)?\s*%", name="says how much D moves (%)")] + _research_checks(),
         category="research", allow_code=True,
         solution=[("insert_cell", {"source": _LI_AXIS_CELL})],
         answer="x, y, z 방향 확산계수는 서로 30% 이내로 비슷해서 이 짧은 AIMD로는 방향 의존성이 있다고 보기 어렵습니다. "
                "fit 구간은 2.1-8.6 ps (lag의 20-80%)이고, 구간을 10-70%에서 30-90%로 옮기면 D가 약 15% 달라집니다.\n"
                "ANSWER: {0}, {1}, {2}"),
    Task("research-li-sites", "Li6PS5Cl argyrodite AIMD 결과야 (frame 간격 0.1 ps). Li가 site 사이를 얼마나 자주 "
         "hop하는지 알고 싶어. site는 몇 개로 잡혔고 hop은 몇 번인지, 평균 체류 시간은 몇 ps인지 알려주고 체류 시간 "
         "분포를 그려줘. site와 hop을 어떻게 정의했는지도 설명해줘.", "paper:li6ps5cl-aimd/XDATCAR.gz",
         [Step("SiteHopping"),
          Matches(r"(?i)densit|밀도|48h|24g|vacan|빈 ?site|빈 ?자리", name="says where the sites come from"),
          Matches(r"(?i)min_stay|site_radius|체류.{0,30}(이상|넘|최소)|rattl|왕복|flicker|되돌아", name="says what counts as a hop")]
         + _research_checks(max_lines=40),
         category="research", allow_code=True,
         solution=[("insert_cell", {"source": _LI_SITES_CELL})],
         answer="site는 Li의 time-averaged density 극대점에서 찾았습니다 (빈 site 포함 314개). site_radius 0.8 Å 안에 "
                "들어오고 0.3 ps 이상 머물러야 hop으로 셌기 때문에 경계에서 왕복하는 진동은 빠집니다."),
    Task("research-carbon-sp3", "같은 폴더의 density-*-T-3000.extxyz 파일들은 밀도(g/cm^3)별로 3000 K에서 quench한 "
         "비정질 탄소야. 각 밀도의 마지막 frame에서 sp, sp2, sp3 비율을 밀도에 따라 비교하는 그래프를 그려주고, "
         "3.0 g/cm^3에서 sp3 비율을 %로 알려줘. bond cutoff는 어떻게 정했는지도 말해줘.",
         "paper:carbon-synth/density-1.0-T-3000.extxyz",
         [Number(ref_carbon_sp3, tol=8.0, name="sp3 % at 3.0 g/cm^3"),
          CellNumber(ref_carbon_sp3, tol=3.0),
          Matches(r"[12]\.\d+\s*(Å|A\b|Angstrom|옹스트롬)", name="reports the bond cutoff")] + _research_checks(),
         category="research", allow_code=True,
         solution=[("insert_cell", {"source": _CARBON_CELL})],
         answer="밀도가 2.5 g/cm^3까지는 거의 모두 sp2이고, 3.0 g/cm^3부터 sp3가 많아집니다. 배위수는 C-C g(r)의 "
                "첫 극소인 1.85 Å을 cutoff로 셌고, 1.8-2.0 Å 사이에서는 비율이 바뀌지 않습니다.\nANSWER: {0}"),
    Task("research-fe-neb", "bcc Fe vacancy 이동 NEB 계산 폴더야 (image 00-04). 이동 barrier와 반응 에너지를 eV로 "
         "구하고 energy profile 그림을 보여줘. 어떤 에너지(OUTCAR의 어느 값)를 썼는지도 알려줘.",
         "paper:fe-vacancy-neb/",
         [Number(lambda p: ref_neb(p, "barrier"), tol=0.02, index=0, name="barrier"),
          Number(lambda p: ref_neb(p, "reaction"), tol=0.01, index=1, name="reaction energy"),
          Matches(r"(?i)sigma|free energy|TOTEN|자유 ?에너지|energy\s*\(", name="says which energy")]
         + _research_checks(),
         category="research", allow_code=True,
         solution=[("insert_cell", {"source": _NEB_CELL})],
         answer="각 image OUTCAR의 energy(sigma->0)를 썼습니다. 이동 barrier는 가운데 image 02에서 나오고, 처음과 끝 "
                "구조의 에너지 차이(반응 에너지)는 거의 0입니다.\nANSWER: {0}, {1}"),
]


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
    Task("recipe-compare-frames", "Compare the g(r) of the first and the last frame on one plot.", CU,
         [Cells(_compares(frames=[{0}, {9, -1}]), name="show(frame=[0, -1])"), Cells(error_cells, 0, name="cells with errors")],
         category="recipe", allow_code=True, setup=_RDF_CELLS,
         solution=[("insert_cell", {"source": "show(pipe, table='coordination-rdf', frame=[0, -1])"}),
                   ("run_cell", {"index": -1})], answer="Done."),
    Task("recipe-partials-stack", "Plot the Cu-Cu, Cu-Zr and Zr-Zr partial g(r) in separate panels, one above the "
         "other.", CUZR,
         [Cells(_compares(columns=3, layout="stack"), name="show(y=[3 partials], layout='stack')"),
          Cells(error_cells, 0, name="cells with errors")],
         category="recipe", allow_code=True,
         solution=[("add_modifier", {"name": "CoordinationAnalysis", "params": {"cutoff": 6.0, "partial": True}}),
                   ("insert_cell", {"source": "show(pipe, table='coordination-rdf', y=['Cu-Cu', 'Cu-Zr', 'Zr-Zr'], "
                                              "layout='stack')"}),
                   ("run_cell", {"index": -1})], answer="Done."),

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
    Task("li-top-msd-paths", "Draw the trajectories of the 5 Li atoms whose MSDs are largest.",
         "paper:li6ps5cl-aimd/XDATCAR.gz",
         [Step("GenerateTrajectoryLines"), Data(_traces_top_msd, name="traced Li are the 5 with the largest MSD")],
         category="paper", setup=["pipe.add(MeanSquaredDisplacement(time_per_frame=0.1, fit_range=(0.215, 1.0)))"],
         solution=[("add_modifier", {"name": "ExpressionSelection",
                                     "params": {"expression": "ParticleType == 'Li'", "top": 5, "by": "MSD"}}),
                   ("add_modifier", {"name": "GenerateTrajectoryLines"})],
         answer="Done: the paths of the five Li atoms with the largest MSD."),

    # ---------------------------------------------------------- whole studies from one sentence
    Task("study-arrhenius", _SERIES_INTRO.format(n=5, folders="T500 to T900") +
         "What is the activation energy of Li diffusion, in eV? Show the Arrhenius plot.",
         "series:li-series/T500/XDATCAR", _study_checks(_ref_ea()), category="study", allow_code=True,
         solution=[("insert_cell", {"source": _study_cell()})], answer="ANSWER: {0}"),
    Task("study-paper", _SERIES_INTRO.format(n=5, folders="T500 to T900") +
         "Work out how Li diffuses in this material and its activation energy, save the Arrhenius plot as "
         "arrhenius.png, and write the paragraph for the results section of our paper (give D in cm^2/s).",
         "series:li-series/T500/XDATCAR",
         _study_checks(_ref_ea(), figure=False) + [
             File("arrhenius.png", _png),
             Mentions(_ref_ea(), tol=0.02, name="Ea in the text"),
             # D from MSD/6t, or the in-plane D = MSD_ab/4t of a layered conductor (1.5 times larger)
             Mentions(_ref_d_cm2(700), rel=0.2, also=1.5, name="D(700 K) in cm^2/s"),
             Matches(_IN_PLANE),
             Prose(80)],
         category="study", allow_code=True,
         solution=[("insert_cell", {"source": _study_cell().removesuffix("fit") +
                                    "fit.figure().savefig('arrhenius.png')\nfit"})],
         answer=("Li diffuses in the ab plane only: the MSD grows along a and b and stays flat along c, so the "
                 "transport is two-dimensional. From five NVT runs between 500 and 900 K the Li tracer diffusion "
                 "coefficient follows an Arrhenius law with an activation energy of {1} eV; at 700 K D = {2} cm^2/s. "
                 "The diffusion coefficients come from linear fits of the time-averaged Li mean squared "
                 "displacement, and the activation energy from a straight-line fit of ln D against 1/T, shown "
                 "in the Arrhenius plot saved as arrhenius.png, which the notebook regenerates from the runs.\n"
                 "ANSWER: {0}")),
    Task("study-hot-start", _SERIES_INTRO.format(n=5, folders="T500 to T900") +
         "What is the activation energy of Li diffusion, in eV? Show the Arrhenius plot.",
         "series:li-series-hot/T500/XDATCAR",
         _study_checks(_ref_ea(skip_unequilibrated=True), tol=0.04) + [Matches(r"(?i)equilibrat|hot|overshoot|transient")],
         category="study", allow_code=True,
         solution=[("insert_cell", {"source": _study_cell(start=", start=check_run(run.pipe).equilibrated or 0")})],
         answer="T500 is not equilibrated at the start (check_run), so its first frames are left out.\nANSWER: {0}"),
    Task("study-mislabel", _SERIES_INTRO.format(n=5, folders="T500 to T900") +
         "What is the activation energy of Li diffusion, in eV? Show the Arrhenius plot.",
         "series:li-series-mislabel/T500/XDATCAR",
         _study_checks(_ref_ea(), tol=0.04) + [Matches(r"(?is)T900.{0,200}600|600.{0,200}T900")],
         category="study", allow_code=True,
         solution=[("insert_cell", {"source": _study_cell()})],
         answer="The INCAR in T900 sets TEBEG = 600 K, so that run is a second 600 K run.\nANSWER: {0}"),
    Task("study-scf", _SERIES_INTRO.format(n=5, folders="T500 to T900") +
         "What is the activation energy of Li diffusion, in eV? Show the Arrhenius plot.",
         "series:li-series-scf/T500/XDATCAR",
         _study_checks(_ref_ea(exclude=("T600",)), tol=0.03) + [Matches(r"(?is)T600|600\s*K")],
         category="study", allow_code=True,
         solution=[("insert_cell", {"source": _study_cell(
             skip="if not check_run(run.pipe).ok and run.label == 'T600':\n        continue\n    ")})],
         answer="T600 hit NELM at most ionic steps (check_run), so it is left out of the fit.\nANSWER: {0}"),
    Task("study-llzo", "This is a GROMACS MD run of the garnet LLZO (Li7La3Zr2O12) at 500 K; the runs at 600, 700 "
         "and 800 K are traj_600.xtc, traj_700.xtc and traj_800.xtc in the same folder, each with its .gro. Frames "
         "are 50.8 fs apart and the Li motion is diffusive from 10 ps on. What is the activation energy of Li "
         "diffusion, in eV? Show the Arrhenius plot.", "paper:llzo-arrhenius/traj_500.xtc",
         _study_checks(0.118, tol=0.03), category="study", allow_code=True,
         solution=[("insert_cell", {"source": (
             "from ewald import examples, load_runs, msd, arrhenius\n"
             "folder = examples.fetch('llzo-arrhenius', quiet=True)\n"
             "runs = load_runs(str(folder / 'traj_*.xtc'), label=r'traj_(\\d+)')\n"
             "d = msd(runs, time_per_frame=0.0508, fit_range=(10 / (runs.pipe.num_frames * 0.0508), 1.0))\n"
             "fit = arrhenius(d, attribute='MeanSquaredDisplacement.diffusion_coefficient.Li')\n"
             "print(f'Ea = {fit.Ea:.4f} eV')\nfit")})],
         answer="ANSWER: {0}"),
    Task("study-timestep", _SERIES_INTRO.format(n=5, folders="T500 to T900") +
         "What is the activation energy of Li diffusion, in eV? Show the Arrhenius plot.",
         "series:li-series-timestep/T500/XDATCAR", _study_checks(_ref_ea(), tol=0.04), category="study",
         allow_code=True, solution=[("insert_cell", {"source": _study_cell()})], answer="ANSWER: {0}"),
    Task("study-path", "Find the Li diffusion pathway in this VASP MD run of Li in a sulfide host: along which "
         "directions do the Li ions move, and how long is one hop, in Angstrom? Show the pathway in the 3D view.",
         "series:li-series/T900/XDATCAR",
         [Number(2.0, tol=0.3, name="hop length"),
          Matches(_IN_PLANE),
          Any(Step("GenerateTrajectoryLines"), Step("SpatialBinning"), Step("CreateIsosurface")),
          Cells(error_cells, 0, name="cells with errors")],
         category="study", allow_code=True,
         solution=[("add_modifier", {"name": "GenerateTrajectoryLines", "params": {"only_selected": False}})],
         answer="Li hops between sites 2.0 A apart along a and b only; nothing moves along c, so the pathway is "
                "a two-dimensional net in the ab plane.\nANSWER: {0}"),
    Task("study-redo", _SERIES_INTRO.format(n=5, folders="T500 to T900") +
         "What is the activation energy of Li diffusion, in eV? Show the Arrhenius plot.",
         "series:li-series/T500/XDATCAR",
         _study_checks(_ref_ea(exclude=("T500",))) + [Cells(_fit_cells, lambda n: n <= 0, name="fit cells added", since=0)],
         followups=["The 500 K run is too short to trust. Leave it out and redo the fit and the plot."],
         category="study", allow_code=True,
         solution=[("insert_cell", {"source": _study_cell()}), (ANSWER_TURN, {"text": "Done."}),
                   ("edit_cell", {"index": 2, "source": _study_cell(
                       skip="if run.label == 'T500':\n        continue\n    ")})],
         answer="Cell 3 now leaves T500 out.\nANSWER: {0}"),
    Task("study-one-temperature", "These four VASP MD runs of Li in a sulfide host (folders ENCUT300 to ENCUT600 in "
         "the current folder) are the Li diffusion data we have. Fit the activation energy of Li diffusion from them.",
         "series:li-encut/ENCUT300/XDATCAR",
         [NoAnswer(), Not(Matches(_EA_CLAIM)), Matches(r"(?i)same temperature|one temperature|single temperature|"
                                                       r"all (four |4 )?(runs )?.{0,40}600\s*K|600\s*K.{0,80}(all|each|every)")],
         category="study", allow_code=True,
         solution=[("insert_cell", {"source": "from ewald import load_runs\nruns = load_runs('*/XDATCAR')\n"
                                              "print(runs.temperatures())"})],
         answer="All four runs are at 600 K (TEBEG = TEEND = 600 in every INCAR); they test the cutoff, so "
                "they cannot give an activation energy. Runs at three or more temperatures are needed."),
    Task("study-no-diffusion", "Compute the Li diffusion coefficient in cm^2/s from this VASP MD run of Li in a "
         "sulfide host, and its activation energy.", "series:li-solid/T300/XDATCAR",
         [Not(Matches(_EA_CLAIM)), Matches(r"(?i)no (long-range )?(li )?diffusion|not diffus|does not diffuse|"
                                           r"doesn't diffuse|vibrat|plateau|no hops|rattl|localized|stays? (on|at|in)")],
         category="study", allow_code=True,
         solution=[("insert_cell", {"source": "from ewald import msd\nmsd(pipe)"})],
         answer="The Li MSD plateaus at about 0.07 A^2: Li only vibrates around its sites and does not diffuse "
                "in this run, so there is no diffusion coefficient to fit, and one temperature cannot give an "
                "activation energy."),
] + RESEARCH

#: Task ids by category.
CATEGORIES = sorted({t.category for t in TASKS})

__all__ = ["CATEGORIES", "RESEARCH", "TASKS"]
