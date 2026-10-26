"""Synthetic VASP MD series for the AI eval's ``study`` tasks (:mod:`ewald.llm.eval_tasks`).

Each series is a set of run folders (``T500/``, ``T600/``, ...) with INCAR, KPOINTS, OSZICAR and
XDATCAR, written into the task's working folder before the model starts. Li atoms hop between
empty sites in the ab plane at a rate set by ``D0 exp(-Ea / kT)``, among S atoms that only
vibrate, so the answer
is known; some series carry a trap a careful analyst catches (a run that starts hot, a folder
whose name disagrees with its INCAR, a run whose SCF never converged, runs that are all at one
temperature, Li that never leaves its site, runs that store frames at different intervals). Everything is seeded, so a series is the same on
every machine; the graders compute their reference values from the written files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KB = 8.617333262e-5   # eV/K


@dataclass
class RunSpec:
    """One run of a series: its folder, the INCAR temperature and the temperature its atoms
    actually move at (``moves_at``, default the INCAR one), and traps."""

    folder: str
    temperature: float
    moves_at: float | None = None
    hot_frames: int = 0           # the first frames run hot (thermostat overshoot), atoms faster
    scf_unconverged: bool = False  # most ionic steps hit NELM; the atoms move erratically
    diffusive: bool = True        # False: Li only vibrates around its site
    potim: float | None = None    # fs per MD step (default the series')
    nblock: int | None = None     # MD steps between stored frames (default the series')


@dataclass
class Series:
    """A series of runs and how its atoms move (``ea`` in eV, ``d0`` in A^2/ps)."""

    runs: list
    ea: float = 0.30
    d0: float = 60.0
    frames: int = 1000
    potim: float = 2.0
    nblock: int = 5
    n_li: int = 64
    a: float = 2.0              # Li site spacing (A)
    grid: int = 8               # sites per edge; the cell is grid * a
    hops: str = "xy"            # Li hops along these axes only (a layered conductor)
    seed: int = 0
    notes: dict = field(default_factory=dict)

    @property
    def cell(self) -> float:
        return self.grid * self.a

    @property
    def n_s(self) -> int:
        return (self.grid // 2) ** 3

    @property
    def time_per_frame(self) -> float:
        """ps between stored frames."""
        return self.potim * self.nblock / 1000.0

    def steps(self, spec: RunSpec) -> tuple:
        """``(potim, nblock, ps per frame)`` of run ``spec``."""
        potim, nblock = spec.potim or self.potim, spec.nblock or self.nblock
        return potim, nblock, potim * nblock / 1000.0


def _ladder(*temperatures, **kw) -> list:
    return [RunSpec(f"T{t}", float(t), **kw) for t in temperatures]


#: The series the tasks use, by name.
SERIES = {
    # clean: five temperatures, Ea = 0.30 eV
    "li-series": Series(_ladder(500, 600, 700, 800, 900)),
    # T500 starts 120 frames hot (2000 K overshoot); check_run flags it, the fit must skip it
    "li-series-hot": Series([RunSpec("T500", 500.0, hot_frames=120)] + _ladder(600, 700, 800, 900), seed=1),
    # folder T900 holds a copy of the 600 K input: its INCAR (and its atoms) are at 600 K
    "li-series-mislabel": Series(_ladder(500, 600, 700, 800) + [RunSpec("T900", 600.0)], seed=2),
    # T600 never converged its SCF: its D is off the trend and check_run says why
    "li-series-scf": Series(_ladder(500) + [RunSpec("T600", 600.0, scf_unconverged=True)] + _ladder(700, 800, 900),
                            seed=3),
    # the runs store frames 10, 10, 10, 20 and 50 fs apart (POTIM and NBLOCK differ)
    "li-series-timestep": Series([RunSpec("T500", 500.0), RunSpec("T600", 600.0),
                                  RunSpec("T700", 700.0, potim=1.0, nblock=10), RunSpec("T800", 800.0, nblock=10),
                                  RunSpec("T900", 900.0, potim=1.0, nblock=50)], seed=6),
    # a convergence test: four runs, all at 600 K, named by cutoff
    "li-encut": Series([RunSpec(f"ENCUT{e}", 600.0) for e in (300, 400, 500, 600)], seed=4),
    # Li vibrates only: no diffusion to fit
    "li-solid": Series([RunSpec("T300", 300.0, diffusive=False)], seed=5),
}


def _trajectory(series: Series, spec: RunSpec, rng):
    """Li hopping between the empty sites of a simple cubic grid (spacing ``a``; hops along the
    axes in ``series.hops`` only), S on the centres of every other cube; both vibrate."""
    a, n = series.a, series.grid
    frames, dt = series.frames, series.steps(spec)[2]
    t_move = spec.moves_at or spec.temperature
    # rate per frame of one Li trying a hop, so that a lone Li has D = d0 exp(-Ea/kT)
    dims = len(series.hops)
    def p_hop(t):
        return min(0.5, 2 * dims * series.d0 * np.exp(-series.ea / (KB * t)) * dt / a ** 2)
    p = np.full(frames, p_hop(t_move))
    p[:spec.hot_frames] = p_hop(2000.0)
    if spec.scf_unconverged:
        p *= 10.0
    occupied = np.zeros((n, n, n), bool)
    flat = rng.choice(n ** 3, size=series.n_li, replace=False)
    site = np.stack(np.unravel_index(flat, (n, n, n)), axis=1)
    occupied[tuple(site.T)] = True
    image = np.zeros_like(site)            # lattice vectors crossed, so the path is unwrapped
    axes = ["xyz".index(c) for c in series.hops]
    li = np.empty((frames, series.n_li, 3))
    for f in range(frames):
        if spec.diffusive and f > 0:
            for i in np.flatnonzero(rng.random(series.n_li) < p[f]):
                step = np.zeros(3, int)
                step[axes[rng.integers(len(axes))]] = rng.choice((-1, 1))
                new = site[i] + step
                wrapped = new % n
                if not occupied[tuple(wrapped)]:
                    occupied[tuple(site[i])] = False
                    occupied[tuple(wrapped)] = True
                    image[i] += (new - wrapped) // n
                    site[i] = wrapped
        li[f] = (site + image * n) * a
    amp = 0.12 * np.sqrt(spec.temperature / 500.0)
    li += rng.normal(0, amp, size=li.shape)
    ns = n // 2
    cubes = np.array([(i, j, k) for i in range(ns) for j in range(ns) for k in range(ns)], float)
    s0 = (2 * cubes + 0.5) * a
    s = s0 + rng.normal(0, 0.7 * amp, size=(frames, len(s0), 3))
    return li, s


def write_run(folder: Path, series: Series, spec: RunSpec, rng) -> Path:
    """Write one run of ``series`` into ``folder``; returns its XDATCAR."""
    folder.mkdir(parents=True, exist_ok=True)
    t_set = spec.temperature
    potim, nblock, _ = series.steps(spec)
    frames = series.frames
    nsteps = frames * nblock
    (folder / "INCAR").write_text(
        "SYSTEM = Li walk in S\n"
        f"IBRION = 0 ; NSW = {nsteps}\nPOTIM = {potim:g}\nNBLOCK = {nblock}\n"
        f"TEBEG = {t_set:g} ; TEEND = {t_set:g}\n"
        "MDALGO = 2 ! Nose-Hoover\nSMASS = 1.0\nNELM = 40\nEDIFF = 1E-5\n", encoding="utf-8", newline="\n")
    (folder / "KPOINTS").write_text("Gamma only\n0\nGamma\n1 1 1\n", encoding="utf-8", newline="\n")

    hot_steps = spec.hot_frames * nblock
    lines = []
    for step in range(1, nsteps + 1):
        hot = step <= hot_steps
        nscf = 40 if spec.scf_unconverged and rng.random() < 0.8 else 12 + int(rng.integers(0, 4))
        lines += [f"DAV: {k:3d}    -0.1E+02   -0.1E-02   -0.1E-01   48   0.1E+00" for k in range(1, nscf + 1)]
        t = (2000.0 if hot else t_set) * (1 + rng.normal(0, 0.05))
        e0 = -150.0 + (4.0 if hot else 0.0) + rng.normal(0, 0.02)
        lines.append(f"{step:5d} T= {t:8.1f} E= {e0 + 1.5:.8E} F= {e0:.8E} E0= {e0:.8E}  "
                     f"EK= 0.15E+01 SP= 0.0E+00 SK= 0.0E+00")
    (folder / "OSZICAR").write_text("       N       E                     dE\n" + "\n".join(lines) + "\n",
                                    encoding="utf-8", newline="\n")

    li, s = _trajectory(series, spec, rng)
    pos = np.concatenate([li, s], axis=1)

    L = series.cell
    out = ["Li walk in S", "           1", f"  {L:.6f} 0 0", f"  0 {L:.6f} 0", f"  0 0 {L:.6f}",
           "   Li   S", f"   {series.n_li}   {series.n_s}"]
    for k in range(frames):
        out.append(f"Direct configuration= {k + 1:5d}")
        out += [" ".join(f"{x:.8f}" for x in row) for row in (pos[k] / L) % 1.0]
    path = folder / "XDATCAR"
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    return path


def write_series(name: str, where) -> Path:
    """Write series ``name`` (:data:`SERIES`) into folder ``where``; returns ``where``."""
    series = SERIES[name]
    where = Path(where)
    rng = np.random.default_rng(series.seed)
    for spec in series.runs:
        write_run(where / spec.folder, series, spec, rng)
    return where


__all__ = ["KB", "RunSpec", "SERIES", "Series", "write_run", "write_series"]
