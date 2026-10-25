"""What a run is, read from its input files, and a health check of its output.

A trajectory file alone does not say how it was made: the time between frames, the ensemble,
the target temperature and whether the run converged live in the input (INCAR, ``in.lammps``)
and in the log files next to it. :func:`run_info` finds them::

    from ewald import load, run_info, check_run
    pipe = load("md/XDATCAR")
    run_info(pipe)          # <RunInfo VASP md NVT, 300 K, 1 fs steps, 0.001 ps per frame>
    check_run(pipe)         # converged? equilibrated? energy drift? close contacts?

Each code's files are read by its :mod:`ewald.codes` entry; the built-in ones:
VASP: ``INCAR`` next to the file (else the parameters echoed at the top of OUTCAR or in
vasprun.xml's ``<incar>``), ``KPOINTS``, ``OSZICAR``, ``OUTCAR``. LAMMPS: ``log.lammps``, which
echoes every input command, else an input file (``in.*``, ``*.in``). Files on SSH hosts work the
same way. Nothing here runs code from the files; they are only parsed.

:func:`run_info` gives :class:`RunInfo`; ``time_per_frame`` (ps) is what the trajectory
analyses (:func:`ewald.msd`, :func:`ewald.vacf`) then use by default. :func:`check_run` gives
:class:`RunCheck`, a list of findings with what each one rests on, for people and for the AI.
"""
from __future__ import annotations

import glob as _glob
import os
import posixpath
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from .io import remote
from .io.base import head_lines, open_binary

__all__ = ["RunInfo", "RunCheck", "Finding", "run_info", "check_run", "read_incar", "read_lammps_input",
           "detect_equilibration", "statistical_inefficiency"]

#: LAMMPS ``units`` style -> (default timestep, picoseconds per time unit); lj is reduced (None)
LAMMPS_UNITS = {"lj": (0.005, None), "real": (1.0, 1e-3), "metal": (0.001, 1.0), "si": (1e-8, 1e12),
                "cgs": (1e-8, 1e12), "electron": (0.001, 1e-3), "micro": (2.0, 1e6), "nano": (0.00045, 1e3)}


@dataclass
class RunInfo:
    """How a run was made, as far as its input files say. ``None`` means not known.

    ``code`` names the :mod:`ewald.codes` entry ("vasp", "lammps" or a plugin's code);
    ``kind`` "md", "relax", "static", "neb", "phonon";
    ``ensemble`` "NVE", "NVT", "NPT", "NPH"; ``timestep`` is the MD step in fs;
    ``time_per_frame`` the time between stored frames in ps; ``temperature`` the target
    (start, end) in K; ``steps`` the number of ionic/MD steps asked for; ``units`` the LAMMPS
    units style; ``parameters`` the raw input values that were read; ``files`` where they came
    from; ``notes`` how each value was inferred.
    """

    code: str | None = None
    kind: str | None = None
    ensemble: str | None = None
    timestep: float | None = None
    time_per_frame: float | None = None
    temperature: tuple | None = None
    pressure: float | None = None
    steps: int | None = None
    units: str | None = None
    kpoints: str | None = None
    parameters: dict = field(default_factory=dict)
    files: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """One line: code, kind, ensemble, temperature, time step and time per frame."""
        parts = [p for p in ((self.code or "").upper(), self.kind, self.ensemble) if p]
        if self.temperature:
            t0, t1 = self.temperature
            parts.append(f"{t0:g} K" if t0 == t1 else f"{t0:g}->{t1:g} K")
        if self.timestep:
            parts.append(f"{self.timestep:g} fs steps")
        if self.time_per_frame:
            parts.append(f"{self.time_per_frame:g} ps per frame")
        return " ".join(parts[:3]) + ("" if len(parts) <= 3 else ", " + ", ".join(parts[3:])) if parts else "unknown run"

    def __repr__(self):
        return f"<RunInfo {self.summary()}>"


# ----------------------------------------------------------------------- input files
def _parse_value(text: str):
    t = text.strip().split()[0] if text.strip() else ""
    if t.upper() in (".TRUE.", "T", ".T."):
        return True
    if t.upper() in (".FALSE.", "F", ".F."):
        return False
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t.replace("d", "e").replace("D", "E"))
    except ValueError:
        return text.strip()


def read_incar(path) -> dict:
    """INCAR tags as ``{TAG: value}`` (numbers converted; ``#``/``!`` comments and ``;`` separated
    tags handled)."""
    out = {}
    with open_binary(path) as fh:
        text = fh.read().decode("utf-8", "replace")
    for line in text.splitlines():
        line = re.split(r"[#!]", line, maxsplit=1)[0]
        for part in line.split(";"):
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip().upper()
                if re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
                    out[key] = _parse_value(value)
    return out


_OUTCAR_TAGS = ("IBRION", "NSW", "POTIM", "NBLOCK", "TEBEG", "TEEND", "SMASS", "MDALGO", "ISIF", "EDIFF",
                "EDIFFG", "NELM", "ISPIN", "ENCUT", "IMAGES", "PSTRESS", "ANDERSEN_PROB", "LANGEVIN_GAMMA")


def _incar_from_outcar(path) -> dict:
    """The INCAR values VASP echoes near the top of an OUTCAR (``POTIM  =   1.0000``)."""
    out = {}
    pat = re.compile(r"\b(" + "|".join(_OUTCAR_TAGS) + r")\s*=\s*([-+.0-9EeDd]+)")
    for raw in head_lines(path, 4000):
        line = raw.decode("utf-8", "replace")
        if "Iteration" in line:  # the electronic steps start; the echo is over
            break
        for key, value in pat.findall(line):
            out.setdefault(key, _parse_value(value))
    return out


def _incar_from_vasprun(path) -> dict:
    out, inside = {}, False
    for raw in head_lines(path, 1000):
        line = raw.decode("utf-8", "replace")
        if "<incar>" in line:
            inside = True
        elif "</incar>" in line:
            break
        elif inside:
            m = re.search(r'name="(\w+)">\s*([^<]*)<', line)
            if m:
                out[m.group(1).upper()] = _parse_value(m.group(2))
    return out


def read_lammps_input(path) -> dict:
    """The commands that matter for timing and ensembles from a LAMMPS input or log file (a log
    echoes every command): ``units``, ``timestep``, ``fixes`` ``[(style, args)]``, ``dump_every``,
    ``runs`` (steps of each ``run``), ``minimize``, ``neb``, plus ``warnings``/``errors`` counts
    and the first error line."""
    out = {"fixes": [], "runs": [], "minimize": False, "neb": False, "warnings": 0, "errors": 0}
    with open_binary(path) as fh:
        for raw in fh:
            line = raw.decode("utf-8", "replace").split("#", 1)[0].strip()
            if not line:
                continue
            tok = line.split()
            word = tok[0]
            if word.startswith("WARNING"):
                out["warnings"] += 1
            elif word.startswith("ERROR"):
                out["errors"] += 1
                out.setdefault("error", line)
            elif word == "units" and len(tok) > 1:
                out["units"] = tok[1]
            elif word == "timestep" and len(tok) > 1:
                v = _parse_value(tok[1])
                if isinstance(v, (int, float)):
                    out["timestep"] = float(v)
            elif word == "fix" and len(tok) > 3:
                out["fixes"].append((tok[3], tok[4:]))
            elif word == "unfix" and len(tok) > 1:
                pass  # kept simple: a fix that was undone still says what the run was meant to be
            elif word == "dump" and len(tok) > 4 and tok[4].isdigit():
                out.setdefault("dump_every", int(tok[4]))
            elif word == "run" and len(tok) > 1 and tok[1].isdigit():
                out["runs"].append(int(tok[1]))
            elif word == "minimize":
                out["minimize"] = True
            elif word in ("neb", "neb/spin"):
                out["neb"] = True
    return out


# ----------------------------------------------------------------------- finding files
def _folder_file(path: str, name: str) -> str | None:
    """``name`` in the folder of ``path`` if it exists there (locally or on the same host)."""
    path = str(path)
    if remote.is_remote(path):
        rp = remote.parse_remote(path)
        cand = str(rp.with_path(posixpath.join(posixpath.dirname(rp.path), name)))
        try:
            return cand if remote.exists(cand) else None
        except OSError:
            return None
    cand = os.path.join(os.path.dirname(os.path.abspath(path)), name)
    return cand if os.path.isfile(cand) else None


def _folder_glob(path: str, patterns) -> list[str]:
    path = str(path)
    if remote.is_remote(path):
        rp = remote.parse_remote(path)
        found = []
        for pat in patterns:
            try:
                found += remote.glob(str(rp.with_path(posixpath.join(posixpath.dirname(rp.path), pat))))
            except OSError:
                pass
        return found
    folder = os.path.dirname(os.path.abspath(path))
    return sorted({f for pat in patterns for f in _glob.glob(os.path.join(folder, pat)) if os.path.isfile(f)})


def _source_paths(source) -> list[str]:
    """The files of a pipeline (or a source, or a path)."""
    if isinstance(source, (str, os.PathLike)):
        return [str(source)]
    src = getattr(source, "source", source)
    paths = getattr(src, "paths", None)
    if paths:
        return [str(p) for p in paths]
    path = getattr(src, "path", None)
    return [str(path)] if path else []


def _basename(path: str) -> str:
    rp = remote.parse_remote(path) if remote.is_remote(path) else None
    return posixpath.basename(rp.path) if rp else os.path.basename(path)


def _code_of(path: str):
    """The :class:`ewald.codes.SimulationCode` of ``path``: by the file's name or content, else by
    the files next to it (an INCAR, a log.lammps, another code's markers)."""
    from . import codes

    return codes.code_of(path) or codes.code_for_folder(path, _folder_file)


# ----------------------------------------------------------------------- run_info
def run_info(source, refresh: bool = False) -> RunInfo:
    """:class:`RunInfo` of a pipeline (or a file path): read once, then cached on the pipeline
    (``refresh=True`` reads the files again)."""
    cached = getattr(source, "_run_info", None)
    if cached is not None and not refresh:
        return cached
    paths = _source_paths(source)
    info = RunInfo()
    if paths:
        path = paths[0]
        code = _code_of(path)
        fill = code.hook("run_info") if code is not None else None
        if code is not None:
            info.code = code.name
        try:
            if fill is not None:
                fill(info, path)
        except (OSError, ValueError, UnicodeError) as exc:
            info.notes.append(f"could not read the run's input files: {exc}")
        if hasattr(source, "compute"):
            with _compute_lock(source):  # reading frames runs numba's parallel parser
                _frame_timing(info, source, path)
    if hasattr(source, "compute"):
        try:
            source._run_info = info
        except AttributeError:
            pass
    return info


def _vasp_info(info: RunInfo, path: str):
    info.code = "vasp"
    incar_path = _folder_file(path, "INCAR")
    params, origin = {}, None
    if incar_path:
        params, origin = read_incar(incar_path), incar_path
    else:
        for name, reader in (("OUTCAR", _incar_from_outcar), ("vasprun.xml", _incar_from_vasprun)):
            cand = path if _basename(path).upper().startswith(name.upper()) else _folder_file(path, name)
            if cand:
                params = reader(cand)
                if params:
                    origin = cand
                    info.notes.append(f"no INCAR next to the file; parameters as echoed in {_basename(cand)}")
                    break
    if origin:
        info.files.append(origin)
    info.parameters = params
    kp = _folder_file(path, "KPOINTS")
    if kp:
        info.files.append(kp)
        lines = [ln.decode("utf-8", "replace").strip() for ln in head_lines(kp, 5)]
        if len(lines) >= 4:
            info.kpoints = f"{lines[2]} {lines[3]}".strip()
    if not params:
        return
    nsw = params.get("NSW")
    ibrion = params.get("IBRION", 0 if nsw else -1)  # VASP's default: MD when NSW > 0
    info.steps = int(nsw) if isinstance(nsw, (int, float)) and nsw else None
    if isinstance(params.get("IMAGES"), int) and params["IMAGES"] > 0:
        info.kind = "neb"
    elif nsw == 0 or ibrion == -1:
        info.kind = "static"
    elif ibrion == 0:
        info.kind = "md"
    elif ibrion in (1, 2, 3):
        info.kind = "relax"
    elif ibrion in (5, 6, 7, 8):
        info.kind = "phonon"
    if info.kind == "md":
        potim = params.get("POTIM", 0.5)
        info.timestep = float(potim)
        if "POTIM" not in params:
            info.notes.append("POTIM not set: VASP's default 0.5 fs assumed")
        tebeg = params.get("TEBEG")
        if isinstance(tebeg, (int, float)):
            info.temperature = (float(tebeg), float(params.get("TEEND", tebeg)))
        info.ensemble = _vasp_ensemble(params)
        if info.ensemble == "NPT" and isinstance(params.get("PSTRESS"), (int, float)):
            info.pressure = float(params["PSTRESS"]) / 10.0  # kB -> GPa


def _vasp_ensemble(p: dict) -> str | None:
    mdalgo = p.get("MDALGO", 0)
    isif = p.get("ISIF", 0)
    if mdalgo in (0, None):
        smass = p.get("SMASS", -3)
        return "NVE" if smass == -3 else "NVT"
    if mdalgo == 1:
        return "NVE" if p.get("ANDERSEN_PROB", 0) == 0 else "NVT"
    if mdalgo == 3:
        return "NPT" if isif == 3 else "NVT"
    if mdalgo in (2, 4, 5, 13):
        return "NVT"
    return None


def _lammps_info(info: RunInfo, path: str):
    info.code = "lammps"
    source = _folder_file(path, "log.lammps")
    if source is None:
        cands = _folder_glob(path, ("in.*", "*.in", "input*.lmp"))
        source = cands[0] if cands else None
    if source is None:
        info.notes.append("no log.lammps or input file next to the dump; timing taken from LAMMPS defaults")
        return
    info.files.append(source)
    p = read_lammps_input(source)
    info.parameters = {k: v for k, v in p.items() if k != "fixes"}
    info.parameters["fixes"] = [" ".join([style, *args]) for style, args in p["fixes"]]
    units = p.get("units", "lj")
    info.units = units
    default_dt, ps_per_unit = LAMMPS_UNITS.get(units, (None, None))
    dt = p.get("timestep", default_dt)
    if "timestep" not in p and default_dt is not None:
        info.notes.append(f"no timestep command: the {units} default {default_dt:g} assumed")
    if dt is not None and ps_per_unit is not None:
        info.timestep = dt * ps_per_unit * 1000.0  # fs
    elif units == "lj":
        info.notes.append("lj units: times are in reduced units (tau), not ps")
    if p["runs"]:
        info.steps = int(sum(p["runs"]))
    styles = [s for s, _ in p["fixes"]]
    if p["neb"]:
        info.kind = "neb"
    elif p["minimize"] and not p["runs"]:
        info.kind = "relax"
    elif p["runs"]:
        info.kind = "md"
    thermostat = None
    for style, args in p["fixes"]:
        base = style.split("/")[0]
        if base in ("npt", "nvt", "nph") or style.startswith(("rigid/nvt", "rigid/npt")):
            info.ensemble = base.upper() if base in ("npt", "nvt", "nph") else style.split("/")[1].upper()
            if "temp" in args:
                k = args.index("temp")
                thermostat = args[k + 1:k + 3]
            for key in ("iso", "aniso", "tri", "x"):
                if key in args:
                    k = args.index(key)
                    try:
                        info.pressure = float(args[k + 1])
                    except (ValueError, IndexError):
                        pass
                    break
        elif style in ("langevin", "temp/berendsen", "temp/rescale", "temp/csvr", "temp/csld", "gld"):
            thermostat = args[:2] if style != "temp/rescale" else args[1:3]
    if info.ensemble is None and "nve" in [s.split("/")[0] for s in styles]:
        info.ensemble = "NVT" if thermostat else "NVE"
    if thermostat and len(thermostat) == 2:
        try:
            info.temperature = (float(thermostat[0]), float(thermostat[1]))
        except ValueError:
            pass
    if units == "lj" and info.temperature is not None:
        info.notes.append("lj units: the target temperature is in reduced units")
    if info.pressure is not None and units in ("metal",):
        info.pressure = info.pressure * 1e-4  # bar -> GPa
    elif info.pressure is not None and units == "real":
        info.pressure = info.pressure * 1.01325e-4  # atm -> GPa


def _frame_timing(info: RunInfo, pipe, path: str):
    """Time between stored frames from the frames' ``Timestep`` attributes and the time step."""
    try:
        n = pipe.num_frames
    except Exception:  # noqa: BLE001 - a broken source has no timing
        return
    if info.kind not in ("md", None) or n < 1:
        return
    steps = None
    if n >= 2:
        try:
            a = pipe.source_frame(0).attributes.get("Timestep")
            b = pipe.source_frame(1).attributes.get("Timestep")
            if a is not None and b is not None and b > a:
                steps = int(b) - int(a)
        except Exception:  # noqa: BLE001
            steps = None
    code = _registered(info.code)
    hook = code.hook("frame_steps") if code is not None else None
    if hook is not None:
        steps = hook(info, path, steps)
    if steps is None:
        steps = 1
    if info.timestep is not None and info.kind == "md":
        info.time_per_frame = steps * info.timestep / 1000.0
        info.notes.append(f"{steps} MD step(s) of {info.timestep:g} fs between stored frames")
    elif info.units == "lj" and info.kind == "md":   # LAMMPS reduced units
        dt = info.parameters.get("timestep", LAMMPS_UNITS["lj"][0])
        info.parameters["time_per_frame_tau"] = steps * dt


def _vasp_frame_steps(info: RunInfo, path: str, steps):
    if _basename(path).upper().startswith("XDATCAR"):
        nblock = info.parameters.get("NBLOCK", 1)
        nblock = int(nblock) if isinstance(nblock, (int, float)) and nblock > 0 else 1
        if steps is None or steps == 1:
            steps = nblock  # XDATCAR numbers its configurations, one per NBLOCK steps
    return steps


def _lammps_frame_steps(info: RunInfo, path: str, steps):
    return info.parameters.get("dump_every") if steps is None else steps


def _registered(name):
    """The registered code ``name`` (None when unknown or None)."""
    if not name:
        return None
    from . import codes

    return codes.codes().get(name)


# ----------------------------------------------------------------------- statistics
def _autocorrelation(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0] if acf[0] > 0 else np.zeros(n)


def statistical_inefficiency(x) -> float:
    """g = 1 + 2 sum (1 - t/N) C(t) up to the first negative C(t): the number of samples per
    independent one (Chodera et al. 2007; ``N/g`` effective samples)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return 1.0
    c = _autocorrelation(x)
    g = 1.0
    for t in range(1, n - 1):
        if c[t] <= 0 and t > 3:
            break
        g += 2.0 * c[t] * (1.0 - t / n)
    return max(1.0, g)


def detect_equilibration(x, candidates: int = 60) -> tuple[int, float, float]:
    """Start of the equilibrated part of a time series: the ``t0`` that maximizes the number of
    effective samples ``(N - t0) / g`` of ``x[t0:]`` (Chodera, JCTC 12, 1799, 2016). Returns
    ``(t0, g, n_eff)``; ``t0`` is tried at ``candidates`` points over the first half."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return 0, 1.0, float(n)
    best = (0, 1.0, -1.0)
    for t0 in np.unique(np.linspace(0, n // 2, min(candidates, n // 2 + 1)).astype(int)):
        g = statistical_inefficiency(x[t0:])
        neff = (n - t0) / g
        if neff > best[2]:
            best = (int(t0), g, neff)
    return best


# ----------------------------------------------------------------------- check_run
@dataclass
class Finding:
    """One result of :func:`check_run`: ``level`` "ok", "info", "warning" or "error"."""

    level: str
    title: str
    detail: str = ""

    def __str__(self):
        mark = {"ok": "ok  ", "info": "info", "warning": "WARN", "error": "FAIL"}.get(self.level, self.level)
        return f"[{mark}] {self.title}" + (f": {self.detail}" if self.detail else "")


@dataclass
class RunCheck:
    """Findings of :func:`check_run` with the :class:`RunInfo` they rest on; ``equilibrated``
    is the first equilibrated frame of an MD run (None when not checked)."""

    info: RunInfo
    findings: list = field(default_factory=list)
    equilibrated: int | None = None
    num_frames: int = 0

    @property
    def ok(self) -> bool:
        """No warnings or errors."""
        return all(f.level in ("ok", "info") for f in self.findings)

    @property
    def warnings(self) -> list:
        return [f for f in self.findings if f.level in ("warning", "error")]

    def to_dict(self) -> dict:
        return {"run": self.info.to_dict(), "ok": self.ok, "equilibrated_from_frame": self.equilibrated,
                "num_frames": self.num_frames, "findings": [asdict(f) for f in self.findings]}

    def __str__(self):
        head = f"Run check: {self.info.summary()}, {self.num_frames} frame(s)"
        return "\n".join([head] + [f"  {f}" for f in self.findings])

    def __repr__(self):
        return str(self)

    def _repr_markdown_(self):
        icons = {"ok": "✓", "info": "·", "warning": "⚠", "error": "✗"}
        lines = [f"**Run check**: {self.info.summary()}, {self.num_frames} frame(s)", ""]
        lines += [f"- {icons.get(f.level, '')} **{f.title}**" + (f": {f.detail}" if f.detail else "")
                  for f in self.findings]
        return "\n".join(lines)


def check_run(pipe, close_contact: float = 0.5) -> RunCheck:
    """Check a run (a :class:`~ewald.pipeline.Pipeline` from :func:`ewald.load`) for the usual
    problems: files still being written, SCF steps that hit NELM, a relaxation that did not reach
    EDIFFG, MD that is not equilibrated (Chodera's method on the potential energy), temperature off
    the thermostat target, energy drift where energy is conserved, LAMMPS warnings and errors, and
    atoms closer than ``close_contact`` times the sum of their covalent radii in the first and last
    frame. Reads files and frames only; changes nothing."""
    with _compute_lock(pipe):  # the numba kernels below must not run beside the GUI's compute thread
        return _check_run(pipe, close_contact)


def _compute_lock(pipe):
    import contextlib

    lock = getattr(pipe, "_compute_lock", None)
    return lock if lock is not None else contextlib.nullcontext()


def _check_run(pipe, close_contact: float) -> RunCheck:
    info = run_info(pipe)
    check = RunCheck(info)
    add = check.findings.append
    src = getattr(pipe, "source", None)
    try:
        check.num_frames = pipe.num_frames
    except Exception:  # noqa: BLE001
        check.num_frames = 0
    add(Finding("info", "Run", info.summary() + (f"; {info.kpoints}" if info.kpoints else "")))
    for note in info.notes:
        add(Finding("info", "Assumed", note))
    incomplete = list(getattr(src, "incomplete", []) or [])
    if incomplete:
        add(Finding("warning", "Files still being written or cut off", "; ".join(incomplete)))
    paths = _source_paths(pipe)
    path = paths[0] if paths else None
    thermo = getattr(getattr(src, "thermo", None), "table", None)
    code = _registered(info.code)
    hook = code.hook("check") if code is not None else None
    if hook is not None:
        try:
            hook(check, info, path)
        except (OSError, ValueError, UnicodeError) as exc:
            add(Finding("info", f"{code.title} checks skipped", str(exc)))
    if thermo is not None and info.kind == "md":
        _check_md(check, info, thermo, pipe)
    if check.num_frames:
        frames = sorted({0, check.num_frames - 1})
        for k in frames:
            try:
                _check_contacts(check, pipe.source_frame(k), k, close_contact)
            except Exception as exc:  # noqa: BLE001 - the check is advice; never fail on it
                add(Finding("info", f"Close contacts not checked at frame {k}", str(exc)))
    if not check.warnings:
        add(Finding("ok", "No problems found", "in the checks above"))
    return check


def _check_lammps(check: RunCheck, info: RunInfo, path: str):
    add = check.findings.append
    p = info.parameters
    if p.get("errors"):
        add(Finding("error", "LAMMPS stopped with an error", str(p.get("error", ""))))
    if p.get("warnings"):
        add(Finding("warning", f"{p['warnings']} LAMMPS warning(s) in the log", "see log.lammps"))


def _check_vasp(check: RunCheck, info: RunInfo, path: str):
    add = check.findings.append
    if not path:
        return
    osz = path if _basename(path).upper().startswith("OSZICAR") else _folder_file(path, "OSZICAR")
    nelm = info.parameters.get("NELM", 60)
    if osz:
        counts = _scf_counts(osz)
        if counts:
            hit = [k + 1 for k, c in enumerate(counts) if c >= nelm]
            if hit:
                shown = ", ".join(map(str, hit[:8])) + (" ..." if len(hit) > 8 else "")
                add(Finding("warning", f"{len(hit)} of {len(counts)} ionic steps hit NELM = {nelm}",
                            f"SCF not converged at step(s) {shown} (OSZICAR)"))
            else:
                add(Finding("ok", "SCF converged at every ionic step",
                            f"at most {max(counts)} of NELM = {nelm} iterations (OSZICAR)"))
    if info.kind == "relax":
        outcar = path if _basename(path).upper().startswith("OUTCAR") else _folder_file(path, "OUTCAR")
        if outcar:
            done = _tail_contains(outcar, b"reached required accuracy")
            if done:
                add(Finding("ok", "Relaxation reached EDIFFG", "OUTCAR: reached required accuracy"))
            else:
                add(Finding("warning", "Relaxation did not reach EDIFFG",
                            f"no 'reached required accuracy' in OUTCAR (NSW = {info.steps})"))


def _scf_counts(path) -> list[int]:
    """Electronic iterations of each ionic step in an OSZICAR."""
    counts, last = [], 0
    with open_binary(path) as fh:
        for raw in fh:
            tok = raw.split()
            if not tok:
                continue
            if tok[0].endswith(b":") and len(tok) > 1 and tok[1].isdigit():
                last = int(tok[1])
            elif tok[0].isdigit() and len(tok) > 1 and tok[1].startswith((b"F=", b"T=")):  # an ionic step
                counts.append(last)
                last = 0
    return counts


def _tail_contains(path, needle: bytes, size: int = 1 << 20) -> bool:
    with open_binary(path) as fh:
        try:
            fh.seek(0, 2)
            end = fh.tell()
            fh.seek(max(0, end - size))
        except (OSError, ValueError):
            pass
        return needle in fh.read()


def _column(table: dict, *names):
    for n in names:
        if n in table:
            col = np.asarray(table[n], float)
            if np.isfinite(col).sum() >= 2:
                return n, col
    return None, None


def _check_md(check: RunCheck, info: RunInfo, table: dict, pipe):
    add = check.findings.append
    steps = np.asarray(table["Step"], float)
    name, pot = _column(table, "PotEng", "Energy", "Free Energy")
    if name is not None and np.isfinite(pot).sum() < 20:
        add(Finding("info", "Equilibration not checked", f"only {int(np.isfinite(pot).sum())} {name} values"))
        name = None
    if name is not None:
        ok = np.isfinite(pot)
        t0, g, neff = detect_equilibration(pot[ok])
        s0 = steps[ok][t0]
        frac = t0 / max(ok.sum(), 1)
        frame = _frame_of_step(pipe, info, s0)
        check.equilibrated = frame
        where = f"step {int(s0)}" + (f" (frame {frame})" if frame is not None else "")
        if frac > 0.45:
            add(Finding("warning", "Not clearly equilibrated",
                        f"{name} still drifts: the equilibrated part would start at {where}, "
                        f"{frac:.0%} into the run; {neff:.0f} independent samples"))
        else:
            add(Finding("ok", f"Equilibrated from {where}",
                        f"{name}: {frac:.0%} of the run discarded, {neff:.0f} independent samples "
                        f"(statistical inefficiency {g:.1f})"))
    else:
        t0 = 0
    tname, temp = _column(table, "Temp", "Temperature")
    if tname is not None and info.temperature is not None and info.ensemble in ("NVT", "NPT"):
        target = info.temperature[1]
        sel = temp[t0:] if name is not None else temp
        sel = sel[np.isfinite(sel)]
        if len(sel):
            mean, sd = float(sel.mean()), float(sel.std())
            off = abs(mean - target)
            level = "warning" if target > 0 and off > max(0.05 * target, 2 * sd / np.sqrt(max(len(sel), 1)) + 10) else "ok"
            add(Finding(level, "Temperature " + ("off the target" if level == "warning" else "at the target"),
                        f"mean {mean:.1f} ± {sd:.1f} K over the equilibrated part, target {target:g} K"))
    conserved = None
    if info.code == "vasp":
        conserved = _column(table, "Total Energy")  # E= in OSZICAR includes the thermostat
    elif info.ensemble == "NVE":
        conserved = _column(table, "TotEng", "Econserve")
    if conserved and conserved[0] is not None:
        cname, e = conserved
        ok = np.isfinite(e)
        n_atoms = _atom_count(pipe)
        dt_fs = info.timestep
        if ok.sum() >= 3 and n_atoms and dt_fs:
            t_ps = steps[ok] * dt_fs / 1000.0
            slope = np.polyfit(t_ps, e[ok], 1)[0] / n_atoms * 1000.0  # meV/atom/ps
            level = "warning" if abs(slope) > 1.0 else "ok"
            add(Finding(level, "Energy drift " + ("too large" if level == "warning" else "small"),
                        f"{cname}: {slope:+.3g} meV/atom/ps (limit 1)"))


def _frame_of_step(pipe, info: RunInfo, step) -> int | None:
    """The stored frame at MD step ``step`` (the first one at or after it)."""
    try:
        n = pipe.num_frames
        paths = _source_paths(pipe)
        if info.code == "vasp" and paths and _basename(paths[0]).upper().startswith("XDATCAR"):
            # XDATCAR numbers its configurations; the thermo table counts ionic steps
            if info.time_per_frame and info.timestep:
                per = info.time_per_frame * 1000.0 / info.timestep
                return int(min(max(np.ceil(step / per) - 1, 0), n - 1))
            return int(min(max(step - 1, 0), n - 1))
        a = pipe.source_frame(0).attributes.get("Timestep")
        b = pipe.source_frame(1).attributes.get("Timestep") if n > 1 else None
        if a is None:
            return None
        if b is None or b <= a:
            return 0
        return int(min(max(np.ceil((step - a) / (b - a)), 0), n - 1))
    except Exception:  # noqa: BLE001
        return None


def _atom_count(pipe) -> int | None:
    try:
        return pipe.source_frame(0).count
    except Exception:  # noqa: BLE001
        return None


def _check_contacts(check: RunCheck, data, frame: int, factor: float):
    from .neighbors import find_pairs
    from .radii import element_of, element_radius

    if data.count < 2:
        return
    types = np.asarray(data.types).astype(str)
    radius_of = {}
    for t in np.unique(types):
        sym = element_of(t)
        r = element_radius(sym, "covalent") if sym else None
        radius_of[t] = float(r) if r else 0.75
    rmax = max(radius_of.values())
    cutoff = 2 * rmax * factor
    if data.cell.is_valid and data.cell.pbc.any():
        cutoff = min(cutoff, 0.99 * float(np.min(data.cell.perpendicular_widths()[data.cell.pbc])) / 2)
    i, j, _d, dist = find_pairs(np.asarray(data["Position"]), data.cell, cutoff, half=True)
    i, j, dist = np.asarray(i), np.asarray(j), np.asarray(dist)
    if len(dist):
        ri = np.array([radius_of[t] for t in types[i]])
        rj = np.array([radius_of[t] for t in types[j]])
        bad = dist < factor * (ri + rj)
        if bad.any():
            k = int(np.argmin(dist / (ri + rj)))
            add = check.findings.append
            add(Finding("error", f"Atoms too close at frame {frame}",
                        f"{int(bad.sum())} pair(s) under {factor:g} x the covalent radius sum; closest: atoms "
                        f"{int(i[k])} and {int(j[k])} at {dist[k]:.2f} Å"))
            return
    check.findings.append(Finding("ok", f"No close contacts at frame {frame}",
                                  f"no pair under {factor:g} x the covalent radius sum"))
