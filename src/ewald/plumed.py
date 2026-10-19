"""PLUMED: collective variables on ewald trajectories, and PLUMED's own output files.

Three parts, usable without the GUI:

- :class:`PlumedEngine` drives the PLUMED library through its official Python binding
  (``import plumed``) the way an MD code does: ewald's positions, cell, masses and charges go
  in through the MD-engine interface, one frame per step, exactly as ``plumed driver`` would
  read them. :func:`run` evaluates a ``plumed.dat`` over a frame range and returns every value
  it computes; the :class:`~ewald.modifiers.Plumed` modifier is built on it.
- :func:`read_colvar` and :func:`read_hills` parse PLUMED's column files (``#! FIELDS``
  header, ``#! SET`` constants, restarts that repeat the header).
- :func:`fes_from_hills` sums metadynamics hills on a grid like ``plumed sum_hills`` (the same
  stretched-Gaussian kernels, periodic CVs and well-tempered rescaling); :func:`sum_hills` runs
  the ``plumed sum_hills`` tool itself and :func:`read_fes` reads its output; :func:`plot_fes`
  draws a 1D curve or a 2D map.

Units: ewald works in Angstrom, ps and amu; PLUMED's are nm, ps, amu and kJ/mol. The engine
declares Angstrom as the MD length unit, so PLUMED converts (``setMDLengthUnits 0.1``), and
the values it returns are in PLUMED units (distances in nm), as in a COLVAR file.

The ``plumed`` package and the PLUMED kernel library are optional. The kernel is found through
``PLUMED_KERNEL`` (``conda install -c conda-forge plumed py-plumed`` sets it; for a source
build point it at ``<prefix>/lib/libplumedKernel.so``), or pass ``kernel=``.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

#: Boltzmann constant in kJ/(mol K), PLUMED's energy unit per kelvin.
KB_KJMOL = 0.0083144626181532

# PLUMED's Gaussian truncation (src/tools/Tools.h): kernels are cut at 0.5*r^2 = 6.25 and
# "stretched" so that they reach zero there.
_DP2CUTOFF = 6.25
_STRETCH_A = 1.00193418799744762399
_STRETCH_B = -.00193418799744762399


# ----------------------------------------------------------------------------- the library
class PlumedNotFound(RuntimeError):
    """The ``plumed`` package or the PLUMED kernel library is missing."""


def available(kernel: str | None = None) -> bool:
    """True when the PLUMED Python binding imports and its kernel library loads."""
    try:
        _new_plumed(kernel)
    except Exception:
        return False
    return True


def _new_plumed(kernel: str | None = None):
    try:
        import plumed
    except ImportError as exc:
        raise PlumedNotFound("PLUMED support needs the optional `plumed` package and the PLUMED library: "
                          "conda install -c conda-forge plumed py-plumed, or pip install plumed and set "
                          "PLUMED_KERNEL to <prefix>/lib/libplumedKernel.so") from exc
    kernel = kernel or None
    try:
        p = plumed.Plumed(kernel=kernel) if kernel else plumed.Plumed()
        version = np.zeros(1, dtype=np.intc)
        p.cmd("getApiVersion", version)
    except Exception as exc:
        raise PlumedNotFound(f"the PLUMED kernel library could not be loaded ({exc}); set PLUMED_KERNEL to "
                           "<prefix>/lib/libplumedKernel.so or pass kernel=") from exc
    if version[0] < 1:
        raise PlumedNotFound("the PLUMED kernel library could not be loaded; set PLUMED_KERNEL to "
                           "<prefix>/lib/libplumedKernel.so or pass kernel=")
    return p, int(version[0])


def api_version(kernel: str | None = None) -> int:
    """API version of the loaded PLUMED kernel (10 for PLUMED 2.10; per-atom values need >= 6)."""
    p, api = _new_plumed(kernel)
    p.finalize()
    return api


def plumed_executable() -> str | None:
    """Path of the ``plumed`` command line tool (next to the kernel, or on PATH)."""
    kernel = os.environ.get("PLUMED_KERNEL")
    if kernel:
        exe = Path(kernel).resolve().parent.parent / "bin" / "plumed"
        if exe.exists():
            return str(exe)
    return shutil.which("plumed")


def _masses(data) -> np.ndarray:
    if "Mass" in data:
        return np.asarray(data["Mass"], dtype=float)
    from ase.data import atomic_masses, atomic_numbers

    table = {}
    out = np.empty(data.count)
    for i, t in enumerate(data.types):
        t = str(t)
        if t not in table:
            z = atomic_numbers.get(t) or atomic_numbers.get(t.capitalize())
            table[t] = float(atomic_masses[z]) if z else 1.0
        out[i] = table[t]
    return out


class PlumedEngine:
    """One PLUMED instance fed ewald frames through the MD-engine interface.

    ``input`` is the text of a ``plumed.dat``. PLUMED runs in ``workdir`` (its output files,
    ``plumed.log`` and any relative ``FILE=`` / ``REFERENCE=`` paths are there). Atom ``i`` of
    PLUMED (1-based, as in ``ATOMS=1,2``) is particle ``i`` of the frame, or the particle with
    'Particle Identifier' ``i`` when ``by_identifier`` (the LAMMPS convention). ``timestep`` is
    the time between frames in ps; frame ``k`` is MD step ``k``, as in ``plumed driver``.
    ``temperature`` (K) sets kBT for METAD and friends.

    >>> eng = PlumedEngine("d: DISTANCE ATOMS=1,2", natoms=n)
    >>> eng.step(data, 0); eng.value("d")
    """

    def __init__(self, input: str, natoms: int, *, workdir: str | os.PathLike = ".", timestep: float = 1.0,
                 temperature: float = 0.0, kernel: str | None = None, log: str = "plumed.log",
                 grab: tuple = ()):
        self.natoms = int(natoms)
        self.workdir = str(workdir)
        self._p, self.api = _new_plumed(kernel)
        p = self._p
        self._cwd = os.getcwd()
        os.chdir(self.workdir)   # PLUMED opens files relative to the working directory
        try:
            p.cmd("setMDEngine", "ewald")
            p.cmd("setNatoms", self.natoms)
            p.cmd("setMDLengthUnits", 0.1)   # ewald length unit: 1 A = 0.1 nm
            p.cmd("setMDTimeUnits", 1.0)     # ps
            p.cmd("setMDEnergyUnits", 1.0)   # kJ/mol (only used for the bias PLUMED returns)
            p.cmd("setMDMassUnits", 1.0)     # amu
            p.cmd("setMDChargeUnits", 1.0)   # e
            p.cmd("setTimestep", float(timestep))
            if temperature and temperature > 0:
                p.cmd("setKbT", KB_KJMOL * float(temperature))
            p.cmd("setLogFile", log)
            p.cmd("setNoVirial")
            p.cmd("init")
            p.cmd("readInputLines", input)
        finally:
            os.chdir(self._cwd)
        self._grab = {}
        for name in grab:
            self._add_grab(name)
        self._pos = np.zeros((self.natoms, 3))
        self._forces = np.zeros((self.natoms, 3))
        self._masses = np.ones(self.natoms)
        self._charges = np.zeros(self.natoms)
        self._box = np.zeros((3, 3))
        self._bias = np.zeros(1)

    def _shape(self, name: str):
        """Rank and shape of value ``name`` (adds PLUMED's GET action for it; API >= 6)."""
        if self.api < 6:
            raise RuntimeError(f"reading values directly needs PLUMED 2.10 or later (API {self.api})")
        os.chdir(self.workdir)
        try:
            rank = np.zeros(1, dtype=np.int64)   # PLUMED checks the type: these must be 8-byte ints
            self._p.cmd("getDataRank " + name, rank)
            shape = np.zeros(max(int(rank[0]), 1), dtype=np.int64)
            if rank[0] > 0:
                self._p.cmd("getDataShape " + name, shape)
        finally:
            os.chdir(self._cwd)
        return int(rank[0]), tuple(int(v) for v in shape[:int(rank[0])])

    def _add_grab(self, name: str):
        """Register a buffer that receives value ``name`` at every step."""
        rank, shape = self._shape(name)
        buf = np.zeros(shape if rank > 0 else (1,))
        self._p.cmd("setMemoryForData " + name, buf)   # PLUMED writes here at every calc: keep it alive
        self._grab[name] = buf

    def step(self, data, step: int, order: np.ndarray | None = None):
        """Hand frame ``data`` to PLUMED as MD step ``step`` and evaluate.

        ``order[i]`` is the PLUMED atom index (0-based) of particle ``i``; None means the same.
        """
        n = data.count
        idx = np.arange(n) if order is None else np.asarray(order)
        if n and idx.max() >= self.natoms:
            raise ValueError(f"frame has a particle for PLUMED atom {idx.max() + 1} but PLUMED was set up for "
                             f"{self.natoms} atoms (the particle count must not grow)")
        pos = np.zeros((self.natoms, 3))
        pos[idx] = data.positions
        masses = np.ones(self.natoms)
        masses[idx] = _masses(data)
        charges = np.zeros(self.natoms)
        if "Charge" in data:
            charges[idx] = np.asarray(data["Charge"], dtype=float)
        self._pos, self._masses, self._charges = pos, masses, charges
        self._forces = np.zeros((self.natoms, 3))
        p = self._p
        os.chdir(self.workdir)
        try:
            p.cmd("setStep", int(step))
            if data.cell.is_valid and np.any(data.cell.pbc):
                self._box = np.ascontiguousarray(data.cell.matrix, dtype=float)   # rows a, b, c in A
                p.cmd("setBox", self._box)
            p.cmd("setPositions", self._pos)
            p.cmd("setMasses", self._masses)
            p.cmd("setCharges", self._charges)
            p.cmd("setForces", self._forces)
            p.cmd("calc")
            p.cmd("getBias", self._bias)
        finally:
            os.chdir(self._cwd)

    @property
    def bias(self) -> float:
        """Total bias of the last step, kJ/mol."""
        return float(self._bias[0])

    @property
    def forces(self) -> np.ndarray:
        """Bias forces of the last step on each PLUMED atom, kJ/(mol A)."""
        return self._forces.copy()

    def value(self, name: str):
        """Value ``name`` of the last step (must be in ``grab``): a float or an array."""
        buf = self._grab[name]
        return float(buf[0]) if buf.shape == (1,) else buf.copy()

    def close(self):
        if self._p is not None:
            os.chdir(self.workdir)
            try:
                self._p.finalize()
            finally:
                os.chdir(self._cwd)
            self._p = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ----------------------------------------------------------------------------- input text
_COMMENT = re.compile(r"#.*$")


def input_text(input: str) -> tuple[str, Path | None]:
    """The text of ``input`` (a ``plumed.dat`` path, or the input itself) and its directory."""
    s = str(input)
    if "\n" not in s and len(s) < 4096 and s.strip() and os.path.isfile(os.path.expanduser(s.strip())):
        path = Path(os.path.expanduser(s.strip())).resolve()
        return path.read_text(), path.parent
    return s, None


def _logical_lines(text: str) -> list[str]:
    """Input lines without comments, with ``...`` continuations joined."""
    out, buf = [], None
    for raw in text.splitlines():
        line = _COMMENT.sub("", raw).strip()
        if buf is not None:
            if line.startswith("..."):
                out.append(buf)
                buf = None
            else:
                buf += " " + line
            continue
        if not line:
            continue
        if line.endswith("..."):
            buf = line[:-3]
        else:
            out.append(line)
    if buf is not None:
        out.append(buf)
    return out


def labels(text: str, base: Path | None = None) -> list[str]:
    """Labels defined in a PLUMED input (``d: DISTANCE ...`` or ``LABEL=d``), in order,
    including those of ``INCLUDE FILE=`` files (relative to ``base``)."""
    out = []
    for line in _logical_lines(_with_includes(text, base or Path.cwd())):
        m = re.match(r"^([A-Za-z_][\w.-]*):\s", line)
        label = m.group(1) if m else None
        if label is None:
            m = re.search(r"(?:^|\s)LABEL=(\S+)", line)
            label = m.group(1) if m else None
        if label and label not in out:
            out.append(label)
    return out


def _is_include(words) -> bool:
    return any(w.upper() == "INCLUDE" for w in words[:2])


def _with_includes(text: str, base: Path, depth: int = 0) -> str:
    if depth > 8:
        return text
    lines = []
    for line in _logical_lines(text):
        words = line.split()
        if _is_include(words):
            m = re.search(r"FILE=(\S+)", line)
            path = base / m.group(1) if m else None
            if path is not None and path.is_file():
                lines.append(_with_includes(path.read_text(), base, depth + 1))
                continue
        lines.append(line)
    return "\n".join(lines)


def untrusted_input_problems(input: str) -> list[str]:
    """Reasons not to run ``input`` when it comes from Claude or a session file rather than the
    user: ``LOAD`` loads (or compiles) native code into the process, and file keys with an
    absolute or ``..`` path read or write outside the input's folder (``PRINT FILE=~/.bashrc``)."""
    text, base = input_text(input)
    problems = []
    for line in _logical_lines(_with_includes(text, base or Path.cwd())):
        words = line.split()
        action = words[1] if words and words[0].endswith(":") and len(words) > 1 else (words[0] if words else "")
        if action.upper() == "LOAD":
            problems.append(f"{line!r}: LOAD runs native code")
        for m in re.finditer(r"(?:^|\s)(\w*FILE\w*|INCLUDE|REFERENCE\w*|STRUCTURE|MOLINFO|PDB)=(\S+)", line, re.I):
            value = os.path.expanduser(m.group(2))
            if os.path.isabs(value) or ".." in Path(value).parts:
                problems.append(f"{m.group(1)}={m.group(2)}: only files next to the input are allowed")
    return problems


_FILE_KEYS = re.compile(r"(?:^|\s)(?:FILE|REFERENCE|STRUCTURE|MOLINFO|TEMPLATE|GRID_RFILE|RFILE|PDB|"
                        r"INCLUDE|COEFFS|BASIS|REFERENCE_\w+)=(\S+)", re.IGNORECASE)


def _referenced_files(text: str, base: Path) -> list[Path]:
    """Existing files that ``text`` names through FILE=, REFERENCE=, ... relative to ``base``,
    following INCLUDE files."""
    found, todo = [], [text]
    while todo:
        for line in _logical_lines(todo.pop()):
            for value in _FILE_KEYS.findall(line):
                for name in value.split(","):
                    path = (base / name) if not os.path.isabs(name) else Path(name)
                    if path.is_file() and path not in found and not os.path.isabs(name):
                        found.append(path)
                        if _is_include(line.split()):
                            todo.append(path.read_text())
    return found


# ----------------------------------------------------------------------------- run
def run(input: str, frames, *, steps=None, timestep: float = 1.0, temperature: float = 0.0,
        by_identifier: bool = True, per_atom: bool = True, workdir: str | os.PathLike | None = None,
        kernel: str | None = None) -> dict:
    """Evaluate a PLUMED input over ``frames`` (an iterable of DataCollection).

    ``input`` is ``plumed.dat`` text or a path to one. Without ``workdir`` PLUMED runs in a new
    temporary directory into which the files the input refers to (``REFERENCE=ref.pdb``,
    ``INCLUDE FILE=``, a HILLS file to restart from, ...) are copied, so the input's own
    ``PRINT`` / ``METAD`` output never overwrites files next to it. ``steps[k]`` is the MD step
    of frame ``k`` (default ``k``).

    Returns ``{"values": {name: array (frames,)}, "per_atom": {label: array (frames, N)},
    "bias": array, "labels": [...], "workdir": path or None, "log": text}``. ``values`` holds
    every scalar of the input's labels, components included (``d.x``, ``m.bias``), in PLUMED
    units, as ``PRINT`` would write them; a label whose value is one number per atom (such as
    ``COORDINATIONNUMBER``) goes to ``per_atom`` instead, in particle order.
    """
    text, base = input_text(input)
    base = base or Path.cwd()
    frames = iter(frames)
    try:
        first = next(frames)
    except StopIteration:
        raise ValueError("no frames to evaluate") from None
    order, natoms = _atom_order(first, by_identifier)
    scalars, vectors = _probe(text, base, natoms, timestep, temperature, kernel)
    if not per_atom:
        scalars, vectors = scalars + vectors, []
    tmp = None
    if workdir is None:
        workdir = tmp = _scratch(text, base, "ewald-plumed-")
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    colvar = workdir / "ewald-plumed-values.dat"
    if colvar.exists():
        colvar.unlink()
    lines = text.rstrip("\n") + "\n"
    if scalars:
        # every value of the input's labels at every step, at full precision
        lines += f"PRINT ARG={','.join(n + '.*' for n in scalars)} STRIDE=1 FILE={colvar.name} FMT=%.17g\n"
    try:
        engine = PlumedEngine(lines, natoms, workdir=workdir, timestep=timestep, temperature=temperature,
                              kernel=kernel, grab=tuple(vectors))
    except PlumedNotFound:
        raise
    except Exception as exc:
        raise RuntimeError(f"PLUMED could not read the input: {exc}") from exc
    per_atom_out = {name: [] for name in vectors}
    bias = []
    k = 0
    try:
        for data in _chain(first, frames):
            if data.count != first.count:
                raise ValueError(f"frame {k} has {data.count} particles, the first one {first.count}; PLUMED "
                                 "needs a fixed particle count")
            o = _atom_order(data, by_identifier)[0] if order is not None else None
            engine.step(data, k if steps is None else int(steps[k]), o)
            bias.append(engine.bias)
            for name in vectors:
                v = engine.value(name)
                per_atom_out[name].append(v[o] if o is not None else v)
            k += 1
    except Exception as exc:
        log = _tail(workdir / "plumed.log")
        raise RuntimeError(f"PLUMED failed at frame {k}: {exc}" + (f"\n--- plumed.log ---\n{log}" if log else "")) from exc
    finally:
        engine.close()
    values = {}
    if scalars and colvar.exists():
        values = {key: col for key, col in read_colvar(colvar).items() if key != "time"}
        colvar.unlink()
    out = {"values": values, "per_atom": {n: np.asarray(v) for n, v in per_atom_out.items()},
           "bias": np.asarray(bias), "labels": scalars + vectors, "workdir": None if tmp else str(workdir),
           "log": _tail(workdir / "plumed.log", 400)}
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _scratch(text: str, base: Path, prefix: str) -> str:
    """A new temporary directory holding copies of the files ``text`` refers to."""
    tmp = tempfile.mkdtemp(prefix=prefix)
    for path in _referenced_files(text, base):
        dest = Path(tmp) / path.relative_to(base)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    return tmp


def _probe(text, base, natoms, timestep, temperature, kernel):
    """Which labels of the input carry values: (scalar labels, per-atom vector labels).

    Uses a throwaway PLUMED instance in a scratch directory, so labels without values
    (GROUP, MOLINFO, ...) and virtual atoms are left out without breaking the real run."""
    names = labels(text, base)
    if not names:
        return [], []
    tmp = _scratch(text, base, "ewald-plumed-probe-")
    try:
        try:
            eng = PlumedEngine(text, natoms, workdir=tmp, timestep=timestep, temperature=temperature,
                               kernel=kernel, log="probe.log")
        except PlumedNotFound:
            raise
        except Exception as exc:
            log = _tail(Path(tmp) / "probe.log")
            raise RuntimeError(f"PLUMED could not read the input: {exc}"
                               + (f"\n--- plumed.log ---\n{log}" if log else "")) from exc
        scalars, vectors = [], []
        try:
            for i, name in enumerate(names):
                try:
                    eng._p.cmd("readInputLine", f"ewald_probe{i}: PRINT ARG={name}.* FILE={tmp}/probe{i}.dat")
                except Exception:
                    continue   # no values: a group, a virtual atom, MOLINFO, ...
                try:
                    rank, shape = eng._shape(name)
                except Exception:
                    rank, shape = 0, ()   # several components: printed as scalars
                (vectors if rank == 1 and shape == (natoms,) else scalars).append(name)
        finally:
            eng.close()
        return scalars, vectors
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _chain(first, rest):
    yield first
    yield from rest


def _atom_order(data, by_identifier: bool):
    if by_identifier and "Particle Identifier" in data:
        ids = np.asarray(data["Particle Identifier"]).astype(np.int64)
        if ids.size and ids.min() >= 1 and len(np.unique(ids)) == ids.size:
            return ids - 1, int(ids.max())
    return None, data.count


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


# ----------------------------------------------------------------------------- files
def read_colvar(path) -> dict:
    """Columns of a PLUMED COLVAR-style file as ``{field: array}``.

    Reads the ``#! FIELDS`` header (repeated headers from restarts are merged; a field missing
    in part of the file is NaN there) and keeps ``#! SET`` constants in the returned dict's
    ``.constants`` attribute (a :class:`ColumnData`).
    """
    fields: list[str] = []
    order: list[str] = []
    rows: list[dict] = []
    constants: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#!"):
                words = s[2:].split()
                if words and words[0] == "FIELDS":
                    fields = words[1:]
                    for f in fields:
                        if f not in order:
                            order.append(f)
                elif words and words[0] == "SET" and len(words) >= 3:
                    constants[words[1]] = " ".join(words[2:])
                continue
            if s.startswith("#"):
                continue
            parts = s.split()
            if not fields:
                fields = [f"col{i}" for i in range(len(parts))]
                order.extend(f for f in fields if f not in order)
            if len(parts) != len(fields):
                continue   # a truncated last line of a running simulation
            rows.append(dict(zip(fields, parts)))
    out = ColumnData()
    for f in order:
        out[f] = np.array([_num(r.get(f)) for r in rows], dtype=float)
    out.constants = constants
    return out


class ColumnData(dict):
    """``{field: array}`` read from a PLUMED file, plus ``constants`` from ``#! SET`` lines."""
    constants: dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constants = {}


def _num(s) -> float:
    if s is None:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return {"pi": np.pi, "-pi": -np.pi}.get(s.lower(), np.nan)


def _const(s: str) -> float:
    """A ``#! SET`` value: a number or an expression in pi such as ``-pi`` or ``2*pi``."""
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return float(eval(s, {"__builtins__": {}}, {"pi": np.pi}))   # noqa: S307 - only names "pi"


class Hills:
    """A metadynamics HILLS file: centers, widths, heights and the CVs' periodicity."""

    def __init__(self, cvs, centers, sigmas, heights, biasf, periods, kernel="stretched-gaussian", time=None):
        self.cvs = list(cvs)
        self.centers = np.asarray(centers, float).reshape(len(heights), -1)
        self.sigmas = np.asarray(sigmas, float).reshape(len(heights), -1)
        self.heights = np.asarray(heights, float)
        self.biasf = np.asarray(biasf, float) if biasf is not None else None
        self.periods = list(periods)   # (min, max) per CV, or None when not periodic
        self.kernel = kernel
        self.time = None if time is None else np.asarray(time, float)

    def __len__(self):
        return len(self.heights)

    def __repr__(self):
        return f"Hills(cvs={self.cvs}, n={len(self)})"


def read_hills(path) -> Hills:
    """Read a HILLS file written by METAD (diagonal, non-multivariate hills)."""
    t = read_colvar(path)
    c = t.constants
    if str(c.get("multivariate", "false")).lower() == "true":
        raise ValueError("multivariate hills are not supported by fes_from_hills; use sum_hills() (plumed sum_hills)")
    cvs = [f[len("sigma_"):] for f in t if f.startswith("sigma_")]
    if not cvs:
        raise ValueError(f"{path} has no sigma_<cv> columns; is it a HILLS file?")
    periods = []
    for cv in cvs:
        lo, hi = c.get(f"min_{cv}"), c.get(f"max_{cv}")
        periods.append((_const(lo), _const(hi)) if lo is not None and hi is not None else None)
    return Hills(cvs, np.column_stack([t[cv] for cv in cvs]), np.column_stack([t[f"sigma_{cv}"] for cv in cvs]),
                 t["height"], t.get("biasf"), periods, c.get("kerneltype", "stretched-gaussian"), t.get("time"))


def fes_from_hills(hills, bins=None, ranges=None, *, min_to_zero: bool = True, stride: int | None = None,
                   max_hills: int | None = None) -> dict:
    """Free energy surface from metadynamics hills, F(s) = -sum_i G_i(s), on a grid.

    ``hills`` is a :class:`Hills` or a path. The heights written to HILLS by a well-tempered
    METAD are already rescaled by gamma/(gamma-1), so this is the FES directly, as computed by
    ``plumed sum_hills``. ``bins`` is the number of bins per CV (default 100) and ``ranges``
    ``[(min, max), ...]`` (default: the periodic domain, else the hill centers +- 3 sigma).
    A periodic CV gets ``bins`` points, a non-periodic one ``bins + 1`` (PLUMED's grids).
    ``max_hills`` uses only the first hills (the FES at an earlier time); ``stride`` also
    returns ``"history"``: the FES after every ``stride`` hills, shape (n, *grid).

    Returns ``{"cvs": names, "grid": [axis arrays], "fes": array (grid shape)}``; for one CV
    also ``table`` = ``{cv: axis, "Free Energy": fes}`` for ewald's Plots tab; for two,
    ``table`` holds the flattened grid (cv1, cv2, "Free Energy").
    """
    if not isinstance(hills, Hills):
        hills = read_hills(hills)
    ncv = len(hills.cvs)
    if max_hills is not None:
        hills = Hills(hills.cvs, hills.centers[:max_hills], hills.sigmas[:max_hills], hills.heights[:max_hills],
                      None, hills.periods, hills.kernel)
    if bins is None:
        bins = 100
    bins = [int(bins)] * ncv if np.isscalar(bins) else [int(b) for b in bins]
    axes = []
    for d in range(ncv):
        per = hills.periods[d]
        if ranges is not None and ranges[d] is not None:
            lo, hi = ranges[d]
        elif per is not None:
            lo, hi = per
        else:
            lo = float(np.min(hills.centers[:, d] - 3 * hills.sigmas[:, d]))
            hi = float(np.max(hills.centers[:, d] + 3 * hills.sigmas[:, d]))
        n = bins[d]
        # PLUMED grid: dx = (max-min)/nbin; nbin points when periodic, nbin+1 otherwise
        axes.append(lo + (hi - lo) / n * np.arange(n if per is not None else n + 1))
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([m.ravel() for m in mesh], axis=1)
    stretched = "stretched" in str(hills.kernel).lower()
    bias = np.zeros(len(points))
    history = []
    chunk = max(1, int(2e7 // max(len(points), 1)))
    nh = len(hills)
    for start in range(0, nh, chunk if stride is None else min(chunk, stride)):
        stop = min(start + (chunk if stride is None else min(chunk, stride)), nh)
        diff = points[None, :, :] - hills.centers[start:stop, None, :]
        for d, per in enumerate(hills.periods):
            if per is not None:
                period = per[1] - per[0]
                diff[..., d] -= period * np.round(diff[..., d] / period)
        r2 = np.sum((diff / hills.sigmas[start:stop, None, :]) ** 2, axis=2)
        dp = 0.5 * r2
        if stretched:
            k = np.where(dp < _DP2CUTOFF, _STRETCH_A * np.exp(-dp) + _STRETCH_B, 0.0)
        else:
            k = np.exp(-dp)
        bias += hills.heights[start:stop] @ k
        if stride is not None and (stop % stride == 0 or stop == nh):
            history.append(-bias.copy())
    shape = tuple(len(a) for a in axes)
    fes = -bias.reshape(shape)
    if min_to_zero:
        fes = fes - fes.min()
        history = [h - h.min() for h in history]
    out = {"cvs": hills.cvs, "grid": axes, "fes": fes}
    if stride is not None:
        out["history"] = np.array([h.reshape(shape) for h in history])
    out["table"] = _fes_table(hills.cvs, axes, fes)
    return out


def _fes_table(cvs, axes, fes) -> dict:
    if len(cvs) == 1:
        return {cvs[0]: axes[0], "Free Energy": fes}
    mesh = np.meshgrid(*axes, indexing="ij")
    table = {cv: m.ravel() for cv, m in zip(cvs, mesh)}
    table["Free Energy"] = fes.ravel()
    return table


def sum_hills(hills, bins=None, ranges=None, *, min_to_zero: bool = True, workdir=None) -> dict:
    """Run ``plumed sum_hills`` on a HILLS file and read the result like :func:`fes_from_hills`.

    Needs the ``plumed`` command line tool (:func:`plumed_executable`). Supports everything the
    tool does, including multivariate hills.
    """
    exe = plumed_executable()
    if exe is None:
        raise RuntimeError("the `plumed` command line tool was not found (PATH, or next to PLUMED_KERNEL)")
    path = Path(hills).resolve()
    tmp = tempfile.mkdtemp(prefix="ewald-sumhills-") if workdir is None else str(workdir)
    try:
        cmd = [exe, "sum_hills", "--hills", str(path), "--outfile", "fes.dat"]
        if bins is not None:
            cmd += ["--bin", ",".join(str(int(b)) for b in np.atleast_1d(bins))]
        if ranges is not None:
            cmd += ["--min", ",".join(repr(float(r[0])) for r in ranges),
                    "--max", ",".join(repr(float(r[1])) for r in ranges)]
        if min_to_zero:
            cmd.append("--mintozero")
        res = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"plumed sum_hills failed:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
        return read_fes(Path(tmp) / "fes.dat")
    finally:
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)


def read_fes(path) -> dict:
    """A grid file written by ``plumed sum_hills`` (``fes.dat``) as ``{"cvs", "grid", "fes", "table"}``."""
    t = read_colvar(path)
    fields = list(t)
    fcol = next((f for f in fields if "free" in f), fields[-1])
    cvs = fields[:fields.index(fcol)]
    shape = tuple(int(float(t.constants[f"nbins_{cv}"])) if f"nbins_{cv}" in t.constants
                  else len(np.unique(t[cv])) for cv in cvs)

    def grid(col):   # PLUMED writes the first CV fastest
        return np.asarray(col).reshape(shape[::-1]).T

    axes = [grid(t[cv])[(0,) * d + (slice(None),) + (0,) * (len(cvs) - d - 1)] for d, cv in enumerate(cvs)]
    fes = grid(t[fcol])
    return {"cvs": cvs, "grid": axes, "fes": fes, "table": _fes_table(cvs, axes, fes)}


def plot_fes(fes: dict, ax=None, *, levels: int = 20, cmap: str = "viridis", units: str = "kJ/mol"):
    """Draw a FES from :func:`fes_from_hills` / :func:`read_fes`: a curve for one CV, a filled
    contour map for two. Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(5, 3.6))
    cvs, axes, f = fes["cvs"], fes["grid"], np.asarray(fes["fes"])
    if len(cvs) == 1:
        ax.plot(axes[0], f, lw=1.6)
        ax.set_xlabel(cvs[0])
        ax.set_ylabel(f"Free energy ({units})")
    elif len(cvs) == 2:
        cs = ax.contourf(axes[0], axes[1], f.T, levels=levels, cmap=cmap)
        ax.contour(axes[0], axes[1], f.T, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
        ax.figure.colorbar(cs, ax=ax, label=f"Free energy ({units})")
        ax.set_xlabel(cvs[0])
        ax.set_ylabel(cvs[1])
    else:
        raise ValueError("plot_fes draws one or two CVs")
    return ax
