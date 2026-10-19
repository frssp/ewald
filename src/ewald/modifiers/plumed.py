"""PLUMED collective variables in the pipeline (optional ``plumed`` package and PLUMED library).

``Plumed`` runs a ``plumed.dat`` over the trajectory through PLUMED's MD-engine interface (see
:mod:`ewald.plumed`); ``LoadColvar`` puts an existing COLVAR file next to the frames;
``FreeEnergySurface`` sums a metadynamics HILLS file into a free energy surface. Their
per-frame values are attributes (``Plumed.d``, ``Colvar.d``), so ``TimeSeries``, expressions
and the Claude tools see them, and the whole series is a table for the Plots tab.
"""
from __future__ import annotations

import os

import numpy as np

from .. import plumed as _plumed
from .trajectory import _TrajectoryModifier
from ..pipeline import Modifier

TEMPLATE = """# PLUMED input: one action per line, distances in nm, atoms numbered from 1
d: DISTANCE ATOMS=1,2
"""


def _stamp(path) -> tuple:
    try:
        st = os.stat(os.path.expanduser(str(path)))
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return ()


class Plumed(_TrajectoryModifier):
    """Collective variables computed by PLUMED for every frame, from a ``plumed.dat``.

    ``input`` is the PLUMED input text or the path of a ``plumed.dat`` (a one-line input may
    separate actions with ``;``). PLUMED reads the frames as an MD code would hand them over:
    positions and cell in Angstrom (converted to nm), masses from 'Mass' or the element,
    charges from 'Charge'; PLUMED atom ``i`` is the particle with 'Particle Identifier' ``i``
    (the LAMMPS numbering) when ``by_identifier`` and identifiers exist, else the i-th particle.
    Frame ``k`` is MD step ``k`` and ``time_per_frame`` ps after the previous one, as in
    ``plumed driver``; ``temperature`` (K) sets kBT for METAD and other biases.

    PLUMED runs once over frames ``start:stop:step`` (it is stateful: METAD deposits hills,
    averages accumulate) and the result is cached. Every value of the input's labels becomes the
    attribute ``Plumed.<name>`` of its frame (``Plumed.d``, ``Plumed.d.x``, ``Plumed.metad.bias``,
    PLUMED units: nm, rad, kJ/mol) and a column of ``tables['plumed']`` (with ``Frame``); a label
    with one value per atom (e.g. ``COORDINATIONNUMBER``) becomes the particle property
    ``Plumed <label>``; ``Plumed.bias`` is the total bias. Relative files in the input
    (``REFERENCE=``, ``INCLUDE``) are read from the directory of the ``plumed.dat`` (or the
    current one); PLUMED's own output goes to a scratch directory unless ``workdir`` is given.
    ``kernel`` is the path of ``libplumedKernel`` if ``PLUMED_KERNEL`` is not set.
    """

    multiline_parameters = ("input",)

    def __init__(self, input: str = TEMPLATE, start: int = 0, stop: int | None = None, step: int = 1,
                 time_per_frame: float = 1.0, temperature: float = 0.0, by_identifier: bool = True,
                 workdir: str = "", kernel: str = ""):
        self.input = input
        self.start = start
        self.stop = stop
        self.step = step
        self.time_per_frame = time_per_frame
        self.temperature = temperature
        self.by_identifier = by_identifier
        self.workdir = workdir
        self.kernel = kernel

    def _text(self) -> str:
        text = str(self.input or "")
        if not text.strip():
            raise ValueError("give a PLUMED input: plumed.dat text or the path of a plumed.dat")
        if "\n" not in text and ";" in text and not os.path.isfile(os.path.expanduser(text.strip())):
            text = "\n".join(part.strip() for part in text.split(";")) + "\n"
        return text

    def modify(self, data, ctx):
        text = self._text()
        stamp = _stamp(text.strip()) if "\n" not in text.strip() else ()
        key = (ctx.upstream_key(), repr(self), stamp)
        cache = self.__dict__.get("_result")
        if cache is None or cache[0] != key:
            cache = (key, self._compute(ctx, text))
            self._result = cache
        frames, res = cache[1]
        table = {"Frame": np.asarray(frames)}
        table.update(res["values"])
        table["Bias"] = res["bias"]
        data.tables["plumed"] = table
        if ctx.frame not in frames:
            return
        row = frames.index(ctx.frame)
        for name, col in res["values"].items():
            data.attributes[f"Plumed.{name}"] = float(col[row])
        data.attributes["Plumed.bias"] = float(res["bias"][row])
        for label, values in res["per_atom"].items():
            if values.shape[1] == data.count:
                data.particles[f"Plumed {label}"] = np.array(values[row])

    def _compute(self, ctx, text):
        frames = list(self._frames(ctx))
        res = _plumed.run(text, (ctx.input_frame(k) for k in frames), steps=frames,
                          timestep=self.time_per_frame * self.step, temperature=self.temperature,
                          by_identifier=self.by_identifier, workdir=self.workdir or None,
                          kernel=self.kernel or None)
        return frames, res


class LoadColvar(Modifier):
    """Values from a PLUMED COLVAR file (or any ``#! FIELDS`` file) attached to the frames.

    ``align="index"``: frame ``k`` takes row ``offset + k * stride`` (``stride`` = COLVAR rows per
    stored frame). ``align="time"``: frame ``k`` takes the row whose ``time`` is nearest to the
    frame's time, ``Timestep * md_timestep`` when the frame has a 'Timestep' attribute and
    ``md_timestep`` (ps) is given, else ``k * time_per_frame``; rows further than half a COLVAR
    interval away count as missing. Each field becomes the attribute ``Colvar.<field>`` (NaN
    for a missing row) and the whole file is ``tables['colvar']``.
    """

    def __init__(self, file: str = "COLVAR", align: str = "index", stride: int = 1, offset: int = 0,
                 time_per_frame: float = 1.0, md_timestep: float = 0.0):
        self.file = file
        self.align = align
        self.stride = stride
        self.offset = offset
        self.time_per_frame = time_per_frame
        self.md_timestep = md_timestep

    def _table(self):
        path = os.path.expanduser(str(self.file))
        key = (path, _stamp(path))
        if self.__dict__.get("_cache", (None,))[0] != key:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"COLVAR file not found: {path}")
            self._cache = (key, _plumed.read_colvar(path))
        return self._cache[1]

    def row(self, frame: int, data=None) -> int | None:
        """The row of the file that belongs to ``frame``, or None."""
        table = self._table()
        n = len(next(iter(table.values()))) if table else 0
        if self.align == "index":
            r = int(self.offset + frame * self.stride)
            return r if 0 <= r < n else None
        if self.align != "time":
            raise ValueError(f"align must be 'index' or 'time', not {self.align!r}")
        if "time" not in table:
            raise ValueError(f"{self.file} has no 'time' column; use align='index'")
        t = table["time"]
        if data is not None and self.md_timestep > 0 and "Timestep" in data.attributes:
            when = float(data.attributes["Timestep"]) * self.md_timestep
        else:
            when = frame * self.time_per_frame
        r = int(np.argmin(np.abs(t - when)))
        spacing = np.median(np.diff(t)) if len(t) > 1 else np.inf
        return r if abs(t[r] - when) <= 0.5 * spacing + 1e-9 * max(1.0, abs(when)) else None

    def modify(self, data, ctx):
        table = self._table()
        data.tables["colvar"] = dict(table)
        r = self.row(ctx.frame, data)
        for name, col in table.items():
            data.attributes[f"Colvar.{name}"] = float(col[r]) if r is not None else float("nan")


class FreeEnergySurface(Modifier):
    """Free energy surface from a metadynamics HILLS file, F(s) = -sum of the hills.

    Sums the hills on a grid exactly as ``plumed sum_hills`` does (PLUMED's stretched-Gaussian
    kernels, periodic CVs, well-tempered heights), for one or two CVs, or runs ``plumed
    sum_hills`` itself when ``use_plumed`` (any number of CVs, multivariate hills). ``bins`` per
    CV, ``ranges`` ``[(min, max), ...]`` or None for the periodic domain / hill extent.
    ``up_to_frame`` uses only the hills deposited up to the current frame's time (``time`` of
    HILLS against ``frame * time_per_frame``), so stepping through the trajectory shows the FES
    converge. Writes ``tables['fes']`` (CV columns and ``Free Energy`` in kJ/mol, minimum at 0)
    and the attributes ``FreeEnergySurface.hills`` and ``FreeEnergySurface.max``.
    """

    def __init__(self, hills: str = "HILLS", bins: int = 100, ranges=None, use_plumed: bool = False,
                 up_to_frame: bool = False, time_per_frame: float = 1.0):
        self.hills = hills
        self.bins = bins
        self.ranges = ranges
        self.use_plumed = use_plumed
        self.up_to_frame = up_to_frame
        self.time_per_frame = time_per_frame

    def surface(self, frame: int | None = None) -> dict:
        """The FES dict of :func:`ewald.plumed.fes_from_hills` (cached per file and parameters)."""
        path = os.path.expanduser(str(self.hills))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"HILLS file not found: {path}")
        hills = self._hills(path)
        n = len(hills)
        if self.up_to_frame and frame is not None and hills.time is not None:
            n = int(np.searchsorted(hills.time, frame * self.time_per_frame, side="right"))
        key = (path, _stamp(path), repr(self), n)
        if self.__dict__.get("_fes", (None,))[0] != key:
            if self.use_plumed:
                if n != len(hills):
                    raise ValueError("up_to_frame needs use_plumed=False")
                fes = _plumed.sum_hills(path, self.bins, self.ranges)
            else:
                fes = _plumed.fes_from_hills(hills, self.bins, self.ranges, max_hills=max(n, 1))
            fes["hills"] = n
            self._fes = (key, fes)
        return self._fes[1]

    def _hills(self, path):
        key = (path, _stamp(path))
        if self.__dict__.get("_hills_cache", (None,))[0] != key:
            self._hills_cache = (key, _plumed.read_hills(path))
        return self._hills_cache[1]

    def modify(self, data, ctx):
        fes = self.surface(ctx.frame)
        data.tables["fes"] = dict(fes["table"])
        data.attributes["FreeEnergySurface.hills"] = fes["hills"]
        data.attributes["FreeEnergySurface.max"] = float(np.nanmax(fes["fes"]))
