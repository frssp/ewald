"""Data pipeline: source -> modifiers -> DataCollection."""
from __future__ import annotations

import functools
import operator
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

from .backend import Backend, get_backend
from .data import DataCollection


#: added to numpy's "output array is read-only" when a step writes into a frame's array in place
READ_ONLY_HINT = ("the arrays of a frame are shared with the pipeline's cache, so a step must replace them "
                  "instead of writing into them: data.positions = data.positions + shift, or "
                  "pos = data.positions.copy(); pos[0] += 1; data.positions = pos")


class ComputeCancelled(Exception):
    """Raised inside :meth:`Pipeline.compute` when its ``cancel`` event is set (the GUI asked for
    a newer frame or parameters while this one was still being computed)."""


@dataclass
class ComputeContext:
    """What a modifier gets besides the data: frame index, pipeline, compute backend."""

    frame: int
    pipeline: "Pipeline"
    backend: Backend
    cache: dict = field(default_factory=dict)
    modifier_index: int | None = None  # position of the running modifier in pipeline.modifiers
    cancel: threading.Event | None = None  # set by the caller to stop a computation early

    @property
    def xp(self):
        return self.backend.xp

    def source_frame(self, frame: int) -> DataCollection:
        """Unmodified input data of another frame (e.g. a reference configuration)."""
        self.check_cancel()
        return self.pipeline.source_frame(frame)

    def input_frame(self, frame: int) -> DataCollection:
        """Data at ``frame`` as it enters the running modifier: the source plus every
        modifier above it. Trajectory modifiers use this to see upstream selections,
        transformations and tables at other frames."""
        self.check_cancel()
        return self.pipeline.compute(frame, device=self.backend, upto=self.modifier_index, cancel=self.cancel)

    def check_cancel(self):
        """Raise :class:`ComputeCancelled` if the caller gave up on this computation. Long
        modifiers call it in their loops; :meth:`input_frame` calls it for them."""
        if self.cancel is not None and self.cancel.is_set():
            raise ComputeCancelled

    def upstream_key(self) -> tuple:
        """Hashable fingerprint of the input (source, frame count, upstream modifiers and
        their parameters); trajectory modifiers cache their result under it."""
        mods = self.pipeline.modifiers[:self.modifier_index]
        return (id(self.pipeline.source), self.pipeline.num_frames, self.backend.name,
                tuple((type(m).__name__, repr(m)) for m in mods if m.enabled))


@functools.lru_cache(maxsize=None)
def _aliases(cls) -> dict:
    """``ALIASES`` of ``cls`` and its bases (a subclass wins)."""
    out = {}
    for klass in reversed(cls.__mro__):
        out.update(klass.__dict__.get("ALIASES", {}))
    return out


def _translate(cls, kwargs: dict):
    """Rewrite old argument names in ``kwargs`` to ``cls``'s current ones, in place."""
    for old, new in _aliases(cls).items():
        if old in kwargs:
            if new in kwargs:
                raise TypeError(f"{cls.__name__}() got both {new!r} and its old name {old!r}")
            kwargs[new] = kwargs.pop(old)
    legacy = getattr(cls, "_legacy_arguments", None)
    if legacy is not None:
        legacy(kwargs)


class Modifier:
    """Base class. Subclasses implement ``modify(data, ctx)`` and return new data or None."""

    enabled: bool = True
    status: str | None = None  # last error message, None when the last evaluation succeeded
    last_time: float | None = None  # seconds spent in the last evaluation
    last_output: dict | None = None  # what the last evaluation changed, see :func:`output_summary`
    # Whether the pipeline may reuse this modifier's output for the same input and settings
    # (see :func:`_modifier_key`). None: built-in modifiers yes, plugins and scripts no.
    cacheable: bool | None = None

    #: Old or OVITO argument names a subclass still accepts, as constructor keywords and as
    #: attributes: ``{"number_of_bins": "bins"}`` (one name per idea across ewald, D55). A subclass
    #: whose old arguments do not map one to one (``fix_xrange`` + ``xrange_start`` + ``xrange_end``
    #: -> ``range``) also defines ``_legacy_arguments(kwargs)``, which rewrites them in place.
    ALIASES: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        init = cls.__dict__.get("__init__")
        if init is None or not (_aliases(cls) or hasattr(cls, "_legacy_arguments")):
            return

        @functools.wraps(init)
        def __init__(self, *args, **kw):
            _translate(cls, kw)   # idempotent: a subclass's wrapper may have translated them already
            init(self, *args, **kw)

        cls.__init__ = __init__

    def __getattr__(self, name):
        new = _aliases(type(self)).get(name)
        if new is None or name.startswith("__"):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self, new)

    def __setattr__(self, name, value):
        object.__setattr__(self, _aliases(type(self)).get(name, name), value)

    def modify(self, data: DataCollection, ctx: ComputeContext):
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__

    def parameters(self) -> dict:
        """Constructor arguments and their current values (used by the GUI and sessions)."""
        import inspect

        sig = inspect.signature(type(self).__init__)
        return {k: getattr(self, k) for k in list(sig.parameters)[1:]
                if hasattr(self, k) and sig.parameters[k].kind not in (inspect.Parameter.VAR_KEYWORD,
                                                                       inspect.Parameter.VAR_POSITIONAL)}

    def _check_positive(self, *names: str):
        """Raise ValueError unless each named setting is a number above zero (a cutoff, a count):
        ``cutoff=-1`` would otherwise run and give zeros everywhere."""
        for name in names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ValueError(f"{type(self).__name__}: {name} must be a number above 0, not {value!r}")

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in self.parameters().items())
        return f"{self.name}({params})"


class PythonModifier(Modifier):
    """Wrap a user function ``f(frame, data)`` (OVITO-compatible signature).

    The function may change ``data`` (set or replace its arrays; writing into them raises,
    since they are shared with the cached frame) or return a new DataCollection.
    A third parameter, if present, receives the ComputeContext. ``pipe.add(f)`` wraps ``f`` in
    one of these. :class:`~ewald.modifiers.PythonScript` is the same step kept as source text
    (what a session or notebook stores and runs only when trusted).
    """

    def __init__(self, function: Callable, **params):
        self.function = function
        self.params = params

    @property
    def name(self) -> str:
        return getattr(self.function, "__name__", "PythonModifier")

    def modify(self, data, ctx):
        return self._call(ctx.frame, data, ctx)

    def _call(self, first, data, ctx):
        import inspect

        try:  # positional slots not filled by keyword parameters decide whether ctx is passed
            params = inspect.signature(self.function).parameters.values()
            nargs = 3 if any(p.kind == p.VAR_POSITIONAL for p in params) else \
                sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name not in self.params
                    for p in params)
        except (TypeError, ValueError):
            nargs = 2
        args = (first, data, ctx)[: max(2, min(nargs, 3))]
        before = _figure_numbers()
        try:
            out = self.function(*args, **self.params)
        finally:
            figures = _take_figures(before)
        target = out if isinstance(out, DataCollection) else data
        for k, png in enumerate(figures):
            name = f"Figure: {self.name}" + (f" ({k + 1})" if k else "")
            target.attributes[name] = StepFigure(png, self.name)
        return out


class StepFigure:
    """A matplotlib figure a Python step drew (``plt.plot(...)`` in an ``@each_frame`` /
    ``@all_frames`` function or a ``[each frame]`` / ``[all frames]`` cell), as PNG bytes. The
    step keeps it in ``data.attributes["Figure: <step name>"]`` (``"... (2)"`` for a second one):
    the figure itself is closed, so steps never leave figures open or put them under another
    cell. In the GUI an each-frame step's figure follows the frame in the Plots panel and an
    all-frames step's figure stays under its cell; in Jupyter it shows as the image."""

    def __init__(self, png: bytes, step: str = ""):
        self.png = png
        self.step = step

    def _repr_png_(self):
        return self.png

    def __repr__(self):
        return f"<figure of step {self.step!r}>"


def _figure_numbers() -> set:
    import sys

    plt = sys.modules.get("matplotlib.pyplot")
    return set(plt.get_fignums()) if plt is not None else set()


def _take_figures(before: set) -> list:
    """PNG bytes of the pyplot figures opened since ``before``; they are closed."""
    import io
    import sys

    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return []
    out = []
    for num in plt.get_fignums():
        if num in before:
            continue
        fig = plt.figure(num)
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi="figure", bbox_inches="tight")
            out.append(buf.getvalue())
        finally:
            plt.close(fig)
    return out


class Frames:
    """The frames an :func:`all_frames` step reads: ``len(frames)``, ``frames[k]`` (the data at
    frame ``k`` as it enters the step, every step above applied), iteration and slices
    (``frames[::10]``). Frames are computed one at a time when read, so lazy readers stay lazy."""

    def __init__(self, ctx: ComputeContext, indices: range | None = None):
        self._ctx = ctx
        self.indices = range(ctx.pipeline.num_frames) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        if isinstance(k, slice):
            return Frames(self._ctx, self.indices[k])
        return self._ctx.input_frame(self.indices[k])

    def __iter__(self):
        return (self._ctx.input_frame(k) for k in self.indices)

    def __repr__(self):
        return f"Frames({len(self)} frames)"


class AllFramesModifier(PythonModifier):
    """A Python step that reads every frame once: ``f(frames, data)`` (see :func:`all_frames`).

    The first evaluation runs ``f`` with :class:`Frames` and the data of the current frame and
    keeps what it set (tables, attributes, per-particle properties); every later frame gets the
    kept result without running ``f`` again, until a step above it, the source or its
    parameters change. Plots of the result then only move their frame marker during playback."""

    def modify(self, data, ctx):
        key = (ctx.upstream_key(), id(self.function), repr(self.params))
        cache = self.__dict__.get("_result")
        if cache is None or cache[0] != key:
            before = _contents(data)
            out = self._call(Frames(ctx), data, ctx)
            result = out if isinstance(out, DataCollection) else data
            cache = (key, _changes(before, result))
            self._result = cache
            if out is not None:
                return out
            return None
        tables, attributes, particles = cache[1]
        for k, v in tables.items():
            data.tables[k] = dict(v)
        data.attributes.update(attributes)
        for k, v in particles.items():
            if len(v) != data.count:
                raise ValueError(f"{self.name}: per-particle result '{k}' has {len(v)} values, but frame "
                                 f"{ctx.frame} has {data.count} particles")
            data.particles[k] = v
        return None


def _contents(data) -> tuple:
    return ({k: _table_id(v) for k, v in data.tables.items()}, {k: id(v) for k, v in data.attributes.items()},
            {k: id(v) for k, v in data.particles.items()})


def _changes(before: tuple, data) -> tuple:
    """Tables, attributes and particle properties of ``data`` that are new or replaced since ``before``."""
    tables, attributes, particles = before
    return ({k: dict(v) for k, v in data.tables.items() if tables.get(k) != _table_id(v)},
            {k: v for k, v in data.attributes.items() if attributes.get(k) != id(v)},
            {k: v for k, v in data.particles.items() if particles.get(k) != id(v)})


def each_frame(function: Callable) -> Callable:
    """Mark a Python step that runs at every frame, ``f(frame, data)`` (the default for a
    function added with ``pipe.add``; the decorator says so in the notebook)::

        @each_frame
        def coordination(frame, data): ...
        pipe.add(coordination)
    """
    function._ewald_frames = "each"
    return function


def all_frames(function: Callable) -> Callable:
    """Mark a Python step that reads the whole trajectory once, ``f(frames, data)``:
    ``frames`` is a :class:`Frames` (the data of every frame as it enters the step) and
    ``data`` the current frame, on which ``f`` sets its result (a table, attributes, a
    per-particle property). The result is computed once and reused at every frame::

        @all_frames
        def mean_height(frames, data):
            z = [d["Position"][:, 2].mean() for d in frames]
            data.tables["height"] = {"Frame": np.arange(len(z)), "Mean z": np.array(z)}
        pipe.add(mean_height)
    """
    function._ewald_frames = "all"
    return function


class ModifierList(list):
    """List that wraps plain callables in PythonModifier on insertion."""

    @staticmethod
    def _wrap(m):
        if isinstance(m, Modifier):
            return m
        if callable(m):
            return AllFramesModifier(m) if getattr(m, "_ewald_frames", None) == "all" else PythonModifier(m)
        raise TypeError(f"{m!r} is not a Modifier or callable")

    def append(self, m):
        super().append(self._wrap(m))

    def insert(self, i, m):
        super().insert(i, self._wrap(m))

    def extend(self, ms):
        super().extend(self._wrap(m) for m in ms)

    def __setitem__(self, i, m):
        super().__setitem__(i, [self._wrap(x) for x in m] if isinstance(i, slice) else self._wrap(m))


def _upstream(data: DataCollection) -> tuple:
    """What a step's input holds, for :func:`_keep_upstream`."""
    return dict(data.particles), {k: dict(v) for k, v in data.tables.items()}, dict(data.attributes)


def _keep_upstream(mod, upstream: tuple, data: DataCollection):
    """A step a verb added (``rdf(pipe)``, ``cna(pipe)``, see :mod:`ewald.verbs`) never replaces what
    the steps before it wrote: ``rdf(pipe, cutoff=6)`` after the user's own
    ``CoordinationAnalysis(cutoff=3)`` must not change their "Coordination". Particle properties and
    attributes that existed before keep their values; a table of the same name keeps the earlier one,
    and the verb's goes in beside it as ``"<name> (<verb>)"`` (``mod._verb_tables`` maps the names)."""
    particles, tables, attributes = upstream
    renamed = {}
    for name, arr in particles.items():
        if name in data.particles and data.particles[name] is not arr and len(arr) == data.count:
            data.particles[name] = arr
    shadowed = {}
    for name, value in attributes.items():
        if data.attributes.get(name, value) is not value:
            shadowed[name] = data.attributes[name]
        data.attributes[name] = value
    mod._verb_attributes = shadowed   # the step's own values of those, for the verb's Table
    for name, table in tables.items():
        out = data.tables.get(name)
        if out is not None and (out.keys() != table.keys() or any(out[c] is not a for c, a in table.items())):
            new = f"{name} ({getattr(mod, '_verb_name', None) or mod.name})"
            data.tables[new] = out
            data.tables[name] = table
            renamed[name] = new
    mod._verb_tables = renamed


def _snapshot(data: DataCollection) -> tuple:
    """Cheap fingerprint of a collection (array identities, not contents): modifiers
    replace arrays instead of writing into them, so identity tells what changed."""
    return (data.count, {k: id(v) for k, v in data.particles.items()},
            {k: id(v) for k, v in data.attributes.items()}, {k: _table_id(v) for k, v in data.tables.items()},
            int(data.selection.sum()) if "Selection" in data.particles else None)


def _table_id(table) -> tuple:
    """Identity of a table's columns: ``copy()`` makes new dicts around the same arrays."""
    return tuple((k, id(v)) for k, v in table.items()) if isinstance(table, dict) else (id(table),)


def output_summary(before: tuple, data: DataCollection, frame: int) -> dict:
    """What one modifier did to the data: particle count before and after, properties,
    attributes and tables it added or replaced, and the selection count. Notebook cells
    show this under each step."""
    count, props, attrs, tables, _ = before
    if data.count != count:  # deleting particles replaces every array: report only new ones
        changed = [k for k in data.particles if k not in props]
    else:
        changed = [k for k, v in data.particles.items() if props.get(k) != id(v)]
    return {"frame": frame, "count_before": count, "count": data.count, "properties": changed,
            "attributes": [k for k, v in data.attributes.items() if attrs.get(k) != id(v)],
            "tables": [k for k, v in data.tables.items() if tables.get(k) != _table_id(v)],
            "selected": int(data.selection.sum()) if "Selection" in data.particles else None}


_RUN_STATE = ("enabled", "status", "last_time", "last_output")
_UNKEYABLE = object()


def _freeze(value, depth: int = 0):
    """A hashable stand-in for a modifier setting, or ``_UNKEYABLE`` when its state can't be
    read off the value (functions, arbitrary objects): such a modifier is always recomputed."""
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    if depth > 6:
        return _UNKEYABLE
    import enum
    import os

    import numpy as np

    if isinstance(value, (np.generic, enum.Enum, os.PathLike)):
        return (type(value).__name__, str(value))
    if isinstance(value, np.ndarray):
        import hashlib

        if value.dtype == object:
            return _freeze(value.tolist(), depth + 1)
        digest = hashlib.blake2b(np.ascontiguousarray(value).view(np.uint8).ravel(), digest_size=16).digest()
        return ("ndarray", value.shape, value.dtype.str, digest)
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        items = sorted(value.items(), key=lambda kv: repr(kv[0])) if isinstance(value, dict) else \
            sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        out = []
        for item in items:
            f = _freeze(item, depth + 1)
            if f is _UNKEYABLE:
                return _UNKEYABLE
            out.append(f)
        return (type(value).__name__, tuple(out))
    return _UNKEYABLE


def _modifier_key(m):
    """What a modifier's output depends on besides its input: its class and every public
    attribute (constructor arguments and anything set later, e.g. by the GUI), or None when
    the output must not be reused (scripts, plugins, settings that are objects or functions)."""
    if not m.enabled:
        return ("off", id(m))
    cacheable = m.cacheable
    if cacheable is None:
        cacheable = type(m).__module__.startswith("ewald.")
    if not cacheable:
        return None
    state = _freeze({k: v for k, v in vars(m).items() if not k.startswith("_") and k not in _RUN_STATE})
    return None if state is _UNKEYABLE else (id(m), type(m), state)


class Pipeline:
    """A source plus an ordered list of modifiers, evaluated lazily per frame.

    >>> pipe = import_file("dump.lammpstrj", type_map={1: "Cu"})
    >>> pipe.modifiers.append(CoordinationAnalysis(cutoff=3.0))
    >>> data = pipe.compute(10, device="cuda")
    """

    def __init__(self, source, modifiers=(), device: str | None = None, cache_size: int = 8,
                 cache_memory: float | None = 2 * 2**30):
        """``cache_size`` caps how many source frames stay in memory and ``cache_memory``
        their total array bytes (None = no byte limit); the newest frame is always kept,
        so multi-million-atom trajectories hold a frame or two instead of eight."""
        self.source = source
        self.modifiers = ModifierList()
        self.modifiers.extend(modifiers)
        self.device = device
        self._cache: OrderedDict[int, DataCollection] = OrderedDict()
        self._cache_lock = threading.RLock()  # the GUI computes on a worker thread too
        # One computation at a time per pipeline: modifiers keep caches on themselves, and numba's
        # fallback threading layer (workqueue) aborts on parallel kernels entered from two threads.
        self._compute_lock = threading.RLock()
        self._cache_size = cache_size
        self.cache_memory = cache_memory
        # Results after the leading modifiers whose output can be reused (see _modifier_key),
        # keyed by source, frame, device and those modifiers' settings: showing the same frame
        # again (view cells, plots, the GUI, a kernel) skips the analysis.
        # Steps slower than ``checkpoint_seconds`` also keep their own result, so changing a later
        # step (its notebook cell) starts there instead of at the source frame.
        self._outputs: OrderedDict[tuple, DataCollection] = OrderedDict()
        self.cache_outputs = True

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_cache_lock", None)
        state.pop("_compute_lock", None)
        state["_outputs"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache_lock, self._compute_lock = threading.RLock(), threading.RLock()
        self.__dict__.setdefault("_outputs", OrderedDict())
        self.__dict__.setdefault("cache_outputs", True)

    def add(self, *modifiers):
        """Append modifiers (``pipe.add(m)`` is ``pipe.modifiers.append(m)``) and return the last
        argument, so it also works as a decorator: ``@pipe.add`` on ``def f(frame, data)``."""
        self.modifiers.extend(modifiers)
        return modifiers[-1] if modifiers else None

    @property
    def num_frames(self) -> int:
        """Frames of the source, or of the trajectory a modifier loads on top of it
        (a modifier with a ``frame_count()`` method, such as LoadTrajectory)."""
        n = self.source.num_frames
        for m in self.modifiers:
            count = m.frame_count() if m.enabled and hasattr(m, "frame_count") else None
            if count:
                n = count
        return n

    def __len__(self):
        return self.num_frames

    def _frame_index(self, frame) -> int:
        """``frame`` as an index into the frames: negative counts from the end, as in a list;
        out of range raises IndexError (``compute(999)`` on 5 frames used to give frame 4)."""
        n = self.num_frames
        try:
            i = operator.index(frame)
        except TypeError:
            raise TypeError(f"frame must be an int, not {type(frame).__name__}") from None
        if not -n <= i < n:
            raise IndexError(f"frame {i} out of range: the pipeline has {n} frame{'s' * (n != 1)}")
        return i % n

    def __getitem__(self, index):
        """``pipe[i]`` is ``pipe.compute(i)`` (negative from the end, out of range raises IndexError);
        ``pipe[a:b:c]`` is a list of those frames, as ``ase.io.read(path, index=":")`` gives."""
        n = self.num_frames
        if isinstance(index, slice):
            return [self.compute(i) for i in range(*index.indices(n))]
        if not isinstance(index, int) and not hasattr(index, "__index__"):
            raise TypeError(f"pipeline frames are indexed by int or slice, not {type(index).__name__}")
        return self.compute(index)

    def __iter__(self):
        """Every frame, computed one at a time (``for data in pipe``); see :meth:`frames`."""
        return self.frames()

    def to_ase(self, frames=None) -> list:
        """The computed frames as a list of ASE ``Atoms`` (``frames``: indices or a slice; default all),
        ready for ``ase.io.write``; per-particle numeric properties go into ``atoms.arrays``."""
        if frames is None:
            frames = slice(None)
        picked = self[frames] if isinstance(frames, slice) else [self[i] for i in frames]
        return [data.to_ase() for data in picked]

    def source_frame(self, frame: int) -> DataCollection:
        frame %= self.source.num_frames
        with self._cache_lock:
            data = self._cache.get(frame)
            if data is not None:
                self._cache.move_to_end(frame)
                return data.copy()
        data = self.source.compute(frame)  # outside the lock: reading a frame can take seconds
        data = data.copy().freeze()  # modifiers that write in place fail instead of changing the cache
        with self._cache_lock:
            self._cache[frame] = data
            self._cache.move_to_end(frame)
            self._trim_cache()
        return data.copy()

    def refresh(self) -> int:
        """Pick up frames a running simulation appended to the source file(s) since they were
        opened (LAMMPS dump, XDATCAR, OUTCAR, extxyz; local or over SSH) and drop cached frames
        that may have changed. Returns how many frames were added (negative if the file was
        rewritten shorter); call it again whenever you want to look for more::

            pipe = import_file("dump.lammpstrj")   # while LAMMPS is running
            ...
            if pipe.refresh():
                data = pipe.compute(-1)            # the newest frame
        """
        refresh = getattr(self.source, "refresh", None)
        if refresh is None:
            return 0
        with self._compute_lock:  # not while a worker thread reads the source
            before = self.num_frames
            first = refresh()
            if first is not None:
                self.invalidate(first)
            return self.num_frames - before

    def invalidate(self, first: int = 0):
        """Forget cached source frames from ``first`` on (their file changed), and every
        cached modifier result. Steps that failed for a missing package try again."""
        for mod in self.modifiers:
            mod.__dict__.pop("_missing_dependency", None)
        with self._cache_lock:
            for frame in [f for f in self._cache if f >= first]:
                del self._cache[frame]
            self._outputs.clear()

    def _trim_cache(self):
        while len(self._cache) > max(self._cache_size, 1):
            self._cache.popitem(last=False)
        while len(self._outputs) > 4 * max(self._cache_size, 1):  # results of several steps per frame
            self._outputs.popitem(last=False)
        if self.cache_memory is not None:
            def total():  # an array shared by a frame and the results computed from it counts once
                seen = {}
                for d in (*self._cache.values(), *self._outputs.values()):
                    for a in d.particles.values():
                        seen[id(a)] = getattr(a, "nbytes", 0)
                return sum(seen.values())

            while self._outputs and total() > self.cache_memory:  # results go first: recomputable
                self._outputs.popitem(last=False)
            while len(self._cache) > 1 and total() > self.cache_memory:
                self._cache.popitem(last=False)

    checkpoint_seconds = 0.02

    def compute(self, frame: int = 0, device: str | None = None, errors: str = "raise",
                upto: int | None = None, cancel: threading.Event | None = None) -> DataCollection:
        """Evaluate the pipeline at ``frame`` (negative counts from the end, as in a list; out of
        range raises IndexError).

        ``errors="skip"`` keeps going past a failing modifier (its message is stored
        in ``modifier.status``); the GUI uses this so one bad modifier never blanks the view.
        ``upto`` stops before the modifier at that index. ``cancel`` is an event a caller on
        another thread sets to stop early; the computation then raises :class:`ComputeCancelled`
        between modifiers or wherever a modifier calls ``ctx.check_cancel()``.
        """
        with self._compute_lock:
            return self._compute(frame, device, errors, upto, cancel)

    def _compute(self, frame, device, errors, upto, cancel) -> DataCollection:
        frame = self._frame_index(frame)
        backend = get_backend(device or self.device)
        mods = list(self.modifiers[:upto])
        keys = []  # settings of the leading modifiers whose output can be reused
        if self.__dict__.get("cache_outputs", True):
            for mod in mods:
                key = _modifier_key(mod)
                if key is None:
                    break
                keys.append(key)
        base = (id(self.source), frame, backend.name, self.num_frames)
        start, data = 0, None
        with self._cache_lock:
            for n in range(len(keys), 0, -1):  # the longest prefix computed before
                hit = self._outputs.get(base + (tuple(keys[:n]),))
                if hit is not None:
                    self._outputs.move_to_end(base + (tuple(keys[:n]),))
                    start, data = n, hit.copy()
                    break
        if data is None:
            data = self.source_frame(frame)
        failed = False
        for index, mod in enumerate(mods):
            if index < start:
                continue
            if index == len(keys) and keys and not failed and start < len(keys):
                self._store_output(base + (tuple(keys),), data)
            if cancel is not None and cancel.is_set():
                raise ComputeCancelled
            if not mod.enabled:
                continue
            missing = mod.__dict__.get("_missing_dependency")
            if errors != "raise" and missing is not None and missing[0] == repr(mod):
                # a package it needs is not installed: retrying every frame repeats the same error;
                # changing the step's settings or Pipeline.invalidate() tries once more
                mod.status, failed = missing[1], True
                continue
            ctx = ComputeContext(frame, self, backend, modifier_index=index, cancel=cancel)
            before = _snapshot(data)
            upstream = _upstream(data) if getattr(mod, "_verb", None) else None
            t0 = time.perf_counter()
            try:
                result = mod.modify(data, ctx)
            except ComputeCancelled:
                raise
            except Exception as exc:
                mod.last_time = time.perf_counter() - t0
                read_only = isinstance(exc, ValueError) and "read-only" in str(exc)
                if read_only:
                    exc = ValueError(f"{exc}: {READ_ONLY_HINT}")
                mod.status = f"{type(exc).__name__}: {exc}"
                failed = True
                if isinstance(exc, ImportError):
                    mod._missing_dependency = (repr(mod), mod.status)
                if errors == "raise":
                    if read_only:
                        raise exc from exc.__context__
                    raise
                continue
            mod.status = None
            mod.__dict__.pop("_missing_dependency", None)
            mod.last_time = time.perf_counter() - t0
            if result is not None:
                data = result
            if upstream is not None:
                _keep_upstream(mod, upstream, data)
            mod.last_output = output_summary(before, data, frame)
            if index < len(keys) - 1 and not failed and mod.last_time >= self.checkpoint_seconds:
                self._store_output(base + (tuple(keys[:index + 1]),), data)
        if len(keys) == len(mods) and keys and not failed and start < len(keys):
            self._store_output(base + (tuple(keys),), data)
        return data

    def _store_output(self, key, data):
        with self._cache_lock:
            # ``data`` goes on to later steps and callers: freezing it keeps them from writing
            # into the cached arrays it shares; they change their own copy
            self._outputs[key] = data.freeze().copy()
            self._outputs.move_to_end(key)
            self._trim_cache()

    def frames(self, start: int = 0, stop: int | None = None, step: int = 1, device=None):
        stop = self.num_frames if stop is None else stop
        for i in range(start, stop, step):
            yield self.compute(i, device=device)

    @property
    def view(self):
        """How this pipeline's structure is drawn (:class:`ewald.view3d.View`): style, colors, radii,
        bonds, cell, camera, ... Notebook ``view(pipe, ...)`` cells, the GUI and Claude set it;
        :meth:`render`, ``snapshot(pipe)`` and the Jupyter picture draw with it."""
        v = self.__dict__.get("_view")
        if v is None:
            from .view3d import View

            v = self.__dict__["_view"] = View(self)
        return v

    @view.setter
    def view(self, value):
        from .view3d import View

        if not isinstance(value, View):
            raise TypeError("pipe.view is a View; use view(pipe, ...) or pipe.view.update(...) to change it")
        self.__dict__["_view"] = value.copy(self)

    @property
    def views(self):
        """This pipeline's view tabs (:class:`ewald.view3d.Views`, ``pipe.views["Top"]``): more 3D views of
        the same structure, each the main :attr:`view` with its own changes; ``view(pipe, tab=..., ...)`` sets one."""
        v = self.__dict__.get("_views")
        if v is None:
            from .view3d import Views

            v = self.__dict__["_views"] = Views(self)
        return v

    def render(self, filename=None, frame: int | None = None, tab: str | None = None, **kwargs):
        """Draw ``frame`` (default: the view's frame) with :attr:`view` (or the view tab ``tab``);
        ``kwargs`` override it for this call."""
        return (self.view if tab is None else self.views[tab]).render(filename, frame, **kwargs)

    def _repr_png_(self):  # Jupyter shows frame 0 as a picture (ewald.display.options)
        from .display import pipeline_png

        return pipeline_png(self)

    def __repr__(self):
        mods = ", ".join(m.name for m in self.modifiers)
        return f"<Pipeline frames={self.num_frames} modifiers=[{mods}]>"
