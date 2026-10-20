"""Data pipeline: source -> modifiers -> DataCollection."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

from .backend import Backend, get_backend
from .data import DataCollection


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


class Modifier:
    """Base class. Subclasses implement ``modify(data, ctx)`` and return new data or None."""

    enabled: bool = True
    status: str | None = None  # last error message, None when the last evaluation succeeded
    last_time: float | None = None  # seconds spent in the last evaluation
    last_output: dict | None = None  # what the last evaluation changed, see :func:`output_summary`
    # Whether the pipeline may reuse this modifier's output for the same input and settings
    # (see :func:`_modifier_key`). None: built-in modifiers yes, plugins and scripts no.
    cacheable: bool | None = None

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

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in self.parameters().items())
        return f"{self.name}({params})"


class PythonModifier(Modifier):
    """Wrap a user function ``f(frame, data)`` (OVITO-compatible signature).

    The function may change ``data`` (set or replace its arrays; writing into them raises,
    since they are shared with the cached frame) or return a new DataCollection.
    A third parameter, if present, receives the ComputeContext.
    """

    def __init__(self, function: Callable, **params):
        self.function = function
        self.params = params

    @property
    def name(self) -> str:
        return getattr(self.function, "__name__", "PythonModifier")

    def modify(self, data, ctx):
        import inspect

        try:  # positional slots not filled by keyword parameters decide whether ctx is passed
            params = inspect.signature(self.function).parameters.values()
            nargs = 3 if any(p.kind == p.VAR_POSITIONAL for p in params) else \
                sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name not in self.params
                    for p in params)
        except (TypeError, ValueError):
            nargs = 2
        args = (ctx.frame, data, ctx)[: max(2, min(nargs, 3))]
        return self.function(*args, **self.params)


class ModifierList(list):
    """List that wraps plain callables in PythonModifier on insertion."""

    @staticmethod
    def _wrap(m):
        if isinstance(m, Modifier):
            return m
        if callable(m):
            return PythonModifier(m)
        raise TypeError(f"{m!r} is not a Modifier or callable")

    def append(self, m):
        super().append(self._wrap(m))

    def insert(self, i, m):
        super().insert(i, self._wrap(m))

    def extend(self, ms):
        super().extend(self._wrap(m) for m in ms)

    def __setitem__(self, i, m):
        super().__setitem__(i, [self._wrap(x) for x in m] if isinstance(i, slice) else self._wrap(m))


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
        cached modifier result."""
        with self._cache_lock:
            for frame in [f for f in self._cache if f >= first]:
                del self._cache[frame]
            self._outputs.clear()

    def _trim_cache(self):
        while len(self._cache) > max(self._cache_size, 1):
            self._cache.popitem(last=False)
        while len(self._outputs) > max(self._cache_size, 1):
            self._outputs.popitem(last=False)
        if self.cache_memory is not None:
            def total():
                return sum(d.nbytes for d in self._cache.values()) + sum(d.nbytes for d in self._outputs.values())

            while self._outputs and total() > self.cache_memory:  # results go first: recomputable
                self._outputs.popitem(last=False)
            while len(self._cache) > 1 and total() > self.cache_memory:
                self._cache.popitem(last=False)

    def compute(self, frame: int = 0, device: str | None = None, errors: str = "raise",
                upto: int | None = None, cancel: threading.Event | None = None) -> DataCollection:
        """Evaluate the pipeline at ``frame``.

        ``errors="skip"`` keeps going past a failing modifier (its message is stored
        in ``modifier.status``); the GUI uses this so one bad modifier never blanks the view.
        ``upto`` stops before the modifier at that index. ``cancel`` is an event a caller on
        another thread sets to stop early; the computation then raises :class:`ComputeCancelled`
        between modifiers or wherever a modifier calls ``ctx.check_cancel()``.
        """
        with self._compute_lock:
            return self._compute(frame, device, errors, upto, cancel)

    def _compute(self, frame, device, errors, upto, cancel) -> DataCollection:
        frame %= self.num_frames
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
            ctx = ComputeContext(frame, self, backend, modifier_index=index, cancel=cancel)
            before = _snapshot(data)
            t0 = time.perf_counter()
            try:
                result = mod.modify(data, ctx)
            except ComputeCancelled:
                raise
            except Exception as exc:
                mod.last_time = time.perf_counter() - t0
                mod.status = f"{type(exc).__name__}: {exc}"
                failed = True
                if errors == "raise":
                    raise
                continue
            mod.status = None
            mod.last_time = time.perf_counter() - t0
            if result is not None:
                data = result
            mod.last_output = output_summary(before, data, frame)
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

    def render(self, filename=None, frame: int | None = None, **kwargs):
        """Draw ``frame`` (default: the view's frame) with :attr:`view`; ``kwargs`` override it for this call."""
        return self.view.render(filename, frame, **kwargs)

    def _repr_png_(self):  # Jupyter shows frame 0 as a picture (ewald.display.options)
        from .display import pipeline_png

        return pipeline_png(self)

    def __repr__(self):
        mods = ", ".join(m.name for m in self.modifiers)
        return f"<Pipeline frames={self.num_frames} modifiers=[{mods}]>"
