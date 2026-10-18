"""Data pipeline: source -> modifiers -> DataCollection."""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

from .backend import Backend, get_backend
from .data import DataCollection


@dataclass
class ComputeContext:
    """What a modifier gets besides the data: frame index, pipeline, compute backend."""

    frame: int
    pipeline: "Pipeline"
    backend: Backend
    cache: dict = field(default_factory=dict)
    modifier_index: int | None = None  # position of the running modifier in pipeline.modifiers

    @property
    def xp(self):
        return self.backend.xp

    def source_frame(self, frame: int) -> DataCollection:
        """Unmodified input data of another frame (e.g. a reference configuration)."""
        return self.pipeline.source_frame(frame)

    def input_frame(self, frame: int) -> DataCollection:
        """Data at ``frame`` as it enters the running modifier: the source plus every
        modifier above it. Trajectory modifiers use this to see upstream selections,
        transformations and tables at other frames."""
        return self.pipeline.compute(frame, device=self.backend, upto=self.modifier_index)

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

    The function may modify ``data`` in place or return a new DataCollection.
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

        try:
            nargs = len(inspect.signature(self.function).parameters)
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


class Pipeline:
    """A source plus an ordered list of modifiers, evaluated lazily per frame.

    >>> pipe = import_file("dump.lammpstrj", type_map={1: "Cu"})
    >>> pipe.modifiers.append(CoordinationAnalysis(cutoff=3.0))
    >>> data = pipe.compute(10, device="cuda")
    """

    def __init__(self, source, modifiers=(), device: str | None = None, cache_size: int = 8):
        self.source = source
        self.modifiers = ModifierList()
        self.modifiers.extend(modifiers)
        self.device = device
        self._cache: OrderedDict[int, DataCollection] = OrderedDict()
        self._cache_size = cache_size

    @property
    def num_frames(self) -> int:
        return self.source.num_frames

    def __len__(self):
        return self.num_frames

    def source_frame(self, frame: int) -> DataCollection:
        frame %= self.num_frames
        if frame in self._cache:
            self._cache.move_to_end(frame)
        else:
            self._cache[frame] = self.source.compute(frame)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return self._cache[frame].copy()

    def compute(self, frame: int = 0, device: str | None = None, errors: str = "raise",
                upto: int | None = None) -> DataCollection:
        """Evaluate the pipeline at ``frame``.

        ``errors="skip"`` keeps going past a failing modifier (its message is stored
        in ``modifier.status``); the GUI uses this so one bad modifier never blanks the view.
        ``upto`` stops before the modifier at that index.
        """
        frame %= self.num_frames
        backend = get_backend(device or self.device)
        data = self.source_frame(frame)
        for index, mod in enumerate(self.modifiers[:upto]):
            if not mod.enabled:
                continue
            ctx = ComputeContext(frame, self, backend, modifier_index=index)
            t0 = time.perf_counter()
            try:
                result = mod.modify(data, ctx)
            except Exception as exc:
                mod.last_time = time.perf_counter() - t0
                mod.status = f"{type(exc).__name__}: {exc}"
                if errors == "raise":
                    raise
                continue
            mod.status = None
            mod.last_time = time.perf_counter() - t0
            if result is not None:
                data = result
        return data

    def frames(self, start: int = 0, stop: int | None = None, step: int = 1, device=None):
        stop = self.num_frames if stop is None else stop
        for i in range(start, stop, step):
            yield self.compute(i, device=device)

    def render(self, filename=None, frame: int = 0, **kwargs):
        from .render import render

        return render(self.compute(frame), filename, **kwargs)

    def __repr__(self):
        mods = ", ".join(m.name for m in self.modifiers)
        return f"<Pipeline frames={self.num_frames} modifiers=[{mods}]>"
