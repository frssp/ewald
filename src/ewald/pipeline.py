"""Data pipeline: source -> modifiers -> DataCollection."""
from __future__ import annotations

import contextlib
import functools
import operator
import sys
import threading
import time
import types
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
        """Unmodified input data of another frame (e.g. a reference configuration); negative
        counts from the end, out of range raises IndexError."""
        self.check_cancel()
        return self.pipeline.source_frame(frame)

    def reference_frame(self, frame: int = 0, offset: int | None = None) -> DataCollection:
        """Source data of a step's reference configuration: ``frame``, or with ``offset`` the
        frame that far from the current one. An offset that leaves the trajectory raises
        IndexError at either end (``offset=-1`` at frame 0 used to take the last frame)."""
        if offset is None:
            return self.source_frame(frame)
        n, k = self.pipeline.num_frames, self.frame + operator.index(offset)
        if not 0 <= k < n:
            raise IndexError(f"frame_offset={offset} at frame {self.frame} points to frame {k}: "
                             f"the pipeline has frames 0 to {n - 1}")
        return self.source_frame(k)

    def input_frame(self, frame: int) -> DataCollection:
        """Data at ``frame`` as it enters the running modifier: the source plus every
        modifier above it. Trajectory modifiers use this to see upstream selections,
        transformations and tables at other frames."""
        self.check_cancel()
        return self.pipeline.compute(frame, device=self.backend, upto=self.modifier_index, cancel=self.cancel)

    def input_frames(self, frames):
        """:meth:`input_frame` of each of ``frames``, one at a time, for a step that reads a
        whole trajectory. The frames are not kept: no cached frames or step results (reading
        20000 small frames spent more time on the cache's bookkeeping than on the file, and
        pushed the frames on screen out of the cache), and the steps above keep the status and
        summary of the frame being computed."""
        for frame in frames:
            self.check_cancel()
            with self.pipeline._compute_lock, self.pipeline._tokens_once():
                data = self.pipeline._compute(frame, self.backend, "raise", self.modifier_index, self.cancel, sweep=True)
            yield data

    def check_cancel(self):
        """Raise :class:`ComputeCancelled` if the caller gave up on this computation. Long
        modifiers call it in their loops; :meth:`input_frame` calls it for them."""
        if self.cancel is not None and self.cancel.is_set():
            raise ComputeCancelled

    def upstream_key(self) -> tuple:
        """Hashable fingerprint of the input (source, frame count, upstream modifiers and
        their parameters); trajectory modifiers cache their result under it.

        A Python step above counts with what its function computes from: its code, defaults,
        closure and the globals it reads (:func:`_token`), so a new value of a variable the
        function uses gives a new key. What cannot be read that way (a random generator, an
        open file, a pipeline) counts by identity: a change inside it is not seen."""
        pipe = self.pipeline
        mods = pipe.modifiers[:self.modifier_index]
        source = pipe._source_key()
        # contents only where they can be told (a file source, steps' settings and the files
        # they name), so that a new pipeline running the same cells finds the result
        # (:mod:`ewald.result_store`); a Python step's repr shows its function's address, and
        # its token says what the function computes
        return (_Same(pipe.source) if source is None else source, pipe.num_frames, self.backend.name,
                tuple((type(m).__name__, "" if isinstance(m, PythonModifier) else repr(m), pipe._step_token(m),
                       _stamps_or_same(m)) for m in mods if m.enabled))

    def shared(self, key, compute):
        """``compute()``, or the result a step in this or another pipeline computed under the
        same ``key`` (made from :meth:`upstream_key`; :mod:`ewald.result_store`). Steps that
        compute over every frame keep their result with it, so Run all does not compute it again."""
        return self.pipeline._shared_result(key, compute)


@functools.lru_cache(maxsize=None)
def _aliases(cls) -> dict:
    """``ALIASES`` of ``cls`` and its bases (a subclass wins)."""
    out = {}
    for klass in reversed(cls.__mro__):
        out.update(klass.__dict__.get("ALIASES", {}))
    return out


@functools.lru_cache(maxsize=None)
def _settings(cls) -> tuple:
    """Names of ``cls``'s constructor arguments."""
    import inspect

    try:
        return tuple(inspect.signature(cls.__init__).parameters)[1:]
    except (TypeError, ValueError):
        return ()


@functools.lru_cache(maxsize=4096)
def _misspelled(cls, name: str) -> str | None:
    """The setting of ``cls`` that ``name`` is a near miss of (``cutof`` -> ``cutoff``, an old
    name -> its current one), or None: ``name`` is a setting or nothing like one."""
    import difflib

    settings, aliases = _settings(cls), _aliases(cls)
    if name in settings:
        return None
    close = difflib.get_close_matches(name, (*settings, *aliases), n=1, cutoff=0.8)
    return aliases.get(close[0], close[0]) if close else None


def _in_method_of(obj, frame) -> bool:
    """Whether ``frame`` runs a method of ``obj``: its first argument is ``obj`` and its code is
    that of a function (property, wrapped ``__init__``) defined on ``obj``'s class. A helper
    ``def tune(step): step.cutof = 1`` is not one."""
    code = frame.f_code
    if not (code.co_argcount > 0 and frame.f_locals.get(code.co_varnames[0]) is obj):
        return False
    for klass in type(obj).__mro__:
        found = klass.__dict__.get(code.co_name)
        for part in (found, *(getattr(found, name, None) for name in ("__func__", "fget", "fset", "__wrapped__"))):
            if getattr(part, "__code__", None) is code:
                return True
    return False


def _init_parameters(init) -> tuple:
    """Argument names of a constructor (none when its signature cannot be read)."""
    import inspect

    try:
        return tuple(inspect.signature(init).parameters)
    except (TypeError, ValueError):
        return ()


def _check_output(output) -> str:
    """``output=`` as a name: a non-empty string."""
    if not isinstance(output, str) or not output.strip():
        raise ValueError(f"output must be a name such as 'cna', not {output!r}")
    return output.strip()


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

    #: Instance attributes where a modifier keeps results across frames (trajectory lines, unwrap
    #: checkpoints, a frozen property); :meth:`forget` drops them.
    CACHE_ATTRS: tuple = ("_result", "_unwrap", "_cache")

    def forget(self):
        """Drop what this modifier kept across evaluations (:attr:`CACHE_ATTRS`), so the next
        evaluation starts from scratch. Settings stay."""
        for name in self.CACHE_ATTRS:
            self.__dict__.pop(name, None)

    #: The name this step's results go under (D106): with ``output="cna"`` the properties,
    #: attributes and tables it writes are called "Structure Type (cna)", "structures (cna)", ...
    #: and what earlier steps wrote under the plain names stays. Every step takes it as a
    #: constructor keyword; None (the default) keeps the plain names. Steps whose own ``output``
    #: argument names what they write (``ComputeProperty``, ``GridDifference``) keep that meaning.
    output: str | None = None
    _OWN_OUTPUT = False   # whether the constructor itself has an ``output`` argument

    def __init__(self, **kwargs):
        # the constructor of steps that have none of their own: ``output=`` only
        output = kwargs.pop("output", None)
        if kwargs:
            raise TypeError(f"{type(self).__name__}() got unexpected keyword arguments: {', '.join(kwargs)}")
        if output is not None:
            self.output = _check_output(output)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        init = cls.__dict__.get("__init__")
        if init is None:    # inherited: a base's constructor already takes ``output``
            return          # (a @dataclass adds its own __init__ only after this runs)
        cls._OWN_OUTPUT = "output" in _init_parameters(init)
        own = cls._OWN_OUTPUT

        @functools.wraps(init)
        def __init__(self, *args, **kw):
            _translate(cls, kw)   # idempotent: a subclass's wrapper may have translated them already
            output = kw.pop("output", None) if not own else None
            init(self, *args, **kw)
            if output is not None:
                self.output = _check_output(output)

        cls.__init__ = __init__

    def __getattr__(self, name):
        new = _aliases(type(self)).get(name)
        if new is None or name.startswith("__"):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self, new)

    def __setattr__(self, name, value):
        cls = type(self)
        name = _aliases(cls).get(name, name)
        if name not in self.__dict__ and name[:1] != "_" and not hasattr(cls, name):
            # a new public name that is nearly a setting: ``m.cutof = 9`` used to be kept and
            # ignored. The modifier's own methods may name their attributes as they like.
            meant = _misspelled(cls, name)
            if meant is not None and not _in_method_of(self, sys._getframe(1)):
                raise AttributeError(f"{cls.__name__} has no setting {name!r}. Did you mean {meant!r}? "
                                     f"Settings: {', '.join(_settings(cls))}")
        object.__setattr__(self, name, value)

    def _input_token(self):
        """What the result of this step depends on that ``repr(self)`` does not show, for
        :meth:`ComputeContext.upstream_key`: None (nothing, the constructor arguments say it
        all), a hashable value (:func:`_token`), or ``_UNKEYABLE`` when it cannot be told: the
        steps below then never reuse what they computed from this one."""
        return None

    def modify(self, data: DataCollection, ctx: ComputeContext):
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__

    def parameters(self) -> dict:
        """Constructor arguments and their current values (used by the GUI and sessions)."""
        import inspect

        sig = inspect.signature(type(self).__init__)
        params = {k: getattr(self, k) for k in list(sig.parameters)[1:]
                  if hasattr(self, k) and sig.parameters[k].kind not in (inspect.Parameter.VAR_KEYWORD,
                                                                         inspect.Parameter.VAR_POSITIONAL)}
        if not type(self)._OWN_OUTPUT and self.output is not None:
            params["output"] = self.output   # every step's keyword (D106), last
        return params

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


def _no_arguments_signature():
    import inspect

    return inspect.Signature([inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)])


Modifier.__init__.__signature__ = _no_arguments_signature()   # ``output=`` is every step's, not listed


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

    def _input_token(self):
        # repr(self) shows the function's address only: the same function computes something
        # else once a variable it reads has changed
        return _token((self.function, self.params))

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
        _check_result(self.name, out)
        target = out if isinstance(out, DataCollection) else data
        for k, png in enumerate(figures):
            name = f"Figure: {self.name}" + (f" ({k + 1})" if k else "")
            target.attributes[name] = StepFigure(png, self.name)
        return out


def _check_result(step: str, out):
    """Raise TypeError unless ``out`` is what a step may return (``return data.positions * 2``
    used to fail later, in whatever read the array as a frame first)."""
    if out is not None and not isinstance(out, DataCollection):
        raise TypeError(f"step {step!r} returned {type(out).__name__}: a step sets its result on data "
                        "(data.positions = ..., data['Name'] = values) and returns None, or returns a "
                        "DataCollection")


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
        return self._ctx.input_frames(self.indices)

    def __repr__(self):
        return f"Frames({len(self)} frames)"


class AllFramesModifier(PythonModifier):
    """A Python step that reads every frame once: ``f(frames, data)`` (see :func:`all_frames`).

    The first evaluation runs ``f`` with :class:`Frames` and the data of the current frame and
    keeps what it set (tables, attributes, per-particle properties); every later frame gets the
    kept result without running ``f`` again, until a step above it, the source or its
    parameters change. Plots of the result then only move their frame marker during playback."""

    def modify(self, data, ctx):
        # the function and its parameters count as in upstream_key: by what they compute from,
        # not by the function's address
        pipe = ctx.pipeline
        upstream = ctx.upstream_key()
        cache = self.__dict__.get("_result")
        if cache is None or cache[0] != (upstream, pipe._step_token(self)):
            # the same function in a new pipeline running the same cells (Run all) computed it
            kept = pipe._shared_get(("all-frames", upstream, pipe._step_token(self)))
            if kept is not None:
                cache = self._result = ((upstream, pipe._step_token(self)), kept)
        if cache is None or cache[0] != (upstream, pipe._step_token(self)):
            before = _contents(data)
            out = self._call(Frames(ctx), data, ctx)
            result = out if isinstance(out, DataCollection) else data
            # the token as the run left it: a function that fills a list of its own
            # (``calls.append(...)``) would never be reused under the one from before
            cache = ((upstream, pipe._step_token(self, again=True)), _changes(before, result))
            self._result = cache
            pipe._shared_put(("all-frames", *cache[0]), cache[1])
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


def _steps(value, call: str):
    """``value`` as what goes into a pipeline's list of steps: an iterable of them. One step,
    text or anything that is not iterable is the mistake ``pipe.modifiers = step``."""
    if isinstance(value, (Modifier, str, bytes)) or callable(value) or not hasattr(value, "__iter__"):
        raise TypeError(f"pipe.modifiers is a list of steps: {call}, or pipe.add(step)")
    return value


class ModifierList(list):
    """List that wraps plain callables in PythonModifier on insertion."""

    def __init__(self, modifiers=()):
        super().__init__(self._wrap(m) for m in modifiers)

    @staticmethod
    def _wrap(m):
        if isinstance(m, Modifier):
            return m
        if isinstance(m, type):
            # a class is callable, so it used to be wrapped as a Python step and fail at compute
            name = m.__name__
            if issubclass(m, Modifier):
                raise TypeError(f"{name} is the modifier class, not a step: call it, pipe.add({name}(...))")
            raise TypeError(f"{name} is a class, not a step: add a Modifier instance or a function f(frame, data)")
        if callable(m):
            return AllFramesModifier(m) if getattr(m, "_ewald_frames", None) == "all" else PythonModifier(m)
        raise TypeError(f"{m!r} is not a Modifier or callable")

    def append(self, m):
        super().append(self._wrap(m))

    def insert(self, i, m):
        super().insert(i, self._wrap(m))

    def extend(self, ms):
        super().extend(self._wrap(m) for m in _steps(ms, "pipe.modifiers += [step]"))

    def __iadd__(self, ms):   # pipe.modifiers += [f]
        self.extend(ms)
        return self

    def __setitem__(self, i, m):
        super().__setitem__(i, [self._wrap(x) for x in m] if isinstance(i, slice) else self._wrap(m))


def _upstream(data: DataCollection) -> tuple:
    """What a step's input holds, for :func:`_keep_upstream`."""
    return (dict(data.particles), {k: dict(v) for k, v in data.tables.items()}, dict(data.attributes),
            dict(getattr(data, "units", {})))


def _keep_upstream(mod, upstream: tuple, data: DataCollection):
    """A step a verb added (``rdf(pipe)``, ``cna(pipe)``, see :mod:`ewald.verbs`) never replaces what
    the steps before it wrote: ``rdf(pipe, cutoff=6)`` after the user's own
    ``CoordinationAnalysis(cutoff=3)`` must not change their "Coordination". Particle properties and
    attributes that existed before keep their values; a table of the same name keeps the earlier one,
    and the verb's goes in beside it as ``"<name> (<verb>)"`` (``mod._verb_tables`` maps the names)."""
    particles, tables, attributes, units = upstream
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
            from .units import rename_table

            rename_table(data, name, new)   # the step's units go with its table, the earlier ones stay
            data.units.update({k: u for k, u in units.items() if isinstance(k, tuple) and k[0] == name})
    mod._verb_tables = renamed


#: Names several steps write on purpose, each over the last one: what the atoms look like, what
#: is selected, the structure itself. ``output=`` leaves them as they are and replacing one is no
#: overwrite to warn about (D106).
SHARED_NAMES = frozenset({
    "Position", "Particle Type", "Particle Identifier", "Periodic Image", "Velocity", "Force", "Mass",
    "Charge", "Color", "Selection", "Transparency", "Radius", "Color.rule",
})


def _shared(name: str) -> bool:
    return name in SHARED_NAMES or name.startswith("ColorCoding.")


def _refs(data: DataCollection) -> tuple:
    """What a collection holds before a step (the arrays themselves, not copies)."""
    return (data.count, dict(data.particles), dict(data.attributes), dict(data.tables), dict(data.units))


def _differs(old, new) -> bool:
    """Whether ``new`` holds other values than ``old`` (a step that rebuilt an array with the
    same numbers did not overwrite it)."""
    import numpy as np

    if old is new:
        return False
    if isinstance(old, dict) or isinstance(new, dict):
        return not (isinstance(old, dict) and isinstance(new, dict) and old.keys() == new.keys()
                    and not any(_differs(old[k], new[k]) for k in old))
    try:
        if isinstance(old, np.ndarray) or isinstance(new, np.ndarray):
            a, b = np.asarray(old), np.asarray(new)
            if a.shape != b.shape:
                return True
            try:
                return not np.array_equal(a, b, equal_nan=True)
            except TypeError:   # text: no NaN
                return not np.array_equal(a, b)
        return bool(old != new)
    except Exception:  # noqa: BLE001 - values that do not compare: count them as changed
        return True


def _apply_output(mod, refs: tuple, data: DataCollection) -> dict:
    """Move what ``mod`` wrote to names with ``" (<mod.output>)"`` after them and put back what
    the steps above had under the plain names (D106). Returns ``{"properties"|"attributes"|
    "tables": {old: new}}``."""
    from .units import rename_table

    count, props, attrs, tables, units = refs
    tag = f" ({mod.output})"
    same = data.count == count     # a step that changed the atom count replaced every array
    moved = {"properties": {}, "attributes": {}, "tables": {}}
    for kind, before, now in (("properties", props, data.particles), ("attributes", attrs, data.attributes),
                              ("tables", tables, data.tables)):
        for name in [k for k, v in now.items() if not _shared(k) and before.get(k) is not v]:
            if kind == "properties" and name in before and not same:
                continue
            new = name + tag
            items = []
            for k, v in now.items():   # the earlier value in its place, this step's right after it
                if k == name:
                    items += [(name, before[name])] if name in before else []
                    items.append((new, v))
                elif k != new:
                    items.append((k, v))
            now.clear()
            now.update(items)
            moved[kind][name] = new
            if kind == "tables":
                rename_table(data, name, new)
                data.units.update({k: u for k, u in units.items() if isinstance(k, tuple) and k[0] == name})
            elif kind == "attributes" and name in data.units:
                data.units[new] = data.units.pop(name)
                if name in units:
                    data.units[name] = units[name]
    rule = data.attributes.get("Color.rule")
    if isinstance(rule, dict) and rule is not attrs.get("Color.rule") and rule.get("property") in moved["properties"]:
        data.attributes["Color.rule"] = {**rule, "property": moved["properties"][rule["property"]]}
    mod._output_names = moved
    return moved


def _note_writer(writers: dict, mod):
    """Record that ``mod`` wrote what its :attr:`Modifier.last_output` lists."""
    out = mod.last_output or {}
    for kind in ("properties", "attributes", "tables"):
        for name in out.get(kind, ()):
            writers[(kind, name)] = mod


def _check_overwrite(mod, refs: tuple, data: DataCollection, writers: dict):
    """Warn (once per name) when ``mod`` replaced a property, attribute or table that an earlier
    step wrote, with other values: the earlier result is gone below this step (D106). Steps that
    name their results (``output=``, a step's own ``output`` argument) and Python steps, which
    write the names they mean to, are not checked."""
    if mod.output is not None or isinstance(mod, PythonModifier) or data.count != refs[0]:
        return
    _, props, attrs, tables, _ = refs
    lost = []
    for kind, label, before, now in (("properties", "property", props, data.particles),
                                     ("attributes", "attribute", attrs, data.attributes),
                                     ("tables", "table", tables, data.tables)):
        for name, value in now.items():
            earlier = writers.get((kind, name))
            if (earlier is not None and earlier is not mod and name in before and not _shared(name)
                    and _differs(before[name], value)):
                lost.append((kind, label, name, earlier))
    warned = mod.__dict__.setdefault("_overwrite_warned", set())
    lost = [x for x in lost if (x[0], x[2]) not in warned]
    if not lost:
        return
    warned.update((x[0], x[2]) for x in lost)
    import warnings

    steps = " and ".join(dict.fromkeys(("an earlier " if x[3].name == mod.name else "") + x[3].name for x in lost))
    names = ", ".join(f"{label} {name!r}" for _, label, name, _ in lost)
    first = lost[0][2]
    warnings.warn(f"{mod.name} replaced what {steps} wrote above it: {names}. To keep both, name this "
                  f"step's results: {mod.name}(..., output='name') writes {first + ' (name)'!r} instead.",
                  UserWarning, stacklevel=2)


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
_NOTHING = object()



class _Same:
    """Token of a value whose state cannot be read off it (a random generator, an open file, a
    pipeline): equal to the token of that same object only. It holds the object, so the address
    never becomes another object's. A copy or pickle of it equals nothing."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return type(other) is _Same and other.value is self.value

    def __hash__(self):
        return id(self.value)

    def __repr__(self):
        return f"<the {type(self.value).__name__} at {id(self.value):#x}>"

    def __reduce__(self):
        return (_Same, (object(),))

    def __deepcopy__(self, memo):
        return _Same(object())


_HASHED_BYTES = 1 << 18   # a read-only array above this goes into a token by identity, not by content
_PLAIN = (list, tuple, set, frozenset, dict)


def _freeze(value, depth: int = 0, tokens=None):
    """A hashable stand-in for a modifier setting, or ``_UNKEYABLE`` when its state can't be
    read off the value (functions, arbitrary objects): such a modifier is always recomputed.
    With ``tokens`` (:class:`_Tokens`) the value is one a Python step computes from: what is
    not plain data gets a stand-in from there, and nothing is ``_UNKEYABLE``."""
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    if depth > 6:
        return _UNKEYABLE if tokens is None else _Same(value)
    import enum
    import os

    import numpy as np

    if isinstance(value, (np.generic, enum.Enum, os.PathLike)):
        return (type(value).__name__, str(value))
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            token = _freeze(value.tolist(), depth + 1, tokens) if tokens is None or depth < 6 else _Same(value)
        elif tokens is not None and value.nbytes > _HASHED_BYTES and _read_only(value):
            token = ("ndarray", value.shape, value.dtype.str, _Same(value))   # nobody can change it
        else:
            token = ("ndarray", value.shape, value.dtype.str, _digest(value))
        return token if tokens is None else tokens.with_state(value, np.ndarray, token, depth)
    if isinstance(value, _PLAIN):
        if tokens is not None and isinstance(value, (list, tuple)) and len(value) > 1024:
            try:   # a long list of numbers: as one array, not element by element
                array = np.asarray(value)
            except Exception:  # noqa: BLE001 - ragged or mixed: element by element
                array = None
            if array is not None and array.dtype.kind in "biufc":
                return (type(value).__name__, array.shape, array.dtype.str, _digest(array))
        items = sorted(value.items(), key=lambda kv: repr(kv[0])) if isinstance(value, dict) else \
            sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        out = []
        for item in items:
            # a dict's pairs are made here: their two parts are what the dict holds
            f = tuple(_freeze(part, depth + 1, tokens) for part in item) if isinstance(value, dict) else \
                _freeze(item, depth + 1, tokens)
            if f is _UNKEYABLE or isinstance(value, dict) and any(part is _UNKEYABLE for part in f):
                return _UNKEYABLE
            out.append(f)
        token = (type(value).__name__, tuple(out))
        return token if tokens is None else tokens.with_state(value, _PLAIN, token, depth)
    return _UNKEYABLE if tokens is None else tokens.other(value, depth)


def _digest(array) -> bytes:
    import hashlib

    import numpy as np

    return hashlib.blake2b(np.ascontiguousarray(array).view(np.uint8).ravel(), digest_size=16).digest()


def _read_only(array) -> bool:
    """Whether nobody can write into ``array``: it and the arrays it is a view of are read-only
    (the arrays of a frame; a file mapped with ``mmap_mode="r"``)."""
    import mmap

    import numpy as np

    while isinstance(array, np.ndarray):
        if array.flags.writeable:
            return False
        array = array.base
    return array is None or isinstance(array, (bytes, mmap.mmap))


def _state(value):
    """The instance attributes of ``value`` (its ``__dict__`` and slots), or None when it keeps
    its state where Python cannot read it (an object of a compiled extension)."""
    state, found = {}, False
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        state.update(attributes)
        found = True
    for klass in type(value).__mro__:
        slots = klass.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in ("__dict__", "__weakref__"):
                continue
            found = True
            try:
                state[name] = getattr(value, name)
            except AttributeError:   # not set, or a name the class mangles
                pass
    return state if found else None


_TOKEN_DEPTH = 4       # functions followed from a step's function (helper of a helper ...)
_TOKEN_OBJECTS = 2000  # objects whose attributes are read for one token; the rest go by identity


def _token(value):
    """Fingerprint of a value a Python step computes from, for the keys all-frames steps cache
    under: equal tokens mean the step computes the same.

    - Plain data as in :func:`_freeze`: numbers, strings, containers, arrays by content (a large
      read-only array by identity, since nobody can change it).
    - A function by its code and what the code reads (:meth:`_Tokens.function`).
    - A module by name, and so is what installed code exports (:func:`_installed`).
    - An object by its class and its attributes (a frame, a dataclass, ``SimpleNamespace``).
    - By identity (:class:`_Same`) what is left: an object whose state Python cannot read (a
      random generator, an open file), a pipeline, a class, a helper more than ``_TOKEN_DEPTH``
      calls away. A change inside one of those is not seen.
    """
    return _Tokens().of(value)


def _function_token(function):
    """:func:`_token` of a Python function."""
    return _Tokens().function(function)


class _Tokens:
    """One fingerprint in the making (:func:`_token`)."""

    def __init__(self):
        self.functions = []            # being fingerprinted, outermost first
        self.left = _TOKEN_OBJECTS

    def of(self, value, depth: int = 0):
        return _freeze(value, depth, self)

    def with_state(self, value, plain, token, depth: int):
        """``token`` of a container or array, plus the attributes a subclass of it carries (the
        ``attributes`` of a :class:`~ewald.Table`, the mask of a masked array)."""
        if type(value) in (plain if isinstance(plain, tuple) else (plain,)):
            return token
        if depth > 5:
            return (token, _Same(value))
        try:
            state = _state(value)
        except Exception:  # noqa: BLE001
            return (token, _Same(value))
        return (token, self.of(state, depth + 1)) if state else (type(value).__name__, token)

    def other(self, value, depth: int):
        """Token of what is not plain data. Deep inside a structure it is the object itself:
        the parts a token is made from must not go deeper than :func:`_freeze` reads."""
        try:
            token = self._other(value, depth) if depth < 6 else None
        except Exception:  # noqa: BLE001 - an object that cannot even be asked for its name
            token = None
        return _Same(value) if token is None else token

    def _other(self, value, depth: int):
        import collections
        import datetime
        import decimal
        import fractions
        import inspect
        import re

        import numpy as np

        if isinstance(value, types.ModuleType):
            return ("module", value.__name__)
        if value is Ellipsis or value is NotImplemented:
            return (repr(value),)
        if isinstance(value, range):
            return ("range", value.start, value.stop, value.step)
        if isinstance(value, slice):
            return self.of(("slice", value.start, value.stop, value.step), depth + 1)
        if isinstance(value, bytearray):
            return ("bytearray", bytes(value))
        if isinstance(value, re.Pattern):
            return ("re", value.pattern, value.flags)
        if isinstance(value, (datetime.date, datetime.time, datetime.timedelta, datetime.tzinfo,
                              decimal.Decimal, fractions.Fraction, np.dtype)):
            return (type(value).__name__, repr(value))
        if isinstance(value, collections.deque):
            return self.of(("deque", tuple(value), value.maxlen), depth + 1)
        if isinstance(value, functools.partial):
            return self.of(("partial", value.func, value.args, value.keywords), depth + 1)
        if isinstance(value, types.MethodType):
            return self.of(("method", value.__func__, value.__self__), depth + 1)
        module = getattr(value, "__module__", None)
        name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
        if isinstance(module, str) and isinstance(name, str) and _installed(module):
            found = sys.modules.get(module)
            for part in name.split("."):
                found = getattr(found, part, None)
            if found is value:   # numpy.linalg.norm, math.sqrt, ewald.DataCollection
                return ("installed", module, name)
        if isinstance(value, types.FunctionType):
            return self.function(value)
        if isinstance(value, (type, Pipeline)):
            return None
        if isinstance(value, types.BuiltinFunctionType):   # rng.normal, a list's append
            owner = getattr(value, "__self__", None)
            return None if owner is None else ("builtin", value.__name__, self.of(owner, depth + 1))
        for inner in ("__wrapped__", "py_func"):   # functools.lru_cache, numba.njit
            wrapped = inspect.getattr_static(value, inner, None)
            if isinstance(wrapped, types.FunctionType):
                return ("wrapped", type(value).__name__, self.function(wrapped))
        return self.instance(value, depth)

    def instance(self, value, depth: int):
        """Token of an object from its class and its attributes; None when they cannot be read."""
        if self.left <= 0:
            return None
        self.left -= 1
        state = _state(value)
        if state is None:
            return None
        if isinstance(value, Modifier):   # its settings, not what it kept from the last run
            state = {k: v for k, v in state.items() if not k.startswith("_") and k not in _RUN_STATE}
        return ("object", self.of(type(value), depth + 1), self.of(state, depth + 1))

    def function(self, function):
        """Token of a Python function: a digest of its code (constants and nested functions
        included) and the tokens of its defaults, its closure cells and the module globals the
        code reads, so it changes when any of them does. Functions among those are followed
        ``_TOKEN_DEPTH`` deep (further ones count by identity); one that is being
        fingerprinted already (recursion) is not followed again."""
        if any(function is f for f in self.functions):
            return ("recursion",)
        if len(self.functions) >= _TOKEN_DEPTH:
            return _Same(function)
        digest, names = _code_info(function.__code__)
        values = [("defaults", function.__defaults__), ("kwdefaults", function.__kwdefaults__)]
        for k, cell in enumerate(function.__closure__ or ()):
            try:
                values.append((k, cell.cell_contents))
            except ValueError:   # a variable the enclosing function has not set yet
                values.append((k, "<empty cell>"))
        scope = function.__globals__
        values += [(name, scope[name]) for name in names if name in scope]   # the others are builtins
        self.functions.append(function)
        try:
            return (digest, *((name, self.of(value)) for name, value in values))
        finally:
            self.functions.pop()


@functools.lru_cache(maxsize=256)
def _code_info(code) -> tuple:
    """Digest of a code object and the names it reads as globals, both with the code nested
    in it (lambdas, inner functions, comprehensions). Attribute names are not globals:
    ``data.cell`` does not read a variable ``cell``."""
    import dis
    import hashlib
    import marshal

    names, todo = set(), [code]
    while todo:
        c = todo.pop()
        names.update(i.argval for i in dis.get_instructions(c)
                     if i.opname.startswith("LOAD_") and i.opname.endswith(("GLOBAL", "GLOBALS", "NAME"))
                     and isinstance(i.argval, str))
        todo.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
    # version 2 writes no back-references, which depend on reference counts
    return hashlib.blake2b(marshal.dumps(code, 2), digest_size=16).digest(), tuple(sorted(names))


@functools.lru_cache(maxsize=None)
def _installed(module: str) -> bool:
    """Whether ``module`` is installed code (the standard library, site-packages, ewald itself)
    rather than the user's script, notebook or own module. What installed code exports is
    taken as fixed and goes into a token by name, as the module does: ``from numpy.linalg
    import norm`` counts like ``np.linalg.norm``. The user's own functions are read."""
    import os
    import pathlib

    top = module.partition(".")[0]
    if top == "__main__":   # the script or notebook, even when started as ``python -m package``
        return False
    if top in sys.stdlib_module_names or top in ("ewald", "builtins"):
        return True
    file = getattr(sys.modules.get(module), "__file__", None)
    if not file:
        return False
    path = pathlib.Path(os.path.realpath(file))
    return any(path.is_relative_to(root) for root in _install_roots())


@functools.lru_cache(maxsize=1)
def _install_roots() -> tuple:
    """The folders packages are installed into: the interpreter's and the user's site-packages,
    and the folder a frozen app unpacks its packages into."""
    import os
    import sysconfig

    roots = [getattr(sys, "_MEIPASS", None)]
    try:
        roots += [sysconfig.get_paths().get(k) for k in ("stdlib", "platstdlib", "purelib", "platlib")]
        site = sys.modules.get("site")   # not there in a frozen app
        roots += [*getattr(site, "getsitepackages", list)(), getattr(site, "getusersitepackages", str)()]
    except Exception:  # noqa: BLE001 - an interpreter without an install scheme: fewer roots
        pass
    return tuple(dict.fromkeys(os.path.realpath(r) for r in roots if r))


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


def _stamps_or_same(m):
    stamps = _file_stamps(m)
    return _Same(m) if stamps is None else stamps


def _shared_step_key(m, key):
    """``key`` (:func:`_modifier_key` of ``m``) without the step's identity, plus the files its
    settings name (:func:`_file_stamps`): what a step in another pipeline, made by running the
    same cell again, has too. None when the files cannot be told (a remote path)."""
    if key[0] == "off":
        return ("off",)
    stamps = _file_stamps(m)
    return None if stamps is None else (f"{type(m).__module__}.{type(m).__qualname__}", key[2], stamps)


_GLOB = frozenset("*?[")


def _file_stamps(m) -> tuple | None:
    """(path, size, mtime in ns) of every local file a setting of ``m`` names (a string or path,
    alone or in a list; wildcards expanded), so a result computed from a file is not reused
    once the file changed. None when a setting names a remote file, whose stamp would cost a
    round trip: results of such a step stay in their own pipeline."""
    import glob
    import os

    names = []
    for k, v in vars(m).items():
        if k.startswith("_") or k in _RUN_STATE:
            continue
        for item in (v if isinstance(v, (list, tuple)) else (v,)):
            if isinstance(item, os.PathLike):
                item = os.fspath(item)
            if isinstance(item, str) and 0 < len(item) < 4096 and "\n" not in item and "\0" not in item:
                names.append(item)
    if not names:
        return ()
    from .io.remote import is_remote

    stamps = []
    for name in names:
        if is_remote(name):
            return None
        try:
            found = sorted(glob.glob(name)) if _GLOB.intersection(name) else [name]
        except (OSError, ValueError, TypeError):
            continue
        for path in found:
            try:
                st = os.stat(path)
            except (OSError, ValueError):
                continue
            if os.path.isfile(path):
                stamps.append((os.path.normcase(os.path.abspath(path)), st.st_size, st.st_mtime_ns))
    return tuple(stamps)


def _portable(key) -> bool:
    """Whether ``key`` holds contents only, so it can be equal to a key made in another
    pipeline: no object by identity (:class:`_Same`, a bare ``object()`` token)."""
    if isinstance(key, _Same) or type(key) is object:
        return False
    if isinstance(key, (tuple, list, frozenset)):
        return all(_portable(k) for k in key)
    return True


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
        state.pop("_trajectories", None)
        state.pop("_token_memo", None)    # per compute, and keyed by id() like the next
        state.pop("_step_tokens", None)
        state.pop("_restless", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache_lock, self._compute_lock = threading.RLock(), threading.RLock()
        self.__dict__.setdefault("_outputs", OrderedDict())
        self.__dict__.setdefault("cache_outputs", True)

    @property
    def modifiers(self) -> ModifierList:
        """The steps, in order (a :class:`ModifierList`: functions put into it become Python
        steps). Assigning any iterable does the same: ``pipe.modifiers = [f, CoordinationAnalysis(3.0)]``."""
        return self.__dict__["modifiers"]

    @modifiers.setter
    def modifiers(self, value):
        self.__dict__["modifiers"] = value if isinstance(value, ModifierList) else \
            ModifierList(_steps(value, "pipe.modifiers = [step]"))

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
        """Unmodified data of ``frame`` as the source gives it (negative counts from the end, out of
        range raises IndexError, as in :meth:`compute`; ``source_frame(999)`` on 10 frames used
        to give frame 9)."""
        return self._source_frame(self._frame_index(frame))

    def _source_frame(self, frame: int) -> DataCollection:
        # A pipeline can have more frames than its source (LoadTrajectory puts a trajectory on a
        # one-frame structure file): each of them starts from a frame of the source, hence the modulo.
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
            self._trim_cache(data)
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
        with self._compute_lock:
            before = self.num_frames
            self.refresh_source()
            return self.num_frames - before

    def refresh_source(self, source=None) -> int | None:
        """:meth:`refresh`, returning the first frame that may have changed (None: nothing did).
        The source is refreshed under the compute lock, so a worker thread computing a frame
        never reads a reader whose frame index is being extended (live reload runs this on a
        pool thread while the GUI computes on its own worker). ``source``: refresh only if it is
        still this pipeline's source (a probe started before another file was opened)."""
        with self._compute_lock:  # not while a worker thread reads the source
            if source is not None and source is not self.source:
                return None
            refresh = getattr(self.source, "refresh", None)
            if refresh is None:
                return None
            first = refresh()
            if first is not None:
                self.invalidate(first)
            return first

    def invalidate(self, first: int = 0):
        """Forget cached source frames from ``first`` on (their file changed), and every
        cached modifier result. Steps that failed for a missing package try again."""
        for mod in self.modifiers:
            mod.__dict__.pop("_missing_dependency", None)
        with self._cache_lock:
            for frame in [f for f in self._cache if f >= first]:
                del self._cache[frame]
            self._outputs.clear()
            self.__dict__.pop("_trajectories", None)   # the stacked frames trajectory steps share

    def clear_caches(self):
        """Start over: forget every cached source frame and step result (:meth:`invalidate`) and
        what each modifier kept across frames (:meth:`Modifier.forget`), so the next
        :meth:`compute` recomputes everything. The GUI's View › Redraw from scratch calls it.
        What other pipelines kept for the same files (:mod:`ewald.result_store`) goes too."""
        source = self._source_key()
        if source is not None:
            from .result_store import shared

            shared.drop(source)
        self.invalidate()
        for mod in self.modifiers:
            mod.forget()

    def _trim_cache(self, added: DataCollection | None = None):
        """Drop the oldest entries above the count and memory limits; ``added`` is the entry
        just stored."""
        while len(self._cache) > max(self._cache_size, 1):
            self._cache.popitem(last=False)
        while len(self._outputs) > 4 * max(self._cache_size, 1):  # results of several steps per frame
            self._outputs.popitem(last=False)
        if self.cache_memory is not None:
            # Adding up every cached array on every stored frame cost more than reading a small
            # frame (a sweep over 20000 frames of 200 atoms). The largest entry stored so far
            # times the number of entries bounds the total: under the budget, nothing to count.
            if added is not None:
                size = sum(getattr(a, "nbytes", 0) for a in added.particles.values())
                self._entry_bytes = max(self.__dict__.get("_entry_bytes", 0), size)
                if (len(self._cache) + len(self._outputs)) * self._entry_bytes <= self.cache_memory:
                    return

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

    #: Whether results are also kept by content across pipelines (:mod:`ewald.result_store`),
    #: so a notebook's Run all reuses what the cells computed before.
    share_results = True

    def _source_key(self):
        """The source's content key (``FileSource.content_key``), or None."""
        take = getattr(self.source, "content_key", None)
        if take is None:
            return None
        try:
            return take()
        except Exception:  # noqa: BLE001 - not told: results stay in this pipeline
            return None

    def _sharing(self) -> bool:
        return self.__dict__.get("cache_outputs", True) and self.share_results

    def _shared_get(self, key):
        from .result_store import shared

        return shared.get(key) if self._sharing() and _portable(key) else None

    def _shared_put(self, key, value):
        from .result_store import shared

        if self._sharing() and _portable(key):
            shared.put(key, value, tag=self._source_key())

    def _shared_result(self, key, compute):
        from .result_store import shared

        if not self._sharing() or not _portable(key):
            return compute()
        hit = shared.get(key, _NOTHING)
        if hit is not _NOTHING:
            return hit
        value = compute()
        shared.put(key, value, tag=self._source_key())
        return value

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
        with self._compute_lock, self._tokens_once():
            return self._compute(frame, device, errors, upto, cancel)

    @contextlib.contextmanager
    def _tokens_once(self):
        """While the outermost :meth:`compute` runs, each step's token (:meth:`_step_token`) is
        taken once: a step that reads every frame computes the steps above it for each of them,
        and must find the same key at all of those."""
        if "_token_memo" in self.__dict__:
            yield
            return
        memo = self.__dict__["_token_memo"] = {}
        try:
            yield
        finally:
            self.__dict__.pop("_token_memo", None)
            self._note_restless(memo)

    def _step_token(self, mod, again: bool = False):
        """:meth:`Modifier._input_token` of ``mod`` as it goes into a cache key; a fresh
        ``object()``, equal to nothing stored, when it cannot be told. When the token differs
        from the last one seen, the stacked trajectory the trajectory steps share is dropped
        as well: its key names the steps above by ``repr``, which does not show the change.
        ``again``: take it anew (the step has just run), not the one of this compute."""
        memo = self.__dict__.get("_token_memo")
        if memo is not None and id(mod) in memo and not again:
            return memo[id(mod)][1]
        token = self._take_token(mod)
        last = self.__dict__.setdefault("_step_tokens", {})
        changed = last.get(id(mod), (None, None))[1] != token
        if memo is not None:
            memo[id(mod)] = (mod, token, changed or (id(mod) in memo and memo[id(mod)][2]))
        if changed:
            if len(last) > 4 * len(self.modifiers) + 16:   # steps that were removed since
                here = {id(m) for m in self.modifiers}
                for state in (last, self.__dict__.get("_restless", {})):
                    for key in [k for k in state if k not in here]:
                        del state[key]
            last[id(mod)] = (mod, token)
            with self._cache_lock:
                self.__dict__.pop("_trajectories", None)
        return token

    _SLOW_TOKEN = 0.05   # seconds

    def _take_token(self, mod):
        if self.__dict__.get("_restless", {}).get(id(mod), (None, 0))[1] >= 2:
            return _Same(mod)
        start = time.perf_counter()
        try:
            token = mod._input_token()
        except Exception:  # noqa: BLE001 - not told is the safe answer
            token = _UNKEYABLE
        if token is _UNKEYABLE:
            return object()
        spent = time.perf_counter() - start
        if spent > self._SLOW_TOKEN and not getattr(mod, "_slow_token_told", False):
            import warnings

            mod._slow_token_told = True
            warnings.warn(
                f"step {mod.name!r}: checking whether what it reads has changed takes {spent:.2f} s at "
                "every compute, because an all-frames step below it reuses its result only while "
                "that is unchanged. The step reads large arrays or lists that can be written to: "
                "make an array read-only (array.flags.writeable = False) and it is not read again.",
                RuntimeWarning, stacklevel=2)
        return token

    def _note_restless(self, memo: dict):
        """After a compute in which a step's token changed, take it once more: when it differs
        again, the step changes what it reads as it runs (it appends to a list, counts its
        calls), and no result computed below it would ever be reused. Seen in two computes,
        such a step counts by identity from then on, like an object that cannot be read."""
        for mod, token, changed in memo.values():
            if not changed or type(token) in (object, _Same):
                continue
            try:
                now = mod._input_token()
            except Exception:  # noqa: BLE001
                continue
            if now is not _UNKEYABLE and now != token:
                restless = self.__dict__.setdefault("_restless", {})
                restless[id(mod)] = (mod, restless.get(id(mod), (mod, 0))[1] + 1)

    def _sweep_frame(self, frame: int) -> DataCollection:
        """:meth:`source_frame` for a sweep over the frames: a cached frame when there is one,
        else read and not kept."""
        frame %= self.source.num_frames
        with self._cache_lock:
            data = self._cache.get(frame)
        if data is None:
            return self.source.compute(frame).copy().freeze()   # nobody else holds it
        return data.copy()

    def _compute(self, frame, device, errors, upto, cancel, sweep: bool = False) -> DataCollection:
        """``sweep``: one frame of many for a trajectory-wide step (``ComputeContext.input_frames``):
        nothing is cached or looked up, and the steps keep their status and summary."""
        frame = self._frame_index(frame)
        backend = get_backend(device or self.device)
        mods = list(self.modifiers[:upto])
        keys = []  # settings of the leading modifiers whose output can be reused
        if self.__dict__.get("cache_outputs", True) and not sweep:
            for mod in mods:
                key = _modifier_key(mod)
                if key is None:
                    break
                keys.append(key)
        base = (id(self.source), frame, backend.name, self.num_frames)
        shared = self._shared_keys(mods, keys, frame, backend) if keys else []
        start, data = 0, None
        with self._cache_lock:
            for n in range(len(keys), 0, -1):  # the longest prefix computed before
                hit = self._outputs.get(base + (tuple(keys[:n]),))
                if hit is not None:
                    self._outputs.move_to_end(base + (tuple(keys[:n]),))
                    start, data = n, hit.copy()
                    break
        if data is None and shared:   # computed by another pipeline (Run all) from the same contents
            from .result_store import shared as store

            for n in range(len(shared), 0, -1):
                hit = store.get(shared[n - 1])
                if hit is not None:
                    data, summaries = hit[0].copy(), hit[1]
                    start = n
                    for mod, (summary, spent) in zip(mods, summaries):
                        if mod.enabled and summary is not None:
                            mod.last_output, mod.last_time, mod.status = summary, spent, None
                    with self._cache_lock:
                        self._outputs[base + (tuple(keys[:n]),)] = hit[0]
                        self._outputs.move_to_end(base + (tuple(keys[:n]),))
                    break
        if data is None:
            data = self._sweep_frame(frame) if sweep else self._source_frame(frame)
        failed = False
        writers = {}   # (kind, name) -> the step that last wrote it, for _check_overwrite
        for index, mod in enumerate(mods):
            if index < start:
                if mod.enabled and not sweep:
                    _note_writer(writers, mod)   # computed before: its last summary says what it wrote
                continue
            if index == len(keys) and keys and not failed and start < len(keys):
                self._store_output(base + (tuple(keys),), data, shared, mods)
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
            before = None if sweep else _snapshot(data)
            named = mod.output is not None and not mod._OWN_OUTPUT
            refs = _refs(data) if named or not sweep else None
            upstream = _upstream(data) if getattr(mod, "_verb", None) else None
            t0 = time.perf_counter()
            try:
                result = mod.modify(data, ctx)
                _check_result(mod.name, result)
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
            if result is not None:
                data = result
            if named:
                _apply_output(mod, refs, data)
            if upstream is not None:
                _keep_upstream(mod, upstream, data)
            if sweep:
                continue
            _check_overwrite(mod, refs, data, writers)
            mod.status = None
            mod.__dict__.pop("_missing_dependency", None)
            mod.last_time = time.perf_counter() - t0
            mod.last_output = output_summary(before, data, frame)
            _note_writer(writers, mod)
            if index < len(keys) - 1 and not failed and mod.last_time >= self.checkpoint_seconds:
                self._store_output(base + (tuple(keys[:index + 1]),), data, shared, mods)
        if len(keys) == len(mods) and keys and not failed and start < len(keys):
            self._store_output(base + (tuple(keys),), data, shared, mods)
        return data

    def _shared_keys(self, mods, keys, frame, backend) -> list:
        """Content keys (:mod:`ewald.result_store`) of the prefixes ``keys`` covers, as far as
        they can be told: empty without a file source or with sharing off."""
        if not self._sharing():
            return []
        from .result_store import ResultStore

        if not ResultStore.enabled():
            return []
        source = self._source_key()
        if source is None:
            return []
        base = ("frame", source, frame, backend.name, self.num_frames)
        out, steps = [], []
        for mod, key in zip(mods, keys):
            step = _shared_step_key(mod, key)
            if step is None:
                break
            steps.append(step)
            out.append(base + (tuple(steps),))
        return out

    def _store_output(self, key, data, shared=(), mods=()):
        with self._cache_lock:
            # ``data`` goes on to later steps and callers: freezing it keeps them from writing
            # into the cached arrays it shares; they change their own copy
            kept = self._outputs[key] = data.freeze().copy()
            self._outputs.move_to_end(key)
            self._trim_cache(data)
        n = len(key[-1])
        if len(shared) >= n:   # for a new pipeline running the same cells, with what each step did
            from .result_store import shared as store

            summaries = tuple((m.last_output, m.last_time) if m.enabled else (None, None) for m in mods[:n])
            store.put(shared[n - 1], (kept, summaries), tag=self._source_key())

    def frames(self, start: int | None = None, stop: int | None = None, step: int | None = None, device=None):
        """Iterator over the computed frames ``start:stop:step``, which count as in a slice
        (``pipe.frames(-2)``: the last two; a ``stop`` past the end ends at the last frame)."""
        try:
            indices = range(*slice(start, stop, step).indices(self.num_frames))
        except TypeError:
            bad = next((v for v in (start, stop, step) if v is not None and not hasattr(v, "__index__")), None)
            raise TypeError(f"frames(start, stop, step) takes whole numbers or None, not {bad!r}") from None
        except ValueError:
            raise ValueError("frames(step=0): the step cannot be zero") from None
        return (self.compute(i, device=device) for i in indices)

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
