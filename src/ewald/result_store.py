"""Results kept across pipelines, by what they were computed from.

A notebook's *Run all* (and *Restart*) starts a new namespace: ``pipe = load(...)`` makes a new
:class:`~ewald.pipeline.Pipeline` with new step objects, and their own caches start empty. The
pipeline also puts the results it may reuse here, under a key made of contents only: the
source's files (local, or remote over SSH) with their size and modification time
(``FileSource.content_key``), the frame, the device, and each step's class and settings, with the
size and modification time of local files a setting names. A new pipeline over the same files
with the same steps then finds them, so Run all after editing one cell recomputes only what
that cell changes. All-frames steps (``msd``,
``rdf`` over a trajectory, Python steps that read every frame) keep their whole-trajectory
result here the same way.

Nothing is shared whose key names an object by identity (a source without files, a step
that reads a random generator or a pipeline): such a key is only ever equal inside one pipeline.
The store is in memory only (one per process: a notebook run in a separate kernel keeps its
own, and a kernel restart empties it) and bounded by :attr:`ResultStore.max_bytes` and
:attr:`ResultStore.max_entries`; ``EWALD_SHARED_RESULTS=0`` turns it off, and
``ewald.pipeline.Pipeline.clear_caches`` (View › Redraw from scratch) drops what was kept for
that pipeline's source.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict

_MISSING = object()


def _arrays(value, out: dict, depth: int = 0):
    """Add ``id -> nbytes`` of the arrays held by ``value`` (a DataCollection, a table, a
    tuple or dict of them) to ``out``."""
    if depth > 6:
        return
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int) and hasattr(value, "dtype"):
        out[id(value)] = nbytes
        return
    if isinstance(value, dict):
        for v in value.values():
            _arrays(v, out, depth + 1)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _arrays(v, out, depth + 1)
    elif hasattr(value, "particles") and hasattr(value, "tables"):   # a DataCollection
        _arrays(dict(value.particles), out, depth + 1)
        _arrays(dict(value.tables), out, depth + 1)
        for grid in getattr(value, "grids", {}).values():
            _arrays(dict(getattr(grid, "properties", {})), out, depth + 1)


class ResultStore:
    """An LRU of results by content key (see the module docstring). Each entry carries a tag,
    the content key of its source, so :meth:`drop` can forget one source's results."""

    def __init__(self, max_bytes: int | None = 512 << 20, max_entries: int = 256):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._entries: OrderedDict = OrderedDict()   # key -> (tag, value, {id: nbytes})
        self._lock = threading.Lock()
        self.hits = 0

    @staticmethod
    def enabled() -> bool:
        return os.environ.get("EWALD_SHARED_RESULTS", "1").strip().lower() not in ("0", "false", "no", "off")

    def __len__(self):
        return len(self._entries)

    def get(self, key, default=None):
        if not self.enabled():
            return default
        with self._lock:
            entry = self._entries.get(key, _MISSING)
            if entry is _MISSING:
                return default
            self._entries.move_to_end(key)
            self.hits += 1
            return entry[1]

    def put(self, key, value, tag=None):
        if not self.enabled() or self.max_entries <= 0:
            return
        sizes: dict = {}
        _arrays(value, sizes)
        if self.max_bytes is not None and sum(sizes.values()) > self.max_bytes:
            return   # would push out everything else
        with self._lock:
            self._entries[key] = (tag, value, sizes)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            if self.max_bytes is not None:
                while len(self._entries) > 1 and self._total() > self.max_bytes:
                    self._entries.popitem(last=False)

    def _total(self) -> int:
        seen: dict = {}   # an array shared by several results (a frame and steps after it) counts once
        for _, _, sizes in self._entries.values():
            seen.update(sizes)
        return sum(seen.values())

    def nbytes(self) -> int:
        with self._lock:
            return self._total()

    def drop(self, tag):
        """Forget every result of the source with content key ``tag``."""
        with self._lock:
            for key in [k for k, e in self._entries.items() if e[0] == tag]:
                del self._entries[key]

    def clear(self):
        with self._lock:
            self._entries.clear()
            self.hits = 0


#: The process's store; :class:`~ewald.pipeline.Pipeline` reads and fills it.
shared = ResultStore()
