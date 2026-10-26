"""Pipeline evaluation off the GUI thread.

The main window asks for a frame with :meth:`ComputeRunner.request`. The pipeline runs on one
worker thread; a newer request cancels the running one (:class:`ewald.pipeline.ComputeCancelled`
stops it between modifiers or at the next ``ctx.input_frame``), so dragging a slider or scrubbing
the timeline never queues stale work. Results arrive through the ``finished`` signal on the GUI
thread. A request that finishes within ``budget`` seconds is returned synchronously instead, so
small systems update in the same event-loop turn as before, without a flicker of the "computing"
state.

``prepare`` runs on the worker right after a frame is computed (the main window searches the
bonds the 3D view will draw there, so the GUI thread finds them in ``bonds.cached_bonds``), and
``prefetch`` names a frame to compute after it while nothing newer is asked for, so playback
finds the next frame in the pipeline's cache.
"""
from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout  # not the builtin on Python 3.10
from dataclasses import dataclass, field

from PySide6 import QtCore

from ..pipeline import ComputeCancelled

# Seconds the GUI waits for a result before it goes on and lets the worker finish in the
# background; None waits for every result (tests that read ``window.data`` right after a change).
BUDGET: float | None = 0.05


@dataclass
class Job:
    """One requested evaluation: ``data`` (or ``error``) is filled in by the worker."""

    pipeline: object
    frame: int
    device: str | None
    interactive: bool = False
    generation: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)
    data: object = None
    error: BaseException | None = None
    seconds: float = 0.0
    prepare: object = None               # callable(data), run on the worker after computing
    prefetch: int | None = None          # frame to compute next, while no newer job comes

    @property
    def cancelled(self) -> bool:
        return isinstance(self.error, ComputeCancelled)


class ComputeRunner(QtCore.QObject):
    """Runs ``pipeline.compute`` jobs on one worker thread, newest request wins."""

    finished = QtCore.Signal(object)     # Job, on the GUI thread
    busyChanged = QtCore.Signal(bool)    # True while a job runs longer than the budget

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ewald-compute")
        self._current: Job | None = None
        self._ahead: Job | None = None   # the read-ahead job running now
        self._generation = 0
        self._busy = False
        self._done.connect(self._deliver, QtCore.Qt.QueuedConnection)

    _done = QtCore.Signal(object)        # worker thread -> GUI thread

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def busy(self) -> bool:
        return self._busy

    def request(self, pipeline, frame: int, device=None, interactive: bool = False,
                budget: float | None = ..., prepare=None, prefetch: int | None = None) -> Job | None:
        """Start computing ``frame``; cancel whatever ran before. Returns the finished job when
        it completes within ``budget`` seconds (default :data:`BUDGET`), else None and the job
        comes later through ``finished``. ``prepare(data)`` runs on the worker after the frame
        is computed; ``prefetch`` is computed (and prepared) after it unless a newer request
        comes first."""
        self.cancel()
        ahead = self._ahead
        if ahead is not None and not (ahead.pipeline is pipeline and ahead.frame == frame):
            ahead.cancel.set()  # reading ahead a frame nobody wants any more
        self._generation += 1
        job = Job(pipeline, frame, device, interactive, self._generation, prepare=prepare, prefetch=prefetch)
        self._current = job
        future = self._pool.submit(self._run, job)
        budget = BUDGET if budget is ... else budget
        try:
            future.result(timeout=budget)
        except FutureTimeout:
            self._set_busy(True)
            return None
        if job.generation != self._generation:
            return None
        self._current = None
        self._set_busy(False)
        return job

    def cancel(self):
        """Stop the running job, if any; its result is dropped."""
        if self._current is not None:
            self._current.cancel.set()
            self._current = None

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the worker is idle (tests, closing the window); True if it is."""
        done = self._pool.submit(lambda: None)
        try:
            done.result(timeout=timeout)
        except FutureTimeout:
            return False
        QtCore.QCoreApplication.sendPostedEvents(self)
        return True

    def shutdown(self):
        self.cancel()
        if self._ahead is not None:
            self._ahead.cancel.set()
        self._pool.shutdown(wait=True, cancel_futures=True)

    # ------------------------------------------------------------ worker side
    def _run(self, job: Job):
        if job.cancel.is_set():
            job.error = ComputeCancelled()
        else:
            t = time.perf_counter()
            try:
                job.data = job.pipeline.compute(job.frame, device=job.device, errors="skip", cancel=job.cancel)
            except ComputeCancelled as exc:
                job.error = exc
            except Exception as exc:  # source failed (bad file/frame)
                job.error = exc
                traceback.print_exc()
            job.seconds = time.perf_counter() - t
            self._prepare(job)
        self._done.emit(job)
        if job.prefetch is not None and job.error is None:
            self._pool.submit(self._read_ahead, job)  # its own task: request() waits for this job only

    @staticmethod
    def _prepare(job: Job):
        if job.prepare is not None and job.data is not None and not job.cancel.is_set():
            try:
                job.prepare(job.data)
            except Exception:  # noqa: BLE001 - only a head start; the GUI thread redoes it and reports
                pass

    def _read_ahead(self, job: Job):
        """Compute ``job.prefetch`` into the pipeline's cache (worker thread)."""
        if job.generation != self._generation:  # a newer request is queued behind: let it run
            return
        ahead = Job(job.pipeline, job.prefetch, job.device, True, job.generation, prepare=job.prepare)
        self._ahead = ahead
        if job.generation != self._generation:  # a request came in between: it saw no read-ahead
            ahead.cancel.set()
        try:
            ahead.data = ahead.pipeline.compute(ahead.frame, device=ahead.device, errors="skip", cancel=ahead.cancel)
            self._prepare(ahead)
        except Exception:  # noqa: BLE001 - cancelled, or the frame fails: its own request reports it
            pass
        finally:
            self._ahead = None

    # ------------------------------------------------------------ GUI side
    def _deliver(self, job: Job):
        if job is not self._current or job.cancelled:
            return  # superseded, or already returned synchronously by request()
        self._current = None
        self._set_busy(False)
        self.finished.emit(job)

    def _set_busy(self, busy: bool):
        if busy != self._busy:
            self._busy = busy
            self.busyChanged.emit(busy)


_warmed = False


def warm_up_jit():
    """Load the numba kernels every file open needs (the text-table parser, the neighbor list)
    on a background thread, once per process: loading them from numba's cache costs about half a
    second on first use, and compiling them on a fresh install several seconds, which would
    otherwise land on the first file the user opens."""
    global _warmed
    if _warmed:
        return
    _warmed = True

    def run():
        try:
            import numpy as np

            from ..data import Cell
            from ..io._textparse import parse_table
            from ..neighbors import find_pairs

            parse_table(b"1 1 0.0 0.0 0.0\n", 1, 5, [0, 1, 2, 3, 4])
            pos = np.random.default_rng(0).random((64, 3)) * 8.0
            find_pairs(pos, Cell(np.eye(3) * 8.0), 2.5, method="numba")
        except Exception:  # noqa: BLE001 - only a head start; the real call reports errors
            pass

    threading.Thread(target=run, name="ewald-jit-warmup", daemon=True).start()
