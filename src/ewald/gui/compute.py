"""Pipeline evaluation off the GUI thread.

The main window asks for a frame with :meth:`ComputeRunner.request`. The pipeline runs on one
worker thread; a newer request cancels the running one (:class:`ewald.pipeline.ComputeCancelled`
stops it between modifiers or at the next ``ctx.input_frame``), so dragging a slider or scrubbing
the timeline never queues stale work. Results arrive through the ``finished`` signal on the GUI
thread. A request that finishes within ``budget`` seconds is returned synchronously instead, so
small systems update in the same event-loop turn as before, without a flicker of the "computing"
state.
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
                budget: float | None = ...) -> Job | None:
        """Start computing ``frame``; cancel whatever ran before. Returns the finished job when
        it completes within ``budget`` seconds (default :data:`BUDGET`), else None and the job
        comes later through ``finished``."""
        self.cancel()
        self._generation += 1
        job = Job(pipeline, frame, device, interactive, self._generation)
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
        self._done.emit(job)

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
