"""Live reload: follow a trajectory a running simulation is still writing (like OVITO and VMD).

Every few seconds the pipeline source is asked for new frames (:meth:`Pipeline.refresh`,
done in a worker thread so a slow SSH link never stalls the window). When frames were
appended the timeline grows; if it sat on the last frame it moves on to the newest one.
"""
from __future__ import annotations

from PySide6 import QtCore


class _Probe(QtCore.QRunnable):
    def __init__(self, source, done):
        super().__init__()
        self.source, self.done = source, done

    def run(self):
        try:
            first, error = self.source.refresh(), None
        except Exception as exc:  # network down, file replaced by a directory, ...
            first, error = None, f"{type(exc).__name__}: {exc}"
        self.done.emit(self.source, first, error)


class LiveReload(QtCore.QObject):
    """Polls ``window.pipeline`` and grows ``window.timeline`` (see :meth:`attach`)."""

    grew = QtCore.Signal(int, int)  # first changed frame, new number of frames
    failed = QtCore.Signal(str)
    _done = QtCore.Signal(object, object, object)

    def __init__(self, parent=None, interval: float = 2.0, enabled: bool = True):
        super().__init__(parent)
        self.pipeline = None
        self.enabled = enabled
        self._busy = False
        self._error = None
        self._incomplete = []
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(int(interval * 1000))
        self.timer.timeout.connect(self.poll)
        self._done.connect(self._finished)

    @classmethod
    def attach(cls, win) -> "LiveReload":
        """Live reload for a MainWindow: follows its pipeline, extends its timeline."""
        live = cls(win)
        live.grew.connect(lambda first, n: _follow(win, first, n))
        live.failed.connect(lambda text: win.message(f"Live reload: {text}", error=True))
        return live

    def configure(self, enabled: bool, interval: float):
        self.enabled = enabled
        self.timer.setInterval(int(interval * 1000))
        self._restart()

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        self._error = None
        self._incomplete = []
        self._restart()

    def report_incomplete(self):
        """Tell the window when the files belong to a calculation still running (or stopped),
        and when such a calculation has finished."""
        notes = list(getattr(getattr(self.pipeline, "source", None), "incomplete", None) or [])
        if notes == self._incomplete:
            return
        win = self.parent()
        if win is not None and hasattr(win, "message"):
            if notes:
                win.message("Calculation incomplete: " + "; ".join(notes), warning=True)
            else:
                win.message("Calculation finished: all of the file is read")
        self._incomplete = notes

    def _restart(self):
        source = getattr(self.pipeline, "source", None)
        if self.enabled and hasattr(source, "refresh"):
            self.timer.start()
        else:
            self.timer.stop()

    def poll(self, wait: bool = False):
        """Look for new frames now (``wait=True`` runs in this thread, for tests)."""
        if wait and self._busy:  # let a look the timer started finish first
            QtCore.QThreadPool.globalInstance().waitForDone()
            QtCore.QCoreApplication.processEvents()
        if self._busy or self.pipeline is None or not hasattr(self.pipeline.source, "refresh"):
            return
        self._busy = True
        probe = _Probe(self.pipeline.source, self._done)
        if wait:
            probe.run()
            QtCore.QCoreApplication.processEvents()
        else:
            QtCore.QThreadPool.globalInstance().start(probe)

    def _finished(self, source, first, error):
        self._busy = False
        if self.pipeline is None or source is not self.pipeline.source:
            return  # another file was opened meanwhile
        if error != self._error:
            self._error = error
            if error:
                self.failed.emit(error)
        if first is not None:
            self.pipeline.invalidate(first)
            self.grew.emit(first, self.pipeline.num_frames)
        self.report_incomplete()


def _follow(win, first: int, n: int):
    tl = win.timeline
    old = tl.slider.maximum() + 1
    at_end = win.frame >= old - 1
    tl.set_num_frames(n, keep_selection=True)
    if n != old:
        win.message(f"Live reload: {n - old:+d} frame(s), {n} now")
    if (at_end and not tl.play.isChecked()) or win.frame >= n:
        tl._set(n - 1) if win.frame != n - 1 else win.recompute()
    elif win.frame >= first:
        win.recompute()
