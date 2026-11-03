"""Notice when the window stops responding, and write down what every thread was doing.

A freeze inside a graphics driver (the first maximize on some integrated GPUs, a laptop on
battery that has to wake or switch its GPU) or a long call on the GUI thread looks the same to
the user: the window does not react. :class:`FreezeWatch` beats once a second on the GUI thread.
Each beat re-arms :func:`faulthandler.dump_traceback_later`, whose own C thread writes every
Python thread's stack to ``freezes.log`` when no beat came for ``threshold`` seconds, even while
the GUI thread is stuck in C++ or the process is killed afterwards. When the beats come back, a
line with the length of the freeze, the power source and the 3D view's GPU follows the stacks,
and :attr:`froze` tells the window, which says so in the status bar.

Only ``ewald gui`` starts it (pytest's own ``faulthandler_timeout`` uses the same, single,
faulthandler timer).
"""
from __future__ import annotations

import faulthandler
import os
import sys
import time
from pathlib import Path

from PySide6 import QtCore

LOG_NAME = "freezes.log"
MAX_BYTES = 1_000_000   # start over above this, so the log stays small enough to send


def log_path() -> Path:
    from ..settings import config_dir

    return config_dir() / LOG_NAME


class FreezeWatch(QtCore.QObject):
    froze = QtCore.Signal(float, str)   # seconds without a beat, the log path

    BEAT_MS = 1000

    def __init__(self, threshold: float = 5.0, path=None, context=None, parent=None,
                 clock=time.monotonic, dump=True):
        super().__init__(parent)
        self.threshold = float(threshold)
        self.path = Path(path) if path is not None else log_path()
        self.context = context or (lambda: "")     # one line about the machine, at the time of a freeze
        self.clock = clock
        self.dump = dump
        self.freezes: list[float] = []
        self._file = None
        self._last = None
        self._timer = QtCore.QTimer(self, interval=self.BEAT_MS)
        self._timer.timeout.connect(self.beat)

    @property
    def running(self) -> bool:
        return self._file is not None

    def start(self):
        if self.threshold <= 0 or self.running:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self.path.stat().st_size > MAX_BYTES:
                self.path.unlink()
        except OSError:
            pass
        self._file = open(self.path, "a", encoding="utf-8", newline="\n")
        from .. import __version__

        self._write(f"EWALD {__version__} started (pid {os.getpid()}, Python {sys.version.split()[0]}, "
                    f"{sys.platform}); reporting freezes longer than {self.threshold:g} s")
        self._last = self.clock()
        self._arm()
        self._timer.start()
        return self

    def stop(self):
        self._timer.stop()
        if self.dump:
            faulthandler.cancel_dump_traceback_later()
        if self._file is not None:
            self._file.close()
            self._file = None

    def _arm(self):
        if self.dump and self._file is not None:
            self._file.flush()
            faulthandler.dump_traceback_later(self.threshold, repeat=False, file=self._file)

    def _write(self, line: str):
        self._file.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
        self._file.flush()

    def beat(self):
        """One beat of the GUI thread (the timer's slot; tests call it with a fake clock)."""
        if self._file is None:
            return
        now = self.clock()
        gap = now - self._last
        self._last = now
        if gap >= self.threshold:
            self.freezes.append(gap)
            try:
                extra = self.context()
            except Exception as exc:  # the report must not fail because a probe did
                extra = f"(no details: {exc})"
            self._write(f"the window did not respond for {gap:.1f} s (the stacks above, if any, were "
                        f"taken {self.threshold:g} s into it; a computer that slept shows up here too)"
                        + (f"; {extra}" if extra else ""))
            self.froze.emit(gap, str(self.path))
        self._arm()


def machine_context(window=None) -> str:
    """Power source and the 3D view's OpenGL renderer, for the line after a freeze."""
    from ..gpu_select import power_source

    parts = [f"power: {power_source() or 'no battery or unknown'}"]
    if window is not None:
        try:
            gl = window.viewport.opengl_info() if window.viewport.view_built else {}
            if gl.get("renderer"):
                parts.append(f"3D view: {gl['renderer']}")
            screen = window.screen()
            parts.append(f"window {window.width()}x{window.height()} at {screen.devicePixelRatio():g}x"
                         + (" maximized" if window.isMaximized() else ""))
        except Exception:
            pass
    return ", ".join(parts)
