"""Background threads that may outlive the widget that started them.

A ``QThread`` that is the child of a widget dies with it: closing a window or a dialog while the
thread still runs destroys a running thread (Qt aborts the process), and a thread that emits on
a deleted ``QObject`` writes into freed memory. :func:`detach` keeps a thread alive, unparented,
until it has finished; connect its signals to the widget's methods, which Qt disconnects when the
widget goes. :data:`relay` is the same for threads that are not ``QThread``s (``QRunnable``,
``threading``): an object that lives as long as the application, to emit through.
"""
from __future__ import annotations

import weakref

from PySide6 import QtCore

_RUNNING: set = set()


def detach(thread: QtCore.QThread) -> QtCore.QThread:
    """Unparent ``thread`` and hold it until it has finished (call before ``start``)."""
    thread.setParent(None)
    _RUNNING.add(thread)
    ref = weakref.ref(thread)
    thread.finished.connect(lambda: _reap(ref))   # runs on the GUI thread (queued)
    return thread


def _reap(ref):
    thread = ref()
    if thread is None:
        return
    try:
        thread.wait()   # finished() is emitted just before the thread ends
    except RuntimeError:  # already deleted
        pass
    # dropped on the next turn of the event loop, not inside a slot of its own signal
    QtCore.QTimer.singleShot(0, lambda: _RUNNING.discard(thread))


def alive(obj: QtCore.QObject) -> bool:
    """False once Qt has deleted the C++ side of ``obj`` (a closed tab or dialog)."""
    try:
        obj.objectName()
    except RuntimeError:
        return False
    return True


def running() -> int:
    """How many detached threads have not been released yet (tests)."""
    return len(_RUNNING)


class _Relay(QtCore.QObject):
    done = QtCore.Signal(object, object)   # token, payload


_relay: _Relay | None = None


def relay() -> _Relay:
    """The application-lifetime object worker threads emit through (create it on the GUI thread)."""
    global _relay
    if _relay is None:
        _relay = _Relay()
        app = QtCore.QCoreApplication.instance()
        if app is not None and _relay.thread() is not app.thread():
            _relay.moveToThread(app.thread())
    return _relay
