"""The freeze report: stacks of every thread in freezes.log when the window stops responding."""
import faulthandler
import time

import pytest

pytest.importorskip("PySide6")
from PySide6 import QtWidgets  # noqa: E402

from ewald.gui.freeze_watch import FreezeWatch, log_path  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_a_long_gap_between_beats_is_logged_and_signalled(qapp, tmp_path):
    clock = Clock()
    path = tmp_path / "freezes.log"
    w = FreezeWatch(5, path=path, clock=clock, dump=False, context=lambda: "power: battery")
    seen = []
    w.froze.connect(lambda s, p: seen.append((s, p)))
    w.start()
    clock.t += 1
    w.beat()
    clock.t += 12.5
    w.beat()
    w.stop()
    text = path.read_text(encoding="utf-8")
    assert "started" in text and "did not respond for 12.5 s" in text and "power: battery" in text
    assert seen == [(12.5, str(path))] and w.freezes == [12.5]


def test_off_at_zero(qapp, tmp_path):
    w = FreezeWatch(0, path=tmp_path / "freezes.log").start()
    assert not w.running and not (tmp_path / "freezes.log").exists()


def test_faulthandler_writes_the_stuck_threads_stack(qapp, tmp_path):
    path = tmp_path / "freezes.log"
    w = FreezeWatch(0.3, path=path).start()
    try:
        time.sleep(1.0)           # the GUI thread is stuck: no beat re-arms the timer
        w.beat()
    finally:
        w.stop()
        faulthandler.cancel_dump_traceback_later()
    text = path.read_text(encoding="utf-8")
    assert "Timeout" in text and "test_faulthandler_writes_the_stuck_threads_stack" in text
    assert "did not respond" in text


def test_a_big_log_starts_over(qapp, tmp_path):
    path = tmp_path / "freezes.log"
    path.write_text("x" * 1_100_000, encoding="utf-8")
    FreezeWatch(5, path=path, dump=False).start().stop()
    assert path.stat().st_size < 1000


def test_log_lives_in_the_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("EWALD_CONFIG_DIR", str(tmp_path))
    assert log_path() == tmp_path / "freezes.log"
