"""The "unsaved changes" question when a notebook tab or the window closes: Save (Save All for
several), Don't Save, Cancel; a cancelled or failed save keeps everything open."""
import os

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from ewald.gui import notebook_panel as npn  # noqa: E402

B = QtWidgets.QMessageBox


@pytest.fixture(scope="module")
def app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    for w in app.topLevelWidgets():
        w.close()
        w.deleteLater()
    app.processEvents()


def _click(monkeypatch, which, seen=None):
    """Make the next QMessageBox "exec" press the standard button ``which`` (None: close it)."""
    def exec_(box):
        if seen is not None:
            seen.append({b: box.button(b).text() for b in (B.Save, B.SaveAll, B.Discard, B.Cancel)
                         if box.button(b)})
            seen[-1]["default"] = box.standardButton(box.defaultButton())
            seen[-1]["escape"] = box.standardButton(box.escapeButton())
            seen[-1]["text"] = box.text()
        if which is not None:
            box.button(which).click()
        return 0
    monkeypatch.setattr(B, "exec", exec_)


def test_ask_unsaved_offers_save_dont_save_cancel(app, monkeypatch):
    seen = []
    for button, answer in ((B.Save, "save"), (B.Discard, "discard"), (B.Cancel, "cancel"), (None, "cancel")):
        _click(monkeypatch, button, seen)
        assert npn.ask_unsaved(None, "Quit", ["Notebook 1"]) == answer
    one = seen[0]
    assert set(one) - {"default", "escape", "text"} == {B.Save, B.Discard, B.Cancel}
    assert one[B.Discard] == "Don't Save"
    assert one["default"] == B.Save and one["escape"] == B.Cancel
    assert "Notebook 1" in one["text"]

    seen.clear()
    _click(monkeypatch, B.SaveAll, seen)
    assert npn.ask_unsaved(None, "Quit", ["a", "b"]) == "save"      # several: Save All
    assert B.SaveAll in seen[0] and B.Save not in seen[0] and "a\nb" in seen[0]["text"]


@pytest.fixture
def panel(app, monkeypatch):
    monkeypatch.setattr(npn, "CONFIRM_UNSAVED", True)
    p = npn.NotebookPanel()
    views = [p.new_notebook("a"), p.new_notebook("b"), p.new_notebook("clean")]
    views[0].notebook.dirty = views[1].notebook.dirty = True
    yield p, views
    monkeypatch.setattr(npn, "CONFIRM_UNSAVED", False)
    p.deleteLater()


def test_confirm_discard_saves_each_dirty_tab_or_cancels(panel, monkeypatch):
    p, (a, b, clean) = panel
    saved, result = [], {}

    def save_view():                                   # the window's Save: the current tab
        saved.append(p.view.notebook.name)
        return result.get(p.view.notebook.name, "/x.ewd")

    p.save_view = save_view
    asked = []
    for answer, ok in (("cancel", False), ("discard", True)):
        monkeypatch.setattr(npn, "ask_unsaved", lambda parent, title, names, a=answer: asked.append(names) or a)
        assert p.confirm_discard(title="Quit") is ok
    assert asked[0] == ["a", "b"] and saved == []          # only the dirty ones are listed

    monkeypatch.setattr(npn, "ask_unsaved", lambda *a: "save")
    assert p.confirm_discard() is True and saved == ["a", "b"]
    saved.clear()
    result["a"] = None                                     # Save as cancelled (or the save failed)
    assert p.confirm_discard() is False and saved == ["a"]  # b is not saved, nothing closes
    assert p.confirm_discard([clean]) is True               # nothing unsaved: no question


def test_close_tab_uses_the_same_question(panel, monkeypatch):
    p, (a, b, clean) = panel
    p.save_view = lambda: None
    monkeypatch.setattr(npn, "ask_unsaved", lambda *a: "cancel")
    p.close_tab(p.tabs.indexOf(a))
    assert a in p.views()
    monkeypatch.setattr(npn, "ask_unsaved", lambda *a: "save")
    p.close_tab(p.tabs.indexOf(a))                         # save cancelled: the tab stays
    assert a in p.views()
    monkeypatch.setattr(npn, "ask_unsaved", lambda *a: "discard")
    p.close_tab(p.tabs.indexOf(a))
    assert a not in p.views()


def test_save_without_a_window_needs_a_path(panel, monkeypatch, tmp_path):
    p, (a, b, clean) = panel
    monkeypatch.setattr(npn, "ask_unsaved", lambda *a: "save")
    assert p.confirm_discard([a]) is False                 # never saved and no window to ask a name
    a.notebook.save(str(tmp_path / "a.py"))
    a.notebook.dirty = True
    assert p.confirm_discard([a]) is True and not a.notebook.dirty


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_quit_with_an_untitled_notebook_asks_for_a_name(app, monkeypatch, tmp_path):
    from ewald.gui.app import MainWindow

    monkeypatch.setattr(npn, "CONFIRM_UNSAVED", True)
    w = MainWindow()
    try:
        w.show()
        view = w.panel.new_notebook("draft")
        view.notebook.dirty = True
        monkeypatch.setattr(npn, "ask_unsaved", lambda *a: "save")
        names = iter([("", ""), (str(tmp_path / "draft.py"), "")])
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: next(names))
        assert w.close() is False and w.isVisible()        # Save as dialog cancelled: no quit
        assert w.close() is True                           # saved, then quits
        assert os.path.exists(tmp_path / "draft.py") and not view.notebook.dirty
    finally:
        monkeypatch.setattr(npn, "CONFIRM_UNSAVED", False)
        w.close()
