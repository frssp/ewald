"""Remotes side bar (gui/remotes_panel.py): saved SSH computers and their file browser."""
import os

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from ewald import keychain, settings  # noqa: E402
from ewald.io import profiles, remote  # noqa: E402
from ewald.io.profiles import Profile  # noqa: E402


@pytest.fixture(scope="module")
def app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    for w in app.topLevelWidgets():
        w.close()
        w.deleteLater()
    app.processEvents()


@pytest.fixture
def kr(monkeypatch):
    store = {}

    class K:
        get_password = staticmethod(lambda s, u: store.get((s, u)))
        set_password = staticmethod(lambda s, u, v: store.__setitem__((s, u), v))
        delete_password = staticmethod(lambda s, u: store.pop((s, u)))
        get_keyring = staticmethod(lambda: K)

    monkeypatch.setattr(keychain, "_keyring", lambda: K)
    return store


def _names(panel):
    return [panel.files.topLevelItem(i).text(0) for i in range(panel.files.topLevelItemCount())]


def test_panel_lists_profiles_and_browses(app, server, kr, tmp_path):
    from ewald.gui.remotes_panel import RemotesPanel

    profiles.save(Profile("work", host="hpc", folder="/data"))
    panel = RemotesPanel()
    rows = [panel.list.item(i).data(QtCore.Qt.UserRole) for i in range(panel.list.count())]
    assert rows == [("profile", "work")]  # 'hpc' from ~/.ssh/config is named by the profile
    opened, local = [], []
    panel.openRequested.connect(opened.append)
    panel.downloadedOpen.connect(local.append)

    assert panel.connect_to("work") and panel.cwd == "/data"
    assert {"XDATCAR", "POSCAR"} <= set(_names(panel))
    panel.filter.setText("xdat")
    assert _names(panel) == ["XDATCAR"]
    panel.filter.setText("")
    assert not panel.browse("/nowhere") and panel.status.text().startswith("✗")
    item = panel.files.findItems("POSCAR", QtCore.Qt.MatchExactly)[0]
    panel._activated(item)
    assert opened == ["work:/data/POSCAR"]
    panel.files.setCurrentItem(item)
    assert panel.selected_url() == "work:/data/POSCAR"
    path = panel.download("work:/data/POSCAR", str(tmp_path), then_open=True)
    assert os.path.isfile(path) and local == [path]

    panel.bookmark()
    assert profiles.get("work").bookmarks == ["/data"]
    places = [panel.places.itemData(i) for i in range(panel.places.count())]
    assert ("dir", "/data") in places and ("file", "work:/data/POSCAR") in places
    panel.up()
    assert panel.cwd == "/"
    panel._place(places.index(("dir", "/data")))
    assert panel.cwd == "/data"
    assert "●" in panel.list.item(0).text()  # connected

    panel.delete("work", confirm=False)
    rows = [panel.list.item(i).data(QtCore.Qt.UserRole) for i in range(panel.list.count())]
    assert rows == [("config", "hpc")]
    assert panel.connect_to("hpc") and panel.cwd == "/home/alice"


def test_profile_dialog_saves_secret_in_keychain(app, server, kr):
    from ewald.gui.remotes_panel import ProfileDialog

    dlg = ProfileDialog()
    dlg.name.setText("lab")
    dlg.host.setEditText("hpc")
    dlg.auth["password"].setChecked(True)
    assert not dlg.form.isRowVisible(dlg.key_widget) and dlg.form.isRowVisible(dlg.pw_row)
    dlg.password.setText("pw-1")
    dlg._save()
    assert dlg.result() == QtWidgets.QDialog.Accepted
    assert kr[("ewald", "ssh-password:lab")] == "pw-1"
    assert "pw-1" not in (settings.config_dir() / "remote_profiles.json").read_text()

    dlg = ProfileDialog(profiles.get("lab"))
    assert "saved" in dlg.password.placeholderText() and dlg.forget["password"].isVisibleTo(dlg)
    dlg.name.setText("lab2")
    dlg.auth["auto"].setChecked(True)  # key auth from ~/.ssh/config works for hpc
    assert dlg.test() and dlg.status.text().startswith("✓")
    dlg._save()
    assert profiles.names() == ["lab2"] and kr[("ewald", "ssh-password:lab2")] == "pw-1"

    dlg = ProfileDialog(profiles.get("lab2"))
    dlg.forget["password"].setChecked(True)
    dlg._save()
    assert ("ewald", "ssh-password:lab2") not in kr

    bad = ProfileDialog()
    bad.name.setText("x")
    bad._save()
    assert bad.result() != QtWidgets.QDialog.Accepted and "name" in bad.status.text()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_main_window_has_remotes_side_bar(app, server, kr, monkeypatch, vasp_dir):
    from ewald.gui import notebook_panel
    from ewald.gui.app import MainWindow

    monkeypatch.setattr(notebook_panel, "CONFIRM_UNSAVED", False)
    profiles.save(Profile("work", host="hpc", folder="/data"))
    w = MainWindow()
    try:
        assert "Remotes" in w.activity_buttons and w.remotes_dock.isHidden()
        w.show_remotes()
        assert not w.remotes_dock.isHidden()
        w.remotes_panel.connect_to("work")
        w.remotes_panel.open("work:/data/POSCAR")
        assert w.pipeline is not None and w.pipeline.source.path == "work:/data/POSCAR"
        import shutil

        shutil.copytree(vasp_dir / "neb", os.path.join(server.root, "data", "neb"))
        w.remotes_panel.nebRequested.emit("work:/data/neb")  # folder menu › Open as NEB run
        assert w.pipeline.source.path == "work:/data/neb" and w.pipeline.num_frames == 5
    finally:
        w.close()
        remote.disconnect_all()
