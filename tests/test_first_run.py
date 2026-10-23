"""The first-run notice's stamp (``ewald.gui.first_run``), without Qt: when a start counts as the
first one after an install or update. The notice window itself is tested in tests/test_gui.py."""
import json
import ntpath
import os

import pytest

from ewald.gui import first_run


def test_first_start_then_not_until_something_changes():
    assert first_run.is_first_run()          # no startup.json yet
    first_run.mark_done()
    assert not first_run.is_first_run()
    saved = json.loads(first_run._path().read_text(encoding="utf-8"))
    assert saved == first_run.stamp()
    changed = dict(saved, ewald="99.0")       # an update
    assert first_run.is_first_run(changed)
    changed = dict(saved, packages=dict(saved["packages"], numba=saved["packages"]["numba"] + 1))
    assert first_run.is_first_run(changed)    # a reinstalled library


def test_editing_an_ewald_source_counts_as_an_update(tmp_path, monkeypatch):
    """An editable install (git pull) keeps the version, so the sources' mtimes are part of the stamp."""
    import ewald

    pkg = tmp_path / "ewald"
    for d in first_run.SOURCE_DIRS:
        (pkg / d).mkdir(parents=True, exist_ok=True)
    src = pkg / "gui" / "app.py"
    src.write_text("", encoding="utf-8")
    monkeypatch.setattr(ewald, "__file__", str(pkg / "__init__.py"))
    first_run.mark_done()
    assert not first_run.is_first_run()
    st = os.stat(src)
    os.utime(src, (st.st_atime, st.st_mtime + 10))
    assert first_run.is_first_run()


def test_broken_or_unwritable_stamp_only_shows_the_notice_again(monkeypatch):
    path = first_run._path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert first_run.is_first_run()

    def refuse(*a, **k):
        raise PermissionError("read-only")

    from ewald import fileops

    monkeypatch.setattr(fileops, "write_text", refuse)
    first_run.mark_done()                     # no exception
    assert first_run.is_first_run()


def test_stamp_reads_crlf_and_windows_executables():
    """A stamp written on Windows (CRLF, backslash paths, any case) still compares equal there."""
    assert first_run._executable(r"C:\Users\me\.venv\Scripts\Python.EXE", ntpath) == \
        r"c:\users\me\.venv\scripts\python.exe"
    current = first_run.stamp()
    path = first_run._path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(current, indent=1).replace("\n", "\r\n").encode("utf-8"))
    assert not first_run.is_first_run(current)


def test_help_never_shows_the_notice(monkeypatch):
    monkeypatch.setattr(first_run, "show_notice", lambda app: pytest.fail("notice shown"))
    assert first_run.start(["--help"]) is None
    first_run.finish()                        # nothing to close: no stamp written
    assert not first_run._path().exists()
