"""Replacing files safely on every platform.

Settings, notebooks, sessions and caches are written to a temporary file next to the target
and then moved over it with :func:`replace`, so a crash never leaves half a file. On Windows
that move fails with ``PermissionError`` while another program has the target open (an editor,
a virus scanner, a backup or sync client such as OneDrive), which on Linux and macOS never
happens. :func:`replace` retries for a moment, since scanners and sync clients let go quickly,
and then raises :class:`FileInUseError` naming the programs that hold the file (from the
Windows Restart Manager), so the message says what to close.

>>> from ewald.fileops import write_text, replace
>>> write_text("settings.json", text)      # temp file + replace
>>> replace("out.ewd.tmp", "out.ewd")
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

#: Delays between attempts of :func:`replace` on Windows (seconds; about 1.5 s in all).
RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.75)


class FileInUseError(PermissionError):
    """A file could not be replaced because another program has it open (Windows)."""

    def __init__(self, path, programs: list[str], cause: OSError | None = None):
        self.path = os.fspath(path)
        self.programs = list(programs)
        who = ", ".join(self.programs) if self.programs else "another program"
        super().__init__(f"{self.path} is open in {who}; close it there and save again"
                         + (f" ({cause.strerror or cause})" if cause is not None else ""))


def _windows() -> bool:
    return sys.platform == "win32"


def replace(src, dst, *, delays=None) -> None:
    """``os.replace(src, dst)``, retried on Windows while another program holds either file;
    raises :class:`FileInUseError` (a ``PermissionError``) naming that program in the end."""
    delays = RETRY_DELAYS if delays is None else delays
    for attempt in range(len(delays) + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError as e:
            if not _windows():
                raise
            if attempt == len(delays):
                programs = holders(dst) or holders(src)
                raise FileInUseError(dst, programs, e) from e
            time.sleep(delays[attempt])


def write_text(path, text: str, *, delays=None) -> Path:
    """Write ``text`` (UTF-8, LF line ends) to ``path`` through a temporary file and :func:`replace`."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        replace(tmp, path, delays=delays)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


def holders(path) -> list[str]:
    """Names of the programs that have ``path`` open (Windows Restart Manager); [] elsewhere
    or when Windows cannot tell."""
    if not _windows():
        return []
    try:
        return _restart_manager_list(os.path.abspath(os.fspath(path)))
    except Exception:   # an old Windows, a missing rstrtmgr.dll, a path it cannot register
        return []


def _restart_manager_list(path: str) -> list[str]:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    rm = ctypes.WinDLL("rstrtmgr")

    class RM_UNIQUE_PROCESS(ctypes.Structure):
        _fields_ = [("dwProcessId", wintypes.DWORD), ("ProcessStartTime", wintypes.FILETIME)]

    class RM_PROCESS_INFO(ctypes.Structure):
        _fields_ = [("Process", RM_UNIQUE_PROCESS),
                    ("strAppName", wintypes.WCHAR * 256),          # CCH_RM_MAX_APP_NAME + 1
                    ("strServiceShortName", wintypes.WCHAR * 64),  # CCH_RM_MAX_SVC_NAME + 1
                    ("ApplicationType", ctypes.c_int),
                    ("AppStatus", wintypes.ULONG),
                    ("TSSessionId", wintypes.DWORD),
                    ("bRestartable", wintypes.BOOL)]

    session = wintypes.DWORD()
    key = ctypes.create_unicode_buffer(33)                         # CCH_RM_SESSION_KEY + 1
    if rm.RmStartSession(ctypes.byref(session), 0, key):
        return []
    try:
        files = (wintypes.LPCWSTR * 1)(path)
        if rm.RmRegisterResources(session, 1, files, 0, None, 0, None):
            return []
        needed, count, reasons = wintypes.UINT(0), wintypes.UINT(0), wintypes.DWORD()
        err = rm.RmGetList(session, ctypes.byref(needed), ctypes.byref(count), None, ctypes.byref(reasons))
        if err not in (0, 234) or not needed.value:                # 234: ERROR_MORE_DATA
            return []
        info = (RM_PROCESS_INFO * needed.value)()
        count = wintypes.UINT(needed.value)
        if rm.RmGetList(session, ctypes.byref(needed), ctypes.byref(count), info, ctypes.byref(reasons)):
            return []
        names = []
        for i in range(count.value):
            name = info[i].strAppName or f"process {info[i].Process.dwProcessId}"
            if name not in names:
                names.append(name)
        return names
    finally:
        rm.RmEndSession(session)
