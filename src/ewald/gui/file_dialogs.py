"""Save-dialog helpers that behave the same on every platform."""
from __future__ import annotations

import fnmatch
import os
import re


def suffixed(path: str, chosen_filter: str = "") -> str:
    """``path`` with the first extension of the dialog filter the user picked when the typed
    name has none and matches none of its patterns (``POSCAR`` stays ``POSCAR``). The native
    Windows dialog adds the extension itself, Qt's own dialog (Linux) may not, so ``plot``
    would otherwise be saved without a format on one platform only."""
    if not path:
        return path
    name = os.path.basename(path)
    patterns = re.findall(r"[^\s()]*\*[^\s()]*", (chosen_filter or "").rsplit("(", 1)[-1])
    if os.path.splitext(name)[1] or any(fnmatch.fnmatchcase(name.lower(), p.lower()) for p in patterns):
        return path
    ext = next((p[1:] for p in patterns if p.startswith("*.")), "")
    return path + ext


def open_in_editor(path) -> bool:
    """Open a text file in the user's editor (:func:`ewald.fileops.editor_commands`); False when
    nothing could open it."""
    import subprocess

    from PySide6 import QtCore, QtGui

    from ..fileops import editor_commands

    for way in editor_commands(path):
        if way == "url":
            if QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path))):
                return True
            continue
        try:
            subprocess.Popen(way, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            continue
    return False
