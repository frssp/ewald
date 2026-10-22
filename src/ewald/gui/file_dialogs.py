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
