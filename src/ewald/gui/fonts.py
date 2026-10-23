"""The UI fonts bundled in ``ewald/assets/fonts`` (IBM Plex Sans and Mono, JetBrains Mono; SIL
OFL 1.1, see the README there), registered with Qt so the theme's font families resolve to the
same files on every system. A family the user installed themselves is still found by name."""
from __future__ import annotations

from pathlib import Path

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
_loaded: list[str] = []


def load_fonts() -> list[str]:
    """Register every bundled font with Qt (once; needs a QGuiApplication) and return the
    families now available from them."""
    if _loaded:
        return list(_loaded)
    from PySide6 import QtGui

    families = []
    for path in sorted(FONT_DIR.glob("*.ttf")):
        font_id = QtGui.QFontDatabase.addApplicationFont(str(path))
        if font_id >= 0:
            families += [f for f in QtGui.QFontDatabase.applicationFontFamilies(font_id) if f not in families]
    _loaded.extend(families)
    return families
