"""The UI fonts bundled in ``ewald/assets/fonts`` (IBM Plex Sans and Mono, JetBrains Mono; SIL
OFL 1.1, see the README there), registered with Qt so the theme's font families resolve to the
same files on every system. A family the user installed themselves is still found by name.

Every family a style sheet or a font names must exist: when Qt meets a missing one (as the
primary family, or as a fallback for a glyph the first font lacks) it builds its whole family
alias table, which costs about 200 ms on macOS and logs "Populating font family aliases took
... Replace uses of missing font family". :func:`css` and :func:`installed` drop the families
this system does not have, so the fallback lists in ``theme.QSS`` can name fonts of all three
platforms (Segoe UI, Helvetica Neue, Consolas, Menlo, ...)."""
from __future__ import annotations

import re
from pathlib import Path

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
_loaded: list[str] = []
_families: set[str] | None = None   # _known()'s answer: QFontDatabase.families() took 0.1 s on Windows


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
    global _families
    _families = None
    return families


# CSS generic families: Qt does not resolve them in style sheets, so they become the system's
# own UI or fixed-width family (QFontDatabase.SystemFont names)
GENERIC = {"sans-serif": "GeneralFont", "serif": "GeneralFont", "system-ui": "GeneralFont",
           "monospace": "FixedFont"}
_FAMILY = re.compile(r"(font-family:\s*)([^;}]+)")


def _known() -> set[str] | None:
    """Lower-cased families Qt has (bundled, installed), or None without a QGuiApplication."""
    from PySide6 import QtGui

    global _families
    if QtGui.QGuiApplication.instance() is None:
        return None
    if _families is None:
        # "IBM Plex Sans [IBM]" when two foundries ship a family; the name alone matches either
        _families = {re.sub(r"\s*\[.*\]$", "", f).lower() for f in QtGui.QFontDatabase.families()}
    return _families


def _system(kind: str) -> str:
    from PySide6 import QtGui

    return QtGui.QFontDatabase.systemFont(getattr(QtGui.QFontDatabase.SystemFont, kind)).family()


def installed(families, known: set[str] | None = None) -> list[str]:
    """``families`` without those this system lacks, generic names (``monospace``) replaced by
    the system family, duplicates dropped; never empty (the system UI font at worst). Unchanged
    without a QGuiApplication, where nothing can be looked up."""
    known = _known() if known is None else known
    families = [f.strip().strip("\"'") for f in families if f.strip().strip("\"'")]
    if known is None:
        return families
    out = []
    for f in families:
        name = _system(GENERIC[f.lower()]) if f.lower() in GENERIC else f if f.lower() in known else None
        if name and name.lower() not in {o.lower() for o in out}:
            out.append(name)
    return out or [_system("GeneralFont")]


def css(text: str, known: set[str] | None = None) -> str:
    """Style sheet ``text`` with each ``font-family`` list cut to :func:`installed` families."""
    known = _known() if known is None else known
    if known is None:
        return text

    def fix(m):
        names = installed(m.group(2).split(","), known)
        return m.group(1) + ", ".join(f'"{n}"' for n in names)

    return _FAMILY.sub(fix, text)
