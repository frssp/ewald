"""One header for every panel: dock title bars, the workbench's tab-group heads and the notebook tabs.

All of them share :func:`height`, the left inset of their first label (:data:`INSET`) and
:func:`button` (an icon-only tool button of :data:`BUTTON` px with a :data:`ICON` px line icon from
``icons.py``). Glyph labels (▾, ⧉, ⏮) are not used: their size and baseline depend on whichever
fallback font the platform picks (Segoe UI Symbol or a color emoji font on Windows), which is what
made the panel corners line up differently there. Restyle through the object names in ``theme.py``.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

BUTTON = 24  # header buttons, square
ICON = 16    # line icons inside them
INSET = 12   # left padding of the first title or tab label
MIN_HEIGHT = 34


def height(widget: QtWidgets.QWidget | None = None) -> int:
    """Height of a panel header: room for a tab label in the UI font plus padding, at least 34 px."""
    from .theme import current_font_size

    app = QtWidgets.QApplication.instance()
    px = current_font_size(app) if app is not None else 12
    return max(MIN_HEIGHT, round(px * 1.4) + 18)


def button(icon_name: str, tip: str, checkable: bool = False, name: str = "headerButton") -> QtWidgets.QToolButton:
    """An icon-only header button (``icon_name`` from ``icons.PATHS``)."""
    from .icons import icon

    b = QtWidgets.QToolButton(toolTip=tip, autoRaise=True, objectName=name, checkable=checkable)
    b.setProperty("ewald_icon", icon_name)
    b.setIcon(icon(icon_name))
    b.setIconSize(QtCore.QSize(ICON, ICON))
    b.setFixedSize(BUTTON, BUTTON)
    b.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
    return b


def set_icon(b: QtWidgets.QAbstractButton, icon_name: str):
    """Swap the icon of a :func:`button` (fold state, play/pause)."""
    from .icons import icon

    b.setProperty("ewald_icon", icon_name)
    b.setIcon(icon(icon_name))
