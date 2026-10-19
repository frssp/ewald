"""Line icons for toolbars and menus, drawn in the theme colors.

One stroke style (24 px grid, 1.7 px round strokes) so every button reads as one set; the
paths are ewald's own. ``Preferences › General › Toolbar buttons`` chooses icons with text
(the default: text on mode switches and rarely used buttons, icons alone on the tool rail),
icons only or text only (:func:`button_style`).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .theme import TOKENS

PATHS = {
    "open": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "remote": '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
    "save": '<path d="M5 3h11l3 3v13a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M7 3v5h8V3M7 21v-7h10v7"/>',
    "analyze": '<circle cx="8" cy="9" r="3"/><circle cx="16" cy="15" r="3"/><path d="M10.5 10.5l3 3"/>',
    "render": '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    "orbit": '<path d="M20 12a8 8 0 1 1-3-6.2"/><path d="M20 4v5h-5"/>',
    "pick": '<path d="m5 3 14 7-6 2-2 6z"/>',
    "distance": '<circle cx="5" cy="12" r="2"/><circle cx="19" cy="12" r="2"/><path d="M7 12h10" stroke-dasharray="2 2"/>',
    "angle": '<path d="M4 20 18 6M4 20h16"/><path d="M10 20a6 6 0 0 0-1.8-4.2"/>',
    "bond": '<circle cx="6" cy="18" r="3"/><circle cx="18" cy="6" r="3"/><path d="m8 16 8-8"/>',
    "clear": '<path d="M6 6l12 12M18 6 6 18"/>',
    "fit": '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    "center": '<circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>',
    "select": '<rect x="4" y="4" width="16" height="16" rx="2" stroke-dasharray="3 2"/><path d="m9 12 2 2 4-4"/>',
    "delete": '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
    "play": '<path d="M7 4v16l13-8z"/>',
    "pause": '<path d="M8 5v14M16 5v14"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
    "isosurface": '<path d="M12 3c5 0 8 4 8 9s-3 9-8 9-8-4-8-9 3-9 8-9z"/><path d="M8 12c2-3 6-3 8 0"/>',
    "chart": '<path d="M4 20V4M4 20h16"/><path d="m7 15 4-5 3 3 5-7"/>',
    "claude": '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6"/>',
    "example": '<path d="M4 5a2 2 0 0 1 2-2h12v16H6a2 2 0 0 0-2 2z"/><path d="M4 21V5M8 7h6"/>',
    "image": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2"/>',
    "plugins": '<path d="M9 4h4v3a2 2 0 1 0 4 0V4h3v6h-3a2 2 0 1 0 0 4h3v6h-6v-3a2 2 0 1 0-4 0v3H4v-6h3a2 2 0 1 0 0-4H4V4z"/>',
    "notebook": '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6M9 12h6M9 16h4"/>',
    "eye": '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
}

APP_ICON_DIR = Path(__file__).resolve().parent.parent / "assets" / "icon"
APP_ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)

STYLES = ("Icons and text", "Icons", "Text")


def svg(name: str, color: str) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
            f'{PATHS[name]}</svg>')


def _pixmap(name: str, color: str, size: int) -> QtGui.QPixmap:
    from PySide6 import QtSvg

    pix = QtGui.QPixmap(size, size)
    pix.fill(QtCore.Qt.transparent)
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg(name, color).encode()))
    painter = QtGui.QPainter(pix)
    renderer.render(painter)
    painter.end()
    return pix


@lru_cache(maxsize=None)
def icon(name: str, color: str | None = None) -> QtGui.QIcon:
    """Icon ``name`` in the secondary text color, accent when checked, text color on hover;
    ``color`` fixes one color (e.g. on the accent-filled primary button)."""
    ic = QtGui.QIcon()
    for size in (16, 24, 32, 48):  # crisp at 1x and 2x screens
        if color:
            ic.addPixmap(_pixmap(name, color, size))
            continue
        ic.addPixmap(_pixmap(name, TOKENS["secondary"], size), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        ic.addPixmap(_pixmap(name, TOKENS["text"], size), QtGui.QIcon.Active, QtGui.QIcon.Off)
        for mode in (QtGui.QIcon.Normal, QtGui.QIcon.Active):
            ic.addPixmap(_pixmap(name, TOKENS["accent"], size), mode, QtGui.QIcon.On)
        ic.addPixmap(_pixmap(name, TOKENS["muted"], size), QtGui.QIcon.Disabled, QtGui.QIcon.Off)
    return ic


@lru_cache(maxsize=None)
def app_icon() -> QtGui.QIcon:
    """The EWALD app icon (window title bar, taskbar, dock), from the PNGs in ``ewald/assets/icon``;
    sizes up to 32 px are the simplified small variant (see ``make_icons.py`` there)."""
    ic = QtGui.QIcon()
    for size in APP_ICON_SIZES:
        path = APP_ICON_DIR / f"ewald-{size}.png"
        if path.exists():
            ic.addFile(str(path), QtCore.QSize(size, size))
    return ic


def button_style(button: QtWidgets.QAbstractButton, style: str, text_in_hybrid: bool = True):
    """Show ``button`` as icon, text or both for Preferences › General › Toolbar buttons.
    ``text_in_hybrid`` False keeps the default style to the icon (tool rail, frequent buttons)."""
    show_icon = style != "Text"
    show_text = style == "Text" or (style == "Icons and text" and text_in_hybrid)
    name = button.property("ewald_icon")
    button.setIcon(icon(name, button.property("ewald_icon_color")) if show_icon and name else QtGui.QIcon())
    if isinstance(button, QtWidgets.QToolButton):
        button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon if show_icon and show_text
                                  else QtCore.Qt.ToolButtonIconOnly if show_icon else QtCore.Qt.ToolButtonTextOnly)
    elif isinstance(button, QtWidgets.QPushButton):
        label = button.property("ewald_label") or button.text()
        button.setProperty("ewald_label", label)
        button.setText(label if show_text or not show_icon else "")
