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

from .theme import STYLE, TOKENS

PATHS = {
    "open": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "remote": '<rect x="4" y="4" width="16" height="7" rx="1.8"/><rect x="4" y="13" width="16" height="7" rx="1.8"/>'
              '<path d="M8 7.5h.01M8 16.5h.01M12 7.5h4M12 16.5h4"/>',
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
    "play": '<path d="M7.5 5.5v13l10.5-6.5z" fill="currentColor"/>',
    "pause": '<rect x="6.5" y="5.5" width="3.5" height="13" rx="1" fill="currentColor"/>'
             '<rect x="14" y="5.5" width="3.5" height="13" rx="1" fill="currentColor"/>',
    "stop": '<rect x="6.5" y="6.5" width="11" height="11" rx="2" fill="currentColor"/>',
    "first": '<path d="M5 5.5v13"/><path d="M12.5 6.5v11L7.5 12z" fill="currentColor"/>'
             '<path d="M19.5 6.5v11L14.5 12z" fill="currentColor"/>',
    "last": '<path d="M19 5.5v13"/><path d="M11.5 6.5v11l5-5.5z" fill="currentColor"/>'
            '<path d="M4.5 6.5v11l5-5.5z" fill="currentColor"/>',
    "prev": '<path d="M7 5.5v13"/><path d="M17.5 6.5v11L10.5 12z" fill="currentColor"/>',
    "next": '<path d="M17 5.5v13"/><path d="M6.5 6.5v11l7-5.5z" fill="currentColor"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
    "isosurface": '<path d="M12 3c5 0 8 4 8 9s-3 9-8 9-8-4-8-9 3-9 8-9z"/><path d="M8 12c2-3 6-3 8 0"/>',
    "chart": '<path d="M4 20V4M4 20h16"/><path d="m7 15 4-5 3 3 5-7"/>',
    "ai": '<path d="M6.5 4h11A2.5 2.5 0 0 1 20 6.5v8a2.5 2.5 0 0 1-2.5 2.5H11l-4.5 3.5V17A2.5 2.5 0 0 1 4 14.5v-8A2.5 2.5 0 0 1 6.5 4z"/>'
          '<path d="M12 6.9Q12.6 9.9 15.6 10.5Q12.6 11.1 12 14.1Q11.4 11.1 8.4 10.5Q11.4 9.9 12 6.9z" fill="currentColor" stroke-width="1"/>',
    "workflow": '<rect x="3.5" y="3.5" width="7" height="5" rx="1.2"/><rect x="13.5" y="9.5" width="7" height="5" rx="1.2"/>'
                '<rect x="3.5" y="15.5" width="7" height="5" rx="1.2"/><path d="M10.5 6H14a3 3 0 0 1 3 3v.5M17 14.5v.5a3 3 0 0 1-3 3h-3.5"/>',
    "example": '<path d="M4 5a2 2 0 0 1 2-2h12v16H6a2 2 0 0 0-2 2z"/><path d="M4 21V5M8 7h6"/>',
    "image": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M10.3 3.5h3.4l.5 2.4 1.6.9 2.3-.8 1.7 2.9-1.8 1.6v1.9l1.8 1.6-1.7 2.9'
                '-2.3-.8-1.6.9-.5 2.4h-3.4l-.5-2.4-1.6-.9-2.3.8-1.7-2.9 1.8-1.6v-1.9L4.2 8.9l1.7-2.9 2.3.8 1.6-.9z"/>',
    "plugins": '<path d="M9 4h4v3a2 2 0 1 0 4 0V4h3v6h-3a2 2 0 1 0 0 4h3v6h-6v-3a2 2 0 1 0-4 0v3H4v-6h3a2 2 0 1 0 0-4H4V4z"/>',
    "notebook": '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6M9 12h6M9 16h4"/>',
    "eye": '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "split": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 4v16"/>',
    "maximize": '<path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7"/>',
    "fold": '<path d="M6 9l6 6 6-6"/>',
    "chevron_down": '<path d="M6 9l6 6 6-6"/>',
    "chevron_up": '<path d="M6 15l6-6 6 6"/>',
    "chevron_right": '<path d="M9 6l6 6-6 6"/>',
    "chevron_left": '<path d="M15 6l-6 6 6 6"/>',
    "float": '<rect x="3" y="8" width="13" height="13" rx="2"/><path d="M12 3h9v9M21 3l-8 8"/>',
    "menu": '<path d="M4 7h16M4 12h16M4 17h16"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "grip": '<g fill="currentColor" stroke="none"><circle cx="9" cy="6" r="1.5"/><circle cx="15" cy="6" r="1.5"/>'
            '<circle cx="9" cy="12" r="1.5"/><circle cx="15" cy="12" r="1.5"/><circle cx="9" cy="18" r="1.5"/>'
            '<circle cx="15" cy="18" r="1.5"/></g>',
    "send": '<path d="M12 19V5M6 11l6-6 6 6"/>',
    "arrow_left": '<path d="M19 12H5M11 6l-6 6 6 6"/>',
    "arrow_right": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    "arrow_up": '<path d="M12 19V5M6 11l6-6 6 6"/>',
    "arrow_down": '<path d="M12 5v14M6 13l6 6 6-6"/>',
    "arrow_up_right": '<path d="M7 17L17 7M8 7h9v9"/>',
    "roll": '<path d="M20 12a8 8 0 1 1-3-6.2"/><path d="M20 4v5h-5"/>',
    "reset": '<path d="M4 12a8 8 0 1 0 3-6.2"/><path d="M4 4v5h5"/>',
    "refresh": '<path d="M20 12a8 8 0 1 1-3-6.2"/><path d="M20 4v5h-5"/>',
    "home": '<path d="M4 11.5 12 4.5l8 7"/><path d="M6.5 10v9.5h11V10"/>',
    "folder_up": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M12 16v-6M9 12.5l3-3 3 3"/>',
    "star": '<path d="m12 3.5 2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z"/>',
    "more": '<circle cx="5" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="12" r="1.6"/>',
}
PATHS["claude"] = PATHS["ai"]   # the name plugins used before "ai" (a multi-provider AI is not only Claude)

APP_ICON_DIR = Path(__file__).resolve().parent.parent / "assets" / "icon"
APP_ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)

STYLES = ("Icons and text", "Icons", "Text")


def svg(name: str, color: str, stroke: float | None = None) -> str:
    stroke = stroke or STYLE["icon_stroke"]   # the UI theme's line width
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke:g}" stroke-linecap="round" stroke-linejoin="round">'
            f'{PATHS[name].replace("currentColor", color)}</svg>')


def _pixmap(name: str, color: str, size: int) -> QtGui.QPixmap:
    from PySide6 import QtSvg

    pix = QtGui.QPixmap(size, size)
    pix.fill(QtCore.Qt.transparent)
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg(name, color).encode()))
    painter = QtGui.QPainter(pix)
    renderer.render(painter)
    painter.end()
    return pix


def icon(name: str, color: str | None = None) -> QtGui.QIcon:
    """Icon ``name`` in the secondary text color, accent when checked, text color on hover;
    ``color`` fixes one color (e.g. on the accent-filled primary button). Cached per UI theme."""
    return _icon(name, color, TOKENS["secondary"], TOKENS["text"], TOKENS["accent"], TOKENS["muted"],
                 STYLE["icon_stroke"])


@lru_cache(maxsize=None)
def _icon(name: str, color, *_palette) -> QtGui.QIcon:
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


def refresh(root: QtWidgets.QWidget):
    """Redraw the icons of every button under ``root`` that names one (``ewald_icon``) in the
    current UI theme; the toolbar's :func:`button_style` buttons are left to it."""
    for b in root.findChildren(QtWidgets.QAbstractButton):
        name = b.property("ewald_icon")
        if name and name in PATHS and not b.icon().isNull():
            color = b.property("ewald_icon_color")  # a token name follows the theme, a hex stays
            b.setIcon(icon(name, TOKENS.get(color, color) if color else None))


def badged(name: str, count: int) -> QtGui.QIcon:
    """:func:`icon` with a round count badge in its lower right corner (the activity bar's
    "updates available"); ``count`` 0 gives the plain icon."""
    if count <= 0:
        return icon(name)
    base, out = icon(name), QtGui.QIcon()
    text = str(count) if count < 10 else "9+"
    for size in (16, 24, 32, 48):
        for mode in (QtGui.QIcon.Normal, QtGui.QIcon.Active, QtGui.QIcon.Disabled):
            for state in (QtGui.QIcon.Off, QtGui.QIcon.On):
                pix = base.pixmap(size, size, mode, state)
                painter = QtGui.QPainter(pix)
                painter.setRenderHint(QtGui.QPainter.Antialiasing)
                d = size * 0.62
                rect = QtCore.QRectF(size - d, size - d, d, d)
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(TOKENS["accent"]))
                painter.drawEllipse(rect)
                font = painter.font()
                font.setPixelSize(max(6, int(d * (0.78 if len(text) == 1 else 0.58))))
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QtGui.QColor("white"))
                painter.drawText(rect, QtCore.Qt.AlignCenter, text)
                painter.end()
                out.addPixmap(pix, mode, state)
    return out


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
    color = button.property("ewald_icon_color")  # a token name follows the theme, a hex stays
    button.setIcon(icon(name, TOKENS.get(color, color) if color else None) if show_icon and name else QtGui.QIcon())
    if isinstance(button, QtWidgets.QToolButton):
        button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon if show_icon and show_text
                                  else QtCore.Qt.ToolButtonIconOnly if show_icon else QtCore.Qt.ToolButtonTextOnly)
    elif isinstance(button, QtWidgets.QPushButton):
        label = button.property("ewald_label") or button.text()
        button.setProperty("ewald_label", label)
        button.setText(label if show_text or not show_icon else "")
