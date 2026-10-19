"""UI themes from the design tokens (project design files, redesign canvas).

Paper (light, the default) and Graphite (dark); ``@ui_theme`` plugins add more (:func:`palette`). ``accent`` (teal) marks what you can act on,
``data`` (orange) marks results computed from the data. ``TOKENS`` is the active palette; it is
updated in place by :func:`use`, so ``from .theme import TOKENS`` sees the change.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

PALETTES = {
    "paper": {
        "bg": "#FBFBF9", "panel": "#F3F4F1", "raised": "#FFFFFF", "border": "#D5D8D1",
        "text": "#15181C", "secondary": "#3D434A", "muted": "#5C636B", "accent": "#0B7A6F",
        "on_accent": "#FFFFFF", "accent_soft": "#E0EFEC", "data": "#C8551F", "data_soft": "#F7E4D9",
        "selection": "#C99A06", "pick": "#C8551F", "error": "#C62828", "ok": "#2E7D32",
        "viewport": "#ECEDE9", "hover": "#E9EBE6", "pressed": "#DFE2DC", "accent_hover": "#0A6B61",
        "accent_pressed": "#085C53",
    },
    "graphite": {
        "bg": "#14161A", "panel": "#1B1E23", "raised": "#23272E", "border": "#30353D",
        "text": "#E7E9EC", "secondary": "#AEB4BE", "muted": "#8C939E", "accent": "#3FB8A6",
        "on_accent": "#0B1220", "accent_soft": "#1E3A36", "data": "#F08A4B", "data_soft": "#3A2A20",
        "selection": "#E8C547", "pick": "#FFD84D", "error": "#FF6B6B", "ok": "#5BD08A",
        "viewport": "#14161A", "hover": "#2A2F37", "pressed": "#333943", "accent_hover": "#55C6B5",
        "accent_pressed": "#33A493",
    },
}
DEFAULT = "paper"
TOKENS = dict(PALETTES[DEFAULT])


# Interaction shades: (token, from, toward, amount). Built-in palettes pick them by hand;
# ``@ui_theme`` plugins that leave them out get them mixed from their own colors (:func:`derive`).
DERIVED = (("hover", "raised", "text", 0.07), ("pressed", "raised", "text", 0.12),
           ("accent_hover", "accent", "text", 0.15), ("accent_pressed", "accent", "text", 0.28))


def _mix(a: str, b: str, t: float) -> str:
    ca, cb = (tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in (a, b))
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ca, cb))


def derive(tokens: dict, own=()) -> dict:
    """``tokens`` with the interaction shades not in ``own`` mixed from its base colors."""
    out = dict(tokens)
    for key, a, b, t in DERIVED:
        if key not in own:
            out[key] = _mix(out[a], out[b], t)
    return out


# Small line images the style sheet needs as files (QSS has no inline images), in palette colors.
_IMAGES = {
    "chevron": ("secondary", '<path d="M6 9l6 6 6-6"/>'),
    "chevron_up": ("secondary", '<path d="M6 15l6-6 6 6"/>'),
    "check": ("on_accent", '<path d="M5 12.5l4.5 4.5L19 7.5"/>'),
    "close": ("muted", '<path d="M7 7l10 10M17 7 7 17"/>'),
    "close_hover": ("text", '<path d="M7 7l10 10M17 7 7 17"/>'),
}


def image(name: str, tokens: dict | None = None) -> str:
    """Path of the style-sheet image ``name`` drawn in the colors of ``tokens`` (the active palette)."""
    import os
    import tempfile

    key, path_d = _IMAGES[name]
    color = (tokens or TOKENS)[key]
    path = os.path.join(tempfile.gettempdir(), f"ewald-ui-{name}-{color.lstrip('#')}.svg")
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
                     f'stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
                     f'{path_d}</svg>')
    return path.replace("\\", "/")


def palette(name: str) -> dict:
    """The design tokens of UI theme ``name``: a built-in palette or a ``@ui_theme`` plugin
    (:func:`ewald.plugins.ui_theme`); unknown or broken themes fall back to the default."""
    if name in PALETTES:
        return dict(PALETTES[name])
    from .. import plugins

    found = plugins.ui_themes().get(name)
    if found is not None:
        try:
            return found.palette()
        except Exception as exc:  # a broken theme must not keep the window from opening
            log.warning("UI theme %r: %s; using %s", name, exc, DEFAULT)
    elif name:
        log.warning("unknown UI theme %r; using %s", name, DEFAULT)
    return dict(PALETTES[DEFAULT])


def use(name: str) -> dict:
    """Make ``name`` the active palette (unknown names fall back to the default)."""
    TOKENS.clear()
    TOKENS.update(palette(name))
    return TOKENS

QSS = """
* {{ font-family: "IBM Plex Sans", "Inter", "Segoe UI", "Helvetica Neue", sans-serif; font-size: 12px; }}
QMainWindow, QWidget {{ background: {panel}; color: {text}; }}
QMainWindow::separator {{ background: {border}; width: 1px; height: 1px; }}
QMenuBar {{ background: {bg}; color: {secondary}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {raised}; color: {text}; }}
QMenu {{ background: {raised}; border: 1px solid {border}; padding: 4px; }}
QMenu::separator {{ height: 1px; background: {border}; margin: 4px 8px; }}
QToolBar {{ background: {bg}; border: none; border-bottom: 1px solid {border}; spacing: 4px; padding: 4px 6px; }}
QToolBar::separator {{ background: {border}; width: 1px; margin: 6px 6px; }}

/* buttons: flat, no bevels; ghost tool buttons, quiet secondary push buttons, one filled primary */
QToolButton {{ background: transparent; color: {secondary}; border: none; border-radius: 6px; padding: 5px 8px; }}
QToolButton:hover {{ background: {hover}; color: {text}; }}
QToolButton:pressed {{ background: {pressed}; }}
QToolButton:checked {{ background: {accent_soft}; color: {accent}; }}
QToolButton:disabled {{ color: {muted}; background: transparent; }}
QToolButton#groupButton {{ padding: 2px; border-radius: 4px; }}
QToolButton[popupMode="2"] {{ padding-right: 18px; }}
QToolButton::menu-indicator {{ image: url("{img_chevron}"); width: 9px; height: 9px;
    subcontrol-origin: padding; subcontrol-position: center right; right: 5px; }}
QToolButton::menu-indicator:pressed, QToolButton::menu-indicator:open {{ top: 0; left: 0; }}
QToolButton#cellMenu {{ font-size: 16px; padding: 0 6px; color: {muted}; }}
QToolButton#cellMenu::menu-indicator {{ image: none; width: 0; }}
QToolButton#cellMenu:hover {{ color: {text}; }}
QPushButton {{ background: {raised}; color: {text}; border: 1px solid {border}; border-radius: 6px; padding: 5px 14px;
    min-height: 16px; }}
QPushButton:hover {{ background: {hover}; }}
QPushButton:pressed {{ background: {pressed}; }}
QPushButton:checked {{ background: {accent_soft}; color: {accent}; border-color: {accent}; }}
QPushButton:disabled {{ color: {muted}; background: {panel}; }}
QPushButton:focus {{ outline: none; }}
QPushButton::menu-indicator {{ image: url("{img_chevron}"); width: 9px; height: 9px;
    subcontrol-origin: padding; subcontrol-position: center right; right: 8px; }}
QPushButton#primary {{ background: {accent}; color: {on_accent}; border: 1px solid {accent}; font-weight: 600; }}
QPushButton#primary:hover {{ background: {accent_hover}; border-color: {accent_hover}; }}
QPushButton#primary:pressed {{ background: {accent_pressed}; border-color: {accent_pressed}; }}
QPushButton#primary:disabled {{ background: {border}; border-color: {border}; color: {muted}; }}
QPushButton#searchBox {{ text-align: left; color: {muted}; background: {panel}; border: 1px solid {border};
    border-radius: 6px; padding: 5px 12px; min-width: 320px; font-weight: normal; }}
QPushButton#searchBox:hover {{ border-color: {muted}; background: {panel}; color: {secondary}; }}
QToolTip {{ background: {raised}; color: {text}; border: 1px solid {border}; border-radius: 4px; padding: 4px 6px; }}

/* fields */
QLineEdit, QPlainTextEdit, QTextEdit, QAbstractSpinBox, QComboBox {{
    background: {bg}; color: {text}; border: 1px solid {border}; border-radius: 6px; padding: 3px 6px;
    selection-background-color: {accent}; selection-color: {on_accent}; }}
QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover {{ border-color: {muted}; }}
QLineEdit:focus, QPlainTextEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {{ border-color: {accent}; }}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {{ color: {muted}; background: {panel}; }}
QComboBox {{ padding-right: 22px; }}
QComboBox::drop-down {{ border: none; width: 20px; subcontrol-origin: padding; subcontrol-position: center right; }}
QComboBox::down-arrow {{ image: url("{img_chevron}"); width: 10px; height: 10px; }}
QComboBox::down-arrow:on {{ image: url("{img_chevron_up}"); }}
QComboBox QAbstractItemView {{ background: {raised}; border: 1px solid {border}; border-radius: 6px; padding: 4px;
    outline: none; selection-background-color: {hover}; selection-color: {text}; }}
QAbstractSpinBox {{ padding-right: 18px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border; width: 16px; border: none;
    background: transparent; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 6px; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 6px; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {hover}; }}
QAbstractSpinBox::up-arrow {{ image: url("{img_chevron_up}"); width: 8px; height: 8px; }}
QAbstractSpinBox::down-arrow {{ image: url("{img_chevron}"); width: 8px; height: 8px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off,
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{ image: none; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {border}; }}
QTabBar::tab {{ background: transparent; color: {muted}; padding: 6px 12px; border: none;
               border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {text}; border-bottom-color: {accent}; }}
QTabBar::tab:hover {{ color: {text}; }}
QTabBar::close-button {{ image: url("{img_close}"); subcontrol-position: right; border-radius: 4px;
    width: 14px; height: 14px; margin-left: 4px; }}
QTabBar::close-button:hover {{ image: url("{img_close_hover}"); background: {hover}; }}
QDockWidget {{ color: {muted}; titlebar-close-icon: none; }}
QDockWidget::title {{ background: {bg}; padding: 4px 8px; border-bottom: 1px solid {border}; }}
QWidget#dockTitle {{ background: {bg}; border-bottom: 1px solid {border}; }}
QLabel#dockTitleText {{ color: {muted}; }}
QToolButton#dockFold {{ color: {muted}; border: none; padding: 0 3px; }}
QToolButton#dockFold:hover {{ color: {accent}; }}
QTreeView, QListView, QTableView, QTableWidget {{ background: {bg}; alternate-background-color: {panel};
    border: 1px solid {border}; gridline-color: {border}; selection-background-color: {raised};
    selection-color: {text}; outline: none; }}
QTreeView::item {{ padding: 3px 2px; }}
QTreeView::item:hover {{ background: {hover}; }}
QTreeView::item:selected {{ background: {accent_soft}; color: {text}; }}
QHeaderView::section {{ background: {panel}; color: {muted}; border: none; border-right: 1px solid {border};
                        border-bottom: 1px solid {border}; padding: 4px 6px; }}
QGroupBox {{ border: 1px solid {border}; border-radius: 6px; margin-top: 14px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {secondary}; }}
QSlider::groove:horizontal {{ height: 4px; background: {border}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 10px; height: 10px; margin: -5px 0; border-radius: 7px; background: {raised};
    border: 2px solid {accent}; }}
QSlider::handle:horizontal:hover {{ background: {accent_soft}; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: transparent; border: none; width: 10px; height: 10px; }}
QScrollBar::handle {{ background: {border}; border-radius: 3px; min-height: 24px; min-width: 24px; margin: 2px; }}
QScrollBar::handle:hover {{ background: {muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; border: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QStatusBar {{ background: {bg}; color: {muted}; border-top: 1px solid {border}; }}
QLabel#muted {{ color: {muted}; }}
QLabel#section {{ color: {muted}; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
QFrame#rail {{ background: {bg}; border-right: 1px solid {border}; }}
QFrame#viewbar {{ background: {bg}; border-bottom: 1px solid {border}; }}
QCheckBox {{ color: {text}; spacing: 6px; }}
QCheckBox::indicator, QTreeView::indicator, QTableView::indicator, QListView::indicator {{ width: 14px; height: 14px;
    border: 1.5px solid {muted}; border-radius: 4px; background: {bg}; }}
QCheckBox::indicator:hover, QTreeView::indicator:hover, QTableView::indicator:hover {{ border-color: {accent}; }}
QCheckBox::indicator:checked, QTreeView::indicator:checked, QTableView::indicator:checked,
QListView::indicator:checked {{ background: {accent}; border-color: {accent}; image: url("{img_check}"); }}
QCheckBox::indicator:disabled {{ background: {panel}; border-color: {border}; }}
QFrame#rail QToolButton {{ font-size: 18px; color: {secondary}; }}
QFrame#rail QToolButton:checked {{ color: {accent}; background: {raised}; }}

/* notebook */
QWidget#notebookPage, QScrollArea#notebookScroll, QScrollArea#notebookScroll > QWidget > QWidget {{ background: {panel}; }}
QFrame#cell {{ background: {raised}; border: 1px solid {border}; border-radius: 8px; }}
QToolBar#activity-bar {{ background: {panel}; border: none; border-right: 1px solid {border}; padding: 6px 2px; spacing: 6px; }}
QToolBar#activity-bar QToolButton {{ padding: 6px; border-radius: 6px; }}
QToolBar#activity-bar QToolButton:checked {{ background: {accent_soft}; }}
QFrame#pluginRow {{ background: {raised}; border: 1px solid {border}; border-radius: 6px; }}
QFrame#pluginRow:hover {{ border-color: {accent}; }}
QFrame#pluginRow QLabel {{ background: transparent; }}
QFrame#pluginRow[incompatible="true"] QLabel {{ color: {muted}; }}
QLabel#pluginSummary {{ color: {secondary}; }}
QLabel#pluginKinds {{ color: {data}; font-size: 11px; }}
QPushButton#pluginInstall {{ background: {accent}; color: {on_accent}; border: none; border-radius: 6px; padding: 3px 12px;
    font-weight: 600; }}
QPushButton#pluginInstall:hover {{ background: {accent_hover}; }}
QPushButton#pluginInstall:disabled {{ background: {border}; color: {muted}; }}
QPushButton#pluginButton {{ border-radius: 6px; padding: 3px 12px; }}
QFrame#cell[current="true"] {{ border: 2px solid {accent}; }}
QFrame#cell[state="error"] {{ border-color: {error}; }}
QFrame#cell[state="stale"] {{ background: {panel}; }}
QFrame#cell QLabel, QFrame#cell QCheckBox {{ background: transparent; }}
QToolButton#cellRun {{ color: {accent}; padding: 2px 6px; border-radius: 12px; }}
QToolButton#cellRun:hover {{ background: {accent_soft}; color: {accent}; }}
QLabel#cellCounter {{ color: {muted}; font-size: 11px; }}
QFrame#restrictedBanner {{ background: {data_soft}; border-bottom: 1px solid {data}; }}
QFrame#restrictedBanner QLabel {{ color: {text}; background: transparent; }}
QToolButton#cellFold {{ color: {muted}; padding: 0 4px; border: none; }}
QToolButton#cellFold:hover {{ color: {accent}; }}
QLabel#cellFolded {{ color: {secondary}; font-style: italic; padding: 3px 0; }}
QFrame#cell[current="true"] QLabel#cellCounter {{ color: {accent}; }}
QPlainTextEdit#cellEditor {{ background: {panel}; border: 1px solid {border}; border-radius: 6px; padding: 4px 6px; }}
QLabel#cellError {{ color: {error}; }}
QLabel#cellValue, QLabel#cellText {{ color: {secondary}; }}
QLabel#cellMarkdown {{ color: {text}; font-size: 13px; padding: 2px 4px; }}
QFrame#cell[kind="markdown"] {{ background: transparent; border-color: transparent; }}
QLabel#cellTitle, QLabel#stepTitle {{ font-weight: 600; font-size: 14px; }}
QLabel#stepBadge {{ background: {data_soft}; color: {data}; border-radius: 4px; padding: 1px 6px; font-size: 11px; }}
QLabel#stepSummary {{ color: {secondary}; }}
QLabel#stepResult {{ color: {text}; }}
QLabel#stepError {{ color: {error}; }}
QToolButton#segment {{ background: {panel}; color: {muted}; border: none; border-radius: 5px; padding: 3px 10px;
    font-size: 11px; }}
QToolButton#segment:hover {{ color: {text}; background: {hover}; }}
QToolButton#segment:checked {{ background: {accent_soft}; color: {accent}; font-weight: 600; }}
QPushButton#addCell {{ background: transparent; border: 1px solid transparent; border-radius: 14px; padding: 4px 12px;
    color: {secondary}; font-weight: normal; }}
QPushButton#addCell:hover {{ background: {accent_soft}; color: {accent}; }}
QPushButton#addCell::menu-indicator {{ image: none; width: 0; }}
QTabWidget#notebookTabs QTabBar::tab {{ font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace; padding: 7px 14px; }}
QTabWidget#notebookTabs QTabBar::tab:selected {{ color: {accent}; border-bottom: 2px solid {accent}; font-weight: 600; }}
QToolButton#tabNew {{ font-size: 16px; color: {muted}; padding: 0 8px; border-radius: 6px; }}
QToolButton#tabNew:hover {{ color: {accent}; }}
"""


def stylesheet(font_size: int = 12) -> str:
    images = {f"img_{name}": image(name) for name in _IMAGES}
    return QSS.format(**TOKENS, **images).replace("font-size: 12px", f"font-size: {font_size}px")


def current_font_size(app) -> int:
    """The UI font size of the style sheet in use (12 before any)."""
    import re

    m = re.search(r"font-size: (\d+)px", app.styleSheet() or "")
    return int(m.group(1)) if m else 12


def apply(app, name: str | None = None, font_size: int | None = None):
    from PySide6 import QtGui

    if name is not None:
        use(name)
    if not app.property("ewald_fusion"):  # style().name() is empty once a style sheet wraps it
        app.setStyle("Fusion")
        app.setProperty("ewald_fusion", True)
    pal = app.palette()
    for role, key in ((QtGui.QPalette.Window, "panel"), (QtGui.QPalette.Base, "bg"),
                      (QtGui.QPalette.Text, "text"), (QtGui.QPalette.WindowText, "text"),
                      (QtGui.QPalette.Button, "raised"), (QtGui.QPalette.ButtonText, "text"),
                      (QtGui.QPalette.Highlight, "accent"), (QtGui.QPalette.ToolTipBase, "raised"),
                      (QtGui.QPalette.ToolTipText, "text"), (QtGui.QPalette.PlaceholderText, "muted")):
        pal.setColor(role, QtGui.QColor(TOKENS[key]))
    if pal != app.palette():
        app.setPalette(pal)
    css = stylesheet(font_size or current_font_size(app))
    if app.styleSheet() != css:  # re-polishing every widget of every window is slow; skip when unchanged
        app.setStyleSheet(css)
