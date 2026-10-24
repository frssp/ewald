"""UI themes from the design tokens (project design files, redesign canvas).

Graphite (dark, the default), Paper (light), Midnight, Solarized light/dark and High contrast;
``@ui_theme`` plugins add more (:func:`palette`). ``accent`` (blue in Paper and Graphite) marks what you can act on,
``data`` (orange) marks results computed from the data. ``TOKENS`` is the active palette; it is
updated in place by :func:`use`, so ``from .theme import TOKENS`` sees the change.
"""
from __future__ import annotations

import logging
import re
from string import Template

from ..ui_tokens import (DEFAULT, DENSITY, DERIVED, DESCRIPTIONS, PALETTES, STYLE_DEFAULTS, _mix,  # noqa: F401
                         derive)

log = logging.getLogger(__name__)

TOKENS = dict(PALETTES[DEFAULT])
#: What the active UI theme sets besides colors (fonts, radius, density, icon stroke, extra QSS);
#: updated in place by :func:`use` like :data:`TOKENS`.
STYLE = dict(STYLE_DEFAULTS)


# Small line images the style sheet needs as files (QSS has no inline images), in palette colors.
_IMAGES = {
    "chevron": ("secondary", '<path d="M6 9l6 6 6-6"/>'),
    "chevron_up": ("secondary", '<path d="M6 15l6-6 6 6"/>'),
    "chevron_right": ("secondary", '<path d="M9 6l6 6-6 6"/>'),
    "chevron_left": ("secondary", '<path d="M15 6l-6 6 6 6"/>'),
    "check": ("on_accent", '<path d="M5 12.5l4.5 4.5L19 7.5"/>'),
    "close": ("muted", '<path d="M7 7l10 10M17 7 7 17"/>'),
    "close_hover": ("text", '<path d="M7 7l10 10M17 7 7 17"/>'),
    # the grip of a resize handle: a short pill on its rule (a vertical, a horizontal bar), wider
    # on hover. Qt squares SVG style-sheet images, so these are written as PNG (plus @2x) at
    # their own 7 x 32 px size; the fourth field is the stroke width.
    "grip_v": ("muted", '<path d="M3.5 4v24"/>', "0 0 7 32", 3),
    "grip_h": ("muted", '<path d="M4 3.5h24"/>', "0 0 32 7", 3),
    "grip_v_hover": ("accent", '<path d="M3.5 3.5v25"/>', "0 0 7 32", 5),
    "grip_h_hover": ("accent", '<path d="M3.5 3.5h25"/>', "0 0 32 7", 5),
}


def image(name: str, tokens: dict | None = None) -> str:
    """Path of the style-sheet image ``name`` drawn in the colors of ``tokens`` (the active palette)."""
    import hashlib
    import os
    import tempfile

    key, path_d, *box = _IMAGES[name]
    box, stroke = box or (None, 2.4)
    color = (tokens or TOKENS)[key]
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="{}" fill="none" stroke="{}" stroke-width="{}" '
           'stroke-linecap="round" stroke-linejoin="round">{}</svg>')
    # the file name carries a hash of the drawing, so a changed image never reuses an old file
    digest = hashlib.sha1(repr((path_d, box, stroke)).encode()).hexdigest()[:8]
    stem = os.path.join(tempfile.gettempdir(), f"ewald-ui-{name}-{color.lstrip('#')}-{digest}")
    if box is None:
        path = stem + ".svg"
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(svg.format("0 0 24 24", color, stroke, path_d))
        return path.replace("\\", "/")
    path = stem + ".png"
    if not os.path.exists(path):
        from PySide6 import QtCore, QtGui, QtSvg

        w, h = (int(v) for v in box.split()[2:])
        renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.format(box, color, stroke, path_d).encode()))
        for scale, out in ((1, path), (2, stem + "@2x.png")):
            img = QtGui.QImage(w * scale, h * scale, QtGui.QImage.Format_ARGB32_Premultiplied)
            img.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(img)
            renderer.render(painter)
            painter.end()
            img.save(out)
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


def style(name: str) -> dict:
    """The non-color settings of UI theme ``name`` (:data:`ewald.ui_tokens.STYLE_DEFAULTS` for
    the built-in themes and for a plugin whose style keys are broken)."""
    if name in PALETTES or not name:
        return dict(STYLE_DEFAULTS)
    from .. import plugins

    found = plugins.ui_themes().get(name)
    if found is not None:
        try:
            return found.style()
        except Exception as exc:
            log.warning("UI theme %r: %s; using the default fonts and sizes", name, exc)
    return dict(STYLE_DEFAULTS)


def use(name: str) -> dict:
    """Make ``name`` the active palette and style (unknown names fall back to the default)."""
    TOKENS.clear()
    TOKENS.update(palette(name))
    STYLE.clear()
    STYLE.update(style(name))
    return TOKENS

QSS = """
* {{ font-family: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", sans-serif; font-size: 12px; }}
QMainWindow, QWidget {{ background: {panel}; color: {text}; }}
/* resize handles between docks and split panels: a 7 px hit area (the split cursor shows over all
   of it) with a 1 px rule that thickens into a pill-shaped grip in the middle; on hover both turn
   accent and widen. QSS draws no borders on these, so the rule is a hard-stopped gradient;
   {rule} sits between the border and muted tokens so the rule and its grip read as one. */
QMainWindow::separator {{ width: 7px; height: 7px; }}
QMainWindow::separator:vertical, QSplitter::handle:horizontal {{ width: 7px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {panel}, stop:0.428 {panel},
        stop:0.429 {rule}, stop:0.571 {rule}, stop:0.572 {panel}, stop:1 {panel});
    image: url({img_grip_v}); }}
QMainWindow::separator:horizontal, QSplitter::handle:vertical {{ height: 7px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {panel}, stop:0.428 {panel},
        stop:0.429 {rule}, stop:0.571 {rule}, stop:0.572 {panel}, stop:1 {panel});
    image: url({img_grip_h}); }}
QMainWindow::separator:vertical:hover, QSplitter::handle:horizontal:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {panel}, stop:0.428 {panel},
        stop:0.429 {accent}, stop:0.571 {accent}, stop:0.572 {panel}, stop:1 {panel});
    image: url({img_grip_v_hover}); }}
QMainWindow::separator:horizontal:hover, QSplitter::handle:vertical:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {panel}, stop:0.428 {panel},
        stop:0.429 {accent}, stop:0.571 {accent}, stop:0.572 {panel}, stop:1 {panel});
    image: url({img_grip_h_hover}); }}
QMenuBar {{ background: {bg}; color: {secondary}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {raised}; color: {text}; }}
QMenu {{ background: {raised}; border: 1px solid {border}; padding: 4px; }}
QMenu::separator {{ height: 1px; background: {border}; margin: 4px 8px; }}
QMenu::item:disabled {{ color: {muted}; }}
QToolBar {{ background: {bg}; border: none; border-bottom: 1px solid {border}; spacing: 4px; padding: 4px 6px; }}
QToolBar::separator {{ background: {border}; width: 1px; margin: 6px 6px; }}
QToolBar QLabel, QWidget#toolbarSpacer {{ background: transparent; }}

/* buttons: flat, no bevels; ghost tool buttons, quiet secondary push buttons, one filled primary */
QToolButton {{ background: transparent; color: {secondary}; border: none; border-radius: 6px; padding: 5px 8px; }}
QToolButton:hover {{ background: {hover}; color: {text}; }}
QToolButton:pressed {{ background: {pressed}; }}
QToolButton:checked {{ background: {accent_soft}; color: {accent}; }}
QToolButton:disabled {{ color: {muted}; background: transparent; }}
QToolButton[popupMode="2"] {{ padding-right: 18px; }}
QToolButton::menu-indicator {{ image: url("{img_chevron}"); width: 9px; height: 9px;
    subcontrol-origin: padding; subcontrol-position: center right; right: 5px; }}
QToolButton::menu-indicator:pressed, QToolButton::menu-indicator:open {{ top: 0; left: 0; }}
QToolButton::menu-arrow {{ image: url("{img_chevron}"); width: 9px; height: 9px; }}
QToolButton::menu-button {{ border: none; width: 14px; }}
QMenu::right-arrow {{ image: url("{img_chevron_right}"); width: 10px; height: 10px; }}
QToolButton#cellMenu {{ padding: 0; }}
QToolButton#cellMenu::menu-indicator {{ image: none; width: 0; }}
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
QTabBar QToolButton {{ background: {bg}; border-radius: 0; padding: 0; }}
QTabBar QToolButton::left-arrow {{ image: url("{img_chevron_left}"); width: 12px; height: 12px; }}
QTabBar QToolButton::right-arrow {{ image: url("{img_chevron_right}"); width: 12px; height: 12px; }}
QTabBar::close-button {{ image: url("{img_close}"); subcontrol-position: right; border-radius: 4px;
    width: 14px; height: 14px; margin-left: 4px; }}
QTabBar::close-button:hover {{ image: url("{img_close_hover}"); background: {hover}; }}
QDockWidget {{ color: {muted}; titlebar-close-icon: none; }}
QDockWidget::title {{ background: {bg}; padding: 4px 8px; border-bottom: 1px solid {border}; }}
/* panel headers (header.py): dock title bars and tab-group heads share height, inset and buttons */
QWidget#panelHeader {{ background: {bg}; border: none; border-bottom: 1px solid {border}; }}
QLabel#panelTitle {{ color: {secondary}; font-weight: 600; background: transparent; padding: 0; }}
QWidget#panelHeader QTabBar {{ background: transparent; }}
QWidget#panelHeader QTabBar::tab {{ padding: 0 12px; margin: 0; }}
QToolButton#headerButton, QToolButton#groupButton {{ padding: 0; border-radius: 4px; }}
QToolButton#headerButton::menu-indicator {{ image: none; width: 0; }}
QToolButton#minimalExit {{ background: {raised}; border: 1px solid {border}; padding: 3px 10px; }}
QToolButton#minimalExit:hover {{ background: {hover}; }}
QLabel#toast {{ background: {text}; color: {bg}; border-radius: 8px; padding: 8px 14px; }}
QToolButton#timelineSelection {{ color: {pick}; padding: 2px 6px; }}
QTreeView, QListView, QTableView, QTableWidget {{ background: {bg}; alternate-background-color: {panel};
    border: 1px solid {border}; gridline-color: {border}; selection-background-color: {raised};
    selection-color: {text}; outline: none; }}
QTreeView::item {{ padding: 3px 2px; }}
QTreeView::branch {{ background: transparent; }}
QTreeView::branch:closed:has-children {{ image: url("{img_chevron_right}"); }}
QTreeView::branch:open:has-children {{ image: url("{img_chevron}"); }}
QTreeView::item:hover {{ background: {hover}; }}
QTreeView::item:selected {{ background: {accent_soft}; color: {text}; }}
QHeaderView::section {{ background: {panel}; color: {muted}; border: none; border-right: 1px solid {border};
                        border-bottom: 1px solid {border}; padding: 4px 6px; }}
QHeaderView::down-arrow {{ image: url("{img_chevron}"); width: 10px; height: 10px; }}
QHeaderView::up-arrow {{ image: url("{img_chevron_up}"); width: 10px; height: 10px; }}
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
QLabel#welcomeTitle {{ font-size: 20px; font-weight: 600; }}
QToolButton#welcomeRecent {{ color: {accent}; border: none; padding: 3px 0px; text-align: left; }}
QToolButton#welcomeRecent:hover {{ color: {accent_hover}; text-decoration: underline; }}
QFrame#rail {{ background: {bg}; border-right: 1px solid {border}; }}
QFrame#viewbar {{ background: {bg}; border-bottom: 1px solid {border}; }}
QFrame#viewbar QToolButton {{ padding: 4px 6px; }}
QFrame#axisGroup QToolButton {{ padding: 4px 3px; min-width: 14px; }}
QCheckBox {{ color: {text}; spacing: 6px; }}
QCheckBox::indicator, QGroupBox::indicator, QTreeView::indicator, QTableView::indicator, QListView::indicator {{ width: 14px; height: 14px;
    border: 1.5px solid {muted}; border-radius: 4px; background: {bg}; }}
QCheckBox::indicator:hover, QGroupBox::indicator:hover, QTreeView::indicator:hover, QTableView::indicator:hover {{ border-color: {accent}; }}
QCheckBox::indicator:checked, QGroupBox::indicator:checked, QTreeView::indicator:checked, QTableView::indicator:checked,
QListView::indicator:checked {{ background: {accent}; border-color: {accent}; image: url("{img_check}"); }}
QCheckBox::indicator:disabled {{ background: {panel}; border-color: {border}; }}
QFrame#rail QToolButton {{ font-size: 18px; color: {secondary}; }}
QFrame#rail QToolButton:checked {{ color: {accent}; background: {raised}; }}

/* View tab: how the structure is drawn, as cards */
QScrollArea#viewScroll, QScrollArea#viewScroll > QWidget > QWidget, QWidget#viewPage {{ background: {panel}; }}
QFrame#viewSection {{ background: {raised}; border: 1px solid {border}; border-radius: 8px; }}
QFrame#viewSection QLabel, QFrame#viewSection QCheckBox, QFrame#viewSection QGroupBox {{ background: transparent; }}
QWidget#viewBody {{ background: transparent; }}
QLabel#viewSectionTitle {{ color: {secondary}; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; }}
QWidget#viewSectionHead {{ background: transparent; min-height: 26px; }}
QToolButton#viewFold {{ padding: 0; border-radius: 4px; background: transparent; }}
QToolButton#viewFold:hover {{ background: {hover}; }}
QLabel#viewStatus {{ color: {muted}; }}
QWidget#viewGroup {{ background: transparent; }}
QFrame#viewRule {{ background: {border}; border: none; }}
QCheckBox#viewGroupTitle {{ color: {secondary}; font-weight: 600; }}
QCheckBox#viewGroupTitle[set="true"] {{ color: {accent}; }}
QTableWidget#viewTable {{ border: 1px solid {border}; border-radius: 6px; background: {bg}; }}
QTableWidget#viewTable::item {{ padding: 0 6px; border-bottom: 1px solid {border}; }}
QTableWidget#viewTable QHeaderView::section {{ background: {raised}; border: none; border-bottom: 1px solid {border};
    padding: 4px 6px; font-size: 11px; }}
QTableWidget#viewTable QAbstractSpinBox {{ border: none; border-radius: 0; background: transparent; padding: 0 6px 0 0; }}
QTableWidget#viewTable QAbstractSpinBox::up-button, QTableWidget#viewTable QAbstractSpinBox::down-button {{ width: 0; }}
QTableWidget#viewTable QAbstractSpinBox::up-arrow, QTableWidget#viewTable QAbstractSpinBox::down-arrow {{ image: none;
    width: 0; height: 0; }}
QWidget#viewCell {{ background: transparent; }}
QLabel#viewSectionMark {{ color: {accent}; font-size: 11px; }}
QLabel#viewLabel {{ color: {secondary}; }}
QLabel#viewLabel[set="true"] {{ color: {accent}; font-weight: 600; }}
QGroupBox[set="true"]::title {{ color: {accent}; }}
QLabel#viewValue {{ color: {text}; }}
QLabel#viewError {{ color: {error}; }}
QLineEdit[invalid="true"] {{ border: 1px solid {error}; }}
QPushButton#viewAdd::menu-indicator {{ image: none; width: 0; }}

/* notebook */
QWidget#notebookPage, QScrollArea#notebookScroll, QScrollArea#notebookScroll > QWidget > QWidget {{ background: {panel}; }}
QFrame#cell {{ background: {raised}; border: 1px solid {border}; border-radius: 8px; }}
QToolBar#activity-bar {{ background: {panel}; border: none; border-right: 1px solid {border}; padding: 6px 2px; spacing: 6px; }}
QToolBar#activity-bar QToolButton {{ padding: 6px; border-radius: 6px; border-left: 2px solid transparent; }}
QToolBar#activity-bar QToolButton:hover {{ background: {hover}; }}
QToolBar#activity-bar QToolButton:checked {{ background: {accent_soft}; border-left: 2px solid {accent};
    border-top-left-radius: 0; border-bottom-left-radius: 0; }}
QFrame#pluginRow {{ background: {raised}; border: 1px solid {border}; border-radius: 6px; }}
QFrame#pluginRow:hover {{ border-color: {accent}; }}
QFrame#pluginRow[fresh="true"] {{ border: 1px solid {accent}; background: {accent_soft}; }}
QFrame#pluginRow QLabel {{ background: transparent; }}
QFrame#pluginRow[incompatible="true"] QLabel {{ color: {muted}; }}
QFrame#restartBar {{ background: {accent_soft}; border: 1px solid {accent}; border-radius: 6px; }}
QFrame#restartBar QLabel {{ background: transparent; color: {text}; }}
QLabel#pluginSummary {{ color: {secondary}; }}
QLabel#pluginKinds {{ color: {data}; font-size: 11px; }}
QPushButton#pluginInstall {{ background: {accent}; color: {on_accent}; border: none; border-radius: 6px; padding: 3px 12px;
    font-weight: 600; }}
QPushButton#pluginInstall:hover {{ background: {accent_hover}; }}
QPushButton#pluginInstall:disabled {{ background: {border}; color: {muted}; }}
QPushButton#pluginButton {{ border-radius: 6px; padding: 3px 12px; }}
QFrame#cell[current="true"] {{ border-color: {accent}; }}
QFrame#cell[state="error"] {{ border-color: {error}; }}
QFrame#cell[state="stale"] {{ background: {panel}; }}
QFrame#cell[state="running"] {{ border-color: {data}; }}
QFrame#cell[state="running"] QLabel#cellCounter {{ color: {data}; font-weight: 600; }}
QFrame#cell[state="error"] QLabel#cellCounter {{ color: {error}; }}
QWidget#cellGutter, QWidget#cellTools, QWidget#cellContent, QLabel#cellImage {{ background: transparent; }}
/* what a cell printed or returned: its own area under the code, set off by a rule */
QWidget#cellOutput {{ background: transparent; border: none; border-left: 2px solid {rule}; }}
QPlainTextEdit#cellEditor, QPlainTextEdit#cellText, QPlainTextEdit#cellValue, QPlainTextEdit#cellError,
QLabel#cellCounter {{ font-family: "JetBrains Mono", "Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono", monospace; }}
QFrame#cell QLabel, QFrame#cell QCheckBox {{ background: transparent; }}
QFrame#plotCard {{ background: {panel}; border: 1px solid {border}; border-radius: 6px; }}
QFrame#plotCard:hover {{ border-color: {accent}; }}
QLabel#plotCardTitle {{ color: {text}; font-weight: 600; }}
QPushButton#plotCardLink {{ color: {accent}; background: transparent; border: none; padding: 0; text-align: left; }}
QPushButton#plotCardLink:hover {{ text-decoration: underline; }}
QToolButton#cellRun {{ padding: 0; border-radius: 12px; }}
QToolButton#cellRun:hover {{ background: {accent_soft}; color: {accent}; }}
QLabel#cellCounter {{ color: {muted}; font-size: 11px; }}
QFrame#firstRunNotice {{ background: {raised}; border: 1px solid {border}; }}
QFrame#firstRunNotice QLabel {{ background: transparent; }}
QLabel#firstRunTitle {{ color: {text}; font-size: 15px; font-weight: 600; }}
QLabel#firstRunBody {{ color: {secondary}; }}
QFrame#restrictedBanner {{ background: {data_soft}; border-bottom: 1px solid {data}; }}
QFrame#restrictedBanner QLabel {{ color: {text}; background: transparent; }}
QFrame#runBar {{ border-bottom: 1px solid {border}; }}
QFrame#runBar QLabel {{ color: {secondary}; background: transparent; }}
QToolButton#cellFold {{ padding: 0; border: none; }}
QLabel#cellFolded {{ color: {secondary}; font-style: italic; padding: 3px 0; }}
QLabel#cellDrag {{ padding: 2px 0; background: transparent; }}
QFrame#cellDropLine {{ background: {accent}; border-radius: 1px; }}
QFrame#cell[current="true"] QLabel#cellCounter {{ color: {accent}; }}
QPlainTextEdit#cellEditor {{ background: {panel}; border: none; border-radius: 6px; padding: 4px 6px; }}
QFrame#cell[state="stale"] QPlainTextEdit#cellEditor {{ background: {raised}; }}
QLabel#cellError {{ color: {error}; border-left: 3px solid {error}; padding: 2px 8px; }}
QLabel#cellValue, QLabel#cellText {{ color: {secondary}; }}
QPlainTextEdit#cellText, QPlainTextEdit#cellValue, QPlainTextEdit#cellError {{ background: transparent; border: none;
    padding: 0; color: {secondary}; }}
QPlainTextEdit#cellError {{ color: {error}; }}
QLabel#cellRich {{ color: {text}; }}
QScrollArea#cellRichScroll, QScrollArea#cellRichScroll > QWidget > QWidget {{ background: transparent; }}
QProgressBar#cellProgress {{ border: none; border-radius: 3px; background: {raised}; max-height: 6px; }}
QProgressBar#cellProgress::chunk {{ background: {accent}; border-radius: 3px; }}
QLabel#cellMarkdown {{ color: {text}; font-size: 13px; padding: 2px 4px; }}
QFrame#cell[kind="markdown"] {{ background: transparent; border-color: transparent; }}
QFrame#cell[selected="true"] {{ background: {accent_soft}; }}
QFrame#cell[flash="true"] {{ background: {accent_soft}; border-color: {accent}; }}
QLabel#cellTitle, QLabel#stepTitle {{ font-weight: 600; font-size: 14px; }}
QLabel#stepBadge {{ background: {data_soft}; color: {data}; border-radius: 4px; padding: 1px 6px; font-size: 11px; }}
QLabel#stepSummary {{ color: {secondary}; }}
QLabel#stepResult {{ color: {text}; }}
QLabel#stepError {{ color: {error}; }}
QToolButton#segment {{ background: {panel}; color: {muted}; border: none; border-radius: 5px; padding: 3px 10px;
    font-size: 11px; }}
QToolButton#segment:hover {{ color: {text}; background: {hover}; }}
QToolButton#segment:checked {{ background: {accent_soft}; color: {accent}; font-weight: 600; }}
QToolButton#cellFrames {{ background: {accent_soft}; color: {accent}; border: none; border-radius: 5px;
    padding: 1px 8px; font-weight: 600; }}
QToolButton#cellFrames::menu-indicator {{ image: none; width: 0; }}
QPushButton#addCell {{ background: transparent; border: 1px solid transparent; border-radius: 14px; padding: 4px 12px;
    color: {secondary}; font-weight: normal; }}
QPushButton#addCell:hover {{ background: {accent_soft}; color: {accent}; }}
QPushButton#addCell::menu-indicator {{ image: none; width: 0; }}
QTabBar#notebookTabBar::tab {{ font-family: "JetBrains Mono", "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace;
    padding: 0 12px; }}
QTabBar#notebookTabBar::tab:selected {{ color: {accent}; border-bottom: 2px solid {accent}; font-weight: 600; }}
QWidget#notebookTabRow, QTabBar#notebookTabBar {{ background: transparent; }}
QWidget#chatPanel, QWidget#chatPage, QScrollArea#chatScroll, QWidget#chatWelcome {{ background: {bg}; }}
QLabel#chatContext {{ color: {muted}; background: transparent; font-size: 11px; }}
QLabel#chatContext[picked="true"] {{ color: {pick}; }}
QToolButton#chatHeaderButton {{ color: {secondary}; border-radius: 6px; padding: 3px 10px; }}
QToolButton#chatHeaderButton:hover {{ background: {hover}; color: {text}; }}
QFrame#chatUserBubble {{ background: {accent_soft}; border-radius: 14px; border-top-right-radius: 4px; }}
QLabel#chatUserText {{ color: {text}; background: transparent; }}
QLabel#chatMeta {{ color: {muted}; font-size: 10px; background: transparent; }}
QLabel#chatAvatar, QLabel#chatAvatarLarge {{ background: {accent}; color: {on_accent}; font-weight: 700;
    border-radius: 13px; }}
QLabel#chatAvatarLarge {{ border-radius: 22px; font-size: 20px; }}
QLabel#chatAuthor {{ color: {text}; background: transparent; font-weight: 600; }}
QLabel#chatText {{ color: {text}; background: transparent; }}
QLabel#chatError {{ color: {error}; background: transparent; }}
QLabel#chatNarration {{ color: {secondary}; background: transparent; }}
QLabel#chatNote {{ color: {muted}; background: transparent; font-size: 11px; }}
QLabel#chatThinking {{ color: {muted}; background: transparent; font-style: italic; }}
QLabel#chatStep {{ background: {panel}; color: {secondary}; border: 1px solid {border}; border-radius: 10px;
    padding: 2px 9px; font-size: 11px; }}
QToolButton#chatCellLink {{ background: {accent_soft}; color: {accent}; border: 1px solid transparent;
    border-radius: 10px; padding: 1px 8px 1px 6px; font-size: 11px; font-weight: 600; }}
QToolButton#chatCellLink:hover {{ border-color: {accent}; }}
QToolButton#aiCellPill {{ background: {accent}; color: {on_accent}; border: none; border-radius: 12px;
    padding: 3px 12px 3px 8px; font-size: 11px; font-weight: 600; }}
QToolButton#aiCellPill:hover {{ background: {accent_hover}; }}
QToolButton#chatCellLink:disabled {{ background: transparent; color: {muted}; font-weight: 400;
    text-decoration: line-through; }}
QFrame#chatCode {{ background: {panel}; border: 1px solid {border}; border-radius: 8px; }}
QLabel#chatCodeLang {{ color: {muted}; font-size: 11px; background: transparent; }}
QToolButton#chatCodeButton {{ color: {secondary}; border-radius: 5px; padding: 2px 8px; font-size: 11px; }}
QToolButton#chatCodeButton:hover {{ background: {accent_soft}; color: {accent}; }}
QPlainTextEdit#chatCodeEditor {{ background: transparent; border: none; padding: 2px 8px; }}
QFrame#chatComposer {{ background: {raised}; border: 1px solid {border}; border-radius: 16px; }}
QPlainTextEdit#chatInput {{ background: transparent; border: none; }}
QToolButton#chatSend {{ background: {accent}; color: {on_accent}; border-radius: 15px; font-weight: 700; font-size: 15px; }}
QToolButton#chatSend:hover {{ background: {accent_hover}; }}
QToolButton#chatSend:disabled {{ background: {border}; color: {muted}; }}
QLabel#chatHint {{ color: {muted}; background: transparent; font-size: 10px; }}
QLabel#chatWelcomeTitle {{ color: {text}; background: transparent; font-size: 15px; font-weight: 600; }}
QPushButton#chatStarter {{ background: {raised}; border: 1px solid {border}; border-radius: 14px; padding: 5px 14px;
    color: {secondary}; font-weight: normal; }}
QPushButton#chatStarter:hover {{ border-color: {accent}; color: {accent}; }}
"""


_RADIUS = re.compile(r"(border(?:-\w+)*-radius: )(\d+)px")
_PADDING = re.compile(r"(padding(?:-\w+)?: )([^;}]+)")


def _restyle(css: str, st: dict) -> str:
    """``css`` with the theme's fonts, corner radius and spacing (:data:`STYLE`) in place of ewald's."""
    if st["font"]:
        css = css.replace('font-family: "IBM Plex Sans"', f'font-family: "{st["font"]}", "IBM Plex Sans"')
    if st["mono_font"]:
        css = css.replace('"JetBrains Mono",', f'"{st["mono_font"]}", "JetBrains Mono",')
    if st["radius"] != STYLE_DEFAULTS["radius"]:
        scale = st["radius"] / STYLE_DEFAULTS["radius"]
        # pills and round avatars (radius above 8 px) keep their shape
        css = _RADIUS.sub(lambda m: m.group(0) if int(m.group(2)) > 8
                          else f"{m.group(1)}{round(int(m.group(2)) * scale)}px", css)
    factor = DENSITY[st["density"]]
    if factor != 1:
        css = _PADDING.sub(lambda m: m.group(1) + re.sub(
            r"(\d+)px", lambda n: f"{max(0, round(int(n.group(1)) * factor))}px", m.group(2)), css)
    return css


#: How far a resize handle's rule moves from the border color toward the grip's (muted): the
#: plain border is too faint to carry the grip.
RULE = 0.45


def stylesheet(font_size: int = 12) -> str:
    images = {f"img_{name}": image(name) for name in _IMAGES}
    rule = _mix(TOKENS["border"], TOKENS["muted"], RULE)
    css = _restyle(QSS.format(**TOKENS, **images, rule=rule), STYLE).replace("font-size: 12px", f"font-size: {font_size}px")
    if STYLE["qss"]:   # the theme's own rules last, so they win
        css += "\n/* UI theme */\n" + Template(STYLE["qss"]).safe_substitute(TOKENS) + "\n"
    return css


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
    # the theme's font size applies while the user's (Preferences > General) is the default
    css = stylesheet(font_size or (STYLE["font_size"] if name is not None else 0) or current_font_size(app))
    from . import fonts

    css = fonts.css(css)   # only families this system has: a missing one costs Qt ~200 ms on macOS
    if app.styleSheet() != css:  # re-polishing every widget of every window is slow; skip when unchanged
        app.setStyleSheet(css)
