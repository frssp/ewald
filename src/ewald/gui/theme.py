"""UI themes from the design tokens (project design files, redesign canvas).

Paper (light, the default) and Graphite (dark). ``accent`` (teal) marks what you can act on,
``data`` (orange) marks results computed from the data. ``TOKENS`` is the active palette; it is
updated in place by :func:`use`, so ``from .theme import TOKENS`` sees the change.
"""
from __future__ import annotations

PALETTES = {
    "paper": {
        "bg": "#FBFBF9", "panel": "#F3F4F1", "raised": "#FFFFFF", "border": "#D5D8D1",
        "text": "#15181C", "secondary": "#3D434A", "muted": "#5C636B", "accent": "#0B7A6F",
        "on_accent": "#FFFFFF", "accent_soft": "#E0EFEC", "data": "#C8551F", "data_soft": "#F7E4D9",
        "selection": "#C99A06", "pick": "#C8551F", "error": "#C62828", "ok": "#2E7D32",
        "viewport": "#ECEDE9",
    },
    "graphite": {
        "bg": "#14161A", "panel": "#1B1E23", "raised": "#23272E", "border": "#30353D",
        "text": "#E7E9EC", "secondary": "#AEB4BE", "muted": "#8C939E", "accent": "#3FB8A6",
        "on_accent": "#0B1220", "accent_soft": "#1E3A36", "data": "#F08A4B", "data_soft": "#3A2A20",
        "selection": "#E8C547", "pick": "#FFD84D", "error": "#FF6B6B", "ok": "#5BD08A",
        "viewport": "#14161A",
    },
}
DEFAULT = "paper"
TOKENS = dict(PALETTES[DEFAULT])


def use(name: str) -> dict:
    """Make ``name`` the active palette (unknown names fall back to the default)."""
    TOKENS.clear()
    TOKENS.update(PALETTES.get(name, PALETTES[DEFAULT]))
    return TOKENS

QSS = """
* {{ font-family: "IBM Plex Sans", "Inter", "Segoe UI", "Helvetica Neue", sans-serif; font-size: 12px; }}
QMainWindow, QWidget {{ background: {panel}; color: {text}; }}
QMainWindow::separator {{ background: {border}; width: 1px; height: 1px; }}
QMenuBar {{ background: {bg}; color: {secondary}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {raised}; color: {text}; }}
QMenu {{ background: {raised}; border: 1px solid {border}; padding: 4px; }}
QMenu::separator {{ height: 1px; background: {border}; margin: 4px 8px; }}
QToolBar {{ background: {bg}; border: none; border-bottom: 1px solid {border}; spacing: 4px; padding: 3px; }}
QToolButton {{ background: transparent; color: {secondary}; border: 1px solid transparent; border-radius: 4px;
              padding: 4px 7px; }}
QToolButton:hover {{ background: {raised}; color: {text}; }}
QToolButton:checked {{ background: {raised}; color: {accent}; border-color: {border}; }}
QPushButton {{ background: {raised}; color: {text}; border: 1px solid {border}; border-radius: 4px; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {accent}; }}
QPushButton#primary {{ background: {accent}; color: {on_accent}; border: none; font-weight: 600; }}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {bg}; color: {text}; border: 1px solid {border}; border-radius: 4px; padding: 3px 5px;
    selection-background-color: {accent}; selection-color: {on_accent}; }}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {accent}; }}
QComboBox QAbstractItemView {{ background: {raised}; border: 1px solid {border}; selection-background-color: {raised};
                               selection-color: {accent}; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {border}; }}
QTabBar::tab {{ background: transparent; color: {muted}; padding: 6px 12px; border: none;
               border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {text}; border-bottom-color: {accent}; }}
QTabBar::tab:hover {{ color: {text}; }}
QDockWidget {{ color: {muted}; titlebar-close-icon: none; }}
QDockWidget::title {{ background: {bg}; padding: 4px 8px; border-bottom: 1px solid {border}; }}
QTreeView, QListView, QTableView, QTableWidget {{ background: {bg}; alternate-background-color: {panel};
    border: 1px solid {border}; gridline-color: {border}; selection-background-color: {raised};
    selection-color: {text}; outline: none; }}
QTreeView::item {{ padding: 3px 2px; }}
QTreeView::item:selected {{ background: {raised}; border-left: 2px solid {accent}; }}
QHeaderView::section {{ background: {panel}; color: {muted}; border: none; border-right: 1px solid {border};
                        border-bottom: 1px solid {border}; padding: 4px 6px; }}
QGroupBox {{ border: 1px solid {border}; border-radius: 6px; margin-top: 14px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {secondary}; }}
QSlider::groove:horizontal {{ height: 4px; background: {border}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 12px; margin: -5px 0; border-radius: 6px; background: {text}; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: {bg}; width: 10px; height: 10px; }}
QScrollBar::handle {{ background: {border}; border-radius: 4px; min-height: 20px; min-width: 20px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QStatusBar {{ background: {bg}; color: {muted}; border-top: 1px solid {border}; }}
QLabel#muted {{ color: {muted}; }}
QLabel#section {{ color: {muted}; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
QFrame#rail {{ background: {bg}; border-right: 1px solid {border}; }}
QFrame#viewbar {{ background: {bg}; border-bottom: 1px solid {border}; }}
QCheckBox {{ color: {text}; }}
QCheckBox::indicator, QTreeView::indicator, QTableView::indicator {{ width: 13px; height: 13px; border: 1px solid {muted};
    border-radius: 3px; background: {bg}; }}
QCheckBox::indicator:checked, QTreeView::indicator:checked {{ background: {accent}; border-color: {accent}; }}
QFrame#rail QToolButton {{ font-size: 18px; color: {secondary}; }}
QFrame#rail QToolButton:checked {{ color: {accent}; background: {raised}; }}
QPushButton#primary:hover {{ background: {accent}; border: none; }}

/* notebook */
QWidget#notebookPage, QScrollArea#notebookScroll, QScrollArea#notebookScroll > QWidget > QWidget {{ background: {panel}; }}
QFrame#cell {{ background: {raised}; border: 1px solid {border}; border-radius: 8px; }}
QToolBar#activity-bar {{ background: {panel}; border: none; border-right: 1px solid {border}; padding: 6px 2px; spacing: 6px; }}
QToolBar#activity-bar QToolButton {{ padding: 6px; border-radius: 6px; }}
QToolBar#activity-bar QToolButton:checked {{ background: {accent_soft}; }}
QFrame#pluginRow {{ background: {raised}; border: 1px solid {border}; border-radius: 6px; }}
QFrame#pluginRow:hover {{ border-color: {accent}; }}
QFrame#pluginRow QLabel {{ background: transparent; }}
QLabel#pluginSummary {{ color: {secondary}; }}
QLabel#pluginKinds {{ color: {data}; font-size: 11px; }}
QPushButton#pluginInstall {{ background: {accent}; color: {on_accent}; border: none; border-radius: 4px; padding: 2px 10px; }}
QPushButton#pluginButton {{ border-radius: 4px; padding: 2px 10px; }}
QFrame#cell[current="true"] {{ border: 2px solid {accent}; }}
QFrame#cell[state="error"] {{ border-color: {error}; }}
QFrame#cell[state="stale"] {{ background: {panel}; }}
QFrame#cell QLabel, QFrame#cell QCheckBox {{ background: transparent; }}
QToolButton#cellRun {{ color: {accent}; padding: 2px 5px; }}
QLabel#cellCounter {{ color: {muted}; font-size: 11px; }}
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
QToolButton#segment {{ border: 1px solid {border}; border-radius: 4px; padding: 2px 8px; font-size: 11px; }}
QToolButton#segment:checked {{ background: {accent_soft}; color: {accent}; border-color: {accent}; }}
QPushButton#addCell {{ border: 1px dashed {muted}; border-radius: 6px; padding: 5px 10px; color: {secondary}; }}
QPushButton#addCell:hover {{ border-color: {accent}; color: {accent}; }}
QPushButton#addCell::menu-indicator {{ image: none; width: 0; }}
QTabWidget#notebookTabs QTabBar::tab {{ font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace; padding: 7px 14px; }}
QTabWidget#notebookTabs QTabBar::tab:selected {{ color: {accent}; border-bottom: 2px solid {accent}; font-weight: 600; }}
QToolButton#tabNew {{ font-size: 16px; color: {muted}; padding: 2px 10px; }}
QToolButton#tabNew:hover {{ color: {accent}; }}
"""


def stylesheet(font_size: int = 12) -> str:
    return QSS.format(**TOKENS).replace("font-size: 12px", f"font-size: {font_size}px")


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
