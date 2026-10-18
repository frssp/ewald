"""Dark theme from the design tokens (docs: /design/gui-design.md)."""
from __future__ import annotations

TOKENS = {
    "bg": "#14161A", "panel": "#1B1E23", "raised": "#23272E", "border": "#30353D",
    "text": "#E7E9EC", "secondary": "#AEB4BE", "muted": "#8C939E", "accent": "#5AA9FF",
    "selection": "#E8C547", "pick": "#FFD84D", "error": "#FF6B6B", "ok": "#5BD08A",
}

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
QPushButton#primary {{ background: {accent}; color: #0B1220; border: none; font-weight: 600; }}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {bg}; color: {text}; border: 1px solid {border}; border-radius: 4px; padding: 3px 5px;
    selection-background-color: {accent}; selection-color: #0B1220; }}
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
"""


def stylesheet() -> str:
    return QSS.format(**TOKENS)


def apply(app):
    from PySide6 import QtGui

    app.setStyle("Fusion")
    pal = app.palette()
    for role, key in ((QtGui.QPalette.Window, "panel"), (QtGui.QPalette.Base, "bg"),
                      (QtGui.QPalette.Text, "text"), (QtGui.QPalette.WindowText, "text"),
                      (QtGui.QPalette.Button, "raised"), (QtGui.QPalette.ButtonText, "text"),
                      (QtGui.QPalette.Highlight, "accent"), (QtGui.QPalette.ToolTipBase, "raised"),
                      (QtGui.QPalette.ToolTipText, "text"), (QtGui.QPalette.PlaceholderText, "muted")):
        pal.setColor(role, QtGui.QColor(TOKENS[key]))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet())
