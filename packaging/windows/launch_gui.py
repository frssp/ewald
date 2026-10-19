"""Frozen entry point for ewald-gui.exe (windowed, no console).

With EWALD_SMOKE_TEST set, the window opens and the app quits after a few seconds with
exit code 0; the release workflow uses this to check the frozen GUI starts.
"""
import os
import sys


def main():
    from ewald.gui.app import main as gui_main

    if os.environ.get("EWALD_SMOKE_TEST"):
        from PySide6 import QtCore, QtWidgets

        def quit_later():
            app = QtWidgets.QApplication.instance()
            if app is None:
                QtCore.QTimer.singleShot(200, quit_later)
            else:
                QtCore.QTimer.singleShot(4000, app.quit)

        # the timer fires once the event loop runs, i.e. after the window is shown
        QtCore.QTimer.singleShot(0, quit_later)
    return gui_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
