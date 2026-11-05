"""``ewald://`` links in the GUI: hand them to a running EWALD, confirm, open.

A browser starts ``ewald-gui ewald://open?...`` for every click. When EWALD is already
running, that new process passes the link to it over a local socket (``QLocalServer``: a
named pipe on Windows, a socket file only this user may use elsewhere) and exits, so the
calculation opens in the window the user already has instead of a second window. On macOS
the running app gets the link as a ``QFileOpenEvent`` instead (:class:`EwaldApplication`).

Parsing and the rules on what a link may open are in :mod:`ewald.deeplink`.
"""
from __future__ import annotations

import getpass
import hashlib
import html
import json

from PySide6 import QtCore, QtNetwork, QtWidgets

from .. import deeplink

FORWARD_TIMEOUT_MS = 1500


def server_name() -> str:
    """One name per user (two users on one machine each have their own EWALD)."""
    try:
        user = getpass.getuser()
    except (OSError, KeyError, ImportError):  # no user name known: still one server per machine
        user = "user"
    return "ewald-links-" + hashlib.sha256(user.encode("utf-8")).hexdigest()[:12]


def forward(links, timeout_ms: int = FORWARD_TIMEOUT_MS) -> bool:
    """Give ``links`` to an EWALD that is already running; False when none answers."""
    sock = QtNetwork.QLocalSocket()
    sock.connectToServer(server_name())
    if not sock.waitForConnected(timeout_ms):
        return False
    sock.write(json.dumps({"links": list(links)}).encode("utf-8") + b"\n")
    ok = sock.waitForBytesWritten(timeout_ms) and sock.waitForReadyRead(timeout_ms)
    answer = bytes(sock.readAll().data()).strip() if ok else b""
    sock.disconnectFromServer()
    return answer == b"ok"


class LinkServer(QtCore.QObject):
    """Listens for links from later ``ewald-gui`` starts and calls ``handler(link)`` for each."""

    def __init__(self, handler, parent=None):
        super().__init__(parent)
        self.handler = handler
        self.server = QtNetwork.QLocalServer(self)
        self.server.setSocketOptions(QtNetwork.QLocalServer.UserAccessOption)
        self.server.newConnection.connect(self._accept)
        self.listening = self._listen()

    def _listen(self) -> bool:
        name = server_name()
        # ask first: with UserAccessOption, Qt on Unix renames its socket over an existing one,
        # so a plain listen() would take the links away from the EWALD that already runs
        probe = QtNetwork.QLocalSocket()
        probe.connectToServer(name)
        if probe.waitForConnected(300):   # another EWALD of this user listens: it keeps the links
            probe.disconnectFromServer()
            return False
        if self.server.listen(name):
            return True
        QtNetwork.QLocalServer.removeServer(name)   # a socket file left by a crashed EWALD
        return self.server.listen(name)

    def _accept(self):
        while self.server.hasPendingConnections():
            sock = self.server.nextPendingConnection()
            sock.setParent(self)
            sock.readyRead.connect(lambda s=sock: self._read(s))
            sock.disconnected.connect(sock.deleteLater)

    def _read(self, sock):
        if not sock.canReadLine():
            if sock.bytesAvailable() > deeplink.MAX_LENGTH * 8:   # never a link: drop it
                sock.abort()
            return
        line = bytes(sock.readLine().data())
        try:
            links = [s for s in json.loads(line.decode("utf-8")).get("links", []) if isinstance(s, str)]
        except (ValueError, AttributeError):
            links = []
        sock.write(b"ok\n" if links else b"bad\n")
        sock.flush()
        for link in links[:8]:
            QtCore.QTimer.singleShot(0, lambda link=link: self.handler(link))

    def close(self):
        self.server.close()


def confirm(window, link: deeplink.DeepLink) -> bool:
    """Ask before opening; the box shows exactly what will be read. "Don't ask again" turns
    ``general.confirm_links`` off."""
    box = QtWidgets.QMessageBox(window)
    box.setObjectName("linkConfirm")
    box.setWindowTitle("Open link")
    box.setIcon(QtWidgets.QMessageBox.Question)
    shown = f"<b>{html.escape(link.title)}</b><br>" if link.title else ""
    box.setText(f"A web page asks EWALD to open:<br><br>{shown}<code>{html.escape(link.location)}</code>")
    box.setInformativeText("Read like Open location (Ctrl+L); nothing in the link is run.")
    box.setTextFormat(QtCore.Qt.RichText)
    check = QtWidgets.QCheckBox("Don't ask again for links")
    box.setCheckBox(check)
    open_button = box.addButton("Open", QtWidgets.QMessageBox.AcceptRole)
    box.addButton(QtWidgets.QMessageBox.Cancel)
    box.setDefaultButton(open_button)
    box.exec()
    accepted = box.clickedButton() is open_button
    if accepted and check.isChecked():
        window.settings["general.confirm_links"] = False
        window.settings.save()
    return accepted


def bring_to_front(window):
    if window.isMinimized():
        window.showNormal()
    window.raise_()
    window.activateWindow()


def open_link(window, text: str):
    """Open the location an ``ewald://`` link names in ``window``, after the user's OK."""
    bring_to_front(window)
    try:
        link = deeplink.parse(text, schemes=window.location_schemes())
    except deeplink.DeepLinkError as exc:
        window.message(f"Link not opened: {exc}", error=True)
        QtWidgets.QMessageBox.warning(window, "Link not opened", str(exc))
        return None
    if window.settings["general.confirm_links"] and not confirm(window, link):
        window.message("Link not opened (cancelled)")
        return None
    return window.open_location_dialog(link.location)
