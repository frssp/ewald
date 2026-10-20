"""The settings page of a plugin (:func:`ewald.plugins.settings_page`) as a dialog, generated
from its fields: Plugins side bar › Settings, Edit › Plugin settings."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .. import plugins


class PluginSettingsDialog(QtWidgets.QDialog):
    """One plugin settings page. Save calls the page's ``save(values)``; a ``ValueError`` it
    raises is shown under the fields and the dialog stays open, so the user can fix the entry.
    Secret fields start empty whatever is stored; left empty, they keep the saved secret."""

    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.page = page
        self.setWindowTitle(f"{page.title} settings")
        self.setMinimumWidth(460)
        lay = QtWidgets.QVBoxLayout(self)
        if page.description:
            lay.addWidget(QtWidgets.QLabel(page.description, objectName="muted", wordWrap=True))
        form = QtWidgets.QFormLayout()
        lay.addLayout(form)
        self.widgets: dict = {}
        try:
            values = page.values()
            problem = ""
        except Exception as exc:          # a broken plugin must not keep the dialog from opening
            values, problem = {f.name: f.empty() for f in page.fields}, f"Could not read the settings: {exc}"
        for f in page.fields:
            w = self._widget(f, values.get(f.name))
            if f.help:
                w.setToolTip(f.help)
            self.widgets[f.name] = w
            form.addRow(f.label, w)
            if f.help:
                form.addRow("", QtWidgets.QLabel(f.help, objectName="muted", wordWrap=True))
        self.status = QtWidgets.QLabel(problem, wordWrap=True, objectName="cellError" if problem else "muted")
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.status.setVisible(bool(problem))
        lay.addWidget(self.status)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        for label, act in page.actions.items():
            b = self.buttons.addButton(label, QtWidgets.QDialogButtonBox.ActionRole)
            b.clicked.connect(lambda _=False, act=act, label=label: self._run(act, label, close=False))
        self.buttons.accepted.connect(lambda: self._run(page.save, "Save", close=True))
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.message = ""

    def _widget(self, f, value):
        if f.kind == "bool":
            return QtWidgets.QCheckBox(checked=bool(value))
        if f.kind == "choice":
            w = QtWidgets.QComboBox()
            w.addItems([str(c) for c in f.choices])
            if value in f.choices:
                w.setCurrentIndex(list(f.choices).index(value))
            return w
        if f.kind == "int":
            w = QtWidgets.QSpinBox(maximum=2**31 - 1, minimum=-(2**31))
            w.setValue(int(value or 0))
            return w
        if f.kind == "multiline":
            w = QtWidgets.QPlainTextEdit(str(value or ""))
            w.setPlaceholderText(f.placeholder)
            w.setFixedHeight(80)
            return w
        w = QtWidgets.QLineEdit()
        if f.kind == "secret":
            w.setEchoMode(QtWidgets.QLineEdit.Password)
            w.setPlaceholderText("saved; type to replace" if value else (f.placeholder or "not set"))
        else:
            w.setText("" if value is None else str(value))
            w.setPlaceholderText(f.placeholder)
        return w

    def values(self) -> dict:
        """What the fields hold now (secrets as typed; "" when left empty)."""
        out = {}
        for f in self.page.fields:
            w = self.widgets[f.name]
            if f.kind == "bool":
                out[f.name] = w.isChecked()
            elif f.kind == "choice":
                out[f.name] = f.choices[w.currentIndex()]
            elif f.kind == "int":
                out[f.name] = w.value()
            elif f.kind == "multiline":
                out[f.name] = w.toPlainText()
            else:
                out[f.name] = w.text().strip()
        return out

    def _run(self, function, label, close):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            message, error = function(self.values()) or "", False
        except ValueError as exc:
            message, error = str(exc), True
        except Exception as exc:
            message, error = f"{label} failed: {exc}", True
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.message = str(message)
        self.status.setText(self.message)
        self.status.setObjectName("cellError" if error else "muted")
        self.status.style().polish(self.status)
        self.status.setVisible(bool(self.message))
        if not error:
            for f in self.page.fields:      # a saved secret is not shown again
                if f.kind == "secret" and self.widgets[f.name].text():
                    self.widgets[f.name].clear()
                    self.widgets[f.name].setPlaceholderText("saved; type to replace")
            if close:
                self.accept()


def edit(name: str, parent=None, exec_: bool = True) -> PluginSettingsDialog:
    """Open the settings page ``name`` (modal unless ``exec_`` is False, as in tests)."""
    dlg = PluginSettingsDialog(plugins.settings_pages()[name], parent)
    if exec_:
        dlg.exec()
    else:
        dlg.show()
    return dlg
