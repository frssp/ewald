"""The OPTIMADE side bar: pick a database, type a formula, ``Li-P-S``, ``Li P S`` or a filter,
and open what it finds. A double click (or Open) opens one structure as File › Open from ›
Location does; Insert cell adds ``pipe = load("optimade:...")`` to the notebook without running
it; Open all opens every hit as one frame each. Searches run off the GUI thread."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..optimade import providers as P
from ..optimade.search import search
from .workers import detach


def OptimadePanel(context, parent=None):
    HIT = QtCore.Qt.UserRole

    class _Job(QtCore.QThread):
        done = QtCore.Signal(object, str)

        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def run(self):
            try:
                self.done.emit(self.fn(), "")
            except Exception as exc:  # noqa: BLE001 (a failed search shows in the side bar, never a crash)
                self.done.emit(None, str(exc))

    class Panel(QtWidgets.QWidget):
        def __init__(self):
            super().__init__(parent)
            self.context = context
            self.jobs = []
            self.results = None
            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(8, 8, 8, 8)
            lay.setSpacing(6)
            self.db = QtWidgets.QComboBox(toolTip="Database (more under Preferences › OPTIMADE)")
            lay.addWidget(self.db)
            row = QtWidgets.QHBoxLayout()
            self.search = QtWidgets.QLineEdit(placeholderText="LiFePO4, Li-P-S, Li P S or a filter")
            self.search.setClearButtonEnabled(True)
            self.search.setToolTip("A formula, a chemical system (Li-P-S: exactly these elements), elements it "
                                   "contains (Li P S), or an OPTIMADE filter such as\n"
                                   'elements HAS ALL "Si","O" AND nsites<=12')
            self.search.returnPressed.connect(self.run_search)
            row.addWidget(self.search, 1)
            go = QtWidgets.QToolButton(text="Search")
            go.clicked.connect(self.run_search)
            row.addWidget(go)
            lay.addLayout(row)
            self.tree = QtWidgets.QTreeWidget(objectName="optimadeResults", columnCount=3, rootIsDecorated=False)
            self.tree.setHeaderLabels(["Formula", "Id", "Sites"])
            self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            self.tree.itemDoubleClicked.connect(lambda *_: self.open())
            self.tree.itemSelectionChanged.connect(self._selected)
            lay.addWidget(self.tree, 1)
            buttons = QtWidgets.QHBoxLayout()
            self.buttons = {}
            for text, slot, tip in (("Open", self.open, "Open the selected structure (double click does the same)"),
                                    ("Insert cell", self.insert_cell, 'A notebook cell pipe = load("optimade:..."), not run'),
                                    ("Open all", self.open_all, "Every hit as one frame each")):
                b = QtWidgets.QPushButton(text, toolTip=tip)
                b.clicked.connect(lambda _=False, slot=slot: slot())
                b.setEnabled(False)
                buttons.addWidget(b)
                self.buttons[text] = b
            lay.addLayout(buttons)
            self.note = QtWidgets.QLabel(objectName="muted", wordWrap=True)
            self.note.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            lay.addWidget(self.note)
            self.fill_databases()
            context.on("settings", lambda _s: self.fill_databases())

        def fill_databases(self):
            current = self.db.currentData()
            self.db.clear()
            try:
                known = P.providers()
                default = current or P.get(None).name
            except (KeyError, ValueError) as exc:
                self._say(str(exc), error=True)
                known, default = P.BUILTIN, P.DEFAULT
            for p in known.values():
                self.db.addItem(f"{p.title} ({p.name})", p.name)
            i = self.db.findData(default)
            self.db.setCurrentIndex(max(i, 0))

        # -------------------------------------------------------------- search
        def run_search(self):
            db, text = self.db.currentData(), self.search.text().strip()
            self._say("Searching…")
            self.tree.clear()
            for b in self.buttons.values():
                b.setEnabled(False)
            job = detach(_Job(lambda: search(text, db=db)))
            job.done.connect(self._show)
            job.finished.connect(lambda: self.jobs.remove(job) if job in self.jobs else None)
            self.jobs.append(job)
            job.start()

        def wait(self, ms: int = 10000):
            """Block until every search has answered (tests, screenshots)."""
            while self.jobs:
                self.jobs[0].wait(ms)
                QtWidgets.QApplication.processEvents()

        def _show(self, results, error):
            self.results = results
            if error:
                self._say(error, error=True)
                return
            for h in results:
                item = QtWidgets.QTreeWidgetItem([h.formula, h.id, "" if h.nsites is None else str(h.nsites)])
                item.setData(0, HIT, h)
                item.setToolTip(0, h.uri)
                self.tree.addTopLevelItem(item)
            for col in range(3):
                self.tree.resizeColumnToContents(col)
            total = results.total
            more = f" of {total}" if total is not None and total > len(results) else ""
            self._say(f"{len(results)}{more} found · {results.filter or 'all'}" if results else
                      f"Nothing found · {results.filter}")
            self.buttons["Open all"].setEnabled(len(results) > 1)

        def _selected(self):
            n = len(self.selected())
            self.buttons["Open"].setEnabled(n == 1)
            self.buttons["Insert cell"].setEnabled(n >= 1)

        def selected(self):
            return [it.data(0, HIT) for it in self.tree.selectedItems() if it.data(0, HIT) is not None]

        # -------------------------------------------------------------- actions
        def open(self):
            hits = self.selected()
            if len(hits) == 1:
                self.context.open(hits[0].uri)

        def insert_cell(self):
            hits = self.selected()
            if not hits:
                return
            if len(hits) == 1:
                code = f'pipe = load("{hits[0].uri}")  # {hits[0].formula}'
            else:
                code = "\n".join(f'{_var(h)} = load("{h.uri}")  # {h.formula}' for h in hits) + f"\npipe = {_var(hits[0])}"
            self.context.insert_cell(code)

        def open_all(self):
            if self.results:
                self.context.open(f"optimade:{self.results.db}?{self.search.text().strip()}")

        def _say(self, text, error=False):
            self.note.setText(text)
            self.note.setObjectName("cellError" if error else "muted")
            self.note.style().polish(self.note)

    def _var(h):
        name = "".join(c if c.isalnum() else "_" for c in h.id).strip("_").lower() or "s"
        return name if name[0].isalpha() else f"s_{name}"

    return Panel()
