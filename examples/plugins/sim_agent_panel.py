"""Example GUI plugin: a "Sim agent" side bar with its own activity-bar button, an "Agent jobs"
tab under the 3D view, menu entries and a page in Preferences. See docs/plugins/gui.md.

Copy this file into ewald's plugin folder (``ewald plugins`` prints where it is) and restart, or
install it from the Plugins side bar: a new button appears at the bottom of the activity bar
(the icon column at the left edge, under AI chat), Tools gets a "Sim agent" submenu, the 3D
view's right-click menu gets "Send picked atoms to Sim agent" while atoms are picked, and
Preferences gets a "Sim agent" page.

The side bar shows the structure on screen and turns a task into a notebook cell; the jobs tab
lists every task sent this session. A real agent plugin would send the task to its server.
"""
from ewald.plugin_api import SettingsField, menu_item, panel, plugin_settings, settings_page

__ewald__ = {"id": "sim-agent-example", "name": "Sim agent (example)", "version": "0.1.0",
             "description": "A side bar, a bottom tab, menu entries and a Preferences page from a plugin",
             "plugin-api": 1}

ICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="5" y="8" width="14" height="11" rx="3"/><path d="M12 4v4M9 13h.01M15 13h.01M9.5 16h5"/>'
        '<circle cx="12" cy="3.5" r="1"/></svg>')


JOBS = []          # tasks sent this session, shared by the side bar, the jobs tab and the menus
_LISTENERS = []


def send_task(context, text):
    """Queue ``text`` (a real plugin would post it to the server) and tell every view of JOBS."""
    s = plugin_settings("sim-agent")
    JOBS.append({"task": text, "frame": context.frame, "server": s["server"]})
    for update in list(_LISTENERS):
        update()
    if s["insert_cells"]:
        context.insert_cell(f"# sim-agent task: {text}\n# (a real plugin would send it to {s['server']})")
    context.message(f"queued {text!r}")


def _follow(widget, update):
    """Call ``update()`` whenever JOBS changes, until ``widget`` is deleted."""
    _LISTENERS.append(update)
    widget.destroyed.connect(lambda *_: update in _LISTENERS and _LISTENERS.remove(update))
    update()


@settings_page("sim-agent", title="Sim agent", fields=[
    SettingsField("server", "Server", default="http://localhost:8000", help="Where the agent listens."),
    SettingsField("token", "Access token", kind="secret", help="Kept in the system keychain."),
    SettingsField("model", "Model", kind="choice", choices=("default", "fast", "accurate"),
                  help="What the agent runs its tasks with."),
    SettingsField("insert_cells", "Insert tasks as cells", kind="bool", default=True,
                  help="Send turns a task into a notebook cell instead of only listing it.")])
class SimAgentSettings:
    """The agent behind the Sim agent side bar."""
    # no load()/save(): ewald stores the values (the token in the keychain); plugin_settings() reads them


@panel(title="Sim agent", icon=ICON, shortcut="Ctrl+Alt+A", width=320)
def sim_agent(context):
    """Send simulation tasks to an agent and follow the structure on screen."""
    from PySide6 import QtWidgets

    box = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(12, 8, 12, 12)
    server = QtWidgets.QLabel()
    structure = QtWidgets.QLabel("No structure loaded", wordWrap=True)
    tasks = QtWidgets.QListWidget()
    task = QtWidgets.QLineEdit(placeholderText="e.g. relax this cell with PBE")
    send = QtWidgets.QPushButton("Send")
    settings = QtWidgets.QPushButton("Settings…")

    def show_server(_settings=None):
        s = plugin_settings("sim-agent")
        server.setText(f"Server: {s['server']}  ({s['model']})" + ("" if s["token"] else "  no token"))

    def show_data(data):
        structure.setText(f"{data.count} atoms, frame {context.frame}")

    def submit():
        text = task.text().strip()
        if text:
            task.clear()
            send_task(context, text)

    def show_jobs():
        tasks.clear()
        tasks.addItems([job["task"] for job in JOBS])

    send.clicked.connect(submit)
    task.returnPressed.connect(submit)
    settings.clicked.connect(lambda: context.open_preferences("Sim agent"))
    context.on("settings", show_server)
    context.on("data", show_data)
    _follow(box, show_jobs)
    show_server()
    if context.data is not None:
        show_data(context.data)
    for w in (server, structure, QtWidgets.QLabel("Tasks"), tasks, task, send, settings):
        lay.addWidget(w)
    return box


@panel(title="Agent jobs", icon="send", area="bottom")
def agent_jobs(context):
    """Tasks sent to the sim agent this session (a tab under the 3D view)."""
    from PySide6 import QtWidgets

    table = QtWidgets.QTableWidget(0, 3)
    table.setHorizontalHeaderLabels(["Task", "Frame", "Server"])
    table.horizontalHeader().setStretchLastSection(True)
    table.verticalHeader().hide()

    def show_jobs():
        table.setRowCount(len(JOBS))
        for row, job in enumerate(JOBS):
            for col, key in enumerate(("task", "frame", "server")):
                table.setItem(row, col, QtWidgets.QTableWidgetItem(str(job[key])))

    _follow(table, show_jobs)
    return table


@menu_item(title="Send picked atoms to Sim agent", menu="3D view", when="picks", icon="send")
def send_picked(context):
    """Send a relaxation task for the atoms picked in the 3D view."""
    send_task(context, f"relax atoms {', '.join(map(str, context.picks))}")


@menu_item(title="Send structure…", menu="Tools/Sim agent", when="data")
def send_structure(context):
    """Ask for a task and send it with the structure on screen."""
    from PySide6 import QtWidgets

    text, ok = QtWidgets.QInputDialog.getText(context.window, "Sim agent", "Task for the structure on screen:")
    if ok and text.strip():
        send_task(context, text.strip())


@menu_item(title="Settings…", menu="Tools/Sim agent")
def agent_settings(context):
    """Open the Sim agent page of Preferences."""
    context.open_preferences("Sim agent")
