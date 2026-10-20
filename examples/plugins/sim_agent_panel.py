"""Example GUI plugin: a "Sim agent" side bar with its own activity-bar button and a page in
Preferences. See docs/plugins/gui.md.

Copy this file into ewald's plugin folder (``ewald plugins`` prints where it is) and restart, or
install it from the Plugins side bar: a new button appears at the bottom of the activity bar
(the icon column at the left edge, under AI chat), and Preferences gets a "Sim agent" page.

The panel shows the structure on screen, keeps a task list and turns a task into a notebook
cell. A real agent plugin would send the task to its server instead.
"""
from ewald.plugin_api import SettingsField, panel, plugin_settings, settings_page

__ewald__ = {"id": "sim-agent-example", "name": "Sim agent (example)", "version": "0.1.0",
             "description": "A side bar and a Preferences page contributed by a plugin", "plugin-api": 1}

ICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="5" y="8" width="14" height="11" rx="3"/><path d="M12 4v4M9 13h.01M15 13h.01M9.5 16h5"/>'
        '<circle cx="12" cy="3.5" r="1"/></svg>')


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
        if not text:
            return
        tasks.addItem(text)
        task.clear()
        s = plugin_settings("sim-agent")
        if s["insert_cells"]:
            context.insert_cell(f"# sim-agent task: {text}\n# (a real plugin would send it to {s['server']})")
        context.message(f"queued {text!r}")

    send.clicked.connect(submit)
    task.returnPressed.connect(submit)
    settings.clicked.connect(lambda: context.open_preferences("Sim agent"))
    context.on("settings", show_server)
    context.on("data", show_data)
    show_server()
    if context.data is not None:
        show_data(context.data)
    for w in (server, structure, QtWidgets.QLabel("Tasks"), tasks, task, send, settings):
        lay.addWidget(w)
    return box
