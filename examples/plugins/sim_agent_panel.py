"""Example GUI plugin: a "Sim agent" side bar with its own activity-bar button and a Preferences
page. See docs/plugins/gui.md.

Copy this file into ewald's plugin folder (``ewald plugins`` prints where it is) and restart, or
install it from the Plugins side bar: a new button appears at the bottom of the activity bar
(the icon column at the left edge, under AI chat), and Preferences gets a "Sim agent" page.

The panel shows the structure on screen, keeps a task list and turns a task into a notebook
cell. A real agent plugin would send the task to its server (``simagent.server``) instead.
"""
from ewald.plugin_api import Option, panel, settings_page

__ewald__ = {"id": "sim-agent-example", "name": "Sim agent (example)", "version": "0.1.0",
             "description": "A side bar and a Preferences page contributed by a plugin", "plugin-api": 1}

ICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="5" y="8" width="14" height="11" rx="3"/><path d="M12 4v4M9 13h.01M15 13h.01M9.5 16h5"/>'
        '<circle cx="12" cy="3.5" r="1"/></svg>')

settings_page("Sim agent", [
    Option("simagent.server", "http://localhost:8000", "Server", "Where the agent listens."),
    Option("simagent.model", "default", "Model", "What the agent runs its tasks with.",
           choices=("default", "fast", "accurate")),
    Option("simagent.timeout", 60.0, "Timeout", "Seconds to wait for an answer.", minimum=1, maximum=3600),
    Option("simagent.insert_cells", True, "Insert tasks as cells",
           "Send turns a task into a notebook cell instead of only listing it."),
], description="The agent behind the Sim agent side bar.",
    sections=[("Connection", ["simagent.server", "simagent.timeout"]),
              ("Tasks", ["simagent.model", "simagent.insert_cells"])], icon=ICON)


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

    def show_server(s):
        server.setText(f"Server: {s['simagent.server']}  ({s['simagent.model']})")

    def show_data(data):
        structure.setText(f"{data.count} atoms, frame {context.frame}")

    def submit():
        text = task.text().strip()
        if not text:
            return
        tasks.addItem(text)
        task.clear()
        if context.settings["simagent.insert_cells"]:
            context.insert_cell(f"# sim-agent task: {text}\n# (a real plugin would send it to "
                                f"{context.settings['simagent.server']})")
        context.message(f"queued {text!r}")

    send.clicked.connect(submit)
    task.returnPressed.connect(submit)
    settings.clicked.connect(lambda: context.open_preferences("Sim agent"))
    context.on("settings", show_server)
    context.on("data", show_data)
    show_server(context.settings)
    if context.data is not None:
        show_data(context.data)
    for w in (server, structure, QtWidgets.QLabel("Tasks"), tasks, task, send, settings):
        lay.addWidget(w)
    return box
