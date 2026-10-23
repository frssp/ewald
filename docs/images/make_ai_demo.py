"""Record docs/images/ai-demo.gif: Ask AI classifies the surface sites of a Pt nanoparticle.

    xvfb-run -a -s "-screen 0 1920x1200x24" python docs/images/make_ai_demo.py

The window, the notebook, the tools and every cell are real: the AI's tool calls insert, edit
and run notebook cells in the GUI as they would with a live model. Only the model is scripted
(``SCRIPT`` below: what a model answered when asked the same questions), so the recording needs
no API key and comes out the same every time. The viewport zoom in/out is done on the camera
between the turns, as a user would with the mouse wheel. Needs ffmpeg (or imageio-ffmpeg) for
the GIF palette.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ai-demo.gif")
SIZE = (1440, 860)      # window size; the GIF is scaled to WIDTH
WIDTH = 960
FPS = 8

NOTEBOOK = '''\
# %% a Pt nanoparticle (201 atoms)
from ewald import each_frame, examples, load
from ewald.modifiers import *
from ewald.paint import hide, paint
from ewald.views import show

pipe = load(examples.path("Pt_nanoparticle.xyz"))
'''

SITES = '''\
@pipe.add
@each_frame
def sites(frame, data):
    paint(data, "Coordination == 6", "crimson")  # vertex
    paint(data, "Coordination == 7", "orange")   # edge
    paint(data, "Coordination == 8", "teal")     # (100)
    paint(data, "Coordination == 9", "seagreen") # (111)'''

QUESTIONS = [
    "Find the vertex, edge and facet atoms of this nanoparticle and color them.",
    "Cut it open so I can see the core, and add a coordination histogram.",
]

# What the model answers, turn by turn: (text, [(tool, input), ...]). An empty tool list ends the turn.
SCRIPT = [
    [("I'll count the Pt neighbors of every atom first: the coordination number tells the site type.",
      [("add_modifier", {"name": "CoordinationAnalysis", "params": {"cutoff": 3.3}})]),
     ("Coordination is 6, 7, 8, 9 or 12. That is a truncated octahedron: vertices have 6 neighbors, "
      "edges 7, (100) facets 8, (111) facets 9 and the core 12. I'll paint each kind in a step cell.",
      [("insert_cell", {"source": SITES})]),
     ("Done. Of the **201** Pt atoms:\n\n"
      "- **24** vertices (CN 6, crimson)\n- **36** edge atoms (CN 7, orange)\n"
      "- **6** (100) facet atoms (CN 8, teal)\n- **56** (111) facet atoms (CN 9, sea green)\n"
      "- **79** core atoms (CN 12, silver)\n\n"
      "Cell 2 counts the neighbors and cell 3 paints the sites; change a color there and run it again.",
      [])],
    [("I'll hide every atom in front of the middle plane (x > 0 Å) in the painting cell.",
      [("edit_cell", {"index": 2, "old_text": "# (111)",
                      "new_text": "# (111)\n    hide(data, \"Position.X > 0\")      # front half"})]),
     ("And a histogram of the coordination numbers:",
      [("insert_cell", {"source": 'show(pipe, histogram="Coordination")'})]),
     ("The cut shows the silver **CN 12** core inside a shell of surface sites. Cell 3 now hides the "
      "front half and cell 4 is the histogram: five bars, at 6, 7, 8, 9 and 12.", [])],
]


def scripted_provider():
    """A provider whose model plays ``SCRIPT``: text streams in word by word, tool calls go to the
    real tools."""
    from ewald.llm.chat import Chat, Reply, ToolCall
    from ewald.llm.providers import Provider

    turns = [list(t) for t in SCRIPT]

    class ScriptedChat(Chat):
        def user(self, text, images=None):
            self.turn = turns.pop(0)     # a new question
            self.messages.append("user")

        def send(self, tool_choice=None, on_text=None):
            text, calls = self.turn.pop(0)
            time.sleep(0.8)              # "thinking"
            for i, word in enumerate(text.split(" ")):
                if on_text:
                    on_text(("" if i == 0 else " ") + word)
                time.sleep(0.045)
            self.messages.append("assistant")
            calls = [ToolCall(f"t{len(self.messages)}{k}", name, args) for k, (name, args) in enumerate(calls)]
            return Reply(text, calls, "tool_use" if calls else "end")

        def tool_results(self, results):
            for _call, result, *_ in results:
                if getattr(result, "is_error", False):
                    raise RuntimeError(f"scripted tool call failed: {result.text()}")
            time.sleep(0.4)
            self.messages.append("tool results")

    return Provider("scripted", "Scripted demo", models={"demo": "Demo"}, assistant="AI",
                    make_client=lambda *a, **k: None, chat_class=ScriptedChat, key_required=False)


def record():
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ewald import plugins, settings
    from ewald.gui import notebook_panel
    from ewald.gui.app import MainWindow

    notebook_panel.CONFIRM_UNSAVED = False
    plugins.register_ai_provider("scripted", scripted_provider())
    s = settings.Settings()
    s["llm.provider"] = "scripted"
    s["llm.run_cells"] = True          # Preferences > AI > Let the AI run cells
    path = os.path.join(os.getcwd(), "pt_nanoparticle.py")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(NOTEBOOK)

    w = MainWindow(settings=s)
    w.resize(*SIZE)
    w.show()
    frames_dir = tempfile.mkdtemp()
    shots = []

    def settle(seconds=0.0, n=None):
        """Pump events for ``seconds``, grabbing a frame every 1/FPS s."""
        end = time.monotonic() + seconds
        while True:
            app.processEvents()
            app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            if not shots or time.monotonic() - shots[-1][0] >= 1 / FPS:
                shot = os.path.join(frames_dir, f"{len(shots):05d}.png")
                w.grab().save(shot)
                shots.append((time.monotonic(), shot))
            if time.monotonic() >= end:
                return
            QtCore.QThread.msleep(10)

    def quiet(n=40):
        for _ in range(n):
            app.processEvents()
            QtCore.QThread.msleep(30)

    quiet()
    w.open_notebook(path, run=True)
    w.ask_claude()                     # the AI chat side bar beside the notebook
    w.resizeDocks([w.ask_dock, w.notebook_dock], [300, 500], QtCore.Qt.Horizontal)
    quiet(80)
    w.statusBar().clearMessage()
    shots.clear()
    settle(1.5)

    cam = lambda: w.viewport.plotter  # noqa: E731

    def zoom(factor, seconds, turn=0.0):
        steps = max(1, int(seconds * FPS))
        for _ in range(steps):
            cam().camera.zoom(factor ** (1 / steps))
            if turn:
                cam().camera.azimuth = cam().camera.azimuth + turn / steps
            cam().render()
            settle(1 / FPS)

    def ask(question):
        box = w.ask_panel.input
        box.setFocus()
        for ch in question:
            box.insertPlainText(ch)
            settle(0.03)
        settle(0.6)
        w.ask_panel.ask()
        while w.ask_panel.busy():
            settle(0.1)
        settle(2.0)

    ask(QUESTIONS[0])
    zoom(2.2, 2.5, turn=25)      # zoom in on the vertices and edges
    settle(1.5)
    zoom(1 / 2.2, 2.0, turn=-25)
    settle(1.0)
    ask(QUESTIONS[1])
    zoom(1.8, 2.0, turn=-20)     # into the cut
    settle(1.5)
    zoom(1 / 1.8, 1.5, turn=20)
    settle(2.5)
    w.close()
    return shots


def to_gif(shots, out):
    """Frames -> GIF with one palette for the whole clip (ffmpeg palettegen/paletteuse); the
    frame times are the real ones."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    work = os.path.dirname(shots[0][1])
    lst = os.path.join(work, "frames.txt")
    with open(lst, "w", encoding="utf-8", newline="\n") as f:
        for (t, shot), (t2, _) in zip(shots, shots[1:] + [(shots[-1][0] + 2.0, None)]):
            f.write(f"file '{shot}'\nduration {max(t2 - t, 0.02):.3f}\n")
        f.write(f"file '{shots[-1][1]}'\n")
    scale = f"scale={WIDTH}:-1:flags=lanczos"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-vf",
                    f"{scale},split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=none:diff_mode=rectangle", out], check=True)
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    if os.environ.get("_EWALD_AI_DEMO"):
        to_gif(record(), OUT)
        sys.exit(0)
    work = tempfile.mkdtemp()
    env = dict(os.environ, _EWALD_AI_DEMO="1", EWALD_CONFIG_DIR=tempfile.mkdtemp(), MPLBACKEND="Agg",
               EWALD_PLUGIN_INDEX="")
    subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, cwd=work, check=True)
