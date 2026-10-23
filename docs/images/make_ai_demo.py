"""Record docs/images/ai-demo.gif: Ask AI classifies the surface sites of a Pt nanoparticle.

    xvfb-run -a -s "-screen 0 1920x1200x24" python docs/images/make_ai_demo.py

The window, the notebook, the tools and every cell are real: the AI's tool calls insert, edit
and run notebook cells in the GUI as they would with a live model. Only the model is scripted
(``SCRIPT`` below, written for this recording), so it needs no API key and comes out the same
every time. Needs ffmpeg (or imageio-ffmpeg) for the GIF palette.

Like a screen recording, the picture zooms to where things happen: the chat while a question is
typed and an answer streams in, the notebook cell the AI writes or edits, and back out to the
whole window to show the result. The window is grabbed at full size; ``FOCUS`` names the region
each moment, and ``frame_crops`` eases between them.
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
WIDTH = 720
FPS = 6
ZOOM = {"type": 2.0, "chat": 2.0, "cell": 1.6, "full": 1.0}   # how far each region is zoomed in
EASE = 0.35             # seconds: how fast the picture follows a new region

# where the picture looks now: ("type",) the chat input, ("chat",) the newest chat message,
# ("cell", index; -1 = last) or ("full",). The scripted
# model sets it from its thread, the recording loop reads it at every grab.
FOCUS = [("full",)]

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
            FOCUS[0] = ("chat",)
            time.sleep(0.8)              # "thinking"
            for i, word in enumerate(text.split(" ")):
                if on_text:
                    on_text(("" if i == 0 else " ") + word)
                time.sleep(0.045)
            self.messages.append("assistant")
            if calls:                    # look at the cell the tool writes
                FOCUS[0] = ("cell", calls[0][1].get("index", -1))
                time.sleep(0.5)
            calls = [ToolCall(f"t{len(self.messages)}{k}", name, args) for k, (name, args) in enumerate(calls)]
            return Reply(text, calls, "tool_use" if calls else "end")

        def tool_results(self, results):
            for _call, result, *_ in results:
                if getattr(result, "is_error", False):
                    raise RuntimeError(f"scripted tool call failed: {result.text()}")
            time.sleep(2.2)              # the cell runs and shows its output
            self.messages.append("tool results")

    return Provider("scripted", "Scripted demo", models={"demo": "Demo"}, assistant="AI",
                    make_client=lambda *a, **k: None, chat_class=ScriptedChat, key_required=False)


def record():
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ewald import plugins, settings
    from ewald.gui import notebook_panel
    from ewald.gui.ask_panel import Message
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

    def region():
        """(center x, center y, zoom) of the window region ``FOCUS`` names, in window pixels."""
        what = FOCUS[0]
        zoom = ZOOM[what[0]]
        cw, ch = w.width() / zoom, w.height() / zoom
        if what[0] == "type":
            box = w.ask_panel.input
            top = box.mapTo(w, QtCore.QPoint(0, 0)).y()
            return cw / 2, top + box.height() + 12 - ch / 2, zoom
        if what[0] == "chat":        # the bottom of the newest message near the bottom of the picture
            shown = [m for m in w.ask_panel.findChildren(Message) if m.isVisible()]
            if shown:
                bottom = max(m.mapTo(w, QtCore.QPoint(0, m.height())).y() for m in shown)
                return cw / 2, bottom + 40 - ch / 2, zoom
            return cw / 2, ch / 2, zoom
        if what[0] == "cell":
            view = w.panel.view
            cells = view.notebook.cells
            cell = view.widget_for(cells[what[1]]) if cells else None
            dock = w.notebook_dock
            x = dock.mapTo(w, QtCore.QPoint(0, 0)).x() + dock.width() / 2
            if cell is None or not cell.isVisible():
                return x, w.height() / 2, zoom
            top = cell.mapTo(w, QtCore.QPoint(0, 0)).y()
            return x, min(top + cell.height() / 2, top + ch / 2 - 40), zoom
        return w.width() / 2, w.height() / 2, 1.0

    def settle(seconds=0.0, n=None):
        """Pump events for ``seconds``, grabbing a frame (and the region to show) every 1/FPS s."""
        end = time.monotonic() + seconds
        while True:
            app.processEvents()
            app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            if not shots or time.monotonic() - shots[-1][0] >= 1 / FPS:
                shot = os.path.join(frames_dir, f"{len(shots):05d}.png")
                w.grab().save(shot)
                shots.append((time.monotonic(), shot, region()))
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

    def ask(question):
        box = w.ask_panel.input
        box.setFocus()
        FOCUS[0] = ("type",)
        settle(0.8)
        for ch in question:
            box.insertPlainText(ch)
            settle(0.03)
        settle(0.6)
        w.ask_panel.ask()
        while w.ask_panel.busy():
            settle(0.1)
        FOCUS[0] = ("chat",)
        settle(3.0)                  # read the answer
        FOCUS[0] = ("full",)
        settle(3.0)                  # the whole window: structure, cells and chat

    settle(1.0)
    ask(QUESTIONS[0])
    ask(QUESTIONS[1])
    w.close()
    return shots, (w.width(), w.height())


def frame_crops(shots, size):
    """Crop every frame to its region, easing towards each new one, scaled to ``WIDTH``."""
    from PIL import Image

    W, H = size
    out_size = (WIDTH, round(WIDTH * H / W))
    cx, cy, zoom = W / 2, H / 2, 1.0
    last = shots[0][0]
    frames = []
    for t, path, (tx, ty, tz) in shots:
        a = 1 - pow(2.718281828, -(t - last) / EASE)
        last = t
        cx, cy, zoom = cx + a * (tx - cx), cy + a * (ty - cy), zoom + a * (tz - zoom)
        cw, ch = W / zoom, H / zoom
        x0 = min(max(cx - cw / 2, 0), W - cw)
        y0 = min(max(cy - ch / 2, 0), H - ch)
        image = Image.open(path).convert("RGB")
        scale = image.width / W          # device pixels per window pixel
        box = tuple(round(v * scale) for v in (x0, y0, x0 + cw, y0 + ch))
        cropped = path[:-4] + "-crop.png"
        image.crop(box).resize(out_size, Image.LANCZOS).save(cropped)
        frames.append((t, cropped))
    return frames


def to_gif(shots, out):
    """Frames -> GIF with one palette for the whole clip (ffmpeg palettegen/paletteuse); the
    frame times are the real ones."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    shots = frame_crops(*shots)
    work = os.path.dirname(shots[0][1])
    lst = os.path.join(work, "frames.txt")
    with open(lst, "w", encoding="utf-8", newline="\n") as f:
        for (t, shot), (t2, _) in zip(shots, shots[1:] + [(shots[-1][0] + 2.0, None)]):
            f.write(f"file '{shot}'\nduration {max(t2 - t, 0.02):.3f}\n")
        f.write(f"file '{shots[-1][1]}'\n")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-vf",
                    "split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];"
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
