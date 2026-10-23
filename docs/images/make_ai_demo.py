"""Record docs/images/ai-demo.gif: Ask AI analyses three runs, one scene each.

    xvfb-run -a -s "-screen 0 1920x1200x24" python docs/images/make_ai_demo.py [scene ...]

Scenes (``SCENES``): Li diffusion in a Li6PS5Cl AIMD run (VASP XDATCAR), sp/sp2/sp3 carbon in a
melt-quench MD run (extxyz), and the barrier of a vacancy hop in fcc Fe (VASP NEB folder). The
data are the published datasets of ``ewald.examples.fetch`` (downloaded once, cached).

The window, the notebook, the tools and every cell are real: the AI's tool calls insert, edit
and run notebook cells in the GUI as they would with a live model, and the numbers in its answers
are what those cells compute. Only the model is scripted (each scene's ``script``, written for
this recording), so it needs no API key and comes out the same every time. After the answer the
trajectory plays in the structure view.

Like a screen recording, the picture zooms in so the text reads, but calmly (few, slow moves):
while the AI works it shows the chat and the notebook side by side and only pans up and down to
the typed question, the answer or the cell being written; then it zooms out to the whole window
and over to the structure view while the run plays. Each scene runs in its own process and grabs
the full window; ``FOCUS`` names the region each moment, ``frame_crops`` eases between them, a
title card goes before each scene.
Needs Pillow; gifsicle, when installed, shrinks the GIF (frame differences, --lossy).
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ai-demo.gif")
SIZE = (1440, 860)      # window size; the GIF is scaled to WIDTH
WIDTH = 760
FPS = 6
ZOOM = {"type": 1.5, "chat": 1.5, "cell": 1.5, "view": 1.5, "full": 1.0}   # how far each region is zoomed in
EASE = 0.9              # seconds: how fast the picture follows a new region (slow: calm)
TITLE_SECONDS = 2.2
SPEED = 1.7             # the GIF plays this much faster than the recording (grabs slow the window)

HEADER = '''\
import numpy as np

from ewald import all_frames, each_frame, examples, load
from ewald.modifiers import *
from ewald.paint import fade, paint
from ewald.views import show, view

'''

# Each scene: the notebook it starts from, the question, and what the model answers, reply by
# reply: (text, [(tool, input), ...]); an empty tool list ends the answer. ``play``: the frames
# shown after the answer (start, stop, step).
SCENES = {
    "li-diffusion": dict(
        title="Li diffusion in Li6PS5Cl",
        subtitle="VASP AIMD (XDATCAR): 416 atoms, 140 frames",
        notebook="# %% Li6PS5Cl argyrodite, AIMD at 0.1 ps per frame\n" + HEADER
                 + 'pipe = load(examples.fetch("li6ps5cl-aimd") / "XDATCAR.gz")\n',
        question="Which Li ions hop in this run? Draw the paths of the most mobile ones colored by time, "
                 "and give me the Li diffusion coefficient.",
        script=[
            ("I'll measure how far every atom moves from the first frame.",
             [("add_modifier", {"name": "CalculateDisplacements", "params": {"reference_frame": 0}})]),
            ("Now I'll select the six Li that get farthest during the run and fade the rest.",
             [("insert_cell", {"source": '''\
@pipe.add
@all_frames
def mobile_li(frames, data):
    li = np.flatnonzero(data["Particle Type"] == "Li")
    travel = np.max([d["Displacement Magnitude"][li] for d in frames], axis=0)
    selected = np.zeros(data.count, int)
    selected[li[np.argsort(travel)[-6:]]] = 1
    data["Selection"] = selected
    fade(data, "Selection == 0", 0.85)'''})]),
            ("Their paths, colored by time and drawn up to the current frame:",
             [("add_modifier", {"name": "GenerateTrajectoryLines", "params": {}}),
              ("set_view", {"trajectory_color_by": "Time", "trajectory_colormap": "plasma",
                            "trajectory_trail": True, "trajectory_width": 4, "bonds": False})]),
            ("And the mean squared displacement per element, fitted from 3 ps on:",
             [("add_modifier", {"name": "MeanSquaredDisplacement",
                                "params": {"time_per_frame": 0.1, "fit_range": [0.214, 1.0]}})]),
            ("The six fastest Li travel 7.6-10.3 Å from where they start, twice the median Li "
             "(3.9 Å), hopping between neighboring cages. **D\\*(Li) = 0.125 Å²/ps = 1.25 × 10⁻⁵ cm²/s**, "
             "in line with kinisi's 1.23(10) × 10⁻⁵ on the same run; P, S and Cl stay on their sites "
             "(D\\* below 0.0005 Å²/ps).", []),
        ],
        play=(0, 140, 5),
    ),
    "carbon-quench": dict(
        title="sp, sp² and sp³ carbon",
        subtitle="melt-quench MD (extxyz): 200 C atoms, 211 frames",
        notebook="# %% amorphous carbon at 2.0 g/cm3, quenched and annealed at 3000 K\n" + HEADER
                 + 'pipe = load(examples.fetch("carbon-synth") / "density-2.0-T-3000.extxyz")\n',
        question="Color the atoms by hybridization and plot how the sp, sp2 and sp3 fractions change "
                 "during the run.",
        script=[
            ("Hybridization follows the number of C neighbors within 1.85 Å: 2 is sp, 3 is sp², 4 is sp³.",
             [("add_modifier", {"name": "CoordinationAnalysis", "params": {"cutoff": 1.85}})]),
            ("Painting them in every frame:",
             [("insert_cell", {"source": '''\
@pipe.add
@each_frame
def hybridization(frame, data):
    paint(data, "Coordination == 2", "gold")       # sp
    paint(data, "Coordination == 3", "seagreen")   # sp2
    paint(data, "Coordination == 4", "royalblue")  # sp3'''})]),
            ("And the fractions over the whole run, as a plot:",
             [("insert_cell", {"source": '''\
@pipe.add
@all_frames
def fractions(frames, data):
    cn = [d["Coordination"] for d in frames]
    data.tables["hybridization"] = {
        "Time": np.array([d.attributes["time"] for d in frames]),
        "sp": np.array([np.mean(c == 2) for c in cn]),
        "sp2": np.array([np.mean(c == 3) for c in cn]),
        "sp3": np.array([np.mean(c == 4) for c in cn]),
    }'''}),
              ("insert_cell", {"source": 'show(pipe, table="hybridization")'})]),
            ("The run starts as a hot gas: 36 % sp, 22 % sp², 12 % sp³ and 30 % with fewer or more "
             "neighbors. By frame 50 it is **80 % sp²**, and it ends as an almost pure sp² network "
             "(**99 %**) at this density.", []),
        ],
        play=(0, 211, 7),
    ),
    "fe-neb": dict(
        title="A vacancy hop in fcc Fe",
        subtitle="VASP climbing-image NEB folder: 5 images",
        notebook="# %% fcc Fe with one vacancy, VTST NEB folder 00-04\n" + HEADER
                 + 'pipe = load(examples.fetch("fe-vacancy-neb"))\n',
        question="What is the barrier of this NEB, and which atom moves?",
        script=[
            ("The folder loads as a band with its energy profile. I'll plot it.",
             [("insert_cell", {"source": 'show(pipe, table="NEB spline", y="Energy")'})]),
            ("Now the atom that moves most between the end images:",
             [("insert_cell", {"source": '''\
band = load(examples.fetch("fe-vacancy-neb"))
band.modifiers.clear()                     # the images alone, no overlay copies
first, last = band.compute(0), band.compute(band.num_frames - 1)
step = first.cell.minimum_image(last.positions - first.positions)
hop = int(np.argmax(np.linalg.norm(step, axis=1)))
print(f"atom {hop} moves {np.linalg.norm(step[hop]):.2f} Å")'''})]),
            ("And its path through the band, seen along [110]:",
             [("add_modifier", {"name": "GenerateTrajectoryLines", "params": {}}),
              ("set_view", {"frame": 2, "direction": [1, 1, 0], "bonds": False, "trajectory_trail": True,
                            "trajectory_width": 4})]),
            ("The barrier is **325 meV**, with the saddle at image 2, halfway along a 3.00 Å path; the "
             "reverse barrier is 326 meV, so both ends are equivalent sites. **Atom 7** jumps 2.45 Å into "
             "the vacancy, one nearest-neighbor distance (2.51 Å).", []),
        ],
        play=(0, 5, 1),
    ),
}

# where the picture looks now: ("type",) the chat input, ("chat",) the newest chat message,
# ("cell", index; -1 = last), ("view",) the structure view or ("full",). The scripted model sets it from its thread, the
# recording loop reads it at every grab.
FOCUS = [("full",)]


def scripted_provider(script):
    """A provider whose model plays ``script``: text streams in word by word, tool calls go to the
    real tools."""
    from ewald.llm.chat import Chat, Reply, ToolCall
    from ewald.llm.providers import Provider

    replies = list(script)

    class ScriptedChat(Chat):
        def user(self, text, images=None):
            self.messages.append("user")

        def send(self, tool_choice=None, on_text=None):
            text, calls = replies.pop(0)
            FOCUS[0] = ("chat",)
            time.sleep(0.7)              # "thinking"
            for i, word in enumerate(text.split(" ")):
                if on_text:
                    on_text(("" if i == 0 else " ") + word)
                time.sleep(0.04)
            self.messages.append("assistant")
            if calls:                    # look at the cell the tool writes
                FOCUS[0] = ("cell", calls[0][1].get("index", -1))
                time.sleep(0.4)
            calls = [ToolCall(f"t{len(self.messages)}{k}", name, args) for k, (name, args) in enumerate(calls)]
            return Reply(text, calls, "tool_use" if calls else "end")

        def tool_results(self, results):
            for _call, result, *_ in results:
                if getattr(result, "is_error", False):
                    raise RuntimeError(f"scripted tool call failed: {result.text()}")
            time.sleep(1.5)              # the cell runs and shows its output
            self.messages.append("tool results")

    return Provider("scripted", "Scripted demo", models={"demo": "Demo"}, assistant="AI",
                    make_client=lambda *a, **k: None, chat_class=ScriptedChat, key_required=False)


def record(name, frames_dir):
    """Record scene ``name`` into ``frames_dir``: PNGs and shots.json [(time, png, region)]."""
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ewald import plugins, settings
    from ewald.gui import notebook_panel
    from ewald.gui.app import MainWindow
    from ewald.gui.ask_panel import Message

    scene = SCENES[name]
    notebook_panel.CONFIRM_UNSAVED = False
    plugins.register_ai_provider("scripted", scripted_provider(scene["script"]))
    s = settings.Settings()
    s["llm.provider"] = "scripted"
    s["llm.run_cells"] = True          # Preferences > AI > Let the AI run cells
    path = os.path.join(os.getcwd(), f"{name.replace('-', '_')}.py")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(scene["notebook"])

    w = MainWindow(settings=s)
    w.resize(*SIZE)
    w.show()
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
        if what[0] == "cell":        # same columns as the chat (the picture only moves up and down)
            view = w.panel.view
            cells = view.notebook.cells
            cell = view.widget_for(cells[what[1]]) if cells else None
            x = cw / 2
            if cell is None or not cell.isVisible():
                return x, w.height() / 2, zoom
            top = cell.mapTo(w, QtCore.QPoint(0, 0)).y()
            return x, min(top + cell.height() / 2, top + ch / 2 - 40), zoom
        if what[0] == "view":
            v = w.viewport
            c = v.mapTo(w, QtCore.QPoint(v.width() // 2, v.height() // 2))
            return w.width() - cw / 2, c.y(), zoom
        return w.width() / 2, w.height() / 2, 1.0

    def settle(seconds=0.0):
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
    settle(0.8)

    box = w.ask_panel.input
    box.setFocus()
    FOCUS[0] = ("type",)
    settle(0.8)
    for ch in scene["question"]:
        box.insertPlainText(ch)
        settle(0.025)
    settle(0.5)
    w.ask_panel.ask()
    while w.ask_panel.busy():
        settle(0.1)
    FOCUS[0] = ("chat",)
    settle(3.5)                        # read the answer
    FOCUS[0] = ("full",)
    settle(1.2)
    FOCUS[0] = ("view",)               # the run plays in the structure view
    settle(0.6)
    start, stop, step = scene["play"]
    short = stop - start < 10          # a NEB band: there and back, slowly
    for frame in [*range(start, stop, step), *(range(stop - 2, start - 1, -1) if short else ())]:
        w.set_frame(frame)
        settle(0.6 if short else 1 / FPS)
    settle(1.5)
    w.close()
    with open(os.path.join(frames_dir, "shots.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"size": [w.width(), w.height()], "shots": shots}, f)


def title_card(number, scene, size, path):
    """A card naming the scene, the size of the cropped frames."""
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", size, "#16181d")
    draw = ImageDraw.Draw(image)
    bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    regular = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    h = size[1]
    draw.text((48, h * 0.30), f"Ask AI  ·  {number} / {len(SCENES)}", font=ImageFont.truetype(regular, 20),
              fill="#5ca8ff")
    draw.text((48, h * 0.40), scene["title"], font=ImageFont.truetype(bold, 38), fill="#f2f4f8")
    draw.text((48, h * 0.55), scene["subtitle"], font=ImageFont.truetype(regular, 22), fill="#aab2c0")
    image.save(path)


def frame_crops(shots, size, work):
    """Crop every frame to its region, easing towards each new one, scaled to ``WIDTH``."""
    from PIL import Image

    W, H = size
    out_size = (WIDTH, round(WIDTH * H / W))
    cx, cy, zoom = W / 2, H / 2, 1.0
    last = shots[0][0]
    frames = []
    for t, path, (tx, ty, tz) in shots:
        a = 1 - math.exp(-(t - last) / EASE)
        last = t
        cx, cy, zoom = cx + a * (tx - cx), cy + a * (ty - cy), zoom + a * (tz - zoom)
        cw, ch = W / zoom, H / zoom
        x0 = min(max(cx - cw / 2, 0), W - cw)
        y0 = min(max(cy - ch / 2, 0), H - ch)
        image = Image.open(path).convert("RGB")
        scale = image.width / W          # device pixels per window pixel
        box = tuple(round(v * scale) for v in (x0, y0, x0 + cw, y0 + ch))
        cropped = os.path.join(work, os.path.basename(path)[:-4] + "-crop.png")
        image.crop(box).resize(out_size, Image.LANCZOS).save(cropped)
        frames.append((t, cropped))
    return frames, out_size


def to_gif(names, dirs, out):
    """Title card + cropped frames of every scene -> one GIF: one palette for the whole clip, the
    recorded frame times sped up by ``SPEED``; then gifsicle -O3 --lossy if installed."""
    from PIL import Image

    entries = []                         # (image, milliseconds)
    for k, (name, folder) in enumerate(zip(names, dirs), 1):
        with open(os.path.join(folder, "shots.json"), encoding="utf-8") as f:
            rec = json.load(f)
        frames, out_size = frame_crops(rec["shots"], rec["size"], folder)
        card = os.path.join(folder, "title.png")
        title_card(k, SCENES[name], out_size, card)
        entries.append((card, TITLE_SECONDS * 1000))
        for (t, png), (t2, _) in zip(frames, frames[1:] + [(frames[-1][0] + 1 / FPS, None)]):
            entries.append((png, max(t2 - t, 0.02) * 1000 / SPEED))
    images = [Image.open(png).convert("RGB") for png, _ in entries]
    sample = images[:: max(1, len(images) // 40)]
    w, h = images[0].size
    montage = Image.new("RGB", (w, h * len(sample)))
    for k, image in enumerate(sample):
        montage.paste(image, (0, h * k))
    palette = montage.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    frames = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    frames[0].save(out, save_all=True, append_images=frames[1:], loop=0, disposal=1,
                   duration=[round(ms) for _, ms in entries])
    if shutil.which("gifsicle"):
        subprocess.run(["gifsicle", "-O3", "--lossy=60", "-b", out], check=True)
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB, {sum(ms for _, ms in entries) / 1000:.0f} s")


if __name__ == "__main__":
    if os.environ.get("_EWALD_AI_DEMO"):
        record(os.environ["_EWALD_AI_DEMO"], os.environ["_EWALD_AI_DEMO_DIR"])
        sys.exit(0)
    names = sys.argv[1:] or list(SCENES)
    dirs = []
    for name in names:
        folder = tempfile.mkdtemp()
        env = dict(os.environ, _EWALD_AI_DEMO=name, _EWALD_AI_DEMO_DIR=folder, EWALD_CONFIG_DIR=tempfile.mkdtemp(),
                   MPLBACKEND="Agg", EWALD_PLUGIN_INDEX="")
        subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, cwd=tempfile.mkdtemp(), check=True)
        dirs.append(folder)
    to_gif(names, dirs, OUT)
