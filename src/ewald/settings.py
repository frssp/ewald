"""User preferences shared by the GUI and scripts (design: /design/settings-menu.md).

Stored as one JSON file holding only the values that differ from the defaults::

    from ewald import settings
    s = settings.load()
    s["viewport.background"] = "white"
    s.save()

Headless ``render()`` / ``Pipeline.compute()`` never read these implicitly; the
GUI applies them, scripts read them explicitly.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from . import fileops

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Option:
    key: str
    default: object
    label: str
    help: str = ""
    choices: tuple = ()          # allowed values (combo box)
    minimum: float | None = None
    maximum: float | None = None
    kind: str = ""               # editor hint: color, file, dir, table, ui_theme (installed UI themes),
                                 # ai_provider, ai_model (model ids the chosen provider suggests),
                                 # ai_profiles (saved AI connections)

    @property
    def page(self) -> str:
        return PAGE_OF_KEY.get(self.key) or PAGE_OF_PREFIX[self.key.split(".")[0]]


# python.plot_style choices -> (ewald.plotstyle style, colors from the UI theme?)
PLOT_STYLES = ("Publication", "Publication in UI theme colors", "Matplotlib default")


def plot_style(choice: str, tokens: dict | None = None) -> tuple:
    """``(style, tokens)`` for :meth:`ewald.notebook.Notebook.set_plot_style` from a
    ``python.plot_style`` value and the active UI theme's ``tokens``."""
    if choice == "Matplotlib default":
        return "default", None
    return "publication", (dict(tokens) if tokens and "theme" in choice else None)


# Preferences pages in order (2026-10-23 menu review); plugin settings pages follow "Plugins"
PAGES = ("General", "Codes", "3D view", "Atoms & bonds", "Rendering", "Plots", "Notebooks", "Compute", "Python", "AI",
         "Plugins")
PAGE_OF_PREFIX = {"general": "General", "viewport": "3D view", "elements": "Atoms & bonds", "bonds": "Atoms & bonds",
                  "render": "Rendering", "plots": "Plots", "compute": "Compute", "performance": "Compute",
                  "python": "Python", "llm": "AI", "codes": "Codes"}
PAGE_OF_KEY = {"python.plot_style": "Plots", "python.run_notebooks": "Notebooks", "python.save_figures": "Notebooks",
               "python.bundle_data_mb": "Notebooks", "python.output_scroll_after": "Notebooks",
               "python.output_box_lines": "Notebooks", "python.follow_running": "Notebooks", "python.plugin_catalogs": "Plugins"}
PAGE_ALIASES = {"Viewports": "3D view", "Elements": "Atoms & bonds"}   # names before 2026-10-23

OPTIONS = (
    Option("general.ui_theme", "graphite", "UI theme",
           "Graphite: dark. Paper: light. Midnight, Solarized light/dark and High contrast are built in too; "
           "UI theme plugins (@ui_theme) add more.", kind="ui_theme"),
    Option("general.startup_workspace", "Notebook", "Startup workspace", choices=("Notebook", "Render")),
    Option("general.toolbar_style", "Icons and text", "Toolbar buttons",
           "Icons and text: text on mode switches and rarely used buttons, icons alone on the tool rail.",
           choices=("Icons and text", "Icons", "Text")),
    Option("general.activity_hidden", "", "Hidden activity bar buttons",
           "Side bar names, comma separated (right-click the activity bar to change); "
           "hidden side bars stay in the Window menu."),
    Option("general.font_size", 12, "UI font size", "Pixels; applies to every panel.", minimum=9, maximum=20),
    Option("general.restore_layout", True, "Restore window layout", "Reopen docks and window size as you left them."),
    Option("general.live_reload", True, "Live reload",
           "Follow trajectories a running simulation is still writing (LAMMPS dump, XDATCAR, OUTCAR, "
           "extxyz; local or SSH): new frames extend the timeline."),
    Option("general.live_reload_interval", 2.0, "Live reload interval", "Seconds between looks at the file.",
           minimum=0.5, maximum=600),
    Option("general.docs_url", "", "Documentation URL",
           "Where Help opens the manual. Empty = the docs folder on GitHub; or a hosted or built site."),

    Option("viewport.follow_theme", True, "Background follows the UI theme",
           "Use the UI theme's viewport color (dark in Graphite, light in Paper). Off: the Background color."),
    Option("viewport.background", "#ECEDE9", "Background", "Used when the background does not follow the UI theme.",
           kind="color"),
    Option("viewport.projection", "ortho", "Projection", choices=("ortho", "persp")),
    Option("viewport.initial_view", "iso", "Initial view direction", choices=("iso", "x", "y", "z", "front")),
    Option("viewport.style_preset", "Ball & stick", "Default style",
           "Style of a newly opened structure. Simple (ASE) draws flat outlined discs on white, like ASE's GUI.",
           choices=("Ball & stick", "Space-filling", "Simple (ASE)", "Points")),
    Option("viewport.rotation_step", 15.0, "Step rotation", "Degrees per click of the rotate buttons (VESTA).",
           minimum=0.5, maximum=90),
    Option("viewport.show_axes", True, "Show axes tripod"),
    Option("viewport.render_frame", False, "Render frame in the 3D view",
           "Also outside the Render page: the part of the view the picture shows, outside it dimmed."),
    Option("viewport.show_overlay", True, "Show timestep and time",
           "Beside the frame number under the 3D view, for trajectories whose file has them."),
    Option("viewport.backend", "auto", "Viewport renderer",
           "VTK: the full viewport (OpenGL 3.2+). Legacy GL: OpenGL 1.x display lists like VESTA, which "
           "work over SSH X11 forwarding (ssh -X) but have no impostors, smoothing or 3D text. Auto uses "
           "Legacy GL on an ssh -X display. EWALD_VIEWPORT overrides this. Takes effect at the next start.",
           choices=("auto", "vtk", "legacy-gl")),
    Option("viewport.gpu", "auto", "GPU for the 3D view",
           "On a laptop with two GPUs (integrated Intel/AMD plus NVIDIA or AMD): High performance draws on the "
           "discrete GPU. On Windows it sets the Graphics settings choice for this Python (Settings › Display › "
           "Graphics), on Linux PRIME render offload. Auto leaves the choice to the system. EWALD_GPU overrides "
           "this. Takes effect at the next start. CUDA compute always uses the NVIDIA GPU.",
           choices=("auto", "high-performance", "power-saving")),
    Option("viewport.hover_info", True, "Atom info on hover",
           "Resting the cursor on an atom shows its element, index and Cartesian and fractional "
           "coordinates in a tooltip and the status bar. Nothing is computed while the view moves."),
    Option("viewport.hover_delay", 0.25, "Hover delay", "Seconds the cursor rests before the atom info appears.",
           minimum=0.05, maximum=3),
    Option("viewport.sphere_drawing", "impostor", "Sphere drawing",
           "Impostor: GPU spheres that are round at any zoom and fast for any number of atoms. "
           "Mesh: triangle spheres (Sphere resolution) up to the impostor threshold. "
           "A style's own Shape (mesh/impostor) wins over this.", choices=("impostor", "mesh")),
    Option("viewport.sphere_resolution", 16, "Sphere resolution",
           "Mesh sphere detail when Sphere drawing is Mesh and in the OpenGL 1.x viewport (its Simple (ASE) "
           "discs use at least 36); a view's sphere_resolution (View › Atoms › Sphere detail) wins; exports use "
           "full quality.", minimum=4, maximum=40),
    Option("viewport.mesh_limit", 150_000, "Impostor threshold",
           "Above this many atoms spheres are drawn as GPU impostors (and arrows as lines).",
           minimum=1000, maximum=100_000_000),
    Option("viewport.anti_aliasing", True, "Smooth edges", "FXAA on the viewport: removes jagged sphere and "
           "bond outlines at almost no cost."),
    Option("viewport.arrow_resolution", 6, "Arrow resolution",
           "Sides per vector arrow while interacting; exports use full quality.", minimum=3, maximum=32),
    Option("viewport.interactive_detail", "auto", "Detail while moving",
           "While rotating, zooming, scrubbing or playing, draw a coarse scene (impostors; above "
           "'Atoms while moving' every k-th atom and no bonds) and refine when idle; playback keeps "
           "one level until it stops. Auto does this only when a full frame misses the target frame "
           "rate. Exports always use full quality.",
           choices=("auto", "coarse", "full")),
    Option("viewport.target_fps", 30, "Target frame rate", "Auto detail switches to the coarse scene "
           "when a full-detail frame takes longer than 1/fps.", minimum=1, maximum=240),
    Option("viewport.interactive_atoms", 1_000_000, "Atoms while moving",
           "The coarse scene draws every k-th atom so at most this many remain.",
           minimum=1000, maximum=100_000_000),
    Option("viewport.max_bond_sticks", 8_000_000, "Bond cylinders at most",
           "Above this many bond cylinders the 3D view leaves the bonds out and says so, instead of "
           "running out of GPU memory (about 0.6 KB each; a bond whose two halves share a color is "
           "one cylinder). Pictures and the ray tracer still draw them.", minimum=0, maximum=1_000_000_000),
    Option("viewport.refine_delay", 0.3, "Refine after", "Idle seconds before full detail returns.",
           minimum=0.0, maximum=10.0),

    Option("elements.color_scheme", "Jmol", "Color scheme", choices=("Jmol", "CPK")),
    Option("elements.radius_source", "covalent", "Radius preset",
           "Covalent (Cordero 2008), atomic (Slater 1964; calc.: Clementi 1967), vdW (Alvarez 2013; Bondi 1964) "
           "or Shannon (1976) ionic/crystal radii by oxidation state and CN.",
           choices=("covalent", "atomic", "atomic (calc.)", "vdW", "vdW (Bondi)", "ionic", "ionic (crystal)")),
    Option("elements.radius_scale", 0.8, "Radius scale", "Multiplies the tabulated radius.", minimum=0.05, maximum=3),
    Option("elements.overrides", {}, "Per-element overrides", "Color and radius per element; wins over the scheme.",
           kind="table"),
    Option("bonds.length_factor", 1.15, "Bond length factor", "Max bond = factor × (r_A + r_B), like VESTA.",
           minimum=0.5, maximum=2),
    Option("bonds.radius", 0.15, "Bond radius", "Stick radius in ball-and-stick views.", minimum=0.01, maximum=1),
    Option("bonds.auto_limit", 250_000, "Automatic bonds up to",
           "Opening a structure with more atoms turns bonds off (search and sticks get slow); "
           "turn them on under Bonds.", minimum=0, maximum=100_000_000),

    Option("render.renderer", "raytrace", "Default renderer", "Vector writes SVG/PDF/EPS with editable text.",
           choices=("raytrace", "opengl", "vector")),
    Option("render.quality", "Publication", "Image quality",
           "Draft = half size, no AO/shadows; Publication = 16 rays/pixel, 8 AO rays per hit, shadows. "
           "Custom uses the values below.", choices=("Draft", "Standard", "Publication", "Custom")),
    Option("render.movie_quality", "Same as image", "Movie quality", "Lower it to export long movies faster.",
           choices=("Same as image", "Draft", "Standard", "Publication")),
    Option("render.samples", 16, "Anti-aliasing rays/pixel", "Used when quality is Custom.", minimum=1, maximum=256),
    Option("render.ao_samples", 32, "Ambient occlusion rays", "Used when quality is Custom.", minimum=0, maximum=512),
    Option("render.noise_threshold", 0.01, "Noise threshold",
           "Adaptive sampling: pixels below this error stop early (0 = off). Used when quality is Custom.",
           minimum=0.0, maximum=0.1),
    Option("render.shadows", True, "Shadows", "Used when quality is Custom."),
    Option("render.size_preset", "1-column (3.5 in, 4:3)", "Size preset",
           choices=("1-column (3.5 in, 4:3)", "2-column (7.2 in, 16:9)", "Square (3.5 in)", "Slide 16:9 (1920×1080)")),
    Option("render.dpi", 600, "Resolution", "Print resolution used by the size presets.", minimum=72, maximum=2400),
    Option("render.background", "white", "Background", choices=("white", "transparent", "black", "viewport")),
    Option("render.output_dir", "", "Output folder", "Empty = current folder.", kind="dir"),
    Option("render.type_legend", False, "Particle type legend", "One color swatch per type in the corner."),
    Option("render.scale_bar", False, "Scale bar", "Round length in Å or nm; exact in orthographic views."),
    Option("plots.size", "1-column (3.5 in)", "Saved plot size",
           "Journal column width of plots saved from the Plots tab and view cells (PDF, SVG, PNG); "
           "PNG uses the DPI of the image quality.",
           choices=("1-column (3.5 in)", "1.5-column (5.0 in)", "2-column (7.2 in)")),
    Option("render.movie_format", "MP4 (H.264)", "Movie format", "AVI uses Motion JPEG for PowerPoint.",
           choices=("MP4 (H.264)", "GIF", "AVI (Motion JPEG)", "MOV (H.264)", "PNG sequence")),

    Option("compute.device", "cpu", "Device", "Where analysis and ray-traced pictures run: the window, the "
           "notebook's pipe and the Render page. CPU is the safe default; OpenCL ray traces on Intel, AMD and "
           "NVIDIA GPUs and Metal on a Mac's GPU, both keeping the analysis on the CPU; Auto uses the CUDA GPU "
           "when CuPy finds one, else Metal on a Mac or an OpenCL GPU for ray tracing, else the CPU.",
           choices=("cpu", "cuda", "opencl", "metal", "auto")),
    Option("performance.cache_memory", 2048, "Frame cache memory",
           "Upper bound for cached trajectory frames (MB); the current frame is always kept.",
           minimum=64, maximum=1_048_576),
    Option("performance.background_compute", True, "Compute in the background",
           "Evaluate the pipeline on a worker thread so the window stays responsive; a newer frame or "
           "parameter change cancels the running one."),
    Option("compute.threads", 0, "CPU threads", "Numba threads for CPU kernels; 0 = all cores.",
           minimum=0, maximum=1024),

    Option("python.run_notebooks", "never", "Run notebooks when opened",
           "'never': opening a notebook runs nothing; it shows the cells and the results saved in the "
           "file, and Run all (or the bar over the cells) runs it. 'ask': asks every time. 'always': a "
           "notebook you saved or trusted before (the same file, unchanged) runs at once, as in Jupyter. "
           "A file you have not trusted is Python code you have not seen: it never runs without asking "
           "and otherwise opens restricted (an .ewd's plugins load only once you trust it).",
           choices=("never", "ask", "always")),
    Option("python.plot_style", "Publication", "Style of your own plots",
           "matplotlib figures that notebook cells and the console draw (plt.plot ...): the publication "
           "style of saved plots, in print colors or the UI theme's, or matplotlib's defaults; the style "
           "file below goes on top. rcParams a cell sets always win.", choices=PLOT_STYLES),
    Option("python.output_scroll_after", 50, "Scroll outputs longer than (lines)",
           "A cell's printed text, value or traceback longer than this scrolls in a box, as in "
           "Jupyter. Click the rule left of an output (or press Shift+O on the cell) to scroll it or "
           "show every line, double-click it (or O) to hide it. 0 scrolls nothing by itself.",
           minimum=0, maximum=100000),
    Option("python.output_box_lines", 20, "Scrolled output height (lines)",
           "How many lines the box of a scrolled output shows.", minimum=3, maximum=10000),
    Option("python.follow_running", "follow", "Follow the running cell",
           "While cells run (Run all, Run below, or cells the AI runs outside an answer), follow: scroll "
           "to a cell that starts off screen, until you scroll yourself. never: leave the scroll alone; "
           "the chip beside Interrupt still says which cell runs and goes there when clicked.",
           choices=("follow", "never")),
    Option("python.save_figures", True, "Save figures with notebooks",
           "An .ewd notebook keeps every view cell as SVG and PNG, with the plotted data as CSV."),
    Option("python.bundle_data_mb", 50.0, "Copy data files up to (MB)",
           "An .ewd notebook carries data files up to this size; larger ones stay links (path, size, "
           "fingerprint). 0 keeps links only.", minimum=0.0, maximum=100000.0),
    Option("python.plugin_catalogs", "", "Plugin catalogs",
           "URLs or paths of plugin index files (JSON) listed in the Plugins panel, separated by ';', "
           "after the default index (EWALD_PLUGIN_INDEX overrides it; empty turns it off)."),
    Option("python.startup_script", "", "Console startup script", "Run in the Python console when the GUI starts.",
           kind="file"),
    Option("python.extra_paths", "", "Extra module paths", "Folders added to sys.path, separated by ';'.", kind="dir"),

    # the API key itself is not an option: it lives in the OS keychain (ewald.llm.credentials)
    Option("llm.provider", "anthropic", "Provider",
           "The AI service behind Ask AI, the AI chat and ewald ask: Anthropic (Claude), OpenAI, Google "
           "Gemini, an OpenAI-compatible server (Ollama, LM Studio, vLLM, OpenRouter) "
           "or a plugin's (an API gateway, for example).",
           kind="ai_provider"),
    Option("llm.model", "", "Model", "Empty: the provider's default. Any model id the service offers works.",
           kind="ai_model"),
    Option("llm.profiles", {}, "Saved connections",
           "Named sets of provider, model, address and headers; pick one to fill in the fields below. "
           "ewald ask --profile NAME uses one from the command line. API keys stay in the keychain.",
           kind="ai_profiles"),
    Option("llm.base_url", "", "Server URL",
           "Empty: the provider's own address. For an OpenAI-compatible server, its address, "
           "e.g. http://localhost:11434/v1 (Ollama) or http://localhost:1234/v1 (LM Studio)."),
    Option("llm.route_prefix", "", "Model prefix",
           "Put in front of the model id in each request, e.g. openai/ for a vendor-qualified route. "
           "Empty: the provider's default, usually none."),
    Option("llm.key_header", "", "API key header",
           "Send the API key in this HTTP header instead of Authorization: Bearer, e.g. api-key (Azure OpenAI) "
           "or a gateway's ticket header. Empty: Authorization: Bearer."),
    Option("llm.headers", "", "Extra headers",
           "HTTP headers sent with every request, as Name: value pairs separated by ';' (proxies, gateways); "
           "{uuid} in a value becomes a new id in each request. Put secrets in the API key field, not here."),
    Option("llm.timeout", 0.0, "Timeout",
           "Seconds one request may take. 0: the provider's default.",
           minimum=0.0, maximum=3600.0),
    Option("llm.effort", "high", "Effort",
           "How hard the model thinks before answering (Claude, OpenAI reasoning models); lower is faster and cheaper.",
           choices=("low", "medium", "high", "xhigh", "max")),
    Option("llm.stream", True, "Stream answers",
           "The AI chat shows each answer while it arrives, and Stop ends it at once. Off: the answer "
           "appears when it is complete (for a server or gateway that cannot stream; ewald also answers "
           "in one piece by itself when an OpenAI-compatible server refuses streaming)."),
    Option("llm.code_cells", "insert", "Code in answers",
           "insert: each Python block the AI chat writes becomes a new notebook cell right away (not run; "
           "Ctrl+Z removes it). button: the block gets a + Insert as cell button instead.",
           choices=("insert", "button")),
    Option("llm.run_cells", False, "Let the AI run cells",
           "The AI chat can read, insert and edit notebook cells; with this on it can also run them (any "
           "Python, like Shift+Enter). Off: cells it writes wait until you run them."),
    Option("llm.follow_cells", "follow", "Follow the AI's cells",
           "How the notebook scrolls while the AI chat writes and runs cells (reading them never scrolls). "
           "follow: to a cell it writes when that cell is off screen, until you scroll yourself during the "
           "answer. end: once, to the last cell it wrote, when the answer is done. never: not at all. "
           "When the notebook does not go there, a pill at its edge shows the cell and jumps to it.",
           choices=("follow", "end", "never")),
    Option("llm.read_folders", "", "Folders the AI may read",
           "Files the AI chat opens itself (load_structure, workflows, plugin tools) must lie in one of these "
           "folders, separated by ';'. Empty: any folder. Cells you run are not limited.", kind="dir"),
    Option("llm.confirm_hosts", True, "Ask before the AI connects to a host",
           "Before the AI chat opens a file on an SSH host for the first time in a chat, ask whether it may."),
)
OPTION = {o.key: o for o in OPTIONS}


def option(key: str, load_plugins: bool = True) -> Option | None:
    """The :class:`Option` of ``key``: ewald's own (:data:`OPTIONS`), else a simulation code's
    ``codes.<name>.*`` row (:mod:`ewald.codes`; built-in codes need no plugin loading, plugin
    codes load the plugins unless ``load_plugins`` is False). None when unknown."""
    opt = OPTION.get(key)
    if opt is not None or not key.startswith("codes."):
        return opt
    from . import codes

    opt = codes.builtin_option(key)
    if opt is None and load_plugins:
        name = key.split(".")[1] if key.count(".") >= 2 else ""
        code = codes.codes().get(name)
        opt = next((o for o in code.options if o.key == key), None) if code else None
    return opt


def all_options() -> list[Option]:
    """:data:`OPTIONS` plus every registered simulation code's options (Preferences › Codes)."""
    from . import codes

    return [*OPTIONS, *(o for c in codes.codes().values() for o in c.options)]


# ---------------------------------------------------------------- devices / optional components
def _probe_cuda():
    if importlib.util.find_spec("cupy") is None:
        return False, "CuPy not installed: pip install 'ewald[gpu]'"
    from .backend import gpu_available

    if not gpu_available():
        return False, "CuPy installed but no CUDA device found"
    try:
        import cupy

        name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
        return True, name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return True, "CUDA device"


def _probe_opencl():
    if importlib.util.find_spec("pyopencl") is None:
        return False, "pyopencl not installed: pip install 'ewald[opencl]'"
    from .raytrace_gpu import opencl_devices

    devices = opencl_devices()
    if not devices:
        return False, "pyopencl installed but no OpenCL device found (install the GPU driver)"
    return True, f"ray tracing on {devices[0].name.strip()}; analysis on the CPU"


def _probe_metal():
    if sys.platform != "darwin":
        return False, "macOS only"
    if importlib.util.find_spec("Metal") is None:
        return False, "PyObjC's Metal bindings not installed: pip install 'ewald[metal]'"
    from .raytrace_gpu import metal_device

    dev = metal_device()
    if dev is None:
        return False, "no Metal GPU found"
    return True, f"ray tracing on {dev.name()}; analysis on the CPU"


# key, label, probe() -> (available, detail). New backends (e.g. Apple Metal) add one row here.
DEVICES = (
    ("cpu", "CPU (NumPy / Numba)", lambda: (True, f"{os.cpu_count()} cores")),
    ("cuda", "NVIDIA CUDA (CuPy)", _probe_cuda),
    ("opencl", "OpenCL GPU (ray tracing)", _probe_opencl),
    ("metal", "Apple GPU (Metal, ray tracing)", _probe_metal),
    ("auto", "Auto (GPU if available)", lambda: (True, "falls back to CPU")),
)

# module, purpose, install command
COMPONENTS = (
    ("ewald._native", "PTM, Voronoi, CSP matching, interval CNA (C++ extension)",
     "install a C++ compiler, then reinstall ewald (docs/user/install.md)"),
    ("cupy", "GPU compute (CUDA)", "pip install 'ewald[gpu]'"),
    ("pyopencl", "GPU ray tracing on any GPU (OpenCL)", "pip install 'ewald[opencl]'"),
    ("ovito", "Dislocation analysis (DXA)", "pip install 'ewald[gui,ovito]'"),
    ("imageio_ffmpeg", "Movie export (.mp4)", "pip install imageio-ffmpeg (part of the ewald install)"),
    ("qtconsole", "IPython console in the GUI", "pip install 'ewald[gui]'"),
)


def component_status():
    """[(module, purpose, installed, version, install command)] without importing heavy packages."""
    from importlib import metadata

    dist = {"cupy": None, "imageio_ffmpeg": "imageio-ffmpeg"}
    out = []
    for mod, purpose, cmd in COMPONENTS:
        found = importlib.util.find_spec(mod) is not None
        version = ""
        if found and not mod.startswith("ewald."):
            names = [dist.get(mod) or mod] if mod != "cupy" else ["cupy-cuda12x", "cupy-cuda11x", "cupy"]
            for n in names:
                try:
                    version = metadata.version(n)
                    break
                except metadata.PackageNotFoundError:
                    pass
        out.append((mod, purpose, found, version, cmd))
    return out


# ---------------------------------------------------------------- storage
def read_text_any(path) -> str:
    """A text file EWALD wrote: UTF-8, or the locale's encoding for files older versions wrote
    on Windows (cp949, cp1252, ...) before every file was UTF-8."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        import locale

        return raw.decode(locale.getpreferredencoding(False), errors="replace")


def config_dir() -> Path:
    if os.environ.get("EWALD_CONFIG_DIR"):
        return Path(os.environ["EWALD_CONFIG_DIR"])
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "ewald"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ewald"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "ewald"


def default_path() -> Path:
    return config_dir() / "settings.json"


def _valid(opt: Option, value):
    """Return the value coerced to the option's type, or raise ValueError."""
    d = opt.default
    if isinstance(d, bool):
        if not isinstance(value, bool):
            raise ValueError("expected true/false")
    elif isinstance(d, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("expected a number")
        value = type(d)(value)
        if (opt.minimum is not None and value < opt.minimum) or (opt.maximum is not None and value > opt.maximum):
            raise ValueError(f"out of range [{opt.minimum}, {opt.maximum}]")
    elif isinstance(d, dict):
        if not isinstance(value, dict):
            raise ValueError("expected a mapping")
        value = {str(k): dict(v) for k, v in value.items()}
    else:
        value = str(value)
    if opt.choices and value not in opt.choices:
        raise ValueError(f"expected one of {opt.choices}")
    return value


# Options that no longer exist, dropped quietly from older settings files: the OVITO-like
# "classic" window layout was removed, so the notebook layout is the only one.
RETIRED = {"general.layout", "llm.user_id", "llm.system_name",   # these two: gateway headers, now llm.headers
           "plots.font_size",   # now font.size of the plot style file (ewald.plotstyle)
           "general.lammps_type_map"}   # a default LAMMPS type map fits almost no run (2026-10-25)
# Keys renamed since: old -> new (LAMMPS settings moved to the Codes page, 2026-10-25)
RENAMED_KEYS = {"general.lammps_type_prompt": "codes.lammps.type_prompt"}
# Values renamed since: key -> {old: new}
RENAMED_VALUES = {"general.startup_workspace": {"Analyze": "Notebook"}}


class Settings:
    """Mapping of option key -> value with defaults; ``save()`` writes only changed keys."""

    def __init__(self, values: dict | None = None, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else default_path()
        self._values: dict = {}
        self._later: dict = {}   # codes.<plugin code>.* values, checked once that code's options are known
        for k, v in (values or {}).items():
            if k in RETIRED:
                continue
            k = RENAMED_KEYS.get(k, k)
            v = RENAMED_VALUES.get(k, {}).get(v, v) if isinstance(v, str) else v
            if k.startswith("codes.") and option(k, load_plugins=False) is None:
                self._later[k] = v   # a plugin's code: kept (and saved) until its plugin is loaded
                continue
            try:
                self[k] = v
            except (KeyError, ValueError) as exc:
                log.warning("ignoring setting %s=%r: %s", k, v, exc)

    def _option(self, key) -> Option:
        opt = option(key)
        if opt is None:
            raise KeyError(f"unknown setting {key!r}")
        if key in self._later:
            v = self._later.pop(key)
            try:
                self[key] = v
            except ValueError as exc:
                log.warning("ignoring setting %s=%r: %s", key, v, exc)
        return opt

    def __getitem__(self, key):
        opt = self._option(key)
        v = self._values.get(key, opt.default)
        return dict(v) if isinstance(v, dict) else v

    def __setitem__(self, key, value):
        opt = option(key)
        if opt is None:
            raise KeyError(f"unknown setting {key!r}")
        self._later.pop(key, None)
        value = _valid(opt, value)
        if value == opt.default:
            self._values.pop(key, None)
        else:
            self._values[key] = value

    def get(self, key, default=None):
        return self[key] if option(key) is not None else default

    def is_default(self, key) -> bool:
        return key not in self._values and key not in self._later

    def reset(self, keys=None):
        for k in list(self._values) + list(self._later) if keys is None else keys:
            self._values.pop(k, None)
            self._later.pop(k, None)

    def changed(self) -> dict:
        return {**self._later, **{k: self[k] for k in self._values}}

    def copy(self) -> "Settings":
        return Settings(self.changed(), self.path)

    def update(self, other: "Settings | dict"):
        values = other.changed() if isinstance(other, Settings) else other
        self._values.clear()
        self._later.clear()
        for k, v in values.items():
            if k.startswith("codes.") and option(k, load_plugins=False) is None:
                self._later[k] = v
            else:
                self[k] = v

    def save(self, path=None):
        p = Path(path) if path else self.path
        p.parent.mkdir(parents=True, exist_ok=True)
        fileops.write_text(p, json.dumps(self.changed(), indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        return p

    def __repr__(self):
        return f"<Settings {self.path} changed={sorted(self._values)}>"

    # ---------------------------------------------------------- derived values
    def element_color(self, symbol: str):
        """RGB (0..1) for an element under the color scheme and overrides, or None for non-elements."""
        o = self["elements.overrides"].get(symbol, {})
        if "color" in o:
            return o["color"]
        from ase.data import atomic_numbers

        z = atomic_numbers.get(symbol)
        if not z:
            return None
        if self["elements.color_scheme"] == "CPK":
            from ase.data.colors import cpk_colors as table
        else:
            from ase.data.colors import jmol_colors as table
        return tuple(float(c) for c in table[z])

    def element_radius(self, symbol: str, preset: str | None = None):
        """Scaled radius of an element under ``preset`` (default: the Radius preset) and the overrides.

        Ionic presets use the element's most common oxidation state at CN VI here; per-type values
        for a structure come from :meth:`type_radii`.
        """
        o = self["elements.overrides"].get(symbol, {})
        if "radius" in o:
            return float(o["radius"])
        from .radii import element_radius

        r = element_radius(symbol, preset or self["elements.radius_source"])
        return None if r is None else float(r * self["elements.radius_scale"])

    def type_radii(self, data, preset: str | None = None, oxidation_states: dict | None = None,
                   coordination: dict | None = None):
        """``(radii, table)``: scaled per-type radii of ``data`` with overrides applied, and the
        :class:`ewald.radii.RadiusTable` they came from (oxidation states, CN, fallbacks)."""
        from .radii import element_of, resolve

        table = resolve(data, preset or self["elements.radius_source"], oxidation_states, coordination)
        overrides = self["elements.overrides"]
        out = {}
        for t, r in table.radii.items():
            o = overrides.get(t) or overrides.get(element_of(t)) or {}
            out[t] = float(o["radius"]) if "radius" in o else float(r * self["elements.radius_scale"])
        return out, table


def load(path=None) -> Settings:
    p = Path(path) if path else default_path()
    values = {}
    if p.exists():
        try:
            values = json.loads(read_text_any(p))
        except (OSError, ValueError) as exc:
            log.warning("could not read %s: %s; using defaults", p, exc)
    return Settings(values if isinstance(values, dict) else {}, p)
