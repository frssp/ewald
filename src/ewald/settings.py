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
        return PAGE_OF_PREFIX[self.key.split(".")[0]]


# python.plot_style choices -> (ewald.plotstyle style, colors from the UI theme?)
PLOT_STYLES = ("Publication", "Publication in UI theme colors", "Matplotlib default")


def plot_style(choice: str, tokens: dict | None = None) -> tuple:
    """``(style, tokens)`` for :meth:`ewald.notebook.Notebook.set_plot_style` from a
    ``python.plot_style`` value and the active UI theme's ``tokens``."""
    if choice == "Matplotlib default":
        return "default", None
    return "publication", (dict(tokens) if tokens and "theme" in choice else None)


PAGES = ("General", "Viewports", "Elements", "Rendering", "Compute", "Python", "AI")
PAGE_OF_PREFIX = {"general": "General", "viewport": "Viewports", "elements": "Elements", "bonds": "Elements",
                  "render": "Rendering", "plots": "Rendering", "compute": "Compute", "performance": "Compute",
                  "python": "Python", "llm": "AI"}

OPTIONS = (
    Option("general.ui_theme", "graphite", "UI theme",
           "Graphite: dark. Paper: light. Midnight, Solarized light/dark and High contrast are built in too; "
           "UI theme plugins (@ui_theme) add more.", kind="ui_theme"),
    Option("general.startup_workspace", "Notebook", "Startup workspace", choices=("Notebook", "Render")),
    Option("general.toolbar_style", "Icons and text", "Toolbar buttons",
           "Icons and text: text on mode switches and rarely used buttons, icons alone on the tool rail.",
           choices=("Icons and text", "Icons", "Text")),
    Option("general.font_size", 12, "UI font size", "Pixels; applies to every panel.", minimum=9, maximum=20),
    Option("general.restore_layout", True, "Restore window layout", "Reopen docks and window size as you left them."),
    Option("general.lammps_type_prompt", "ask", "LAMMPS type names",
           "Ask for a type → element map when opening a LAMMPS dump.", choices=("ask", "never")),
    Option("general.lammps_type_map", "", "Default LAMMPS type map", "e.g. 1=Cu,2=Zr; pre-fills the prompt."),
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
    Option("viewport.show_overlay", True, "Show frame / timestep text"),
    Option("viewport.backend", "auto", "Viewport renderer",
           "VTK: the full viewport (OpenGL 3.2+). Legacy GL: OpenGL 1.x display lists like VESTA, which "
           "work over SSH X11 forwarding (ssh -X) but have no impostors, smoothing or 3D text. Auto uses "
           "Legacy GL on an ssh -X display. EWALD_VIEWPORT overrides this. Takes effect at the next start.",
           choices=("auto", "vtk", "legacy-gl")),
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
           "Mesh sphere detail when Sphere drawing is Mesh; exports use full quality.", minimum=4, maximum=40),
    Option("viewport.mesh_limit", 150_000, "Impostor threshold",
           "Above this many atoms spheres are drawn as GPU impostors (and arrows as lines).",
           minimum=1000, maximum=100_000_000),
    Option("viewport.anti_aliasing", True, "Smooth edges", "FXAA on the viewport: removes jagged sphere and "
           "bond outlines at almost no cost."),
    Option("viewport.arrow_resolution", 6, "Arrow resolution",
           "Sides per vector arrow while interacting; exports use full quality.", minimum=3, maximum=32),
    Option("viewport.interactive_detail", "auto", "Detail while moving",
           "While rotating, zooming, scrubbing or playing, draw a coarse scene (impostors, no bonds, "
           "at most 'Atoms while moving') and refine when idle. Auto does this only when a full "
           "frame misses the target frame rate. Exports always use full quality.",
           choices=("auto", "coarse", "full")),
    Option("viewport.target_fps", 30, "Target frame rate", "Auto detail switches to the coarse scene "
           "when a full-detail frame takes longer than 1/fps.", minimum=1, maximum=240),
    Option("viewport.interactive_atoms", 1_000_000, "Atoms while moving",
           "The coarse scene draws every k-th atom so at most this many remain.",
           minimum=1000, maximum=100_000_000),
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
    Option("bonds.auto_limit", 100_000, "Automatic bonds up to",
           "Opening a structure with more atoms turns bonds off (search and sticks get slow); "
           "turn them on under Bonds.", minimum=0, maximum=100_000_000),

    Option("render.renderer", "raytrace", "Default renderer", "Vector writes SVG/PDF/EPS with editable text.",
           choices=("raytrace", "opengl", "vector")),
    Option("render.quality", "Publication", "Image quality",
           "Draft = half size, no AO/shadows; Publication = 16 rays/pixel, 8 AO rays per hit, shadows. "
           "Custom uses the three values below.", choices=("Draft", "Standard", "Publication", "Custom")),
    Option("render.movie_quality", "Same as image", "Movie quality", "Lower it to export long movies faster.",
           choices=("Same as image", "Draft", "Standard", "Publication")),
    Option("render.samples", 16, "Anti-aliasing rays/pixel", "Used when quality is Custom.", minimum=1, maximum=256),
    Option("render.ao_samples", 32, "Ambient occlusion rays", "Used when quality is Custom.", minimum=0, maximum=512),
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
    Option("plots.font_size", 8.0, "Saved plot font size", "Points; 7-9 pt is what journals ask for.",
           minimum=5, maximum=24),
    Option("render.movie_format", "MP4 (H.264)", "Movie format", "AVI uses Motion JPEG for PowerPoint.",
           choices=("MP4 (H.264)", "GIF", "AVI (Motion JPEG)", "MOV (H.264)", "PNG sequence")),

    Option("compute.device", "cpu", "Device", "Where analysis runs: the window and the notebook's pipe. "
           "CPU is the safe default; Auto uses the CUDA GPU when CuPy finds one, else the CPU.",
           choices=("cpu", "cuda", "auto")),
    Option("performance.cache_memory", 2048, "Frame cache memory",
           "Upper bound for cached trajectory frames (MB); the current frame is always kept.",
           minimum=64, maximum=1_048_576),
    Option("performance.background_compute", True, "Compute in the background",
           "Evaluate the pipeline on a worker thread so the window stays responsive; a newer frame or "
           "parameter change cancels the running one."),
    Option("compute.threads", 0, "CPU threads", "Numba threads for CPU kernels; 0 = all cores.",
           minimum=0, maximum=1024),

    Option("python.run_notebooks", "ask", "Run notebooks when opened",
           "As in Jupyter, a notebook you saved or trusted before (the same file, unchanged) runs when "
           "opened; any other file is Python code you have not seen, so it asks first (an .ewd's "
           "plugins load only if you agree). 'never' runs nothing when opening and opens unknown files "
           "restricted. 'always' is the same as 'ask': a file you have not trusted always asks.",
           choices=("ask", "always", "never")),
    Option("python.plot_style", "Publication", "Style of your own plots",
           "matplotlib figures that notebook cells and the console draw (plt.plot ...): the publication "
           "style of saved plots, in print colors or the UI theme's, or matplotlib's defaults. rcParams a "
           "cell sets always win.", choices=PLOT_STYLES),
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
           "Gemini, an OpenAI-compatible server (Ollama, LM Studio, vLLM, OpenRouter), an API gateway "
           "with a credential ticket, or a plugin's.",
           kind="ai_provider"),
    Option("llm.model", "", "Model", "Empty: the provider's default. Any model id the service offers works.",
           kind="ai_model"),
    Option("llm.profiles", {}, "Saved connections",
           "Named sets of provider, model, address and headers; pick one to fill in the fields below. "
           "ewald ask --profile NAME uses one from the command line. API keys stay in the keychain.",
           kind="ai_profiles"),
    Option("llm.base_url", "", "Server URL",
           "Empty: the provider's own address. For an OpenAI-compatible server, its address, "
           "e.g. http://localhost:11434/v1 (Ollama) or http://localhost:1234/v1 (LM Studio); "
           "for a gateway, $GATEWAY_BASE_URL when empty."),
    Option("llm.route_prefix", "", "Model prefix",
           "Put in front of the model id in each request, e.g. openai/ for a gpt-oss gateway route. "
           "Empty: $GATEWAY_ROUTE_PREFIX for a gateway, else none."),
    Option("llm.user_id", "", "User id",
           "Sent as the User-Id header to an API gateway. Empty: $GATEWAY_USER_ID, else your login name."),
    Option("llm.system_name", "", "System name",
           "Sent as the X-System-Name header to an API gateway. Empty: $GATEWAY_SYSTEM_NAME, else ewald."),
    Option("llm.headers", "", "Extra headers",
           "HTTP headers sent with every request, as Name: value pairs separated by ';' (proxies, gateways). "
           "Put secrets in the API key field, not here."),
    Option("llm.timeout", 0.0, "Timeout",
           "Seconds one request may take. 0: the provider's default (a gateway: $GATEWAY_TIMEOUT, else 120).",
           minimum=0.0, maximum=3600.0),
    Option("llm.effort", "high", "Effort",
           "How hard the model thinks before answering (Claude, OpenAI reasoning models); lower is faster and cheaper.",
           choices=("low", "medium", "high", "xhigh", "max")),
    Option("llm.code_cells", "insert", "Code in answers",
           "insert: each Python block the AI chat writes becomes a new notebook cell right away (not run; "
           "Ctrl+Z removes it). button: the block gets a + Insert as cell button instead.",
           choices=("insert", "button")),
    Option("llm.run_cells", False, "Let the AI run cells",
           "The AI chat can read, insert and edit notebook cells; with this on it can also run them (any "
           "Python, like Shift+Enter). Off: cells it writes wait until you run them."),
)
OPTION = {o.key: o for o in OPTIONS}


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


# key, label, probe() -> (available, detail). New backends (e.g. Apple Metal) add one row here.
DEVICES = (
    ("cpu", "CPU (NumPy / Numba)", lambda: (True, f"{os.cpu_count()} cores")),
    ("cuda", "NVIDIA CUDA (CuPy)", _probe_cuda),
    ("auto", "Auto (GPU if available)", lambda: (True, "falls back to CPU")),
)

# module, purpose, install command
COMPONENTS = (
    ("ewald._native", "PTM, Voronoi, CSP matching, interval CNA (C++ extension)",
     "install a C++ compiler, then reinstall ewald (docs/user/install.md)"),
    ("cupy", "GPU compute (CUDA)", "pip install 'ewald[gpu]'"),
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
RETIRED = {"general.layout"}
# Values renamed since: key -> {old: new}
RENAMED_VALUES = {"general.startup_workspace": {"Analyze": "Notebook"}}


class Settings:
    """Mapping of option key -> value with defaults; ``save()`` writes only changed keys."""

    def __init__(self, values: dict | None = None, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else default_path()
        self._values: dict = {}
        for k, v in (values or {}).items():
            if k in RETIRED:
                continue
            v = RENAMED_VALUES.get(k, {}).get(v, v) if isinstance(v, str) else v
            try:
                self[k] = v
            except (KeyError, ValueError) as exc:
                log.warning("ignoring setting %s=%r: %s", k, v, exc)

    def __getitem__(self, key):
        if key not in OPTION:
            raise KeyError(key)
        v = self._values.get(key, OPTION[key].default)
        return dict(v) if isinstance(v, dict) else v

    def __setitem__(self, key, value):
        if key not in OPTION:
            raise KeyError(f"unknown setting {key!r}")
        value = _valid(OPTION[key], value)
        if value == OPTION[key].default:
            self._values.pop(key, None)
        else:
            self._values[key] = value

    def get(self, key, default=None):
        return self[key] if key in OPTION else default

    def is_default(self, key) -> bool:
        return key not in self._values

    def reset(self, keys=None):
        for k in list(self._values) if keys is None else keys:
            self._values.pop(k, None)

    def changed(self) -> dict:
        return {k: self[k] for k in self._values}

    def copy(self) -> "Settings":
        return Settings(self.changed(), self.path)

    def update(self, other: "Settings | dict"):
        values = other.changed() if isinstance(other, Settings) else other
        self._values.clear()
        for k, v in values.items():
            self[k] = v

    def save(self, path=None):
        p = Path(path) if path else self.path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.changed(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                     encoding="utf-8", newline="\n")
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
