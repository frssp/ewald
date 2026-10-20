"""Feature inventory: docs/features.md, generated from the code.

``python -m ewald inventory -o docs/features.md`` rewrites the file from the registries
(modifiers, readers, writers, renderers, CLI parser, AI tools and providers, settings). The GUI section
needs a live main window, so it is regenerated only with ``--gui`` (PySide6 and a display) and
otherwise carried over from the existing file. ``tests/test_docs.py`` and ``tests/test_gui.py``
fail when the file is stale, like ``llms.txt``.
"""
from __future__ import annotations

import argparse
import inspect
import re
from pathlib import Path

GUI_START, GUI_END = "<!-- gui:start -->", "<!-- gui:end -->"

# modifier module -> section of the inventory (a new module shows up under its own name until added here)
CATEGORIES = {
    "analysis": "Analysis",
    "structure": "Structure identification",
    "local_order": "Structure identification",
    "voronoi": "Structure identification",
    "wigner_seitz": "Defects and clusters",
    "cluster": "Defects and clusters",
    "ovito_bridge": "OVITO bridge (needs the optional `ovito` package)",
    "trajectory": "Trajectory",
    "bonds": "Bonds and polyhedra",
    "volume": "Volumetric data",
    "coloring": "Coloring",
    "selection": "Selection",
    "transform": "Modification",
    "datasets": "Combining files",
    "script": "Python",
}

HEADER = """# Feature inventory

Everything ewald has on `main`, generated from the code by
`python -m ewald inventory -o docs/features.md --gui`. Do not edit it by hand: `tests/test_docs.py`
(and `tests/test_gui.py` for the GUI section) fail when it is out of date. Open work and ideas are
in the [TODO list](todo.md); how each feature works is in the rest of this manual.
"""


ABBREVIATIONS = ("al.", "e.g.", "i.e.", "etc.", "vs.", "cf.", "...")


def first_sentence(doc: str | None) -> str:
    """First sentence of a docstring's first paragraph (not cut at "et al." or "e.g.")."""
    text = " ".join((inspect.cleandoc(doc or "").split("\n\n") or [""])[0].split())
    for m in re.finditer(r"\.(?=\s|$)", text):
        head = text[:m.end()]
        if not head.endswith(ABBREVIATIONS):
            text = head
            break
    return text.replace("|", "\\|")


def _table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def modifiers_section() -> list[str]:
    from . import modifiers as mods

    groups: dict[str, list] = {}
    for name in mods.__all__:
        cls = getattr(mods, name)
        if name in ("Modifier", "PythonModifier"):
            continue
        module = cls.__module__.rsplit(".", 1)[-1]
        groups.setdefault(CATEGORIES.get(module, module), []).append(cls)
    n = sum(len(v) for v in groups.values())
    out = ["## Modifiers", "",
           f"{n} built-in modifiers in `ewald.modifiers` (OVITO spellings such as "
           "`CommonNeighborAnalysisModifier` also import). Details: [Modifiers](user/modifiers.md).", ""]
    order = list(dict.fromkeys(CATEGORIES.values()))
    for cat in sorted(groups, key=lambda c: (order.index(c) if c in order else len(order), c)):
        out += [f"### {cat}", ""]
        out += _table(("Modifier", "What it does"), [(f"`{c.__name__}`", first_sentence(c.__doc__)) for c in groups[cat]])
        out.append("")
    return out


def io_section() -> list[str]:
    from . import plugins

    readers = [r for r in plugins.readers().values() if r.origin == "built-in"]
    readers.sort(key=lambda r: (r.priority is None, r.priority or 0, r.name))
    out = ["## File formats", "", "### Readers", "",
           "Tried in this order when no `format=` is given; reader plugins slot in at priority 50. "
           "Details: [File formats](user/formats.md).", ""]
    out += _table(("Format", "File names", "Reads", "`format=` aliases"),
                  [(f"`{r.name}`", " ".join(f"`{p}`" for p in r.patterns)
                    or ("(by content)" if r.sniff is not None else "(by `format=` only)"),
                    r.description.replace("|", "\\|"), ", ".join(f"`{a}`" for a in r.aliases))
                   for r in readers])
    writers = [w for w in plugins.writers().values() if w.origin == "built-in"]
    writers.sort(key=lambda w: (w.priority is None, w.priority or 0, w.name))
    out += ["", "Remote paths (`host:/path`, `ssh://...`) work with every reader; LAMMPS dumps and XDATCARs "
            "are streamed, other formats are downloaded to a local cache.", "", "### Writers", "",
            "`export_file(source, path, format=...)` and File › Export data pick a writer by `format=` or by "
            "file name; *one frame* writers need e.g. `frames=[-1]`.", ""]
    out += _table(("Format", "File names", "Writes", "`format=` aliases"),
                  [(f"`{w.name}`" + ("" if w.multi_frame else " *one frame*"),
                    " ".join(f"`{p}`" for p in w.patterns) or "(by `format=` only)",
                    w.description.replace("|", "\\|"), ", ".join(f"`{a}`" for a in w.aliases))
                   for w in writers])
    return out + [""]


def rendering_section() -> list[str]:
    from .movie import IMAGE_SUFFIXES, MOVIE_SUFFIXES
    from .render import QUALITY_PRESETS, RENDERERS, VECTOR_SUFFIXES, render

    out = ["## Rendering and output", "", "Details: [Rendering figures](user/rendering.md).", "", "### Renderers", ""]
    out += _table(("`renderer=`", "What it is"), [(f"`{k}`", v) for k, v in RENDERERS.items()])
    out += ["", "### Visual elements", "",
            "What gets drawn; every renderer draws every element. Plugins add more "
            "([Visual element plugins](plugins/visuals.md)).", ""]
    from .plugins import _builtin_visuals

    out += _table(("Element", "What it draws"), [(f"`{name}`", p.description.replace("|", "\\|"))
                                                for name, p in _builtin_visuals().items()])
    from .plugins import _builtin_overlays, _builtin_themes

    out += ["", "### Overlays", "",
            "`text`, `legend` (color bar), `type_legend` and `scale_bar` are `render()` options; overlay "
            "elements are added with `overlays=[...]` or the Render page's \"+ Overlay\". Plugins add more "
            "([Overlay plugins](plugins/overlays.md)).", ""]
    out += _table(("Overlay", "What it draws"), [(f"`{name}`", p.description.replace("|", "\\|"))
                                                for name, p in _builtin_overlays().items()])
    out += ["", "### Themes and color maps", "",
            "`theme=` (View › Theme in the GUI) sets element colors, background, cell color and radius "
            "preset; every `cmap` takes a matplotlib name or a color map plugin "
            "([Color maps and themes](plugins/styles.md)).", ""]
    out += _table(("Theme", "What it sets"), [(f"`{name}`", p.description.replace("|", "\\|"))
                                              for name, p in _builtin_themes().items()])
    out += ["", "### Output formats", "",
            "- Images: " + " ".join(f"`{s}`" for s in IMAGE_SUFFIXES) + "; vector pages: "
            + " ".join(f"`{s}`" for s in VECTOR_SUFFIXES) + " (true vectors with `renderer=\"vector\"`, "
            "an embedded image otherwise)",
            "- Movies (`render_animation`, `ewald movie`): " + " ".join(f"`{s}`" for s in MOVIE_SUFFIXES)
            + " (an image suffix writes a numbered sequence; video needs ffmpeg)",
            "- Sessions: `.ewald` (source, modifiers, style, camera, render settings)", "",
            "### Quality presets", ""]
    keys = list(next(iter(QUALITY_PRESETS.values())))
    out += _table(["`quality=`"] + [f"`{k}`" for k in keys],
                  [[f"`{name}`"] + [p[k] for k in keys] for name, p in QUALITY_PRESETS.items()])
    params = [p for p in inspect.signature(render).parameters.values() if p.kind is p.KEYWORD_ONLY]
    out += ["", "### `render()` options", "",
            ", ".join(f"`{p.name}`" for p in params) + ".", ""]
    from .render import build_scene

    taken = {p.name for p in params}
    scene = [p for p in inspect.signature(build_scene).parameters.values()
             if p.kind is p.KEYWORD_ONLY and p.name not in taken and p.name not in ("only_visible", "frame")]
    out += ["Scene keywords `render()` also takes: " + ", ".join(f"`{p.name}`" for p in scene) + ".", ""]
    return out


def cli_section() -> list[str]:
    from .__main__ import build_parser

    out = ["## Command line", "", "Details: [Command line](user/cli.md).", ""]
    sub = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    helps = {a.dest: a.help for a in sub._choices_actions}
    rows = []
    for name, p in sub.choices.items():
        pos = " ".join(a.dest.upper() if a.nargs is None else f"[{a.dest.upper()}...]" if a.nargs in ("*", "+")
                       else f"[{a.dest.upper()}]" for a in p._actions if not a.option_strings)
        flags = " ".join(f"`{a.option_strings[-1]}`" for a in p._actions
                         if a.option_strings and a.dest != "help")
        rows.append((f"`ewald {name}{' ' + pos if pos else ''}`", (helps.get(name) or "").replace("|", "\\|"), flags))
    return out + _table(("Command", "What it does", "Options"), rows) + [""]


def claude_section() -> list[str]:
    from .llm.providers import BUILTIN
    from .llm.notebook_tools import CELL_TOOLS
    from .llm.tools import CELL_WRITERS, Toolset

    ts = Toolset(None, allow_code=True)

    out = ["## AI integration", "",
           "The same tools serve Ask AI and the AI chat, `ewald ask`, the Python `ewald.llm.Agent` and `ewald mcp` "
           "(MCP server), whichever provider answers. They work on a notebook (the AI chat's current tab, or "
           "a session notebook of their own) and change the structure and every other result only through "
           "its cells: tools marked *cell* write a cell and run it (a name and JSON values, no model-written "
           "code), *notebook* tools read and edit cells, and `run_cell` (*code*) runs a cell the AI wrote, "
           "only with `allow_code=True` or Preferences > AI > Let the AI run cells. The others only read. "
           "Details: [AI integration](user/claude.md).", "",
           "Providers (Preferences > AI, `--provider`; `@ai_provider` plugins add more):", ""]
    out += _table(("Provider", "Name", "Default model", "Key"),
                  [(p.label, f"`{p.name}`", f"`{p.default_model}`",
                    (f"`{p.env_var}` or keychain" if p.env_var else "optional") if p.key_name else "none")
                   for p in BUILTIN.values()])
    out += ["", "Tools:", ""]
    rows = [(f"`{name}`" + (" *cell*" if name in CELL_WRITERS else " *notebook*" if name in CELL_TOOLS else "")
             + (" *code*" if name == "run_cell" else ""), first_sentence(t.description))
            for name, t in ts.tools.items()]
    out += _table(("Tool", "What it does"), rows)
    out += ["", "Also: `ewald describe` / `ewald narrate` (rule-based descriptors and a trajectory timeline, "
            "AI-written when a client is available), Modifier Forge (`ewald forge`) and the `llms.txt` API "
            "reference (`ewald api`).", ""]
    return out


def plugins_section() -> list[str]:
    from . import plugins

    examples = sorted((Path(__file__).resolve().parents[2] / "examples" / "plugins").glob("*.py"))
    out = ["## Plugins", "",
           f"Entry point groups `{plugins.READER_GROUP}`, `{plugins.WRITER_GROUP}`, `{plugins.MODIFIER_GROUP}`, "
           f"`{plugins.VISUAL_GROUP}`, `{plugins.OVERLAY_GROUP}`, `{plugins.COLORMAP_GROUP}`, "
           f"`{plugins.THEME_GROUP}`, `{plugins.TOOL_GROUP}`, `{plugins.UI_THEME_GROUP}`, "
           f"`{plugins.AI_PROVIDER_GROUP}`, `{plugins.PANEL_GROUP}`, `{plugins.SETTINGS_PAGE_GROUP}` and "
           f"`{plugins.WORKFLOW_GROUP}`, or a file in "
           "`<config dir>/plugins/` with `@reader` / `@writer` / `@modifier` / `@visual` / `@overlay` / "
           "`@colormap` / `@theme` / `@ai_tool` (or `@claude_tool`) / `@ui_theme` / `@ai_provider` / `@panel` "
           "(an activity-bar side bar) / `@settings_page` (a Preferences page) / `@workflow_source` (workflows "
           "a side bar lists and a notebook opens); `ewald plugins` lists "
           "what loaded. "
           "Details: [Plugins](plugins/index.md).", ""]
    for path in examples:
        out.append(f"- `examples/plugins/{path.name}`: {first_sentence(inspect.getdoc(_module_doc(path)))}")
    return out + [""]


def _module_doc(path: Path):
    import ast

    class _Doc:
        __doc__ = ast.get_docstring(ast.parse(path.read_text()))

    return _Doc


def settings_section() -> list[str]:
    from .settings import OPTIONS, PAGES

    out = ["## Preferences", "",
           "Edit › Preferences, one page per group; stored as JSON with only changed values. "
           "Scripts read them with `ewald.settings.load()`.", ""]
    for page in PAGES:
        out += [f"### {page}", ""]
        out += _table(("Key", "Label", "Default"),
                      [(f"`{o.key}`", o.label, f"`{o.default!r}`") for o in OPTIONS if o.page == page])
        out.append("")
    return out


def gui_section(window) -> list[str]:
    """Menus, toolbar, workspaces and panels of a live ``gui.app.MainWindow``."""
    from PySide6 import QtWidgets

    from .gui.viewport import Viewport

    def clean(text):  # drop mnemonics, keep a literal "&&" as "&"
        return " ".join(re.sub(r"&(&?)", r"\1", text).split())

    def walk(menu, depth):
        lines = []
        for act in menu.actions():
            if act.isSeparator() or not act.text():
                continue
            key = act.shortcut().toString()
            label = clean(act.text()) + (f" (`{key}`)" if key else "")
            lines.append("  " * depth + f"- {label}")
            if act.menu():
                lines += walk(act.menu(), depth + 1)
        return lines

    out = [GUI_START, "## Desktop GUI", "", "`ewald gui [FILE]`. Details: [Desktop GUI](user/gui.md).", "",
           "### Menus", ""]
    for act in window.menuBar().actions():
        if act.menu():
            out += [f"**{clean(act.text())}**", ""] + walk(act.menu(), 0) + [""]
    if hasattr(window, "viewport_menu"):
        out += ["**Viewport (right click)**, entries shown only when they apply (picked atoms, phonon "
                "modes, NEB band, volumetric grid, trajectory):", ""] + walk(window.viewport_menu(everything=True), 0) + [""]
    tb = window.findChild(QtWidgets.QToolBar, "main-toolbar")
    items = [clean(a.text()) for a in tb.actions() if a.text()]
    items += [clean(w.text()) for w in tb.findChildren(QtWidgets.QAbstractButton) if w.text().strip()]
    out += ["### Toolbar", "", ", ".join(dict.fromkeys(items)) + ", Compute device switch (CPU / CUDA / auto).", "",
            "### Panels", "",
            "- Workspaces: " + ", ".join(b.text() for b in window.ws_buttons),
            "- Docks: " + ", ".join(sorted(d.windowTitle() for d in window.findChildren(QtWidgets.QDockWidget))),
            "- Editor area tabs: " + ", ".join(window.workbench.titles("editor")),
            "- Panel tabs: " + ", ".join(window.workbench.titles("panel"))
            + " (tabs drag between groups: centre = tab, edge = split; View › Layout)",
            "- Viewport tools: " + "; ".join(tip for _key, _icon, tip in Viewport.TOOLS), "", GUI_END]
    return out


def inventory(gui_lines: list[str] | None = None) -> str:
    """The whole document; ``gui_lines`` is the GUI section (from :func:`gui_section` or the old file)."""
    from . import __all__ as api

    lines = [HEADER, "## Python API", "",
             "`import ewald`: " + ", ".join(f"`{n}`" for n in api)
             + ". Details: [Python scripting](user/scripting.md) and the [API reference](reference/api.md).", ""]
    lines += modifiers_section() + io_section() + rendering_section() + cli_section() + claude_section()
    lines += plugins_section() + settings_section()
    lines += gui_lines or [GUI_START, "## Desktop GUI", "",
                           "(not generated yet: run `python -m ewald inventory -o docs/features.md --gui`)", GUI_END]
    return "\n".join(lines).rstrip() + "\n"


def saved_gui_lines(text: str) -> list[str] | None:
    m = re.search(re.escape(GUI_START) + r".*?" + re.escape(GUI_END), text, re.S)
    return m.group(0).split("\n") if m else None


def live_gui_lines() -> list[str]:
    """Build a main window (default settings, no plugins) and read its menus."""
    import os
    import tempfile

    from PySide6 import QtWidgets

    from . import settings as settings_mod

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    saved = {k: os.environ.get(k) for k in ("EWALD_CONFIG_DIR", "EWALD_NO_PLUGINS")}
    with tempfile.TemporaryDirectory() as tmp:
        os.environ.update(EWALD_CONFIG_DIR=tmp, EWALD_NO_PLUGINS="1")
        try:
            from .gui.app import MainWindow

            w = MainWindow(settings=settings_mod.Settings())
            lines = gui_section(w)
            w.close()
            w.deleteLater()
            app.processEvents()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return lines


def update(output: str | None = None, gui: bool = False):
    """``ewald inventory``: print the inventory, or rewrite ``output`` (keeping its GUI section unless ``gui``)."""
    path = Path(output) if output else None
    old = path.read_text() if path is not None and path.exists() else ""
    text = inventory(live_gui_lines() if gui else saved_gui_lines(old))
    if path is None:
        print(text, end="")
    elif text != old:
        path.write_text(text)
        print("wrote", path)
    else:
        print(path, "is up to date")
