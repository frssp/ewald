"""Command line: ``python -m ewald render INPUT OUTPUT.png [--frame N] ...``."""
from __future__ import annotations

import argparse
import sys

from . import import_file
from .render import RENDERERS


def build_parser():
    """The ``ewald`` argument parser (also read by ``ewald inventory`` for docs/features.md)."""
    from . import __title__, __version__

    ap = argparse.ArgumentParser(prog="ewald", description=__title__)
    ap.add_argument("--version", action="version", version=f"ewald {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render a frame to an image (.png/.jpg/.tif/.svg/.pdf/.eps)")
    r.add_argument("input", help="structure/trajectory file or .ewald session")
    r.add_argument("output")
    r.add_argument("--frame", type=int, default=None)
    r.add_argument("--size", type=int, nargs=2, default=None, help="pixels (default 1600 1200)")
    r.add_argument("--direction", default=None, help="view direction (default iso)")
    r.add_argument("--bonds", type=float, default=None, help="bond cutoff in Angstrom")
    r.add_argument("--color-by", default=None, help="property or expression to color code")
    r.add_argument("--cmap", default="viridis")
    r.add_argument("--type-map", default=None, help="LAMMPS types, e.g. 1=Cu,2=Zr")
    r.add_argument("--ao", action="store_true", help="screen-space ambient occlusion")
    mv = sub.add_parser("movie", help="render frames to .mp4/.gif/.avi/.mov or a PNG sequence")
    mv.add_argument("input", help="structure/trajectory file or .ewald session")
    mv.add_argument("output")
    mv.add_argument("--fps", type=float, default=10)
    mv.add_argument("--start", type=int, default=0)
    mv.add_argument("--stop", type=int, default=None)
    mv.add_argument("--every", type=int, default=1, help="frame stride")
    mv.add_argument("--size", type=int, nargs=2, default=None, help="pixels (default 1280 960)")
    mv.add_argument("--direction", default=None, help="view direction (default iso)")
    mv.add_argument("--bonds", type=float, default=None, help="bond cutoff in Angstrom")
    mv.add_argument("--color-by", default=None, help="property or expression to color code")
    mv.add_argument("--cmap", default="viridis")
    mv.add_argument("--type-map", default=None, help="LAMMPS types, e.g. 1=Cu,2=Zr")
    mv.add_argument("--time-per-frame", type=float, default=None, help="for [Time] in --text")
    for parser in (r, mv):
        parser.add_argument("--renderer", choices=tuple(RENDERERS), default=None)
        parser.add_argument("--allow-code", action="store_true", help="run Python modifiers stored in a session")
        parser.add_argument("--quality", choices=("draft", "standard", "publication"), default=None,
                            help="sampling preset (default publication; draft renders at half size)")
        parser.add_argument("--text", default=None, help="label, e.g. 'frame [Frame]  t = [Time] ps'")
        parser.add_argument("--type-legend", action="store_true", help="swatch per particle type")
        parser.add_argument("--scale-bar", nargs="?", type=float, const=0.0, default=None, metavar="LENGTH",
                            help="scale bar; optional length in Angstrom (default: a round length)")
        parser.add_argument("--theme", default=None,
                            help="element colors, background and cell color, e.g. VESTA, Journal, 'Dark slide'")
        parser.add_argument("--overlay", action="append", default=None, metavar="NAME",
                            help="overlay element, e.g. 'Coordinate tripod' or Attributes (repeatable)")
    r.add_argument("--dpi", type=float, default=None, help="stored in PNG/TIFF; page size of vector output")
    g = sub.add_parser("gui", help="open the desktop viewer")
    g.add_argument("input", nargs="?", help="structure/trajectory file or .ewald session")
    g.add_argument("--type-map", default=None)
    g.add_argument("--device", default=None)
    ex = sub.add_parser("examples", help="list the shipped example files (POSCAR, CIF, extxyz, LAMMPS, ...)")
    ex.add_argument("--copy", default=None, metavar="DIR", help="copy them into DIR")
    ex.add_argument("--papers", action="store_true", help="list the published datasets that --fetch downloads")
    ex.add_argument("--fetch", default=None, metavar="NAME", help="download a published dataset (checksummed)")
    i = sub.add_parser("info", help="print frames and properties")
    i.add_argument("input")
    i.add_argument("--type-map", default=None)
    d = sub.add_parser("describe", help="JSON summary of a frame or, with --trajectory, of the run")
    d.add_argument("input")
    d.add_argument("--frame", type=int, default=0)
    d.add_argument("--trajectory", action="store_true", help="scan frames and report events")
    d.add_argument("--type-map", default=None)
    c = sub.add_parser("check", help="what a run is (from INCAR / log.lammps) and whether it converged and equilibrated")
    c.add_argument("input", nargs="+", help="trajectory or output files (several: one check each)")
    c.add_argument("--json", action="store_true", help="print the findings as JSON")
    c.add_argument("--type-map", default=None)
    q = sub.add_parser("ask", help="ask AI about a structure (Claude by default; needs the provider's package + key)")
    q.add_argument("input")
    q.add_argument("question", nargs="+")
    q.add_argument("--type-map", default=None)
    q.add_argument("--model", default=None)
    _ai_arguments(q)
    q.add_argument("--allow-code", action="store_true", help="let the AI run Python in this process")
    q.add_argument("--script", default=None, help="write the resulting pipeline as a Python script")
    q.add_argument("--select", default=None, metavar="EXPR",
                   help="ask about these particles, e.g. 'ParticleIdentifier == 17' or 'Position.Z > 20'")
    q.add_argument("--ids", default=None, help="ask about these Particle Identifiers (comma-separated)")
    q.add_argument("--frame", type=int, default=0, help="frame for --select/--ids (default 0)")
    n = sub.add_parser("narrate", help="timeline of events in a trajectory (an AI writes it if available)")
    n.add_argument("input")
    n.add_argument("--type-map", default=None)
    n.add_argument("--no-llm", action="store_true", help="rule-based titles only")
    n.add_argument("--render", action="store_true", help="show the AI renders of the frames around events")
    n.add_argument("--model", default=None)
    _ai_arguments(n)
    n.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    a = sub.add_parser("api", help="print the Python API reference for LLMs (llms.txt)")
    a.add_argument("--with-plugins", action="store_true", help="append the modifier plugins installed here")
    pl = sub.add_parser("plugins", help="list installed plugins and any that failed to load")
    pl.add_argument("--json", action="store_true")
    pl.add_argument("--enable", metavar="ID", help="switch a disabled plugin back on")
    pl.add_argument("--disable", metavar="ID", help="switch a plugin off (it stays installed; its code is not imported)")
    f = sub.add_parser("forge", help="have an AI write a modifier plugin, test it and save it")
    f.add_argument("request", nargs="+", help="the analysis, in words")
    f.add_argument("--yes", action="store_true", help="run and save the generated code without asking")
    f.add_argument("--draft", action="store_true", help="print the AI's first draft; run and save nothing")
    f.add_argument("--overwrite", action="store_true", help="replace a plugin file of the same name")
    f.add_argument("--rounds", type=int, default=4, help="attempts to fix failing checks (default 4)")
    f.add_argument("--model", default=None)
    _ai_arguments(f)
    m = sub.add_parser("mcp", help="serve the analysis tools over MCP (stdio)")
    m.add_argument("input", nargs="?")
    m.add_argument("--allow-code", action="store_true")
    rm = sub.add_parser("remote", help="saved SSH profiles: list, add, remove, test, ls, get")
    rsub = rm.add_subparsers(dest="remote_cmd", required=True)
    rsub.add_parser("list", help="saved profiles and unsaved ~/.ssh/config hosts")
    ra = rsub.add_parser("add", help="add or update a profile (its name is used as host: NAME:/path)")
    ra.add_argument("name")
    ra.add_argument("--host", default="", help="host name, IP or ~/.ssh/config alias (default: NAME)")
    ra.add_argument("--user", default="")
    ra.add_argument("--port", type=int, default=None)
    ra.add_argument("--key", default="", help="private key file (auth = key)")
    ra.add_argument("--password", action="store_true",
                    help="password login; asks for the password and saves it in the OS keychain")
    ra.add_argument("--passphrase", action="store_true", help="ask for the key passphrase and save it in the keychain")
    ra.add_argument("--jump", default="", help="ProxyJump: another profile, an alias or user@host[:port]")
    ra.add_argument("--folder", default="", help="start folder of the file browser")
    rr = rsub.add_parser("remove", help="delete a profile and its saved password/passphrase")
    rr.add_argument("name")
    rt = rsub.add_parser("test", help="connect with a profile (or ~/.ssh/config alias) and print the remote home")
    rt.add_argument("name")
    rl = rsub.add_parser("ls", help="list a remote folder, e.g. nurion:/scratch/me")
    rl.add_argument("path")
    rg = rsub.add_parser("get", help="download a remote file for local analysis")
    rg.add_argument("path")
    rg.add_argument("dest", nargs="?", default=".", help="file or folder (default: current folder)")
    inv = sub.add_parser("inventory", help="print or update the feature inventory (docs/features.md)")
    inv.add_argument("--output", "-o", default=None, help="rewrite this file, e.g. docs/features.md")
    inv.add_argument("--gui", action="store_true",
                     help="regenerate the GUI section from a live main window (needs PySide6 and a display); "
                          "otherwise it is kept from --output")
    return ap


# optional package -> the extra that installs it (for "ewald CMD" without that extra)
EXTRAS = {"PySide6": "gui", "pyvistaqt": "gui", "qtconsole": "gui", "anthropic": "llm", "openai": "llm", "mcp": "llm",
          "anyio": "llm", "h5py": "formats", "MDAnalysis": "formats",
          "cupy": "gpu", "imageio_ffmpeg": "movie"}


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except ImportError as exc:  # a missing optional dependency: one line, not a traceback
        top = (getattr(exc, "name", None) or "").split(".")[0]
        if top in EXTRAS:
            sys.exit(f"ewald {args.cmd}: needs {top}: pip install 'ewald[{EXTRAS[top]}]'")
        if "pip install" in str(exc):
            sys.exit(f"ewald {args.cmd}: {exc}")
        raise


def _remote_cmd(args):
    import getpass
    import time

    from .io import profiles, remote

    remote._require_paramiko()
    cmd = args.remote_cmd
    if cmd == "list":
        for p in profiles.profiles():
            saved = [k for k in ("password", "passphrase") if profiles.secret(p.name, k)]
            extra = (f"  via {p.jump}" if p.jump else "") + (f"  [{', '.join(saved)} saved]" if saved else "")
            print(f"{p.name:16s} {p.target}  ({profiles.AUTH[p.auth]}){extra}")
        hosts = profiles.unsaved_config_hosts()
        if hosts:
            print(f"~/.ssh/config: {' '.join(hosts)}")
        if not profiles.profiles() and not hosts:
            print("No profiles. Add one with: ewald remote add NAME --host HOST --user USER")
        return
    if cmd == "add":
        old = profiles.get(args.name)
        auth = "password" if args.password else "key" if args.key else (old.auth if old else "auto")
        prof = profiles.Profile(args.name, host=args.host or (old.host if old else ""),
                                user=args.user or (old.user if old else ""),
                                port=args.port or (old.port if old else None), auth=auth,
                                key_file=args.key or (old.key_file if old else ""),
                                jump=args.jump or (old.jump if old else ""),
                                folder=args.folder or (old.folder if old else ""),
                                bookmarks=old.bookmarks if old else [])
        pw = getpass.getpass(f"Password for {prof.target}: ") if args.password else None
        phrase = getpass.getpass(f"Passphrase for {prof.key_file}: ") if args.passphrase else None
        try:
            profiles.save(prof, password=pw, passphrase=phrase)
        except ValueError as exc:
            sys.exit(f"ewald remote add: {exc}")
        where = profiles.secret_location(prof.name, "password" if pw else "passphrase") if (pw or phrase) else None
        print(f"Saved {prof.name} ({prof.target})" + (f"; secret in the {where}" if where else ""))
        print(f"Open files as {prof.name}:/path; browse with: ewald remote ls {prof.name}")
        return
    if cmd == "remove":
        if profiles.get(args.name) is None:
            sys.exit(f"ewald remote remove: no profile named {args.name!r}")
        profiles.delete(args.name)
        print(f"Removed {args.name}")
        return
    if cmd == "test":
        prof = profiles.get(args.name) or profiles.from_ssh_config(args.name)
        ok, msg = profiles.test(prof)
        print(msg)
        return 0 if ok else 1
    if cmd == "ls":
        path = args.path if remote.is_remote(args.path) else args.path + ":"
        rp = remote.parse_remote(path)
        if not rp.path or rp.path == ".":
            prof = profiles.get(rp.host)
            path = str(rp.with_path(prof.folder if prof and prof.folder else remote.home(path)))
        for name, is_dir, size, mtime in remote.listdir(path):
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
            print(f"{'' if is_dir else size:>12} {stamp}  {name}{'/' if is_dir else ''}")
        return
    if cmd == "get":
        print(remote.download(args.path, args.dest))
        return


def _run(args):
    if args.cmd == "gui":
        from .gui import main as gui_main

        extra = ([args.input] if args.input else []) + (["--type-map", args.type_map] if args.type_map else [])
        return gui_main(extra + (["--device", args.device] if args.device else []))
    if args.cmd == "api":
        from .llm.reference import api_reference

        print(api_reference(with_plugins=args.with_plugins), end="")
        return
    if args.cmd == "examples":
        return _examples(args.copy, args.papers, args.fetch)
    if args.cmd == "plugins":
        return _print_plugins(args.json, enable=args.enable, disable=args.disable)
    if args.cmd == "forge":
        return _forge(args)
    if args.cmd == "inventory":
        from .inventory import update

        return update(args.output, gui=args.gui)
    if args.cmd == "remote":
        return _remote_cmd(args)
    if args.cmd == "mcp":
        from .llm.mcp_server import serve

        return serve(args.allow_code, args.input)

    kw = {}
    if args.type_map:
        kw["type_map"] = dict(item.split("=") for item in args.type_map.split(","))
    if args.cmd in ("render", "movie"):
        return _render_cmd(args, kw)
    if args.cmd == "check":
        import json

        from .runinfo import check_run

        checks = {}
        for path in args.input:
            check = check_run(import_file(path, **kw))
            checks[path] = check
            if not args.json:
                print(f"{path}\n{check}\n")
        if args.json:
            print(json.dumps({p: c.to_dict() for p, c in checks.items()}, indent=1, default=float))
        return 0 if all(c.ok for c in checks.values()) else 1
    pipe = import_file(args.input, **kw)
    if args.cmd == "describe":
        import json

        from .llm.describe import describe_frame, describe_trajectory

        out = describe_trajectory(pipe) if args.trajectory else describe_frame(pipe.compute(args.frame), frame=args.frame)
        print(json.dumps(out, indent=1, default=float))
        return
    if args.cmd == "narrate":
        import json

        from .llm.narrate import narrate

        client, ai = None, _ai_options(args, effort=False)
        if not args.no_llm:
            try:
                client = ai["provider"].client(model=ai["model"])
            except Exception as exc:  # no SDK or no credentials: fall back to rule-based titles
                print(f"(no AI client: {exc}; using rule-based titles)")
        tl = narrate(pipe, client, render=args.render, provider=ai["provider"], model=ai["model"])
        print(json.dumps(tl.to_dict(), indent=1, default=float) if args.json else tl.to_markdown())
        return
    if args.cmd == "ask":
        from .llm import Agent, Toolset

        ts = Toolset(pipe, allow_code=args.allow_code)
        ts._source_call = {"path": args.input, "kwargs": kw}
        agent = Agent(ts, **_ai_options(args),
                      on_event=lambda kind, p: print(f"  -> {p['name']}({p['input']})")
                      if kind == "tool_call" else None)
        question = " ".join(args.question)
        if args.select or args.ids:
            data = pipe.compute(args.frame % pipe.num_frames)
            ids = [int(x) for x in args.ids.split(",")] if args.ids else None
            indices = ts._selection_indices(data, ids=ids, expression=args.select)
            if not len(indices):
                print("the selection is empty")
                return 1
            print(agent.ask_about_selection(question, indices, frame=args.frame))
        else:
            print(agent.ask(question))
        if args.script:
            with open(args.script, "w") as fh:
                fh.write(ts.to_script())
            print("wrote", args.script)
        return
    if args.cmd == "info":
        d = pipe.compute(0)
        print(f"{args.input}: {pipe.num_frames} frames, {d.count} particles")
        print("properties:", ", ".join(d.particles))
        print("cell:\n", d.cell.matrix, "\npbc:", d.cell.pbc)
        return



def _render_cmd(args, reader_kwargs):
    """``render`` and ``movie``: a file or a session, with command-line flags overriding the session."""
    from .session import is_session_file, load_session

    if is_session_file(args.input):
        sess = load_session(args.input, allow_code=args.allow_code)
        pipe, opts, frame = sess.pipeline, sess.render_kwargs(), sess.frame
        if sess.camera and args.direction is None:
            opts["camera"] = sess.camera
        text = args.text if args.text is not None else sess.text
        tpf = sess.render.get("time_per_frame")
    else:
        pipe, opts, frame, text, tpf = import_file(args.input, **reader_kwargs), {}, 0, args.text, None
    if args.color_by:
        from .modifiers import ColorCoding

        pipe.modifiers.append(ColorCoding(args.color_by, cmap=args.cmap))
    for key, value in (("size", tuple(args.size) if args.size else None), ("direction", args.direction),
                       ("bonds", args.bonds), ("renderer", args.renderer), ("quality", args.quality)):
        if value is not None:
            opts[key] = value
    if args.type_legend:
        opts["type_legend"] = True
    if args.scale_bar is not None:
        opts["scale_bar"] = args.scale_bar or True
    if args.theme:
        opts["theme"] = args.theme
    if args.overlay:
        opts["overlays"] = list(opts.get("overlays") or []) + list(args.overlay)
    if args.cmd == "movie":
        from .render import render_animation

        opts.setdefault("size", (1280, 960))
        frames = range(args.start, pipe.num_frames if args.stop is None else args.stop, args.every)
        n = render_animation(pipe, args.output, frames=frames, fps=args.fps, text=text,
                             time_per_frame=args.time_per_frame if args.time_per_frame is not None else tpf,
                             progress=lambda i, n: print(f"\rframe {i}/{n}", end="", flush=True), **opts)
        print(f"\nwrote {args.output} ({n} frames)")
        return
    from .overlays import format_label
    from .render import render

    frame = args.frame if args.frame is not None else frame
    data = pipe.compute(frame)
    if args.ao:
        opts["ambient_occlusion"] = True
    if args.dpi:
        opts["dpi"] = args.dpi
    render(data, args.output, text=format_label(text, frame, data, tpf) or None, **opts)
    print("wrote", args.output)


def _examples(dest=None, papers=False, fetch=None):
    from . import examples
    from .examples import papers as published

    if fetch:
        print(examples.fetch(fetch))
        return
    if papers:
        for p in published.PAPERS:
            print(f"{p.name}  ({p.size / 1e6:.1f} MB, {p.license})\n  {p.title}\n  {p.citation}\n"
                  f"  reproduces: {p.reproduces}\n  script: examples/papers/{p.script}\n")
        print(f"downloaded into {published.data_dir()}")
        return
    if dest:
        print(f"copied {len(examples.EXAMPLES)} files to {examples.copy(dest)}")
        return
    width = max(len(ex.file) for ex in examples.EXAMPLES)
    for ex in examples.EXAMPLES:
        print(f"{ex.file:<{width}}  {ex.format}: {ex.description}")
    print(f"\nin {examples.FILES}; open one with: ewald gui {examples.EXAMPLES[0].path}")
    print("published data from papers: ewald examples --papers")


def _print_plugins(as_json=False, enable=None, disable=None):
    from . import plugins

    for plugin_id, on in ((enable, True), (disable, False)):
        if plugin_id:
            if plugin_id not in plugins.installed():
                print(f"no installed plugin {plugin_id!r}; `ewald plugins` lists the ids")
                return 1
            plugins.set_enabled(plugin_id, on)
    st = plugins.status()
    if as_json:
        import json

        print(json.dumps(st, indent=1))
        return
    print("plugin folder:", st["plugin_dir"])
    for p in st["plugins"]:
        made = ", ".join(f"{len(v)} {k[:-1] if len(v) == 1 else k}" for k, v in p["contributions"].items() if v)
        print(f"plugin {p['id']:<20} {p['version'] or '-':<9} {p['kind']:<8} {p['state']:<12} "
              f"{(made or p['description'])[:60]}  [{p['origin']}]")
    if not st["readers"]:
        print("no reader plugins")
    for r in st["readers"]:
        pats = " ".join(r["patterns"]) or "(format= only)"
        print(f"reader {r['name']:<16} {pats:<24} {r['description']}  [{r['origin']}]")
    for m in st["modifiers"]:
        print(f"modifier {m['name']:<24} {m['category']:<14} {m['description'][:60]}  [{m['origin']}]")
    for w in st.get("writers", []):
        pats = " ".join(w["patterns"]) or "(format= only)"
        print(f"writer {w['name']:<16} {pats:<24} {w['description']}  [{w['origin']}]")
    for v in st.get("visuals", []):
        print(f"visual {v['name']:<24} {v['category']:<14} {v['description'][:60]}  [{v['origin']}]")
    for v in st.get("overlays", []):
        print(f"overlay {v['name']:<23} {v['category']:<14} {v['description'][:60]}  [{v['origin']}]")
    for v in st.get("colormaps", []):
        print(f"colormap {v['name']:<22} {v['kind']:<14} {v['description'][:60]}  [{v['origin']}]")
    for v in st.get("themes", []):
        print(f"theme {v['name']:<25} {'':<14} {v['description'][:60]}  [{v['origin']}]")
    for v in st.get("ui_themes", []):
        print(f"ui theme {v['name']:<22} {v['base']:<14} {v['description'][:60]}  [{v['origin']}]")
    for v in st.get("ai_providers", []):
        print(f"ai provider {v['name']:<19} {v['label'][:14]:<14} {v['description'][:60]}  [{v['origin']}]")
    for origin, err in st["errors"].items():
        print(f"FAILED {origin}: {err}")
    return 1 if st["errors"] else None


def _ai_arguments(parser):
    parser.add_argument("--provider", default=None,
                        help="AI provider: anthropic, openai, gemini, openai-compatible, gateway or a plugin's "
                             "(default: Preferences > AI)")
    parser.add_argument("--profile", default=None, metavar="NAME",
                        help="a connection saved in Preferences > AI (provider, model, address, headers)")
    parser.add_argument("--base-url", default=None, help="server address (OpenAI-compatible server, gateway)")


def _ai_options(args, effort: bool = True) -> dict:
    """provider (connected), model and effort for an AI command: the flags, else the saved
    connection ``--profile``, else Preferences > AI."""
    from .llm.providers import from_settings

    ai = from_settings(profile=args.profile, provider=args.provider, model=args.model, base_url=args.base_url)
    if not effort:
        ai.pop("effort")
    return ai


def _forge(args):
    from .llm.forge import forge

    request = " ".join(args.request)
    if not (args.draft or args.yes):
        answer = input("The AI's code will run on this machine (a subprocess with a time limit, "
                       "not a sandbox) to test it. Continue? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("cancelled; use --draft to only see the code")
            return 1

    def show(kind, result):
        if kind == "draft":
            print(f"round {result.rounds}: {result.class_name} ({result.module_name}.py)", flush=True)
        elif not result.ok:
            failed = [i["name"] for i in result.report.get("checks", []) + result.report.get("tests", [])
                      if not i["ok"]]
            print("  failed: " + ", ".join(failed), flush=True)

    def confirm(result):
        if args.yes:
            return True
        print(f"# ---- {result.module_name}.py ----\n{result.module_source}")
        answer = input("Tests passed. Save this plugin? It is loaded every time ewald starts. [y/N] ")
        return answer.strip().lower() in ("y", "yes")

    result = forge(request, **_ai_options(args, effort=False), run_code=not args.draft, max_rounds=args.rounds,
                   overwrite=args.overwrite, on_event=show, confirm=confirm)
    if args.draft:
        print(result.module_source)
        print("# ---- tests ----")
        print(result.test_source)
        return
    print(result.summary())
    if result.ok and result.path:
        print(f"use it: from ewald.modifiers import {result.class_name}  (also in the GUI and the AI tools)")
    elif result.ok:
        print("not saved")
    return None if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
