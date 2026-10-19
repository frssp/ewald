"""Command line: ``python -m ewald render INPUT OUTPUT.png [--frame N] ...``."""
from __future__ import annotations

import argparse

from . import import_file


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ewald")
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
        parser.add_argument("--renderer", choices=("opengl", "raytrace", "vector"), default=None)
        parser.add_argument("--allow-code", action="store_true", help="run Python modifiers stored in a session")
        parser.add_argument("--quality", choices=("draft", "standard", "publication"), default=None,
                            help="sampling preset (default publication; draft renders at half size)")
        parser.add_argument("--text", default=None, help="label, e.g. 'frame [Frame]  t = [Time] ps'")
        parser.add_argument("--type-legend", action="store_true", help="swatch per particle type")
        parser.add_argument("--scale-bar", nargs="?", type=float, const=0.0, default=None, metavar="LENGTH",
                            help="scale bar; optional length in Angstrom (default: a round length)")
    r.add_argument("--dpi", type=float, default=None, help="stored in PNG/TIFF; page size of vector output")
    g = sub.add_parser("gui", help="open the desktop viewer")
    g.add_argument("input", nargs="?", help="structure/trajectory file or .ewald session")
    g.add_argument("--type-map", default=None)
    g.add_argument("--device", default=None)
    i = sub.add_parser("info", help="print frames and properties")
    i.add_argument("input")
    i.add_argument("--type-map", default=None)
    d = sub.add_parser("describe", help="JSON summary of a frame or, with --trajectory, of the run")
    d.add_argument("input")
    d.add_argument("--frame", type=int, default=0)
    d.add_argument("--trajectory", action="store_true", help="scan frames and report events")
    d.add_argument("--type-map", default=None)
    q = sub.add_parser("ask", help="ask Claude about a structure (needs anthropic + credentials)")
    q.add_argument("input")
    q.add_argument("question", nargs="+")
    q.add_argument("--type-map", default=None)
    q.add_argument("--model", default=None)
    q.add_argument("--allow-code", action="store_true", help="let Claude run Python in this process")
    q.add_argument("--script", default=None, help="write the resulting pipeline as a Python script")
    q.add_argument("--select", default=None, metavar="EXPR",
                   help="ask about these particles, e.g. 'ParticleIdentifier == 17' or 'Position.Z > 20'")
    q.add_argument("--ids", default=None, help="ask about these Particle Identifiers (comma-separated)")
    q.add_argument("--frame", type=int, default=0, help="frame for --select/--ids (default 0)")
    n = sub.add_parser("narrate", help="timeline of events in a trajectory (Claude writes it if available)")
    n.add_argument("input")
    n.add_argument("--type-map", default=None)
    n.add_argument("--no-llm", action="store_true", help="rule-based titles only")
    n.add_argument("--render", action="store_true", help="show Claude renders of the frames around events")
    n.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    a = sub.add_parser("api", help="print the Python API reference for LLMs (llms.txt)")
    a.add_argument("--with-plugins", action="store_true", help="append the modifier plugins installed here")
    pl = sub.add_parser("plugins", help="list installed plugins and any that failed to load")
    pl.add_argument("--json", action="store_true")
    f = sub.add_parser("forge", help="have Claude write a modifier plugin, test it and save it")
    f.add_argument("request", nargs="+", help="the analysis, in words")
    f.add_argument("--yes", action="store_true", help="run the generated code without asking")
    f.add_argument("--draft", action="store_true", help="print Claude's first draft; run and save nothing")
    f.add_argument("--overwrite", action="store_true", help="replace a plugin file of the same name")
    f.add_argument("--rounds", type=int, default=4, help="attempts to fix failing checks (default 4)")
    f.add_argument("--model", default=None)
    m = sub.add_parser("mcp", help="serve the analysis tools over MCP (stdio)")
    m.add_argument("input", nargs="?")
    m.add_argument("--allow-code", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "gui":
        from .gui import main as gui_main

        extra = ([args.input] if args.input else []) + (["--type-map", args.type_map] if args.type_map else [])
        return gui_main(extra + (["--device", args.device] if args.device else []))
    if args.cmd == "api":
        from .llm.reference import api_reference

        print(api_reference(with_plugins=args.with_plugins), end="")
        return
    if args.cmd == "plugins":
        return _print_plugins(args.json)
    if args.cmd == "forge":
        return _forge(args)
    if args.cmd == "mcp":
        from .llm.mcp_server import serve

        return serve(args.allow_code, args.input)

    kw = {}
    if args.type_map:
        kw["type_map"] = dict(item.split("=") for item in args.type_map.split(","))
    if args.cmd in ("render", "movie"):
        return _render_cmd(args, kw)
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

        client = None
        if not args.no_llm:
            try:
                import anthropic

                client = anthropic.Anthropic()
            except Exception as exc:  # no SDK or no credentials: fall back to rule-based titles
                print(f"(no Claude client: {exc}; using rule-based titles)")
        tl = narrate(pipe, client, render=args.render)
        print(json.dumps(tl.to_dict(), indent=1, default=float) if args.json else tl.to_markdown())
        return
    if args.cmd == "ask":
        from .llm import Agent, Toolset

        ts = Toolset(pipe, allow_code=args.allow_code)
        ts._source_call = {"path": args.input, "kwargs": kw}
        agent = Agent(ts, **({"model": args.model} if args.model else {}),
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


def _print_plugins(as_json=False):
    from . import plugins

    st = plugins.status()
    if as_json:
        import json

        print(json.dumps(st, indent=1))
        return
    print("plugin folder:", st["plugin_dir"])
    if not st["readers"]:
        print("no reader plugins")
    for r in st["readers"]:
        pats = " ".join(r["patterns"]) or "(format= only)"
        print(f"reader {r['name']:<16} {pats:<24} {r['description']}  [{r['origin']}]")
    for m in st["modifiers"]:
        print(f"modifier {m['name']:<24} {m['category']:<14} {m['description'][:60]}  [{m['origin']}]")
    for origin, err in st["errors"].items():
        print(f"FAILED {origin}: {err}")
    return 1 if st["errors"] else None


def _forge(args):
    from .llm.forge import DEFAULT_MODEL, forge

    request = " ".join(args.request)
    if not (args.draft or args.yes):
        answer = input("Claude's code will run on this machine (a subprocess with a time limit, "
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

    result = forge(request, model=args.model or DEFAULT_MODEL, run_code=not args.draft, max_rounds=args.rounds,
                   overwrite=args.overwrite, on_event=show)
    if args.draft:
        print(result.module_source)
        print("# ---- tests ----")
        print(result.test_source)
        return
    print(result.summary())
    if result.ok:
        print(f"use it: from ewald.modifiers import {result.class_name}  (also in the GUI and Claude's tools)")
    return None if result.ok else 1


if __name__ == "__main__":
    main()
