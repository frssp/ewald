"""Command line: ``python -m ewald render INPUT OUTPUT.png [--frame N] ...``."""
from __future__ import annotations

import argparse

from . import import_file


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ewald")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render a frame to an image")
    r.add_argument("input")
    r.add_argument("output")
    r.add_argument("--frame", type=int, default=0)
    r.add_argument("--size", type=int, nargs=2, default=(1600, 1200))
    r.add_argument("--direction", default="iso")
    r.add_argument("--bonds", type=float, default=None, help="bond cutoff in Angstrom")
    r.add_argument("--color-by", default=None, help="property or expression to color code")
    r.add_argument("--cmap", default="viridis")
    r.add_argument("--type-map", default=None, help="LAMMPS types, e.g. 1=Cu,2=Zr")
    r.add_argument("--ao", action="store_true", help="screen-space ambient occlusion")
    g = sub.add_parser("gui", help="open the desktop viewer")
    g.add_argument("input", nargs="?")
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
        print(agent.ask(" ".join(args.question)))
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
    if args.color_by:
        from .modifiers import ColorCoding

        pipe.modifiers.append(ColorCoding(args.color_by, cmap=args.cmap))
    pipe.render(args.output, frame=args.frame, size=tuple(args.size), direction=args.direction,
                bonds=args.bonds, ambient_occlusion=args.ao)
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
