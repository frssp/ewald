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
    i = sub.add_parser("info", help="print frames and properties")
    i.add_argument("input")
    i.add_argument("--type-map", default=None)
    args = ap.parse_args(argv)

    kw = {}
    if args.type_map:
        kw["type_map"] = dict(item.split("=") for item in args.type_map.split(","))
    pipe = import_file(args.input, **kw)
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


if __name__ == "__main__":
    main()
