"""Size and time of vector figures (SVG/PDF) against the number of drawn shapes.

    python benchmarks/vector_size.py                  # fcc Cu 32 .. 8788 atoms, with and without bonds
    python benchmarks/vector_size.py --max-repeat 8   # stop at 8x8x8 cells (2048 atoms)

Prints the measured file size and time next to the estimate of ``ewald.vector.vector_cost``,
whose per-shape constants (``SHAPE_SECONDS``, ``SHAPE_BYTES``) come from this script.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time

from ase.build import bulk

from ewald import DataCollection
from ewald import scene as ir
from ewald.render import render
from ewald.vector import vector_cost


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-repeat", type=int, default=13)
    args = ap.parse_args()
    print(f"{'atoms':>6} {'bonds':>5} {'fmt':>4} {'shapes':>7} {'MB':>7} {'est MB':>7} {'s':>6} {'est s':>6}")
    with tempfile.TemporaryDirectory() as tmp:
        for n in [r for r in (2, 3, 4, 6, 8, 10, 13) if r <= args.max_repeat]:
            data = DataCollection.from_ase(bulk("Cu", "fcc", a=3.61, cubic=True).repeat((n, n, n)))
            for bonds in (None, 2.8):
                for ext in (".svg", ".pdf"):
                    out = os.path.join(tmp, "fig" + ext)
                    t = time.perf_counter()
                    render(data, out, renderer="vector", bonds=bonds, size=(800, 600), large_vector="ignore")
                    dt = time.perf_counter() - t
                    est = vector_cost(ir.build(data, bonds=bonds), out)
                    print(f"{data.count:6d} {'yes' if bonds else 'no':>5} {ext[1:]:>4} {est['shapes']:7d} "
                          f"{os.path.getsize(out) / 1e6:7.2f} {est['bytes'] / 1e6:7.2f} {dt:6.1f} {est['seconds']:6.1f}",
                          flush=True)


if __name__ == "__main__":
    main()
