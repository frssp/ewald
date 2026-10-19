"""Regenerate the PLUMED reference files in this folder (never edit them by hand).

Needs the ``plumed`` command line tool (PLUMED 2.10, on PATH). Runs ``plumed driver`` with
well-tempered METAD over a random-walk trajectory to make HILLS_1d (one periodic CV) and
HILLS_2d (a distance and a torsion), then ``plumed sum_hills`` for the free energy surfaces
fes_1d.dat and fes_2d.dat that tests/test_plumed.py compares ewald.plumed.fes_from_hills with.

    python tests/data/plumed/make_plumed_reference.py
"""
import os
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

INPUT = """d: DISTANCE ATOMS=1,2
t: TORSION ATOMS=1,2,3,4
METAD ARG={args} SIGMA={sigma} HEIGHT=1.0 PACE=2 BIASFACTOR=8 TEMP=300 FILE={hills}
"""


def trajectory(path, nframes=300, seed=3):
    rng = np.random.default_rng(seed)
    x = np.array([[0, 0, 0], [1.5, 0, 0], [2.0, 1.4, 0], [3.4, 1.6, 0.5]], float)
    with open(path, "w") as fh:
        for _ in range(nframes):
            x = x + rng.normal(0, 0.15, x.shape)
            fh.write("4\n20 20 20\n")
            for p in x:
                fh.write("C %.10f %.10f %.10f\n" % tuple(p))


def main():
    os.chdir(HERE)
    trajectory("traj.xyz")
    for name, args, sigma in (("1d", "t", "0.35"), ("2d", "d,t", "0.03,0.35")):
        hills = f"HILLS_{name}"
        Path("plumed.dat").write_text(INPUT.format(args=args, sigma=sigma, hills=hills))
        for f in (hills,):
            if os.path.exists(f):
                os.remove(f)
        subprocess.run(["plumed", "driver", "--plumed", "plumed.dat", "--ixyz", "traj.xyz", "--length-units", "A",
                        "--kt", "2.494338785"], check=True, capture_output=True)
        cmd = ["plumed", "sum_hills", "--hills", hills, "--outfile", f"fes_{name}.dat", "--mintozero"]
        cmd += ["--bin", "60"] if name == "1d" else ["--bin", "30,40", "--min", "0.0,-pi", "--max", "0.6,pi"]
        subprocess.run(cmd, check=True, capture_output=True)
    for f in ("traj.xyz", "plumed.dat"):
        os.remove(f)
    for f in HERE.glob("bck.*"):
        f.unlink()


if __name__ == "__main__":
    main()
