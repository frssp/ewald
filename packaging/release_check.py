"""Release check: run before tagging, and in CI on the tag. See docs/developer/release.md.

    python packaging/release_check.py               # checks, builds sdist + wheel, installs the wheel in a
                                                    # fresh venv and smoke-tests it (dist/ holds the files)
    python packaging/release_check.py --check-only  # version, tag and CHANGELOG checks only (no build)
    python packaging/release_check.py --notes       # print the CHANGELOG section of this version

The version lives in one place, ``__version__`` in src/ewald/__init__.py (pyproject.toml reads it).
CHANGELOG.md must have a ``## [X.Y.Z] - YYYY-MM-DD`` section and a ``[X.Y.Z]:`` link for it. On a
``v*`` tag (``GITHUB_REF`` or ``--tag``) the tag must be ``v`` + that version.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"^\d+\.\d+\.\d+((a|b|rc)\d+)?$")

# files the wheel must carry (package data that is easy to drop from pyproject.toml by accident)
WHEEL_MUST_HAVE = ["ewald/tables/radii.json", "ewald/tables/colors.json", "ewald/assets/icon/ewald.svg",
                   "ewald/assets/icon/ewald.ico", "ewald/assets/icon/ewald.icns", "ewald/assets/icon/ewald-256.png",
                   "ewald/examples/files/POSCAR.SrTiO3", "ewald/_csrc/README.md",
                   "ewald/assets/fonts/IBMPlexSans-Regular.ttf", "ewald/assets/fonts/IBM-Plex-LICENSE.txt"]

SMOKE = r"""
import sys, ewald
from ewald import import_file
from ewald.examples import path as example
from ewald import modifiers as m
print("ewald", ewald.__version__, "from", ewald.__file__)
assert ewald.__version__ == sys.argv[1], ewald.__version__
try:
    from ewald import _native  # noqa: F401
    native = True
except ImportError:
    native = False
print("native extension:", native)
pipe = import_file(example("Cu_void.lammpstrj"))
pipe.modifiers.append(m.CommonNeighborAnalysis())
data = pipe.compute(0)
print("Cu_void:", pipe.num_frames, "frames,", data.count, "atoms, CNA ok")
if native:
    pipe = import_file(example("Pt_nanoparticle.xyz"))
    pipe.modifiers.append(m.PolyhedralTemplateMatching())
    pipe.compute(0)
    print("PTM ok")
print("SMOKE OK")
"""


def version() -> str:
    tree = ast.parse((ROOT / "src" / "ewald" / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__version__" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("no __version__ in src/ewald/__init__.py")


def changelog_section(ver: str) -> str | None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    m = re.search(rf"^## \[{re.escape(ver)}\][^\n]*\n(.*?)(?=^## \[|^\[[^\]]+\]: )", text, re.M | re.S)
    return m.group(1).strip() if m else None


def check(tag: str | None) -> list[str]:
    errors = []
    ver = version()
    if not SEMVER.match(ver):
        errors.append(f"__version__ {ver!r} is not X.Y.Z (or X.Y.ZaN/bN/rcN)")
    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dynamic = pyproject.get("tool", {}).get("setuptools", {}).get("dynamic", {}).get("version", {})
    if "version" in pyproject["project"] or dynamic.get("attr") != "ewald.__version__":
        errors.append("pyproject.toml must take the version from ewald.__version__ (dynamic), not set its own")
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if not re.search(rf"^## \[{re.escape(ver)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", text, re.M):
        errors.append(f"CHANGELOG.md has no '## [{ver}] - YYYY-MM-DD' section")
    elif not changelog_section(ver):
        errors.append(f"CHANGELOG.md section [{ver}] is empty")
    if not re.search(rf"^\[{re.escape(ver)}\]: \S+", text, re.M):
        errors.append(f"CHANGELOG.md has no '[{ver}]: <url>' link at the bottom")
    if tag and tag != f"v{ver}":
        errors.append(f"tag {tag} does not match __version__ {ver} (expected v{ver})")
    return errors


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def build_and_smoke(dist: Path, keep_venv: bool) -> None:
    ver = version()
    if dist.exists():
        for f in dist.glob(f"ewald-{ver}*"):
            f.unlink()
    run([sys.executable, "-m", "build", "--outdir", dist, ROOT], env={**os.environ, "EWALD_REQUIRE_NATIVE": "1"})
    sdist = dist / f"ewald-{ver}.tar.gz"
    wheels = sorted(dist.glob(f"ewald-{ver}-*.whl"))
    if not sdist.exists() or not wheels:
        raise SystemExit(f"build did not produce ewald-{ver}.tar.gz and a wheel in {dist}")
    run([sys.executable, "-m", "twine", "check", "--strict", sdist, *wheels])
    names = set(zipfile.ZipFile(wheels[0]).namelist())
    missing = [f for f in WHEEL_MUST_HAVE if f not in names]
    if missing:
        raise SystemExit(f"wheel {wheels[0].name} is missing {missing}")
    if not any(n.startswith("ewald/_native") for n in names):
        raise SystemExit(f"wheel {wheels[0].name} has no ewald._native")
    print(f"wheel contents ok ({len(names)} files)")

    env_dir = Path(tempfile.mkdtemp(prefix="ewald-release-venv-"))
    try:
        venv.create(env_dir, with_pip=True)
        py = env_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
        run([py, "-m", "pip", "install", "-q", wheels[0]])
        out = subprocess.run([str(py), "-m", "ewald", "--version"], capture_output=True, text=True, check=True)
        if out.stdout.strip() != f"ewald {ver}":
            raise SystemExit(f"'ewald --version' printed {out.stdout.strip()!r}")
        print(out.stdout.strip())
        # run outside the checkout so the installed wheel is imported, not src/
        run([py, "-c", SMOKE, ver], cwd=env_dir)
    finally:
        if keep_venv:
            print("venv kept at", env_dir)
        else:
            shutil.rmtree(env_dir, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-only", action="store_true", help="version/tag/CHANGELOG checks only")
    ap.add_argument("--notes", action="store_true", help="print the CHANGELOG section of this version")
    ap.add_argument("--tag", help="tag to check against (default: GITHUB_REF when it is refs/tags/...)")
    ap.add_argument("--dist", type=Path, default=ROOT / "dist", help="output folder (default: dist/)")
    ap.add_argument("--keep-venv", action="store_true", help="keep the test venv")
    args = ap.parse_args(argv)

    if args.notes:
        section = changelog_section(version())
        if section is None:
            raise SystemExit(f"CHANGELOG.md has no section for {version()}")
        print(section)
        return
    ref = os.environ.get("GITHUB_REF", "")
    tag = args.tag or (ref[len("refs/tags/"):] if ref.startswith("refs/tags/") else None)
    errors = check(tag)
    for e in errors:
        print("error:", e, file=sys.stderr)
    if errors:
        raise SystemExit(1)
    print(f"version {version()} ok" + (f", tag {tag} ok" if tag else ""))
    if not args.check_only:
        build_and_smoke(args.dist, args.keep_venv)


if __name__ == "__main__":
    main()
