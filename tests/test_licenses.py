"""License notices of ported and vendored code stay in place and ship with the package."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "ewald"
CSRC = PKG / "_csrc"


def test_ported_ovito_sources_carry_the_notice():
    for path in [*CSRC.glob("*.cpp"), *PKG.rglob("*.py")]:
        text = path.read_text(encoding="utf-8")
        if "OVITO GmbH" not in text or path.name == "test_licenses.py":
            continue
        if path.suffix == ".cpp":
            assert "MIT License" in text and "LICENSE.OVITO-MIT.txt" in text, path
        else:
            assert "MIT option" in text and "LICENSE.OVITO-MIT.txt" in text, path


def test_every_vendored_library_has_a_license_file():
    for lib in (d for d in CSRC.iterdir() if d.is_dir() and d.name != "__pycache__"):
        assert (lib / "LICENSE").is_file(), lib


def test_license_files_are_package_data():
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["ewald"]
    shipped = {p for pattern in data for p in PKG.glob(pattern)}
    wanted = {CSRC / "README.md", CSRC / "LICENSE.OVITO-MIT.txt", *CSRC.glob("*/LICENSE")}
    assert wanted <= shipped
