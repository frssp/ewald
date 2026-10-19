"""The release metadata stays consistent: one version source, a CHANGELOG section for it, and the
checks packaging/release_check.py runs before a tag (docs/developer/release.md)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import ewald

ROOT = Path(__file__).resolve().parents[1]


def release_check():
    spec = importlib.util.spec_from_file_location("release_check", ROOT / "packaging" / "release_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_version_and_changelog_agree():
    rc = release_check()
    assert rc.version() == ewald.__version__
    assert rc.check(None) == []
    assert "Unreleased" not in rc.changelog_section(ewald.__version__)


def test_tag_must_match_version():
    rc = release_check()
    assert rc.check(f"v{ewald.__version__}") == []
    errors = rc.check("v99.0.0")
    assert len(errors) == 1 and "does not match" in errors[0]


def test_cli_version():
    out = subprocess.run([sys.executable, "-m", "ewald", "--version"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == f"ewald {ewald.__version__}"


def test_release_notes_are_the_changelog_section():
    out = subprocess.run([sys.executable, str(ROOT / "packaging" / "release_check.py"), "--notes"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == release_check().changelog_section(ewald.__version__)
    assert out.stdout.strip() and "## [" not in out.stdout
