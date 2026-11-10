"""The public API does not change unnoticed (docs/developer/api-stability.md).

``tests/data/api/current.txt`` is the API of this tree and ``released.txt`` that of the last
release, both written by ``python -m ewald api --signatures``.
"""
import re
import subprocess
import sys
from pathlib import Path

import ewald
from ewald import api_surface

ROOT = Path(__file__).resolve().parent.parent
API = ROOT / "tests" / "data" / "api"


def _read(name: str) -> str:
    return (API / name).read_text(encoding="utf-8")


def test_snapshot_is_current():
    """A signature changed, or a public name was added or removed: the snapshot shows it in the PR."""
    old, new = api_surface.parse(_read("current.txt")), api_surface.parse(api_surface.render())
    gone = [old[k] for k in old if k not in new]
    added = [new[k] for k in new if k not in old]
    changed = [f"{old[k]}\n      -> {new[k]}" for k in old if k in new and old[k] != new[k]]
    told = "".join(f"\n  {title}:\n    " + "\n    ".join(rows[:15]) + (f"\n    ... and {len(rows) - 15} more" if len(rows) > 15 else "")
                   for title, rows in (("removed", gone), ("changed", changed), ("added", added)) if rows)
    assert not told, ("the public API differs from tests/data/api/current.txt:" + told + "\n"
                      "If this is intended, run `python -m ewald api --signatures > tests/data/api/current.txt` "
                      "and, for anything removed or changed, add a line to CHANGELOG.md [Unreleased] that names it "
                      "(docs/developer/api-stability.md).")
    assert _read("current.txt") == api_surface.render(), "run `python -m ewald api --signatures > tests/data/api/current.txt`"


def test_what_the_release_promised_is_kept_or_in_the_changelog():
    """Code written for the last release keeps working, or CHANGELOG.md says what it has to change."""
    released = _read("released.txt")
    changelog = api_surface.changelog_above((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
                                            api_surface.version_of(released))
    missing = api_surface.unrecorded(released, api_surface.render(), changelog)
    assert not missing, (
        f"the API of ewald {api_surface.version_of(released)} (tests/data/api/released.txt) changed in a way that breaks "
        "code written for it, and CHANGELOG.md does not name the change:\n  "
        + "\n  ".join(f"{name}: {what}" for name, _parameter, what in missing)
        + "\nKeep the old spelling working, or add a line under [Unreleased] › Changed that names the function "
          "and the parameter (docs/developer/api-stability.md).")


def test_released_snapshot_is_of_a_release():
    version = api_surface.version_of(_read("released.txt"))
    assert version and re.search(rf"^## \[{re.escape(version)}\] - ", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M)
    assert api_surface.version_of(_read("current.txt")) == ewald.__version__


def test_snapshot_text_is_the_same_everywhere():
    """No memory addresses, annotations or values that depend on the machine or the Python version."""
    text = api_surface.render()
    assert text == api_surface.render()
    assert not re.search(r"0x[0-9a-f]{6,}|\bat 0x|typing\.|np\.float|numpy\.", text)
    assert "->" not in text and ": int" not in text
    names = api_surface.public_api()
    for must in ("ewald.load", "ewald.Pipeline.compute", "ewald.Pipeline.modifiers", "ewald.DataCollection.tables",
                 "ewald.verbs.Table", "ewald.views.show", "ewald.modifiers.CoordinationAnalysis",
                 "ewald.plugin_api.find_pairs", "ewald.plugin_api.PLUGIN_API", "render(size=)", "View.bonds",
                 "ewald.display.options['quality']"):
        assert must in names, must
    assert names["ewald.plugin_api.DataCollection"] == "ewald.plugin_api.DataCollection = ewald.DataCollection"
    assert set(names) >= {f"ewald.modifiers.{n}" for n in ewald.modifiers.__all__}


def test_cli_prints_the_snapshot():
    out = subprocess.run([sys.executable, "-m", "ewald", "api", "--signatures"], capture_output=True,
                         encoding="utf-8", check=True)
    assert out.stdout == api_surface.render()


def test_values_do_not_depend_on_the_numpy_version():
    import enum

    import numpy as np

    class Kind(enum.Enum):
        A = 1

    value = api_surface._value
    assert value(np.float64(0.5)) == "0.5" and value(np.int64(3)) == "3" and value(True) == "True"
    assert value((1, "a", None)) == "(1, 'a', None)" and value((1,)) == "(1,)" and value([]) == "[]"
    assert value({"b": 1.0}) == "{'b': 1.0}" and value(frozenset({"b", "a"})) == "frozenset({'a', 'b'})"
    assert value(np.zeros(3)) == "<ndarray>" and value(object()) == "<object>" and value(Kind.A) == "<Kind.A>"
    assert value(lambda: 0) == "<function>"


OLD = """# The public API of ewald 1.0.0, one name per line (python -m ewald api --signatures).
ewald.rdf(pipe, cutoff=6.0, bins=200, frame=0)
ewald.msd(pipe, frame=0, time_per_frame=None, **kwargs)
ewald.old(a)
ewald.Pipeline.frames(start=0, stop=None, step=1)
ewald.show(source, table=None, panel=None, save=None)
ewald.modifiers.Thing(cutoff=3.2, mode=<Mode.A>, *, quiet=False)
ewald.plugin_api.PLUGIN_API = 1
ewald.Pipeline.num_frames
render(size=)
View.bonds = 'auto'
ewald.display.options['quality']
"""


def _breaking(new_lines: dict) -> list:
    """Replace or drop (None) lines of OLD by their name and return the breaking changes."""
    rows = []
    for line in OLD.splitlines():
        key = api_surface._key(line)
        if key in new_lines:
            if new_lines[key] is not None:
                rows.append(new_lines[key])
        else:
            rows.append(line)
    rows += [v for k, v in new_lines.items() if v is not None and k not in api_surface.parse(OLD)]
    return api_surface.breaking(OLD, "\n".join(rows) + "\n")


def test_additions_break_nothing():
    assert _breaking({}) == []
    assert _breaking({"ewald.rdf": "ewald.rdf(pipe, cutoff=6.0, bins=200, frame=0, partial=False)"}) == []
    assert _breaking({"ewald.rdf": "ewald.rdf(pipe, cutoff=6.0, bins=200, frame=0, *, partial=False, **kwargs)"}) == []
    assert _breaking({"ewald.new": "ewald.new(x)", "ewald.Pipeline.fresh": "ewald.Pipeline.fresh"}) == []
    assert _breaking({"ewald.old": "ewald.old(a=None)"}) == []                      # a default more


def test_what_breaks_is_named_with_its_parameter():
    def one(new_lines):
        found = _breaking(new_lines)
        assert len(found) == 1, found
        return found[0]

    assert one({"ewald.old": None}) == ("ewald.old", None, "removed")
    assert one({"ewald.rdf": "ewald.rdf(pipe, cutoff=6.0, frame=0)"})[:2] == ("ewald.rdf", "bins")
    assert "default of `cutoff` changed from 6.0 to 5.0" in one({"ewald.rdf": "ewald.rdf(pipe, cutoff=5.0, bins=200, frame=0)"})[2]
    assert "required now" in one({"ewald.rdf": "ewald.rdf(pipe, cutoff, bins=200, frame=0)"})[2]
    assert "new required parameter `kind`" in one({"ewald.old": "ewald.old(a, kind)"})[2]
    assert "keyword-only now" in one({"ewald.rdf": "ewald.rdf(pipe, cutoff=6.0, bins=200, *, frame=0)"})[2]
    name, parameter, what = one({"ewald.show": "ewald.show(source, table=None, layout=None, panel=None, save=None)"})
    assert (name, parameter) == ("ewald.show", "layout") and "moves the positional parameters after it" in what
    moved = _breaking({"ewald.show": "ewald.show(source, table=None, save=None, panel=None)"})
    assert {p for _n, p, _w in moved} == {"panel", "save"} and all("moved from position" in w for _n, _p, w in moved)
    assert one({"ewald.msd": "ewald.msd(pipe, frame=0, time_per_frame=None)"})[1] == "kwargs"
    assert one({"ewald.modifiers.Thing": "ewald.modifiers.Thing(cutoff=3.2, mode=<Mode.B>, *, quiet=False)"})[1] == "mode"
    assert one({"ewald.plugin_api.PLUGIN_API": "ewald.plugin_api.PLUGIN_API = 2"})[0] == "ewald.plugin_api.PLUGIN_API"
    assert one({"render(size=)": None}) == ("render(size=)", None, "removed")
    assert one({"View.bonds": "View.bonds = True"})[0] == "View.bonds"
    assert one({"ewald.Pipeline.num_frames": "ewald.Pipeline.num_frames(exact=False)"})[0] == "ewald.Pipeline.num_frames"
    # a parameter that **kwargs still takes is not gone, but it no longer works by position
    assert "keyword-only now" in one({"ewald.rdf": "ewald.rdf(pipe, cutoff=6.0, bins=200, **kwargs)"})[2]


def test_the_changelog_has_to_name_the_function_and_the_parameter():
    new = OLD.replace("ewald.rdf(pipe, cutoff=6.0,", "ewald.rdf(pipe, cutoff=5.0,").replace("render(size=)\n", "") \
             .replace("ewald.display.options['quality']\n", "")
    assert [n for n, _p, _w in api_surface.unrecorded(OLD, new, "")] == ["ewald.rdf", "render(size=)", "ewald.display.options['quality']"]
    assert len(api_surface.unrecorded(OLD, new, "- `rdf` is faster.")) == 3                   # the parameter is not named
    assert len(api_surface.unrecorded(OLD, new, "- the cutoff of `xrdf` changed.")) == 3      # nor the function
    log = ("- `rdf(pipe)`: the default `cutoff` is 5 A.\n- `render()` no longer takes `size`.\n"
           "- `ewald.display.options` has no `quality` any more.")
    assert api_surface.unrecorded(OLD, new, log) == []


def test_only_the_changelog_above_the_release_counts():
    log = "# Changelog\n\n## [Unreleased]\n\n- new\n\n## [1.1.0] - 2027-01-01\n\n- mid\n\n## [1.0.0] - 2026-12-01\n\n- old\n"
    above = api_surface.changelog_above(log, "1.0.0")
    assert "new" in above and "mid" in above and "old" not in above
    assert api_surface.changelog_above(log, "0.9.0") == log and api_surface.changelog_above(log, None) == log


def test_release_check_wants_the_released_snapshot(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("release_check", ROOT / "packaging" / "release_check.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    assert rc.check(None) == []
    monkeypatch.setattr(rc, "version", lambda: "9.9.9")
    assert any("released.txt is not the API of 9.9.9" in e for e in rc.check(None))
