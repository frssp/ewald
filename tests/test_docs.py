"""The manual (docs/) stays in step with the code: every page is in the nav, every modifier is
documented, links resolve, and the example plugin modifier does what docs/plugins says."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from ewald import Pipeline, StaticSource, manual
from ewald import modifiers as mods
from ewald.backend import Backend

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def nav_pages():
    return re.findall(r":\s*([\w/.-]+\.md)\s*$", (ROOT / "mkdocs.yml").read_text(), re.M)


def manual_pages():
    return sorted(p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*.md")
                  if p.relative_to(DOCS).parts[0] not in ("design", "research"))


def test_every_page_is_in_the_nav_and_exists():
    nav = nav_pages()
    assert sorted(nav) == manual_pages()
    assert all((DOCS / p).exists() for p in manual.PAGES.values())


def test_every_modifier_is_in_the_user_manual():
    text = (DOCS / "user" / "modifiers.md").read_text()
    names = [n for n in mods.__all__ if n not in ("Modifier", "PythonModifier")]
    missing = [n for n in names if f"`{n}" not in text]
    assert not missing, f"add these to docs/user/modifiers.md: {missing}"


def test_relative_links_resolve():
    broken = []
    for page in manual_pages():
        path = DOCS / page
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", path.read_text()):
            if "://" in target:
                continue
            if not (path.parent / target).resolve().exists():
                broken.append(f"{page}: {target}")
    assert not broken, broken


def test_help_urls():
    assert manual.url("user/gui.md") == manual.DEFAULT_BASE + "user/gui.md"
    assert manual.url("reference/api.md").endswith("/blob/main/llms.txt")
    assert manual.url("index.md", "https://example.org/ewald") == "https://example.org/ewald/"
    assert manual.url("plugins/index.md", "https://example.org/ewald/") == "https://example.org/ewald/plugins/"
    assert manual.url("user/gui.md", "https://example.org/") == "https://example.org/user/gui/"
    assert manual.url("user/gui.md", "file:///tmp/site") == "file:///tmp/site/user/gui/index.html"


# ---- the example in docs/plugins/modifiers.md and docs/plugins/testing.md
@pytest.fixture
def local_composition(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples" / "plugins"))
    sys.modules.pop("local_composition", None)
    from local_composition import LocalComposition

    return LocalComposition


@pytest.fixture
def l12():
    atoms = bulk("Cu", "fcc", a=3.75, cubic=True).repeat(3)
    symbols = np.array(atoms.get_chemical_symbols())
    corner = np.all(np.isclose(atoms.get_scaled_positions() * 3 % 1, 0), axis=1)
    symbols[corner] = "Au"
    atoms.set_chemical_symbols(symbols)
    return atoms


def test_example_modifier_plugin(local_composition, l12):
    pipe = Pipeline(StaticSource(l12))
    before = pipe.source_frame(0).positions.copy()
    mod = local_composition(cutoff=3.0, element="Cu")
    pipe.modifiers.append(mod)
    data = pipe.compute(0, device=Backend("test", np))
    frac = data["Local Cu Fraction"]
    assert np.allclose(frac[data.types == "Au"], 1.0)
    assert np.allclose(frac[data.types == "Cu"], 8 / 12)
    assert data.attributes["LocalComposition.mean"] == pytest.approx(frac.mean())
    assert np.array_equal(pipe.source_frame(0).positions, before)
    assert mod.parameters() == {"cutoff": 3.0, "element": "Cu"}
    assert repr(mod) == "LocalComposition(cutoff=3.0, element='Cu')"
