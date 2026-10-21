"""Element names in the example notebooks and paper scripts match the structures they open.

Every element named in a bond table, color, radius, oxidation state, transparency or type selection of
``examples/**/*.py`` must be an element of the structure the script loads: a shipped example file
(``examples.path(...)``), or for a paper script the formulas of its entry in
``ewald.examples.papers.PAPERS``. A copied cell (Mg-B bonds in the TiOF2 script) fails here.
"""
import ast
import re
from pathlib import Path

import numpy as np
import pytest
from ase.data import chemical_symbols

from ewald import examples, import_file
from ewald.examples.papers import PAPERS

ROOT = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = sorted(p for p in ROOT.rglob("*.py") if "plugins" not in p.parts)
ELEMENTS = set(chemical_symbols[1:])
TYPE_KEYS = {"colors", "type_colors", "radii", "type_radii", "oxidation_states", "transparency",
             "type_transparency"}   # keyword arguments: {type: value}
TYPE_LISTS = {"pairs", "bond_pairs", "center_types", "types", "type_a", "type_b"}   # element names inside


def _strings(node):
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def named_elements(path: Path) -> dict[str, int]:
    """Element names the script's calls set something for, with the line of their first use."""
    found = {}
    for call in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(call, ast.Call):
            continue
        for kw in call.keywords:
            if kw.arg in TYPE_KEYS and isinstance(kw.value, ast.Dict):
                names = [s for k in kw.value.keys if k is not None for s in _strings(k)]
            elif kw.arg in TYPE_LISTS:
                names = _strings(kw.value)
            else:
                continue
            for s in names:
                if s in ELEMENTS:
                    found.setdefault(s, kw.value.lineno)
    return found


def _formula_elements(text: str) -> set[str]:
    """Element symbols of the chemical formulas in a paper's title and description (Li6PS5Cl, Nb27 O54 F27)."""
    out = set()
    for word in re.findall(r"[A-Za-z0-9]+", text):
        parts = re.findall(r"[A-Z][a-z]?\d*", word)
        if parts and "".join(parts) == word:
            syms = [re.sub(r"\d", "", p) for p in parts]
            if all(s in ELEMENTS for s in syms):
                out.update(syms)
    return out


def structure_elements(path: Path) -> set[str] | None:
    """The elements of what the script opens, or None when it opens nothing this test knows."""
    paper = next((p for p in PAPERS if p.script == path.name), None)
    if paper is not None:
        return _formula_elements(f"{paper.title} {paper.description}")
    names = re.findall(r"examples\.path\(\s*[\"']([^\"']+)[\"']", path.read_text(encoding="utf-8"))
    if not names:
        return None
    out = set()
    for name in names:
        data = import_file(str(examples.path(name))).compute(0)
        out.update(str(t) for t in np.unique(np.asarray(data.types)))
    return out


def test_formula_elements():
    assert _formula_elements("MgB2 phonons from VASP finite displacements") == {"Mg", "B"}
    assert _formula_elements("Nb27 O54 F27, Li6PS5Cl and TiOF2") == {"Nb", "O", "F", "Li", "P", "S", "Cl", "Ti"}


def test_named_elements_finds_bond_tables_and_colors(tmp_path):
    script = tmp_path / "s.py"
    script.write_text('view(pipe, colors={"Mg": "red"})\nCreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.2)])\n'
                      'paint(data, data.types == "Ti", "gold")\n')
    assert set(named_elements(script)) == {"Mg", "Ti", "O"}


@pytest.mark.parametrize("path", SCRIPTS, ids=[str(p.relative_to(ROOT)) for p in SCRIPTS])
def test_example_names_only_its_own_elements(path):
    named = named_elements(path)
    if not named:
        return
    have = structure_elements(path)
    assert have is not None, f"{path.name} names elements {sorted(named)} but opens no file this test knows"
    stray = {s: line for s, line in named.items() if s not in have}
    assert not stray, f"{path.relative_to(ROOT)}: {', '.join(f'{s} (line {n})' for s, n in stray.items())} " \
                      f"not in its structure ({', '.join(sorted(have))})"
