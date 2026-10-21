"""Regenerate radii.json from pymatgen's periodic table (pymatgen is MIT licensed).

    python src/ewald/tables/make_radii.py [path/to/pymatgen/core]

Needs only the data files of pymatgen-core (``periodic_table.json.gz`` and ``icsd_bv.yaml``),
not an importable pymatgen; without an argument the installed package is used. Covalent and
van der Waals radii are not stored here: they come from ASE at runtime.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

SOURCES = {
    "atomic": "J. C. Slater, J. Chem. Phys. 41, 3199 (1964), empirical atomic radii",
    "atomic_calculated": "E. Clementi, D. L. Raimondi, W. P. Reinhardt, J. Chem. Phys. 47, 1300 (1967), "
                         "calculated atomic radii",
    "shannon": "R. D. Shannon, Acta Cryst. A32, 751 (1976), effective ionic and crystal radii by "
               "oxidation state, coordination number and spin; [crystal, ionic] in Angstrom",
    "oxidation_occurrence": "Oxidation-state counts in the ICSD (pymatgen icsd_bv.yaml, 'occurrence'), "
                            "used as priors for charge-balanced guesses",
    "extracted_from": "pymatgen-core periodic_table.json.gz and icsd_bv.yaml (MIT license)",
}


def _pymatgen_core(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    import pymatgen.core

    return Path(pymatgen.core.__file__).parent


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    core = _pymatgen_core(argv[0] if argv else None)
    table = json.loads(gzip.open(core / "periodic_table.json.gz").read())
    atomic, calculated, shannon = {}, {}, {}
    for sym, row in table.items():
        if isinstance(row.get("Atomic radius"), (int, float)):
            atomic[sym] = row["Atomic radius"]
        if isinstance(row.get("Atomic radius calculated"), (int, float)):
            calculated[sym] = row["Atomic radius calculated"]
        if isinstance(row.get("Shannon radii"), dict):
            shannon[sym] = {oxi: {cn: {spin: [v["crystal_radius"], v["ionic_radius"]] for spin, v in spins.items()}
                                  for cn, spins in cns.items()} for oxi, cns in row["Shannon radii"].items()}
    # icsd_bv.yaml is a two-level mapping; parse the 'occurrence' block without needing PyYAML
    occurrence, block = {}, None
    for line in (core / "icsd_bv.yaml").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(" "):
            block = line.rstrip(":")
        elif block == "occurrence" and (m := re.match(r"\s+([A-Z][a-z]?)(\d*)([+-]):\s*(\d+)", line)):
            sym, n, sign, count = m.groups()
            oxi = int(n or 1) * (1 if sign == "+" else -1)
            if int(count):
                occurrence.setdefault(sym, {})[str(oxi)] = int(count)
    out = {"_sources": SOURCES, "atomic": atomic, "atomic_calculated": calculated, "shannon": shannon,
           "oxidation_occurrence": occurrence}
    path = Path(__file__).with_name("radii.json")
    path.write_text(json.dumps(out, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path}: {len(atomic)} atomic, {len(calculated)} calculated, {len(shannon)} Shannon elements, "
          f"{len(occurrence)} with ICSD oxidation states")


if __name__ == "__main__":
    main()
