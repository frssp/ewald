"""Regenerate scattering.json from pymatgen's diffraction tables (pymatgen is MIT licensed).

    python src/ewald/tables/make_scattering.py [path/to/pymatgen/analysis/diffraction]

Needs only the two data files of ``pymatgen.analysis.diffraction`` (``atomic_scattering_params.json``
and ``neutron_scattering_length.json``), not an importable pymatgen; without an argument the
installed package is used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SOURCES = {
    "xray": "X-ray atomic scattering factors f(s) = Z - 41.78214 s^2 sum_i a_i exp(-b_i s^2), "
            "s = sin(theta)/lambda in 1/Angstrom; [a_i, b_i] pairs from M. De Graef and M. E. McHenry, "
            "Structure of Materials (Cambridge, 2007), as tabulated by pymatgen",
    "neutron": "Bound coherent neutron scattering lengths in fm, V. F. Sears, Neutron News 3, 26 (1992) "
               "(NIST table), as tabulated by pymatgen",
    "extracted_from": "pymatgen/analysis/diffraction/atomic_scattering_params.json and "
                      "neutron_scattering_length.json (MIT license)",
}


def _folder(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    import pymatgen.analysis.diffraction as d

    return Path(d.__file__).parent


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    folder = _folder(argv[0] if argv else None)
    xray = json.loads((folder / "atomic_scattering_params.json").read_text(encoding="utf-8"))
    neutron = json.loads((folder / "neutron_scattering_length.json").read_text(encoding="utf-8"))
    out = {"_sources": SOURCES,
           "xray": {k: xray[k] for k in sorted(xray)},
           "neutron": {k: float(neutron[k]) for k in sorted(neutron)}}
    path = Path(__file__).with_name("scattering.json")
    path.write_text(json.dumps(out, indent=None, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path} ({len(out['xray'])} X-ray, {len(out['neutron'])} neutron entries)")


if __name__ == "__main__":
    main()
