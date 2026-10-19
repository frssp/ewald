"""Regenerate colors.json (element color schemes) from pymatgen (MIT licensed).

    python src/ewald/tables/make_colors.py [path/to/pymatgen/vis/ElementColorSchemes.yaml]

Needs only that data file, not an importable pymatgen; without an argument the installed
package is used. Jmol and CPK colors are not stored here: they come from ASE at runtime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SOURCES = {
    "VESTA": "K. Momma and F. Izumi, J. Appl. Crystallogr. 44, 1272 (2011), VESTA default element colors",
    "extracted_from": "pymatgen vis/ElementColorSchemes.yaml (MIT license)",
}


def _yaml_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    import pymatgen.vis

    return Path(pymatgen.vis.__file__).parent / "ElementColorSchemes.yaml"


def main(argv=None):
    import yaml

    argv = sys.argv[1:] if argv is None else argv
    schemes = yaml.safe_load(_yaml_path(argv[0] if argv else None).read_text())
    out = {"sources": SOURCES,
           "VESTA": {el: [round(c / 255, 4) for c in rgb] for el, rgb in sorted(schemes["VESTA"].items())}}
    target = Path(__file__).with_name("colors.json")
    rows = ",\n".join(f"  {json.dumps(el)}: {json.dumps(rgb)}" for el, rgb in out["VESTA"].items())
    text = '{\n "sources": ' + json.dumps(SOURCES, indent=2).replace("\n", "\n ") + ',\n "VESTA": {\n' + rows + "\n }\n}\n"
    target.write_text(text)
    print(f"wrote {target} ({len(out['VESTA'])} elements)")


if __name__ == "__main__":
    main()
