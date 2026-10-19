# Native kernels (`ewald._native`)

Built by `setup.py` with pybind11 (`pip install -e .`); a C++17 compiler is
required, and installation fails without one.

| Path | Origin | License |
| --- | --- | --- |
| `native.cpp`, `cna.cpp` | Ported from OVITO 3.16 (`PTMAlgorithm.cpp`, `PolyhedralTemplateMatchingModifier.cpp`, `VoronoiAnalysisModifier.cpp`, `CommonNeighborAnalysisModifier.cpp`), Copyright 2026 OVITO GmbH | GPLv3 or MIT; used under MIT, see `LICENSE.OVITO-MIT.txt` |
| `ptm/` | Polyhedral Template Matching library by P. M. Larsen, as vendored in OVITO 3.16 (unchanged) | MIT, see `ptm/LICENSE` |
| `voro++/` | voro++ `cell.*`, `common.*`, `config.hh` by C. H. Rycroft (LBNL), as vendored in OVITO 3.16 (unchanged) | BSD-3 style, see `voro++/LICENSE` |
| `mwm_csp/` | Minimum-weight-matching centrosymmetry by P. M. Larsen and the matching solver by D. Pereira, as vendored in OVITO 3.16 (unchanged) | MIT, see `mwm_csp/LICENSE` (copied from the file headers) |

The license files and this README are installed with the package (`package-data` in
`pyproject.toml`), since the compiled `_native` module contains this code.
