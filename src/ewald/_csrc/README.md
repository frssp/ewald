# Native kernels (`ewald._native`)

Built by `setup.py` with pybind11 when a C++17 compiler is available
(`pip install -e .`). Without it the package still installs; CNA then uses the
array kernel and PTM/Voronoi raise an ImportError.

| Path | Origin | License |
| --- | --- | --- |
| `native.cpp`, `cna.cpp` | Ported from OVITO 3.16 (`PTMAlgorithm.cpp`, `PolyhedralTemplateMatchingModifier.cpp`, `VoronoiAnalysisModifier.cpp`, `CommonNeighborAnalysisModifier.cpp`), Copyright 2026 OVITO GmbH | GPLv3 or MIT; used under MIT, see `LICENSE.OVITO-MIT.txt` |
| `ptm/` | Polyhedral Template Matching library by P. M. Larsen, as vendored in OVITO 3.16 (unchanged) | MIT, see `ptm/LICENSE` |
| `voro++/` | voro++ `cell.*`, `common.*`, `config.hh` by C. H. Rycroft (LBNL), as vendored in OVITO 3.16 (unchanged) | BSD-3 style, see `voro++/LICENSE` |
