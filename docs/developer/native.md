# Native code

`ewald._native` is a pybind11 extension built by `setup.py` from `src/ewald/_csrc/`. It holds
C++ ports of OVITO 3.16's common neighbor analysis, polyhedral template matching, Voronoi
analysis and minimum-weight-matching centrosymmetry, plus the libraries OVITO vendors for them:

| Path | Origin | License |
|---|---|---|
| `native.cpp`, `cna.cpp` | ported from OVITO 3.16 | OVITO's GPLv3-or-MIT, used under MIT (`LICENSE.OVITO-MIT.txt`) |
| `ptm/` | Polyhedral Template Matching by P. M. Larsen, unchanged | MIT |
| `voro++/` | voro++ by C. H. Rycroft, unchanged | BSD-3 style |
| `mwm_csp/` | matching CSP by P. M. Larsen and D. Pereira, unchanged | MIT |

Ported code keeps its notice. `_csrc/README.md` is the authoritative list.

## Building

`pip install -e .` compiles the extension with the system C++17 compiler. It is **not** rebuilt
automatically: after editing anything in `_csrc/`, run `pip install -e .` again. A new source
file has to be added to the `sources` list in `setup.py`.

## Python side

The modifiers in `modifiers/structure.py`, `modifiers/voronoi.py` and `modifiers/local_order.py`
compute neighbors in Python (`find_pairs`, `_knn.py`), call into `_native` for the per-atom
classification, and assemble OVITO-named properties, attributes and colors. With
`device="cuda"`, the CNA signatures run as CuPy kernels instead.

## Testing against OVITO

Ported analyses are checked against OVITO's own results on a set of test structures, stored in
`tests/data/ovito_reference.npz` and `ovito_cluster_reference.npz`. Regenerate them with
`python tests/data/make_ovito_reference.py` (needs `pip install ovito`), never by hand. When you
port another OVITO analysis, add its cases to that script and compare property by property,
including structure type ids and attribute names. Histogram, binning, reduction, bond
distribution, ring and Steinhardt references live in `tests/data/analysis_reference.npz`,
written by `python tests/data/make_analysis_reference.py` (needs `pip install ovito
freud-analysis`).
