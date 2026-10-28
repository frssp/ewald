"""An OPTIMADE structure resource (JSON:API object) as an EWALD frame."""
from __future__ import annotations

import warnings

import numpy as np

from ..data import Cell, DataCollection

#: attributes copied into ``data.attributes["OPTIMADE"]`` when the server gives them
_KEEP = ("chemical_formula_reduced", "chemical_formula_descriptive", "chemical_formula_anonymous",
         "nelements", "nsites", "last_modified", "immutable_id", "structure_features")
VACUUM = 10.0     # Angstrom of box around a non-periodic direction


def _species(attrs) -> dict:
    """species name -> (symbol, occupancy): the symbol of largest concentration; a site that is
    mostly vacancy keeps its next element so that it can still be drawn."""
    out = {}
    for sp in attrs.get("species") or ():
        symbols = list(sp.get("chemical_symbols") or [sp.get("name") or "X"])
        conc = list(sp.get("concentration") or [1.0] * len(symbols))
        pairs = sorted(zip(conc, symbols), reverse=True)
        real = [(c, s) for c, s in pairs if s != "vacancy"] or [(1.0, "X")]
        c, s = real[0]
        out[sp.get("name")] = (s, float(c) if c is not None else 1.0)
    return out


def _cell(attrs, positions) -> Cell:
    dims = attrs.get("dimension_types")
    lattice = attrs.get("lattice_vectors")
    pbc = np.array([bool(d) for d in dims], bool) if dims is not None else np.array([lattice is not None] * 3)
    if lattice is None:
        lattice = [[None] * 3] * 3
    matrix = np.zeros((3, 3))
    lo, hi = (positions.min(0), positions.max(0)) if len(positions) else (np.zeros(3), np.zeros(3))
    origin = np.zeros(3)
    for i, vec in enumerate(lattice):
        if vec is None or any(x is None for x in vec) or not np.any(vec):
            # no vector (molecule, slab, wire): an axis box around the atoms
            matrix[i, i] = (hi[i] - lo[i]) + 2 * VACUUM
            origin[i] = lo[i] - VACUUM
        else:
            matrix[i] = np.asarray(vec, float)
    return Cell(matrix, origin=origin, pbc=pbc)


def to_data(resource: dict, provider: str = "", uri: str = "", url: str = "") -> DataCollection:
    attrs = resource.get("attributes") or {}
    positions = attrs.get("cartesian_site_positions")
    sites = attrs.get("species_at_sites")
    if positions is None or sites is None:
        raise ValueError(f"{uri or resource.get('id')}: the server gave no cartesian_site_positions/species_at_sites")
    positions = np.asarray([[np.nan if x is None else x for x in p] for p in positions], float).reshape(-1, 3)
    species = _species(attrs)
    symbols, occupancy = [], []
    for name in sites:
        s, c = species.get(name, (name if name else "X", 1.0))
        symbols.append(s)
        occupancy.append(c)
    data = DataCollection(positions, symbols, _cell(attrs, positions))
    occupancy = np.asarray(occupancy, float)
    if np.any(occupancy < 0.999):
        data.particles["Occupancy"] = occupancy
        warnings.warn(f"{uri}: partially occupied sites; each shows its majority element "
                      "(particle property Occupancy)", stacklevel=3)
    if any(species.get(name, (name,))[0] != name for name in set(sites)):   # Fe1/Fe2, mixed sites
        data.particles["Species"] = np.array(list(sites))
    if attrs.get("assemblies"):
        warnings.warn(f"{uri}: alternative site assemblies are not shown; all sites are drawn", stacklevel=3)
    info = {"provider": provider, "id": resource.get("id"), "uri": uri, "url": url}
    info.update({k: attrs[k] for k in _KEEP if attrs.get(k) is not None})
    data.attributes["OPTIMADE"] = info
    if attrs.get("chemical_formula_reduced"):
        data.attributes.setdefault("Formula", attrs["chemical_formula_reduced"])
    return data

