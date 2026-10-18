"""Optional bridge to the `ovito` Python module (``pip install ovito``).

``OvitoModifier`` runs any OVITO modifier inside an ewald pipeline, which gives
access to analyses ewald does not implement itself.  ``DislocationAnalysis``
(DXA) is built on it.  The `ovito` package is not a dependency of ewald; it is
imported only when one of these modifiers is evaluated.
"""
from __future__ import annotations

import numpy as np

from ..pipeline import Modifier

# Properties not copied back from OVITO (ewald owns them or they are OVITO-internal).
_SKIP_BACK = {"Position", "Particle Type"}


def _ovito():
    try:
        import ovito  # noqa: F401
    except ImportError as exc:
        raise ImportError("this modifier needs the optional `ovito` package: pip install ovito") from exc
    import ovito.data
    import ovito.pipeline

    return ovito


def to_ovito(data):
    """Convert an ewald DataCollection into an ``ovito.data.DataCollection``."""
    ovito = _ovito()
    from ovito.data import DataCollection, Particles, SimulationCell

    dc = DataCollection()
    cell = SimulationCell()
    m = np.zeros((3, 4))
    if data.cell.is_valid:
        m[:, :3] = data.cell.matrix.T
        m[:, 3] = data.cell.origin
        pbc = tuple(bool(b) for b in data.cell.pbc)
    else:
        lo, hi = data.positions.min(0) - 1.0, data.positions.max(0) + 1.0
        m[:, :3] = np.diag(hi - lo)
        m[:, 3] = lo
        pbc = (False, False, False)
    cell[...] = m
    cell.pbc = pbc
    dc.objects.append(cell)
    particles = Particles(count=data.count)
    particles.create_property("Position", data=data.positions)
    names = list(dict.fromkeys(data.types))
    ptype = particles.create_property("Particle Type", data=np.searchsorted(sorted(names), data.types) + 1)
    for i, name in enumerate(sorted(names)):
        ptype.types.append(ovito.data.ParticleType(id=i + 1, name=str(name)))
    for name, arr in data.particles.items():
        if name in ("Position", "Particle Type") or arr.dtype.kind not in "biuf":
            continue
        try:
            particles.create_property(name, data=np.asarray(arr))
        except Exception:  # standard properties with a different shape or dtype, e.g. 'Color' as int
            pass
    dc.objects.append(particles)
    return dc


def _table_to_dict(table):
    out = {}
    if table.x is not None:
        out[table.x.name] = np.array(table.x)
    for key in table.keys():
        out[key] = np.array(table[key])
    return out


def from_ovito(dc, data):
    """Copy particle properties, attributes and tables from an OVITO result into ewald ``data``."""
    p = dc.particles
    if p is not None and p.count == data.count:
        for name in p.keys():
            if name not in _SKIP_BACK:
                data.particles[name] = np.array(p[name])
    for key, val in dc.attributes.items():
        if isinstance(val, (int, float, str, bool, np.integer, np.floating)):
            data.attributes[key] = val.item() if isinstance(val, np.generic) else val
    for key, table in dc.tables.items():
        try:
            data.tables[key] = _table_to_dict(table)
        except Exception:
            pass
    return data


class OvitoModifier(Modifier):
    """Run an OVITO modifier (``ovito.modifiers.*``) on the current frame.

    >>> from ovito.modifiers import CentroSymmetryModifier
    >>> pipe.modifiers.append(OvitoModifier(CentroSymmetryModifier(num_neighbors=12)))

    Modifiers that need a reference configuration get one from this pipeline
    when ``reference_frame`` is given.  Particle count must stay unchanged.
    """

    def __init__(self, modifier, reference_frame: int | None = None):
        self.modifier = modifier
        self.reference_frame = reference_frame

    @property
    def name(self) -> str:
        return f"Ovito{type(self.modifier).__name__}"

    def _run(self, data, ctx):
        _ovito()
        from ovito.pipeline import Pipeline, StaticSource

        if self.reference_frame is not None and hasattr(self.modifier, "reference"):
            self.modifier.reference = StaticSource(data=to_ovito(ctx.source_frame(self.reference_frame)))
        pipe = Pipeline(source=StaticSource(data=to_ovito(data)))
        pipe.modifiers.append(self.modifier)
        return pipe.compute()

    def modify(self, data, ctx):
        from_ovito(self._run(data, ctx), data)


class DislocationAnalysis(OvitoModifier):
    """Dislocation extraction algorithm (DXA) via the `ovito` package.

    ``input_crystal_structure``: 'FCC' (default), 'HCP', 'BCC', 'CubicDiamond'
    or 'HexagonalDiamond'.  Writes 'Structure Type' (and 'Color'), the
    ``DislocationAnalysis.*`` attributes (total line length, counts per Burgers
    vector family, ...) and the table ``tables['dislocations']`` with one row
    per dislocation segment: 'Segment Identifier', 'Length', 'Burgers Vector'
    (crystal frame), 'Spatial Burgers Vector', 'Points' (array of (M, 3)
    polylines) and 'Is Loop'.
    """

    def __init__(self, input_crystal_structure: str = "FCC", only_perfect_dislocations: bool = False,
                 circuit_stretchability: int = 9, trial_circuit_length: int = 14,
                 line_smoothing_level: int = 1, line_point_separation: float = 2.5, **kwargs):
        self.input_crystal_structure = input_crystal_structure
        self.only_perfect_dislocations = only_perfect_dislocations
        self.circuit_stretchability = circuit_stretchability
        self.trial_circuit_length = trial_circuit_length
        self.line_smoothing_level = line_smoothing_level
        self.line_point_separation = line_point_separation
        self.extra = kwargs
        self.reference_frame = None

    @property
    def modifier(self):
        _ovito()
        from ovito.modifiers import DislocationAnalysisModifier as DXA

        lattice = getattr(DXA.Lattice, self.input_crystal_structure)
        return DXA(input_crystal_structure=lattice, only_perfect_dislocations=self.only_perfect_dislocations,
                   circuit_stretchability=self.circuit_stretchability,
                   trial_circuit_length=self.trial_circuit_length, line_smoothing_level=self.line_smoothing_level,
                   line_point_separation=self.line_point_separation, **self.extra)

    @modifier.setter
    def modifier(self, value):  # set by OvitoModifier.__init__ only
        pass

    @property
    def name(self) -> str:
        return "DislocationAnalysis"

    def modify(self, data, ctx):
        result = self._run(data, ctx)
        from_ovito(result, data)
        segs = result.dislocations.segments if result.dislocations is not None else []
        data.tables["dislocations"] = {
            "Segment Identifier": np.array([s.id for s in segs], dtype=np.int64),
            "Length": np.array([s.length for s in segs], dtype=float),
            "Burgers Vector": np.array([s.true_burgers_vector for s in segs], dtype=float).reshape(-1, 3),
            "Spatial Burgers Vector": np.array([s.spatial_burgers_vector for s in segs], dtype=float).reshape(-1, 3),
            "Is Loop": np.array([s.is_loop for s in segs], dtype=bool),
            "Points": np.array([np.array(s.points) for s in segs] + [None], dtype=object)[:-1],
        }
