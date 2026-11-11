"""Units as labels: what the numbers of a table, an attribute or an axis are in.

ewald never converts the numbers it reads (D105): a LAMMPS ``real`` dump keeps its kcal/mol, a
``nano`` one its nm. What changes is that results say what they are in. A frame's unit system is
its ``Units`` attribute: the LAMMPS style a dump names in ``ITEM: UNITS`` or that
``load(path, units="real")`` gives; without one, lengths are Angstrom and energies eV, as for
VASP and every other format ewald reads.

Units are stored only where they cannot be told from a column's name: ``data.units`` maps
``(table, column)`` and attribute names to a unit, which the trajectory steps fill for their
time axes ("ps", or "frames" when the time between frames is not known) and diffusion
coefficients. Everything else follows from the name (:data:`QUANTITY`, the columns
:data:`ewald.plotstyle.LABELS` knows) and the unit system: :func:`table_units` gives both,
:attr:`ewald.verbs.Table.units` holds them, and the plots label their axes with them.

Unit strings use matplotlib's mathtext for powers (``"Å$^2$/ps"``), as the axis labels do.
"""
from __future__ import annotations

#: The LAMMPS unit styles (https://docs.lammps.org/units.html) by quantity; ``None`` is what
#: ewald assumes without a style (Angstrom, eV, ps for the time between frames, VASP's units).
SYSTEMS: dict = {
    None: {"length": "Å", "energy": "eV", "time": "ps", "velocity": "Å/ps", "charge": "e"},
    "metal": {"length": "Å", "energy": "eV", "time": "ps", "velocity": "Å/ps", "charge": "e"},
    "real": {"length": "Å", "energy": "kcal/mol", "time": "fs", "velocity": "Å/fs", "charge": "e"},
    "si": {"length": "m", "energy": "J", "time": "s", "velocity": "m/s", "charge": "C"},
    "cgs": {"length": "cm", "energy": "erg", "time": "s", "velocity": "cm/s", "charge": "statC"},
    "electron": {"length": "Bohr", "energy": "Hartree", "time": "fs", "velocity": "Bohr/a.t.u.", "charge": "e"},
    "micro": {"length": "µm", "energy": "pg µm$^2$/µs$^2$", "time": "µs", "velocity": "µm/µs", "charge": "pC"},
    "nano": {"length": "nm", "energy": "ag nm$^2$/ns$^2$", "time": "ns", "velocity": "nm/ns", "charge": "e"},
    "lj": {"length": "σ", "energy": "ε", "time": "τ", "velocity": "σ/τ", "charge": "q*"},
}
STYLES = tuple(s for s in SYSTEMS if s is not None)

#: The quantity of each column name :data:`ewald.plotstyle.LABELS` gives a unit (per-type and
#: per-pair columns such as "MSD Cu" follow their base name).
QUANTITY = {
    "r": "length", "Pair separation": "length", "q": "1/length", "d": "length", "Bond Length": "length",
    "MSD": "length^2", "MSD fit": "length^2", "Charge MSD": "charge^2 length^2", "Charge MSD fit": "charge^2 length^2",
    "MQD": "length^4", "Reaction Coordinate": "length", "Energy": "energy",
    "Displacement Magnitude": "length", "CSP": "length^2",
}

#: what a time axis is in when the time between frames is not known: the frame lag
FRAMES = "frames"


def check_style(units) -> str | None:
    """``units`` as a LAMMPS style name (lower case), None for None; ValueError for anything else."""
    if units is None:
        return None
    style = str(units).strip().lower()
    if style not in STYLES:
        raise ValueError(f"units={units!r} is not a LAMMPS unit style; use one of {', '.join(STYLES)} "
                         "(numbers are not converted: it says what the file's numbers are in)")
    return style


def system(data) -> str | None:
    """The unit system of a frame: its ``Units`` attribute when that names a LAMMPS style, else None."""
    attributes = getattr(data, "attributes", None) or {}
    style = str(attributes.get("Units", "")).strip().lower()
    return style if style in STYLES else None


def unit(quantity: str, style: str | None = None) -> str:
    """The unit of ``quantity`` ("length", "length^2", "1/length", "energy", "time",
    "charge^2 length^2", ...) in unit system ``style``: ``unit("length^2")`` is ``"Å$^2$"``,
    ``unit("energy", "real")`` is ``"kcal/mol"``."""
    names = SYSTEMS.get(style, SYSTEMS[None])
    parts = []
    for part in quantity.split():
        inverse = part.startswith("1/")
        base, _, power = part.removeprefix("1/").partition("^")
        u = names[base]
        if power or inverse:
            exponent = ("-" if inverse else "") + (power or "1")
            u = (f"({u})" if " " in u or "/" in u else u) + (f"$^{{{exponent}}}$" if len(exponent) > 1 else f"$^{exponent}$")
        parts.append(u)
    # "charge^2 length^2" reads as e$^2$ Å$^2$
    return " ".join(parts)


def time_unit(time_per_frame) -> str:
    """The unit of a step's time axis: ps when ``time_per_frame`` is given (ewald's convention for
    it), else :data:`FRAMES` (the times are frame lags)."""
    return "ps" if (time_per_frame or 0) > 0 else FRAMES


def per_time(quantity_unit: str, time_per_frame) -> str:
    """``"Å$^2$/ps"``, or ``"Å$^2$/frame"`` without a time between frames."""
    return f"{quantity_unit}/{'ps' if (time_per_frame or 0) > 0 else 'frame'}"


def _base(name: str) -> str:
    from .plotstyle import _split

    return _split(name)[0]


def column_unit(name: str, style: str | None = None) -> str | None:
    """The unit a column has by its name in unit system ``style`` (None when its name says nothing)."""
    quantity = QUANTITY.get(_base(name))
    return None if quantity is None else unit(quantity, style)


def table_units(data, table: str, columns=None) -> dict:
    """``{column: unit}`` of ``data.tables[table]`` (or of ``columns``): what a step recorded in
    ``data.units``, else what the column's name and the frame's unit system say. Columns whose
    unit is not known are left out."""
    style = system(data)
    recorded = getattr(data, "units", None) or {}
    cols = columns if columns is not None else (data.tables.get(table) or {})
    out = {}
    for col in cols:
        u = recorded.get((table, col))
        if u is None:
            u = column_unit(col, style)
        if u is not None:
            out[col] = u
    return out


def attribute_units(data, names) -> dict:
    """``{attribute: unit}`` for those of ``names`` a step recorded a unit for."""
    recorded = getattr(data, "units", None) or {}
    return {n: recorded[n] for n in names if isinstance(n, str) and n in recorded}


def rename_table(data, old: str, new: str):
    """Move the recorded units of table ``old`` to ``new`` (a verb's table put beside an earlier one)."""
    units = getattr(data, "units", None)
    if not units:
        return
    for (table, col) in [k for k in units if isinstance(k, tuple) and k[0] == old]:
        units[(new, col)] = units.pop((table, col))
