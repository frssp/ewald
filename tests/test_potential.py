"""Planar/macroscopic averages, work function and the potential lineup (ewald.potential).

The lineup tests are ported from sim-agent (frssp/sim_agent tests/test_band_alignment.py), so
the two implementations are held to the same answers: synthetic planar averages with a known
step, each side oscillating with its own period, joined by smooth seams.
"""
import json

import numpy as np
import pytest

from ewald.data import Cell, DataCollection
from ewald.electronic import ElectronicStructure
from ewald.potential import (
    PlanarProfile,
    _alignment_type,
    band_alignment,
    box_filter,
    bulk_edges,
    classify_junction,
    layer_extent,
    layer_period,
    lineup_table,
    potential_lineup,
    read_planar_json,
    scissor_schottky,
    split_sides,
    work_function,
)

L, N = 40.0, 400
P_SUB, P_FILM = 2.12, 3.25
STEP = 0.8


def _smooth_window(z, lo, hi, width=0.8):
    def edge(x):
        return 0.5 * (1 + np.tanh(x / width))
    return edge((z - lo + L / 2) % L - L / 2) * edge((hi - z + L / 2) % L - L / 2)


def _profile(field=0.0, offset=0.0):
    z = np.arange(N) * L / N
    in_film = _smooth_window(z, 21.0, 39.0)
    in_sub = 1.0 - in_film
    v = (-5.0 + STEP * in_film + 1.5 * np.cos(2 * np.pi * z / P_SUB) * in_sub
         + 2.0 * np.cos(2 * np.pi * z / P_FILM) * in_film)
    return v + field * (((z + offset) % L) - L / 2)


SUB_Z = np.linspace(2.0, 18.0, 9) / L
FILM_Z = np.linspace(22.0, 38.0, 9) / L


def _lineup(field=0.0, offset=0.0, **kw):
    return potential_lineup(PlanarProfile(_profile(field, offset), L), SUB_Z, FILM_Z, P_SUB, P_FILM, **kw)


class TestLineup:
    def test_step_is_recovered_through_the_oscillations(self):
        res = _lineup()
        assert res["ok"], res["reason"]
        assert res["delta_v_eV"] == pytest.approx(STEP, abs=2e-3)

    def test_tilted_plateau_is_refused(self):
        res = _lineup(field=0.02)
        assert not res["ok"] and res["code"] == "PLATEAU_NOT_FLAT" and res["delta_v_eV"] is None

    def test_thin_layer_is_refused(self):
        assert _lineup(margin=9.0)["code"] == "THIN_LAYER"

    def test_layer_straddling_the_cell_boundary(self):
        shift = 137
        res = potential_lineup(PlanarProfile(np.roll(_profile(), shift), L), (SUB_Z + shift / N) % 1,
                               (FILM_Z + shift / N) % 1, P_SUB, P_FILM)
        assert res["ok"] and res["delta_v_eV"] == pytest.approx(STEP, abs=2e-3)

    def test_interleaved_atoms_are_not_an_interface(self):
        res = potential_lineup(PlanarProfile(_profile(), L), SUB_Z, SUB_Z + 0.01, P_SUB, P_FILM)
        assert res["code"] == "NOT_LAYERED"

    def test_a_coarse_grid_is_not_called_a_thin_layer(self):
        n = 20
        prof = np.interp(np.arange(n) * L / n, np.arange(N) * L / N, _profile())
        res = potential_lineup(PlanarProfile(prof, L), SUB_Z, FILM_Z, P_SUB, P_FILM, margin=6.5)
        assert res["code"] == "GRID_TOO_COARSE"

    def test_box_filter_removes_a_period_off_the_grid(self):
        z = np.arange(N) * L / N
        assert np.max(np.abs(box_filter(np.cos(2 * np.pi * z / (L / 13)), L, L / 13))) < 1e-10

    def test_layer_extent_wraps(self):
        start, width = layer_extent([0.9, 0.95, 0.02, 0.1])
        assert start == pytest.approx(0.9) and width == pytest.approx(0.2)


METAL = {"vbm": 9.0, "cbm": 9.0, "v_mean": 0.0}
SEMI = {"vbm": 4.0, "cbm": 7.0, "v_mean": 0.0}


class TestAlignment:

    def test_schottky_barriers_sum_to_the_gap(self):
        lineup = _lineup()
        res = band_alignment(lineup, METAL, SEMI, interface_fermi=-1.0)
        v_film = lineup["plateaus"]["film"]["mean_eV"]
        assert res["ok"] and res["junction"] == "metal/semiconductor" and res["semiconductor_side"] == "film"
        assert res["schottky_barrier_p_eV"] == pytest.approx(-1.0 - (4.0 + v_film))
        assert res["schottky_barrier_p_eV"] + res["schottky_barrier_n_eV"] == pytest.approx(3.0)

    def test_bulk_potential_reference_is_subtracted(self):
        a = band_alignment(_lineup(), METAL, SEMI, interface_fermi=0.0)
        b = band_alignment(_lineup(), METAL, {"vbm": 5.0, "cbm": 8.0, "v_mean": 1.0}, interface_fermi=0.0)
        assert a["schottky_barrier_p_eV"] == pytest.approx(b["schottky_barrier_p_eV"])

    def test_refusals(self):
        assert band_alignment(_lineup(), METAL, SEMI, None)["code"] == "FERMI_LEVEL_MISSING"
        assert band_alignment(_lineup(), METAL, METAL, 0.0)["code"] == "NO_BAND_EDGE"
        assert band_alignment(_lineup(field=0.02), SEMI, SEMI, None)["code"] == "PLATEAU_NOT_FLAT"

    def test_a_metal_is_read_from_its_bands_not_its_edge_spacing(self):
        al = {"vbm": 9.0, "cbm": 9.118, "v_mean": 0.0, "is_metal": True}
        assert band_alignment(_lineup(), al, SEMI, 0.0)["junction"] == "metal/semiconductor"

    def test_band_offsets(self):
        sub, film = {"vbm": 2.0, "cbm": 3.4, "v_mean": 0.0}, {"vbm": 1.5, "cbm": 3.6, "v_mean": 0.0}
        lineup = _lineup()
        res = band_alignment(lineup, sub, film, None)
        assert res["valence_band_offset_eV"] == pytest.approx(-0.5 + lineup["delta_v_eV"])
        assert res["alignment_type"] == "II"
        assert classify_junction(0.05, 3.0) == "metal/semiconductor"
        assert _alignment_type(1.0, 2.0, 2.5, 3.5) == "III"

    def test_scissor_matches_sim_agent(self):
        res = {"ok": True, "semiconductor_side": "film", "schottky_barrier_p_eV": 2.168,
               "schottky_barrier_n_eV": 1.788 - 2.168, "bulk_gaps_eV": {"substrate": 0.0, "film": 1.788},
               "band_edges_on_interface_axis": {"film": {"vbm_eV": 1.0, "cbm_eV": 2.788}}}
        sc = scissor_schottky(res, 3.03)
        assert sc["schottky_barrier_n_scissor_eV"] == pytest.approx(0.862)
        assert sc["fermi_above_computed_cbm"] is True
        assert scissor_schottky(res, "abc")["applied"] is False


def _frame(planes, a=4.0, c=L):
    """Atoms on planes ``[(z, ["Ti", "N"]), ...]`` of a tetragonal cell, spread in-plane."""
    pos, types = [], []
    for z, names in planes:
        for i, t in enumerate(names):
            pos.append([a * i / len(names), a * (i % 2) / 2, z])
            types.append(t)
    return DataCollection(pos, types, Cell(np.diag([a, a, c])))


class TestGeometry:
    def test_sides_from_plane_composition(self):
        iface = _frame([(z, ["Ti", "N"]) for z in range(1, 19, 2)] + [(z, ["Ti", "O", "O"]) for z in range(21, 39, 2)])
        sides = split_sides(iface, _frame([(1, ["Ti", "N"])]), _frame([(1, ["Ti", "O", "O"])]))
        assert set(iface.types[sides["substrate"]]) == {"Ti", "N"}
        assert set(iface.types[sides["film"]]) == {"Ti", "O"}
        assert sides["ambiguous"].size == 0

    def test_shared_anion_planes_follow_their_neighbours(self):
        sub = [(z, ["Ga"] if k % 2 == 0 else ["As"]) for k, z in enumerate(np.arange(1, 19.5, 1.5))]
        film = [(z, ["Al"] if k % 2 == 0 else ["As"]) for k, z in enumerate(np.arange(20.5, 39.5, 1.5))]
        iface = _frame(sub + film)
        sides = split_sides(iface, _frame([(1, ["Ga"]), (2, ["As"])]), _frame([(1, ["Al"]), (2, ["As"])]))
        z = iface.positions[:, 2]
        assert z[sides["film"]].min() > 19.0 and z[sides["substrate"]].max() < 20.0
        assert iface.types[sides["ambiguous"]].tolist() in ([], ["As"])   # only a seam plane can be ambiguous

    def test_interleaved_sides_raise(self):
        iface = _frame([(z, ["Ti", "N"] if k % 2 else ["Ti", "O", "O"]) for k, z in enumerate(range(1, 39, 2))])
        with pytest.raises(ValueError, match="interleave"):
            split_sides(iface, _frame([(1, ["Ti", "N"])]), _frame([(1, ["Ti", "O", "O"])]))

    def test_layer_period_is_the_primitive_spacing_whatever_the_reference(self):
        # rock salt along [001]: the conventional cell repeats every a, the crystal every a/2
        # (the face-centring translation); a primitive bulk cell's a x b is not even the facet,
        # which is why the period is read from the layer in the interface (sim-agent's
        # period_along_normal gives a/2 = 2.12 A for TiN(100))
        a = 4.24
        base = [([0, 0, 0], "Ti"), ([.5, .5, 0], "Ti"), ([.5, 0, .5], "Ti"), ([0, .5, .5], "Ti"),
                ([.5, 0, 0], "N"), ([0, .5, 0], "N"), ([0, 0, .5], "N"), ([.5, .5, .5], "N")]
        pos = [(np.array(f) + [0, 0, k]) * a for k in range(4) for f, _ in base]
        types = [t for _ in range(4) for _, t in base]
        slab = DataCollection(np.array(pos), types, Cell(np.diag([a, a, 4 * a + 6.0])))
        assert layer_period(slab, np.arange(len(types))) == pytest.approx(a / 2)
        # fcc(111) in a hexagonal cell: ABC stacking repeats every 3 planes, the crystal every one
        d = 4.05 / np.sqrt(3)
        cell = np.array([[2.864, 0, 0], [1.432, 2.480, 0], [0, 0, 9 * d + 8.0]])
        shifts = [np.zeros(3), (cell[0] + cell[1]) / 3, 2 * (cell[0] + cell[1]) / 3]
        pos = [shifts[k % 3] + [0, 0, k * d] for k in range(9)]
        al = DataCollection(np.array(pos), ["Al"] * 9, Cell(cell))
        assert layer_period(al, np.arange(9)) == pytest.approx(d)

    def test_layer_period_needs_two_periods(self):
        one = DataCollection([[0, 0, 0], [0, 0, 2.0]], ["Ti", "N"], Cell(np.diag([3.0, 3.0, 20.0])))
        with pytest.raises(ValueError, match="repeat"):
            layer_period(one, [0, 1])


class TestReadersAndTables:
    def test_planar_json(self, tmp_path):
        path = tmp_path / "locpot_planar_avg.json"
        path.write_text(json.dumps({"grid": [4, 4, 5], "planar_avg_z": [1, 2, 3, 2, 1]}), encoding="utf-8")
        p = read_planar_json(path, 10.0)
        assert p.values.tolist() == [1, 2, 3, 2, 1] and p.z[1] == pytest.approx(2.0)
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="planar_avg_z"):
            read_planar_json(path, 10.0)

    def test_work_function_is_vacuum_minus_fermi(self):
        z = np.linspace(0, 30, 300, endpoint=False)
        v = np.where((z > 5) & (z < 15), -8.0, 4.0)
        res = work_function(PlanarProfile(v, 30.0), efermi=-0.5)
        assert res["work_function_eV"] == pytest.approx(4.5) and res["plausible"]

    def test_bulk_edges_from_eigenvalues(self):
        e = np.array([[[-1.0, 0.0, 2.0], [-0.9, 0.2, 1.8]]])
        occ = np.array([[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]])
        es = ElectronicStructure(types=np.array(["Si"]), efermi=0.5, eigenvalues=e, occupations=occ,
                                 kpoints=np.zeros((2, 3)), weights=np.ones(2))
        edges = bulk_edges(es, PlanarProfile([1.0, 3.0], 2.0))
        assert edges["vbm"] == pytest.approx(0.2) and edges["cbm"] == pytest.approx(1.8)
        assert edges["is_metal"] is False and edges["v_mean"] == pytest.approx(2.0)

    def test_lineup_table_puts_levels_on_the_fermi_axis(self):
        lineup = _lineup()
        res = band_alignment(lineup, METAL, SEMI, interface_fermi=-1.0)
        sc = scissor_schottky(res, 3.5)
        t = lineup_table(PlanarProfile(_profile(), L), lineup, res, sc, names={"film": "TiO2"})
        assert t.attributes["energy_axis"] == "E - E_F (interface)"
        levels = {lv["name"]: lv for lv in t.attributes["levels"]}
        assert levels["E_F (interface)"]["energy"] == pytest.approx(0.0)
        assert levels["TiO2 VBM"]["energy"] == pytest.approx(-res["schottky_barrier_p_eV"])
        assert "TiO2 CBM* (scissor, approx.)" in levels
        assert not any(name.startswith("substrate V") for name in levels)   # a metal has no band edges
        assert [r["side"] for r in t.attributes["regions"]] == ["substrate", "film"]


def test_work_function_table_is_on_the_vacuum_level():
    from ewald.potential import work_function, work_function_table

    z = np.arange(100) * 0.3
    prof = PlanarProfile(np.where((z > 5) & (z < 20), -8.0, 4.0), 30.0)
    wf = work_function(prof, efermi=-0.5)
    t = work_function_table(prof, wf, "Cu(111)")
    assert t.attributes["result"]["work_function_eV"] == pytest.approx(4.5)
    assert t.attributes["energy_axis"] == "E - E_vac" and t["Planar average"].max() == pytest.approx(0.0)
    assert {lv["name"]: lv["energy"] for lv in t.attributes["levels"]}["E_F"] == pytest.approx(-4.5)


@pytest.mark.parametrize("key", ["tin_mgo_100", "gaas_alas_110"])
def test_sides_and_periods_match_the_builders_metadata(key):
    """Interfaces built by sim-agent's builder (tests/data/make_interface_reference.py): the
    sides and filter periods derived here are the ones sim-agent stores and lines up with,
    although its bulk references are primitive cells whose a x b is not the facet."""
    from pathlib import Path

    from ewald import import_file

    data = Path(__file__).parent / "data"
    ref = json.loads((data / "interface_reference.json").read_text(encoding="utf-8"))[key]
    frame = {role: import_file(str(data / f"interface_{key}{suffix}.extxyz")).compute(0)
             for role, suffix in (("interface", ""), ("substrate", "_substrate"), ("film", "_film"))}
    sides = split_sides(frame["interface"], frame["substrate"], frame["film"])
    assert list(sides["substrate"]) == sorted(ref["substrate"]) and list(sides["film"]) == sorted(ref["film"])
    assert layer_period(frame["interface"], sides["substrate"]) == pytest.approx(ref["period_substrate_A"], abs=1e-4)
    assert layer_period(frame["interface"], sides["film"]) == pytest.approx(ref["period_film_A"], abs=1e-4)
