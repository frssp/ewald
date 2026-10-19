import numpy as np
import pytest

from ewald import Pipeline, StaticSource, import_file, linking
from ewald import modifiers as mods


@pytest.fixture
def cu_data(cu_fcc):
    d = cu_fcc.copy()
    d.particles["Position"] = d.positions + np.random.default_rng(0).normal(0, 0.08, d.positions.shape)
    pipe = Pipeline(StaticSource([d, d]), [mods.CoordinationAnalysis(3.0, partial=True), mods.CreateBonds(3.0),
                                           mods.BondLengthDistribution(), mods.BondAngleDistribution()])
    return pipe, pipe.compute(0)


def test_links_of_common_tables(cu_data):
    _, d = cu_data
    assert linking.link_for(d, "coordination-rdf").kind == "pairs"
    assert linking.link_for(d, "bond-length-distr").kind == "bonds"
    assert linking.link_for(d, "bond-angle-distr").kind == "angles"
    assert linking.link_for(d, hist="Coordination") == linking.Link("property", "Coordination")
    assert linking.link_for(d, hist="Position.Z").kind == "property"
    assert linking.link_for(d, hist="nope") is None
    assert linking.link_for(d, "coordination-rdf", "g(r) Cu-Cu").types == ("Cu", "Cu")


def test_atom_ranges_match_brute_force(cu_data):
    _, d = cu_data
    pos, m = d.positions, d.cell.matrix
    delta = pos[None] - pos[:, None]
    f = delta @ np.linalg.inv(m)
    dist = np.linalg.norm((f - np.round(f)) @ m, axis=-1)
    in_shell = (dist >= 2.5) & (dist <= 2.6)
    expect = np.nonzero(in_shell.any(1))[0]
    got = linking.atoms_in_range(d, linking.link_for(d, "coordination-rdf"), 2.6, 2.5)  # either order
    np.testing.assert_array_equal(got, expect)
    # bonds: the ends of every bond with that length
    L = linking.link_for(d, "bond-length-distr")
    lengths = d.bonds.lengths(d)
    i, j = d.bonds.topology[(lengths >= 2.5) & (lengths <= 2.6)].T
    np.testing.assert_array_equal(linking.atoms_in_range(d, L, 2.5, 2.6), np.unique(np.r_[i, j]))
    # property range == ExpressionSelection of the generated expression
    L = linking.link_for(d, hist="Position.Z")
    expr = linking.selection_expression(L, 1.0, 4.0)
    sel = Pipeline(StaticSource([d]), [mods.ExpressionSelection(expr)]).compute(0).selection
    np.testing.assert_array_equal(np.nonzero(sel)[0], linking.atoms_in_range(d, L, 1.0, 4.0))
    assert linking.selection_expression(linking.link_for(d, "bond-length-distr"), 1, 2) is None


def test_values_of_selection_fill_the_plot(cu_data):
    _, d = cu_data
    L = linking.link_for(d, "bond-angle-distr")
    v = linking.values_of(d, L, [0])
    assert len(v) == 66 and np.all((v > 0) & (v <= 180))  # 12 bonds at an fcc site: 12*11/2 angles
    assert np.isin(0, linking.atoms_in_range(d, L, v.min() - 1e-6, v.min() + 1e-6))
    r = linking.values_of(d, linking.link_for(d, "coordination-rdf"), [0], xmax=3.0)
    assert len(r) == 12
    np.testing.assert_allclose(linking.values_of(d, linking.Link("property", "Coordination"), [3, 4]),
                               d["Coordination"][[3, 4]])


def test_thermo_rows_follow_frame_timesteps(lammps_dir):
    pipe = import_file(lammps_dir / "thermo" / "dump.lammpstrj")
    d = pipe.compute(0)
    L = linking.link_for(d, "Thermo", "Temp", num_frames=pipe.num_frames)
    assert L.kind == "frames"
    np.testing.assert_array_equal(linking.frame_timesteps(pipe), [0, 10, 20, 30, 40])
    np.testing.assert_array_equal(linking.frame_rows(pipe, d, "Thermo"), [0, 2, 4, 6, 8])
    assert linking.frames_in_box(pipe, d, L, 15, 32) == [2, 3]
    assert linking.frame_at(pipe, d, L, 26) == 3
    temp = np.asarray(d.tables["Thermo"]["Temp"])
    hot = [k for k, r in enumerate([0, 2, 4, 6, 8]) if temp[r] > 200]
    assert linking.frames_in_box(pipe, d, L, -1, 100, 200, 1e9) == hot  # Shift+drag: by value
    x, _ = linking.frame_points(pipe, d, L, [1, 4])
    np.testing.assert_array_equal(x, [10, 40])


def test_fes_box_selects_frames_through_colvar(tmp_path, cu_fcc):
    colvar = tmp_path / "COLVAR"
    rows = [(0, 0.1, 0.1), (1, 0.9, 0.1), (2, 0.9, 0.9), (3, 0.15, 0.2), (4, 0.5, 0.5), (5, 0.5, 0.5)]
    colvar.write_text("#! FIELDS time d1 d2\n" + "".join(f"{t} {a} {b}\n" for t, a, b in rows))
    g = np.linspace(0, 1, 5)
    x, y = np.meshgrid(g, g)

    def fes(frame, data):
        data.tables["fes"] = {"d1": x.ravel(), "d2": y.ravel(), "Free Energy": (x + y).ravel()}

    # stride 2: frame k is COLVAR row 2k, so rows 1, 3 and 5 belong to no frame
    class LoadColvar(mods.Modifier):  # stands in for the ewald-plumed plugin's: frame k is COLVAR row stride*k
        def __init__(self, path: str = "", stride: int = 1):
            super().__init__()
            self.path, self.stride = path, stride

        def row(self, frame):
            return frame * self.stride

        def modify(self, data, ctx):
            t = np.loadtxt(self.path)
            data.tables["colvar"] = {"time": t[:, 0], "d1": t[:, 1], "d2": t[:, 2]}
            row = t[self.row(ctx.frame)]
            data.attributes.update({"Colvar.d1": row[1], "Colvar.d2": row[2]})

    pipe = Pipeline(StaticSource([cu_fcc] * 3), [LoadColvar(str(colvar), stride=2), mods.PythonModifier(fes)])
    d = pipe.compute(0)
    L = linking.link_for(d, "fes")
    assert L == linking.Link("frames2d", "fes", "colvar")
    np.testing.assert_array_equal(linking.frame_rows(pipe, d, "colvar"), [0, 2, 4])
    assert linking.frames_in_box(pipe, d, L, 0, 0.3, 0, 0.3) == [0]
    assert linking.frames_in_box(pipe, d, L, 0.4, 1, 0.4, 1) == [1, 2]
    px, py = linking.frame_points(pipe, d, L, [1])
    assert (px.tolist(), py.tolist()) == ([0.9], [0.9])
    # the colvar table itself is a per-frame plot (x = time)
    Lc = linking.link_for(d, "colvar", "d1", num_frames=3)
    assert Lc.kind == "frames" and linking.frames_in_box(pipe, d, Lc, 1.5, 4.5) == [1, 2]
