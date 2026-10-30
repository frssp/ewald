"""Ion transport modifiers: conductivity/Haven ratio, van Hove, F_s(q, t), non-Gaussian parameter, hopping."""
import numpy as np
import pytest

from ewald import DataCollection, Pipeline, StaticSource
from ewald.backend import CPU, Backend
from ewald.data import Cell
from ewald.kernels import (conductivity_from_slope, decay_time, displacement_moments, fibonacci_directions,
                           msd_fft, non_gaussian_parameter, rdf, self_intermediate_scattering, unwrap,
                           van_hove_distinct, van_hove_self)
from ewald.modifiers import (IonicConductivity, MeanSquaredDisplacement, NonGaussianParameter, SelectType,
                             SelfIntermediateScattering, SiteHopping, VanHoveCorrelation)


class _NumpyAsGPU:
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(a):
        return np.asarray(a)


@pytest.fixture(params=["cpu", "fake"])
def backend(request):
    return CPU if request.param == "cpu" else Backend("fake-cuda", _NumpyAsGPU())


BOX = 12.0


def _pipeline(true, types, box=BOX, shuffle_rng=None, charges=None):
    """Frames of a trajectory ``true`` (T, N, 3) wrapped into a cubic box, optionally shuffled."""
    frames = []
    n = true.shape[1]
    for k, x in enumerate(true):
        perm = shuffle_rng.permutation(n) if shuffle_rng is not None else np.arange(n)
        d = DataCollection((x % box)[perm], np.asarray(types)[perm], Cell(np.diag([box] * 3)))
        d.particles["Particle Identifier"] = perm + 1
        if charges is not None:
            d.particles["Charge"] = np.asarray(charges, float)[perm]
        d.attributes["Timestep"] = 10 * k
        frames.append(d)
    return Pipeline(StaticSource(frames))


def _walk(rng, T, n, sigma):
    return np.cumsum(rng.normal(scale=sigma, size=(T, n, 3)), axis=0) + rng.random((n, 3)) * BOX


# --------------------------------------------------------------------------- conductivity

def test_lockstep_cations_give_haven_ratio_one_over_n(rng, backend):
    """N cations moving together: <|sum dr|^2> = N^2 <dr^2>, so H_R = sigma_NE / sigma = 1/N."""
    T, n = 50, 8
    step = np.cumsum(rng.normal(scale=0.3, size=(T, 1, 3)), axis=0)
    base = rng.random((2 * n, 3)) * BOX
    true = np.repeat(base[None], T, axis=0)
    true[:, :n] += step                                    # cations move, anions stay
    types = ["Li"] * n + ["Cl"] * n
    pipe = _pipeline(true, types, shuffle_rng=rng)
    pipe.modifiers.append(IonicConductivity(time_per_frame=0.1, subtract_drift=False))
    data = pipe.compute(3, device=backend)
    assert data.attributes["IonicConductivity.haven_ratio"] == pytest.approx(1 / n, rel=1e-9)
    tab = data.tables["conductivity"]
    expect = msd_fft(step) * n * n
    assert np.allclose(tab["Charge MSD"], expect)
    assert np.allclose(tab["Nernst-Einstein MSD"], msd_fft(step) * n)


def test_conductivity_units_and_charge_sources(rng):
    """sigma_NE = N q^2 D / (V kB T); charges from a dict, the 'Charge' property or guessed."""
    T, n = 80, 40
    true = _walk(rng, T, 2 * n, 0.2)
    types = ["Li"] * n + ["F"] * n
    tpf, temp = 0.05, 600.0
    results = []
    for charges, prop in [({"Li": 1, "F": -1}, None), (None, [1.0] * n + [-1.0] * n), (None, None)]:
        pipe = _pipeline(true, types, charges=prop)
        pipe.modifiers.append(IonicConductivity(temperature=temp, time_per_frame=tpf, charges=charges,
                                                subtract_drift=False))
        results.append(pipe.compute(0).attributes)
    for r in results[1:]:
        assert r["IonicConductivity.conductivity"] == pytest.approx(results[0]["IonicConductivity.conductivity"])
    a = results[0]
    pipe = _pipeline(true, types)
    pipe.modifiers.append(MeanSquaredDisplacement(time_per_frame=tpf, by_type=False))
    d = pipe.compute(0).attributes["MeanSquaredDisplacement.diffusion_coefficient"]   # A^2/ps
    e, kb = 1.602176634e-19, 1.380649e-23
    sigma_si = 2 * n / (BOX ** 3 * 1e-30) * e ** 2 * d * 1e-8 / (kb * temp)           # S/m
    assert a["IonicConductivity.nernst_einstein_conductivity"] == pytest.approx(sigma_si / 100, rel=1e-6)
    assert a["IonicConductivity.volume"] == pytest.approx(BOX ** 3)
    assert 0.2 < a["IonicConductivity.haven_ratio"] < 5     # uncorrelated ions: ~1, noisy
    assert conductivity_from_slope(d, BOX ** 3, temp) == pytest.approx(sigma_si / 100 / (2 * n))


def test_bound_pairs_do_not_conduct(rng):
    """Cation-anion pairs that move together carry no net charge displacement."""
    T, n = 40, 10
    walk = _walk(rng, T, n, 0.3)
    true = np.concatenate([walk, walk + 1.0], axis=1)
    pipe = _pipeline(true, ["Na"] * n + ["Cl"] * n)
    pipe.modifiers.append(IonicConductivity(time_per_frame=0.1, charges={"Na": 1, "Cl": -1}))
    a = pipe.compute(0).attributes
    assert abs(a["IonicConductivity.conductivity"]) < 1e-12 * a["IonicConductivity.nernst_einstein_conductivity"]
    assert np.allclose(pipe.compute(0).tables["conductivity"]["Charge MSD"], 0, atol=1e-12)


def test_conductivity_selection_drift_and_errors(rng):
    T, n = 30, 6
    walk = _walk(rng, T, 2 * n, 0.2)
    true = walk + np.linspace(0, 5, T)[:, None, None]   # uniform drift
    pipe = _pipeline(true, ["Li"] * n + ["Xx"] * n)
    pipe.modifiers.append(IonicConductivity(time_per_frame=0.1))
    with pytest.raises(Exception, match="charge"):
        pipe.compute(0)
    pipe.modifiers[:] = [SelectType("Li"), IonicConductivity(time_per_frame=0.1, charges={"Li": 1, "Xx": -1},
                                                             only_selected=True, subtract_drift=True)]
    a = pipe.compute(0).attributes
    still = _pipeline(walk, ["Li"] * n + ["Xx"] * n)
    still.modifiers[:] = [SelectType("Li"), IonicConductivity(time_per_frame=0.1, charges={"Li": 1, "Xx": -1},
                                                              only_selected=True)]
    b = still.compute(0).attributes
    for key in ("conductivity", "nernst_einstein_conductivity"):   # the drift is removed exactly
        assert a["IonicConductivity." + key] == pytest.approx(b["IonicConductivity." + key])
    lockstep = walk.copy()
    lockstep[:, :n] = walk[:, :1] + (walk[0, :n] - walk[0, :1])[None]    # the Li move together
    lockstep[:, n:] = walk[0, n:][None]
    nodt = _pipeline(lockstep, ["Li"] * n + ["Xx"] * n)
    nodt.modifiers.append(IonicConductivity(charges={"Li": 1, "Xx": -1}, subtract_drift=False))
    a = nodt.compute(0).attributes
    assert np.isnan(a["IonicConductivity.conductivity"])        # no time_per_frame: no S/cm
    assert a["IonicConductivity.haven_ratio"] == pytest.approx(1 / n)
    pipe.modifiers[:] = [IonicConductivity(charges={"Li": 1})]
    with pytest.raises(Exception, match="Xx"):
        pipe.compute(0)


# --------------------------------------------------------------------------- non-Gaussian, F_s, van Hove

def test_non_gaussian_parameter_gaussian_and_hops(rng, backend):
    T, n = 30, 3000
    gauss = np.cumsum(rng.normal(scale=0.2, size=(T, n, 3)), axis=0)
    r2, r4 = displacement_moments(gauss, None, backend)
    a2 = non_gaussian_parameter(r2, r4)
    assert np.isnan(a2[0]) and np.all(np.abs(a2[1:10]) < 0.1)
    # lattice hops of length 1 with probability p per frame: alpha2(1) = 3 / (5 p) - 1
    p = 0.1
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    jumps = dirs[rng.integers(0, 6, size=(T, n))] * (rng.random((T, n)) < p)[..., None]
    hop = np.cumsum(jumps, axis=0)
    r2, r4 = displacement_moments(hop, 50, backend)
    assert non_gaussian_parameter(r2, r4)[1] == pytest.approx(3 / (5 * p) - 1, rel=0.1)
    pipe = _pipeline(hop[:, :200] + 3.0, ["Li"] * 200)
    pipe.modifiers.append(NonGaussianParameter(time_per_frame=2.0))
    data = pipe.compute(0, device=backend)
    assert data.attributes["NonGaussianParameter.max"] > 2
    assert data.tables["non-gaussian"]["Time"][1] == 2.0


def test_self_intermediate_scattering_gaussian(rng, backend):
    """Gaussian steps of variance s^2 per component: F_s(q, t) = exp(-q^2 s^2 t / 2)."""
    T, n, s = 40, 500, 0.1
    x = np.cumsum(rng.normal(scale=s, size=(T, n, 3)), axis=0)
    q = [1.0, 3.0]
    fs = self_intermediate_scattering(x, q, 16, backend)
    t = np.arange(T)
    for row, qv in zip(fs, q):
        assert np.allclose(row[:20], np.exp(-qv * qv * s * s * t[:20] / 2), atol=0.03)
    assert fs[:, 0] == pytest.approx([1.0, 1.0])
    d = fibonacci_directions(16)
    assert np.allclose(np.linalg.norm(d, axis=1), 1) and np.all(d[:, 2] > 0)
    tau = decay_time(t, np.exp(-t / 7.0))
    assert tau == pytest.approx(7.0, rel=0.02)
    pipe = _pipeline(x + 5.0, ["O"] * n)
    pipe.modifiers.append(SelfIntermediateScattering(q=3.0, time_per_frame=0.5))
    data = pipe.compute(0, device=backend)
    expect = 2 / (9 * s * s) * 0.5   # exp(-q^2 s^2 t / 2) = 1/e
    assert data.attributes["SelfIntermediateScattering.relaxation_time"] == pytest.approx(expect, rel=0.1)
    assert "Fs(q=3)" in data.tables["intermediate-scattering"]
    pipe.modifiers[:] = [SelfIntermediateScattering(time_per_frame=0.5)]
    a = pipe.compute(0).attributes
    assert a["SelfIntermediateScattering.q"] == pytest.approx(2 * np.pi / (BOX ** 3 / n) ** (1 / 3))


def test_van_hove_self_normalized_and_gaussian(rng, backend):
    T, n, s = 30, 2000, 0.3
    x = np.cumsum(rng.normal(scale=s, size=(T, n, 3)), axis=0)
    r, p = van_hove_self(x, [1, 4], 4.0, 80, 20, backend)
    dr = r[1] - r[0]
    assert np.allclose(p.sum(1) * dr, 1.0, atol=1e-3)
    # Maxwell distribution of |dr| peaks at sqrt(2 t) s
    for row, m in zip(p, [1, 4]):
        assert r[np.argmax(row)] == pytest.approx(np.sqrt(2 * m) * s, abs=3 * dr)


def test_van_hove_distinct_is_g_of_r_at_zero(rng, backend):
    n = 200
    x0 = rng.random((n, 3)) * BOX
    x = np.stack([x0 + rng.normal(scale=0.05, size=x0.shape) for _ in range(6)])
    cell = Cell(np.diag([BOX] * 3))
    r, g = van_hove_distinct(x, cell.matrix, cell.pbc, [0, 2], 5.0, 50, None, backend)
    ref = np.mean([rdf(f, cell, 5.0, 50)["g(r)"] for f in x], axis=0)
    assert np.allclose(g[0], ref)
    assert np.allclose(r, rdf(x[0], cell, 5.0, 50)["r"])
    pipe = _pipeline(x, ["Na"] * n)
    pipe.modifiers.append(VanHoveCorrelation(times=(2,), part="both", r_max=5.0, nbins=50, time_per_frame=0.1))
    tab = pipe.compute(0, device=backend).tables["van-hove"]
    assert {"r", "4πr²Gs(t=0.2)", "Gs(t=0.2)", "Gd(t=0)", "Gd(t=0.2)"} <= set(tab)
    assert np.allclose(tab["Gd(t=0)"], van_hove_distinct(x, cell.matrix, cell.pbc, [0], 5.0, 50, 20)[1][0])
    pipe.modifiers[:] = [VanHoveCorrelation(part="both", r_max=7.0)]
    with pytest.raises(Exception, match="r_max"):
        pipe.compute(0)
    pipe.modifiers[:] = [VanHoveCorrelation(part="distinct", r_max=7.0)]   # clipped to half the box
    assert pipe.compute(0).tables["van-hove"]["r"][-1] < 6.0
    pipe.modifiers[:] = [VanHoveCorrelation(part="all")]
    with pytest.raises(Exception, match="part"):
        pipe.compute(0)


# --------------------------------------------------------------------------- hopping

def _hopping_trajectory(rng, T=20, noise=0.05):
    """8 ions on a 3 A simple cubic lattice of 64 sites in a 12 A box; ion 0 hops +x at frame 5
    and back at frame 12, ion 1 hops +y (through the boundary) at frame 8."""
    g = np.arange(4) * 3.0 + 1.5
    sites = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    occupied = rng.choice(len(sites), 8, replace=False)
    true = np.repeat(sites[occupied][None], T, axis=0)
    true[5:12, 0, 0] += 3.0
    true[8:, 1, 1] -= 3.0 if sites[occupied[1], 1] < 3 else -3.0
    true = true + rng.normal(scale=noise, size=true.shape)
    return sites, occupied, true


def test_site_hopping_counts_hops_and_residence(rng, backend):
    sites, occupied, true = _hopping_trajectory(rng)
    pipe = _pipeline(true, ["Li"] * 8, shuffle_rng=rng)
    pipe.modifiers.append(SiteHopping(sites=sites, time_per_frame=0.5))
    data = pipe.compute(6, device=backend)
    a = data.attributes
    assert a["SiteHopping.hops"] == 3 and a["SiteHopping.sites"] == 64
    hops = data.tables["site-hops"]
    assert sorted(hops["Frame"]) == [5, 8, 12]
    assert np.allclose(hops["Distance"], 3.0)
    assert a["SiteHopping.mean_residence_time"] == pytest.approx(7 * 0.5)
    assert a["SiteHopping.hop_rate"] == pytest.approx(3 / (8 * 19 * 0.5))
    assert a["SiteHopping.jump_diffusion_coefficient"] == pytest.approx(3 * 9.0 / (6 * 8 * 19 * 0.5))
    ids = np.asarray(data["Particle Identifier"])
    count = np.asarray(data["Hop Count"])
    assert count[ids == 1][0] == 2 and count[ids == 2][0] == 1 and count.sum() == 3
    site = np.asarray(data["Site Index"])
    x0 = (sites[occupied[0]] + [3.0, 0, 0]) % BOX
    assert site[ids == 1][0] == np.argmin(np.linalg.norm(sites - x0, axis=1))
    occ = data.tables["site-occupancy"]["Occupancy"]
    assert occ.sum() == pytest.approx(8.0)
    assert data.tables["residence-times"]["Count"].sum() == 1


def test_site_hopping_default_sites_radius_and_file(rng, tmp_path):
    sites, occupied, true = _hopping_trajectory(rng, noise=0.0)
    true[3, 2, 2] += 1.6        # ion 2 rattles past the boundary between two sites for one frame
    pipe = _pipeline(true, ["Li"] * 8)
    pipe.modifiers.append(SiteHopping(sites=sites))
    assert pipe.compute(0).attributes["SiteHopping.hops"] == 5
    pipe.modifiers[:] = [SiteHopping(sites=sites, site_radius=1.0)]
    assert pipe.compute(0).attributes["SiteHopping.hops"] == 3
    # default sites: the ions' own positions at frame start (no vacancies to hop into -> only
    # ion 0 hopping back counts once it is near its own start site again)
    pipe.modifiers[:] = [SiteHopping(site_radius=1.0)]
    a = pipe.compute(0).attributes
    assert a["SiteHopping.sites"] == 8
    # sites from a structure file, filtered by type
    from ase import Atoms
    from ase.io import write

    lattice = Atoms(["Li"] * 64 + ["O"], positions=np.vstack([sites, [[0.1, 0.1, 0.1]]]), cell=[BOX] * 3, pbc=True)
    path = tmp_path / "POSCAR"
    write(path, lattice, format="vasp")
    pipe.modifiers[:] = [SiteHopping(sites=str(path), site_type="Li", site_radius=1.0)]
    a = pipe.compute(0).attributes
    assert a["SiteHopping.sites"] == 64 and a["SiteHopping.hops"] == 3


def test_site_hopping_density_sites_and_min_stay(rng):
    sites, occupied, true = _hopping_trajectory(rng)
    pipe = _pipeline(true, ["Li"] * 8)
    # sites from the time-averaged density: the 8 start sites and the 2 empty ones visited
    pipe.modifiers.append(SiteHopping(sites="density"))
    data = pipe.compute(0)
    a = data.attributes
    assert a["SiteHopping.sites"] == 10 and a["SiteHopping.hops"] == 3
    occ = data.tables["site-occupancy"]
    found = np.column_stack([occ["X"], occ["Y"], occ["Z"]])
    d = found[:, None] - sites[None]
    d -= BOX * np.round(d / BOX)
    assert np.linalg.norm(d, axis=2).min(axis=1).max() < 0.25      # every site found is a lattice site
    # a one-frame visit to the next site is no hop with min_stay; nor is one at the very end
    true[3, 2, 2] += 1.6
    true[-1, 3, 0] += 3.0
    pipe = _pipeline(true, ["Li"] * 8)
    pipe.modifiers.append(SiteHopping(sites=sites, time_per_frame=0.5))
    assert pipe.compute(0).attributes["SiteHopping.hops"] == 6
    pipe.modifiers[:] = [SiteHopping(sites=sites, time_per_frame=0.5, min_stay=1.0)]
    a = pipe.compute(0).attributes
    assert a["SiteHopping.hops"] == 3 and a["SiteHopping.mean_residence_time"] == pytest.approx(3.5)
    from ewald.modifiers.transport import _drop_short_visits

    assert list(_drop_short_visits(np.array([0, 0, 1, 0, 0, 2, 2, 2, 3]), 2)) == [0, 0, 0, 0, 0, 2, 2, 2, 2]


def test_transport_modifiers_construct_with_defaults_and_cache(rng):
    T, n = 12, 20
    true = _walk(rng, T, n, 0.2)
    pipe = _pipeline(true, ["Li"] * (n // 2) + ["Cl"] * (n // 2))
    pipe.modifiers.extend([IonicConductivity(), VanHoveCorrelation(), SelfIntermediateScattering(),
                           NonGaussianParameter(), SiteHopping()])
    data = pipe.compute(0)
    for name in ("conductivity", "van-hove", "intermediate-scattering", "non-gaussian", "site-hops"):
        assert name in data.tables
    for m in pipe.modifiers:
        assert m.status is None
    first = [m.__dict__["_result"] for m in pipe.modifiers]
    pipe.compute(5)
    assert all(m.__dict__["_result"] is r for m, r in zip(pipe.modifiers, first))
    assert np.allclose(unwrap(true % BOX, np.diag([BOX] * 3)) - true, (true[0] % BOX - true[0])[None])
