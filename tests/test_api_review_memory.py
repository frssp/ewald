"""Memory before a huge allocation, from the API review (A9): ``CoordinationAnalysis(cutoff=40)``
on a 108-atom cell 10.85 A wide died with "Unable to allocate 24.0 GiB", although only 2.5
million pairs lie within 40 A. A cutoff beyond the cell is searched among periodic images, and
the search listed the pairs between any two image atoms before it kept those that start in the
cell. Now it lists only those, and a pair list (or a partial g(r)) that really cannot fit is
refused before it is allocated, with the cutoff that would fit. No test here allocates more
than a few hundred MiB: the budget is patched down instead."""
import os
import re
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

import ewald
import ewald.examples
from ewald import DataCollection, backend, neighbors
from ewald.backend import CPU, Backend, get_backend
from ewald.data import Cell
from ewald.kernels import rdf, trajectory_rdf
from ewald.modifiers import CoordinationAnalysis
from ewald.neighbors import find_pairs

CU = ewald.examples.path("Cu_md.extxyz")
MiB = 2**20
TRICLINIC = np.array([[6.0, 0, 0], [1.5, 5.5, 0], [-1.0, 0.8, 5.0]])


def lattice_sum(pos, matrix, pbc, cutoff):
    """Every (i, j, image) within the cutoff by brute force over the periodic images in reach."""
    widths = Cell(matrix).perpendicular_widths()
    reach = [int(np.ceil(cutoff / w)) if p else 0 for w, p in zip(widths, pbc)]
    inv = np.linalg.inv(matrix)
    frac = pos @ inv
    frac[:, np.asarray(pbc)] %= 1.0
    wrapped = frac @ matrix
    out = []
    for a in range(-reach[0], reach[0] + 1):
        for b in range(-reach[1], reach[1] + 1):
            for c in range(-reach[2], reach[2] + 1):
                delta = wrapped[None] + np.array([a, b, c]) @ matrix - wrapped[:, None]
                dist = np.linalg.norm(delta, axis=-1)
                if (a, b, c) == (0, 0, 0):
                    dist[np.diag_indices(len(pos))] = np.inf     # an atom is not its own neighbor
                i, j = np.nonzero(dist < cutoff)
                out += zip(i.tolist(), j.tolist(), np.round(dist[i, j], 8).tolist())
    return sorted(out)


class _NumpyAsGPU:
    """An array module that is not numpy, so that every dispatch takes its CuPy branch."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(a):
        return np.asarray(a)


@pytest.fixture
def fake_gpu():
    return Backend("fake-cuda", _NumpyAsGPU())


@pytest.fixture
def budget(monkeypatch):
    def set_budget(nbytes):
        monkeypatch.setattr(backend, "MEMORY_BUDGET", nbytes)

    return set_budget


# --------------------------------------------------------------------------- the cutoff beyond the cell

def test_the_reported_call_works_and_counts_the_right_neighbors():
    """The reproduction of A9. 728 periodic images are in reach of 40 A; the count is checked
    against a plain sum over them."""
    pipe = ewald.load(CU)
    pipe.add(CoordinationAnalysis(cutoff=40.0))
    data = pipe.compute(0)                                   # was: MemoryError, 24.0 GiB
    src = pipe.source_frame(0)
    pos, matrix = src.positions, src.cell.matrix
    count = np.zeros(len(pos), dtype=np.int64)
    for shift in np.array(np.meshgrid(*[np.arange(-4, 5)] * 3)).reshape(3, -1).T @ matrix:
        count += (np.linalg.norm(pos[None] + shift - pos[:, None], axis=-1) < 40.0).sum(1)
    np.testing.assert_array_equal(data["Coordination"], count - 1)           # less the atom itself
    assert 22000 < data["Coordination"].mean() < 23500                       # 4/3 pi 40^3 x 108 / V = 22690
    g = data.tables["coordination-rdf"]["g(r)"]
    assert abs(g[len(g) // 2:].mean() - 1.0) < 0.01                          # g(r) -> 1 far out


@pytest.mark.parametrize("method", ["auto", "numba", "xp", "brute"])
def test_search_among_images_lists_no_pair_it_throws_away(monkeypatch, method):
    """The cost follows the pairs that are returned: what the inner kernel finds is what
    find_pairs returns (it used to find 92 times more here, 437 times more in the call above)."""
    if method == "numba" and not neighbors.HAVE_NUMBA:
        pytest.skip("numba not installed")
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.05, seed=1)
    cell = Cell(np.array(atoms.cell))
    found = []
    for name in ("_cell_list_numba", "_cell_list_xp", "_brute_force"):
        real = getattr(neighbors, name)
        monkeypatch.setattr(neighbors, name, lambda *a, _real=real, **k: found.append(_real(*a, **k)) or found[-1])
    i, j, delta, dist = find_pairs(atoms.positions, cell, 9.0, method=method)   # the cell is 7.23 A wide
    assert len(found) == 1 and len(found[0][0]) == len(i)
    got = sorted(zip(i.tolist(), j.tolist(), np.round(dist, 8).tolist()))
    assert got == lattice_sum(atoms.positions, cell.matrix, (True, True, True), 9.0)


@pytest.mark.parametrize("pbc", [(True, True, True), (True, False, True), (False, False, True)])
@pytest.mark.parametrize("half", [False, True])
def test_cutoff_beyond_the_cell_matches_a_sum_over_images(rng, fake_gpu, pbc, half):
    """Pairs, their order and distances for a cutoff of up to 2.5 cell widths, in a triclinic
    cell with open directions, on both backends and by every method."""
    pos = rng.random((24, 3)) @ TRICLINIC
    cell = Cell(TRICLINIC, np.zeros(3), np.array(pbc))
    for cutoff in (3.0, 7.0, 12.3):                           # not 12.0, twice a cell vector (see below)
        ref = lattice_sum(pos, TRICLINIC, pbc, cutoff)
        if half:    # one of (i, j, image) and (j, i, -image)
            assert len(ref) % 2 == 0
        first = None
        for be, method in ((CPU, "auto"), (CPU, "xp"), (CPU, "brute"), (fake_gpu, "auto")):
            i, j, delta, dist = (np.asarray(a) for a in find_pairs(pos, cell, cutoff, be, half=half, method=method))
            np.testing.assert_allclose(np.linalg.norm(delta, axis=1), dist, atol=1e-12)
            assert np.all(np.diff(i) >= 0) and np.all((np.diff(j) >= 0) | (np.diff(i) > 0))   # sorted by (i, j)
            full = sorted(zip(i.tolist(), j.tolist(), np.round(dist, 8).tolist()))
            if half:
                assert 2 * len(i) == len(ref) and np.all(i <= j)
                full = sorted(full + [(b, a, d) for a, b, d in full])
            assert full == ref
            if first is None:
                first = (i, j, delta, dist)
            else:                                           # every method lists them in the same order
                np.testing.assert_array_equal(i, first[0])
                np.testing.assert_array_equal(j, first[1])
                np.testing.assert_allclose(delta, first[2], atol=1e-10)


def test_coordination_beyond_the_cell_matches_brute_force(rng):
    pos = rng.random((12, 3)) @ TRICLINIC
    data = DataCollection.from_ase(Atoms("Cu12", positions=pos, cell=TRICLINIC, pbc=True))
    # below half the cell, beyond it, three cells (not 6.0, the length of a cell vector: whether an
    # atom's own image at exactly the cutoff counts is a matter of rounding, before and now)
    for cutoff in (2.0, 6.3, 14.0):
        out = ewald.Pipeline(ewald.StaticSource(data), [CoordinationAnalysis(cutoff=cutoff)]).compute(0)
        ref = np.bincount([i for i, _, _ in lattice_sum(pos, TRICLINIC, (True,) * 3, cutoff)], minlength=12)
        assert ref.sum() > 0
        np.testing.assert_array_equal(out["Coordination"], ref)


def test_too_many_images_is_refused_before_they_are_listed(cu_fcc, monkeypatch):
    """The list of image shifts was built first and counted afterwards: a cutoff of a thousand
    cell widths made a list of 8e9 tuples before the check that refuses it."""
    class NoProduct:
        @staticmethod
        def product(*a, **k):
            raise AssertionError("the image shifts were listed before their number was checked")

    monkeypatch.setattr(neighbors, "MAX_IMAGE_ATOMS", 1000)
    monkeypatch.setattr(neighbors, "itertools", NoProduct)
    with pytest.raises(ValueError, match="half the periodic cell.*27 periodic images"):
        find_pairs(cu_fcc.positions, cu_fcc.cell, 8.0)


# --------------------------------------------------------------------------- the guard

def no_search(monkeypatch):
    """Fail the test if any kernel runs: the refusal must come from the estimate alone."""
    def fail(*a, **k):
        raise AssertionError("the search ran although the pair list cannot fit")

    for name in ("_cell_list_numba", "_cell_list_xp", "_cell_list_cuda", "_brute_force"):
        monkeypatch.setattr(neighbors, name, fail)


def test_pair_list_over_the_budget_is_refused_before_the_search(cu_fcc, budget, monkeypatch):
    budget(MiB)
    no_search(monkeypatch)
    with pytest.raises(MemoryError) as e:                    # 256 atoms, cell 14.46 A, about 600 neighbors each
        find_pairs(cu_fcc.positions, cu_fcc.cell, 12.0)
    message = str(e.value)
    assert type(e.value) is MemoryError                      # so that "except MemoryError" keeps working
    assert message.startswith("cutoff 12 needs too much memory: about ")
    # how much, why, and what to change
    assert re.search(r"about 1\.\d+e\+05 pairs for 256 atoms \(about 6\d\d neighbors each", message)
    assert "the cutoff is 0.83 times the periodic cell width 14.46" in message
    assert "26 periodic images" in message
    assert re.search(r"take \d+\.\d MiB, and 1\.0 MiB are allowed \(ewald\.backend\.MEMORY_BUDGET\)", message)
    assert re.search(r"Use a smaller cutoff: about \d\.\d+ would fit\.$", message)
    assert len(message) < 420 and "\n" not in message


@pytest.mark.parametrize("half", [False, True])
def test_the_cutoff_named_in_the_message_does_fit(cu_fcc, budget, half):
    budget(MiB)
    for cutoff in (12.0, 30.0):
        with pytest.raises(MemoryError) as e:
            find_pairs(cu_fcc.positions, cu_fcc.cell, cutoff, half=half)
        fits = float(re.search(r"about ([\d.]+) would fit", str(e.value)).group(1))
        assert 2.0 < fits < cutoff
        i, _, _, dist = find_pairs(cu_fcc.positions, cu_fcc.cell, fits, half=half)
        assert len(i) > 0 and dist.max() < fits
        # and it is not far below what fits: twice that cutoff does not (the advice is cautious
        # by up to 4 in the pair count where the smaller cutoff needs no images any more)
        with pytest.raises(MemoryError):
            find_pairs(cu_fcc.positions, cu_fcc.cell, 2 * fits, half=half)


@pytest.mark.parametrize("pbc, cutoff", [
    *(((False, False, False), c) for c in (15.0, 100.0, 1e6, np.inf)), *(((True, True, False), c) for c in (15.0, 100.0)),
], ids=lambda v: "slab" if v is True else None)
def test_the_cutoff_named_fits_with_open_directions_too(rng, budget, pbc, cutoff):
    """Without periodic images in some direction the list stops growing as cutoff^3 (it ends at
    every pair there is), so the advice is searched on the estimate, not scaled from the cutoff
    given: that said "about inf would fit", or a tenth of the cutoff six times in a row."""
    pos = rng.random((6000, 3)) * 40
    cell = Cell(np.diag([40.0, 40.0, 40.0]), pbc=pbc)
    budget(MiB)
    with pytest.raises(MemoryError) as e:
        find_pairs(pos, cell, cutoff)
    fits = float(re.search(r"about ([\d.]+) would fit\.$", str(e.value)).group(1))
    assert 1.0 < fits < 5.0
    assert len(find_pairs(pos, cell, fits)[0]) > 0
    with pytest.raises(MemoryError):
        find_pairs(pos, cell, 2 * fits)


def test_an_infinite_cutoff_needs_an_open_system(cu_fcc):
    with pytest.raises(ValueError, match="cutoff inf in a periodic cell .* give a finite cutoff"):
        find_pairs(cu_fcc.positions, cu_fcc.cell, np.inf)


def test_small_systems_skip_the_estimate(cu_fcc, monkeypatch):
    """A list of every pair of a small system is too small to refuse: no estimate, no guard
    (per-frame analyses of small cells call the search in a loop)."""
    monkeypatch.setattr(neighbors, "_MemoryGuard", lambda *a, **k: pytest.fail("built a guard"))
    monkeypatch.setattr(neighbors, "expected_pairs", lambda *a, **k: pytest.fail("estimated"))
    assert len(find_pairs(cu_fcc.positions, cu_fcc.cell, 3.0)[0]) == 256 * 12


def test_minimum_image_search_is_guarded_too(rng, budget, monkeypatch, fake_gpu):
    """A cutoff within half the cell (no images): the same guard, the same message without the
    part about images, on the CPU and on the GPU path."""
    matrix = np.diag([40.0, 40.0, 40.0])
    pos = rng.random((6000, 3)) * 40
    budget(MiB)
    no_search(monkeypatch)
    for be in (CPU, fake_gpu):
        for half in (False, True):
            with pytest.raises(MemoryError) as e:            # 6000 x 106 neighbors = 6.4e5 pairs = 29 MiB
                find_pairs(pos, Cell(matrix), 6.5, be, half=half)
            assert "periodic" not in str(e.value)
            # a half list is half as long, and each atom has as many neighbors either way
            listed = r"about 3\.\d+e\+05 pairs \(each listed once\)" if half else r"about 6\.\d+e\+05 pairs"
            assert re.search(listed + r" for 6000 atoms \(about 10\d neighbors each\) take (2\d|1\d)\.\d MiB",
                             str(e.value))


@pytest.mark.parametrize("method", ["numba", "xp", "brute"])
def test_a_denser_structure_than_estimated_is_refused_on_its_count(rng, budget, monkeypatch, method):
    """A cluster in a large periodic cell has far more pairs than atoms spread evenly would (the
    estimate, 60 pairs): the kernels check again on what they count, before the result arrays
    are allocated (numba) or grow further (the array code)."""
    if method == "numba" and not neighbors.HAVE_NUMBA:
        pytest.skip("numba not installed")
    pos = 100 + rng.random((400, 3)) * 4.0
    cell = Cell(np.diag([200.0, 200.0, 200.0]))
    assert neighbors.expected_pairs(400, cell.volume, 8.0) < 100
    passes = []
    if method == "numba":
        real = neighbors._numba_pairs
        monkeypatch.setattr(neighbors, "_numba_pairs", lambda *a: passes.append(bool(a[-1])) or real(*a))
    budget(MiB)
    with pytest.raises(MemoryError) as e:
        find_pairs(pos, cell, 8.0, method=method)            # all 400 x 399 pairs: 7.3 MiB
    if method == "numba":
        assert passes == [False]                             # counted, never filled
        assert "1.6e+05 pairs for 400 atoms (399 neighbors each) take 7.3 MiB" in str(e.value)
        assert str(e.value).endswith("Use a smaller cutoff: about 3.92 would fit.")
    else:                                                    # stopped on what was found so far
        assert re.search(r"over \d\.\d+e\+0[45] pairs for 400 atoms \(over \d+ neighbors each\)", str(e.value))
        assert re.search(r"Use a smaller cutoff: less than \d\.\d+ would fit\.$", str(e.value))
    budget(16 * MiB)
    assert len(find_pairs(pos, cell, 8.0, method=method)[0]) == 400 * 399


def test_ordinary_calls_never_ask_the_system(cu_fcc, monkeypatch):
    """Below ``SMALL_ALLOCATION`` nothing is read (a bond search runs for every frame), and the
    result is what it was."""
    def fail():
        raise AssertionError("free memory was read for a small pair list")

    monkeypatch.setattr(backend, "free_memory", fail)
    monkeypatch.setattr(backend, "_device_memory", lambda xp: fail())
    assert backend.MEMORY_BUDGET is None
    i, j, _, dist = find_pairs(cu_fcc.positions, cu_fcc.cell, 3.0)
    assert len(i) == 256 * 12 and np.allclose(dist, 3.615 / np.sqrt(2))
    assert len(find_pairs(cu_fcc.positions, cu_fcc.cell, 8.0)[0]) == 256 * 176     # beyond half the cell
    pipe = ewald.load(CU)
    pipe.add(CoordinationAnalysis(cutoff=8.0))
    assert set(pipe.compute(0)["Coordination"]) == {176}


def test_an_infinite_cutoff_on_an_open_system_still_lists_every_pair(rng):
    pos = rng.random((50, 3)) * 10
    cell = Cell(np.eye(3) * 10, np.zeros(3), np.zeros(3, dtype=bool))
    assert len(find_pairs(pos, cell, np.inf)[0]) == 50 * 49
    assert len(find_pairs(pos, Cell(np.zeros((3, 3))), 1e6, half=True)[0]) == 50 * 49 // 2


def test_estimate_never_exceeds_what_an_even_structure_lists(rng):
    """The guard refuses on the estimate, so it must not be above the truth: exact for a
    periodic cell, below it where open directions cut the neighbor spheres."""
    n, box = 1500, 24.0
    pos = rng.random((n, 3)) * box
    for pbc in [(True, True, True), (True, True, False), (False, False, False)]:
        cell = Cell(np.eye(3) * box, np.zeros(3), np.array(pbc))
        for cutoff in (3.0, 9.0, 20.0):
            true = len(find_pairs(pos, cell, cutoff)[0])
            images = any(pbc) and cutoff > box / 2
            est = neighbors.expected_pairs(n, box ** 3, cutoff, [box] * (3 - sum(pbc)), images=images)
            assert 0.5 * true < est < 1.03 * true, (pbc, cutoff, est, true)
            if all(pbc):
                assert abs(est / true - 1) < 0.03


# --------------------------------------------------------------------------- RDF

def test_rdf_and_trajectory_rdf_go_through_the_guard(budget):
    pipe = ewald.load(CU)
    d = pipe.source_frame(0)
    budget(MiB)
    with pytest.raises(MemoryError, match="cutoff 40 needs too much memory.*728 periodic images"):
        rdf(d.positions, d.cell, 40.0, types=d.types)
    with pytest.raises(MemoryError, match="Use a smaller cutoff"):
        trajectory_rdf(pipe, 40.0, partial=True)
    assert len(trajectory_rdf(pipe, 5.0, bins=50)["g(r)"]) == 50


def test_partial_rdf_of_too_many_types_is_refused(budget):
    """One histogram per ordered pair of types: labels that differ on every atom (identifiers
    passed as types) ask for n^2 of them, 29 TiB for 100 000 atoms."""
    d = ewald.load(CU).source_frame(0)
    budget(MiB)
    with pytest.raises(MemoryError) as e:
        rdf(d.positions, d.cell, 3.0, types=np.arange(d.count))
    assert str(e.value) == (
        "partial g(r) for 108 particle types needs too much memory: 108 x 108 type pairs of 200 bins take "
        "35.6 MiB, and 1.0 MiB are allowed (ewald.backend.MEMORY_BUDGET). Use partial=False (types=None), "
        "fewer bins, or types with fewer distinct values.")
    table = rdf(d.positions, d.cell, 3.0, types=d.types)     # one type: no partial columns to speak of
    assert set(table) == {"r", "g(r)", "g(r) Cu-Cu"}
    np.testing.assert_allclose(table["g(r) Cu-Cu"], table["g(r)"])


# --------------------------------------------------------------------------- the budget

def test_free_memory_of_this_machine():
    free = backend.free_memory()
    assert isinstance(free, int) and 64 * MiB < free < 2**50
    assert CPU.short_of_memory(1e3) is None and CPU.short_of_memory(float("nan")) is None
    have, words = CPU.short_of_memory(2.0**60)
    assert 64 * MiB < have < 2**50 and re.fullmatch(r"\d+\.\d [MGT]iB are free", words)


def test_linux_memory_counts_swap_and_respects_the_cgroup_limit(tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16000000 kB\nMemFree:  100 kB\nMemAvailable:    8000000 kB\n"
                       "SwapTotal: 2000000 kB\nSwapFree:        1000000 kB\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(backend, "_cgroup_limit", lambda: None)
    assert backend._linux_memory(str(meminfo)) == 9_000_000 * 1024
    monkeypatch.setattr(backend, "_cgroup_limit", lambda: 2**30)        # a container or a batch job
    assert backend._linux_memory(str(meminfo)) == 2**30
    monkeypatch.setattr(backend, "_cgroup_limit", lambda: 2**62)        # "no limit" in cgroup v1
    assert backend._linux_memory(str(meminfo)) == 9_000_000 * 1024
    meminfo.write_text("MemTotal: 16000000 kB\r\n", encoding="utf-8", newline="")   # an old kernel: no MemAvailable
    assert backend._linux_memory(str(meminfo)) is None
    assert backend._linux_memory(str(tmp_path / "missing")) is None


def test_cgroup_limit_v2_v1_and_none(tmp_path):
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    root, member = tmp_path / "cgroup", tmp_path / "self"
    write(member, "0::/slurm/uid_7/job_42\n")
    write(root / "slurm" / "uid_7" / "job_42" / "memory.max", "max\n")           # the job itself: unlimited
    write(root / "slurm" / "uid_7" / "memory.max", "4294967296\n")               # its parent: 4 GiB
    write(root / "slurm" / "memory.max", "8589934592\n")
    assert backend._cgroup_limit(str(root), str(member)) == 4 * 2**30
    write(member, "5:cpu,cpuacct:/\n4:memory:/docker/abc\n0::/\n")                # v1 with a v2 root beside it
    write(root / "memory" / "docker" / "abc" / "memory.limit_in_bytes", "1073741824\n")
    write(root / "memory" / "memory.limit_in_bytes", "9223372036854771712\n")
    assert backend._cgroup_limit(str(root), str(member)) == 2**30
    write(member, "0::/\n")
    assert backend._cgroup_limit(str(root), str(member)) is None
    assert backend._cgroup_limit(str(root), str(tmp_path / "missing")) is None   # not Linux, or no /proc


def test_free_memory_on_the_other_platforms(monkeypatch):
    """Windows asks its own API, macOS has only the installed RAM, and a platform that says
    nothing falls back to a constant the message owns up to."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(backend, "_windows_memory", lambda: 3 * 2**30)
    monkeypatch.setattr(backend, "_linux_memory", lambda: pytest.fail("/proc read on Windows"))
    assert backend.free_memory() == 3 * 2**30
    monkeypatch.setattr(backend, "_windows_memory", lambda: None)                 # the call failed
    assert backend.free_memory() is None

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(os, "sysconf", lambda name: {"SC_PHYS_PAGES": 2**21, "SC_PAGE_SIZE": 16384}[name],
                        raising=False)
    assert backend.free_memory() == 32 * 2**30
    monkeypatch.setattr(os, "sysconf", lambda name: (_ for _ in ()).throw(ValueError("unrecognized")))
    assert backend.free_memory() is None
    monkeypatch.delattr(os, "sysconf")
    assert backend.free_memory() is None

    assert CPU.short_of_memory(backend.FALLBACK_MEMORY) is None
    have, words = CPU.short_of_memory(backend.FALLBACK_MEMORY + 1)
    assert have == backend.FALLBACK_MEMORY == 8 * 2**30
    assert words.startswith("8.0 GiB are assumed free") and "ewald.backend.MEMORY_BUDGET" in words


def test_gpu_budget_is_the_device_memory(monkeypatch, fake_gpu):
    class Device:
        mem_info = (3 * 2**30, 8 * 2**30)                    # free, total

    class Pool:
        @staticmethod
        def free_bytes():
            return 2**30                                     # held by CuPy's pool, free for reuse

    class CuPyLike(_NumpyAsGPU):
        cuda = type("cuda", (), {"Device": Device})
        get_default_memory_pool = staticmethod(Pool)

    gpu = Backend("cuda", CuPyLike(), raw_kernels=True)
    assert gpu.short_of_memory(4 * 2**30) is None
    assert gpu.short_of_memory(4 * 2**30 + 1) == (4 * 2**30, "4.0 GiB are free on the GPU")
    # an array module without that interface (the test stand-ins): the host's memory
    monkeypatch.setattr(backend, "free_memory", lambda: 2 * 2**30)
    assert fake_gpu.short_of_memory(3 * 2**30) == (2 * 2**30, "2.0 GiB are free")


def test_format_bytes():
    assert [backend.format_bytes(b) for b in (0, 1023, 1536, 35.6 * MiB, 24 * 2**30, 2.0**45)] == [
        "0 bytes", "1023 bytes", "1.5 KiB", "35.6 MiB", "24.0 GiB", "32.0 TiB"]


# --------------------------------------------------------------------------- device names

def test_unknown_device_names_the_devices():
    with pytest.raises(ValueError) as e:
        ewald.load(CU).compute(0, device="cud")
    assert str(e.value) == "unknown device 'cud'. Did you mean 'cuda'? Devices: cpu, cuda, gpu, auto, opencl, metal, mps, apple"
    with pytest.raises(ValueError, match=r"unknown device 'gpus'\. Did you mean 'gpu'\? Devices: cpu, "):
        get_backend("gpus")
    with pytest.raises(ValueError, match=r"unknown device 'tpu-v4'\. Devices: cpu, cuda, "):
        get_backend("tpu-v4")
    with pytest.raises(TypeError, match="device must be a name or a Backend, not 1. Devices: cpu"):
        get_backend(1)                                       # was: 'int' object has no attribute 'lower'


def test_known_device_names_are_unchanged():
    for name in (None, "", "cpu", "CPU", "opencl", "metal", "mps", "Apple"):
        assert get_backend(name) is CPU
    assert get_backend(CPU) is CPU
    for name in ("gpu", "cuda", "GPU"):                      # "gpu" is another word for "cuda", not a typo
        try:
            assert get_backend(name).name == "cuda"
        except RuntimeError as e:                            # no CuPy or no NVIDIA GPU here
            assert "CuPy" in str(e)
    assert get_backend("auto").name in ("cpu", "cuda")
    assert set(backend.DEVICES) >= {"cpu", "cuda", "gpu", "auto", "opencl", *backend.APPLE_DEVICES}
