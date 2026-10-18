import numpy as np
import pytest

from ewald import Pipeline, StaticSource
from ewald.backend import Backend, get_backend
from ewald.modifiers import (AssignColor, CalculateDisplacements, ColorCoding, ComputeProperty,
                             CoordinationAnalysis, DeleteSelected, ExpressionSelection, Replicate,
                             SelectType, Slice)


def test_fcc_coordination_and_rdf(cu_fcc):
    pipe = Pipeline(StaticSource(cu_fcc), [CoordinationAnalysis(cutoff=3.0, number_of_bins=60)])
    d = pipe.compute(0)
    assert np.all(d["Coordination"] == 12)
    rdf = d.tables["coordination-rdf"]
    assert abs(rdf["r"][np.argmax(rdf["g(r)"])] - 3.615 / np.sqrt(2)) < 0.06


def test_gpu_kernels_run_on_any_array_module(cu_fcc):
    """The CUDA path is the same xp code; run it with numpy standing in for CuPy."""
    fake_gpu = Backend("cuda-sim", np)
    d = Pipeline(StaticSource(cu_fcc), [CoordinationAnalysis(cutoff=3.0)]).compute(0, device=fake_gpu)
    assert np.all(d["Coordination"] == 12)


def test_cuda_request_without_cupy_has_clear_error():
    try:
        import cupy  # noqa: F401
        pytest.skip("CuPy present")
    except ImportError:
        with pytest.raises(RuntimeError, match="CuPy"):
            get_backend("cuda")


def test_selection_delete_and_python_modifier(cu_fcc):
    calls = []

    def tag(frame, data):
        calls.append(frame)
        data.particles["Tag"] = np.arange(data.count)

    pipe = Pipeline(StaticSource(cu_fcc))
    pipe.modifiers.append(tag)
    pipe.modifiers.append(ExpressionSelection("(Position.Z < 3.0) & (ParticleType == 'Cu')"))
    pipe.modifiers.append(DeleteSelected())
    d = pipe.compute(0)
    assert calls == [0]
    assert d.count == cu_fcc.count - int((cu_fcc.positions[:, 2] < 3.0).sum())
    assert "Selection" not in d
    # source frame must stay untouched by modifiers
    assert "Tag" not in pipe.source_frame(0)


def test_expression_error_hint(cu_fcc):
    pipe = Pipeline(StaticSource(cu_fcc), [ExpressionSelection("Position.Z > 1 and Position.X > 1")])
    with pytest.raises(ValueError, match="&"):
        pipe.compute(0)


def test_displacement_across_pbc(cu_fcc):
    moved = cu_fcc.copy()
    pos = cu_fcc.positions.copy()
    L = cu_fcc.cell.matrix[0, 0]
    pos[0, 0] = pos[0, 0] - 0.3 + (L if pos[0, 0] < 0.3 else 0)  # wraps around x
    moved.positions = pos
    moved.particles["Particle Identifier"] = np.arange(cu_fcc.count)
    ref = cu_fcc.copy()
    ref.particles["Particle Identifier"] = np.arange(cu_fcc.count)
    d = Pipeline(StaticSource([ref, moved]), [CalculateDisplacements(0)]).compute(1)
    np.testing.assert_allclose(d["Displacement"][0], [-0.3, 0, 0], atol=1e-9)
    assert d.attributes["MSD"] == pytest.approx(0.09 / cu_fcc.count)


def test_slice_replicate_color(cu_fcc):
    pipe = Pipeline(StaticSource(cu_fcc), [
        Replicate((2, 1, 1)),
        Slice(normal=(0, 0, 1), distance=1.8, width=1.0),
        SelectType("Cu"), AssignColor("red"),
        ComputeProperty("Height", "Position.Z - CellOrigin.Z"),
        ColorCoding("Height", cmap="plasma"),
    ])
    d = pipe.compute(0)
    assert d.count == 2 * int((np.abs(cu_fcc.positions[:, 2] - 1.8) <= 0.5).sum())
    assert d["Color"].shape == (d.count, 3)
    assert d.attributes["ColorCoding.range"][0] == pytest.approx(d["Height"].min())
