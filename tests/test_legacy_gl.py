"""The OpenGL 1.x viewport for SSH X11 forwarding: geometry, camera and picking math (no GL context)."""
import numpy as np
import pytest

pytest.importorskip("pyvista")
from ewald.gui import legacy_gl  # noqa: E402  (Qt-free)
from ewald.gui.legacy_gl import CameraPlotter, align_matrices, pick_ray, ray_spheres  # noqa: E402


def test_unit_geometry():
    v, f = legacy_gl.unit_sphere(12)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1)
    assert f.max() < len(v) and f.dtype == np.uint32
    v, n, f = legacy_gl.unit_cylinder(8)
    assert v[:, 2].min() == 0 and v[:, 2].max() == 1 and f.max() < len(v)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1)


def test_align_matrices_map_unit_cylinder_to_segments():
    start = np.array([[0.0, 0, 0], [1, 2, 3], [0, 0, 0]])
    end = np.array([[0.0, 0, 2], [4, 6, 3], [1, 1e-9, 0]])
    m = align_matrices(start, end, [0.5, 0.2, 0.1])
    for k in range(3):
        mat = m[k].T  # stored column-major
        np.testing.assert_allclose((mat @ [0, 0, 0, 1])[:3], start[k], atol=1e-12)
        np.testing.assert_allclose((mat @ [0, 0, 1, 1])[:3], end[k], atol=1e-9)
        side = (mat @ [1, 0, 0, 0])[:3]  # radius direction: perpendicular, of length r
        assert abs(side @ (end[k] - start[k])) < 1e-9
        np.testing.assert_allclose(np.linalg.norm(side), [0.5, 0.2, 0.1][k])


def test_ray_spheres_returns_nearest_hit():
    centers = np.array([[0.0, 0, 5], [0, 0, 2], [3, 0, 2]])
    radii = np.array([1.0, 0.5, 0.5])
    assert ray_spheres(np.zeros(3), np.array([0, 0, 1.0]), centers, radii) == 1
    assert ray_spheres(np.zeros(3), np.array([0, 1.0, 0]), centers, radii) is None


@pytest.mark.parametrize("parallel", [True, False])
def test_camera_fit_and_pick_ray_through_screen_center(parallel):
    from ewald import DataCollection
    from ewald.render import apply_camera_state, camera_state, setup_camera

    ase = pytest.importorskip("ase.build")
    data = DataCollection.from_ase(ase.bulk("Cu", "fcc", a=3.615, cubic=True).repeat(2))
    p = CameraPlotter()
    p.bounds = np.array([0, 7.23, 0, 7.23, 0, 7.23])
    setup_camera(p, data, "iso", parallel)
    cam = p.camera
    np.testing.assert_allclose(cam.focal_point, [3.615] * 3, atol=1e-6)
    # the view direction is kept, and the whole box is inside the view
    d = np.asarray(cam.focal_point) - cam.position
    np.testing.assert_allclose(d / np.linalg.norm(d), -np.ones(3) / np.sqrt(3), atol=1e-6)
    origin, direction = pick_ray(p, 100, 75, 200, 150)
    np.testing.assert_allclose(direction, d / np.linalg.norm(d), atol=1e-6)
    i = ray_spheres(origin, direction, data.positions, np.full(data.count, 1.28))
    assert i is not None and np.linalg.norm(np.cross(data.positions[i] - cam.focal_point, direction)) < 1.28
    # corners of the box project inside the window
    proj, view = p.matrices(200 / 150)
    corners = np.array([[x, y, z, 1] for x in (0, 7.23) for y in (0, 7.23) for z in (0, 7.23)])
    clip = corners @ (proj.T @ view.T).T
    ndc = clip[:, :3] / clip[:, 3:]
    assert np.abs(ndc).max() <= 1
    state = camera_state(p)  # sessions and the render page see a PyVista-like plotter
    q = CameraPlotter()
    apply_camera_state(q, state)
    assert camera_state(q) == pytest.approx(state)


def test_viewport_backend_choice(monkeypatch):
    from ewald.gui import QtStartupError, viewport_backend

    for k in ("EWALD_VIEWPORT", "SSH_CONNECTION", "SSH_CLIENT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    assert viewport_backend() == "vtk" and viewport_backend("legacy-gl") == "legacy-gl"
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 5 6.7.8.9 22")
    monkeypatch.setenv("DISPLAY", "localhost:10.0")
    assert viewport_backend("auto") == "legacy-gl" and viewport_backend("vtk") == "vtk"
    monkeypatch.setenv("EWALD_VIEWPORT", "vtk")
    assert viewport_backend("legacy-gl") == "vtk"
    monkeypatch.setenv("EWALD_VIEWPORT", "metal")
    with pytest.raises(QtStartupError):
        viewport_backend()


@pytest.mark.parametrize("found, backend", [
    (((4, 6), "NVIDIA GeForce RTX 4070/PCIe/SSE2"), "vtk"),
    (((3, 2), "Intel(R) HD Graphics"), "vtk"),
    (((1, 1), "GDI Generic"), "legacy-gl"),                    # Windows' own OpenGL: VMs, Remote Desktop
    (((3, 1), "Intel(R) HD Graphics 3000"), "legacy-gl"),
    ((None, "OSError: [WinError 126] opengl32.dll"), "legacy-gl"),
])
def test_windows_without_opengl_3_2_uses_the_software_legacy_viewport(monkeypatch, found, backend):
    import ewald.gui as gui

    for k in ("EWALD_VIEWPORT", "SSH_CONNECTION", "SSH_CLIENT", "EWALD_SKIP_GL_PROBE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(gui, "_windows", lambda: True)
    monkeypatch.setattr(gui, "_windows_gl_cache", [])
    probes = []
    monkeypatch.setattr(gui, "_wgl_probe", lambda: probes.append(1) or found)
    assert gui.viewport_backend("auto") == backend
    assert gui.viewport_backend("vtk") == "vtk"                 # an explicit choice is kept
    gui.viewport_backend("auto")
    assert probes == [1]                                        # asked once per process

    attrs = []
    monkeypatch.setattr("PySide6.QtCore.QCoreApplication.setAttribute", lambda a, on=True: attrs.append(a))
    note = gui.use_software_gl_if_needed(backend)
    if backend == "vtk":
        assert note is None and not attrs
    else:
        from PySide6 import QtCore

        assert attrs == [QtCore.Qt.AA_UseSoftwareOpenGL] and "software rendering" in note
        assert (found[1] if found[0] is None else "GDI Generic" if found[0] == (1, 1) else "3.1") in note


def test_opengl_check_is_windows_only(monkeypatch):
    import ewald.gui as gui

    monkeypatch.setattr(gui, "_windows", lambda: False)
    monkeypatch.setattr(gui, "_wgl_probe", lambda: pytest.fail("probed off Windows"))
    assert not gui.windows_needs_software_gl()
    assert gui.parse_gl_version("4.6.0 NVIDIA 551.23") == (4, 6) and gui.parse_gl_version("1.1.0") == (1, 1)
    assert gui.parse_gl_version("garbage") is None
    monkeypatch.setattr(gui, "_windows", lambda: True)
    monkeypatch.setattr(gui, "_windows_gl_cache", [])
    monkeypatch.setenv("EWALD_SKIP_GL_PROBE", "1")
    assert not gui.windows_needs_software_gl()
