"""The standalone apps' build files: what they leave out, and the one Windows build (D80)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _excludes():
    spec = importlib.util.spec_from_file_location("bundle_excludes", ROOT / "packaging" / "bundle_excludes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VTK_UNUSED


def test_left_out_vtk_modules_are_not_imported():
    """No VTK module the apps leave out is imported by ewald's renderers, the GUI or pyvista."""
    if importlib.util.find_spec("vtkmodules") is None:
        return
    code = ("import sys, pyvista, ewald.render, ewald.raytrace, ewald.display\n"
            "import importlib.util\n"
            "if importlib.util.find_spec('pyvistaqt'): import pyvistaqt\n"
            "print('\\n'.join(m for m in sys.modules if m.startswith('vtkmodules.')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, encoding="utf-8", check=True).stdout
    loaded = set(out.split())
    assert loaded and not loaded & set(_excludes())
    for name in _excludes():
        assert name.startswith("vtkmodules.vtk")


def test_one_windows_build_and_gpu_from_the_app():
    flow = (ROOT / ".github" / "workflows" / "windows-release.yml").read_text(encoding="utf-8")
    spec = (ROOT / "packaging" / "windows" / "ewald.spec").read_text(encoding="utf-8")
    assert "matrix" not in flow and "EWALD_VARIANT" not in flow and "EWALD_VARIANT" not in spec
    assert "ewald.exe gpu --install" in flow and "smoke_gpu.py" in flow
    assert '"cupy"' in spec and "VTK_UNUSED" in spec and "stdlib_module_names" in spec
    iss = (ROOT / "packaging" / "windows" / "ewald.iss").read_text(encoding="utf-8")
    assert "Variant" not in iss and "win64-setup" in iss
