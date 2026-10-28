import json

import pytest

from ewald import settings


def test_defaults_roundtrip_and_only_changes_saved(tmp_path):
    s = settings.load()
    assert s.path.parent == tmp_path / "config"
    assert s["viewport.rotation_step"] == 15.0 and s.is_default("viewport.rotation_step")
    s["viewport.background"] = "white"
    s["viewport.rotation_step"] = 30   # int coerced to float
    s["compute.device"] = "cpu"        # equal to default -> not stored
    s.save()
    stored = json.loads(s.path.read_text())
    assert stored == {"viewport.background": "white", "viewport.rotation_step": 30.0}
    again = settings.load()
    assert again["viewport.rotation_step"] == 30.0 and again["viewport.background"] == "white"


def test_validation_and_bad_files(tmp_path):
    s = settings.Settings()
    with pytest.raises(KeyError):
        s["no.such"] = 1
    with pytest.raises(ValueError):
        s["compute.device"] = "tpu"
    with pytest.raises(ValueError):
        s["viewport.rotation_step"] = 500
    with pytest.raises(ValueError):
        s["general.restore_layout"] = 1
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"compute.device": "tpu", "unknown": 3, "render.dpi": 300}))
    loaded = settings.load(p)
    assert loaded["compute.device"] == "cpu" and loaded["render.dpi"] == 300
    p.write_text("{not json")
    assert settings.load(p).changed() == {}


def test_element_scheme_and_overrides():
    from ase.data import covalent_radii, vdw_radii
    from ase.data.colors import cpk_colors, jmol_colors

    s = settings.Settings()
    assert s.element_color("Cu") == pytest.approx(tuple(jmol_colors[29]))
    assert s.element_radius("Cu") == pytest.approx(0.8 * covalent_radii[29])
    assert s.element_color("X1") is None
    s["elements.color_scheme"] = "CPK"
    s["elements.radius_source"] = "vdW (Bondi)"
    s["elements.radius_scale"] = 1.0
    assert s.element_color("O") == pytest.approx(tuple(cpk_colors[8]))
    assert s.element_radius("O") == pytest.approx(vdw_radii[8])
    s["elements.radius_source"] = "ionic"                   # most common state at CN VI: O2-
    assert s.element_radius("O") == pytest.approx(1.40)
    s["elements.overrides"] = {"Cu": {"color": "#c8794a", "radius": 1.28}}
    assert s.element_color("Cu") == "#c8794a" and s.element_radius("Cu") == 1.28


def test_devices_and_components_report():
    keys = [k for k, _label, _probe in settings.DEVICES]
    assert keys[:2] == ["cpu", "cuda"] and set(keys) <= set(settings.OPTION["compute.device"].choices)
    for _k, _label, probe in settings.DEVICES:
        ok, detail = probe()
        assert isinstance(ok, bool) and isinstance(detail, str)
    mods = {row[0] for row in settings.component_status()}
    assert {"cupy", "ovito"} <= mods


def test_every_option_has_a_page():
    for opt in settings.OPTIONS:
        assert opt.page in settings.PAGES
        assert settings.Settings()[opt.key] == opt.default


def test_installable_components(monkeypatch):
    """Optional components with an Install button (D80): CuPy with the CUDA wheels, not on a Mac."""
    monkeypatch.setattr(settings.sys, "platform", "win32")
    assert settings.installable("cupy") == ["cupy-cuda12x[ctk]"]
    assert settings.installable("pyopencl") and settings.installable("ovito") == []
    monkeypatch.setattr(settings.sys, "platform", "darwin")
    assert settings.installable("cupy") == []
