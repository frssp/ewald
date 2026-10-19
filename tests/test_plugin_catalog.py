import json

from ewald import plugin_catalog as pc
from ewald import plugins

PLUGIN = '''"""Shift every particle along z.

More text for the details pane.
"""
from ewald.pipeline import Modifier
from ewald.plugins import modifier


@modifier(category="Test")
class CatalogShift(Modifier):
    def __init__(self, dz: float = 1.0):
        self.dz = dz

    def modify(self, data, ctx):
        out = data.copy()
        out.positions = data.positions + [0, 0, self.dz]
        return out
'''


def _plugin(tmp_path):
    src = tmp_path / "src" / "catalog_shift.py"
    src.parent.mkdir()
    src.write_text(PLUGIN)
    return src


def test_describe_file_reads_kinds_and_summary_without_importing(tmp_path):
    e = pc.describe_file(_plugin(tmp_path))
    assert e.name == "Catalog shift" and e.kinds == ["modifier"]
    assert e.summary == "Shift every particle along z." and "details pane" in e.description


def test_catalog_index_install_search_and_remove(tmp_path):
    src = _plugin(tmp_path)
    index = tmp_path / "src" / "index.json"
    index.write_text(json.dumps({"plugins": [
        {"name": "Catalog shift", "summary": "Shift along z", "kinds": ["modifier"], "source": "catalog_shift.py"},
        {"name": "ewald-nothing", "summary": "A package", "kinds": ["reader"], "source": "pip:ewald-nothing>=1"}]}))
    entries, errors = pc.catalog([str(index), str(tmp_path / "missing.json")])
    ours = [e for e in entries if e.catalog == str(index)]
    assert [e.name for e in ours] == ["Catalog shift", "ewald-nothing"]
    assert ours[0].source == str(src) and ours[1].is_pip and not ours[1].file_name
    assert list(errors) == [str(tmp_path / "missing.json")]
    assert pc.search(ours, "shift z") == [ours[0]] and pc.search(ours, "reader") == [ours[1]]

    assert not pc.is_installed(ours[0])
    assert pc.install_file(ours[0]) == ["CatalogShift"]
    assert pc.is_installed(ours[0]) and "CatalogShift" in plugins.modifiers()
    item = next(i for i in pc.installed() if i.path.endswith("catalog_shift.py"))
    assert item.removable and item.provides == {"modifier": ["CatalogShift"]}
    dest = pc.remove_file(item)
    assert dest.parent.name == ".removed" and dest.exists() and not pc.is_installed(ours[0])
    assert not any(i.path.endswith("catalog_shift.py") for i in pc.installed())


def test_pip_command_uses_this_python():
    import sys

    assert pc.pip_command("install", "ewald-x") == [sys.executable, "-m", "pip", "install", "ewald-x"]
    assert pc.pip_command("uninstall", "ewald-x")[-2:] == ["-y", "ewald-x"]
