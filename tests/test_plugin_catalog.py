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
    assert item.removable and item.provides == {"modifier": ["CatalogShift"]} and item.id == "catalog_shift"
    pc.set_enabled(item, False)
    item = next(i for i in pc.installed() if i.id == "catalog_shift")
    assert item.state == "disabled" and not item.enabled and "CatalogShift" not in plugins.modifiers()
    pc.set_enabled(item, True)
    item = next(i for i in pc.installed() if i.id == "catalog_shift")
    assert item.enabled and "CatalogShift" in plugins.modifiers()
    dest = pc.remove_file(item)
    assert dest.parent.name == ".removed" and dest.exists() and not pc.is_installed(ours[0])
    assert not any(i.path.endswith("catalog_shift.py") for i in pc.installed())


def test_pip_command_uses_this_python():
    import sys

    assert pc.pip_command("install", "ewald-x") == [sys.executable, "-m", "pip", "install", "ewald-x"]
    assert pc.pip_command("uninstall", "ewald-x")[-2:] == ["-y", "ewald-x"]
    assert pc.pip_command("install", ["ewald-a==1", "ewald-b"])[-3:] == ["install", "ewald-a==1", "ewald-b"]


def _versioned(tmp_path, version):
    src = tmp_path / f"v{version}" / "catalog_shift.py"
    src.parent.mkdir()
    src.write_text(PLUGIN.replace("from ewald.plugins import modifier\n",
                                  f"from ewald.plugins import modifier\n\n__ewald__ = {{'version': '{version}'}}\n"))
    return src


def test_entries_match_by_id_offer_updates_and_check_compatibility(tmp_path):
    pc.install_file(_versioned(tmp_path, "0.1"))
    items = pc.installed()
    item = next(i for i in items if i.id == "catalog_shift")
    newer = pc.Entry(name="Shift (renamed)", id="catalog_shift", version="0.2",
                     source=str(_versioned(tmp_path, "0.2")), requires_ewald=">=0.0")
    by_name = pc.Entry(name="catalog_shift", version="0.1")
    future = pc.Entry(name="Future", id="catalog_shift", version="9.0", requires_ewald=">=99")
    unreadable = pc.Entry(name="Odd", requires_ewald="newest please")
    assert pc.match(newer, items) is item and pc.match(by_name, items) is item and pc.is_installed(newer, items)
    assert pc.match(pc.Entry(name="Other", id="other"), items) is None
    assert newer.compatible and not future.compatible and not unreadable.compatible
    assert pc.update_for(item, [by_name, future, newer]) is newer   # newest compatible one
    assert pc.update_for(item, [by_name]) is None

    pc.install_file(newer)                                           # the update replaces the file
    assert next(i for i in pc.installed() if i.id == "catalog_shift").version == "0.2"


def test_default_index_and_index_format(tmp_path, monkeypatch):
    monkeypatch.delenv("EWALD_PLUGIN_INDEX")
    assert pc.catalog_urls("a.json; b.json\nc.json") == [pc.DEFAULT_CATALOG, "a.json", "b.json", "c.json"]
    assert pc.catalog_urls("", default=False) == []
    monkeypatch.setenv("EWALD_PLUGIN_INDEX", "")
    assert pc.catalog_urls("a.json") == ["a.json"]
    newer = tmp_path / "newer.json"
    newer.write_text(json.dumps({"format": 2, "plugins": []}))
    _, errors = pc.catalog([str(newer), str(tmp_path / "gone.json")])
    assert "newer than this ewald reads" in errors[str(newer)]
    assert pc.is_missing(errors[str(tmp_path / "gone.json")]) and not pc.is_missing(errors[str(newer)])
    assert pc.is_missing("HTTPError: HTTP Error 404: Not Found")
    assert pc.Entry(name="x", source="pip:ewald-x[gpu]>=1").package == "ewald-x"
    assert pc.Entry(name="x", source="pip:git+https://github.com/a/b").package == ""


def test_github_token_goes_only_to_github_hosts(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    assert pc.github_token() == "secret-token"
    raw = pc._request("https://raw.githubusercontent.com/frssp/ewald-plugins/main/index.json")
    assert raw.unredirected_hdrs["Authorization"] == "token secret-token"
    assert not raw.headers                                            # not passed on to a redirect
    for url in ("https://example.org/index.json", "http://raw.githubusercontent.com/x",
                "https://raw.githubusercontent.com.evil.org/x"):
        assert not pc._request(url).has_header("Authorization"), url

    env = pc.pip_env("git+https://github.com/frssp/private-plugin")
    assert "secret-token" not in " ".join(pc.pip_command("install", "git+https://github.com/frssp/private-plugin"))
    import base64

    key = next(k for k in env if k.startswith("GIT_CONFIG_KEY_"))
    value = env[key.replace("KEY", "VALUE")]
    assert env[key] == "http.https://github.com/.extraheader"
    assert base64.b64decode(value.split()[-1]).decode() == "x-access-token:secret-token"
    assert pc.pip_env("ewald-x") == {} and pc.pip_env("git+https://gitlab.com/a/b") == {}
    assert pc.pip_env(["ewald-x", "git+https://github.com/frssp/private-plugin"]) == env   # a list: its GitHub source
    monkeypatch.delenv("GITHUB_TOKEN")
    monkeypatch.setattr(pc.shutil, "which", lambda name: None)
    assert pc.github_token() == "" and pc.pip_env("git+https://github.com/a/b") == {}


def test_pip_update_upgrades():
    assert pc.pip_command("update", "ewald-x")[-3:] == ["install", "--upgrade", "ewald-x"]
