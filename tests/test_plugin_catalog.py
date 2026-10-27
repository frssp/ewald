import json
import shutil
import sys

import pytest

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


def test_gh_token_opens_no_console_window_on_windows(monkeypatch):
    """ewald started from a shortcut has no console; a child process without CREATE_NO_WINDOW
    flashes one on Windows."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(pc.shutil, "which", lambda name: "gh")
    monkeypatch.setattr(pc.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)   # as on Windows
    seen = {}

    def run(cmd, **kw):
        seen.update(kw)
        return pc.subprocess.CompletedProcess(cmd, 0, stdout="tok\r\n", stderr="")

    monkeypatch.setattr(pc.subprocess, "run", run)
    assert pc.github_token() == "tok"
    assert seen["creationflags"] == 0x08000000 and seen["encoding"] == "utf-8"


def test_pip_update_upgrades():
    assert pc.pip_command("update", "ewald-x")[-3:] == ["install", "--upgrade", "ewald-x"]


def test_downloads_use_a_tls_context(monkeypatch):
    # company proxies re-sign HTTPS; the OS store (truststore) must be what urlopen is given
    import ssl

    from ewald._tls import ssl_context

    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"[]"

    def fake(req, timeout=None, context=None):
        seen["context"] = context
        return Resp()

    monkeypatch.setattr(pc.urllib.request, "urlopen", fake)
    pc._read("https://example.org/index.json", 5)
    assert isinstance(seen["context"], ssl.SSLContext)
    assert isinstance(ssl_context(), ssl.SSLContext)


def test_pip_command_falls_back_to_uv_without_pip(monkeypatch):
    # a uv-made environment has no pip; uv needs the OS certificate store behind a re-signing
    # proxy, asked for through the environment: --system-certs only exists in newer uv and
    # --native-tls only in older uv, and either flag is an error on the other (user, Windows)
    import sys

    monkeypatch.setattr(pc.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(pc.shutil, "which", lambda name: "/usr/bin/uv")
    cmd = pc.pip_command("install", "ewald-x")
    assert cmd[:3] == ["/usr/bin/uv", "pip", "install"]
    assert not any(a in cmd for a in ("--system-certs", "--native-tls"))
    assert pc.pip_env("ewald-x") == {"UV_NATIVE_TLS": "1", "UV_SYSTEM_CERTS": "1"}
    assert cmd[cmd.index("--python") + 1] == sys.executable and cmd[-1] == "ewald-x"
    assert "--upgrade" in pc.pip_command("update", "ewald-x")
    un = pc.pip_command("uninstall", "ewald-x")
    assert un[:3] == ["/usr/bin/uv", "pip", "uninstall"] and "-y" not in un and un[-1] == "ewald-x"


def test_uv_env_keeps_the_github_token(monkeypatch):
    monkeypatch.setattr(pc.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(pc.shutil, "which", lambda name: "C:\\Users\\me\\.local\\bin\\uv.exe")
    monkeypatch.setattr(pc, "github_token", lambda: "tok")
    env = pc.pip_env("git+https://github.com/frssp/ewald-plugins@abc#subdirectory=packages/ewald-plumed")
    assert env["UV_NATIVE_TLS"] == env["UV_SYSTEM_CERTS"] == "1"
    assert any(v.startswith("AUTHORIZATION: basic ") for v in env.values())
    assert pc.pip_command("install", "x")[0].endswith("uv.exe")


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_real_uv_accepts_the_command(tmp_path):
    # the installed uv parses the command and the environment (a dry run, nothing downloaded)
    import os
    import subprocess

    cmd = [shutil.which("uv"), "pip", "install", "--python", sys.executable, "--dry-run", "--offline", "--no-deps", "ewald"]
    r = subprocess.run(cmd, env=os.environ | pc.UV_TLS_ENV, capture_output=True, encoding="utf-8")
    assert "unexpected argument" not in r.stderr, r.stderr


def test_install_checks_the_catalog_sha256_before_writing(tmp_path):
    from ewald.trust import UntrustedCode, sha256

    src = tmp_path / "src" / "hashed_shift.py"
    src.parent.mkdir()
    src.write_text(PLUGIN.replace("CatalogShift", "HashedShift"))
    good = pc.Entry(name="Hashed shift", source=str(src), sha256=sha256(src.read_bytes()).upper())
    bad = pc.Entry(name="Hashed shift", source=str(src), sha256="0" * 64)
    with pytest.raises(UntrustedCode, match="sha256"):
        pc.install_file(bad)
    assert not (plugins.plugin_dir() / "hashed_shift.py").exists()   # nothing written or imported
    assert pc.install_file(good) == ["HashedShift"]
    pc.remove_file(next(i for i in pc.installed() if i.id == "hashed_shift"))


def test_catalog_file_without_a_version_is_not_offered_its_own_update(tmp_path):
    from ewald.trust import sha256

    src = tmp_path / "src" / "unversioned_shift.py"           # no __ewald__ version in the file
    src.parent.mkdir()
    text = PLUGIN.replace("CatalogShift", "UnversionedShift")
    src.write_bytes(text.replace("\n", "\r\n").encode())      # a Windows checkout of the index repo
    digest = sha256(src.read_bytes())
    entry = pc.Entry(name="Unversioned shift", id="unversioned_shift", version="0.1.0", source=str(src), sha256=digest)
    pc.install_file(entry)
    item = next(i for i in pc.installed() if i.id == "unversioned_shift")
    assert item.version == "0.1.0"                            # remembered from the entry it came from
    assert pc.update_for(item, [entry]) is None
    newer = pc.Entry(name="Unversioned shift", id="unversioned_shift", version="0.2.0", source=str(src), sha256="ab" * 32)
    assert pc.update_for(item, [entry, newer]) is newer

    # a copy dropped into the plugin folder by hand: no record, but the same bytes as the entry
    pc._sources_path().unlink()
    item = next(i for i in pc.installed() if i.id == "unversioned_shift")
    assert item.version == "" and pc.update_for(item, [entry]) is None
    assert pc.update_for(item, [newer]) is newer              # different content, so it is an update
    assert pc.update_for(item, [pc.Entry(name="x", id="unversioned_shift", version="0.3")]) is None   # nothing to compare

    pc.install_file(entry)
    target = plugins.plugin_dir() / src.name
    target.write_text(target.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    assert pc.installed_version(target) == ""                 # edited since: not that version any more
    pc.install_file(entry)
    pc.remove_file(next(i for i in pc.installed() if i.id == "unversioned_shift"))
    assert src.name.lower() not in pc._sources()


def test_restart_reason_names_loaded_packages_pip_changed(monkeypatch):
    before = pc.loaded_versions()
    assert before["numpy"]                                   # imported by ewald itself
    assert pc.restart_reason("install", before=before) == ""   # pip changed nothing loaded
    reason = pc.restart_reason("update", before=dict(before, numpy="0.0.1"))
    assert "numpy 0.0.1 -> " in reason and "already loaded" in reason
    gone = pc.restart_reason("uninstall", before={"ewald-surely-not-installed": "1.0"})
    assert "ewald-surely-not-installed 1.0 (removed)" in gone
    assert pc.restart_reason("enable", before={"numpy": "0.0.1"}) == ""   # only pip actions compare


def test_restart_reason_for_switching_on_a_loaded_package_and_updating_a_folder(tmp_path):
    loaded = pc.Installed("NumPy", "numpy", package="numpy")
    assert "after a restart" in pc.restart_reason("enable", loaded)
    assert pc.restart_reason("enable", pc.Installed("New", "x", package="ewald-surely-not-installed")) == ""
    assert pc.restart_reason("disable", loaded) == ""         # its contributions leave the registries at once
    folder = tmp_path / "pkg_plugin"
    folder.mkdir()
    one = tmp_path / "one.py"
    one.write_text("x = 1\n")
    assert "folder plugin" in pc.restart_reason("update", pc.Installed("Pkg", str(folder), path=str(folder)))
    assert pc.restart_reason("update", pc.Installed("One", str(one), path=str(one))) == ""   # a file is run again
    assert pc.restart_reason("update", None) == ""


def test_restart_command_runs_this_python_or_the_frozen_app(monkeypatch):
    nb = r"C:\Users\kim\Documents\cu void.ewd"                # a Windows path goes through as one argument
    assert pc.restart_command([nb]) == [sys.executable, "-m", "ewald", "gui", nb]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert pc.restart_command([nb]) == [sys.executable, nb]
    assert pc.restart_command() == [sys.executable]


def _dist_info(site, name, version, record=None, top_level=None, crlf=False):
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    nl = "\r\n" if crlf else "\n"
    (info / "METADATA").write_text(f"Metadata-Version: 2.1{nl}Name: {name}{nl}Version: {version}{nl}",
                                   encoding="utf-8", newline="")
    if record is not None:
        (info / "RECORD").write_text(nl.join(record) + nl, encoding="utf-8", newline="")
    if top_level is not None:
        (info / "top_level.txt").write_text(nl.join(top_level) + nl, encoding="utf-8", newline="")


def test_module_distributions_reads_metadata_without_touching_the_installed_files(tmp_path, monkeypatch):
    """The Plugins side bar froze for seconds (minutes on a big Windows environment) because
    importlib.metadata.packages_distributions stats every installed file; ours reads RECORD only."""
    import importlib.metadata as md
    from pathlib import Path

    site = tmp_path / "site"
    _dist_info(site, "ewald-mp", "0.1.0", record=[       # the files themselves are not there
        "ewald_mp/__init__.py,sha256=x,10", "ewald_mp/reader.py,,", "ewald_mp-0.1.0.dist-info/METADATA,,",
        "../../bin/ewald-mp,,", "__pycache__/x.pyc,,"])
    _dist_info(site, "fastext", "2.0", crlf=True, record=[   # written on Windows: CRLF, backslashes, a .pyd
        "fastext\\__init__.py,,", "_fastext.cp313-win_amd64.pyd,,", "fastext.pth,,", "single.py,,"])
    _dist_info(site, "Old-Egg", "1.0", top_level=["oldegg", "oldegg_helpers"])
    _dist_info(site, "editable-thing", "0.3", record=["__editable__.editable_thing-0.3.pth,,"])
    real = md.distributions
    monkeypatch.setattr(md, "distributions", lambda: real(path=[str(site)]))
    stats = []
    real_stat = Path.stat
    monkeypatch.setattr(Path, "stat", lambda self, **k: (stats.append(self), real_stat(self, **k))[1])
    found = pc._module_distributions()
    assert found["ewald_mp"] == ["ewald-mp"]
    assert found["fastext"] == ["fastext"] and found["_fastext"] == ["fastext"] and found["single"] == ["fastext"]
    assert found["oldegg"] == ["Old-Egg"] and found["oldegg_helpers"] == ["Old-Egg"]
    assert found["editable_thing"] == ["editable-thing"]           # an editable install: by its name
    assert ".." not in found and "__pycache__" not in found and "fastext.pth" not in found
    assert not [s for s in stats if "ewald_mp" in s.parts or "fastext" in s.parts]   # no installed file was looked at


def test_github_sources_become_archives_without_git(monkeypatch):
    # the standalone app on a PC without git: pip fetches GitHub's zip of the pinned commit
    src = "git+https://github.com/frssp/ewald-plugins@2ac10da#subdirectory=packages/ewald-mp"
    zip_ = "https://github.com/frssp/ewald-plugins/archive/2ac10da.zip#subdirectory=packages/ewald-mp"
    assert pc.github_archive(src) == zip_
    assert pc.github_archive("ewald-mp @ " + src) == "ewald-mp @ " + zip_
    assert pc.github_archive("git+https://github.com/frssp/ewald-simagent.git@958a94a").endswith(
        "/frssp/ewald-simagent/archive/958a94a.zip")
    for same in ("ewald-x", "git+https://github.com/frssp/x", "git+https://gitlab.com/a/b@v1"):
        assert pc.github_archive(same) == same
    monkeypatch.setattr(pc.shutil, "which", lambda name: None)
    assert pc.pip_command("install", src)[-1] == zip_
    monkeypatch.setattr(pc.shutil, "which", lambda name: "/usr/bin/" + name)
    assert pc.pip_command("install", src)[-1] == src


def test_frozen_app_runs_pip_through_ewald_exe(monkeypatch, tmp_path):
    # ewald-gui.exe is no python: `ewald-gui.exe -m pip` did nothing (user, Windows exe)
    import sys

    app = tmp_path / "ewald"
    app.mkdir()
    for name in ("ewald-gui.exe", "ewald.exe"):
        (app / name).write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(app), raising=False)
    monkeypatch.setattr(sys, "executable", str(app / "ewald-gui.exe"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "local"))
    monkeypatch.setattr(pc.Path, "home", staticmethod(lambda: tmp_path / "home"))
    cmd = pc.pip_command("install", "ewald-x")
    assert cmd[:4] == [str(app / "ewald.exe"), "-m", "pip", "install"] and cmd[-1] == "ewald-x"
    prefix = cmd[cmd.index("--prefix") + 1]
    assert prefix == str(pc.addons_prefix()) and "--no-build-isolation" in cmd
    constraints = pc.Path(cmd[cmd.index("--constraint") + 1])
    assert constraints.parent == pc.addons_prefix() and constraints.is_file()   # nothing bundled in tmp: empty
    assert pc.pip_command("uninstall", "ewald-x") == [str(app / "ewald.exe"), "-m", "pip", "uninstall", "-y", "ewald-x"]
    assert pc.pip_command("update", "ewald-x")[3:5] == ["install", "--upgrade"]


def test_bundled_constraints_pin_what_the_app_ships(monkeypatch, tmp_path):
    import sys
    from pathlib import Path

    import numpy

    bundle = Path(numpy.__file__).resolve().parent.parent      # this site-packages plays the bundle
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(pc, "addons_prefix", lambda: tmp_path / "python")
    pins = pc.bundled_constraints().read_text(encoding="utf-8").split()
    assert f"numpy=={numpy.__version__}" in pins and all("==" in p for p in pins)
