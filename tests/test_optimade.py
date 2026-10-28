"""OPTIMADE databases (ewald.optimade, built in since it left the ewald-optimade plugin):
addresses, search box text, the client against a local server, the reader, the AI tool, the CLI
and the side bar."""
import json
import warnings

import numpy as np
import pytest

from ewald import load
from ewald.optimade import client, parse, providers, search, to_filter
from ewald.optimade.query import parse_formula, reduced_formula
from helpers import optimade_server


@pytest.fixture
def server(monkeypatch):
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    srv = optimade_server.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def testdb(server):
    """The stored OPTIMADE settings point database ``test`` (the default) at the server; the
    lines end in CRLF, as a settings file edited on Windows may."""
    from ewald.settings import config_dir

    d = config_dir() / "plugin-settings"
    d.mkdir(parents=True, exist_ok=True)
    extra = f"# lab server\r\ntest = http://127.0.0.1:{server.server_port}/\r\n"
    (d / "optimade.json").write_text(json.dumps({"default": "test", "extra": extra, "page_size": 10}),
                                     encoding="utf-8", newline="\n")
    return server


# ------------------------------------------------------------------ search box text -> filter
@pytest.mark.parametrize("text, flt", [
    ("LiFePO4", 'chemical_formula_reduced="FeLiO4P"'),
    ("Ca(OH)2", 'chemical_formula_reduced="CaH2O2"'),
    ("Fe4O6", 'chemical_formula_reduced="Fe2O3"'),
    ("NaCl", 'chemical_formula_reduced="ClNa"'),
    ("Si", 'chemical_formula_reduced="Si"'),
    ("Li-P-S", 'elements HAS ALL "Li","P","S" AND nelements=3'),
    ("S-Li-P-Li", 'elements HAS ALL "Li","P","S" AND nelements=3'),
    ("Li P S", 'elements HAS ALL "Li","P","S"'),
    ("Li, S", 'elements HAS ALL "Li","S"'),
    ('elements HAS ALL "Si","O" AND nsites<=12', 'elements HAS ALL "Si","O" AND nsites<=12'),
    ("nsites<4", "nsites<4"),
    ("", ""),
])
def test_to_filter(text, flt):
    assert to_filter(text) == flt


@pytest.mark.parametrize("bad", ["Xx2", "Li-Qq", "Li Zz", "Fe(O", "Fe)2", "Fe2.5O", "fe2o3"])
def test_to_filter_rejects(bad):
    with pytest.raises(ValueError):
        to_filter(bad)


def test_reduced_formula():
    assert reduced_formula(parse_formula("Mg(OH)2")) == "H2MgO2"
    assert reduced_formula(parse_formula("C6H12O6")) == "CH2O"


# ------------------------------------------------------------------ addresses and databases
def test_parse_uri():
    u = parse("optimade:mp/mp-149")
    assert (u.db, u.id, u.query) == ("mp", "mp-149", "")
    assert str(u) == "optimade:mp/mp-149"
    assert parse("optimade:COD/1000041").db == "cod"
    assert parse("optimade:mc3d/mc3d-12/pbe").id == "mc3d-12/pbe"     # ids may hold slashes
    q = parse("optimade:oqmd?Li-P-S")
    assert (q.db, q.id, q.query) == ("oqmd", "", "Li-P-S")
    assert parse("optimade:mp-149", values={}).db == providers.DEFAULT   # no database: the default
    with pytest.raises(ValueError):
        parse("optimade:mp")
    with pytest.raises(ValueError):
        parse("optimade:mp/mp-149?Si")


def test_windows_like_text_is_not_a_path():
    # backslashes are not separators in an address on any platform
    assert parse("optimade:cod/1000041").id == "1000041"
    assert parse("optimade:mp/a\\b", values={}).id == "a\\b"


def test_extra_databases():
    got = providers.parse_extra("# comment\r\nLab = https://optimade.lab.org/v1/\r\n\r\n")
    assert got["lab"].structures_url == "https://optimade.lab.org/v1/structures"
    with pytest.raises(ValueError):
        providers.parse_extra("lab https://x")
    with pytest.raises(ValueError):
        providers.parse_extra("lab = file:///etc/passwd")
    assert set(providers.BUILTIN) <= set(providers.providers({"extra": "lab = https://x.org"}))
    with pytest.raises(KeyError, match="unknown OPTIMADE database"):
        providers.get("nope", {})


def test_every_builtin_is_https():
    for p in providers.BUILTIN.values():
        assert p.base_url.startswith("https://") and not p.base_url.endswith(("/", "/v1"))


def test_next_link_must_be_http():
    with pytest.raises(client.OptimadeError, match="not an http"):
        client.get_json("file:///etc/passwd")


# ------------------------------------------------------------------ against the local server
def test_load_one_structure(testdb):
    pipe = load("optimade:test/mp-149")
    data = pipe.compute(0)
    assert data.count == 2
    assert list(data.types) == ["Si", "Si"]
    assert np.allclose(data.cell.matrix[0], [0, 2.715, 2.715])
    assert data.cell.pbc.all()
    info = data.attributes["OPTIMADE"]
    assert info["provider"] == "test" and info["id"] == "mp-149" and info["uri"] == "optimade:test/mp-149"
    assert info["chemical_formula_reduced"] == "Si"


def test_default_database_from_settings(testdb):
    assert load("optimade:mp-149").compute(0).attributes["OPTIMADE"]["provider"] == "test"


def test_partial_occupancy(testdb):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        data = load("optimade:test/1000041").compute(0)
    assert any("partially occupied" in str(x.message) for x in w)
    assert np.allclose(sorted(set(np.round(data.particles["Occupancy"], 3))), [0.9, 1.0])
    assert list(data.particles["Species"][-4:]) == ["Cl1"] * 4


def test_molecule_and_slab(testdb):
    mol = load("optimade:test/water").compute(0)
    assert not mol.cell.pbc.any() and mol.cell.is_valid
    frac = mol.cell.to_fractional(mol.positions)
    assert (frac > 0).all() and (frac < 1).all()                  # the box holds the molecule
    slab = load("optimade:test/mp-2/slab").compute(0)             # an id with a slash
    assert list(slab.cell.pbc) == [True, True, False] and slab.cell.is_valid


def test_not_found(testdb):
    with pytest.raises(Exception, match="no structure"):
        load("optimade:test/mp-0").compute(0)


def test_search_follows_pages(testdb):
    res = search("", limit=10)
    assert len(res) == 4 and res.total == 4 and res.db == "test"
    assert len([r for r in testdb.requests if r.startswith("/v1/structures?")]) == 2   # pages of 2
    assert "response_fields=" in testdb.requests[0]
    res = search("Si")
    assert [h.uri for h in res] == ["optimade:test/mp-149"] and res[0].nsites == 2
    assert "1 of 1 in test" in repr(res)
    assert [h.id for h in search("Na Cl")] == ["1000041"]
    assert [h.id for h in search("H-O")] == ["water"]
    assert len(search("", limit=3)) == 3


def test_search_error_is_readable(testdb):
    with pytest.raises(client.OptimadeError, match="400 Bad Request cannot parse filter"):
        search(filter="bad filter")


@pytest.mark.filterwarnings("ignore:.*partially occupied")
def test_load_a_query_gives_one_frame_each(testdb):
    pipe = load("optimade:test?C")
    assert pipe.num_frames == 1
    pipe = load("optimade:test?", limit=3)
    ids = [pipe.compute(i).attributes["OPTIMADE"]["id"] for i in range(3)]
    assert ids == ["mp-149", "1000041", "water"]
    with pytest.raises(Exception, match="nothing in"):
        load("optimade:test?Li-P-S").compute(0)


def test_ai_tool(testdb):
    from ewald.llm.tools import Toolset

    ts = Toolset()
    res = ts.call("optimade_search", {"query": "Si"})
    assert not res.is_error, res.content
    got = json.loads(res.content) if isinstance(res.content, str) else res.content
    assert got["hits"][0]["uri"] == "optimade:test/mp-149"
    res = ts.call("optimade_search", {"list_databases": True})
    got = json.loads(res.content) if isinstance(res.content, str) else res.content
    assert "mp" in got["databases"] and got["default"] == "test"


def test_cli(testdb, capsys):
    from ewald.__main__ import main

    assert main(["optimade", "search", "Si", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["hits"][0]["id"] == "mp-149"
    assert main(["optimade", "databases"]) == 0
    assert "Materials Project" in capsys.readouterr().out


def test_built_in_group_and_switching_it_off():
    from ewald import plugins

    p = plugins.installed()["optimade"]
    assert (p.kind, p.loaded, p.errors) == ("built-in", True, [])
    assert p.contributions == {"readers": ["optimade"], "tools": ["optimade_search"],
                               "settings_pages": ["optimade"], "panels": ["optimade"]}
    assert plugins.scheme_reader("optimade:mp/mp-149").name == "optimade"
    assert plugins.panels()["optimade"].origin == "built-in"
    plugins.set_enabled("optimade", False)
    try:
        assert plugins.scheme_reader("optimade:mp/mp-149") is None
        assert "optimade" not in plugins.panels() and "optimade_search" not in plugins.tools()
        assert "optimade" not in plugins.settings_pages()
        assert providers.settings() == {}          # search() still works on the built-in table
    finally:
        plugins.set_enabled("optimade", True)
    assert "optimade_search" in plugins.tools()


def test_old_plugin_package_is_not_loaded(monkeypatch):
    """An ewald-optimade 0.1 left installed would register the same names: it is listed with
    a hint to uninstall it and none of its code runs."""
    from ewald import plugins

    class EP:
        name, value, group = "optimade", "ewald_optimade", "ewald.plugins"

        def load(self):
            raise AssertionError("the old plugin must not be imported")

    real = plugins._entry_point_scan
    monkeypatch.setattr(plugins, "_entry_point_scan",
                        lambda: (lambda group, eps=real(): [EP()] if group == "ewald.plugins" else eps(group)))
    monkeypatch.setattr(plugins, "_package_manifest", lambda value: {"id": "optimade", "name": "OPTIMADE structures"})
    plugins.load_plugins(force=True)
    try:
        (old,) = [p for k, p in plugins.installed().items() if k.startswith("optimade@")]
        assert "built into ewald now" in old.errors[0] and "pip uninstall" in old.errors[0]
        assert plugins.installed()["optimade"].kind == "built-in"
    finally:
        monkeypatch.undo()
        plugins.load_plugins(force=True)


# ------------------------------------------------------------------ side bar
class _Context:
    def __init__(self):
        self.opened, self.cells, self.callbacks = [], [], {}

    def open(self, path):
        self.opened.append(path)

    def insert_cell(self, code):
        self.cells.append(code)

    def on(self, event, fn):
        self.callbacks.setdefault(event, []).append(fn)


def test_side_bar(testdb, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from ewald.optimade.panel import optimade

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ctx = _Context()
    panel = optimade(ctx)
    assert panel.db.currentData() == "test"            # the default from the settings
    assert panel.db.findData("mp") >= 0
    panel.search.setText("Na Cl")
    panel.run_search()
    panel.wait()
    app.processEvents()
    assert panel.tree.topLevelItemCount() == 1
    panel.tree.topLevelItem(0).setSelected(True)
    panel.open()
    panel.insert_cell()
    assert ctx.opened == ["optimade:test/1000041"]
    assert ctx.cells == ['pipe = load("optimade:test/1000041")  # ClNa']
    panel.search.setText("")
    panel.run_search()
    panel.wait()
    app.processEvents()
    assert panel.tree.topLevelItemCount() == 4 and "4 found" in panel.note.text()
    panel.tree.selectAll()
    panel.insert_cell()
    assert ctx.cells[-1].splitlines()[-1] == "pipe = mp_149"
    assert 'mp_2_slab = load("optimade:test/mp-2/slab")' in ctx.cells[-1]
    panel.open_all()
    assert ctx.opened[-1] == "optimade:test?"
    panel.search.setText("Qq")
    panel.run_search()
    panel.wait()
    app.processEvents()
    assert "not element symbols" in panel.note.text()
