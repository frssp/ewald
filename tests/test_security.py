"""Inputs that may come from an untrusted file or from Claude cannot run code or touch the user's files."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from ase.build import bulk

from ewald import DataCollection, Pipeline, StaticSource
from ewald import modifiers as mods
from ewald.modifiers.expression import evaluate
from ewald.pipeline import Modifier
from ewald.session import load_session, modifier_from_dict


@pytest.fixture
def cu():
    return DataCollection.from_ase(bulk("Cu", "fcc", a=3.615, cubic=True).repeat(2))


@pytest.mark.parametrize("expr", [
    'np.testing._private.utils.os.system("echo pwned")', "().__class__.__mro__", 'np.load("x.npy")',
    "np.ctypeslib", "np.lib.npyio", "np.linalg.inv.__globals__", '__import__("os")', "(lambda: 1)()",
    "[p for p in Position]", "Position.ctypes", 'f"{N}"'])
def test_expression_cannot_reach_python(cu, expr):
    with pytest.raises(ValueError, match="not allowed"):
        evaluate(expr, cu)


@pytest.mark.parametrize("expr", [
    '(Position.Z > 2) & (ParticleType == "Cu")', "np.isin(N, [32])",
    "sqrt(Position.X**2 + Position.Y**2)", "np.linalg.norm(Position, axis=1) > 1", "np.where(Position.Z > 1, 1, 0)",
    "abs(Position.X - CellSize.X / 2) < 1", "Position[:, 2] > 1", "Position.T[2] > 1", "~(Position.Z > 1)",
    "Position.Z > Position.Z.mean()", "Position.Z.astype(\"int64\") == 0"])
def test_expressions_still_work(cu, expr):
    np.asarray(evaluate(expr, cu))


def test_missing_component_is_a_name_error(cu):
    with pytest.raises(NameError, match="Available"):
        evaluate("Velocity.X > 0", cu)


def test_expression_selection_in_pipeline_is_checked(cu):
    pipe = Pipeline(StaticSource([cu]), [mods.ExpressionSelection('np.testing._private.utils.os.system("x")')])
    with pytest.raises(ValueError, match="not allowed"):
        pipe.compute(0)


def test_session_entry_must_be_a_modifier(tmp_path):
    marker = tmp_path / "ran"
    entry = {"type": "system", "module": "os", "parameters": {"command": f"touch {marker}"}}
    m, warning = modifier_from_dict(entry)
    assert m is None and "not a modifier" in warning and not marker.exists()
    nested = {"type": "ExpressionSelection", "module": "ewald.modifiers.selection",
              "parameters": {"expression": {"__modifier__": entry}}}
    modifier_from_dict(nested)
    assert not marker.exists()
    session = tmp_path / "evil.ewald"
    session.write_text(json.dumps({"version": 1, "modifiers": [entry]}))
    try:
        load_session(session)
    except Exception:
        pass
    assert not marker.exists()


class Untrusted(Modifier):
    """A modifier that loads files or code (like the ewald-plumed plugin's Plumed)."""

    def __init__(self, input: str = ""):
        super().__init__()
        self.input = input

    def untrusted_problems(self):
        return ["LOAD"] if "LOAD" in self.input else []

    def modify(self, data, ctx):
        return None


def test_session_modifier_with_untrusted_problems_is_disabled():
    entry = {"type": "Untrusted", "module": __name__, "parameters": {"input": "LOAD FILE=/tmp/evil.so"}}
    m, warning = modifier_from_dict(entry)
    assert not m.enabled and "LOAD" in warning
    m, warning = modifier_from_dict({**entry, "parameters": {"input": "d: DISTANCE ATOMS=1,2"}})
    assert m.enabled and warning is None
    assert modifier_from_dict(entry, allow_code=True)[0].enabled


def test_session_with_moved_modifier_says_how_to_install(monkeypatch):
    from ewald import modifiers, plugins

    monkeypatch.setattr(plugins, "modifiers", lambda: {})  # as without the ewald-plumed plugin
    m, warning = modifier_from_dict({"type": "Plumed", "module": "ewald.modifiers.plumed",
                                     "parameters": {"input": "d: DISTANCE ATOMS=1,2"}})
    assert m is None and "ewald-plumed" in warning and "pip install" in warning
    with pytest.raises(ImportError, match="ewald-phonopy"):
        modifiers.PhononMode


def test_claude_export_stays_out_of_user_files(tmp_path, cu):
    from ewald.llm.tools import Toolset

    work, home = tmp_path / "work", tmp_path / "home"
    work.mkdir()
    (home / ".ssh").mkdir(parents=True)
    precious = home / "notes.txt"
    precious.write_text("keep")
    ts = Toolset(Pipeline(StaticSource([cu])), workdir=str(work))
    for path, overwrite in [(str(precious), True), (str(home / ".ssh" / "authorized_keys"), False),
                            ("../home/notes.txt", True), (str(home / ".bashrc"), False)]:
        r = ts.call("export_file", {"path": path, "format": "extxyz", "overwrite": overwrite})
        assert r.is_error, path
    assert precious.read_text() == "keep" and not (home / ".ssh" / "authorized_keys").exists()
    assert not ts.call("export_file", {"path": "out/a.extxyz"}).is_error
    assert ts.call("export_file", {"path": "out/a.extxyz"}).is_error  # exists
    assert not ts.call("export_file", {"path": "out/a.extxyz", "overwrite": True}).is_error
    assert not ts.call("export_file", {"path": str(home / "new.extxyz")}).is_error  # a new file is fine


def test_forge_subprocess_gets_no_secrets(monkeypatch):
    from ewald.llm import forge

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws")
    env = forge._subprocess_env("/tmp/forge-home")
    assert not {"ANTHROPIC_API_KEY", "SSH_AUTH_SOCK", "AWS_SECRET_ACCESS_KEY"} & set(env)
    assert env["HOME"] == "/tmp/forge-home" and env.get("PATH") == os.environ.get("PATH")


def _ewd(path, cells, files=None, manifest=None):
    """A hand-made .ewd file (as someone could send it)."""
    import zipfile

    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {},
          "cells": [{"id": f"c{k}", "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": src} for k, src in enumerate(cells)]}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("notebook.ipynb", json.dumps(nb))
        z.writestr("manifest.json", json.dumps(manifest or {"ewald_notebook": 1, "data": [], "plugins": []}))
        for name, data in (files or {}).items():
            z.writestr(name, data)
    return str(path)


def test_ewd_opens_without_running_code_or_importing_plugins(tmp_path):
    from ewald import plugins
    from ewald.notebook import Notebook

    marker = tmp_path / "pwned"
    plugin = f"open({str(tmp_path / 'plugin-ran')!r}, 'w').close()\n"
    path = _ewd(tmp_path / "evil.ewd", [f"open({str(marker)!r}, 'w').close()"],
                files={"plugins/evil_plugin.py": plugin},
                manifest={"ewald_notebook": 1, "data": [],
                          "plugins": [{"id": "evil_plugin", "file": "plugins/evil_plugin.py"}]})
    nb = Notebook.load(path)
    assert nb.restricted
    with pytest.raises(PermissionError):
        nb.run_all()
    with pytest.raises(PermissionError):
        nb.run_cell(0)
    assert not marker.exists() and not (tmp_path / "plugin-ran").exists()
    assert "evil_plugin" not in plugins.installed()


@pytest.mark.parametrize("name", ["../outside.txt", "/abs.txt", "data/../../outside.txt", "C:/x.txt"])
def test_ewd_with_unsafe_paths_is_refused(tmp_path, name):
    from ewald.notebook import Notebook

    path = _ewd(tmp_path / "bad.ewd", ["x = 1"], files={name: "x"})
    with pytest.raises(ValueError, match="unsafe path"):
        Notebook.load(path)
    assert not (tmp_path.parent / "outside.txt").exists()
