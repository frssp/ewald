"""EWALD behaves the same on Windows, macOS and Linux (docs/developer/portability.md).

CI runs on Linux only while Actions minutes are exhausted, so these tests catch the usual
Windows divergences from Linux: source rules checked on the AST (text files name their
encoding and write LF, no string splitting of file paths, no OS-dependent case rules, no
fork/shell), plus Windows paths, CRLF files and Windows text-mode writes simulated here.
"""
from __future__ import annotations

import ast
import builtins
import io
import ntpath
import os
import types
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "ewald"
DATA = Path(__file__).resolve().parent / "data"

# Reviewed sites that split or join strings on "/" or "\": none of them is a local file path.
# Key: (file under src/ewald, enclosing function). New sites use os.path / pathlib / posixpath
# (remote paths) instead, or are listed here with the reason.
SEP_SITES = {
    ("gui/window_plugins.py", "refresh_plugin_menu_items"): "menu paths ('Tools/Sub/Item'), not files",
    ("gui/window_plugins.py", "add_plugin_items"): "menu paths ('Tools/Sub/Item'), not files",
    ("io/neb.py", "_join"): "remote (POSIX) branch; local paths use os.path.join",
    ("io/neb.py", "_dirname"): "remote (POSIX) branch; local paths use os.path.dirname",
    ("io/neb.py", "_basename"): "remote (POSIX) branch; local paths use os.path.basename",
    ("io/remote.py", "glob"): "path on the SSH host, always POSIX",
    ("io/xdatcar.py", "species_name"): "POTCAR labels such as 'Fe_pv/...'",
    ("kernel_bridge.py", "load_frame"): "array names inside the .npz written by the bridge",
    ("llm/evals.py", "resolve_file"): "'paper:<id>/<file>' task references",
    ("manual.py", "url"): "manual URLs",
    ("notebook_io.py", "_bundle_name"): "zip member names, '/' by the zip format",
    ("plugin_catalog.py", "_join"): "URL branch; local paths use pathlib",
    ("plugins.py", "register_menu_item"): "menu paths ('Tools/Sub/Item'), not files",
    ("runinfo.py", "_lammps_info"): "LAMMPS fix styles such as 'npt/asphere'",
}


# ----------------------------------------------------------------------- source rules
def _sources():
    for path in sorted(SRC.rglob("*.py")):
        yield path.relative_to(SRC).as_posix(), ast.parse(path.read_text(encoding="utf-8"))


def _functions(tree):
    """node -> name of the innermost enclosing function ('' at module level)."""
    owner = {}

    def visit(node, name):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else name
            owner[child] = inner
            visit(child, inner)

    visit(tree, "")
    return owner


def _call_name(node):
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def _mode(node, position):
    for kw in node.keywords:
        if kw.arg == "mode":
            return kw.value
    return node.args[position] if len(node.args) > position else None


def _text_io_problems(node):
    """What a call that opens a text file lacks: 'encoding' (the default is the locale's code
    page on Windows) and, for writes, 'newline' (Windows text mode writes CRLF)."""
    kws = {kw.arg for kw in node.keywords}
    name = _call_name(node)
    f = node.func
    if name in ("open", "fdopen") and (isinstance(f, ast.Name) or (isinstance(f.value, ast.Name)
                                                                    and f.value.id in ("io", "builtins", "os"))):
        mode = _mode(node, 1)
        if mode is None:
            mode = "r"
        elif not (isinstance(mode, ast.Constant) and isinstance(mode.value, str)) or "b" in mode.value:
            return []
        else:
            mode = mode.value
        need = ["encoding"] + (["newline"] if set(mode) & set("wax") else [])
    elif name == "read_text" and not node.args:
        need = ["encoding"]
    elif name == "write_text" and len(node.args) < 2:
        need = ["encoding", "newline"]
    elif name in ("run", "Popen", "check_output") and kws & {"text", "universal_newlines"}:
        need = ["encoding"]
    else:
        return []
    return [k for k in need if k not in kws]


def test_text_files_are_utf8_with_lf():
    """Every text file EWALD opens names encoding="utf-8" and every text write newline="\\n":
    otherwise Windows reads cp949/cp1252 and writes CRLF (a POSCAR VASP then rejects)."""
    bad = []
    for rel, tree in _sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (missing := _text_io_problems(node)):
                bad.append(f"{rel}:{node.lineno} needs {', '.join(missing)}=")
    assert not bad, "text I/O without an explicit encoding / newline:\n" + "\n".join(bad)


def _sep_sites():
    found = {}
    for rel, tree in _sources():
        owner = _functions(tree)
        for node in ast.walk(tree):
            hit = False
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr, value = node.func.attr, node.func.value
                hit = (attr in ("split", "rsplit", "partition", "rpartition") and node.args
                       and isinstance(node.args[0], ast.Constant) and node.args[0].value in ("/", "\\")) or \
                      (attr == "join" and isinstance(value, ast.Constant) and value.value in ("/", "\\"))
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                hit = any(isinstance(s, ast.Constant) and s.value in ("/", "\\") for s in (node.left, node.right))
            if hit:
                found.setdefault((rel, owner.get(node, "")), node.lineno)
    return found


def test_paths_are_not_split_as_strings():
    """Local paths go through os.path / pathlib: splitting on "/" read the wrong file on Windows
    (PR #189). Reviewed non-path sites are listed in SEP_SITES."""
    found = _sep_sites()
    new = sorted(f"{rel}:{line} ({fn or 'module'})" for (rel, fn), line in found.items() if (rel, fn) not in SEP_SITES)
    gone = sorted(k for k in SEP_SITES if k not in found)
    assert not new, ("string splitting/joining on '/' or '\\'; use os.path, pathlib or posixpath (remote), "
                     f"or list a reviewed non-path site in SEP_SITES: {new}")
    assert not gone, f"SEP_SITES lists sites that no longer exist: {gone}"


FORBIDDEN_CALLS = {
    ("fnmatch", "fnmatch"): "ignores case on Windows only; use fnmatchcase (on lower() for case-insensitive)",
    ("os", "fork"): "no fork on Windows",
    ("os", "getuid"): "POSIX only",
    ("os", "killpg"): "POSIX only",
    ("os", "setsid"): "POSIX only",
}
FORBIDDEN_MODULES = {"fcntl", "termios", "pwd", "grp", "resource"}


def test_no_platform_only_calls():
    bad = []
    for rel, tree in _sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in FORBIDDEN_CALLS:
                    bad.append(f"{rel}:{node.lineno} {f.value.id}.{f.attr}: {FORBIDDEN_CALLS[f.value.id, f.attr]}")
                for kw in node.keywords:
                    if kw.arg == "shell" and not (isinstance(kw.value, ast.Constant) and kw.value.value is False):
                        bad.append(f"{rel}:{node.lineno} shell=True: cmd.exe and sh quote differently; pass a list")
                    if kw.arg in ("preexec_fn", "start_new_session"):
                        bad.append(f"{rel}:{node.lineno} {kw.arg}: POSIX only")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                bad += [f"{rel}:{node.lineno} import {n}: POSIX only" for n in names if n.split(".")[0] in FORBIDDEN_MODULES]
                if isinstance(node, ast.ImportFrom) and node.module == "multiprocessing":
                    bad.append(f"{rel}:{node.lineno} multiprocessing: spawn on Windows/macOS, fork on Linux")
    assert not bad, "\n".join(bad)


# ----------------------------------------------------------------------- Windows paths
WINDOWS_PATHS = [r"C:\Users\kim\run\XDATCAR", "C:/Users/kim/run/OUTCAR", r"D:\data\04\OUTCAR", r"\\server\share\POSCAR"]


@pytest.mark.parametrize("path", WINDOWS_PATHS)
def test_windows_paths_are_local_files(path):
    """A drive letter is neither an SSH host ('C:...' as host:path) nor a URI scheme ('c:')."""
    from ewald import plugins
    from ewald.io import remote

    assert not remote.is_remote(path)
    assert plugins.path_scheme(path) is None
    assert plugins.scheme_reader(path) is None
    assert plugins.path_scheme("aiida:1234") == "aiida" and remote.is_remote("cluster:/scratch/XDATCAR")


def test_plugin_source_names_with_backslashes(tmp_path):
    from ewald.plugin_catalog import Entry, source_name

    assert source_name(r"C:\Users\kim\plugins\my_mod.py") == "my_mod.py"
    assert source_name("/home/kim/plugins/my_mod.py") == "my_mod.py"
    assert source_name("https://example.org/a/my_mod.py?raw=1") == "my_mod.py"
    assert Entry(name="x", source=r"D:\p\x.py").file_name == "x.py"


def test_ewd_bundles_files_on_other_drives():
    from ewald.notebook_io import _bundle_name

    assert _bundle_name("run/XDATCAR") == "data/run/XDATCAR"
    assert _bundle_name("../../a/XDATCAR") == "data/a/XDATCAR"
    assert _bundle_name(r"D:\runs\T300\XDATCAR") == "data/runs/T300/XDATCAR"   # relpath fails across drives
    assert _bundle_name(r"..\T300\XDATCAR") == "data/T300/XDATCAR"


def _windows_os():
    """An ``os`` stand-in with Windows path rules for modules that only use os.path / os.sep."""
    return types.SimpleNamespace(**{**vars(os), "path": ntpath, "sep": "\\", "altsep": "/"})


def test_run_labels_with_windows_paths(monkeypatch):
    from ewald import runs

    monkeypatch.setattr(runs, "os", _windows_os())
    labels = runs._default_labels([r"C:\md\T300\XDATCAR", r"C:\md\T600\XDATCAR"])
    assert labels == ["T300", "T600"]
    assert runs._default_labels(["C:/md/T300/XDATCAR"]) == ["T300"]


def test_kernel_names_windows_environments(monkeypatch):
    from ewald import kernel

    monkeypatch.setattr(kernel, "os", _windows_os())
    assert kernel._short(r"C:\Users\kim\venvs\ml\Scripts\python.exe") == "Python (ml)"
    assert kernel._short(r"C:\Users\kim\miniconda3\envs\dft\python.exe") == "Python (dft)"


def test_remote_cache_names_are_valid_windows_names():
    from ewald.io.remote import local_name

    for name in ("XDATCAR", "OUTCAR.gz", "dump.lammpstrj", "vasprun.xml"):
        assert local_name(name) == name   # readers still recognise the file by name
    assert local_name("dump.*.lammpstrj") == "dump._.lammpstrj"
    assert local_name('a:b|c"d<e>f?') == "a_b_c_d_e_f_"
    assert local_name("CON") == "_CON" and local_name("nul.txt") == "_nul.txt"
    assert local_name("trailing. ") == "trailing" and local_name("") == "file"


# ----------------------------------------------------------------------- case
def test_reader_patterns_ignore_case_everywhere():
    """fnmatch.fnmatch follows the OS (case-insensitive on Windows only); readers and writers
    match names the same way on every platform."""
    from ewald import plugins
    from ewald.plugin_kinds import _matches

    assert _matches("structure.CIF", ("*.cif",)) and _matches("poscar_relaxed", ("POSCAR*",))
    assert not _matches("structure.cif", ("*.xyz",))
    assert plugins.find_writer("OUT.XYZ").name == plugins.find_writer("out.xyz").name
    assert plugins.find_writer("poscar").name == plugins.find_writer("POSCAR").name == "poscar"


def test_save_dialog_adds_the_filter_extension():
    pytest.importorskip("PySide6")
    from ewald.gui.file_dialogs import suffixed

    assert suffixed("/a/plot", "PDF (*.pdf);;PNG (*.png)".split(";;")[0]) == "/a/plot.pdf"
    assert suffixed("/a/movie", "Movies (*.mp4 *.gif)") == "/a/movie.mp4"
    assert suffixed("/a/x.png", "PDF (*.pdf)") == "/a/x.png"
    assert suffixed("/a/POSCAR", "VASP POSCAR (POSCAR* CONTCAR* *.vasp)") == "/a/POSCAR"
    assert suffixed("/a/data", "All files (*)") == "/a/data"
    assert suffixed("", "PDF (*.pdf)") == ""


# ----------------------------------------------------------------------- line endings
CRLF_FILES = ["lammps/dump.atom", "lammps/dump.custom.lammpstrj", "lammps/dump.triclinic.lammpstrj",
              "lammps/data.full.lmp", "lammps/data.triclinic.lmp", "vasp/XDATCAR", "vasp/XDATCAR.npt",
              "vasp/POSCAR", "vasp/POSCAR.selective", "vasp/CONTCAR", "vasp/OUTCAR", "vasp/vasprun.xml"]


def _crlf_copy(src: Path, folder: Path) -> Path:
    dest = folder / src.name
    dest.write_bytes(src.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    return dest


def _types(data):
    t = data.particles["Particle Type"] if "Particle Type" in data.particles else None
    return None if t is None else [getattr(x, "name", x) for x in getattr(t, "types", [])]


@pytest.mark.parametrize("rel", CRLF_FILES)
def test_crlf_files_read_like_lf(rel, tmp_path):
    """Files written or edited on Windows have CRLF ends; the native byte readers and the ASE
    fallback give the same frames, positions, cells, types and attributes as for LF."""
    from ewald import load

    a, b = load(str(DATA / rel)), load(str(_crlf_copy(DATA / rel, tmp_path)))
    assert a.num_frames == b.num_frames
    for frame in sorted({0, a.num_frames - 1}):
        da, db = a.compute(frame), b.compute(frame)
        assert da.count == db.count
        np.testing.assert_allclose(db.positions, da.positions)
        np.testing.assert_allclose(np.asarray(db.cell.matrix), np.asarray(da.cell.matrix))
        assert _types(da) == _types(db)
        assert {k: str(v) for k, v in da.attributes.items()} == {k: str(v) for k, v in db.attributes.items()}


def test_crlf_thermo_and_kpoints(tmp_path):
    from ewald.electronic import kpoints_file_labels
    from ewald.io.thermo import read_lammps_log

    log = DATA / "lammps" / "thermo" / "log.lammps"
    a, b = read_lammps_log(log), read_lammps_log(_crlf_copy(log, tmp_path))
    assert list(a) == list(b)
    kp = DATA / "vasp" / "electronic" / "KPOINTS"
    assert kpoints_file_labels(kp) == kpoints_file_labels(_crlf_copy(kp, tmp_path))


@pytest.fixture
def windows_text_mode(monkeypatch):
    """Text files opened without newline= get CRLF ends, as on Windows."""
    real = builtins.open

    def crlf_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if "b" not in mode and newline is None and any(c in mode for c in "wax"):
            newline = "\r\n"
        return real(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(builtins, "open", crlf_open)
    monkeypatch.setattr(io, "open", crlf_open)
    return crlf_open


@pytest.mark.parametrize("name", ["POSCAR", "structure.xyz", "structure.extxyz", "structure.cif", "structure.pdb",
                                  "structure.xsf", "data.lmp", "dump.lammpstrj", "XDATCAR"])
def test_written_files_have_lf_ends_on_windows(name, tmp_path, windows_text_mode):
    """Exports are byte-identical across platforms: no CRLF from Windows text mode."""
    from ewald import export_file, load

    pipe = load(str(DATA / ("vasp/XDATCAR" if name in ("XDATCAR", "dump.lammpstrj", "structure.xyz") else "vasp/POSCAR")))
    out = tmp_path / name
    kwargs = {"frames": [0]} if name in ("POSCAR", "data.lmp", "structure.cif", "structure.pdb", "structure.xsf") else {}
    export_file(pipe, str(out), **kwargs)
    assert b"\r" not in out.read_bytes()


def test_settings_written_by_older_windows_versions_still_load(tmp_path, monkeypatch):
    """Settings files were written in the ANSI code page on Windows before every file was UTF-8."""
    import locale

    from ewald import settings

    monkeypatch.setattr(locale, "getpreferredencoding", lambda do_setlocale=True: "cp949")   # Korean Windows
    p = tmp_path / "settings.json"
    p.write_bytes('{"general.docs_url": "C:\\\\문서"}'.encode("cp949"))
    assert settings.load(p)["general.docs_url"] == "C:\\문서"
    p.write_text('{"x": "Å"}', encoding="utf-8")
    assert settings.read_text_any(p) == '{"x": "Å"}'
