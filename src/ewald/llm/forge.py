"""Modifier Forge: an AI model (Claude by default) writes a modifier plugin and ewald verifies it before saving it.

>>> from ewald.llm.forge import forge
>>> result = forge("Fraction of O neighbours within 2.5 A around each Li atom", run_code=True)
>>> result.ok, result.path            # saved into the plugin folder, now in the GUI menu and tools

The model returns two files through one tool call: the plugin module (a ``Modifier`` subclass
registered with ``@modifier``) and a test module whose ``test_*`` functions build structures
with a known answer and assert it. ewald then runs, in a separate Python process with a time
limit and a temporary config folder,

- the import and the plugin contract (a ``Modifier`` subclass; every constructor argument stored
  under its own name),
- a no-mutation check (the input frame's arrays are unchanged after ``compute``),
- the GPU code path with a NumPy-backed fake CUDA backend, compared with the NumPy result,
- the model's own tests,

and sends any failure back to the model to fix, up to ``max_rounds`` times. Only a version that
passes everything is written to the plugin folder, with its tests next to it in ``_forge_tests/``.

The generated code runs on this machine (in a subprocess, not a security sandbox), so nothing
is executed unless ``run_code=True``; ``ewald forge`` asks first.
"""
from __future__ import annotations

import datetime
import json
import keyword
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5-5"

CONTRACT = """\
Plugin contract (ewald modifiers):
- Import from `ewald.plugin_api` (the supported plugin surface): subclass `Modifier`, decorate the class with `@modifier(category="...")`.
- Store every constructor argument as an attribute of the same name; give every argument a default.
  The GUI form, repr(), caching and generated scripts are built from these attributes.
- `modify(self, data, ctx)`: `data` is a shallow copy of the frame. NEVER write into an existing
  array (no `data.positions[...] = `, no `+=` on a property): arrays are shared with cached frames.
  Build a new array and assign it (`data["Name"] = new`). Return None, or a new DataCollection
  when the particle count changes (e.g. `data.delete(mask)`).
- Write array code against `ctx.xp` (NumPy or CuPy) and pass `backend=ctx.backend` to ewald kernels
  such as `ewald.neighbors.find_pairs(positions, cell, cutoff, backend=be)` -> (i, j, delta, dist)
  (full neighbour list: each pair appears as i->j and j->i, periodic images handled). Convert with
  `ctx.backend.to_numpy(a)` before storing a result in `data`.
- OVITO-style names: per-particle results are properties in Title Case With Spaces; global results
  are attributes "ClassName.quantity"; curves are tables `data.tables["kebab-name"] = {col: array}`.
- Raise an exception with an actionable message for invalid input; never print.
- Import optional heavy packages inside functions.
- The first paragraph of the class docstring is a one-sentence description (shown in the GUI and to Claude).

Tests:
- A separate module importing your plugin module by its module name.
- Functions named test_*() using plain `assert` (no pytest fixtures). Build structures whose answer is
  known analytically: ase.build.bulk/molecule/surface, then `ewald.DataCollection.from_ase(atoms)`,
  `ewald.Pipeline(ewald.StaticSource(data))`, append the modifier, `pipe.compute(0)`. Use numpy for
  tolerances (np.allclose). Cover a perturbed structure as well as a perfect one, and periodic
  boundaries. Keep each test under a few seconds.
"""

SYSTEM_PROMPT = """\
You write analysis modifiers for ewald, an OVITO-style atomistic structure and trajectory library \
used by computational materials scientists (VASP, LAMMPS). The user describes an analysis; you \
deliver a plugin module and a test module through the submit_plugin tool. ewald runs the tests and \
contract checks in a subprocess and sends you any failure to fix.

Get the physics right and say which convention you chose (cutoffs, normalisation, units: Angstrom, \
fs, eV) in the class docstring. Prefer vectorised array code over Python loops over atoms. Tests must \
check numbers against values you can derive by hand for the test structure, not just that code runs.

{contract}
ewald API reference:

{reference}"""

SUBMIT_TOOL = {
    "name": "submit_plugin",
    "description": "Submit the plugin module and its tests. ewald runs them and returns a report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "module_name": {"type": "string", "description": "snake_case module (file) name, e.g. 'li_o_fraction'"},
            "class_name": {"type": "string", "description": "the Modifier subclass name, CamelCase"},
            "category": {"type": "string", "description": "GUI submenu, e.g. 'Analysis'"},
            "module_source": {"type": "string", "description": "complete Python source of the plugin module"},
            "test_source": {"type": "string", "description": "complete Python source of the test module"},
            "notes": {"type": "string", "description": "conventions chosen and known limitations, 1-3 sentences"},
        },
        "required": ["module_name", "class_name", "category", "module_source", "test_source"],
    },
}

_RUNNER = r'''
import importlib, inspect, json, sys, traceback
import numpy as np

workdir, module_name, class_name, test_name = sys.argv[1:5]
sys.path.insert(0, workdir)
report = {"checks": [], "tests": []}

def check(name, fn):
    try:
        detail = fn()
        report["checks"].append({"name": name, "ok": True, "detail": detail or ""})
        return True
    except Exception as exc:
        report["checks"].append({"name": name, "ok": False,
                                 "detail": "".join(traceback.format_exception(exc)[-6:])})
        return False

state = {}

def do_import():
    state["module"] = importlib.import_module(module_name)

def do_class():
    from ewald import Modifier
    from ewald import plugins
    cls = getattr(state["module"], class_name, None)
    if not (isinstance(cls, type) and issubclass(cls, Modifier)):
        raise TypeError(f"{module_name}.{class_name} is not a Modifier subclass")
    if class_name not in plugins._registry.modifiers:
        raise RuntimeError("the class is not registered; decorate it with @modifier(category=...)")
    state["cls"] = cls

def do_parameters():
    cls = state["cls"]
    sig = inspect.signature(cls.__init__)
    params = [p for p in list(sig.parameters.values())[1:] if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    missing_default = [p.name for p in params if p.default is p.empty]
    if missing_default:
        raise ValueError(f"give every constructor argument a default (missing: {missing_default})")
    obj = cls()
    unstored = [p.name for p in params if not hasattr(obj, p.name)]
    if unstored:
        raise ValueError(f"store every constructor argument as an attribute of the same name (missing: {unstored})")
    state["obj"] = obj
    return repr(obj)

def _frame():
    from ase.build import bulk
    from ewald import DataCollection
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((3, 3, 3))
    atoms.rattle(0.05, seed=1)
    symbols = atoms.get_chemical_symbols()
    for k in range(0, len(symbols), 4):
        symbols[k] = "Zr"           # two species, so per-type code paths run
    atoms.set_chemical_symbols(symbols)
    return DataCollection.from_ase(atoms)

def do_no_mutation():
    from ewald import Pipeline, StaticSource
    src = StaticSource(_frame())
    frame = src.frames[0]
    before = {k: v.copy() for k, v in frame.particles.items()}
    cell = frame.cell.matrix.copy()
    pipe = Pipeline(src)
    pipe.modifiers.append(state["obj"])
    out = pipe.compute(0)
    changed = [k for k, v in before.items() if not np.array_equal(v, frame.particles[k])]
    if changed or not np.array_equal(cell, frame.cell.matrix):
        raise AssertionError(f"modify() wrote into input arrays in place: {changed or ['cell']}; "
                             "build new arrays and assign them instead")
    state["cpu_out"] = out
    new = sorted(set(out.particles) - set(before)) + sorted(f"attr:{k}" for k in out.attributes) \
        + sorted(f"table:{k}" for k in out.tables)
    return "outputs: " + ", ".join(new)

def do_gpu_path():
    from ewald import Pipeline, StaticSource
    from ewald.backend import Backend

    class NumpyAsGPU:
        def __getattr__(self, name):
            return getattr(np, name)

        @staticmethod
        def asnumpy(a):
            return np.asarray(a)

    pipe = Pipeline(StaticSource(_frame()))
    pipe.modifiers.append(state["cls"]())
    gpu = pipe.compute(0, device=Backend("fake-cuda", NumpyAsGPU()))
    cpu = state["cpu_out"]
    for k, v in cpu.particles.items():
        g = gpu.particles.get(k)
        if g is None or not isinstance(g, np.ndarray):
            raise AssertionError(f"property {k!r}: GPU path returned {type(g).__name__}, store NumPy arrays "
                                 "(ctx.backend.to_numpy)")
        if v.dtype.kind in "fc" and not np.allclose(v, g, equal_nan=True):
            raise AssertionError(f"property {k!r} differs between the NumPy and the GPU code path")

if check("import", do_import) and check("class registered", do_class) and check("parameters", do_parameters):
    if check("no in-place writes", do_no_mutation):
        check("GPU code path", do_gpu_path)

if all(c["ok"] for c in report["checks"][:2]):
    try:
        tests = importlib.import_module(test_name)
        funcs = [(n, f) for n, f in vars(tests).items() if n.startswith("test_") and callable(f)]
        if not funcs:
            report["checks"].append({"name": "tests", "ok": False, "detail": "no test_* functions found"})
        for n, f in funcs:
            try:
                f()
                report["tests"].append({"name": n, "ok": True, "detail": ""})
            except Exception as exc:
                report["tests"].append({"name": n, "ok": False,
                                        "detail": "".join(traceback.format_exception(exc)[-6:])})
    except Exception as exc:
        report["checks"].append({"name": "import tests", "ok": False,
                                 "detail": "".join(traceback.format_exception(exc)[-6:])})

print("@@REPORT@@" + json.dumps(report))
'''


@dataclass
class ForgeResult:
    request: str
    module_name: str = ""
    class_name: str = ""
    category: str = ""
    module_source: str = ""
    test_source: str = ""
    notes: str = ""
    report: dict = field(default_factory=dict)
    rounds: int = 0
    ok: bool = False
    verified: bool = False
    path: Path | None = None

    def summary(self) -> str:
        lines = [f"{self.class_name or '?'} ({self.module_name or '?'}): "
                 + ("passed" if self.ok else "unverified" if not self.verified else "FAILED")
                 + f" after {self.rounds} round(s)"]
        for item in self.report.get("checks", []) + self.report.get("tests", []):
            lines.append(f"  {'ok  ' if item['ok'] else 'FAIL'} {item['name']}"
                         + (f": {item['detail'].strip().splitlines()[-1]}" if item["detail"] and not item["ok"] else ""))
        if self.path:
            lines.append(f"saved to {self.path}")
        if self.notes:
            lines.append(f"notes: {self.notes}")
        return "\n".join(lines)


def _valid_identifier(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


# environment variables the test subprocess keeps; everything else (API keys, tokens, the SSH
# agent socket, cloud credentials) stays out of reach of the code the model wrote
_ENV_KEEP = ("PATH", "LANG", "LANGUAGE", "TZ", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "WINDIR", "COMSPEC",
             "PATHEXT", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "CUDA_PATH",
             "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "VIRTUAL_ENV",
             "CONDA_PREFIX")


def _subprocess_env(home) -> dict:
    env = {k: v for k, v in os.environ.items() if k in _ENV_KEEP or k.startswith("LC_")}
    env["HOME"] = env["USERPROFILE"] = str(home)  # no ~/.ssh, ~/.aws, ... through the home folder
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"  # no saved API key via keyring
    return env


def verify(module_name: str, class_name: str, module_source: str, test_source: str, timeout: float = 120) -> dict:
    """Run the contract checks and the tests in a subprocess; returns the report dict."""
    if not (_valid_identifier(module_name) and _valid_identifier(class_name)):
        return {"checks": [{"name": "names", "ok": False,
                            "detail": f"module_name {module_name!r} and class_name {class_name!r} "
                                      "must be Python identifiers"}], "tests": []}
    test_name = f"test_{module_name}"
    with tempfile.TemporaryDirectory(prefix="ewald-forge-") as tmp:
        Path(tmp, f"{module_name}.py").write_text(module_source, encoding="utf-8", newline="\n")
        Path(tmp, f"{test_name}.py").write_text(test_source, encoding="utf-8", newline="\n")
        runner = Path(tmp, "_forge_runner.py")
        runner.write_text(_RUNNER, encoding="utf-8", newline="\n")
        env = {**_subprocess_env(tmp), "EWALD_CONFIG_DIR": str(Path(tmp, "config")), "EWALD_NO_PLUGINS": "1",
               "MPLBACKEND": "Agg", "PYTHONUTF8": "1"}
        try:
            proc = subprocess.run([sys.executable, "-I", str(runner), tmp, module_name, class_name, test_name],
                                  cwd=tmp, env=env, capture_output=True, timeout=timeout,
                                  encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return {"checks": [{"name": "time limit", "ok": False,
                                "detail": f"the checks and tests did not finish within {timeout:.0f} s"}], "tests": []}
    out = proc.stdout
    if "@@REPORT@@" not in out:
        return {"checks": [{"name": "run", "ok": False,
                            "detail": (proc.stderr or out)[-3000:] or f"exit code {proc.returncode}"}], "tests": []}
    return json.loads(out.split("@@REPORT@@", 1)[1].strip().splitlines()[0])


def _passed(report: dict) -> bool:
    items = report.get("checks", []) + report.get("tests", [])
    return bool(report.get("tests")) and all(i["ok"] for i in items)


def _system() -> str:
    from .reference import api_reference

    return SYSTEM_PROMPT.format(contract=CONTRACT, reference=api_reference())


def forge(request: str, *, client=None, model: str | None = None, provider=None, base_url: str | None = None,
          run_code: bool = False, max_rounds: int = 4, save: bool = True, overwrite: bool = False,
          timeout: float = 120, max_tokens: int = 16000, on_event=None, confirm=None) -> ForgeResult:
    """Have an AI model write a modifier plugin for ``request``; verify and save it (see module docs).

    ``provider`` is an AI provider (:mod:`ewald.llm.providers`; default Anthropic) and ``model``
    defaults to its first model. ``client`` defaults to the provider's client with the saved
    key; any object with the same ``messages.create`` (Claude) or ``chat.completions.create``
    works (tests pass a fake). With ``run_code=False`` the first draft is returned unverified and
    nothing is run or saved. ``confirm(result) -> bool`` is asked before a verified plugin is
    saved, since a saved plugin is imported on every start; False keeps it unsaved.
    """
    from .. import plugins
    from .providers import get

    p = get(provider)
    model = model or (DEFAULT_MODEL if p.name == "anthropic" else p.default_model)
    if client is None:
        client = p.client(base_url=base_url, model=model)
    emit = on_event or (lambda kind, payload: None)
    chat = p.chat(client, model, system=_system(),
                  tools=[SUBMIT_TOOL], max_tokens=max_tokens, beta=False)
    chat.user(request)
    result = ForgeResult(request)
    for round_ in range(1, max_rounds + 1):
        reply = chat.send(tool_choice="submit_plugin")
        call = next((c for c in reply.calls if c.name == "submit_plugin"), None)
        if call is None:
            if reply.stop == "refusal":
                result.notes = "request declined by the model"
                return result
            chat.user("Submit the plugin with the submit_plugin tool.")
            continue
        args = dict(call.input or {})
        result.rounds = round_
        result.module_name = str(args.get("module_name", "")).strip()
        result.class_name = str(args.get("class_name", "")).strip()
        result.category = str(args.get("category", "") or "Plugins").strip()
        result.module_source = args.get("module_source", "")
        result.test_source = args.get("test_source", "")
        result.notes = args.get("notes", "")
        emit("draft", result)
        if not run_code:
            return result
        result.report = verify(result.module_name, result.class_name, result.module_source,
                               result.test_source, timeout=timeout)
        result.verified = True
        result.ok = _passed(result.report)
        emit("report", result)
        if result.ok:
            break
        chat.tool_results([(call, "Checks or tests failed; fix the code (or a test that is wrong) and submit "
                                  "again.\n" + json.dumps(result.report, indent=1), True)])
    if result.ok and save and (confirm is None or confirm(result)):
        result.path = save_plugin(result, overwrite=overwrite)
        plugins.load_plugins(force=True)
    return result


def save_plugin(result: ForgeResult, overwrite: bool = False) -> Path:
    """Write the verified module into the plugin folder and its tests into ``_forge_tests/``."""
    from .. import plugins

    folder = plugins.plugin_dir()
    target = folder / f"{result.module_name}.py"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True (ewald forge --overwrite) to replace it")
    tests = folder / "_forge_tests"
    tests.mkdir(parents=True, exist_ok=True)
    header = ("# Generated by ewald forge from the request:\n"
              + textwrap.indent(textwrap.fill(result.request, 96), "#   ") + "\n"
              + "# Verified: contract checks and the tests in _forge_tests/"
              + f"test_{result.module_name}.py passed.\n")
    # the manifest goes last: a module may open with a docstring or ``from __future__`` imports
    from ..plugin_api import PLUGIN_API

    manifest = {"id": result.module_name, "description": result.request[:200], "source": "forge",
                "created": datetime.date.today().isoformat(), "plugin_api": PLUGIN_API}
    target.write_text(header + result.module_source.lstrip("\n").rstrip("\n")
                      + f"\n\n\n__ewald__ = {manifest!r}   # plugin manifest (ewald plugins)\n",
                      encoding="utf-8", newline="\n")
    (tests / f"test_{result.module_name}.py").write_text(result.test_source, encoding="utf-8", newline="\n")
    return target


__all__ = ["forge", "verify", "save_plugin", "ForgeResult", "SUBMIT_TOOL", "CONTRACT"]
