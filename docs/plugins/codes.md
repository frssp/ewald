# Simulation code plugins

Everything EWALD knows about one simulation program lives in one entry of the code registry,
`ewald.codes`: which files it writes, which readers read them, its settings, how to tell what a
run was from its input files, what to check in its output, which folders hold its runs and which
per-step result files sit next to its trajectories. VASP and LAMMPS are the two built-in
entries; Quantum ESPRESSO, CP2K, GPAW or an in-house code are added the same way.

Preferences › Codes draws one section per registered code (its description, files, formats and
its settings), so a plugin code shows up there next to VASP and LAMMPS without GUI code.

```python
# <config dir>/plugins/qe_code.py
from ewald.plugin_api import Option, SimulationCode, simulation_code


def qe_run_info(info, path):
    """Fill info (ewald.runinfo.RunInfo) from the pw.x input next to path."""
    ...


@simulation_code
def quantum_espresso():
    return SimulationCode(
        "qe", "Quantum ESPRESSO", "Plane-wave DFT (pw.x).",
        files=("*.pwo", "pw.out"),            # file names it writes (case is ignored)
        markers=("pw.in",),                   # a file next to a trajectory that says "this code's run"
        formats=("espresso-out",),            # reader names, shown in Preferences
        options=[Option("codes.qe.pseudo_dir", "", "Pseudopotential folder")],
        run_info=qe_run_info)
```

The full example is [`examples/plugins/qe_code.py`](https://github.com/frssp/ewald/blob/main/examples/plugins/qe_code.py):
it reads `pw.in` for `run_info` (kind, ensemble, time step, temperature) and makes folders with
`pw.in` and `pw.out` steps of the [Workflows side bar](../user/workflows.md).

## Fields

| Field | What it does |
|---|---|
| `name`, `title`, `description` | one lower-case word (`"qe"`), the display name, one sentence |
| `files` | `fnmatch` patterns of the file names it writes; `ewald.codes.code_of(path)` matches them case-insensitively on every OS |
| `detect(path)` | for files known by content, not name (LAMMPS dumps called anything) |
| `markers` | file names whose presence next to a file says the run is this code's (`INCAR`, `log.lammps`) |
| `formats` | names of the readers of its formats |
| `options` | `Option` rows named `codes.<name>.<key>`; Preferences › Codes edits them, scripts read `settings.load()["codes.<name>.<key>"]` |
| `run_info(info, path)` | fills `ewald.runinfo.RunInfo` (code, kind, ensemble, timestep in fs, temperature, ...) from the input files |
| `frame_steps(info, path, steps)` | MD steps between stored frames when the frames do not say |
| `check(check, info, path)` | appends `ewald.runinfo.Finding`s to `check_run`'s result |
| `workflow_step(name, folder, entries)` | a `WorkflowStep` when `folder` holds one of its runs, else None |
| `thermo` | reader name -> per-step result file names next to such files (`{"xdatcar": ("OSZICAR",)}`), joined to frames by `import_file(..., thermo="auto")` |
| `read_thermo(path)` | reads one of those files into a table with a `Step` column |
| `open_options(path, settings, ask)` | reader keyword arguments when the GUI opens a file; `ask(title, label, text)` asks the user (None = cancelled) |

Hooks are functions or lazy `"module:function"` strings, imported on first use. A plugin code
cannot take the name of a built-in one. Settings of a plugin code that is not installed (yet)
are kept in `settings.json` and checked once the plugin is loaded.

As a package, name the function in an entry point:

```toml
[project.entry-points."ewald.codes"]
qe = "ewald_qe:quantum_espresso"
```

## Built-in codes

`vasp` and `lammps` belong to the built-in plugin groups of the same names, so
`ewald plugins --disable lammps` (or the switch in the Plugins side bar) removes the LAMMPS readers
and the LAMMPS code entry together. From a notebook:

```python
from ewald import codes
codes.codes()                       # {'vasp': <SimulationCode vasp (built-in)>, 'lammps': ..., 'qe': ...}
codes.code_of("md/XDATCAR").title   # 'VASP'
codes.get("lammps").as_dict()       # files, markers, formats, options, hooks
```
