"""Example simulation code plugin: Quantum ESPRESSO (pw.x) next to the built-in VASP and LAMMPS.

Copy this file into ewald's plugin folder (``ewald plugins`` prints where it is). Then
Preferences › Codes lists Quantum ESPRESSO, ``run_info("md/pw.out")`` reads the run's
``pw.in`` (kind, ensemble, time step, temperature), and folders holding a pw.x run show up as
steps in the Workflows side bar. ASE already reads the structures (``espresso-out``).

A code is one :class:`ewald.codes.SimulationCode`: the files it writes, the files that mark
its run folders, its readers, settings rows (``options``, named ``codes.<name>.*``) and hooks.
"""
import os
import re

from ewald.plugin_api import SimulationCode, simulation_code

#: pw.x time unit: Rydberg atomic units, 4.8378e-17 s = 4.8378e-2 fs
RY_FS = 4.8378e-2


def read_pw_input(path) -> dict:
    """``key = value`` pairs of the namelists of a pw.x input (lower-case keys, quotes dropped)."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.split("!", 1)[0]
            for key, value in re.findall(r"([A-Za-z_][\w()]*)\s*=\s*('[^']*'|\"[^\"]*\"|[^,\s]+)", line):
                value = value.strip("'\"")
                try:
                    value = float(value.replace("d", "e").replace("D", "E"))
                except ValueError:
                    pass
                out[key.lower()] = value
    return out


def _input_next_to(path):
    folder = os.path.dirname(os.path.abspath(path))
    for name in ("pw.in", "pw.inp", "espresso.pwi"):
        cand = os.path.join(folder, name)
        if os.path.isfile(cand):
            return cand
    return None


def qe_run_info(info, path):
    """Fill ``info`` (:class:`ewald.runinfo.RunInfo`) from the pw.x input next to ``path``."""
    source = _input_next_to(path)
    if source is None:
        info.notes.append("no pw.in next to the output; run parameters unknown")
        return
    p = read_pw_input(source)
    info.files.append(source)
    info.parameters = p
    calc = str(p.get("calculation", "scf")).lower()
    info.kind = {"md": "md", "vc-md": "md", "relax": "relax", "vc-relax": "relax"}.get(calc, "static")
    if isinstance(p.get("nstep"), float):
        info.steps = int(p["nstep"])
    if info.kind == "md":
        info.timestep = float(p.get("dt", 20.0)) * RY_FS
        if "dt" not in p:
            info.notes.append("dt not set: pw.x's default 20 Ry a.u. assumed")
        thermostat = str(p.get("ion_temperature", "not_controlled")).lower()
        info.ensemble = "NVE" if thermostat == "not_controlled" else ("NPT" if calc == "vc-md" else "NVT")
        if isinstance(p.get("tempw"), float) and thermostat != "not_controlled":
            info.temperature = (p["tempw"], p["tempw"])


def qe_workflow_step(name, folder, entries):
    """A folder with a pw.x input and output is one step whose result is the output."""
    from ewald.plugin_api import WorkflowStep

    files = {e.lower(): e for e in entries}
    out = next((files[f] for f in ("pw.out", "espresso.pwo") if f in files), None)
    if out is None or not any(f in files for f in ("pw.in", "pw.inp", "espresso.pwi")):
        return None
    step = WorkflowStep(name, work_dir=folder, path=os.path.join(folder, out))
    step.scalars["code"] = "qe"
    return step


@simulation_code
def quantum_espresso():
    return SimulationCode(
        "qe", "Quantum ESPRESSO", "Plane-wave DFT (pw.x): SCF, relaxations and Born-Oppenheimer MD.",
        files=("*.pwo", "pw.out", "espresso.pwo"), markers=("pw.in",), formats=("espresso-out",),
        run_info=qe_run_info, workflow_step=qe_workflow_step)
