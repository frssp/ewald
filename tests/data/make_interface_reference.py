"""Regenerate tests/data/interface_*.extxyz and interface_reference.json: interfaces built by
sim-agent's own builder (frssp/sim_agent src/sim_agent/tools/interface_utils.py), with the side
indices and filter periods sim-agent stores for them, so tests/test_potential.py can check that
the lineup's sides and periods derived from the structures alone agree. Never edit the outputs
by hand. Needs a frssp/sim_agent checkout (SIM_AGENT_SRC, default ../sim_agent/src) and pymatgen:

    SIM_AGENT_SRC=~/sim_agent/src python tests/data/make_interface_reference.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.environ.get("SIM_AGENT_SRC", os.path.join(os.path.dirname(__file__), "..", "..", "..", "sim_agent", "src")))
warnings.filterwarnings("ignore")
import sim_agent.tools.interface_utils as iu
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor


def rocksalt(a, c, an): return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), [c, an], [[0,0,0],[.5,.5,.5]])
def zincblende(a, c, an): return Structure.from_spacegroup("F-43m", Lattice.cubic(a), [c, an], [[0,0,0],[.25,.25,.25]])


def build(sub, film, sm, fm, t=14.0, gap=2.0, max_area=60, max_strain=0.05):
    rows = iu.search_matches(sub, film, [sm], [fm], max_area, max_strain, film_thickness=t, substrate_thickness=t, gap=gap)
    rows = [r for r in rows if r["n_usable_terminations"]]
    if not rows: return None
    row = rows[0]
    b = iu._builder(sub, film, fm, sm, 1e-3, max_strain)
    match = iu.match_from_dict(row["match"])
    terms = [x for x in iu.terminations(b, film, sub, match, t, t, gap) if x.seams_equivalent]
    term_key = b.terminations[terms[0].as_dict()["index"]]
    iface, _fsl, _ssl = iu.build_superlattice(sub, film, sm, fm, match, term_key, b, t, t, gap)
    p_sub, p_film = iu.period_along_normal(sub, sm), iu.period_along_normal(film, fm)
    sub_ref = sub.get_primitive_structure(tolerance=0.01)
    film_ref = iu.strained_film_reference(film, fm, match)
    si, fi = iu.side_indices(iface)
    from pymatgen.core import Structure as S
    return S.from_sites(iface.sites), sub_ref, film_ref, {"sub": si, "film": fi, "p_sub": p_sub, "p_film": p_film}

CASES = {
    "TiN(100)/MgO(100)": (rocksalt(4.24, "Ti", "N"), rocksalt(4.21, "Mg", "O"), (1, 0, 0), (1, 0, 0)),
    "GaAs(110)/AlAs(110)": (zincblende(5.65, "Ga", "As"), zincblende(5.66, "Al", "As"), (1, 1, 0), (1, 1, 0)),
}


if __name__ == "__main__":
    import json

    from ase.io import write
    HERE = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for name, key in (("GaAs(110)/AlAs(110)", "gaas_alas_110"), ("TiN(100)/MgO(100)", "tin_mgo_100")):
        iface, sref, fref, meta = build(*CASES[name])
        write(HERE + f"/interface_{key}.extxyz", AseAtomsAdaptor.get_atoms(iface), format="extxyz")
        for role, s in (("substrate", sref), ("film", fref)):
            write(HERE + f"/interface_{key}_{role}.extxyz", AseAtomsAdaptor.get_atoms(s), format="extxyz")
        out[key] = {"source": f"{name}, built by sim-agent tools/interface_utils.py (build_superlattice, "
                    "period_along_normal, side_indices) at frssp/sim_agent 27ff1de",
                    "substrate": meta["sub"], "film": meta["film"],
                    "period_substrate_A": meta["p_sub"], "period_film_A": meta["p_film"]}
    with open(HERE + "/interface_reference.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
