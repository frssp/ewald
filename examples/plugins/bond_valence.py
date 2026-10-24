"""Example AI tool plugin: bond valence sums of the cations in the current frame.

``bond_valence_sums`` gives the AI model (the AI chat, ``ewald ask``, the MCP server) one more
action: for every cation element, the mean, min and max of the bond valence sum
V_i = sum_j exp((R0 - r_ij) / b) over its anion neighbours within ``cutoff``, with b = 0.37 A and
R0 from Brown & Altermatt, Acta Cryst. B41, 244 (1985) and Brese & O'Keeffe, Acta Cryst. B47, 192
(1991) for the oxides below. It only reads the pipeline, so it writes no cell (the AI changes
results only through the notebook). The AI eval suite uses it (``ewald eval --tasks plugin-*``).
See docs/plugins/tools.md.

Copy this file into ewald's plugin folder (``ewald plugins`` prints where it is) and the AI can
call it; from Python, ``bond_valence_sums(toolset)`` with a :class:`ewald.llm.Toolset`.
"""
import numpy as np

from ewald.plugin_api import ai_tool

#: R0 (Angstrom) of cation-O bonds, for the common oxidation state.
R0_OXIDE = {"Li": 1.466, "Na": 1.803, "Mg": 1.693, "Al": 1.651, "Si": 1.624, "K": 2.132, "Ca": 1.967,
            "Ti": 1.815, "V": 1.784, "Cr": 1.724, "Mn": 1.790, "Fe": 1.759, "Co": 1.692, "Ni": 1.654,
            "Cu": 1.679, "Zn": 1.704, "Sr": 2.118, "Zr": 1.937, "Ba": 2.285, "La": 2.172, "Pb": 2.112}


@ai_tool(schema={"type": "object", "properties": {
    "cutoff": {"type": "number", "description": "largest cation-O distance counted, Angstrom (default 3.5)"},
    "frame": {"type": "integer", "description": "frame (default: the current one)"}}})
def bond_valence_sums(toolset, cutoff=3.5, frame=None):
    """Bond valence sums (valence units) of the cations of an oxide in one frame: mean, min and max per
    element, from the cation-O distances within cutoff (R0 of Brown & Altermatt, b = 0.37 A)."""
    from ewald.neighbors import find_pairs

    pipe = toolset._need_pipeline()
    frame = toolset.frame if frame is None else int(frame) % pipe.num_frames
    data = pipe.compute(frame)
    types = np.asarray(data.types).astype(str)
    if not np.any(types == "O"):
        raise ValueError("no O atoms: this tool knows cation-O bonds only")
    i, j, _delta, dist = find_pairs(data.positions, data.cell, float(cutoff))
    i, j, dist = np.asarray(i), np.asarray(j), np.asarray(dist)
    keep = types[j] == "O"
    out = {}
    for element in sorted(set(types[i[keep]]) - {"O"}):
        if element not in R0_OXIDE:
            out[element] = "no R0 for this element"
            continue
        bonds = keep & (types[i] == element)
        s = np.exp((R0_OXIDE[element] - dist[bonds]) / 0.37)
        sums = np.bincount(i[bonds], weights=s, minlength=data.count)[types == element]
        out[element] = {"mean": round(float(sums.mean()), 3), "min": round(float(sums.min()), 3),
                        "max": round(float(sums.max()), 3), "atoms": int(sums.size), "R0": R0_OXIDE[element]}
    return {"frame": frame, "cutoff": float(cutoff), "b": 0.37, "bond_valence_sums": out}
