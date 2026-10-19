"""Write the DOS / band structure fixtures in this folder: vasprun.xml (+ KPOINTS) and vaspout.h5.

Both hold the same made-up spin-polarized GaAs calculation: 2 atoms in the fcc primitive
cell, a line-mode path G-X-W | K-G (6 points per line, so W -> K jumps), 8 bands with an
indirect gap (VBM at G, CBM at X; the down channel 0.1 eV higher), LORBIT = 11 PDOS on 41
energies and band projections. The XML follows what VASP 6 writes (pymatgen's Vasprun reads
it, tests/test_electronic.py compares); the HDF5 file the py4vasp schema
(py4vasp/_raw/definition.py). Run: python tests/data/vasp/electronic/make_electronic.py
"""
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(11)

A = 5.65
LATTICE = 0.5 * A * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
FRAC = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
TYPES = ["Ga", "As"]
ORBITALS = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2"]
POINTS = [("\\Gamma", [0, 0, 0]), ("X", [0.5, 0, 0.5]), ("X", [0.5, 0, 0.5]), ("W", [0.5, 0.25, 0.75]),
          ("K", [0.375, 0.375, 0.75]), ("\\Gamma", [0, 0, 0])]
DIV = 6
EFERMI = 0.5
NB, NE = 8, 41


def kpath():
    pts = []
    for s in range(0, len(POINTS), 2):
        a, b = np.array(POINTS[s][1], float), np.array(POINTS[s + 1][1], float)
        pts += [a + (b - a) * t for t in np.linspace(0, 1, DIV)]
    return np.array(pts)


K = kpath()
NK = len(K)


def bands():
    """(2, nk, nb): 4 valence bands (top at G = 0), 4 conduction bands (bottom at X = 1)."""
    kc = K @ np.linalg.inv(LATTICE).T
    g = np.linalg.norm(kc, axis=1)                              # |k| from Gamma
    x = np.linalg.norm(kc - np.array([0.5, 0, 0.5]) @ np.linalg.inv(LATTICE).T, axis=1)
    e = np.zeros((NK, NB))
    for b in range(4):
        e[:, b] = -(3 - b) * 1.5 - 4 * g ** 2 * (1 + 0.3 * b)
    for b in range(4, NB):
        e[:, b] = 1.0 + (b - 4) * 1.2 + 3 * x ** 2 * (1 + 0.2 * b)
    up = e
    down = e + 0.1
    return np.stack([up, down]) + EFERMI - 0.5  # VBM (up) at 0, CBM (up) at 1, E_F = 0.5 in the gap


EIG = bands()
OCC = (EIG < EFERMI).astype(float)
ENERGIES = np.linspace(-10, 10, NE) + EFERMI
PDOS = rng.random((2, 2, len(ORBITALS), NE)) * np.exp(-0.05 * (ENERGIES - EFERMI) ** 2)  # (atom, spin, orb, E)
PDOS[..., np.abs(ENERGIES - EFERMI) < 0.6] = 0.0                                          # the gap
TDOS = PDOS.sum(axis=(0, 2)) * 1.1                              # interstitial part on top
PROJ = rng.random((2, NK, NB, 2, len(ORBITALS)))
PROJ /= PROJ.sum(axis=(3, 4), keepdims=True) * 1.2              # (spin, k, band, atom, orb), sums to 1/1.2


def f(x):
    return f"{x:16.8f}"


def vasprun():
    out = ['<?xml version="1.0" encoding="ISO-8859-1"?>', "<modeling>", " <generator>",
           '  <i name="program" type="string">vasp </i>', '  <i name="version" type="string">6.4.2  </i>',
           " </generator>", " <incar>", '  <i type="int" name="ISPIN">     2</i>',
           '  <i type="int" name="LORBIT">    11</i>', " </incar>", " <kpoints>",
           '  <generation param="listgenerated" >', f'   <i type="int" name="divisions">     {DIV} </i>']
    out += [f"   <v>{''.join(f(c) for c in p)} </v>" for _, p in POINTS]
    out += ["  </generation>", '  <varray name="kpointlist" >']
    out += [f"   <v>{''.join(f(c) for c in k)} </v>" for k in K]
    out += ["  </varray>", '  <varray name="weights" >']
    out += [f"   <v>{f(1 / NK)} </v>" for _ in K]
    out += ["  </varray>", " </kpoints>", " <parameters>", '  <separator name="electronic" >',
            '   <i name="NELECT">     8.00000000</i>', '   <i type="int" name="NELM">    60</i>',
            '   <i type="int" name="IBRION">    -1</i>', '   <separator name="electronic spin" >',
            '    <i type="int" name="ISPIN">     2</i>', '    <i type="logical" name="LNONCOLLINEAR"> F  </i>',
            "   </separator>", "  </separator>", '  <separator name="dos" >',
            '   <i type="int" name="NEDOS">    41</i>', '   <i type="int" name="LORBIT">    11</i>',
            "  </separator>", " </parameters>", " <atominfo>", "  <atoms>       2 </atoms>",
            "  <types>       2 </types>", '  <array name="atoms" >', '   <dimension dim="1">ion</dimension>',
            '   <field type="string">element</field>', '   <field type="int">atomtype</field>', "   <set>",
            "    <rc><c>Ga</c><c>   1</c></rc>", "    <rc><c>As</c><c>   2</c></rc>", "   </set>", "  </array>",
            '  <array name="atomtypes" >', '   <dimension dim="1">type</dimension>',
            '   <field type="int">atomspertype</field>', '   <field type="string">element</field>',
            "   <field>mass</field>", "   <field>valence</field>",
            '   <field type="string">pseudopotential</field>', "   <set>",
            "    <rc><c>   1</c><c>Ga</c><c>     69.72300000</c><c>      3.00000000</c>"
            "<c>  PAW_PBE Ga 08Apr2002                 </c></rc>",
            "    <rc><c>   1</c><c>As</c><c>     74.92200000</c><c>      5.00000000</c>"
            "<c>  PAW_PBE As 22Sep2009                 </c></rc>", "   </set>", "  </array>", " </atominfo>"]

    def structure(name=None, indent=" "):
        head = f'{indent}<structure name="{name}" >' if name else f"{indent}<structure>"
        lines = [head, f"{indent} <crystal>", f'{indent}  <varray name="basis" >']
        lines += [f"{indent}   <v>{''.join(f(c) for c in v)} </v>" for v in LATTICE]
        lines += [f"{indent}  </varray>", f'{indent}  <i name="volume">{f(abs(np.linalg.det(LATTICE)))} </i>',
                  f"{indent} </crystal>", f'{indent} <varray name="positions" >']
        lines += [f"{indent}  <v>{''.join(f(c) for c in v)} </v>" for v in FRAC]
        lines += [f"{indent} </varray>", f"{indent}</structure>"]
        return lines

    out += structure("initialpos")
    out += [" <calculation>", "  <scstep>", "   <energy>", '    <i name="e_fr_energy">    -9.50000000 </i>',
            '    <i name="e_wo_entrp">    -9.50000000 </i>', '    <i name="e_0_energy">    -9.50000000 </i>',
            "   </energy>", "  </scstep>"]
    out += structure(indent="  ")
    out += ['  <varray name="forces" >'] + ["   <v>       0.00000000       0.00000000       0.00000000 </v>"] * 2
    out += ["  </varray>", '  <varray name="stress" >']
    out += ["   <v>       0.00000000       0.00000000       0.00000000 </v>"] * 3
    out += ["  </varray>", "  <energy>", '   <i name="e_fr_energy">    -9.50000000 </i>',
            '   <i name="e_wo_entrp">    -9.50000000 </i>', '   <i name="e_0_energy">    -9.50000000 </i>',
            "  </energy>", "  <eigenvalues>", "   <array>", '    <dimension dim="1">band</dimension>',
            '    <dimension dim="2">kpoint</dimension>', '    <dimension dim="3">spin</dimension>',
            "    <field>eigene</field>", "    <field>occ</field>", "    <set>"]

    def eig_block(pad):
        lines = []
        for s in range(2):
            lines.append(f'{pad}<set comment="spin {s + 1}">')
            for k in range(NK):
                lines.append(f'{pad} <set comment="kpoint {k + 1}">')
                lines += [f"{pad}  <r>{f(EIG[s, k, b])}{f(OCC[s, k, b])} </r>" for b in range(NB)]
                lines.append(f"{pad} </set>")
            lines.append(f"{pad}</set>")
        return lines

    out += eig_block("     ")
    out += ["    </set>", "   </array>", "  </eigenvalues>", "  <separator> </separator>", "  <dos>",
            f'   <i name="efermi">{f(EFERMI)} </i>', "   <total>", "    <array>",
            '     <dimension dim="1">gridpoints</dimension>', '     <dimension dim="2">spin</dimension>',
            "     <field>energy</field>", "     <field>total</field>", "     <field>integrated</field>", "     <set>"]
    de = ENERGIES[1] - ENERGIES[0]
    for s in range(2):
        out.append(f'      <set comment="spin {s + 1}">')
        integ = np.cumsum(TDOS[s]) * de
        out += [f"       <r>{f(ENERGIES[i])}{f(TDOS[s, i])}{f(integ[i])} </r>" for i in range(NE)]
        out.append("      </set>")
    out += ["     </set>", "    </array>", "   </total>", "   <partial>", "    <array>",
            '     <dimension dim="1">gridpoints</dimension>', '     <dimension dim="2">spin</dimension>',
            '     <dimension dim="3">ion</dimension>', "     <field>energy</field>"]
    out += [f"     <field>{o:>5}</field>" for o in ORBITALS]
    out.append("     <set>")
    for a in range(2):
        out.append(f'      <set comment="ion {a + 1}">')
        for s in range(2):
            out.append(f'       <set comment="spin {s + 1}">')
            out += [f"        <r>{f(ENERGIES[i])}{''.join(f(x) for x in PDOS[a, s, :, i])} </r>" for i in range(NE)]
            out.append("       </set>")
        out.append("      </set>")
    out += ["     </set>", "    </array>", "   </partial>", "  </dos>", "  <projected>", "   <eigenvalues>",
            "    <array>", '     <dimension dim="1">band</dimension>', '     <dimension dim="2">kpoint</dimension>',
            '     <dimension dim="3">spin</dimension>', "     <field>eigene</field>", "     <field>occ</field>",
            "     <set>"]
    out += eig_block("      ")
    out += ["     </set>", "    </array>", "   </eigenvalues>", "   <array>",
            '    <dimension dim="1">ion</dimension>', '    <dimension dim="2">band</dimension>',
            '    <dimension dim="3">kpoint</dimension>', '    <dimension dim="4">spin</dimension>']
    out += [f"    <field>{o:>5}</field>" for o in ORBITALS]
    out.append("    <set>")
    for s in range(2):
        out.append(f'     <set comment="spin{s + 1}">')
        for k in range(NK):
            out.append(f'      <set comment="kpoint {k + 1}">')
            for b in range(NB):
                out.append(f'       <set comment="band {b + 1}">')
                out += [f"        <r>{''.join(f(x) for x in PROJ[s, k, b, a])} </r>" for a in range(2)]
                out.append("       </set>")
            out.append("      </set>")
        out.append("     </set>")
    out += ["    </set>", "   </array>", "  </projected>", " </calculation>"]
    out += structure("finalpos")
    out.append("</modeling>")
    with open(os.path.join(HERE, "vasprun.xml"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    lines = ["G-X-W K-G", str(DIV), "Line-mode", "Reciprocal"]
    for s in range(0, len(POINTS), 2):
        for lab, p in POINTS[s:s + 2]:
            lines.append(" ".join(f"{c:.6f}" for c in p) + f" ! {lab}")
        lines.append("")
    with open(os.path.join(HERE, "KPOINTS"), "w") as fh:
        fh.write("\n".join(lines))


def vaspout():
    with h5py.File(os.path.join(HERE, "vaspout.h5"), "w") as f:
        for k, v in zip(("major", "minor", "patch"), (6, 5, 1)):
            f[f"version/{k}"] = v
        f["results/positions/ion_types"] = np.array([t.ljust(2) for t in TYPES], dtype="S2")
        f["results/positions/number_ion_types"] = np.array([1, 1])
        f["results/positions/scale"] = 1.0
        f["results/positions/lattice_vectors"] = LATTICE
        f["results/positions/position_ions"] = FRAC
        f["intermediate/ion_dynamics/scale"] = 1.0
        f["intermediate/ion_dynamics/lattice_vectors"] = LATTICE[None]
        f["intermediate/ion_dynamics/position_ions"] = FRAC[None]
        f["input/kpoints/mode"] = np.bytes_("line")
        f["input/kpoints/number_kpoints"] = DIV
        f["input/kpoints/coordinates_kpoints"] = np.array([p for _, p in POINTS], float)
        f["input/kpoints/labels_kpoints"] = np.array([lab for lab, _ in POINTS], dtype="S")
        f["input/kpoints/positions_labels_kpoints"] = np.arange(1, len(POINTS) + 1)
        g = f.create_group("results/electron_eigenvalues")
        g["eigenvalues"] = EIG
        g["fermiweights"] = OCC
        g["kpoint_coords"] = K
        g["kpoints_symmetry_weight"] = np.full(NK, 1 / NK)
        d = f.create_group("results/electron_dos")
        d["efermi"] = EFERMI
        d["energies"] = ENERGIES
        d["dos"] = TDOS
        d["dospar"] = np.transpose(PDOS, (1, 0, 2, 3))                 # (spin, atom, orbital, E)
        p = f.create_group("results/projectors")
        p["lchar"] = np.array([o.ljust(5) for o in ORBITALS], dtype="S5")
        p["par"] = np.transpose(PROJ, (0, 3, 4, 1, 2))                  # (spin, atom, orbital, k, band)


if __name__ == "__main__":
    vasprun()
    vaspout()
