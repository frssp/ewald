"""Regenerate tests/data/analysis_reference.npz: OVITO 3.16 and freud results for the analysis modifiers
(Histogram, SpatialBinning, ReduceProperty, BondAngle/BondLengthDistribution, FindRings, Steinhardt).

    pip install ovito ase freud-analysis
    python tests/data/make_analysis_reference.py

Inputs are stored next to the results, so the test suite needs neither OVITO nor freud.
"""
import os

import numpy as np
from ase.build import bulk
from ovito.data import BondType, DataCollection, Particles, SimulationCell
from ovito.modifiers import BondAngleDistributionModifier as BAD
from ovito.modifiers import BondLengthDistributionModifier as BLD
from ovito.modifiers import CreateBondsModifier, FindRingsModifier, ReducePropertyModifier as RED
from ovito.modifiers import HistogramModifier as HIST
from ovito.modifiers import SpatialBinningModifier as SB
from ovito.pipeline import Pipeline, StaticSource

rng = np.random.default_rng(7)
out = {}


def make(pos, cell, pbc, types, names=None, **props):
    dc = DataCollection()
    m = np.zeros((3, 4))
    m[:, :3] = cell.T
    sc = SimulationCell()
    sc[...] = m
    sc.pbc = tuple(bool(b) for b in pbc)
    dc.objects.append(sc)
    p = Particles()
    p.create_property("Position", data=pos)
    tp = p.create_property("Particle Type", data=types)
    for t in sorted(set(types)):
        tp.add_type_id(int(t), p, name=(names or {}).get(t, ""))
    for name, value in props.items():
        p.create_property(name.replace("_", " "), data=value)
    dc.objects.append(p)
    return dc


def run(dc, *mods):
    pipe = Pipeline(source=StaticSource(data=dc))
    for m in mods:
        pipe.modifiers.append(m)
    return pipe.compute()


# --------------------------------------------------------------- histogram, binning, reduce
# Noisy fcc in a triclinic cell, a few atoms outside it; once fully periodic, once open along z.
base = bulk("Cu", "fcc", a=3.6).repeat((5, 4, 4))
for tag, pbc in [("tri", (True, True, True)), ("open", (True, True, False))]:
    pos = base.get_positions() + rng.normal(0, 0.3, (len(base), 3))
    pos[:4] += base.cell[0] * 1.3
    pos[4:7] -= base.cell[2] * 0.2
    n = len(pos)
    mass = rng.uniform(1, 2, n)
    sel = (rng.random(n) < 0.4).astype(np.int32)
    out.update({f"{tag}/pos": pos, f"{tag}/cell": np.array(base.cell), f"{tag}/pbc": np.array(pbc),
                f"{tag}/mass": mass, f"{tag}/sel": sel})
    dc = make(pos, np.array(base.cell), pbc, np.ones(n, np.int32), Mass=mass, Selection=sel)

    for key, kw in [("plain", dict(property="Mass", bin_count=37)),
                    ("fixed", dict(property="Mass", bin_count=10, fix_xrange=True, xrange_start=1.2,
                                   xrange_end=1.9)),
                    ("rel", dict(property="Mass", bin_count=13, normalization=HIST.NormalizationMode.RelativeFrequency)),
                    ("pdf", dict(property="Position.Y", bin_count=21,
                                 normalization=HIST.NormalizationMode.ProbabilityDensity)),
                    ("onlysel", dict(property="Position.Z", bin_count=9, only_selected=True)),
                    ("select", dict(property="Mass", bin_count=8, select_elements=True, selection_start=1.7,
                                    selection_end=1.3))]:
        d = run(dc, HIST(**kw))
        t = d.tables[f"histogram[{kw['property']}]"]
        out[f"{tag}/hist/{key}/xy"] = np.array(t.xy())
        if "Histogram.num_selected" in d.attributes:
            out[f"{tag}/hist/{key}/num_selected"] = np.array(d.attributes["Histogram.num_selected"])
            out[f"{tag}/hist/{key}/Selection"] = np.array(d.particles["Selection"])

    ops = ["Mean", "Sum", "SumVol", "Min", "Max"]
    cases = []
    for direction, counts in [("X", 7), ("Y", 5), ("Z", 9), ("XY", (4, 3)), ("XZ", (3, 5)), ("YZ", (2, 4)),
                              ("XYZ", (3, 4, 5))]:
        for op in ops:
            cases.append((f"{direction}/{op}", dict(property="Mass", direction=direction, bin_count=counts,
                                                     reduction_operation=op)))
    cases += [("Y/deriv", dict(property="Mass", direction="Y", bin_count=6, first_derivative=True)),
              ("Z/deriv", dict(property="Mass", direction="Z", bin_count=6, first_derivative=True)),
              ("X/onlysel", dict(property="Mass", direction="X", bin_count=4, only_selected=True)),
              ("XYZ/unity", dict(property="", direction="XYZ", bin_count=(2, 3, 2), reduction_operation="Sum")),
              ("Z/vector", dict(property="Position", direction="Z", bin_count=4))]
    for key, kw in cases:
        kw = dict(kw)
        kw["direction"] = getattr(SB.Direction, kw["direction"])
        kw["reduction_operation"] = getattr(SB.Operation, kw.get("reduction_operation", "Mean"))
        d = run(dc, SB(**kw))
        if "binning" in d.tables:
            t = d.tables["binning"]
            out[f"{tag}/bin/{key}/x"] = np.array(t.xy())[:, 0]
            out[f"{tag}/bin/{key}/y"] = np.array(t.y)
        else:
            g = d.grids["binning"]
            out[f"{tag}/bin/{key}/grid"] = np.array(g.properties[0]).reshape(g.shape, order="F")

    for op in RED.Operation.__members__:
        for prop, extra in [("Mass", {}), ("Position", {}), ("Mass", dict(only_selected=True))]:
            d = run(dc, RED(property=prop, reduction_operation=getattr(RED.Operation, op), **extra))
            for k, v in d.attributes.items():
                if k.startswith("ReduceProperty."):
                    out[f"{tag}/reduce/{'sel' if extra else 'all'}/{k}"] = np.array(v)

# --------------------------------------------------------------- bond angle / length distributions
dia = bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3))
pos = dia.get_positions() + rng.normal(0, 0.12, (len(dia), 3))
types = np.where(np.arange(len(dia)) % 3 == 0, 1, 2).astype(np.int32)
sel = (rng.random(len(dia)) < 0.3).astype(np.int32)
dc = make(pos, np.array(dia.cell), (True,) * 3, types, names={1: "A", 2: "B"}, Selection=sel)
d = run(dc, CreateBondsModifier(cutoff=2.9))
bonds = d.particles.bonds
topo = np.array(bonds.topology)
bt = np.where(np.arange(len(topo)) % 4 == 0, 1, 2).astype(np.int32)
bsel = (rng.random(len(topo)) < 0.5).astype(np.int32)
out.update({"bond/pos": pos, "bond/cell": np.array(dia.cell), "bond/types": types, "bond/sel": sel,
            "bond/topology": topo, "bond/image": np.array(bonds.pbc_vectors), "bond/bond_type": bt,
            "bond/bond_sel": bsel})
d = d.clone()
b = d.particles_.bonds_
prop = b.create_property("Bond Type", data=bt)
for t, name in [(1, "s"), (2, "l")]:
    prop.types.append(BondType(id=t, name=name))
b.create_property("Selection", data=bsel)
P = BAD.Partition
for key, kw in [("off", {}), ("cos", dict(cosine_mode=True, bins=40)), ("ptype", dict(partition=P.ByParticleType)),
                ("psel", dict(partition=P.ByParticleSelection)), ("btype", dict(partition=P.ByBondType)),
                ("bsel", dict(partition=P.ByBondSelection, bins=30))]:
    r = run(d, BAD(**kw))
    t = r.tables["bond-angle-distr"]
    out[f"bond/angle/{key}/y"] = np.array(t.y)
    out[f"bond/angle/{key}/names"] = np.array(t.y.component_names or ["Count"])
    kw = {k: (getattr(BLD.Partition, v.name) if k == "partition" else v) for k, v in kw.items() if k != "cosine_mode"}
    r = run(d, BLD(**kw))
    t = r.tables["bond-length-distr"]
    out[f"bond/length/{key}/y"] = np.array(t.y)
    out[f"bond/length/{key}/names"] = np.array(t.y.component_names or ["Count"])
r = run(d, BLD(bins=25, length_cutoff=2.6))
out["bond/length/cut/y"] = np.array(r.tables["bond-length-distr"].y)

# --------------------------------------------------------------- rings in disordered Si networks
for tag, noise, cutoff, rmax in [("ringa", 0.35, 2.75, 10), ("ringb", 0.5, 2.7, 12), ("ringc", 0.2, 2.6, 8)]:
    p = dia.get_positions() + rng.normal(0, noise, (len(dia), 3))
    d = run(make(p, np.array(dia.cell), (True,) * 3, np.ones(len(dia), np.int32)),
            CreateBondsModifier(cutoff=cutoff), FindRingsModifier(minimum_ring_size=3, maximum_ring_size=rmax))
    out.update({f"{tag}/pos": p, f"{tag}/cell": np.array(dia.cell), f"{tag}/max": np.array(rmax),
                f"{tag}/topology": np.array(d.particles.bonds.topology),
                f"{tag}/image": np.array(d.particles.bonds.pbc_vectors)})
    h = d.tables["ring-size-histogram"]
    out[f"{tag}/hist"] = np.array(h.xy())
    for k, t in d.tables.items():
        if k.endswith("-rings"):
            out[f"{tag}/{k}"] = np.array(t.y).reshape(-1, int(k.split("-")[0]))

# --------------------------------------------------------------- Steinhardt order parameters (freud)
import freud  # noqa: E402

for tag, atoms, noise in [("fcc", bulk("Cu", "fcc", a=3.615, cubic=True).repeat((4, 4, 4)), 0.15),
                          ("bcc", bulk("Fe", "bcc", a=2.87, cubic=True).repeat((5, 5, 5)), 0.1),
                          ("hcp", bulk("Mg", "hcp", a=3.21, c=5.21).repeat((5, 5, 3)), 0.12),
                          ("liquid", bulk("Cu", "fcc", a=3.615).repeat((5, 5, 5)), 0.5)]:
    atoms.set_cell(atoms.cell.standard_form()[0], scale_atoms=True)  # freud boxes are upper triangular
    p = atoms.get_positions() + rng.normal(0, noise, (len(atoms), 3))
    cell = np.array(atoms.cell)
    out.update({f"st/{tag}/pos": p, f"st/{tag}/cell": cell})
    box = freud.box.Box.from_matrix(cell.T)
    frac = p @ np.linalg.inv(cell)
    pts = (frac - np.floor(frac) - 0.5) @ cell
    for nkey, query in [("k12", dict(num_neighbors=12, exclude_ii=True)),
                        ("k8", dict(num_neighbors=8, exclude_ii=True)),
                        ("rc", dict(r_max=3.2 if tag != "bcc" else 2.6, exclude_ii=True))]:
        for akey, average in [("plain", False), ("avg", True)]:
            st = freud.order.Steinhardt([4, 6, 8], average=average, wl=True, wl_normalize=True)
            st.compute((box, pts), neighbors=query)
            out[f"st/{tag}/{nkey}/{akey}/W"] = np.array(st.particle_order)
            st = freud.order.Steinhardt([4, 6, 8], average=average)
            st.compute((box, pts), neighbors=query)
            out[f"st/{tag}/{nkey}/{akey}/Q"] = np.array(st.particle_order)
        st = freud.order.Steinhardt(6, wl=True, wl_normalize=False)
        st.compute((box, pts), neighbors=query)
        out[f"st/{tag}/{nkey}/raw/W6"] = np.array(st.particle_order)

np.savez_compressed(os.path.join(os.path.dirname(__file__), "analysis_reference.npz"), **out)
print(f"wrote {len(out)} arrays")
