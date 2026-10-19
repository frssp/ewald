import logging

import numpy as np
import pytest
from ase.build import bulk
from ase.spacegroup import crystal

from ewald import radii
from ewald.data import DataCollection


def _srtio3(n=1):
    a = crystal(["Sr", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221,
                cellpar=[3.905] * 3 + [90] * 3)
    return DataCollection.from_ase(a * (n, n, n))


def test_tabulated_presets_match_their_sources():
    from ase.data import covalent_radii
    from ase.data.vdw_alvarez import vdw_radii as alvarez

    assert radii.element_radius("Cu", "covalent") == pytest.approx(covalent_radii[29])
    assert radii.element_radius("Cu", "slater") == 1.35          # Slater 1964
    assert radii.element_radius("Cu", "atomic (calc.)") == 1.45  # Clementi 1967
    assert radii.element_radius("Fe", "vdW") == pytest.approx(alvarez[26])
    assert radii.element_radius("Xe", "atomic") == pytest.approx(covalent_radii[54])  # not in Slater: covalent
    assert radii.element_radius("Fe3+", "atomic") == 1.40 and radii.element_radius("X1") is None
    with pytest.raises(ValueError):
        radii.preset_name("metallic")


def test_shannon_lookup_by_oxidation_cn_and_spin():
    assert radii.shannon_radius("Ti", 4, 6) == (0.605, 6)
    assert radii.shannon_radius("Ti", 4, 6, "crystal") == (0.745, 6)
    assert radii.shannon_radius("O", -2, 2) == (1.35, 2)
    assert radii.shannon_radius("Cl", -1, 8) == (1.81, 6)        # only VI tabulated: nearest CN
    assert radii.shannon_radius("Fe", 2, 6) == (0.78, 6)          # high spin by default
    assert radii.shannon_radius("Fe", 2, 6, spin="low") == (0.61, 6)
    assert radii.shannon_radius("Cu", 0) is None


def test_ionic_radii_guessed_from_structure():
    tab = radii.resolve(_srtio3(2), "ionic")
    assert tab.radii == {"Sr": 1.44, "Ti": 0.605, "O": 1.40}
    assert tab.coordination == {"Sr": 12, "Ti": 6, "O": 6}      # O: 2 Ti + 4 Sr
    assert {t: o for t, (o, _) in tab.oxidation.items()} == {"Sr": 2, "Ti": 4, "O": -2}
    rutile = crystal(["Ti", "O"], [(0, 0, 0), (0.3, 0.3, 0)], spacegroup=136, cellpar=[4.59, 4.59, 2.96, 90, 90, 90])
    assert radii.resolve(DataCollection.from_ase(rutile), "ionic").coordination == {"Ti": 6, "O": 3}
    nacl = radii.resolve(DataCollection.from_ase(bulk("NaCl", "rocksalt", 5.64)), "ionic (crystal)")
    assert nacl.radii == {"Na": 1.16, "Cl": 1.67}


def test_ionic_oxidation_sources_and_fallback(caplog):
    d = _srtio3()
    tab = radii.resolve(d, "ionic", oxidation_states={"Ti": 3}, coordination={"Ti": 6})
    assert tab.oxidation["Ti"] == (3, "given") and tab.radii["Ti"] == 0.67
    assert tab.oxidation["Sr"][0] == 2 and not tab.fallback   # Sr2+ Ti3+ O2- is not neutral: most common state
    named = d.copy()
    named["Particle Type"] = np.array(["Sr2+", "Ti3+", "O2-", "O2-", "O2-"])
    tab = radii.resolve(named, "ionic")
    assert tab.oxidation["Ti3+"] == (3, "type name") and tab.radii["Ti3+"] == 0.67
    charged = d.copy()
    charged["Charge"] = np.array([2.0, 4.0, -2.0, -2.0, -2.0])
    assert radii.resolve(charged, "ionic").oxidation["Ti"] == (4, "Charge")
    with caplog.at_level(logging.WARNING, logger="ewald.radii"):
        cu = radii.resolve(DataCollection.from_ase(bulk("Cu")), "ionic")
    assert cu.fallback == {"Cu": "oxidation state unknown"} and cu.radii["Cu"] == pytest.approx(1.32)
    assert "oxidation_states" in caplog.text
    assert radii.parse_oxidation_states("Sr=2, Ti=+4 O:2-") == {"Sr": 2, "Ti": 4, "O": -2}
    with pytest.raises(ValueError):
        radii.parse_oxidation_states("Ti four")


def test_default_radii_presets_and_overrides():
    from ewald.render import default_radii

    d = _srtio3()
    types = d.types
    r = default_radii(d, radius_preset="ionic", type_radii={"Sr": 2.0})
    assert r[types == "Sr"][0] == 2.0 and r[types == "Ti"][0] == pytest.approx(0.8 * 0.605)
    assert default_radii(d)[types == "O"][0] == pytest.approx(0.8 * 0.66)


def test_radii_table_is_packaged_with_citations():
    t = radii._tables()
    assert "Shannon" in t["_sources"]["shannon"] and "Slater" in t["_sources"]["atomic"]
    assert len(t["shannon"]) > 90 and t["oxidation_occurrence"]["O"]["-2"] > 1000


def test_bond_valence_guess_with_pymatgen():
    pytest.importorskip("pymatgen.core")
    tab = radii.resolve(_srtio3(), "ionic")
    assert tab.oxidation == {"Sr": (2, "bond valence"), "Ti": (4, "bond valence"), "O": (-2, "bond valence")}
