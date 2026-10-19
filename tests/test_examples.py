import filecmp

import numpy as np
import pytest

from ewald import examples
from ewald.__main__ import main


@pytest.mark.parametrize("ex", examples.EXAMPLES, ids=lambda ex: ex.file)
def test_every_example_opens(ex):
    pipe = examples.load(ex.file)
    data = pipe.compute(pipe.num_frames - 1)
    assert data.count > 0 and np.isfinite(data.positions).all()
    assert all(t.isalpha() for t in map(str, data.types))  # element names, no type_map needed


def test_example_contents():
    assert examples.load("Cu_md.extxyz").num_frames == 10
    assert "Force" in examples.load("Cu_md.extxyz").compute(3)
    assert examples.load("XDATCAR.Si").num_frames == 10
    assert examples.load("Cu_void.lammpstrj").num_frames == 5
    chg = examples.load("CHGCAR.Si").compute(0)
    rho = chg.grids["charge-density"]["Charge Density"]
    assert rho.shape == (16, 16, 16)
    assert rho.mean() * chg.cell.volume == pytest.approx(32.0)  # 8 Si x 4 valence electrons
    assert sorted(set(examples.load("POSCAR.SrTiO3").compute(0).types)) == ["O", "Sr", "Ti"]
    with pytest.raises(KeyError, match="POSCAR.SrTiO3"):
        examples.path("nope.xyz")


def test_examples_match_their_generator(tmp_path):
    from ewald.examples import make_examples

    make_examples.main(tmp_path)
    for ex in examples.EXAMPLES:
        assert filecmp.cmp(tmp_path / ex.file, ex.path, shallow=False), ex.file


def test_examples_cli(tmp_path, capsys):
    main(["examples"])
    out = capsys.readouterr().out
    assert all(ex.file in out for ex in examples.EXAMPLES)
    main(["examples", "--copy", str(tmp_path / "ex")])
    assert sorted(p.name for p in (tmp_path / "ex").iterdir()) == sorted(ex.file for ex in examples.EXAMPLES)
