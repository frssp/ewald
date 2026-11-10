"""API review, load and verbs: errors that point the right way, and verbs that fail cleanly.

``load()`` used to answer every failure of a file that also parses as a text table (nearly any
file with a row of numbers) with "it is a table: read it with load_table(...)", also when the call
was wrong and the file fine. A verb whose first compute failed left its step in the pipeline, so
every later compute failed the same way.
"""
import copy
import pickle

import numpy as np
import pytest

import ewald.examples as examples
from ewald import ComputeCancelled, bond_angles, export_file, import_file, load, msd, plugins, rdf, vacf
from ewald.data import DataCollection
from ewald.io import IncompleteFileError
from ewald.io.remote import RemoteError
from ewald.modifiers import CoordinationAnalysis, WrapAtoms
from ewald.runs import Run, Runs
from ewald.verbs import Table

CU = examples.path("Cu_md.extxyz")
DUMP = examples.path("Cu_void.lammpstrj")
DATA = examples.path("CuZr.data")
POSCAR = examples.path("POSCAR.SrTiO3")


@pytest.fixture
def registry(monkeypatch):
    """A reader registry of its own, so readers registered in a test are gone after it."""
    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    yield plugins
    monkeypatch.setattr(plugins, "_registry", plugins._Registry())


@pytest.fixture
def table_file(tmp_path):
    """A collective-variable log: a table, and no structure."""
    path = tmp_path / "HILLS"
    path.write_text("#! FIELDS time phi psi\n1.0 -1.4 2.8\n2.0 -2.5 2.79\n", encoding="utf-8", newline="\n")
    return path


def message(excinfo) -> str:
    return str(excinfo.value)


# ---------------------------------------------------------------- A3: load() and reader keywords
@pytest.mark.parametrize("opener", [load, import_file])
def test_unknown_reader_keyword_names_load_the_format_and_its_keywords(opener):
    with pytest.raises(TypeError) as e:
        opener(DUMP, typemap={1: "Cu"})
    text = message(e)
    assert "load()/import_file() got an unexpected keyword argument 'typemap'" in text
    assert "lammps-dump reader" in text and "Cu_void.lammpstrj" in text
    assert "type_map=" in text and "sort_by_id=" in text          # what the reader does take
    assert "Did you mean 'type_map'?" in text
    assert "load_table" not in text and "LammpsDumpReader" not in text


@pytest.mark.parametrize("keywords", [{"typemap": {1: "Cu"}}, {"frames": ":2"}])
def test_keyword_typo_on_a_structure_file_is_not_called_a_table(keywords):
    with pytest.raises(TypeError) as e:            # was: ValueError "... it is a table ... load_table(...)"
        load(CU, **keywords)
    text = message(e)
    assert f"unexpected keyword argument {next(iter(keywords))!r}" in text
    assert "extxyz reader" in text and "takes no keywords" in text
    assert "load_table" not in text


def test_keyword_error_says_which_format_was_asked_for():
    with pytest.raises(TypeError, match=r"lammps-dump reader \(format='lammps/dump'\)"):
        load(DUMP, format="lammps/dump", typemap={})


def test_misspelled_keyword_of_load_itself_is_suggested():
    with pytest.raises(TypeError, match="Did you mean 'format'"):
        load(DUMP, fromat="lammps-dump")
    with pytest.raises(TypeError, match="Did you mean 'thermo'"):
        load(DUMP, themo=None)


def test_keywords_of_wrapped_builtin_readers_are_checked_against_the_reader():
    with pytest.raises(TypeError) as e:
        load(DATA, typemap={1: "Cu"})
    text = message(e)
    assert "lammps-data reader" in text and "atom_style=" in text and "Did you mean 'type_map'?" in text
    # the keywords the wrapper takes for ASE's names still work, as do the reader's own
    assert load(DATA, style="atomic", units="metal").compute(0).count > 0
    assert load(DATA, atom_style="atomic", type_map={1: "Cu", 2: "Zr"}).compute(0).count > 0
    assert load(DUMP, type_map={1: "Cu"}, sort_by_id=False).compute(0).count == 462


def test_volumetric_reader_no_longer_swallows_unknown_keywords(tmp_path):
    cube = tmp_path / "rho.cube"
    cube.write_text("rho\ntest\n 1 0.0 0.0 0.0\n 2 1.0 0.0 0.0\n 2 0.0 1.0 0.0\n 2 0.0 0.0 1.0\n"
                    " 1 0.0 0.0 0.0 0.0\n 0.1 0.2 0.3 0.4 0.5 0.6\n 0.7 0.8\n", encoding="utf-8", newline="\n")
    assert load(cube, pbc=True).compute(0).cell.pbc.all()
    with pytest.raises(TypeError, match=r"cube reader .* takes pbc=\. Did you mean 'pbc'\?"):
        load(cube, pcb=True)                       # was read as if nothing had been passed


def test_plugin_reader_keywords_come_from_its_signature(registry, tmp_path):
    def read_points(path, scale=1.0, *, origin=(0, 0, 0)):
        return [DataCollection(np.loadtxt(path, ndmin=2) * scale + np.asarray(origin))]

    def read_anything(path, **options):            # passes its options on: nothing to check them against
        return [DataCollection(np.loadtxt(path, ndmin=2) * options.get("factor", 1.0))]

    registry.register_reader("points", read_points, patterns=("*.pts",))
    registry.register_reader("loose", read_anything, patterns=("*.loose",))
    pts = tmp_path / "a.pts"
    pts.write_text("0 0 0\n1 1 1\n", encoding="utf-8", newline="\n")
    loose = tmp_path / "a.loose"
    loose.write_text("0 0 0\n1 1 1\n", encoding="utf-8", newline="\n")

    assert load(pts, scale=2.0, origin=(1, 0, 0)).compute(0).positions[1].tolist() == [3.0, 2.0, 2.0]
    with pytest.raises(TypeError) as e:
        load(pts, scal=2.0)
    text = message(e)
    assert "points reader" in text and "scale=, origin=" in text and "Did you mean 'scale'?" in text
    assert "load_table" not in text                # a.pts also parses as a table of numbers
    assert load(loose, factor=3.0, anything="goes").compute(0).positions[1].tolist() == [3.0, 3.0, 3.0]


def test_ase_formats_still_take_ase_keywords_and_report_unknown_ones():
    assert import_file(DATA, format="lammps-data-ase", style="atomic").compute(0).count > 0
    with pytest.raises(TypeError) as e:
        load(POSCAR, typemap={})                   # goes to ase.io.read, which cannot be checked beforehand
    text = message(e)
    assert "load()/import_file() got an unexpected keyword argument 'typemap'" in text
    assert "ase reader" in text and "read_vasp()" in text and "load_table" not in text


@pytest.mark.parametrize("call, error, says", [
    (lambda: load(CU, format="lammps-dump"), ValueError, "no 'ITEM: TIMESTEP' found"),   # the wrong format for a good file
    (lambda: load(CU, "lammps-dump"), ValueError, "no 'ITEM: TIMESTEP' found"),
    (lambda: load(POSCAR, format="no-such-format"), Exception, "no-such-format"),      # ASE's UnknownFileTypeError
    (lambda: load(examples.path("XDATCAR.Si"), join="bogus"), ValueError, "join must be one of"),
    (lambda: load(DUMP, type_map=5), TypeError, r"type_map maps the file's type ids .* cannot be 5"),
])
def test_a_wrong_call_on_a_structure_file_keeps_its_own_error(call, error, says):
    with pytest.raises(error, match=says) as e:
        call()
    assert "load_table" not in message(e) and "table" not in message(e)
    assert e.value.__cause__ is None               # the error of import_file, not one wrapped by load()
    assert type(e.value).__name__ in ("ValueError", "TypeError", "UnknownFileTypeError")


def test_join_error_is_the_one_import_file_raises():
    with pytest.raises(ValueError, match="join must be one of"):
        load(DUMP, join="bogus")


def test_a_file_that_cannot_be_reached_is_not_probed_as_a_table(registry, table_file, tmp_path, monkeypatch):
    import ewald.verbs

    probes = []
    real = ewald.verbs.load_table
    monkeypatch.setattr(ewald.verbs, "load_table", lambda path: probes.append(path) or real(path))

    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.extxyz", typemap={})
    with pytest.raises(FileNotFoundError):
        load(DUMP, thermo=tmp_path / "no-such-log.lammps")

    def unreachable(path):
        raise RemoteError("connection to cluster was lost")

    def refused(path):
        raise PermissionError(13, "Permission denied", str(path))

    registry.register_reader("unreachable", unreachable, patterns=("HILLS",), priority=1)
    with pytest.raises(RemoteError, match="connection to cluster was lost"):
        load(table_file)                           # was: "it is a table", though nothing had been read
    registry.register_reader("unreachable", refused, patterns=("HILLS",), priority=1)
    with pytest.raises(PermissionError):
        load(table_file)
    assert probes == []


def test_a_file_still_being_written_says_so(tmp_path):
    part = tmp_path / "part.extxyz"                # three atoms announced, one written yet
    part.write_text('3\nLattice="4 0 0 0 4 0 0 0 4" Properties=species:S:1:pos:R:3\nCu 0 0 0\n',
                    encoding="utf-8", newline="\n")
    with pytest.raises(IncompleteFileError, match="incomplete"):
        load(part)                                 # was: "it is a table (columns column 1)"


def test_a_real_table_still_gets_the_load_table_hint(table_file):
    with pytest.raises(ValueError) as e:
        load(table_file)
    text = message(e)
    assert f"load_table({str(table_file)!r})" in text and "columns time, phi, psi" in text
    # no reader knew the file: said in plain words, not with the name of ASE's exception
    assert "is not a structure file any reader recognizes, but it reads as a table" in text
    assert "UnknownFileTypeError" not in text
    assert e.value.__cause__ is not None
    with pytest.raises(ValueError, match=r"load_table\("):
        load(table_file, type_map={1: "Cu"})       # options do not make it a structure
    with pytest.raises(ValueError, match=r"load_table\("):
        load(table_file, format="lammps-dump")


# ---------------------------------------------------------------- A4: verbs are all or nothing
def failing_above(monkeypatch, limit: float, error: BaseException):
    """CoordinationAnalysis fails with ``error`` for a cutoff above ``limit`` (as a 40 A cutoff
    runs out of memory on a 10 A cell), and works below."""
    real = CoordinationAnalysis.modify

    def modify(self, data, ctx):
        if self.cutoff > limit:
            raise error
        return real(self, data, ctx)

    monkeypatch.setattr(CoordinationAnalysis, "modify", modify)


@pytest.mark.parametrize("error", [MemoryError("24 GiB"), KeyboardInterrupt(), ComputeCancelled("newer frame"),
                                   ValueError("bad input")], ids=lambda e: type(e).__name__)
def test_failed_verb_leaves_no_step_behind(monkeypatch, tmp_path, error):
    failing_above(monkeypatch, 10.0, error)
    pipe = load(CU)
    wrap = pipe.add(WrapAtoms())
    with pytest.raises(type(error)) as e:
        rdf(pipe, cutoff=40)
    assert e.value is error                        # the same exception, not a wrapped one
    assert list(pipe.modifiers) == [wrap] and pipe.modifiers[0] is wrap
    # the pipeline still works: before, each of these failed with the same error again
    assert pipe.compute(0).count == 108
    assert msd(pipe).name == "msd"
    export_file(pipe, tmp_path / "out.extxyz", frames=[0])
    assert (tmp_path / "out.extxyz").stat().st_size > 0


def test_failed_second_call_puts_the_earlier_step_back(monkeypatch):
    failing_above(monkeypatch, 10.0, MemoryError("24 GiB"))
    pipe = load(CU)
    wrap = pipe.add(WrapAtoms())
    g = rdf(pipe, cutoff=4)
    last = pipe.add(WrapAtoms())
    earlier = pipe.modifiers[1]
    assert isinstance(earlier, CoordinationAnalysis) and earlier.cutoff == 4

    with pytest.raises(MemoryError, match="24 GiB"):
        rdf(pipe, cutoff=40)
    assert len(pipe.modifiers) == 3
    assert pipe.modifiers[0] is wrap and pipe.modifiers[1] is earlier and pipe.modifiers[2] is last
    np.testing.assert_array_equal(pipe.compute(0).tables["coordination-rdf"]["g(r)"], g["g(r)"])

    again = rdf(pipe, cutoff=5)                    # and a call that works still replaces the step in place
    assert [type(m).__name__ for m in pipe.modifiers] == ["WrapAtoms", "CoordinationAnalysis", "WrapAtoms"]
    assert pipe.modifiers[1].cutoff == 5 and again.x.max() > g.x.max()


def test_verb_whose_step_cannot_run_on_the_data_is_rolled_back():
    pipe = load(CU)
    with pytest.raises(ValueError, match="needs bonds"):
        bond_angles(pipe)                          # no bonds: the step used to stay and break compute()
    assert list(pipe.modifiers) == []
    assert pipe.compute(0).count == 108


def three_runs() -> Runs:
    return Runs([Run(label, CU, load(CU), float(label)) for label in ("300", "600", "900")])


def test_verb_on_runs_is_all_or_nothing(monkeypatch):
    runs = three_runs()
    first = rdf(runs, cutoff=4)
    assert list(first) == ["300", "600", "900"]
    earlier = [r.pipe.modifiers[0] for r in runs]

    error = MemoryError("24 GiB")
    monkeypatch.setattr(runs["900"].pipe, "compute", lambda *a, **k: (_ for _ in ()).throw(error))
    with pytest.raises(MemoryError) as e:
        rdf(runs, cutoff=5)                        # the first two runs compute, the third fails
    assert e.value is error
    for run, step in zip(runs, earlier):           # was: cutoff 5 in two runs, cutoff 4 in the third
        assert list(run.pipe.modifiers) == [step] and run.pipe.modifiers[0] is step and step.cutoff == 4

    fresh = three_runs()
    monkeypatch.setattr(fresh["600"].pipe, "compute", lambda *a, **k: (_ for _ in ()).throw(error))
    with pytest.raises(MemoryError):
        rdf(fresh, cutoff=4)
    assert [list(r.pipe.modifiers) for r in fresh] == [[], [], []]


def test_vacf_on_runs_is_all_or_nothing(monkeypatch):
    runs = three_runs()                            # vacf has its own loop over the runs
    error = KeyboardInterrupt()
    monkeypatch.setattr(runs["600"].pipe, "compute", lambda *a, **k: (_ for _ in ()).throw(error))
    with pytest.raises(KeyboardInterrupt) as e:
        vacf(runs)
    assert e.value is error
    assert [list(r.pipe.modifiers) for r in runs] == [[], [], []]   # was: the step stayed in the first two runs


# ---------------------------------------------------------------- Table: a missing column
def test_missing_table_column_names_the_table_and_its_columns():
    g = rdf(load(CU), cutoff=4)
    with pytest.raises(KeyError) as e:
        g["g"]
    text = e.value.args[0]
    assert "'g'" in text and "'coordination-rdf'" in text and "r, g(r)" in text
    assert "Did you mean 'g(r)'?" in text
    with pytest.raises(KeyError, match=r"Did you mean 'g\(r\)'"):
        g["G(R)"]                                  # only the case differs
    with pytest.raises(KeyError) as e:
        g["energy"]
    assert "r, g(r)" in e.value.args[0] and "Did you mean" not in e.value.args[0]
    t = Table("thermo", {"Step": [0, 1], "Temp": [300.0, 301.0], "PotEng": [-1.0, -1.1]})
    with pytest.raises(KeyError, match="Did you mean 'PotEng'"):
        t["Poteng"]
    with pytest.raises(KeyError, match="Did you mean 'Temp'"):
        t["Tmep"]
    with pytest.raises(KeyError, match="no column 3 in table 'thermo'"):
        t[3]


def test_table_is_still_a_plain_dict_of_columns():
    t = Table("thermo", {"Step": [0, 1], "Temp": [300.0, 301.0]}, frame=2, labels=["a", "b"],
              attributes={"thermo.mean": 300.5})
    assert "g" not in t and t.get("g") is None and t.get("g", 1) == 1
    assert list(t) == ["Step", "Temp"] and t.pop("g", None) is None and t.setdefault("Temp", 0) is t["Temp"]
    for clone in (copy.copy(t), copy.deepcopy(t), pickle.loads(pickle.dumps(t))):
        assert type(clone) is Table and clone.name == "thermo" and clone.columns == ["Step", "Temp"]
        assert clone.frame == 2 and clone.labels == ["a", "b"] and clone.attributes == {"thermo.mean": 300.5}
        np.testing.assert_array_equal(clone["Temp"], t["Temp"])
        with pytest.raises(KeyError, match="no column 'g' in table 'thermo'"):
            clone["g"]
    assert dict(t).keys() == t.keys()
