"""Errors without a hint, from the API review (A7): a misspelt property, table, attribute or
column raised a bare ``KeyError: 'Coord'``, and expressions written the OVITO or C way (``&&``,
``||``, a single ``=``, ``a > 1 | b > 1`` without parentheses) failed with "invalid syntax" or
numpy's "ufunc 'bitwise_or' not supported". The reader of these messages is often a model
writing the next notebook cell, so each one names what is there and the corrected call."""
import copy
import copyreg
import json
import pickle
import re

import numpy as np
import pytest

import ewald
import ewald.examples
from ewald import DataCollection
from ewald.data import Attributes, Cell, Particles, Tables
from ewald.modifiers import CoordinationAnalysis, ExpressionSelection
from ewald.modifiers.expression import evaluate
from ewald.verbs import Table
from ewald.views import compare_data, figure, show, view_data


@pytest.fixture
def pipe():
    p = ewald.load(ewald.examples.path("Cu_md.extxyz"))
    p.add(CoordinationAnalysis(cutoff=2.7))   # at 2.7 A the hot frames have atoms with 9 to 12 neighbors
    return p


@pytest.fixture
def data(pipe):
    return pipe.compute(3)


# ---------------------------------------------------------------- lookups on a frame
@pytest.mark.parametrize("lookup", [lambda d: d["Coord"], lambda d: d.particles["Coord"]])
def test_missing_property_says_what_the_frame_has(data, lookup):
    with pytest.raises(KeyError) as err:
        lookup(data)
    message = err.value.args[0]
    assert message.startswith("no particle property 'Coord'; this frame has: Position, Particle Type, Force, "
                              "Coordination")
    assert message.endswith("Did you mean 'Coordination'?")


@pytest.mark.parametrize("name, meant", [("coordination", "Coordination"), ("ParticleType", "Particle Type"),
                                         ("Forces", "Force"), ("Pos", "Position")])
def test_missing_property_close_matches(data, name, meant):
    with pytest.raises(KeyError, match=re.escape(f"Did you mean {meant!r}?")):
        data[name]


def test_missing_property_without_a_close_match_points_at_steps(data):
    with pytest.raises(KeyError, match=r"no particle property 'Structure Type'; this frame has: .*pipe\.add\(") as err:
        data["Structure Type"]
    assert "Did you mean" not in err.value.args[0]


def test_expression_component_is_not_a_property(data):
    with pytest.raises(KeyError, match=re.escape("data['Position'][:, 2]")):
        data["Position.Z"]
    assert np.array_equal(data["Position"][:, 2], evaluate("Position.Z", data))


def test_missing_table_and_attribute(data):
    with pytest.raises(KeyError, match="no table 'rdf'; this frame has: coordination-rdf. "
                                       "Did you mean 'coordination-rdf'"):
        data.tables["rdf"]
    with pytest.raises(KeyError, match="no attribute 'Temp'; this frame has: Energy, Timestep"):
        data.attributes["Temp"]
    with pytest.raises(KeyError, match="no attribute 'energy'.*Did you mean 'Energy'"):
        data.attributes["energy"]
    empty = DataCollection(np.zeros((2, 3)))
    with pytest.raises(KeyError, match=r"no table 'msd'; this frame has: none\. Analysis steps write tables: pipe\.add"):
        empty.tables["msd"]
    with pytest.raises(KeyError, match="no attribute 'Energy'; this frame has: none"):
        empty.attributes["Energy"]


def test_long_lists_are_cut(data):
    d = data.copy()
    for k in range(30):
        d[f"Extra {k:02d}"] = np.zeros(d.count)
        d.attributes[f"Attr {k:02d}"] = k
    for lookup, total in ((lambda: d["Nope"], len(d.particles)), (lambda: d.attributes["Nope"], len(d.attributes))):
        with pytest.raises(KeyError) as err:
            lookup()
        message = err.value.args[0]
        assert f"... ({total} in all)" in message and "02" in message and "20" not in message
        assert len(message) < 400
    with pytest.raises(KeyError, match="Did you mean 'Extra 25'"):   # the close match may be past the cut
        d["Extra25"]


def test_containers_still_behave_as_dicts(data):
    for box, known in ((data.particles, "Position"), (data.tables, "coordination-rdf"), (data.attributes, "Energy")):
        assert isinstance(box, dict) and known in box and "Nope" not in box
        assert box.get("Nope") is None and box.get("Nope", 1) == 1
        before = list(box)
        with pytest.raises(KeyError):
            box["Nope"]
        assert list(box) == before                    # a failed lookup adds nothing (no defaultdict)
        assert box == dict(box) and repr(box) == repr(dict(box))
        with pytest.raises(KeyError):
            box.pop("Nope")
        assert box.setdefault(known, None) is box[known]
    scalars = {k: v for k, v in data.attributes.items() if isinstance(v, (int, float, str))}
    assert json.loads(json.dumps(Attributes(scalars))) == scalars


def test_every_way_to_get_a_frame_keeps_the_containers(data):
    fresh = DataCollection(np.zeros((3, 3)), ["Cu"] * 3)
    fresh.tables = {"t": {"x": np.arange(3.0)}}       # plain dicts are assigned all over the library
    fresh.attributes = dict(Energy=1.0)
    fresh.particles = dict(fresh.particles)
    frames = [fresh, data, data.copy(), data.take(np.arange(5)), data.delete(np.arange(data.count) < 5),
              data.copy().freeze(), copy.copy(data), copy.deepcopy(data), pickle.loads(pickle.dumps(data)),
              data.apply(ExpressionSelection("Position.Z > 5"))]
    for d in frames:
        assert (type(d.particles), type(d.tables), type(d.attributes)) == (Particles, Tables, Attributes)
        with pytest.raises(KeyError, match="no table 'nope'; this frame has"):
            d.tables["nope"]
        with pytest.raises(KeyError, match="no attribute 'nope'; this frame has"):
            d.attributes["nope"]
    assert fresh.tables == {"t": fresh.tables["t"]} and fresh.attributes == {"Energy": 1.0}
    assert np.array_equal(pickle.loads(pickle.dumps(data))["Coordination"], data["Coordination"])


def test_copy_shares_arrays_and_copies_containers(data):
    d = data.copy()
    d.tables["mine"] = Table("mine", {"x": np.arange(3.0)}, labels=["a", "b", "c"])
    c = d.copy()
    assert c["Position"] is d["Position"] and c.particles is not d.particles
    assert c.tables is not d.tables and c.attributes is not d.attributes
    assert type(c.tables["coordination-rdf"]) is type(d.tables["coordination-rdf"])
    assert c.tables["coordination-rdf"] is not d.tables["coordination-rdf"]     # columns may be replaced
    assert type(c.tables["mine"]) is Table and c.tables["mine"].labels == ["a", "b", "c"]
    c.attributes["new"] = 1
    c.tables["coordination-rdf"]["extra"] = np.zeros(1)
    assert "new" not in d.attributes and "extra" not in d.tables["coordination-rdf"]


class _PickledBefore:
    """Pickles as a DataCollection did before ``tables`` and ``attributes`` were properties: its
    state holds them as plain dicts under their plain names."""

    def __init__(self, state):
        self.state = state

    def __reduce__(self):
        return copyreg._reconstructor, (DataCollection, object, None), self.state


def test_frames_from_before_the_change_still_load():
    state = {"_particles": Particles({"Position": np.zeros((2, 3))}), "attributes": {"Energy": 1.0},
             "tables": {"t": {"x": np.arange(2.0)}}, "cell": Cell(np.eye(3))}     # no bonds, grids, ... either
    old = pickle.loads(pickle.dumps(_PickledBefore(state)))
    by_hand = DataCollection.__new__(DataCollection)                               # as DataCollection.copy builds one
    by_hand.particles, by_hand.attributes, by_hand.tables = state["_particles"], {"Energy": 1.0}, state["tables"]
    by_hand.cell = Cell(np.eye(3))
    for d in (old, by_hand, old.copy(), by_hand.copy(), copy.deepcopy(old)):
        assert type(d) is DataCollection and d.count == 2
        assert d.attributes == {"Energy": 1.0} and list(d.tables) == ["t"]
        assert (type(d.tables), type(d.attributes)) == (Tables, Attributes)
        assert "Energy=1.0" in d.summary()


def test_kernel_frames_round_trip(data, tmp_path):
    from ewald.kernel_bridge import load_frame, save_frame

    path = str(tmp_path / "frame.npz")
    save_frame(data, path)
    back = load_frame(path)
    assert np.array_equal(back.tables["coordination-rdf"]["g(r)"], data.tables["coordination-rdf"]["g(r)"])
    assert back.attributes["Energy"] == data.attributes["Energy"]
    assert (type(back.tables), type(back.attributes)) == (Tables, Attributes)
    with pytest.raises(KeyError, match="no table 'rdf'.*Did you mean 'coordination-rdf'"):
        back.tables["rdf"]


# ---------------------------------------------------------------- show(), figure(), view_data()
NO_COLUMN = r"no column 'g' in table 'coordination-rdf'; this table has: (r, )?g\(r\)\. Did you mean 'g\(r\)'\?"


def test_missing_column_names_the_table_and_its_columns(pipe):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()                        # other tests may have left figures open
    with pytest.raises(KeyError, match=NO_COLUMN):
        view_data(pipe, table="coordination-rdf", y="g")
    with pytest.raises(KeyError, match=NO_COLUMN):
        show(pipe, table="coordination-rdf", y="g")
    with pytest.raises(KeyError, match=NO_COLUMN):
        figure(pipe, table="coordination-rdf", y="g")
    with pytest.raises(KeyError, match=NO_COLUMN):
        view_data(pipe.compute(0), table="coordination-rdf", y="g")
    with pytest.raises(KeyError, match=NO_COLUMN):   # a verb's result is its own table
        view_data(Table("coordination-rdf", pipe.compute(0).tables["coordination-rdf"]), y="g")
    assert plt.get_fignums() == before                # the error comes before a figure is made
    with pytest.raises(KeyError, match=NO_COLUMN):   # one sentence, whichever way the column is asked for
        Table("coordination-rdf", pipe.compute(0).tables["coordination-rdf"])["g"]
    title, columns = view_data(pipe, table="coordination-rdf", y="g(r)")
    assert title == "coordination-rdf" and list(columns) == ["r", "g(r)"]


def test_missing_column_in_a_comparison(pipe):
    with pytest.raises(KeyError, match=NO_COLUMN):
        figure(pipe, table="coordination-rdf", y=["g(r)", "g"])
    with pytest.raises(KeyError, match=NO_COLUMN):
        compare_data(pipe, table="coordination-rdf", y="g", frame=[0, 1])
    partial = DataCollection(np.zeros((2, 3)))
    partial.tables["rdf"] = {"r": np.arange(3.0), "g(r)": np.ones(3), "g(r) Cu-Cu": np.ones(3), "g(r) Cu-Zr": np.ones(3)}
    title, x, series = compare_data(partial, table="rdf", y=["Total", "Cu-Zr"])   # legend labels still name columns
    assert (title, x, [s[3] for s in series]) == ("rdf", "r", ["g(r)", "g(r) Cu-Zr"])
    with pytest.raises(KeyError, match=r"no column 'g\(r\) Cu-Z' in table 'rdf'; this table has: g\(r\), g\(r\) Cu-Cu, "
                                       r"g\(r\) Cu-Zr\. Did you mean 'g\(r\) Cu-Zr'"):
        compare_data(partial, table="rdf", y=["Total", "g(r) Cu-Z"])


def test_missing_table_and_histogram_property_name_a_close_match(pipe):
    with pytest.raises(KeyError, match="no table 'coordination_rdf'; this frame has: coordination-rdf. "
                                       "Did you mean 'coordination-rdf'"):
        view_data(pipe, table="coordination_rdf")
    wrong = "no particle property 'Coordnation'; this frame has: Position, .*Did you mean 'Coordination'"
    with pytest.raises(KeyError, match=wrong):
        view_data(pipe, histogram="Coordnation")
    with pytest.raises(KeyError, match=wrong):
        show(pipe, histogram="Coordnation")
    with pytest.raises(KeyError, match=wrong):
        compare_data(pipe, histogram="Coordnation", frame=[0, 1])
    with pytest.raises(KeyError, match=r"no particle property 'Position\.Z'; this frame has"):   # no column advice here
        view_data(pipe, histogram="Position.Z")
    title, columns = view_data(pipe, histogram="Coordination", frame=3)
    assert title == "Coordination" and int(np.sum(columns["count"])) == 108


# ---------------------------------------------------------------- expressions
@pytest.fixture
def frame(data):
    d = data.copy()
    d["Selection"] = np.arange(d.count) % 3            # numbers, where OVITO takes any nonzero as true
    return d


def _columns(d):
    return d["Position"][:, 2], d["Position"][:, 0], d["Coordination"], d["Selection"]


def test_ovito_logical_operators(frame):
    z, x, c, s = _columns(frame)
    assert 0 < np.sum((z > 5) & (c < 12)) < np.sum(z > 5)      # the frame tells the operators apart
    cases = {
        "Position.Z > 5 && Coordination < 12": (z > 5) & (c < 12),
        "Position.Z > 5 || Coordination < 12": (z > 5) | (c < 12),
        "Position.Z>5&&Coordination<12": (z > 5) & (c < 12),
        # || binds loosest, then &&, both below comparisons (C and OVITO)
        "Position.Z > 5 || Coordination < 12 && Position.X < 3": (z > 5) | ((c < 12) & (x < 3)),
        "Position.Z > 5 && Coordination < 12 || Position.X < 3": ((z > 5) & (c < 12)) | (x < 3),
        "Position.X < 3 || Position.Z > 5 && Coordination < 12 || Position.X > 9":
            (x < 3) | ((z > 5) & (c < 12)) | (x > 9),
        "(Position.Z > 5 || Coordination < 12) && Position.X < 3": ((z > 5) | (c < 12)) & (x < 3),
        "Position.Z > 5 && Position.X < 3 && Coordination < 12": (z > 5) & (x < 3) & (c < 12),
        # nonzero numbers are true: 2 & True would be 0
        "Selection && Position.Z > 5": (s != 0) & (z > 5),
        "Selection || Position.Z > 5": (s != 0) | (z > 5),
        # with the other OVITO syntax, and inside brackets
        "Position.Z > 5 && Coordination < 12 ? 1 : 0": np.where((z > 5) & (c < 12), 1, 0),
        "np.where(Position.Z > 5 && Coordination < 12, 1, 0)": np.where((z > 5) & (c < 12), 1, 0),
        "Position.Z^2 > 25 && Position.X^2 < 9": (z ** 2 > 25) & (x ** 2 < 9),
        "(Position.Z > 5) & (Coordination < 12 || Position.X < 3)": (z > 5) & ((c < 12) | (x < 3)),
        'ParticleType == "Cu" && Position.Z > 5': z > 5,
        "Frame == 3 && N > 100": True,
    }
    for expression, want in cases.items():
        got = np.asarray(evaluate(expression, frame, 3))
        assert got.dtype == np.asarray(want).dtype and np.array_equal(got, want), expression


def test_logical_operators_select_in_a_pipeline(pipe):
    pipe.add(ExpressionSelection("Position.Z > 5 && Coordination < 12"))
    d = pipe.compute(3)
    assert np.array_equal(d.selection, (d["Position"][:, 2] > 5) & (d["Coordination"] < 12)) and d.selection.any()


@pytest.mark.parametrize("expression", ["Position.Z > 5 &&", "|| Position.Z > 5", "Position.Z > 5 && || N > 1"])
def test_logical_operator_without_a_side(frame, expression):
    with pytest.raises(SyntaxError, match="need a condition on both sides"):
        evaluate(expression, frame)


def test_logical_operators_beside_python_syntax(frame):
    z, _, c, _ = _columns(frame)
    # a keyword argument's condition starts after its name
    assert evaluate("np.sum(Position.Z, where=Position.Z > 5 && Coordination < 12)", frame) \
        == np.sum(z, where=(z > 5) & (c < 12)) != np.sum(z)
    # Python's and/or/if bind looser than && and || would: grouped by the writer, never guessed
    assert evaluate("(Frame > 1 && N > 5) or Frame == 0", frame, 3)
    assert evaluate("1 if (Frame > 1 && N > 500) else 2", frame, 3) == 2
    for expression in ("Frame > 5 && N > 5 or Frame == 3", "1 if Frame > 1 && N > 5 else 0"):
        with pytest.raises(SyntaxError, match="beside 'and', 'or' or 'if': put each part in parentheses"):
            evaluate(expression, frame, 3)
    assert evaluate("Frame > 5 && N > 5 || Frame == 3", frame, 3)        # what the first one means in OVITO
    assert np.sum(evaluate('ParticleType == "or" || Position.Z > 5', frame)) == np.sum(z > 5)   # a word in a string


def test_string_literals_are_left_alone():
    d = DataCollection(np.arange(18.0).reshape(6, 3), ["a=b", "a&&b", "x||y", "a=b", "Cu", "?:"])
    count = lambda expression: int(np.sum(evaluate(expression, d)))   # noqa: E731
    assert count('ParticleType == "a=b"') == 2
    assert count("ParticleType == 'a&&b'") == 1 and count('ParticleType == "x||y"') == 1
    assert count('ParticleType != "a=b"') == 4
    assert count('ParticleType == "a&&b" || ParticleType == "a=b"') == 3
    assert count('ParticleType == "a=b" && Position.X > 0') == 1
    assert count('(ParticleType == "a&&b") | (ParticleType == "x||y")') == 2
    assert count('np.isin(ParticleType, ["a=b", "a&&b", "x||y"])') == 4
    with pytest.raises(ValueError, match=re.escape("""use '==': 'ParticleType == "a=b" && Position.X = 0'""".replace(
            "X = 0", "X == 0"))):
        evaluate('ParticleType == "a=b" && Position.X = 0', d)


def test_single_equals_names_the_comparison(frame):
    with pytest.raises(ValueError, match="'Coordination = 12': a single '=' does not compare; use '==': "
                                         "'Coordination == 12'"):
        evaluate("Coordination = 12", frame)
    with pytest.raises(ValueError, match=re.escape("use '==': '(Coordination == 12) & (Position.Z > 5)'")):
        evaluate("(Coordination = 12) & (Position.Z > 5)", frame)
    with pytest.raises(ValueError, match="use '==': 'Coordination == 12 \\? 1 : 0'"):
        evaluate("Coordination = 12 ? 1 : 0", frame)
    # the correction is only shown when it runs: here the rule, not a guess
    for expression in ("Coordinaton = 12", "np.clip(Position.Z, a_min=0, a_max=5) = 3"):
        with pytest.raises(ValueError, match="a single '=' does not compare; use '==', as in") as err:
            evaluate(expression, frame)
        assert "a_min==" not in str(err.value) and "Coordinaton ==" not in str(err.value)


def test_other_operators_with_equals_are_not_guessed(frame):
    z, _, c, _ = _columns(frame)
    for expression, want in {"Coordination <= 11": c <= 11, "Coordination >= 12": c >= 12, "Coordination != 12": c != 12,
                             "Coordination == 12": c == 12, "Coordination<=11": c <= 11,
                             "np.clip(Position.Z, a_min=0, a_max=5) >= 5": z >= 5}.items():
        assert np.array_equal(evaluate(expression, frame), want), expression
    for expression in ("Coordination => 12", "Coordination =< 12", "np.clip(Position.Z, a_min=0, a_max=5) +",
                       "Coordination === 12"):
        with pytest.raises(SyntaxError, match="invalid expression"):
            evaluate(expression, frame)


@pytest.mark.parametrize("expression, fixed", [
    ("Position.Z > 1 | Position.X > 1", "(Position.Z > 1) | (Position.X > 1)"),
    ("Position.Z > 5 & Position.X > 1", "(Position.Z > 5) & (Position.X > 1)"),
    ("Coordination == 12 | Coordination == 11", "(Coordination == 12) | (Coordination == 11)"),   # whole numbers
    ("Position.Z > 5 | Position.X > 5 & Coordination == 12",
     "(Position.Z > 5) | (Position.X > 5) & (Coordination == 12)"),
    ("(Position.Z > 5 | Position.X > 5) & Selection", "((Position.Z > 5) | (Position.X > 5)) & Selection"),
    ("np.where(Position.Z > 5 | Position.X > 5, 1, 0)", "np.where((Position.Z > 5) | (Position.X > 5), 1, 0)"),
    ("Position.Z > 5 | Position.X > 5 ? 1 : 0", "(Position.Z > 5) | (Position.X > 5) ? 1 : 0"),
    ('ParticleType == "Cu" | Position.Z > 5', '(ParticleType == "Cu") | (Position.Z > 5)'),
    ("Position.Z > 5 && Position.X > 5 | Coordination > 11",
     "Position.Z > 5 && (Position.X > 5) | (Coordination > 11)"),
])
def test_missing_parentheses_show_the_expression_with_them(frame, expression, fixed):
    with pytest.raises(ValueError, match="bind tighter than comparisons") as err:
        evaluate(expression, frame)
    assert str(err.value).startswith(repr(expression)) and str(err.value).endswith(f"parentheses: {fixed!r}")
    np.asarray(evaluate(fixed, frame))                 # what the message says to write runs


def test_missing_parentheses_mean_what_c_means(frame):
    z, x, c, _ = _columns(frame)
    assert np.array_equal(evaluate("(Position.Z > 5) | (Position.X > 5) & (Coordination == 12)", frame),
                          (z > 5) | ((x > 5) & (c == 12)))


def test_bitwise_on_decimals_states_the_rule(frame):
    with pytest.raises(ValueError, match=r"'Position.Z \| 1': a side of '&' or '\|' is neither a condition nor a "
                                         r"whole number.*bind tighter.*\(Position.Z > 5\) & \(Coordination < 12\)"):
        evaluate("Position.Z | 1", frame)


def test_other_errors_keep_their_message(frame):
    with pytest.raises(ValueError, match="instead of 'and', 'or', 'not'"):
        evaluate("Position.Z > 5 and Coordination < 12", frame)
    with pytest.raises(NameError, match="'Bogus' is not defined.*Available: .*Coordination"):
        evaluate("Position.Z > 1 | Bogus > 1", frame)
    with pytest.raises(TypeError, match="ufunc 'add'"):
        evaluate("Position.Z + 'a'", frame)
    with pytest.raises(ValueError, match="not allowed"):
        evaluate("Position.Z > 1 && ().__class__", frame)


def test_expressions_that_worked_give_the_same_result(frame):
    z, x, c, s = _columns(frame)
    cases = {
        "(Position.Z > 5) & (Coordination < 12)": (z > 5) & (c < 12),
        "(Position.Z > 5) | (Coordination < 12)": (z > 5) | (c < 12),
        "~(Position.Z > 5)": ~(z > 5),
        "Selection & 1": s & 1,                       # bitwise on whole numbers stays bitwise
        "Selection | 4": s | 4,
        "Selection & Coordination > 11": (s & c) > 11,   # Python's reading, right or not: no silent change
        "Position.Z > 5 ? Coordination : 0": np.where(z > 5, c, 0),
        "Position.Z^2 + Position.X^2 < 30": z ** 2 + x ** 2 < 30,
        "np.clip(Position.Z, a_min=1, a_max=5)": np.clip(z, 1, 5),
        "(Coordination == 12) == (Position.Z > 5)": (c == 12) == (z > 5),
        "Coordination << 1": c << 1,
        "Coordination // 2 >= 5": c // 2 >= 5,
        "Position[:, 2] > 5": z > 5,
        "abs(Position.Z - 5) <= 1": np.abs(z - 5) <= 1,
    }
    for expression, want in cases.items():
        got = np.asarray(evaluate(expression, frame, 3))
        assert got.dtype == np.asarray(want).dtype and np.array_equal(got, want), expression
