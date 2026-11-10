"""search_api finds a step by the words people use for it, from the API review (A8): the search
only matched an entry's own words, so "supercell" missed ``Replicate`` ("Tile the periodic cell"),
"vacancy" found nothing, "write POSCAR" ranked ``export_file`` behind the introduction and a Korean
query returned an empty list. Entries now carry search terms (``SYNONYMS``), queries are read
through a glossary (Korean, notation such as g(r), British spellings), and plurals stem to the
singular. None of it is printed: ``llms.txt`` stays what ``ewald api`` writes."""
from pathlib import Path

import pytest

from ewald import modifiers, plugins
from ewald.llm import Toolset
from ewald.llm import reference
from ewald.llm.reference import GLOSSARY, SYNONYMS, _entry, _expand, _terms, search_reference


@pytest.fixture(autouse=True)
def no_plugins(monkeypatch):
    """The core reference only, whatever plugins this environment has installed."""
    monkeypatch.setattr(plugins, "modifiers", lambda: {})
    monkeypatch.setattr(plugins, "functions", lambda: {})


def found(query: str, limit: int = 3, **kwargs) -> list[str]:
    """Names of the entries (modifier, function or recipe title; '' for guide text) a query finds."""
    return [_entry(hit["text"]) for hit in search_reference(query, limit, **kwargs)]


# ------------------------------------------------------------------ the queries of the review

def test_the_queries_of_the_review():
    assert found("supercell")[0] == "Replicate"                   # was: only PhononMode (its `supercell=`)
    assert found("write POSCAR")[0] == "export_file"              # was: second, behind the introduction
    assert set(found("배위수 분포")[:2]) == {"CoordinationAnalysis", "rdf"}   # was: []
    assert search_reference("zzzz qqqq") == [] and search_reference("안녕하세요") == []   # still nothing to find


@pytest.mark.parametrize("query, first", [
    # a format named with a verb of reading is a question about reading
    ("read cif file", "import_file"), ("open a LAMMPS dump file", "import_file"), ("import POSCAR", "load"),
    ("load a cif", "load"), ("read lammps data file", "import_file"), ("cif 파일 읽어", "import_file"),
    ("read a POSCAR and compute rdf", "rdf"),
    ("write POSCAR", "export_file"), ("convert the trajectory to CIF", "export_file"), ("export xyz", "export_file"),
    ("save the last frame as a LAMMPS data file", "export_file"), ("구조를 POSCAR로 저장", "export_file"),
    # "first shell" asks for a coordination number unless a selection is asked for
    ("first shell coordination number of Li", "CoordinationAnalysis"),
    ("coordination number in the first shell", "CoordinationAnalysis"),
    ("select the first coordination shell", "ExpandSelection"),
    ("select neighbors of the selected atoms", "ExpandSelection"),
    # 평균 is "average", not the "mean" of MeanSquaredDisplacement
    ("프레임 평균", "TimeAveraging"), ("평균 구조", "TimeAveraging"), ("평균제곱변위", "MeanSquaredDisplacement"),
    ("평균 배위수", "CoordinationAnalysis"),
])
def test_a_synonym_does_not_claim_the_neighbouring_question(query, first):
    assert found(query)[0] == first


def test_nothing_claims_the_primitive_cell():
    """No step returns a primitive or standardized cell (SymmetryAnalysis names the space group and
    the Wyckoff positions only), so no synonym may send a model to one for it."""
    claimed = [(name, p) for name, phrases in SYNONYMS.items() for p in phrases
               if {"primitive", "standardize", "standardized", "niggli"} & set(_terms(p))]
    assert not claimed
    assert "SymmetryAnalysis" not in found("primitive cell")


#: What a researcher types -> the entry that does it, and how far down it may be (two entries are
#: equally right for a verb and its modifier). None of these found its entry there before the review.
RESEARCHER = [
    ("repeat the cell 3 times along z", "Replicate", 1),
    ("show periodic images of the structure", "Replicate", 1),
    ("g(r) of the liquid", "CoordinationAnalysis", 2),
    ("pair distribution function", "rdf", 2),
    ("PDF analysis of a glass", "CoordinationAnalysis", 2),
    ("number of neighbors of each atom", "CoordinationAnalysis", 2),
    ("Li diffusivity", "msd", 3),
    ("tracer diffusivity", "MeanSquaredDisplacement", 3),
    ("count vacancies and interstitials", "WignerSeitzAnalysis", 1),
    ("point defects after a cascade", "WignerSeitzAnalysis", 1),
    ("antisite defects", "WignerSeitzAnalysis", 1),
    ("Frenkel pairs", "WignerSeitzAnalysis", 1),
    ("nanocrystalline sample", "GrainSegmentation", 1),
    ("line defects in a crystal", "DislocationAnalysis", 1),
    ("twinning", "IdentifyFCCPlanarFaults", 1),
    ("electron localization function", "CreateIsosurface", 1),
    ("charge transfer at the interface", "GridDifference", 1),
    ("work function from the planar average", "Charge density: isosurface and a lattice-plane slice", 1),
    ("convert the trajectory to CIF", "export_file", 1),
    ("save the last frame as a LAMMPS data file", "export_file", 1),
    ("read an OUTCAR", "load", 1),
    ("merge two structures", "CombineDatasets", 1),
    ("density profile along z", "SpatialBinning", 1),
    ("energy vs time", "TimeSeries", 1),
    ("crop the structure", "Slice", 1),
    ("rotate the cell", "AffineTransformation", 1),
    ("crystallinity of the sample", "CommonNeighborAnalysis", 4),
    ("identify fcc and hcp atoms", "cna", 3),
    ("bond orientational order", "SteinhardtOrderParameters", 1),
    ("diffractogram", "PowderDiffraction", 2),
    ("lattice vibration eigenvector", "PhononMode", 1),
    ("dwell time of Li", "SiteHopping", 1),
    ("convergence of the relaxation", "check_run", 1),
    ("MD time step of the run", "run_info", 1),
    ("make the atoms transparent", "fade", 1),
    ("highlight the selected atoms", "paint", 1),
    ("save a picture", "snapshot", 1),
    ("screenshot of the structure", "snapshot", 1),
    ("colour atoms by a property", "ColorCoding", 1),
    # the same, asked in Korean (particles attached, compounds spaced or not)
    ("배위수의 분포를 계산", "CoordinationAnalysis", 2),
    ("동경분포함수", "rdf", 2),
    ("확산계수를 구해줘", "MeanSquaredDisplacement", 2),
    ("평균 제곱 변위", "MeanSquaredDisplacement", 1),
    ("결합각 분포", "BondAngleDistribution", 2),
    ("결합 길이 분포", "BondLengthDistribution", 1),
    ("슈퍼셀 만들기", "Replicate", 1),
    ("초격자", "Replicate", 1),
    ("공공과 침입형 원자 개수", "WignerSeitzAnalysis", 1),
    ("전위 분석", "DislocationAnalysis", 1),
    ("결정립 크기", "GrainSegmentation", 1),
    ("변형률 계산", "AtomicStrain", 2),
    ("전하밀도 등위면", "CreateIsosurface", 1),
    ("전하밀도 차이", "GridDifference", 1),
    ("궤적을 그려줘", "GenerateTrajectoryLines", 1),
    ("속도자기상관함수", "VelocityAutocorrelation", 1),
    ("구조인자", "StructureFactor", 1),
    ("XRD 회절 패턴", "PowderDiffraction", 2),
    ("공간군 찾기", "SymmetryAnalysis", 1),
    ("다면체 표시", "CoordinationPolyhedra", 1),
    ("클러스터 분석", "ClusterAnalysis", 2),
    ("원자 색칠", "paint", 1),
    ("POSCAR로 내보내기", "export_file", 1),
    ("구조를 저장", "export_file", 1),
    ("원자 숨기기", "hide", 1),
]


@pytest.mark.parametrize("query, entry, within", RESEARCHER, ids=[q for q, _, _ in RESEARCHER])
def test_a_researchers_words_find_the_entry(query, entry, within):
    assert entry in found(query, within), found(query, 5)


#: Queries in an entry's own words, with the entry the search ranked first before the review: the
#: synonyms and the stemming must not push it down (these pass without the review's changes too).
OWN_WORDS = [
    ("common neighbor analysis", "CommonNeighborAnalysis"),
    ("radial distribution function", "CoordinationAnalysis"),
    ("mean squared displacement", "MeanSquaredDisplacement"),
    ("create bonds between atoms", "CreateBonds"),
    ("select atoms by expression", "ExpressionSelection"),
    ("delete selected particles", "DeleteSelected"),
    ("space group and Wyckoff positions", "SymmetryAnalysis"),
    ("ring statistics of an amorphous network", "FindRings"),
    ("isosurface of a charge density grid", "CreateIsosurface"),
    ("Li site occupancy, hops between sites and residence times", "SiteHopping"),
    ("bond angle distribution", "BondAngleDistribution"),
    ("bond length histogram", "BondLengthDistribution"),
    ("velocity autocorrelation", "VelocityAutocorrelation"),
    ("structure factor S(q)", "StructureFactor"),
    ("powder diffraction pattern", "PowderDiffraction"),
    ("ionic conductivity", "IonicConductivity"),
    ("unwrap trajectories", "UnwrapTrajectories"),
    ("time average of rdf", "TimeAveraging"),
    ("atomic strain", "AtomicStrain"),
    ("centrosymmetry parameter", "CentroSymmetry"),
    ("polyhedral template matching", "PolyhedralTemplateMatching"),
    ("cluster analysis", "ClusterAnalysis"),
    ("histogram of a property", "Histogram"),
    ("NEB barrier", "NebEnergyProfile"),
    ("phonon mode animation", "PhononMode"),
    ("load several runs", "load_runs"),
    ("activation energy from temperatures", "arrhenius"),
    ("hide particles", "hide"),
    ("slice of atoms in a slab", "Slice"),
    ("wrap atoms into the cell", "WrapAtoms"),
    ("van Hove correlation", "VanHoveCorrelation"),
    ("non-Gaussian parameter", "NonGaussianParameter"),
    ("grain segmentation", "GrainSegmentation"),
    ("dislocation extraction", "DislocationAnalysis"),
    ("diamond identification", "IdentifyDiamond"),
    ("Steinhardt order parameters", "SteinhardtOrderParameters"),
    ("spatial binning along z", "SpatialBinning"),
    ("trajectory lines", "GenerateTrajectoryLines"),
    ("freeze property", "FreezeProperty"),
    ("export frames to a file", "export_file"),
    ("check run for problems", "check_run"),
    ("charge density difference", "GridDifference"),
]


@pytest.mark.parametrize("query, entry", OWN_WORDS, ids=[q for q, _ in OWN_WORDS])
def test_an_entrys_own_words_still_rank_it_first(query, entry):
    assert found(query, 1) == [entry], found(query, 3)


# ------------------------------------------------------------------ the synonyms table

def entries() -> set[str]:
    recipes = {title for title, _, _ in reference.RECIPES}
    functions = {name for _, names in reference.FUNCTIONS for name in names}
    return (set(modifiers.__all__) - {"Modifier", "PythonModifier"}) | functions | recipes


def test_every_synonym_key_is_an_entry_of_the_reference():
    """A renamed modifier, function or recipe must take its search terms along."""
    assert not set(SYNONYMS) - entries()
    indexed = {_entry(text) for _, text in reference.reference_chunks()}
    assert not set(SYNONYMS) - indexed            # and the search can tell which piece the key means
    verbs = {"rdf": "CoordinationAnalysis", "msd": "MeanSquaredDisplacement", "cna": "CommonNeighborAnalysis"}
    assert all(SYNONYMS[verb] is SYNONYMS[step] for verb, step in verbs.items())   # a verb shares its step's words


def test_every_synonym_finds_its_entry(monkeypatch):
    """Each phrase, searched alone, brings up every entry that lists it: none is shadowed by
    another entry, unreadable to the tokenizer, or rewritten by the glossary."""
    chunks = reference.reference_chunks()
    monkeypatch.setattr(reference, "reference_chunks", lambda with_plugins=False: chunks)   # ~600 searches
    listed: dict[str, list[str]] = {}
    for name, phrases in SYNONYMS.items():
        assert len(set(phrases)) == len(phrases), name
        for phrase in phrases:
            assert phrase == phrase.lower() and _terms(phrase) and _expand(phrase) == phrase, (name, phrase)
            listed.setdefault(phrase, []).append(name)
    lost = {phrase: names for phrase, names in listed.items()
            if not set(names) <= set(found(phrase, len(names) + 2))}
    assert not lost


def test_a_phrase_needs_all_of_its_words():
    """"pair distribution" is CoordinationAnalysis; "distribution" alone is not, or every
    distribution would be the RDF."""
    assert found("bond length distribution", 1) == ["BondLengthDistribution"]
    assert "CoordinationAnalysis" not in found("ring size distribution", 2)
    assert found("shear strain", 1) == ["AtomicStrain"]          # "shear cell" is AffineTransformation
    assert found("shear the cell", 1) == ["AffineTransformation"]
    assert set(found("PDF", 2)) == {"CoordinationAnalysis", "rdf"}     # the pair distribution function,
    assert found("save figure as pdf", 1) == ["show"]                  # unless a figure is saved as one


def test_synonyms_are_not_printed(monkeypatch):
    """The terms go into the index only: the entry reads as before, and llms.txt is ``ewald api``."""
    replicate = next(text for _, text in reference.reference_chunks() if _entry(text) == "Replicate")
    assert "supercell" not in replicate and "Tile the periodic cell" in replicate
    assert "supercell" not in search_reference("supercell", 1)[0]["text"]
    text = (Path(__file__).parents[1] / "llms.txt").read_text(encoding="utf-8")
    assert text == reference.api_reference(), "llms.txt is stale; regenerate with: python -m ewald api > llms.txt"


# ------------------------------------------------------------------ the glossary

def test_korean_terms_are_read_with_particles_and_spaces():
    assert _terms("배위수 분포") == []                             # the tokenizer reads Latin words only
    assert _terms(_expand("배위수의 분포를")) == ["coordination", "number", "distribution"]
    assert _terms(_expand("확산계수를 계산")) == ["diffusion", "coefficient"]      # not "diffusion" twice (확산)
    assert _terms(_expand("평균 제곱 변위")) == _terms(_expand("평균제곱변위")) == ["mean", "squared", "displacement"]
    assert _terms(_expand("정전위")) == ["electrostatic", "potential"]             # not 전위, a dislocation
    assert _terms(_expand("전위 밀도")) == ["dislocation", "density"]
    assert _terms(_expand("RDF 검색")) == ["rdf"] and _terms(_expand("색으로")) == ["color"]   # 검색 is not 색
    assert _terms(_expand("A 그리고 B")) == ["a", "b"] and _terms(_expand("그려줘")) == ["draw"]
    assert _expand("mean squared displacement") == "mean squared displacement"       # English passes through


def test_notation_and_spellings_are_read():
    assert _terms("g(r)") == []                                   # single letters are no words
    assert _terms(_expand("G(r)")) == _terms(_expand("g (r)")) == ["radial", "distribution", "function"]
    assert _terms(_expand("S(q) of a glass"))[:2] == ["structure", "factor"]
    assert _terms(_expand("neighbour colours")) == ["neighbor", "color"]
    assert _terms(_expand("freq6")) == ["freq", "6"] and "steinhardt" in _terms(_expand("Q6"))
    assert found("density of states", 1) == [""] and "DOS" in search_reference("density of states", 1)[0]["text"]


def test_every_glossary_word_is_in_the_index():
    """A glossary entry translates into words the reference (or a synonym) has, else it finds nothing."""
    index = {t for _, text in reference.reference_chunks() for t in _terms(text)}
    index |= {t for phrases in SYNONYMS.values() for p in phrases for t in _terms(p)}
    assert not {term: words for term, words in GLOSSARY.items() if not set(_terms(words)) <= index}
    assert all(term == term.lower() for term in GLOSSARY)         # matched in lower case


# ------------------------------------------------------------------ stemming

def test_plural_and_ing_forms_share_a_term():
    assert _terms("hops Hopping") == ["hop", "hop"]
    assert _terms("sites types particles angles frames images") == _terms("site type particle angle frame image")
    assert _terms("trajectories vacancies properties") == ["trajectory", "vacancy", "property"]
    assert _terms("boxes classes meshes") == ["box", "class", "mesh"]
    assert _terms("analysis radius class filling adding") == ["analysis", "radius", "class", "fill", "add"]
    assert _terms("UnwrapTrajectories bond_angles") == ["unwrap", "trajectory", "bond", "angle"]


# ------------------------------------------------------------------ plugins and the tool

def test_search_with_plugins_when_none_is_installed():
    for query in ("supercell", "슈퍼셀", "ring statistics"):
        assert search_reference(query, 4, with_plugins=True) == search_reference(query, 4)
    assert search_reference("zzzz qqqq", with_plugins=True) == []


def test_plugin_entries_are_found_beside_the_synonyms(monkeypatch):
    class LocalComposition(modifiers.Modifier):
        """Warren-Cowley short-range order parameters of an alloy."""

        def __init__(self, cutoff: float = 3.0):
            self.cutoff = cutoff

    def stacking_energy(pipe, plane=(1, 1, 1)):
        """Generalized stacking fault energy curve of a slab."""

    stacking_energy.__module__ = "ewald_gsf"
    monkeypatch.setattr(plugins, "modifiers", lambda: {
        "LocalComposition": plugins.ModifierPlugin("LocalComposition", LocalComposition, origin="local_composition.py")})
    monkeypatch.setattr(plugins, "functions", lambda: {
        "stacking_energy": plugins.FunctionPlugin("stacking_energy", stacking_energy, "ewald_gsf", "ewald-gsf")})
    assert found("Warren-Cowley short-range order", 1, with_plugins=True) == ["LocalComposition"]
    assert found("generalized stacking fault energy", 1, with_plugins=True) == ["stacking_energy"]
    assert found("합금의 단거리 규칙 Warren-Cowley", 1, with_plugins=True) == ["LocalComposition"]
    assert "LocalComposition" not in found("Warren-Cowley short-range order", 3)     # only when asked for
    assert found("supercell", 1, with_plugins=True) == ["Replicate"]                 # core synonyms as before
    assert found("stacking fault", 1, with_plugins=True) == ["IdentifyFCCPlanarFaults"]


def test_the_tool_and_the_question_hints_read_korean():
    from ewald.llm.notebook_prompt import related_part

    out = Toolset().call("search_api", {"query": "슈퍼셀"}).content
    assert out["results"][0]["text"].startswith("- `Replicate(") and "note" not in out
    assert "nothing matched" in Toolset().call("search_api", {"query": "안녕하세요"}).content["note"]
    part = related_part("Li의 확산계수를 구하고 싶어")
    assert "- `MeanSquaredDisplacement(" in part and part.count("\n- `") <= 4
    assert related_part("안녕하세요") == "" and related_part("그리고 다시 한 번") == ""
