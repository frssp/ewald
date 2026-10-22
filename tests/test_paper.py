"""The JOSS paper in paper/: required sections, length, citations and figures."""
import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"
pytestmark = pytest.mark.skipif(not (PAPER / "paper.md").is_file(), reason="no paper/ (sdist)")

SECTIONS = ["Summary", "Statement of need", "State of the field", "Software design",
            "Research impact statement", "AI usage disclosure", "Acknowledgements", "References"]


def _body():
    text = (PAPER / "paper.md").read_text(encoding="utf-8")
    return text.split("---", 2)[2]


def test_paper_has_the_joss_sections_in_order():
    headings = re.findall(r"^# (.+)$", _body(), re.M)
    assert headings == SECTIONS


def test_paper_length_is_within_joss_limits():
    words = len(_body().split())
    assert 750 <= words <= 1750, words


def test_every_citation_is_in_the_bibliography():
    bib = (PAPER / "paper.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
    cited = set(re.findall(r"@([A-Za-z][\w:-]*\d{4})", _body()))
    assert cited, "no citations found"
    assert cited <= keys, cited - keys


def test_figures_exist():
    for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", _body()):
        assert (PAPER / path).is_file(), path
