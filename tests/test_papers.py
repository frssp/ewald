"""Published paper datasets: registry, checksummed download, CLI. No network access needed."""
import hashlib
import re
from pathlib import Path

import pytest

from ewald import examples
from ewald.__main__ import main
from ewald.examples import papers

ROOT = Path(__file__).resolve().parents[1]


def test_registry_is_pinned_and_complete():
    assert len({p.name for p in papers.PAPERS}) == len(papers.PAPERS)
    for p in papers.PAPERS:
        assert p.files and p.license and p.doi and p.citation
        assert (ROOT / "examples" / "papers" / p.script).is_file(), p.script
        assert len({f.name for f in p.files}) == len(p.files)
        for f in p.files:
            # a commit, not a branch, so the bytes never change under the checksum
            assert re.match(r"https://raw\.githubusercontent\.com/[^/]+/[^/]+/[0-9a-f]{40}/", f.url), f.url
            assert re.fullmatch(r"[0-9a-f]{64}", f.sha256) and f.size > 0
    with pytest.raises(KeyError, match="li6ps5cl-aimd"):
        papers.get("nope")


def test_every_dataset_is_documented():
    page = (ROOT / "docs" / "user" / "papers.md").read_text(encoding="utf-8")
    readme = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    for p in papers.PAPERS:
        assert p.name in page and p.script in page and p.script in readme


def _local_paper(tmp_path, content=b"frame 1\nframe 2\n"):
    src = tmp_path / "upstream.xyz"
    src.write_bytes(content)
    f = papers.RemoteFile("data.xyz", src.as_uri(), hashlib.sha256(content).hexdigest(), len(content))
    return papers.Paper("local", "t", "c", "doi", "src", "MIT", "d", "r", "s.py", (f,))


def test_fetch_downloads_once_and_verifies(tmp_path, monkeypatch):
    paper = _local_paper(tmp_path)
    monkeypatch.setattr(papers, "PAPERS", [paper])
    calls = []
    real = papers._download
    monkeypatch.setattr(papers, "_download", lambda url, dest: (calls.append(url), real(url, dest)))
    folder = examples.fetch("local", tmp_path / "cache", quiet=True)
    assert (folder / "data.xyz").read_bytes() == b"frame 1\nframe 2\n"
    examples.fetch("local", tmp_path / "cache", quiet=True)
    assert len(calls) == 1                       # cached copy with the right checksum is kept
    (folder / "data.xyz").write_bytes(b"edited")
    examples.fetch("local", tmp_path / "cache", quiet=True)
    assert len(calls) == 2 and (folder / "data.xyz").read_bytes().startswith(b"frame")


def test_fetch_rejects_a_changed_upstream_file(tmp_path, monkeypatch):
    paper = _local_paper(tmp_path)
    (tmp_path / "upstream.xyz").write_bytes(b"something else")
    monkeypatch.setattr(papers, "PAPERS", [paper])
    with pytest.raises(ValueError, match="checksum"):
        examples.fetch("local", tmp_path / "cache", quiet=True)
    assert not (tmp_path / "cache" / "data.xyz").exists()


def test_default_folder_follows_the_cache_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("EWALD_CACHE_DIR", str(tmp_path))
    assert papers.data_dir() == tmp_path / "papers"


def test_papers_cli(capsys):
    main(["examples", "--papers"])
    out = capsys.readouterr().out
    assert all(p.name in out and p.script in out for p in papers.PAPERS)
