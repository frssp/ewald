"""ewald:// links (ewald.deeplink): what a web page may ask EWALD to open, and the OS registration."""
import pytest

from ewald import deeplink
from ewald.deeplink import DeepLink, DeepLinkError

SCHEMES = ("fakeloc", "aiida")


def test_parse_reads_location_and_title():
    link = deeplink.parse("ewald://open?location=fakeloc%3A1234%2Fscf&title=PK%201234%20%C2%B7%20Si", SCHEMES)
    assert link == DeepLink("fakeloc:1234/scf", "PK 1234 · Si")
    assert deeplink.parse(link.url(), SCHEMES) == link        # url() round-trips


@pytest.mark.parametrize("url", [
    "ewald:open?location=fakeloc:7",            # some browsers drop the //
    "ewald://open/?location=fakeloc:7",         # Chrome on Windows may add a /
    " EWALD://OPEN?location=fakeloc:7 ",        # scheme and action ignore case, blanks are trimmed
    "ewald://open?location=FakeLoc:7",
])
def test_parse_accepts_browser_variants(url):
    assert deeplink.parse(url, SCHEMES).location.lower() == "fakeloc:7"


@pytest.mark.parametrize("url, why", [
    ("ewald://open?location=/etc/passwd", "installed reader"),             # local paths: never
    ("ewald://open?location=C%3A%5Cdata%5Cx.ewd", "installed reader"),     # a Windows drive is not a scheme
    ("ewald://open?location=host%3A%2Fhome%2Fu%2Fx", "installed reader"),  # SSH host:/path
    ("ewald://open?location=ssh%3A%2F%2Fevil%2Fx", "installed reader"),
    ("ewald://open?location=notebook.py", "installed reader"),
    ("ewald://run?location=fakeloc:1", "Unknown link action"),
    ("ewald://open?location=fakeloc:1&location=fakeloc:2", "exactly one"),
    ("ewald://open?title=x", "exactly one"),
    ("ewald://open?location=fakeloc:1&code=print(1)", "Unknown link field"),
    ("ewald://open?location=fakeloc%3A1%0A2", "control characters"),
    ("ewald://open?location=fakeloc:" + "1" * 3000, "longer than"),
    ("https://example.org/?location=fakeloc:1", "Not an ewald"),
])
def test_parse_refuses(url, why):
    with pytest.raises(DeepLinkError, match=why):
        deeplink.parse(url, SCHEMES)


def test_title_is_plain_short_text():
    link = deeplink.parse("ewald://open?location=fakeloc:1&title=" + "a%09b%1B" + "x" * 500, SCHEMES)
    assert link.title.startswith("a b") and "\x1b" not in link.title
    assert len(link.title) == deeplink.MAX_TITLE


def test_default_schemes_are_the_installed_readers(monkeypatch):
    from ewald import plugins

    plugins.register_reader("fake-scheme", lambda p, **k: [], schemes=("fakeloc",), priority=None)
    try:
        assert "fakeloc" in deeplink.location_schemes()
        assert deeplink.parse("ewald://open?location=fakeloc:1").location == "fakeloc:1"
    finally:
        plugins._registry.readers.pop("fake-scheme", None)
    with pytest.raises(DeepLinkError):
        deeplink.parse("ewald://open?location=fakeloc:1")


def test_is_deep_link():
    assert deeplink.is_deep_link("ewald://open?location=x:1") and deeplink.is_deep_link("EWALD:open")
    assert not deeplink.is_deep_link("ewald.py") and not deeplink.is_deep_link(None)


def test_desktop_entry_quotes_the_command():
    text = deeplink.desktop_entry(["/opt/my apps/ewald-gui"])
    assert 'Exec="/opt/my apps/ewald-gui" %u' in text
    assert "MimeType=x-scheme-handler/ewald;" in text and "NoDisplay=true" in text


def test_register_and_unregister_on_linux(tmp_path, monkeypatch):
    monkeypatch.setattr(deeplink.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(deeplink, "_run_quietly", lambda args: calls.append(args) or True)
    message = deeplink.register()
    entry = tmp_path / "applications" / deeplink.DESKTOP_FILE
    assert str(entry) in message and "x-scheme-handler/ewald" in entry.read_text(encoding="utf-8")
    assert b"\r\n" not in entry.read_bytes()
    assert ["xdg-mime", "default", deeplink.DESKTOP_FILE, "x-scheme-handler/ewald"] in calls
    assert "Removed" in deeplink.unregister() and not entry.exists()
    assert "No ewald" in deeplink.unregister()


def test_register_on_macos_points_to_the_app(monkeypatch):
    monkeypatch.setattr(deeplink.sys, "platform", "darwin")
    with pytest.raises(DeepLinkError, match="EWALD.app"):
        deeplink.register()


def test_cli_register_links(tmp_path, monkeypatch, capsys):
    from ewald.__main__ import main

    monkeypatch.setattr(deeplink.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(deeplink, "_run_quietly", lambda args: True)
    main(["register-links"])
    assert (tmp_path / "applications" / deeplink.DESKTOP_FILE).exists()
    main(["register-links", "--remove"])
    assert not (tmp_path / "applications" / deeplink.DESKTOP_FILE).exists()
    assert "Registered" in capsys.readouterr().out
