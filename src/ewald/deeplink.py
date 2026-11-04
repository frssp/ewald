"""``ewald://open?location=...``: links on a web page that open a location in the desktop app.

A results server or a lab wiki puts such a link behind an "Open in EWALD" button. The browser
asks "Open EWALD?", and the OS hands the link to ``ewald-gui`` (Windows registry, the macOS
app's ``CFBundleURLTypes``, a Linux ``x-scheme-handler/ewald`` desktop entry). The GUI shows
what the link wants to open and opens it on the user's OK, exactly as File › Open from ›
Location (Ctrl+L) would.

Any web page can make the browser launch such a link, so a link is data, never code:

- it opens only a location served by an installed reader's URI scheme (``aiida:1234`` and the
  like): no local path, SSH host or notebook, so a page cannot point EWALD at a file of its
  choosing;
- the location is read with that reader, which only reads; nothing in the link is run;
- ``title`` is plain text shown in the confirmation, cut to :data:`MAX_TITLE` characters.

Pages that link here: put the location in ``location``, URL-encoded, and an optional
``title`` (what the user will recognise, e.g. "PK 1234 · Si bands"). The link format follows
the custom URL scheme convention of RFC 7595 (https://www.rfc-editor.org/rfc/rfc7595).

:func:`register` / :func:`unregister` (``ewald register-links``) register the scheme for a pip
install; the Windows installer and the macOS app register it themselves.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

SCHEME = "ewald"
#: longer links are refused: no real location needs more, and a huge one only hides its tail
MAX_LENGTH = 2048
MAX_TITLE = 120
ACTIONS = ("open",)


class DeepLinkError(ValueError):
    """A link EWALD will not open; the message says why, for the user."""


@dataclass(frozen=True)
class DeepLink:
    location: str
    title: str = ""

    def url(self) -> str:
        """The link back as text (``ewald://open?location=...``)."""
        query = {"location": self.location, **({"title": self.title} if self.title else {})}
        return f"{SCHEME}://open?" + urllib.parse.urlencode(query, quote_via=urllib.parse.quote)


def is_deep_link(text) -> bool:
    """True for ``ewald:...`` text (a link, not a file path)."""
    return isinstance(text, str) and text.strip().lower().startswith(SCHEME + ":")


def _printable(text: str) -> bool:
    return all(ch.isprintable() for ch in text)


def location_schemes() -> list[str]:
    """URI schemes the installed and enabled readers serve: what a link may open."""
    from . import plugins

    return sorted({s.lower() for p in plugins.readers().values() for s in p.schemes})


def parse(url: str, schemes=None) -> DeepLink:
    """Read ``ewald://open?location=LOC[&title=TEXT]`` (also ``ewald:open?...``, as some
    browsers pass it). ``schemes``: the location schemes allowed (default: the installed
    readers'). Raises :class:`DeepLinkError` with a message for the user."""
    if not isinstance(url, str) or not is_deep_link(url):
        raise DeepLinkError(f"Not an {SCHEME}:// link: {url!r}")
    url = url.strip()
    if len(url) > MAX_LENGTH:
        raise DeepLinkError(f"The link is longer than {MAX_LENGTH} characters.")
    parts = urllib.parse.urlsplit(url)
    action = (parts.netloc or parts.path).strip("/").lower()
    if parts.netloc and parts.path.strip("/"):
        action = f"{parts.netloc}/{parts.path.strip('/')}".lower()
    if action not in ACTIONS:
        raise DeepLinkError(f"Unknown link action {action!r}; this version of EWALD knows: "
                            + ", ".join(ACTIONS))
    try:
        query = urllib.parse.parse_qs(parts.query, strict_parsing=bool(parts.query),
                                      errors="strict", max_num_fields=8)
    except ValueError as exc:
        raise DeepLinkError(f"The link's query is malformed: {exc}") from None
    unknown = sorted(set(query) - {"location", "title"})
    if unknown:
        raise DeepLinkError(f"Unknown link field(s): {', '.join(unknown)}")
    locations = query.get("location", [])
    if len(locations) != 1:
        raise DeepLinkError("The link must name exactly one location.")
    location = locations[0].strip()
    if not location or not _printable(location):
        raise DeepLinkError("The link's location is empty or has control characters.")
    from .plugin_kinds import path_scheme

    allowed = location_schemes() if schemes is None else sorted(s.lower() for s in schemes)
    scheme = path_scheme(location)
    if scheme is None or scheme not in allowed:
        raise DeepLinkError(
            f"A link can open only a location of an installed reader ({', '.join(f'{s}:' for s in allowed) or 'none installed'}), "
            f"not {location!r}. Install the plugin that reads it, or open it from File › Open.")
    title = " ".join(query.get("title", [""])[0].split())
    title = "".join(ch for ch in title if ch.isprintable())[:MAX_TITLE]
    return DeepLink(location, title)


# --------------------------------------------------------------------------- registration
def _gui_command() -> list[str]:
    """How the OS should start the GUI with a link: the ``ewald-gui`` launcher next to this
    Python (a pip install), else ``python -m ewald gui``."""
    exe = Path(sys.executable)
    names = ("ewald-gui.exe",) if os.name == "nt" else ("ewald-gui",)
    for folder in (exe.parent, exe.parent / "Scripts", exe.parent / "bin"):
        for name in names:
            if (folder / name).is_file():
                return [str(folder / name)]
    found = shutil.which("ewald-gui")
    if found:
        return [found]
    python = exe.with_name("pythonw.exe") if os.name == "nt" and exe.with_name("pythonw.exe").is_file() else exe
    return [str(python), "-m", "ewald", "gui"]


def _desktop_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base) / "applications"


DESKTOP_FILE = "ewald-links.desktop"


def _desktop_quote(arg: str) -> str:
    """One Exec argument of a desktop entry (freedesktop Desktop Entry spec, "The Exec key":
    https://specifications.freedesktop.org/desktop-entry-spec/latest/exec-variables.html)."""
    if arg and not any(ch in arg for ch in ' \t\n"\'\\><~|&;$*?#()`'):
        return arg
    return '"' + "".join("\\" + ch if ch in '"`$\\' else ch for ch in arg) + '"'


def _icon_path() -> str:
    icon = Path(__file__).parent / "assets" / "icon" / "ewald-256.png"
    return str(icon) if icon.is_file() else "ewald"


def desktop_entry(command=None) -> str:
    """The Linux desktop entry that hands ``ewald:`` links to the GUI (hidden from menus)."""
    exec_line = " ".join(_desktop_quote(a) for a in (command or _gui_command())) + " %u"
    return ("[Desktop Entry]\nType=Application\nName=EWALD\nComment=Open EWALD links\n"
            f"Exec={exec_line}\nIcon={_icon_path()}\nTerminal=false\nNoDisplay=true\n"
            f"MimeType=x-scheme-handler/{SCHEME};\n")


def _run_quietly(args) -> bool:
    if shutil.which(args[0]) is None:
        return False
    try:
        return subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                              timeout=30, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _windows_key() -> str:
    return rf"Software\Classes\{SCHEME}"


def register() -> str:
    """Register ``ewald:`` links for this user to open with this install; returns what was done."""
    command = _gui_command()
    if sys.platform == "win32":
        import winreg

        line = subprocess.list2cmdline([*command, "%1"])
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _windows_key()) as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "URL:EWALD link")
            winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _windows_key() + r"\shell\open\command") as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, line)
        return f"Registered {SCHEME}:// links for this user: {line}"
    if sys.platform == "darwin":
        raise DeepLinkError("On macOS the EWALD app registers ewald:// links itself: install EWALD.app "
                            "(docs/developer/macos-release.md) and open it once.")
    folder = _desktop_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / DESKTOP_FILE
    path.write_text(desktop_entry(command), encoding="utf-8", newline="\n")
    _run_quietly(["update-desktop-database", str(folder)])
    default = _run_quietly(["xdg-mime", "default", DESKTOP_FILE, f"x-scheme-handler/{SCHEME}"])
    note = "" if default else " (xdg-mime not found: your desktop may need a log-out to pick it up)"
    return f"Registered {SCHEME}:// links with {path}{note}"


def unregister() -> str:
    """Undo :func:`register` for this user."""
    if sys.platform == "win32":
        import winreg

        for sub in (r"\shell\open\command", r"\shell\open", r"\shell", ""):
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, _windows_key() + sub)
            except OSError:
                pass
        return f"Removed the {SCHEME}:// link registration of this user."
    if sys.platform == "darwin":
        raise DeepLinkError("On macOS the link registration belongs to EWALD.app; remove the app to remove it.")
    path = _desktop_dir() / DESKTOP_FILE
    if path.exists():
        path.unlink()
        _run_quietly(["update-desktop-database", str(path.parent)])
        return f"Removed {path}"
    return "No ewald:// link registration to remove."
