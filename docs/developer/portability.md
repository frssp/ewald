# Windows, macOS and Linux

EWALD must behave nearly identically on Windows, macOS and Linux: the same files open, the
same results come out, exported files are byte-identical, and the GUI works the same way. The
owner runs EWALD on Windows, while the cloud containers and the CI that still runs are Linux,
so Windows differences are not seen unless a test simulates them. This page is the rule set
and the record of the audit from 2026-10-21 (after PR #189, where the NEB reader split a
Windows path on `/` and read `04/OUTCAR` instead of `POSCAR`).

`tests/test_portability.py` enforces the rules it can check on the source and simulates
Windows where it can (paths, CRLF files, text-mode writes, a cp949 locale).

## Rules

**Paths.** Local paths go through `os.path` or `pathlib`, never `str.split("/")`,
`rsplit("/")`, `"/".join(...)` or `a + "/" + b`. Remote paths on an SSH host are POSIX and
use `posixpath` or the `RemotePath` helpers in `io/remote.py`. A path string that may come
from another platform (a catalog index, an `.ewd` manifest, a notebook) is read with
`PureWindowsPath`, which splits on both separators everywhere. `os.path.relpath` raises
`ValueError` across Windows drives; handle it. A single letter before `:` is a drive, never a
URI scheme or an SSH host (`plugins.path_scheme`, `remote.parse_remote`). The few reviewed
string splits on `/` that are not file paths (menu paths, URLs, HDF5 or zip member names,
LAMMPS fix styles) are listed in `SEP_SITES` with the reason; a new site fails the test until
it uses a path API or is listed.

**Text files.** Every text `open()`, `os.fdopen()`, `Path.read_text()` and `write_text()`
names `encoding="utf-8"`, and every text write also names `newline="\n"`. Without them
Windows reads and writes the ANSI code page (cp949 on Korean Windows, cp1252 in Europe) and
writes CRLF line ends, which some codes on the cluster do not accept in a POSCAR or a LAMMPS
input. Writers that go through ASE use `io._ase_write`, which hands ASE a UTF-8, LF file object.
Text subprocess output also names its encoding (and the child gets `PYTHONUTF8=1` where it is
Python). Readers parse CRLF files exactly like LF files (native byte readers included); the
CRLF tests cover the VASP and LAMMPS readers. Redirected CLI output is UTF-8 everywhere
(`__main__._utf8_output`). Files older versions wrote in the locale encoding on Windows still
load (`settings.read_text_any`).

**Case.** Windows and the default macOS file system ignore case, Linux does not. Name matching
is made case-insensitive on every platform rather than following the OS: reader and writer
patterns (`plugin_kinds._matches`), sniffers (`.lower()` before comparing), and the open dialog
filters list both cases because Qt matches case on Linux only. `fnmatch.fnmatch` is forbidden
(it follows the OS); use `fnmatchcase`, on lower-cased strings where case should not matter.
Two files that differ only by case (`POSCAR`, `poscar`) are one file on Windows; do not rely on
both existing.

**File names.** A name that comes from somewhere else (a remote host, a plugin index) is made a
valid Windows name before it is created locally: `remote.local_name` replaces `<>:"/\|?*` and
control characters, drops trailing dots and spaces and prefixes device names (`CON`, `NUL`).
The same function runs on every platform so caches look alike.

**Processes.** No `fork`, `os.setsid`, `preexec_fn`, `start_new_session`, `shell=True`,
`multiprocessing` (spawn on Windows and macOS, fork on Linux), `fcntl`, `termios`, `pwd`,
`grp` or `resource`. Commands are argument lists. A frozen app has no Python interpreter at
`sys.executable` (see the open items).

**Files in use.** Windows refuses to replace or delete a file another process has open. Close
files before replacing or removing them, and read readers' files per access (no long-lived
handles or memory maps). Replace files only through `ewald.fileops.replace` (or `write_text`,
which writes a temporary file first): on Windows it retries for about 1.5 s, since virus scanners
and sync clients let go quickly, and then raises `FileInUseError` (a `PermissionError`) naming the
programs that hold the file, from the Windows Restart Manager. `test_portability.py` rejects a
bare `os.replace`.

**Secrets.** The fallback secrets file (`keychain.file_set`, when no OS keychain works) is mode
0600 on Linux and macOS and gets an owner-only ACL on Windows (`icacls /inheritance:r /grant:r`),
set before the secrets are written.

**OpenGL and fonts.** On Windows `viewport.backend = auto` reads the driver's OpenGL version from
a hidden WGL window (`gui.windows_opengl`); below 3.2 it uses the Legacy GL viewport with Qt's
software OpenGL. The UI fonts (IBM Plex Sans and Mono, JetBrains Mono) ship in
`ewald/assets/fonts` and are registered at start (`gui.fonts.load_fonts`), so the UI uses the
same fonts everywhere. Style sheets pass through `gui.fonts.css` (fonts set in code through
`gui.fonts.installed`), which drops the families the system lacks: a missing family, even late
in a fallback list, makes Qt build its alias table (about 200 ms on macOS, logged as
"Populating font family aliases ... missing font family"). matplotlib figures from
`plotstyle.figure()` and the Plots canvas are bound to the style (`plotstyle.bind`), so text
drawn outside `plotstyle.context()` (a canvas paint event) never falls back to matplotlib's
default math families (`mathtext.cal` = cursive, Apple Chancery on macOS).

**Numbers.** numpy 2 is the minimum, so the default integer is int64 on Windows as well
(numpy 1 used int32 there, and pair keys like `i * n + j` overflowed past 46k atoms). C++ code
uses fixed-width types (`int64_t`, `size_t`), never `long` (32-bit on Windows).

**GUI.** Restyle through `gui/theme.py`; give font stacks with a Windows, a macOS and a Linux
family; scale pixel sizes by `devicePixelRatioF()`; save dialogs pass their result through
`gui/file_dialogs.suffixed` (Windows adds the filter's extension, Qt's own dialog does not);
shortcuts use `QKeySequence` standard keys or Ctrl, which Qt maps to Cmd on macOS.
OpenGL widgets of one window share with its compositing context, made from the default format
when the first one appears: `gui/viewport.use_vtk_surface_format()` sets VTK's 3.2 core format
before that (`_app()` and `Viewport`). Linux and Windows drivers share across profiles, macOS
does not, and the VTK view composited black there while the rest of the window worked.
See-through atoms are sphere meshes, not impostors (`render.draw_scene`): the impostor shader under
depth peeling draws nothing on macOS. The VTK view has no MSAA: depth peeling into multisampled
framebuffers drops every see-through layer, on Mesa too.

**Tests.** A change that touches paths, files, processes or the GUI adds a test that simulates
the other platforms where that is possible: Windows paths through `ntpath` or
`PureWindowsPath` (see `_windows_os()` in `tests/test_portability.py`), CRLF copies of input
files, the `windows_text_mode` fixture for writers, and a patched `locale` for code pages.

## Audit of 2026-10-21

Fixed in the PR that added this page:

| Area | Problem | Fix |
|---|---|---|
| Text encoding | 60 text `open`/`read_text`/`write_text`/subprocess calls used the locale encoding; `settings.json` (written with `ensure_ascii=False`) could not hold characters outside cp949 on Korean Windows | `encoding="utf-8"` everywhere, enforced by the test; old locale-encoded settings still load |
| Line ends | Every exporter (extxyz, LAMMPS data/dump, XSF, CHGCAR/cube, and POSCAR/CIF/PDB through ASE) wrote CRLF on Windows | `newline="\n"`, `_ase_write` for ASE formats |
| Plugin catalog | A local plugin source `C:\...\x.py` was split on `/`, so the "file name" was the whole path; `is_installed` then looked at the source itself | `plugin_catalog.source_name` |
| `.ewd` save | A data file on another drive kept its absolute path, and the zip member became `data/D:\...` | `notebook_io._bundle_name` |
| Drive letters | `path_scheme("C:\\x")` returned `"c"`; the GUI treated any missing path with `:` as remote ("Downloading C:\...") | schemes need 2+ letters; `remote.is_remote` |
| Name matching | Reader patterns used `fnmatch.fnmatch`, case-insensitive on Windows only | `_matches` (case-insensitive everywhere) |
| Open dialog | Qt filters match case on Linux only | both cases listed |
| Save dialogs | A name typed without extension got one on Windows only | `file_dialogs.suffixed` |
| Remote cache | Remote names with `:` `*` `?` or device names could not be created on Windows | `remote.local_name` |
| CLI output | `ewald api > file` or pipes used the ANSI code page on Windows | UTF-8 when redirected |
| numpy | numpy 1 on Windows made int32 arrays by default | `numpy>=2.0`, `numba>=0.60` |

Open, needing a real Windows (or macOS) machine or a larger change; tracked in
[TODO](../todo.md):

1. Frozen apps (`packaging/windows`, `packaging/macos`): `sys.executable` is `ewald.exe`, so the
   Modifier Forge check subprocess, plugin `pip install` and the default notebook kernel cannot
   start a Python there.
2. OpenGL on Windows without 3.2 now gets the Legacy GL viewport with Qt's software OpenGL,
   but VTK's offscreen pictures (OpenGL renderer, `snapshot`) still use the system OpenGL there;
   the ray tracer and vector output need none. The WGL check is untested on a real VM or
   Remote Desktop session.
3. Windows Credential Manager limits a secret to 2560 bytes.
4. Native extension: without MSVC Build Tools Windows installs lack `ewald._native` (PTM,
   Voronoi, ...) until Windows wheels are published (Actions minutes).
5. Glob and `os.path.exists` follow the file system's case rules (`run*/XDATCAR` also matches
   `Run1/xdatcar` on Windows); kept as is, since the file system decides what can be opened.
6. Not yet seen on real hardware: the Windows look (bundled IBM Plex Sans, per-monitor DPI at 125/150 %),
   native file dialogs, Windows Credential Manager, ffmpeg from imageio-ffmpeg, a Windows
   OpenSSH config and agent, CUDA on Windows, and everything on a real Mac.
