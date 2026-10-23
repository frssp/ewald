"""Which files and SSH hosts the AI's tools may read.

The model picks the paths it passes to ``load_structure``, ``describe_workflow`` and plugin
tools such as ``read_plumed_output``, and a path can come from text the model read (a
structure file's comment, a notebook from someone else). :class:`ReadAccess` limits them:

- ``folders``: local paths must lie inside one of these folders (after resolving ``~``,
  ``..`` and links; case-insensitive on Windows). Empty: any local path, as before.
- ``confirm_host``: called with the host name before the first connection to an SSH host the
  model named; False refuses it. The GUI asks the user; ``None`` allows every host (then
  ``~/.ssh/config`` and known_hosts still decide whether a connection works at all).

Paths with a registered scheme (``aiida:1234``, ``simagent:7``) are not files and go to their
plugin. In the GUI the folders come from Preferences › AI › Folders the AI may read
(``llm.read_folders``) and the host question from ``llm.confirm_hosts``. Plugin tools call
``toolset.check_read(path)`` before opening a path they were given.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Callable, Iterable


class ReadAccess:
    def __init__(self, folders: Iterable[str] = (), confirm_host: Callable[[str], bool] | None = None):
        self.folders = [os.path.realpath(os.path.expanduser(str(f))) for f in folders if str(f).strip()]
        self.confirm_host = confirm_host
        #: Hosts the user allowed (or refused) in this chat: asked once each.
        self.hosts: dict[str, bool] = {}

    @classmethod
    def from_settings(cls, settings, confirm_host=None) -> "ReadAccess":
        """Preferences › AI: ``llm.read_folders`` (folders separated by ';') and
        ``llm.confirm_hosts`` (ask through ``confirm_host`` when on)."""
        folders = [f.strip() for f in str(settings["llm.read_folders"] or "").replace(";", "\n").splitlines()]
        return cls(folders, confirm_host if settings["llm.confirm_hosts"] else None)

    def check(self, path) -> None:
        """Raise ``PermissionError`` unless the AI may read ``path``."""
        from .. import plugins
        from ..io import remote

        path = str(path)
        if plugins.path_scheme(path) and (plugins.scheme_reader(path) is not None or _workflow_scheme(path)):
            return
        rp = remote.parse_remote(path)
        if rp is not None:
            self._check_host(rp.host)
        else:
            self._check_folder(path)

    def _check_host(self, host: str) -> None:
        if self.confirm_host is None:
            return
        key = host.lower()
        if key not in self.hosts:
            self.hosts[key] = bool(self.confirm_host(host))
        if not self.hosts[key]:
            raise PermissionError(f"the user did not allow the AI to connect to {host}")

    def _check_folder(self, path: str) -> None:
        if not self.folders:
            return
        full = os.path.expanduser(path)
        if glob.has_magic(full):   # a wildcard: the folder before the first pattern decides
            fixed = []
            for part in Path(full).parts:
                if glob.has_magic(part):
                    break
                fixed.append(part)
            full = str(Path(*fixed)) if fixed else os.curdir
        full = os.path.normcase(os.path.realpath(full))
        for folder in self.folders:
            root = os.path.normcase(folder)
            try:
                if os.path.commonpath([full, root]) == root:
                    return
            except ValueError:   # another drive on Windows
                continue
        raise PermissionError(f"{path} is outside the folders the AI may read ({', '.join(self.folders)}); "
                              "Preferences › AI › Folders the AI may read")


def _workflow_scheme(path: str) -> bool:
    from .. import workflow

    try:
        return workflow.source_for(path) is not None
    except Exception:
        return False
