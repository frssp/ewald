"""Secrets in the OS keychain: the Claude API key, SSH passwords and key passphrases.

>>> from ewald import keychain
>>> keychain.set_secret("ssh-password:cluster", "hunter2")   # 'keychain', or 'file' without one
>>> keychain.get_secret("ssh-password:cluster")
'hunter2'
>>> keychain.delete_secret("ssh-password:cluster")

Every secret is stored under the service ``ewald`` with its name as the user name. Where no
keychain backend works (a headless Linux box, ssh without a desktop session) secrets go to
``<config dir>/credentials.json``, one entry per name, readable by the owner only (mode 0600;
an owner-only ACL on Windows). Secrets
never go to settings.json, sessions, notebooks or logs.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

SERVICE = "ewald"
log = logging.getLogger(__name__)


def _keyring():
    """The keyring module when a real backend is usable, else None."""
    try:
        import keyring
        from keyring.backends import fail
    except Exception:
        return None
    try:
        backend = keyring.get_keyring()
    except Exception:
        return None
    if isinstance(backend, fail.Keyring) or getattr(backend, "priority", 1) <= 0:
        return None
    if type(backend).__name__ == "ChainerBackend" and not getattr(backend, "backends", ()):
        return None
    return keyring


def keyring_name() -> str:
    """Human name of the keychain in use ('' when there is none)."""
    kr = _keyring()
    if kr is None:
        return ""
    name = type(kr.get_keyring()).__module__.rsplit(".", 1)[-1]
    return {"macOS": "macOS Keychain", "Windows": "Windows Credential Manager",
            "SecretService": "Secret Service (GNOME Keyring / KWallet)", "kwallet": "KWallet"}.get(name, name)


# ----------------------------------------------------------------- 0600 file fallback
def secrets_file() -> Path:
    from .settings import config_dir

    return config_dir() / "credentials.json"


def file_get(name: str) -> str | None:
    try:
        return json.loads(secrets_file().read_text(encoding="utf-8")).get(name) or None
    except (OSError, ValueError, AttributeError):
        return None


def file_set(name: str, value: str | None) -> None:
    """Set or (``None``) remove one entry; the file is removed with its last entry."""
    path = secrets_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        data = {}
    if value is None:
        if name not in data:
            return
        data.pop(name)
    else:
        data[name] = value
    if not data:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600))
    owner_only(path)   # before the secrets go in; an existing file keeps its old mode through O_CREAT
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f)


def owner_only(path) -> bool:
    """Let only the current user read ``path``: mode 0600 on Linux and macOS; on Windows, where
    the mode means nothing, an ACL that grants the current user full control and drops every
    inherited entry (``icacls``). False (and a warning) when Windows refused."""
    if sys.platform != "win32":
        os.chmod(path, 0o600)
        return True
    import getpass
    import subprocess

    user, domain = getpass.getuser(), os.environ.get("USERDOMAIN", "")
    account = f"{domain}\\{user}" if domain else user
    try:
        done = subprocess.run(["icacls", os.fspath(path), "/inheritance:r", "/grant:r", f"{account}:F"],
                              capture_output=True, encoding="utf-8", errors="replace", timeout=30,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        ok, why = done.returncode == 0, (done.stderr or done.stdout).strip()
    except (OSError, subprocess.SubprocessError) as e:
        ok, why = False, str(e)
    if not ok:
        log.warning("could not limit %s to %s (icacls: %s); other accounts on this PC may read it",
                    path, account, why)
    return ok


# ----------------------------------------------------------------- generic secrets
def get_secret(name: str, _kr=...) -> tuple[str | None, str | None]:
    """(value, where) with where 'keychain', 'file' or None."""
    kr = _keyring() if _kr is ... else _kr
    if kr is not None:
        try:
            value = kr.get_password(SERVICE, name)
        except Exception:
            value = None
        if value:
            return value, "keychain"
    value = file_get(name)
    return (value, "file") if value else (None, None)


def set_secret(name: str, value: str, _kr=...) -> str:
    """Save ``value`` under ``name``; returns 'keychain' or 'file'."""
    if not value:
        raise ValueError("empty secret")
    kr = _keyring() if _kr is ... else _kr
    if kr is not None:
        try:
            kr.set_password(SERVICE, name, value)
            file_set(name, None)
            return "keychain"
        except Exception:
            pass
    file_set(name, value)
    return "file"


def delete_secret(name: str, _kr=...) -> None:
    kr = _keyring() if _kr is ... else _kr
    if kr is not None:
        try:
            kr.delete_password(SERVICE, name)
        except Exception:
            pass
    file_set(name, None)
