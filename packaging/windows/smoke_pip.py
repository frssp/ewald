"""Smoke test for plugins in the frozen app: ``ewald run smoke_pip.py``.

Installs a small first-party plugin the way the Plugins side bar does
(:func:`ewald.plugin_catalog.pip_command`: ewald.exe -m pip into the add-on prefix), once from
git and once from GitHub's archive (what happens on a PC without git), and checks that a new
ewald process imports it and lists its entry point.
"""
import subprocess
import sys

from ewald import plugin_catalog as pc

SOURCE = ("git+https://github.com/frssp/ewald-plugins@6b3c056791825ef5ea6a5a1e4fec7adb0d4bd66a"
          "#subdirectory=packages/ewald-plumed")
CHECK = ("import importlib.metadata as m, ewald_plumed; "
         "assert 'plumed' in [e.name for e in m.entry_points(group='ewald.plugins')]; print('plugin ok')")


def run(cmd):
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


for source in (SOURCE, pc.github_archive(SOURCE)):
    run(pc.pip_command("install", source))
    run([pc.app_python(), "-c", CHECK])
    run(pc.pip_command("uninstall", "ewald-plumed"))
print("PIP SMOKE OK")
