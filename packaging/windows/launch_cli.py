"""Frozen entry point for ewald.exe: the ``ewald`` CLI plus ``ewald run SCRIPT`` / ``ewald python``.

``run`` executes a user script with the bundled interpreter, so the installed app can run
pipeline scripts without a separate Python (like ``ovitos``).

ewald.exe also answers the command lines tools give ``sys.executable`` as if it were python:
``-m MODULE``, ``-c CODE`` and ``SCRIPT.py``, after any of python's one-letter switches. pip
needs this: the Plugins side bar runs ``ewald.exe -m pip install ...``, and pip runs its build
backend as ``ewald.exe .../_in_process.py``. A script is run only when it is a path to an
existing ``.py`` file, so it never shadows a CLI command (those are words).
"""
import os
import runpy
import sys

# python switches that take no value and change nothing that matters here
_FLAGS = {"-I", "-E", "-s", "-S", "-u", "-B", "-O", "-OO", "-q", "-b", "-bb", "-P"}
_WITH_VALUE = {"-X", "-W"}


def _as_python(argv):
    """Run ``argv`` as python would if it is a python command line; None if it is not."""
    i = 0
    while i < len(argv) and (argv[i] in _FLAGS or argv[i] in _WITH_VALUE):
        i += 2 if argv[i] in _WITH_VALUE else 1
    rest = argv[i:]
    if not rest:
        return None
    if rest[0] == "-m" and len(rest) > 1:
        sys.argv = [rest[1], *rest[2:]]
        runpy.run_module(rest[1], run_name="__main__", alter_sys=True)
        return 0
    if rest[0] == "-c" and len(rest) > 1:
        sys.argv = ["-c", *rest[2:]]
        exec(compile(rest[1], "<string>", "exec"), {"__name__": "__main__"})  # noqa: S102 (the caller's own code, as python -c)
        return 0
    if rest[0].endswith(".py") and os.path.isfile(rest[0]):
        sys.argv = rest
        sys.path.insert(0, os.path.dirname(os.path.abspath(rest[0])))
        runpy.run_path(rest[0], run_name="__main__")
        return 0
    return None


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "run":
        if len(argv) < 2:
            sys.exit("usage: ewald run SCRIPT.py [args ...]")
        sys.argv = argv[1:]
        runpy.run_path(argv[1], run_name="__main__")
        return 0
    if argv and argv[0] == "python":
        import code

        sys.argv = [""]
        code.interact(local={"__name__": "__main__"})
        return 0
    code = _as_python(argv)
    if code is not None:
        return code
    from ewald.__main__ import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
