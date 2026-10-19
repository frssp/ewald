"""Frozen entry point for ewald.exe: the ``ewald`` CLI plus ``ewald run SCRIPT`` / ``ewald python``.

``run`` executes a user script with the bundled interpreter, so the installed app can run
pipeline scripts without a separate Python (like ``ovitos``).
"""
import runpy
import sys


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
    from ewald.__main__ import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
