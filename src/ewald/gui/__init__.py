"""Desktop GUI (PySide6 + VTK). Install with ``pip install 'ewald[gui]'``."""


def show(pipeline=None, device=None, block: bool = True):
    from .app import show as _show

    return _show(pipeline, device, block)


def main(argv=None):
    from .app import main as _main

    return _main(argv)
