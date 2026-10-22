"""IPython runs the notebook's cells: one :class:`NotebookShell` per in-process notebook.

A cell's source goes through IPython's ``run_cell``, so cells behave as in Jupyter: magics
(``%timeit``, ``%%time``, ``!pip list``), ``display(...)`` and ``update_display``, rich
representations (``_repr_html_``, ``_repr_svg_``, ``_repr_markdown_``, ``_repr_latex_``,
``_repr_mimebundle_``), ``_`` / ``Out[n]`` and IPython's tracebacks. What a cell prints or shows
goes to a :class:`Sink` (the cell's outputs) as it happens, so a long cell shows its output and
progress bars while it runs.

Several notebooks can run at once on different threads (one GUI tab each, plus the console), so
nothing here swaps process-wide state for the length of a run: ``sys.stdout``/``sys.stderr``/
``sys.displayhook`` and the display publisher of IPython's global shell (which ``display()``
uses) are routers that send each thread's output to the sink that thread is running, and
everything else to where it went before.
"""
from __future__ import annotations

import contextlib
import io
import sys
import threading

PROGRESS_MIME = "application/vnd.ewald.progress+json"   # ewald.progress bars: the GUI draws a progress bar

_local = threading.local()
_lock = threading.RLock()


class Sink:
    """Where one running cell's outputs go. ``stream(name, text)`` for stdout/stderr, ``display``
    for ``display()`` and progress bars (``display_id`` + ``update`` replace an earlier one),
    ``result`` for the value of the last line, ``error`` for a traceback."""

    def stream(self, name: str, text: str): ...
    def display(self, data: dict, metadata: dict | None = None, display_id: str | None = None, update: bool = False): ...
    def clear(self, wait: bool = False): ...
    def result(self, data: dict, metadata: dict | None = None): ...
    def error(self, ename: str, evalue: str, text: str): ...


def current() -> Sink | None:
    """The sink of the cell this thread is running, or None outside a cell."""
    return getattr(_local, "sink", None)


@contextlib.contextmanager
def capture(sink: Sink, shell=None):
    """Send this thread's output to ``sink`` (and its last-line values to ``shell``'s display hook)."""
    previous = getattr(_local, "sink", None), getattr(_local, "shell", None)
    _local.sink, _local.shell = sink, shell
    _install()
    try:
        yield sink
    finally:
        _local.sink, _local.shell = previous
        _uninstall()


# ---------------------------------------------------------------- routers
class _Stream(io.TextIOBase):
    """``sys.stdout``/``sys.stderr`` while a cell runs: this thread's text goes to its sink."""

    def __init__(self, name: str, fallback):
        self.name, self.fallback = name, fallback

    def write(self, text):
        sink = current()
        if sink is None:
            return self.fallback.write(text) if self.fallback is not None else len(text)
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
        sink.stream(self.name, text)
        return len(text)

    def flush(self):
        if current() is None and self.fallback is not None:
            with contextlib.suppress(ValueError, OSError):   # the stream it replaced was closed (pytest's capture)
                self.fallback.flush()

    def close(self):   # never closes the stream it replaced (IOBase closes itself when collected)
        pass

    def writable(self):
        return True

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"

    def fileno(self):   # subprocess and faulthandler ask for it: the real stream's
        if self.fallback is None:
            raise io.UnsupportedOperation("fileno")
        return self.fallback.fileno()


def _displayhook(value):
    shell = getattr(_local, "shell", None)
    if current() is not None and shell is not None:
        return shell.displayhook(value)
    return _saved["displayhook"](value)


_saved: dict = {}
_users = [0]


def _install():
    with _lock:
        _users[0] += 1
        if _users[0] > 1:
            return
        _saved.update(stdout=sys.stdout, stderr=sys.stderr, displayhook=sys.displayhook)
        sys.stdout, sys.stderr = _Stream("stdout", sys.stdout), _Stream("stderr", sys.stderr)
        sys.displayhook = _displayhook
        route_display()


def _uninstall():
    with _lock:
        _users[0] -= 1
        if _users[0]:
            return
        # leave whatever someone else installed meanwhile alone
        if isinstance(sys.stdout, _Stream):
            sys.stdout = _saved["stdout"]
        if isinstance(sys.stderr, _Stream):
            sys.stderr = _saved["stderr"]
        if sys.displayhook is _displayhook:
            sys.displayhook = _saved["displayhook"]


class _Publisher:
    """IPython's global display publisher, routed: ``display()`` in a cell goes to the cell's
    sink, anywhere else (the console, a Jupyter kernel) to the publisher it replaced."""

    def __init__(self, inner):
        self._ewald_inner = inner

    def publish(self, data, metadata=None, *args, transient=None, update=False, **kwargs):
        sink = current()
        if sink is None:
            return self._ewald_inner.publish(data, metadata, *args, transient=transient, update=update, **kwargs)
        display_id = (transient or {}).get("display_id")
        sink.display(dict(data), dict(metadata or {}), display_id, update)

    def clear_output(self, wait=False):
        sink = current()
        if sink is None:
            return self._ewald_inner.clear_output(wait)
        sink.clear(wait)

    def __getattr__(self, name):   # session, pub_socket, set_parent ... of a kernel's publisher
        return getattr(self._ewald_inner, name)


def host():
    """IPython's global shell (the one ``display()`` and ``IPython.get_ipython()`` use): a
    Jupyter kernel's or the GUI console's when there is one, else a small private one."""
    from IPython.core.interactiveshell import InteractiveShell

    _shell_classes()
    with _lock:
        if not InteractiveShell.initialized():
            globals()["_Host"].instance(config=_config())
        return InteractiveShell.instance()


def release_host():
    """Drop the private global shell, so another one (the GUI console's in-process kernel) can be
    created; the routing moves onto that one the next time a cell runs."""
    from IPython.core.interactiveshell import InteractiveShell

    with _lock:
        if "_Host" in globals() and isinstance(InteractiveShell._instance, globals()["_Host"]):
            InteractiveShell.clear_instance()


def route_display():
    """Put the router on the global shell's display publisher (once per shell)."""
    shell = host()
    if not isinstance(shell.display_pub, _Publisher):
        shell.display_pub = _Publisher(shell.display_pub)
    if isinstance(shell, globals()["_Host"]) or type(shell).__name__ == "InProcessInteractiveShell":
        register_formatters(shell)   # ours (or the GUI console's): display() looks as a last line does


# ---------------------------------------------------------------- shells
def _config():
    from traitlets.config import Config

    c = Config()
    c.HistoryManager.enabled = False   # no SQLite file under ~/.ipython
    c.HistoryManager.hist_file = ":memory:"
    c.InteractiveShell.colors = "nocolor"   # tracebacks as plain text (cell outputs, .ipynb, the AI)
    c.InteractiveShell.autocall = 0
    c.InteractiveShell.ast_node_interactivity = "last_expr"
    c.InteractiveShell.xmode = "Context"
    return c


def _shell_classes():
    """The shell classes, defined on first use (importing IPython costs about 0.3 s)."""
    if "NotebookShell" in globals():
        return
    from IPython.core.builtin_trap import BuiltinTrap
    from IPython.core.compilerop import CachingCompiler
    from IPython.core.display_trap import DisplayTrap
    from IPython.core.displayhook import DisplayHook
    from IPython.core.displaypub import DisplayPublisher
    from IPython.core.interactiveshell import InteractiveShell

    class _NoDisplayTrap(DisplayTrap):
        """sys.displayhook stays the router (installed while any cell runs), not this shell's hook."""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _NoBuiltinTrap(BuiltinTrap):
        """get_ipython/exit/quit live in the cell namespace, not in the shared builtins."""

        def activate(self):
            pass

        def deactivate(self):
            pass

    class _Hook(DisplayHook):
        """The last line's value: into the running cell's sink, no ``Out[n]:`` prompt."""

        def start_displayhook(self):
            pass

        def write_output_prompt(self):
            pass

        def write_format_data(self, format_dict, md_dict=None):
            sink = current()
            if sink is not None:
                sink.result(dict(format_dict), dict(md_dict or {}))

        def finish_displayhook(self):
            pass

    class _Pub(DisplayPublisher):
        def publish(self, data, metadata=None, *args, transient=None, update=False, **kwargs):
            sink = current()
            if sink is not None:
                sink.display(dict(data), dict(metadata or {}), (transient or {}).get("display_id"), update)

        def clear_output(self, wait=False):
            sink = current()
            if sink is not None:
                sink.clear(wait)

    class _Compiler(CachingCompiler):
        """Code names are the notebook's (``<cell 3>``), so tracebacks name the cell."""

        label = "<cell>"

        def get_code_name(self, raw_code, transformed_code, number):
            return self.label

        def format_code_name(self, name):
            return None

    class NotebookShell(InteractiveShell):
        """An IPython shell whose namespace is a notebook's: it never takes over ``__main__``,
        the prompts, the builtins or ``sys.displayhook``, and writes no history file."""

        displayhook_class = _Hook
        display_pub_class = _Pub

        def init_sys_modules(self):
            pass

        def init_prompts(self):
            pass

        def init_builtins(self):
            super().init_builtins()
            self.builtin_trap = _NoBuiltinTrap(self)   # its shell

        def init_displayhook(self):
            super().init_displayhook()
            self.display_trap = _NoDisplayTrap(hook=self.displayhook)

        def init_instance_attrs(self):
            super().init_instance_attrs()
            self.compile = _Compiler()

        def init_virtualenv(self):
            pass   # a GUI started from a venv needs no warning about it

        def _showtraceback(self, etype, evalue, stb):
            sink = current()
            text = self.InteractiveTB.stb2text(stb)
            if sink is None:
                print(text)
                return
            sink.error(getattr(etype, "__name__", str(etype)), str(evalue), text)

        def init_user_ns(self):
            # a Jupyter kernel's namespace (ewald.kernel_bridge) keeps its own In/Out/get_ipython
            kept = {k: self.user_ns[k] for k in _USER_NS_KEYS if k in self.user_ns}
            super().init_user_ns()
            self.user_ns.update(kept)
            self.user_ns.setdefault("display", _display_function())
            self.user_ns_hidden["display"] = self.user_ns["display"]

        def bind(self, namespace: dict):
            """Run in ``namespace`` (the notebook's dict, used as is)."""
            if self.user_ns is namespace:
                return
            self.init_create_namespaces(user_ns=namespace)
            self.init_user_ns()

    class _Host(NotebookShell):
        """The global shell when nobody else made one (scripts, tests): only for ``display()``."""

    globals().update(NotebookShell=NotebookShell, _Host=_Host)


_USER_NS_KEYS = ("In", "Out", "_ih", "_oh", "_dh", "get_ipython", "exit", "quit", "open", "display")


def _display_function():
    from IPython.display import display

    return display


def __getattr__(name):   # NotebookShell / _Host exist once IPython is imported
    if name in ("NotebookShell", "_Host"):
        _shell_classes()
        return globals()[name]
    raise AttributeError(name)


def bind(shell, namespace: dict):
    """Make any IPython shell (the GUI console's) run in ``namespace``, keeping the In/Out/
    get_ipython already there, so the console and the notebook share their variables."""
    if shell.user_ns is namespace:
        return
    kept = {k: namespace[k] for k in _USER_NS_KEYS if k in namespace}
    shell.init_create_namespaces(user_ns=namespace)
    shell.init_user_ns()
    namespace.update(kept)


def new_shell(namespace: dict):
    """A :class:`NotebookShell` running in ``namespace``."""
    _shell_classes()
    shell = globals()["NotebookShell"](config=_config(), user_ns=namespace)
    register_formatters(shell)
    return shell


def run(shell, source: str, label: str, count: int, sink: Sink, transform=None) -> bool:
    """Run ``source`` as cell ``label`` (``<cell 3>``) with execution count ``count``; outputs go
    to ``sink``. ``transform(tree)`` may rewrite the parsed module (``[each frame]`` cells).
    Returns whether it ran without error."""
    shell.compile.label = label
    shell.execution_count = count
    transformer = _Transformer(transform) if transform is not None else None
    if transformer is not None:
        shell.ast_transformers.append(transformer)
    try:
        with capture(sink, shell):
            result = shell.run_cell(source, store_history=True)
    finally:
        if transformer is not None:
            shell.ast_transformers.remove(transformer)
    if transformer is not None and transformer.error is not None:
        exc = transformer.error
        sink.error(type(exc).__name__, str(exc), f"{type(exc).__name__}: {exc}")
        return False
    return bool(result.success)


class _Transformer:
    """An IPython AST transformer for one run (``visit`` is all IPython calls)."""

    def __init__(self, fn):
        self.fn, self.error = fn, None

    def visit(self, tree):
        try:
            return self.fn(tree)
        except Exception as exc:  # noqa: BLE001 - reported as the cell's error
            self.error = exc
            import ast

            return ast.Module(body=[], type_ignores=[])


# ---------------------------------------------------------------- formatters
def register_formatters(shell):
    """How ewald objects look as a cell's last line: a DataCollection as its summary, an RGB(A)
    image array as a picture, a View as text (the viewport shows it), matplotlib figures as PNG."""
    f = shell.display_formatter.formatters
    if getattr(shell.display_formatter, "_ewald", False):
        return
    shell.display_formatter._ewald = True
    f["text/plain"].for_type_by_name("ewald.data", "DataCollection", _summary_text)
    f["image/png"].for_type_by_name("ewald.data", "DataCollection", lambda data: None)   # text here, a picture in Jupyter
    f["image/png"].for_type_by_name("numpy", "ndarray", _array_png)
    f["image/png"].for_type_by_name("ewald.view3d", "View", lambda view: None)
    f["image/png"].for_type_by_name("matplotlib.figure", "Figure", _figure_png)


def _summary_text(obj, p, cycle):
    p.text(obj.summary())


def _array_png(value):
    import numpy as np

    if not (isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3 and value.shape[2] in (3, 4)):
        return None
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(value).save(buf, format="PNG")
    return buf.getvalue()


def _figure_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi="figure", bbox_inches="tight")
    return buf.getvalue()
