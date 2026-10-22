"""Progress bars that work in a terminal, in Jupyter and under a cell of the EWALD notebook.

:func:`progress` is :class:`tqdm.tqdm` (rate, time left, units) with one change: inside a
notebook cell it is drawn as an updating output under the cell (a progress bar in the desktop
app, an HTML ``<progress>`` in a saved ``.ipynb``) instead of text on stderr::

    from ewald.progress import progress

    for frame in progress(range(pipe.num_frames), desc="frames"):
        ...

    with progress(total=size, unit="B", unit_scale=True, desc="XDATCAR") as bar:
        for block in blocks:
            bar.update(len(block))

Outside a cell (a script, ``ewald examples --fetch``) it is plain tqdm on stderr. Plain
``tqdm`` in a cell works as well: its text line updates in place.
"""
from __future__ import annotations

import html
import io
import os

from tqdm.std import tqdm as _tqdm

from .shell import PROGRESS_MIME, current


class Progress(_tqdm):
    """tqdm that publishes itself as a display output while a notebook cell runs."""

    def __init__(self, *args, **kwargs):
        self._ewald_sink = current()
        if self._ewald_sink is not None:
            kwargs["file"] = io.StringIO()   # nothing on stderr: the bar is an output of the cell
            kwargs.setdefault("ascii", False)
            kwargs.setdefault("ncols", 80)
            self._ewald_id = "progress-" + os.urandom(6).hex()
            self._ewald_shown = False
        super().__init__(*args, **kwargs)

    def bundle(self) -> dict:
        """This bar as a MIME bundle: text, HTML and the values for the GUI's progress bar."""
        d = self.format_dict
        text = self.format_meter(**d)
        total = d.get("total")
        value = {"n": d.get("n", 0), "total": total, "desc": d.get("prefix") or "", "text": text,
                 "rate": d.get("rate"), "elapsed": d.get("elapsed"), "unit": d.get("unit"),
                 "closed": bool(self.disable) or getattr(self, "_ewald_closed", False)}
        bar = (f'<progress value="{d.get("n", 0)}" max="{total}" style="width:20em"></progress> ' if total
               else "<progress style=\"width:20em\"></progress> ")
        return {"text/plain": text, "text/html": bar + f"<code>{html.escape(text)}</code>", PROGRESS_MIME: value}

    def display(self, msg=None, pos=None):
        sink = getattr(self, "_ewald_sink", None)
        if sink is None:
            return super().display(msg, pos)
        sink.display(self.bundle(), {}, self._ewald_id, update=self._ewald_shown)
        self._ewald_shown = True
        return True

    def close(self):
        if getattr(self, "_ewald_sink", None) is not None and not self.disable:
            self._ewald_closed = True
            self.display()
        super().close()


def progress(iterable=None, **kwargs) -> Progress:
    """A progress bar over ``iterable`` or for ``total=...`` manual ``update(n)`` calls; any tqdm
    keyword works (``desc``, ``unit``, ``unit_scale``, ``leave``, ``disable``, ...)."""
    return Progress(iterable, **kwargs)


tqdm = Progress   # ``from ewald.progress import tqdm`` for code written against tqdm
