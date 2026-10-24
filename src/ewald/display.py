"""Pictures for Jupyter (and the GUI notebook): a Pipeline, DataCollection or verb Table as the last
line of a cell shows an image instead of its text repr::

    pipe = load("XDATCAR")
    pipe                        # frame 0, drawn by the built-in ray tracer (no OpenGL needed)
    pipe.compute(5)             # another frame
    rdf(pipe)                   # the g(r) curve

``options`` sets the picture: ``size``, ``renderer`` ('raytrace', 'opengl'), ``quality`` and any
other :func:`ewald.render.render` keyword (``direction=``, ``bonds=``, ``theme=``, ...).
``quality=None`` (the default) is 'standard', or 'draft' above :data:`LARGE` atoms.
``options["enabled"] = False`` goes back to text. A picture that fails falls back to text too.
"""
from __future__ import annotations

import io

options: dict = {"enabled": True, "size": (480, 360), "renderer": "raytrace", "quality": None, "scale": 1.0}
LARGE = 100_000  # quality=None draws more atoms than this as 'draft', fewer as 'standard'


def _quiet_vtk():
    """Silence VTK's 'bad X server connection' warning while the camera is framed without a display."""
    try:
        from vtkmodules.vtkCommonCore import vtkLogger
    except ImportError:
        return None
    level = vtkLogger.GetCurrentVerbosityCutoff()
    vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_ERROR)
    return lambda: vtkLogger.SetStderrVerbosity(level)


_BASE = tuple(options)


def data_png(data, view=None, frame: int = 0) -> bytes | None:
    """PNG bytes of ``data`` drawn with :data:`options` (and a pipeline's ``view`` settings, which
    win over the base options; other keys you put into ``options`` win over the view), or None
    (disabled, empty or failed)."""
    if not options.get("enabled", True) or data is None or not data.count:
        return None
    from PIL import Image

    from .render import render

    kwargs = {k: v for k, v in options.items() if k != "enabled"}
    if view is not None:
        from .view3d import merge

        kwargs = merge({**kwargs, **view.render_kwargs(data, frame)},
                       {k: v for k, v in options.items() if k not in _BASE})
    if kwargs.get("quality") is None:
        kwargs["quality"] = "draft" if data.count > LARGE else "standard"
    restore = _quiet_vtk()
    try:
        img = render(data, **kwargs)
    except Exception:  # noqa: BLE001 - Jupyter then shows the text repr
        return None
    finally:
        if restore:
            restore()
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def pipeline_png(pipe, frame: int | None = None) -> bytes | None:
    if not options.get("enabled", True) or not pipe.num_frames:
        return None
    view = pipe.view
    frame = view.frame if frame is None else frame
    try:
        data = pipe.compute(min(frame, pipe.num_frames - 1))
    except Exception:  # noqa: BLE001
        return None
    return data_png(data, view, frame)


def table_png(table) -> bytes | None:
    """The curve of a verb :class:`~ewald.verbs.Table` (what ``table.plot()`` draws)."""
    if not options.get("enabled", True) or not table:
        return None
    from . import plotstyle
    from .views import _draw

    fig = plotstyle.figure()
    try:
        with plotstyle.context(*fig._ewald_style):
            _draw(fig.add_subplot(), table.name, dict(table), None, show_title=False,
                  labels=getattr(table, "labels", None), attributes=getattr(table, "attributes", None))
        buf = io.BytesIO()
        plotstyle.save(fig, buf, format="png", dpi=150)
    except Exception:  # noqa: BLE001
        return None
    return buf.getvalue()
