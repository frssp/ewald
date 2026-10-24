"""Python script modifier whose code is stored as text (editable in the GUI, saved in sessions)."""
from __future__ import annotations

import numpy as np

from ..pipeline import Modifier

TEMPLATE = '''import numpy as np

def modify(frame, data):
    """Called for every frame. Edit data.particles / data.attributes in place."""
    # Example: flag atoms in the upper half of the cell
    z = data.positions[:, 2]
    data["Upper"] = (z > z.mean()).astype(int)
'''


class PythonScript(Modifier):
    """Run ``modify(frame, data)`` defined in ``source``.

    Extra keyword parameters are passed to ``modify`` so the same script can be
    reused with different settings. In a script, ``pipe.add(function)`` (a
    :class:`~ewald.pipeline.PythonModifier`) is the simpler form of the same step.
    """

    cacheable = False  # arbitrary code: may read files, globals or the clock

    def __init__(self, source: str = TEMPLATE, title: str = "Python script", **params):
        self.source = source
        self.title = title
        self.params = params
        self._compiled_for = None
        self._function = None

    @property
    def name(self) -> str:
        return self.title

    def _compile(self):
        if self._compiled_for != self.source:
            import ewald

            namespace = {"np": np, "ewald": ewald, "__name__": "ewald_script"}
            exec(compile(self.source, f"<{self.title}>", "exec"), namespace)
            if "modify" not in namespace:
                raise NameError("script must define modify(frame, data)")
            self._function = namespace["modify"]
            self._compiled_for = self.source
        return self._function

    def modify(self, data, ctx):
        import inspect

        fn = self._compile()
        nargs = len(inspect.signature(fn).parameters)
        args = (ctx.frame, data, ctx)[: max(2, min(nargs, 3))]
        return fn(*args, **self.params)
