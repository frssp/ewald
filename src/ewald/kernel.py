"""Run a notebook's cells in another Python: a Jupyter kernel per notebook.

A conda environment with a package ewald's GUI cannot import (a different torch, a DFT code's
Python bindings) still works as a notebook tab: its cells run in a kernel started from that
environment's Python, through ``jupyter_client``. The kernel needs ``ipykernel`` and ewald
(without the GUI extras). :mod:`ewald.kernel_bridge` runs the cells there with the usual notebook
rules, and the GUI's pipeline is a :class:`KernelPipeline` whose frames come from the kernel, so
the viewport, plots and local display steps work as usual::

    from ewald.kernel import Kernel
    nb.set_kernel(Kernel(python="/opt/conda/envs/ml/bin/python"))
    nb.run_all()
    nb.pipeline.compute(0)          # computed in the kernel, sent back as arrays

``Kernel.available()`` lists the Jupyter kernels installed on this computer.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading

STARTUP_TIMEOUT = 60.0


class KernelError(RuntimeError):
    """The kernel could not start, died, or the bridge failed (not an error in a cell)."""


def available() -> list[dict]:
    """Installed Jupyter kernels: ``{"name", "display_name", "python"}`` for each Python one."""
    from jupyter_client.kernelspec import KernelSpecManager

    out = []
    for name, info in sorted(KernelSpecManager().get_all_specs().items()):
        spec = info.get("spec", {})
        if spec.get("language", "").lower() != "python":
            continue
        argv = spec.get("argv") or []
        out.append({"name": name, "display_name": spec.get("display_name", name),
                    "python": argv[0] if argv else ""})
    return out


class Kernel:
    """One kernel process: ``python`` (an interpreter path) or ``name`` (an installed kernelspec)."""

    def __init__(self, python: str | None = None, name: str | None = None, display_name: str | None = None):
        if not python and not name:
            python = sys.executable
        self.python, self.name = python, name
        self.display_name = display_name or (name if name else _short(python))
        self._km = self._kc = None
        self._lock = threading.RLock()  # the GUI computes frames on a worker thread too
        self._dir = None

    def __repr__(self):
        return f"<Kernel {self.display_name!r}{' running' if self.alive else ''}>"

    @property
    def spec(self) -> dict:
        """What a notebook file keeps to start the same kernel again."""
        return {"name": self.name, "python": self.python, "display_name": self.display_name}

    @classmethod
    def from_spec(cls, spec: dict) -> Kernel:
        return cls(python=spec.get("python"), name=spec.get("name"), display_name=spec.get("display_name"))

    # ------------------------------------------------------------ process
    @property
    def alive(self) -> bool:
        return self._km is not None and self._km.is_alive()

    def start(self):
        from jupyter_client.manager import KernelManager

        with self._lock:
            if self.alive:
                return
            if self.name:
                km = KernelManager(kernel_name=self.name)
            else:
                km = KernelManager(kernel_name="ewald-python", kernel_spec_manager=_Specs(self.python))
            try:
                km.start_kernel(stderr=open(os.devnull, "w"))  # noqa: SIM115 - the kernel owns it
                kc = km.blocking_client()
                kc.start_channels()
                kc.wait_for_ready(timeout=STARTUP_TIMEOUT)
            except Exception as exc:
                try:
                    km.shutdown_kernel(now=True)
                except Exception:  # noqa: BLE001
                    pass
                raise KernelError(f"could not start {self.display_name}: {exc}") from exc
            self._km, self._kc = km, kc
            self._dir = tempfile.mkdtemp(prefix="ewald-kernel-")
            self._call("import ewald.kernel_bridge as _ewald_bridge",
                       hint=f"{self.display_name} has no ewald: run `{self.python or 'python'} -m pip install ewald` "
                            "in that environment (the GUI extras are not needed)")

    def restart(self):
        with self._lock:
            if self.alive:
                self._kc.stop_channels()
                self._km.restart_kernel(now=True)
                self._kc = self._km.blocking_client()
                self._kc.start_channels()
                self._kc.wait_for_ready(timeout=STARTUP_TIMEOUT)
                self._call("import ewald.kernel_bridge as _ewald_bridge")
            else:
                self.start()

    def interrupt(self):
        if self.alive:
            self._km.interrupt_kernel()

    def shutdown(self):
        with self._lock:
            if self._kc is not None:
                self._kc.stop_channels()
            if self._km is not None:
                try:
                    self._km.shutdown_kernel(now=True)
                except Exception:  # noqa: BLE001 - already gone
                    pass
            self._km = self._kc = None
            if self._dir:
                shutil.rmtree(self._dir, ignore_errors=True)
                self._dir = None

    # ------------------------------------------------------------ requests
    def _call(self, code: str, timeout: float | None = None, hint: str | None = None) -> str:
        """Run bridge code; returns what it printed. A Python error there raises KernelError."""
        with self._lock:
            if not self.alive:
                raise KernelError(f"{self.display_name} is not running")
            kc = self._kc
            msg_id = kc.execute(code, silent=False, store_history=False, allow_stdin=False)
            printed, error = [], None
            while True:
                try:
                    msg = kc.get_iopub_msg(timeout=timeout)
                except Exception as exc:  # queue.Empty on timeout
                    raise KernelError(f"{self.display_name} did not answer") from exc
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                kind, content = msg["msg_type"], msg["content"]
                if kind == "stream":
                    printed.append(content.get("text", ""))
                elif kind == "error":
                    error = f"{content.get('ename')}: {content.get('evalue')}"
                elif kind == "status" and content.get("execution_state") == "idle":
                    break
            if error:
                raise KernelError(hint or f"{self.display_name}: {error}")
            return "".join(printed)

    def run(self, cells: list, index: int, run_below: bool = True, counter: int = 0, pipeline_name: str = "pipe",
            path: str | None = None, data_dirs=()) -> dict:
        """Run cell ``index`` of ``cells`` (``[{id, kind, source}]``) in the kernel; the bridge's result."""
        self.start()
        out = os.path.join(self._dir, "result.json")
        args = [cells, index, bool(run_below), int(counter), out, pipeline_name, path, list(data_dirs)]
        self._call(f"_ewald_bridge.run(*__import__('json').loads({json.dumps(json.dumps(args))}))")
        with open(out, encoding="utf-8") as f:
            return json.load(f)

    def reset(self):
        """A fresh namespace, like restarting the kernel (it is restarted)."""
        if self.alive:
            self.restart()

    def frame(self, pipeline_name: str, frame: int):
        """Frame ``frame`` of the kernel's pipeline, computed there and read back here."""
        from .kernel_bridge import load_frame

        with self._lock:
            path = os.path.join(self._dir or tempfile.gettempdir(), f"frame-{frame}.npz")
            self._call(f"_ewald_bridge.save_frame({pipeline_name}.compute({int(frame)}), {path!r})")
            try:
                return load_frame(path)
            finally:
                os.remove(path)


def _short(python: str | None) -> str:
    """A readable name for an interpreter: its environment folder (``ml`` for .../envs/ml/bin/python)."""
    if not python:
        return "Python"
    if os.path.abspath(python) == os.path.abspath(sys.executable):
        return "This Python"
    parts = os.path.normpath(python).split(os.sep)
    env = parts[-3] if len(parts) >= 3 and parts[-2] in ("bin", "Scripts") else parts[-2] if len(parts) >= 2 else python
    return f"Python ({env})"


def _Specs(python: str):  # noqa: N802 - built lazily so importing ewald.kernel needs no jupyter_client
    from jupyter_client.kernelspec import KernelSpec, KernelSpecManager

    class Specs(KernelSpecManager):
        """One kernel spec: ipykernel in the given Python."""

        def get_kernel_spec(self, kernel_name, *args, **kwargs):
            return KernelSpec(argv=[python, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                              display_name=_short(python), language="python")

    return Specs()


class KernelSource:
    """Pipeline source whose frames the kernel computes (the kernel's ``pipe`` with all its steps)."""

    def __init__(self, kernel: Kernel, pipeline_name: str = "pipe", num_frames: int = 0):
        self.kernel, self.pipeline_name, self._num_frames = kernel, pipeline_name, num_frames

    @property
    def num_frames(self) -> int:
        return self._num_frames

    def compute(self, frame: int):
        data = self.kernel.frame(self.pipeline_name, frame)
        data.attributes.setdefault("SourceFrame", frame)
        return data


def kernel_pipeline(kernel: Kernel, pipeline_name: str = "pipe", num_frames: int = 0):
    """A local Pipeline over the kernel's ``pipe``: its frames come from the kernel; steps added
    here (display steps from the GUI) run locally on top."""
    from .pipeline import Pipeline

    pipe = Pipeline(KernelSource(kernel, pipeline_name, num_frames))
    pipe.kernel = kernel
    return pipe
