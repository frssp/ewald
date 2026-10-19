"""Opening files and examples, export, movies and sessions in the main window."""
from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from .. import examples
from .. import modifiers as mods
from ..io import import_file
from ..render import camera_state


def file_filter() -> str:
    """Open-dialog filter from the reader registry (built-in formats and plugins)."""
    from .. import plugins

    return (f"Simulation files ({' '.join(plugins.file_patterns('structure'))});;"
            f"Volumetric data ({' '.join(plugins.file_patterns('volumetric'))});;"
            "Notebooks (*.py);;Sessions (*.ewald);;All files (*)")


SESSION_FILTER = "ewald session (*.ewald)"
NOTEBOOK_FILTER = "Notebook (*.py)"


def save_filters() -> list[tuple[str, str | None]]:
    """Save-dialog filters from the writer registry: (filter text, writer name or None)."""
    from .. import plugins

    out = []
    for plugin in sorted(plugins.writers().values(), key=lambda p: (p.priority is None, p.priority or 0)):
        if plugin.patterns:
            out.append((f"{plugin.description} ({' '.join(plugin.patterns)})", plugin.name))
    return out + [("All files, format from the name (*)", None)]


def _has_phonon_modes(pipe) -> bool:
    try:
        return "Phonon Modes" in pipe.source_frame(0).attributes
    except Exception:
        return False


class FilesMixin:
    """Part of :class:`ewald.gui.app.MainWindow`."""

    # ---------------------------------------------------------------- pipeline / data
    def open_file(self, path, **kwargs):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            pipe = import_file(path, **kwargs)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.message(f"Load failed: {path}: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Load failed", f"{path}\n\n{exc}")
            return None
        QtWidgets.QApplication.restoreOverrideCursor()
        if _has_phonon_modes(pipe) and not any(isinstance(m, mods.PhononMode) for m in pipe.modifiers):
            pipe.modifiers.append(mods.PhononMode())  # a phonon file opens playing a mode
        self._check_bond_limit = True
        self.style.bond_pairs.clear()  # a new structure gets its own VESTA bond table
        self.set_pipeline(pipe)
        self.setWindowTitle(f"EWALD — {os.path.basename(str(path))}")
        self.message(f"Loaded {path}: {pipe.num_frames} frame(s)")
        return pipe

    def open_dialog(self):
        nb = self.layout_mode == "notebook"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open" if nb else "Load file", "", file_filter())
        if path:
            self.open_any(path)

    def open_any(self, path, ask_types: bool = True, **kwargs):
        """A session, a notebook (.py, notebook layout) or a data file."""
        from ..session import is_session_file

        if is_session_file(path):
            self.load_session(path)
        elif self.layout_mode == "notebook" and str(path).endswith(".py") and os.path.isfile(path):
            self.open_notebook(path)
        elif kwargs or not ask_types:
            self.open_file(path, **kwargs)
        else:
            self._open_asking_types(path)

    # ---------------------------------------------------------------- notebooks
    def open_notebook(self, path, run: bool | None = None):
        """Open a ``.py`` notebook in a new tab. Its cells run when ``python.run_notebooks``
        says so ("ask" asks: a notebook is code)."""
        if run is None:
            policy = self.settings["python.run_notebooks"]
            if policy == "ask":
                run = QtWidgets.QMessageBox.question(
                    self, "Run notebook", f"Run the cells of {os.path.basename(path)} now?\n\n"
                    "A notebook is Python code: run it only if you trust it.") == QtWidgets.QMessageBox.Yes
            else:
                run = policy == "always"
        try:
            view = self.panel.open_notebook(path, run=run)
        except Exception as exc:
            self.message(f"Open failed: {path}: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Open failed", f"{path}\n\n{exc}")
            return None
        self.notebook_dock.show()
        self.setWindowTitle(f"EWALD — {os.path.basename(str(path))}")
        self.message(f"Opened notebook {path}" + ("" if run else " (not run: Notebook › Run all)"))
        return view

    def save_notebook(self, ask: bool = False):
        """Save the current notebook as a ``.py`` file (asks for a name the first time)."""
        nb = self.panel.notebook
        if nb is None:
            return None
        path = nb.path
        if ask or not path:
            start = path or os.path.join(os.path.dirname(self.session_path or "") or os.getcwd(),
                                         nb.name.replace(" ", "_") + ".py")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save notebook", start, NOTEBOOK_FILTER)
            if not path:
                return None
            if not path.endswith(".py"):
                path += ".py"
        try:
            self.panel.save(path)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", f"{path}\n\n{exc}")
            return None
        self.message(f"Saved notebook {path}")
        return path

    def open_neb_dialog(self):
        """A VASP NEB folder (00 .. NN) or a LAMMPS NEB run folder, one frame per image."""
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Load NEB folder")
        if path:
            self.open_file(path)
            self.bottom.setCurrentWidget(self.plots)

    def open_example(self, name):
        """Open one of the shipped example files (File › Open example)."""
        return self.open_file(str(examples.path(name)))

    def open_remote_dialog(self):
        from .remote_dialog import RemoteFileDialog

        dlg = RemoteFileDialog(self)
        if dlg.exec() and dlg.url:
            self._open_asking_types(dlg.url)
        return dlg

    def _open_asking_types(self, path):
        kwargs = {}
        from ..io.lammps_dump import is_lammps_dump

        if is_lammps_dump(path):
            default_map = self.settings["general.lammps_type_map"]
            if self.settings["general.lammps_type_prompt"] == "ask":
                text, ok = QtWidgets.QInputDialog.getText(
                    self, "LAMMPS types", "Optional type → element map (e.g. 1=Cu,2=Zr):", text=default_map)
            else:
                text, ok = default_map, True
            if ok and text.strip():
                kwargs["type_map"] = dict(p.split("=") for p in text.replace(" ", "").split(","))
        self.open_file(path, **kwargs)

    def export_dialog(self):
        if self.pipeline is None:
            return
        from .. import plugins
        from ..io import export_file

        filters = save_filters()
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export data", "export.extxyz", ";;".join(f for f, _ in filters))
        if not path:
            return
        fmt = dict(filters).get(chosen)
        writer = plugins.find_writer(path, fmt)
        all_frames = False
        n = self.pipeline.num_frames
        if writer.multi_frame and n > 1:
            answer = QtWidgets.QMessageBox.question(
                self, "Export trajectory", f"{writer.description}\n\nExport all {n} frames? "
                "(No exports only the current frame.)",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
            if answer == QtWidgets.QMessageBox.Cancel:
                return
            all_frames = answer == QtWidgets.QMessageBox.Yes
        try:
            if all_frames:
                export_file(self.pipeline, path, format=fmt)
            else:
                export_file(self.data, path, format=fmt)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"{path}\n\n{exc}")
            return
        self.message(f"Exported {path} ({writer.name})")

    def export_movie(self, path=None):
        if self.pipeline is None:
            return
        if not path:
            from .render_page import MOVIE_FORMATS

            ext = MOVIE_FORMATS.get(self.render_page.movie_format.currentText(), ".mp4")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export movie", "movie" + ext, "Movies (*.mp4 *.gif *.avi *.mov);;PNG sequence (*.png)")
            if not path:
                return
        from ..render import render_animation

        page = self.render_page
        kw = dict(self.style.scene_kwargs())
        kw.update(page.render_kwargs(movie=True))
        w, h = kw.pop("size")
        frames = range(0, self.pipeline.num_frames, self.timeline.stride.value())
        dlg = QtWidgets.QProgressDialog(f"Rendering {os.path.basename(path)}…", "Cancel", 0, len(frames), self)
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumDuration(0)

        def progress(i, n):
            dlg.setValue(i)
            QtWidgets.QApplication.processEvents()
            return not dlg.wasCanceled()

        try:
            n = render_animation(self.pipeline, path, frames=frames, fps=self.timeline.fps.value(),
                                 device=self.device, size=(w // 2 * 2, h // 2 * 2),
                                 camera=camera_state(self.viewport.plotter), text=page.text.text() or None,
                                 time_per_frame=page.time_per_frame.value(), progress=progress, **kw)
        except Exception as exc:
            self.message(f"Movie export failed: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Movie export failed", str(exc))
            return None
        finally:
            dlg.close()
        self.message(f"Wrote {path} ({n} frames)")
        return path

    # ---------------------------------------------------------------- sessions
    def save_session(self, path=None, ask=False):
        """Save pipeline, style, camera and Render settings to an .ewald file."""
        if self.pipeline is None:
            return None
        path = path or (None if ask else self.session_path)
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save session", "session.ewald", SESSION_FILTER)
            if not path:
                return None
        from ..session import save_session

        try:
            save_session(path, self.pipeline, frame=self.frame, camera=camera_state(self.viewport.plotter),
                         style=self.style, render=self.render_page.state(), device=self.device)
        except (TypeError, OSError) as exc:
            self.message(f"Save failed: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Save session failed", str(exc))
            return None
        self.session_path = path
        self.setWindowTitle(f"EWALD — {os.path.basename(path)}")
        self.message(f"Saved session {path}")
        return path

    def load_session_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load session", "", SESSION_FILTER + ";;All files (*)")
        if path:
            self.load_session(path)

    def load_session(self, path, allow_code=None):
        """Restore a session. Python modifiers run only after the user agrees (or ``allow_code=True``)."""
        import json
        import warnings

        from ..session import CODE_MODIFIERS, load_session

        if allow_code is None:
            try:
                with open(path, encoding="utf-8") as fh:
                    has_code = any(m.get("type") in CODE_MODIFIERS for m in json.load(fh).get("modifiers", []))
            except (OSError, ValueError):
                has_code = False
            allow_code = has_code and QtWidgets.QMessageBox.question(
                self, "Python code in session",
                "This session contains Python modifiers. Run their code?\n\n"
                "Only answer Yes for sessions you trust; otherwise they are loaded disabled.") \
                == QtWidgets.QMessageBox.Yes
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = load_session(path, allow_code=bool(allow_code))
        except Exception as exc:
            self.message(f"Session load failed: {path}: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Load session failed", f"{path}\n\n{exc}")
            return None
        for w in s.warnings:
            self.message(w, error=True)
        if s.style:
            self.style._pref_filled = {}
            for k, v in s.style.items():  # in place: viewport, panel and console share this object
                if k in self.style.__dataclass_fields__:
                    setattr(self.style, k, list(map(list, v)) if k == "bond_pairs" else v)
        if s.device:
            self.device = s.device
            self.device_combo.blockSignals(True)
            self.device_combo.setCurrentText({"cuda": "CUDA", "auto": "auto"}.get(s.device, "CPU"))
            self.device_combo.blockSignals(False)
        self.set_pipeline(s.pipeline)
        if s.frame:
            self.timeline._set(min(s.frame, s.pipeline.num_frames - 1))
        if s.camera:
            from ..render import apply_camera_state

            apply_camera_state(self.viewport.plotter, s.camera)
            self.viewport.render()
        self.render_page.set_state(s.render)
        self._style_changed()
        self.session_path = path
        self.setWindowTitle(f"ewald — {os.path.basename(path)}")
        self.message(f"Loaded session {path}")
        return s
