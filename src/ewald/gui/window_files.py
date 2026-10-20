"""Opening files and examples, export, movies and sessions in the main window."""
from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from .. import examples
from .. import plugins
from ..io import import_file
from ..render import camera_state
from .busy import busy
from .welcome import add_recent


def file_filter() -> str:
    """Open-dialog filter from the reader registry (built-in formats and plugins)."""
    from .. import plugins

    return (f"Simulation files ({' '.join(plugins.file_patterns('structure'))});;"
            f"Volumetric data ({' '.join(plugins.file_patterns('volumetric'))});;"
            "Notebooks (*.ewd *.ipynb *.py);;Sessions (*.ewald);;All files (*)")


SESSION_FILTER = "ewald session (*.ewald)"
NOTEBOOK_FILTERS = {".ewd": "EWALD notebook with results and data (*.ewd)",
                    ".ipynb": "Jupyter notebook (*.ipynb)", ".py": "Python script, code only (*.py)"}
NOTEBOOK_FILTER = ";;".join(NOTEBOOK_FILTERS.values())
NOTEBOOK_EXTENSIONS = tuple(NOTEBOOK_FILTERS)


def save_filters() -> list[tuple[str, str | None]]:
    """Save-dialog filters from the writer registry: (filter text, writer name or None)."""
    from .. import plugins

    out = []
    for plugin in sorted(plugins.writers().values(), key=lambda p: (p.priority is None, p.priority or 0)):
        if plugin.patterns:
            out.append((f"{plugin.description} ({' '.join(plugin.patterns)})", plugin.name))
    return out + [("All files, format from the name (*)", None)]


def _is_neb(path) -> bool:
    from ..io import is_neb

    return is_neb(path)


def _has_phonon_modes(pipe) -> bool:
    try:
        return "Phonon Modes" in pipe.source_frame(0).attributes
    except Exception:
        return False


class FilesMixin:
    """Part of :class:`ewald.gui.app.MainWindow`."""

    # ---------------------------------------------------------------- pipeline / data
    def open_file(self, path, **kwargs):
        remote = ":" in str(path) and not os.path.exists(str(path))     # host:/path, ssh://
        try:
            with busy(self.message, f"{'Downloading' if remote else 'Opening'} {os.path.basename(str(path))}…"):
                pipe = import_file(path, **kwargs)
        except Exception as exc:
            self.message(f"Load failed: {path}: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Load failed", f"{path}\n\n{exc}")
            return None
        phonon = plugins.modifiers().get("PhononMode")   # the ewald-phonopy plugin
        if phonon is not None and _has_phonon_modes(pipe) and not any(isinstance(m, phonon.cls) for m in pipe.modifiers):
            pipe.modifiers.append(phonon.cls())  # a phonon file opens playing a mode
        self._check_bond_limit = True
        self.style.bond_pairs.clear()  # a new structure gets its own VESTA bond table
        self.set_pipeline(pipe)
        self.setWindowTitle(f"EWALD — {os.path.basename(str(path))}")
        self.message(f"Loaded {path}: {pipe.num_frames} frame(s)")
        add_recent(path)
        return pipe

    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open", "", file_filter())
        if path:
            self.open_any(path)

    def open_any(self, path, ask_types: bool = True, **kwargs):
        """A session, a notebook (.ewd, .ipynb, .py) or a data file."""
        from ..session import is_session_file
        if is_session_file(path):
            self.load_session(path)
        elif str(path).lower().endswith(NOTEBOOK_EXTENSIONS) and os.path.isfile(path):
            self.open_notebook(path)
        elif not kwargs and (run := self._neb_run_to_open(path)):
            self.open_neb(run)
        elif kwargs or not ask_types:
            self.open_file(path, **kwargs)
        else:
            self._open_asking_types(path)

    def _neb_run_to_open(self, path) -> str | None:
        """The NEB run folder to open instead of ``path`` when it is one image of a NEB
        (``03/OUTCAR``, ``dump.neb.3``) and the user wants the whole band."""
        from ..io import neb_run_of

        run = neb_run_of(path) if "://" not in str(path) else None
        if run is None:
            return None
        answer = QtWidgets.QMessageBox.question(
            self, "NEB run", f"{os.path.basename(str(path))} is one image of the NEB run in\n{run}\n\n"
            "Open the whole band (one frame per image, with the energy profile)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)
        return run if answer == QtWidgets.QMessageBox.Yes else None

    # ---------------------------------------------------------------- drag and drop
    def dragEnterEvent(self, event):
        """Files and folders (a NEB run folder, for one) can be dropped on the window."""
        if any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return super().dropEvent(event)
        event.acceptProposedAction()
        # open after the drop returns, so the source app is not blocked by dialogs
        QtCore.QTimer.singleShot(0, self, lambda: self.open_dropped(paths))

    def open_dropped(self, paths):
        """Open dropped paths: notebooks each in a tab, otherwise the first data file or folder."""
        notebooks = [p for p in paths if str(p).lower().endswith(NOTEBOOK_EXTENSIONS) and os.path.isfile(p)]
        for p in notebooks:
            self.open_notebook(p)
        rest = [p for p in paths if p not in notebooks]
        if rest:
            if os.path.isdir(rest[0]) and _is_neb(rest[0]):
                self.open_neb(rest[0])
            else:
                self.open_any(rest[0])

    # ---------------------------------------------------------------- notebooks
    def open_notebook(self, path, run: bool | None = None):
        """Open a notebook (``.ewd``, ``.ipynb`` or ``.py``) in a new tab, as Jupyter does: a
        notebook you saved or trusted before (the same bytes, :func:`ewald.trust.is_trusted_file`)
        opens trusted and runs unless ``python.run_notebooks`` is "never"; any other file asks
        first (unless "never") and otherwise opens restricted, showing the results saved in it.
        An ``.ewd``'s plugins load only once it is trusted."""
        from ..trust import is_trusted_file

        trusted = is_trusted_file(path)
        if run is None:
            policy = self.settings["python.run_notebooks"]
            run = False if policy == "never" else True if trusted else self._ask_trust(path)
        try:
            view = self.panel.open_notebook(path, run=run, trusted=trusted)
        except Exception as exc:
            self.message(f"Open failed: {path}: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Open failed", f"{path}\n\n{exc}")
            return None
        nb = view.notebook
        for w in nb.warnings:
            self.message(w, error=True)
        self._apply_notebook_state(nb.state)
        self.notebook_dock.show()
        self.setWindowTitle(f"EWALD — {os.path.basename(str(path))}")
        note = "" if run else (" (restricted: shows saved results; Run all asks to trust it)" if nb.restricted
                               else " (not run: Notebook › Run all)")
        self.message(f"Opened notebook {path}{note}")
        add_recent(path)
        return view

    def _ask_trust(self, path) -> bool:
        """Ask before running a notebook; for an ``.ewd`` also name the plugins it carries."""
        extra = ""
        if str(path).lower().endswith(".ewd"):
            from ..notebook_io import read_manifest

            try:
                carried = [p.get("name") or p.get("id") for p in read_manifest(path).get("plugins", []) if p.get("file")]
            except Exception:  # noqa: BLE001 - the open itself reports a broken file
                carried = []
            if carried:
                extra = "\n\nIt carries plugins, which load when you agree: " + ", ".join(carried)
        return QtWidgets.QMessageBox.question(
            self, "Run notebook", f"Run the cells of {os.path.basename(str(path))} now?\n\n"
            "A notebook is Python code: run it only if you trust it." + extra) == QtWidgets.QMessageBox.Yes

    def _apply_notebook_state(self, state: dict):
        """The viewer state a notebook file saved: style, frame, camera and render settings."""
        if not state:
            return
        if state.get("render"):
            self.render_page.set_state(state["render"])
        for k, v in (state.get("style") or {}).items():
            if k in self.style.__dataclass_fields__:
                setattr(self.style, k, list(map(list, v)) if k == "bond_pairs" else v)
        if self.pipeline is not None and state.get("frame"):
            self.timeline._set(min(int(state["frame"]), self.pipeline.num_frames - 1))
        if state.get("camera") and self.pipeline is not None:
            from ..render import apply_camera_state

            apply_camera_state(self.viewport.plotter, state["camera"])
            self.viewport.render()
        self._style_changed(write=False)

    def notebook_state(self) -> dict:
        """What a notebook file keeps of the viewer: style, frame, camera and render settings."""
        state = {"style": self.style.to_dict(), "frame": int(self.frame), "render": self.render_page.state()}
        if self.pipeline is not None:
            state["camera"] = camera_state(self.viewport.plotter)
        return state

    def _notebook_switched(self, old, new):
        """Each notebook tab keeps its own render settings: the page follows the current tab."""
        if getattr(self, "render_page", None) is None:
            return
        if old is not None:
            old.state["render"] = self.render_page.state()
        if new is not None and new.state.get("render"):
            self.render_page.set_state(new.state["render"])
        if new is None and not self.panel.views():
            self.clear_pipeline()

    def save_notebook(self, ask: bool = False):
        """Save the current notebook (asks for a name the first time). The extension picks the
        format: ``.ewd`` (default: code, results, figures, small data files and plugins), ``.ipynb``
        (code and results) or ``.py`` (code)."""
        nb = self.panel.notebook
        if nb is None:
            return None
        path = nb.path
        if ask or not path:
            ext = os.path.splitext(path or "")[1].lower() if path else ".ewd"
            stem = os.path.splitext(path)[0] if path else os.path.join(
                os.path.dirname(self.session_path or "") or os.getcwd(), nb.name.replace(" ", "_"))
            path, chosen = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save notebook", stem + ext, NOTEBOOK_FILTER, NOTEBOOK_FILTERS.get(ext, ""))
            if not path:
                return None
            if not path.lower().endswith(NOTEBOOK_EXTENSIONS):
                path += next((e for e, f in NOTEBOOK_FILTERS.items() if f == chosen), ".ewd")
        nb.state = self.notebook_state()
        options = {}
        if path.lower().endswith(".ewd"):
            options = {"figures": bool(self.settings["python.save_figures"]),
                       "data_limit_mb": float(self.settings["python.bundle_data_mb"])}
        try:
            self.panel.save(path, **options)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", f"{path}\n\n{exc}")
            return None
        self.message(f"Saved notebook {path}")
        return path

    def open_neb_dialog(self):
        """A VASP NEB folder (00 .. NN) or a LAMMPS NEB run folder, one frame per image."""
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Load NEB folder")
        if path:
            self.open_neb(path)

    def open_neb(self, path):
        """Open a NEB run folder and show its energy profile in the Plots tab."""
        pipe = self.open_file(path)
        if pipe is not None:
            self.workbench.show(self.plots)
        return pipe

    def open_example(self, name):
        """Open one of the shipped example files (File › Open example)."""
        return self.open_file(str(examples.path(name)))

    def location_schemes(self) -> list[str]:
        """URI schemes the installed readers serve (``aiida``, ``simagent``, ...), for Open location."""
        return sorted({s for p in plugins.readers().values() for s in p.schemes})

    def open_location_dialog(self, text: str | None = None):
        """File › Open location: a path or address typed in, ``host:/path`` over SSH or a plugin
        scheme such as ``simagent:1234`` (the same strings ``load()`` takes)."""
        if text is None:
            schemes = self.location_schemes()
            hint = ("A file path, host:/path over SSH"
                    + (", or " + ", ".join(f"{s}:…" for s in schemes) if schemes else "")
                    + "\n(the same text load() takes in a notebook cell)")
            text, ok = QtWidgets.QInputDialog.getText(self, "Open location", hint,
                                                      text=getattr(self, "_last_location", ""))
            if not ok:
                return None
        text = text.strip()
        if not text:
            return None
        self._last_location = text
        if plugins.path_scheme(text) in self.location_schemes():
            return self.open_file(text)      # not a file: no type questions, no NEB lookup
        self.open_any(text)
        return self.pipeline

    def open_remote_dialog(self):
        from .remote_dialog import RemoteFileDialog

        dlg = RemoteFileDialog(self)
        if dlg.exec() and dlg.url:
            self._open_asking_types(dlg.url)
        return dlg

    def show_remotes(self):
        """Show the Remotes side bar (saved SSH computers and their files)."""
        self._show_side_bar(self.remotes_dock)
        self.remotes_panel.list.setFocus()

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
        self.set_pipeline(s.pipeline)
        if s.frame:
            self.timeline._set(min(s.frame, s.pipeline.num_frames - 1))
        if s.camera:
            from ..render import apply_camera_state

            apply_camera_state(self.viewport.plotter, s.camera)
            self.viewport.render()
        self.render_page.set_state(s.render)
        self._style_changed(write=False)
        self.session_path = path
        self.setWindowTitle(f"ewald — {os.path.basename(path)}")
        self.message(f"Loaded session {path}")
        return s
