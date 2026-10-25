"""The 3D view as notebook code: the main window follows ``pipe.view`` (a view cell, Claude's set_view,
the console) and writes what the user changes in the viewport back into the notebook's view cell."""
from __future__ import annotations

from ..render import apply_camera_state, camera_state, setup_camera
from ..view3d import SETTINGS, View

_CAMERA = ("camera", "direction", "zoom", "orthographic")
# render page keys of the picture settings (ewald.view3d 'picture' group)
_PAGE = {"renderer": "renderer_name", "legend": "legend", "type_legend": "type_legend",
         "ambient_occlusion": "ambient_occlusion", "shadows": "shadows", "text": "text_template"}


class ViewSyncMixin:
    _view_seen = None          # (id(view), version) the window applied or wrote last
    _view_camera = None        # camera settings applied last
    _view_frame = None         # frame setting applied last
    _writing_view = False

    def _pending_view(self, data) -> View | None:
        """The pipeline's view when code changed it since the window last looked. A view nothing set
        yet takes the window's style instead (the Preferences' default style, the last file's colors)."""
        pipe = self.pipeline
        if pipe is None:
            return None
        v = pipe.view
        key = (id(v), v.version)
        if self._view_seen is None or key[0] != self._view_seen[0]:   # another pipeline: its own camera
            self._view_camera = self._view_frame = None
        if key == self._view_seen:
            return None
        if not v.version:
            v.update(**self.style.view_settings(data.types if data is not None else ()))
            self._view_seen = (id(v), v.version)
            return None
        self._view_seen = key
        return v

    def follow_view(self, data) -> bool:
        """Before drawing ``data``: take the display settings of a changed ``pipe.view``. True if it changed."""
        v = self._pending_view(data)
        if v is None:
            return False
        self._writing_view = True
        try:
            self.style.apply_view(v, data.types if data is not None else ())
            self.viewport.style_combo.blockSignals(True)
            self.viewport.style_combo.setCurrentText(self.style.preset)
            self.viewport.style_combo.blockSignals(False)
            self._follow_picture(v)
        finally:
            self._writing_view = False
        self._pending_camera = v
        return True

    def _follow_picture(self, v: View):
        page = getattr(self, "render_page", None)
        if page is None:
            return
        st = {_PAGE[k]: getattr(v, k) for k in _PAGE if k in v._values and getattr(v, k) is not None}
        if v.quality is not None:
            st["quality"] = v.quality.capitalize()
        if v.size is not None:
            st["preset"], st["size"] = "Custom (px)", tuple(v.size)
        if v.scale_bar is not None:
            st["scale_bar_on"] = bool(v.scale_bar)
            if not isinstance(v.scale_bar, bool):
                st["scale_bar_length"] = float(v.scale_bar)
        if v.transparent_background:
            st["background_choice"] = "transparent"
        if "region" in v._values:
            st["region"] = v.region
        if v.overlays:
            st["overlays"] = [{"name": o, "params": {}, "enabled": True} if isinstance(o, str) else dict(o)
                              for o in v.overlays]
        if st:
            try:
                page.set_state(st)
            except Exception as exc:  # noqa: BLE001 - a picture setting the page cannot show
                self.message(f"View: {exc}", warning=True)

    def follow_view_camera(self):
        """After drawing: the camera and frame of the view applied last, when they changed."""
        v, self._pending_camera = getattr(self, "_pending_camera", None), None
        if v is None or self.data is None:
            return
        cam = tuple(repr(getattr(v, k)) for k in _CAMERA)
        if cam != self._view_camera:
            self._view_camera = cam
            if any(k in v._values for k in _CAMERA):
                if v.orthographic != self.viewport.ortho and hasattr(self.viewport, "set_ortho"):
                    self.viewport.set_ortho(v.orthographic)
                if v.camera is not None:
                    apply_camera_state(self.viewport.plotter, v.camera)
                else:
                    setup_camera(self.viewport.plotter, self.data, v.direction, v.orthographic, v.zoom)
                self.viewport.render()
        if v.frame != self._view_frame:
            self._view_frame = v.frame
            if "frame" in v._values and v.frame != self.frame and self.pipeline is not None:
                n = self.pipeline.num_frames     # view(pipe, frame=-1) is the last frame, as in pipe.compute(-1)
                self.timeline._set(min(max(v.frame + n if v.frame < 0 else v.frame, 0), n - 1))

    def write_view(self, camera: bool = False, region: bool = False):
        """After a change in the viewport, the View tab or a style menu: put the style into
        ``pipe.view`` and the notebook's view cell (``camera``: the current camera, trajectory frame and
        Render frame too; ``region``: the Render frame the user just changed)."""
        if self._writing_view or self.pipeline is None:
            return
        v = self.pipeline.view
        types = self.data.types if self.data is not None else ()
        keep = {k: x for k, x in v.changed().items() if SETTINGS[k].group in ("camera", "picture")}
        if camera:
            keep["camera"] = {k: (list(x) if isinstance(x, tuple) else x)
                              for k, x in camera_state(self.viewport.plotter).items()}
            keep.pop("direction", None)
            keep.pop("zoom", None)
            keep["frame"] = int(self.frame)
        page = getattr(self, "render_page", None)
        if page is not None and (camera or region):
            r = page.effective_region() if camera else page.region
            if r is None or tuple(r) == (0.0, 0.0, 1.0):
                keep.pop("region", None)
            else:
                keep["region"] = [round(float(x), 4) for x in r]
        wanted = View(**{k: x for k, x in self.style.view_settings(types).items()}, **keep).changed()
        if wanted != v.changed():
            v.reset().update(**wanted)
            if camera:
                self._view_camera = tuple(repr(getattr(v, k)) for k in _CAMERA)
                self._view_frame = v.frame
        self._view_seen = (id(v), v.version)
        view = self.panel.view
        if view is None or view.notebook.pipeline is not self.pipeline:
            return
        nb = view.notebook
        before = nb.view_cell()
        old = before.source if before is not None else None
        cell = nb.set_view_cell(v.changed())
        if cell is None or (cell is before and cell.source == old):
            return
        if cell is before:
            w = view.widget_for(cell)
            if w is not None and w.editor is not None:
                w.editor.blockSignals(True)
                w.editor.setPlainText(cell.source)
                w.editor.blockSignals(False)
        else:
            view.rebuild()
        view.history.commit("Change view", merge=("view", id(nb)))
        view.titleChanged.emit()

    def write_camera(self):
        """View › Write view to notebook: the current camera and frame into the view cell."""
        self.write_view(camera=True)
        self.message("Wrote the camera and frame into the notebook's view cell")

    # ---------------------------------------------------------------- view tabs (pipe.views)
    def follow_view_tabs(self, data):
        """Before drawing ``data``: one 3D viewport tab per view tab of the pipeline (``view(pipe,
        tab=..., ...)``), each drawn with its view; tabs the pipeline no longer has close."""
        pipe = self.pipeline
        tabs = self.__dict__.setdefault("view_tabs", {})
        names = pipe.views.names() if pipe is not None else []
        for name in [n for n in tabs if n not in names]:
            self._drop_view_tab(name)
        types = data.types if data is not None else ()
        for name in names:
            t = tabs.get(name) or self._make_view_tab(name)
            v = pipe.views[name]
            key = (id(v.base), v.base.version, v.version)
            if key != t.seen:
                t.seen = key
                t.style.apply_view(v.resolved(), types)
                t.viewport.style_combo.blockSignals(True)
                t.viewport.style_combo.setCurrentText(t.style.preset)
                t.viewport.style_combo.blockSignals(False)
                cam = tuple(repr(getattr(v, k)) for k in _CAMERA)
                if cam != t.camera:
                    t.camera, t.pending_camera = cam, True
            if data is not None:
                if t.style.bonds_enabled:
                    t.style.ensure_bond_pairs(data.types)
                t.viewport.overlay_text = self.viewport.overlay_text
                t.viewport.set_data(data)
                self._view_tab_camera(name)

    def _make_view_tab(self, name: str):
        from ..view_style import ViewStyle
        from .app import make_viewport
        from .prefs_apply import apply_viewport_settings

        style = ViewStyle()
        vp = make_viewport(style, self.settings["viewport.backend"])
        apply_viewport_settings(vp, self.settings)
        from .icons import button_style

        mode = self.settings["general.toolbar_style"]
        for b, text_in_hybrid in vp.styled_buttons:   # the tool rail as in the main viewport
            button_style(b, mode, text_in_hybrid)
        vp.set_button_style(mode)
        t = ViewTab(name, vp, style)
        self.view_tabs[name] = t
        vp.hovered.connect(lambda text: self.hover_label.setText(text))
        vp.picked.connect(lambda text: self.pick_label.setText(text))   # linked plots follow the main view only
        vp.styleChanged.connect(lambda n=name: self.write_view_tab(n))
        vp.contextMenuRequested.connect(lambda n=name: self._view_tab_menu(n))
        vp.viewBuilt.connect(lambda n=name: self._view_tab_camera(n))
        self.timeline.play.toggled.connect(vp.set_playing)
        group = self.workbench.group_of("viewport")
        self.workbench.register(t.view_id, vp, f"Structure · {name}", home="editor", closable=True,
                                native=True, group=group)
        return t

    def _drop_view_tab(self, name: str):
        t = self.view_tabs.pop(name, None)
        if t is None:
            return
        try:
            self.timeline.play.toggled.disconnect(t.viewport.set_playing)
        except (RuntimeError, TypeError):
            pass
        if t.view_id in self.workbench.specs:
            self.workbench.unregister(t.view_id)

    def _view_tab_camera(self, name: str):
        """The tab's camera from its view, once its 3D widget exists and shows the data."""
        t = self.view_tabs.get(name)
        if t is None or not t.pending_camera or self.pipeline is None or name not in self.pipeline.views:
            return
        vp = t.viewport
        if not vp.view_built or vp.data is None or vp.parked:
            return
        t.pending_camera = False
        v = self.pipeline.views[name]
        if any(k in v.resolved()._values for k in _CAMERA):
            if v.orthographic != vp.ortho and hasattr(vp, "set_ortho"):
                vp.set_ortho(v.orthographic)
            if v.camera is not None:
                apply_camera_state(vp.plotter, v.camera)
            else:
                setup_camera(vp.plotter, vp.data, v.direction, v.orthographic, v.zoom)
            vp.render()

    def _view_tab_shown(self, view_id: str):
        for name, t in getattr(self, "view_tabs", {}).items():
            if t.view_id == view_id:
                from PySide6 import QtCore

                QtCore.QTimer.singleShot(0, lambda n=name: self._view_tab_camera(n))   # after it unparks

    def _view_tab_closed(self, view_id: str):
        """Closing a view tab removes it from the pipeline and its view cell from the notebook (Undo brings
        the cell back; running it opens the tab again)."""
        name = next((n for n, t in getattr(self, "view_tabs", {}).items() if t.view_id == view_id), None)
        if name is None:
            return
        t = self.view_tabs.pop(name)
        try:
            self.timeline.play.toggled.disconnect(t.viewport.set_playing)
        except (RuntimeError, TypeError):
            pass
        if self.pipeline is not None:
            self.pipeline.views.remove(name)
        view = self.panel.view
        if view is None or view.notebook.pipeline is not self.pipeline:
            return
        cell = view.notebook.view_cell(name)
        if cell is not None:
            view.notebook.delete(cell)
            view.rebuild()
            view.history.commit("Close view tab")
            view.titleChanged.emit()

    def new_view_tab(self, name: str | None = None):
        """Window › New structure tab: another 3D view of the pipeline, written into the notebook as a
        ``view(pipe, tab=...)`` cell. Returns the tab's name."""
        pipe = self.pipeline
        if pipe is None:
            self.message("Open a structure first", warning=True)
            return None
        if name is None:
            n = 2
            while f"View {n}" in pipe.views:
                n += 1
            name = f"View {n}"
        pipe.views.add(name)
        self._write_view_tab_cell(name)
        self.follow_view_tabs(self.data)
        t = self.view_tabs.get(name)
        if t is not None:
            self.workbench.show(t.view_id)
        return name

    def write_view_tab(self, name: str, camera: bool = False):
        """After a change in a view tab's viewport (its style menu; ``camera``: its camera too): the tab's
        own settings, what differs from the main view, into ``pipe.views[name]`` and its view cell."""
        pipe, t = self.pipeline, getattr(self, "view_tabs", {}).get(name)
        if pipe is None or t is None or name not in pipe.views:
            return
        v, types = pipe.views[name], self.data.types if self.data is not None else ()
        main = pipe.view.resolved()
        keep = {k: x for k, x in v.changed().items() if SETTINGS[k].group in ("camera", "picture")}
        shown = View(**t.style.view_settings(types))
        own = {k: x for k, x in shown.settings().items()
               if SETTINGS[k].group not in ("camera", "picture") and repr(x) != repr(getattr(main, k))}
        if camera:
            keep["camera"] = {k: (list(x) if isinstance(x, tuple) else x)
                              for k, x in camera_state(t.viewport.plotter).items()}
            keep.pop("direction", None)
            keep.pop("zoom", None)
            keep["orthographic"] = bool(t.viewport.ortho)
        wanted = {k: x for k, x in {**own, **keep}.items()}
        if wanted != v.changed():
            v.reset().update(**{k: wanted[k] for k in SETTINGS if k in wanted})
        t.seen = (id(v.base), v.base.version, v.version)
        t.camera = tuple(repr(getattr(v, k)) for k in _CAMERA)
        t.viewport.redraw()
        self._write_view_tab_cell(name)

    def _write_view_tab_cell(self, name: str):
        view = self.panel.view
        if view is None or view.notebook.pipeline is not self.pipeline:
            return
        nb = view.notebook
        before = nb.view_cell(name)
        old = before.source if before is not None else None
        cell = nb.set_view_cell(self.pipeline.views[name].changed(), tab=name)
        if cell is None or (cell is before and cell.source == old):
            return
        if cell is before:
            w = view.widget_for(cell)
            if w is not None and w.editor is not None:
                w.editor.blockSignals(True)
                w.editor.setPlainText(cell.source)
                w.editor.blockSignals(False)
        else:
            view.rebuild()
        view.history.commit("Change view tab", merge=("view-tab", id(nb), name))
        view.titleChanged.emit()

    def _view_tab_menu(self, name: str):
        from PySide6 import QtGui, QtWidgets

        t = self.view_tabs.get(name)
        if t is None:
            return
        menu = QtWidgets.QMenu(t.viewport)
        menu.addAction("Write view to notebook", lambda: self.write_view_tab(name, camera=True)).setToolTip(
            f"Write this tab's camera and style into its view(pipe, tab={name!r}, ...) cell")
        menu.addAction("Reset view", t.viewport.reset_view)
        menu.addSeparator()
        menu.addAction("New structure tab", self.new_view_tab)
        menu.addAction("Close tab", lambda: self.workbench.close_view(t.view_id))
        menu.setToolTipsVisible(True)
        menu.exec(QtGui.QCursor.pos())
        menu.deleteLater()


class ViewTab:
    """One view tab of the window: its viewport, the style it draws with and what it applied last."""

    def __init__(self, name: str, viewport, style):
        self.name, self.viewport, self.style = name, viewport, style
        self.view_id = f"viewport:{name}"
        self.seen = None             # (id(base), base version, tab version) applied last
        self.camera = None           # camera settings applied last
        self.pending_camera = False
