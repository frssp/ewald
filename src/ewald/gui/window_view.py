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
