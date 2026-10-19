"""Animation export: GIF, MP4, AVI, MOV, WebM and numbered PNG/JPG/TIFF sequences.

GIF and image sequences use Pillow (always installed). Video containers pipe raw
frames into ffmpeg: the binary from ``imageio-ffmpeg`` (``pip install 'ewald[movie]'``)
when installed, else ``ffmpeg`` on PATH.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

# container -> ffmpeg output arguments
VIDEO_CODECS = {
    ".mp4": ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart"],
    ".mov": ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"],
    ".m4v": ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"],
    ".avi": ["-c:v", "mjpeg", "-q:v", "2", "-pix_fmt", "yuvj420p"],  # plays in PowerPoint / Windows players
    ".webm": ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", "-pix_fmt", "yuv420p"],
}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MOVIE_SUFFIXES = (".gif",) + tuple(VIDEO_CODECS) + IMAGE_SUFFIXES


def ffmpeg_executable() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def sequence_pattern(filename: str) -> str:
    """'frames/img.png' -> 'frames/img_{:04d}.png'; '%04d' and '{...}' patterns are kept."""
    name = str(filename)
    if "{" in name:
        return name
    if "%" in name:
        import re

        return re.sub(r"%0?(\d*)d", lambda m: "{:0" + (m.group(1) or "1") + "d}", name)
    root, ext = os.path.splitext(name)
    return f"{root}_{{:04d}}{ext}"


class MovieWriter:
    """Append RGB(A) uint8 frames; the format follows the file extension."""

    def __init__(self, filename, fps: float = 10, background=(255, 255, 255), loop: int = 0):
        self.filename = str(filename)
        self.ext = os.path.splitext(self.filename)[1].lower()
        if self.ext not in MOVIE_SUFFIXES:
            raise ValueError(f"unsupported movie format {self.ext!r}; use one of {', '.join(MOVIE_SUFFIXES)}")
        self.fps = float(fps)
        self.background = np.asarray(background, float)
        self.loop = loop
        self.count = 0
        self._gif = []
        self._proc = None
        if self.ext in VIDEO_CODECS and ffmpeg_executable() is None:
            raise RuntimeError(f"{self.ext} export needs ffmpeg: pip install 'ewald[movie]' "
                               "(or install ffmpeg), or write a .gif or PNG sequence instead")

    def _flatten(self, img):
        img = np.asarray(img)
        if img.shape[-1] == 4:  # composite transparency over the background color
            a = img[..., 3:4] / 255.0
            img = (img[..., :3] * a + self.background[:3] * (1 - a) + 0.5).astype(np.uint8)
        return np.ascontiguousarray(img[..., :3], dtype=np.uint8)

    def append(self, img):
        from PIL import Image

        img = np.asarray(img, np.uint8)
        if self.ext in IMAGE_SUFFIXES:
            keep_alpha = img.shape[-1] == 4 and self.ext in (".png", ".tif", ".tiff")
            Image.fromarray(img if keep_alpha else self._flatten(img)).save(
                sequence_pattern(self.filename).format(self.count))
        elif self.ext == ".gif":
            self._gif.append(Image.fromarray(self._flatten(img)).quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                                                          dither=Image.Dither.NONE))
        else:
            rgb = self._flatten(img)
            h, w = rgb.shape[:2]
            if self._proc is None:
                self.size = (w - w % 2, h - h % 2)  # yuv420p needs even dimensions
                cmd = [ffmpeg_executable(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{self.size[0]}x{self.size[1]}", "-r", f"{self.fps:g}", "-i", "-", "-an",
                       *VIDEO_CODECS[self.ext], "-r", f"{self.fps:g}", self.filename]
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            rgb = rgb[:self.size[1], :self.size[0]]
            if rgb.shape[:2] != (self.size[1], self.size[0]):
                raise ValueError("all movie frames must have the same size")
            self._proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
        self.count += 1

    def close(self):
        if self.ext == ".gif" and self._gif:
            self._gif[0].save(self.filename, save_all=True, append_images=self._gif[1:], loop=self.loop,
                              duration=max(20, int(round(1000 / self.fps))), disposal=1, optimize=False)
            self._gif = []
        if self._proc is not None:
            self._proc.stdin.close()
            err = self._proc.stderr.read().decode(errors="replace")
            if self._proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.filename}: {err.strip()}")
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def render_animation(pipeline, filename, *, frames=None, fps: float = 10, device=None, text=None,
                     time_per_frame: float | None = None, progress=None, **kwargs):
    """Render frames of a pipeline into ``filename`` and return the number of frames written.

    Formats by extension: .gif, .mp4, .avi, .mov, .webm, or .png/.jpg/.tif for a
    numbered sequence (``out.png`` -> ``out_0000.png``; ``frame_%03d.png`` also works).
    ``frames``: iterable of frame indices (default all). ``text`` is a label template
    expanded per frame (``"t = [Time] ps"``, see :func:`ewald.overlays.format_label`)
    or a callable ``f(frame, data) -> str``. ``progress(i, n)`` is called after each
    frame; returning False stops early (the file written so far is kept).
    Other keywords go to :func:`ewald.render.render` (size, camera, renderer, ``quality``
    ='draft'|'standard'|'publication', ...); 'draft' renders half-size frames for quick previews.
    """
    from .overlays import format_label
    from .render import _rgb, render

    idx = list(range(pipeline.num_frames) if frames is None else frames)
    bg = (np.asarray(_rgb(kwargs.get("background", "white"))) * 255).astype(int)
    if os.path.splitext(str(filename))[1].lower() not in IMAGE_SUFFIXES:
        kwargs["transparent_background"] = False  # only image sequences keep alpha
    kwargs.pop("filename", None)
    written = 0
    with MovieWriter(filename, fps=fps, background=bg) as writer:
        for n, f in enumerate(idx):
            data = pipeline.compute(f, device=device)
            label = text(f, data) if callable(text) else format_label(text, f, data, time_per_frame)
            writer.append(render(data, None, text=label or None, **kwargs))
            written += 1
            if progress is not None and progress(n + 1, len(idx)) is False:
                break
    return written
