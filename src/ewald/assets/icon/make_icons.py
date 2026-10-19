"""Regenerate the EWALD app icon files from ewald.svg and ewald-small.svg.

    pip install cairosvg pillow
    python src/ewald/assets/icon/make_icons.py

The SVGs are the sources; the PNG sizes, ewald.ico (Windows) and ewald.icns (macOS) are
rendered from them, never edited by hand. Sizes up to 32 px use ewald-small.svg, whose
heavier marks stay readable in title bars and taskbars.
"""
from __future__ import annotations

import io
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIZES = (16, 24, 32, 48, 64, 128, 256, 512, 1024)
SMALL = 32  # this size and below come from ewald-small.svg


def render(size: int):
    import cairosvg
    from PIL import Image

    src = HERE / ("ewald-small.svg" if size <= SMALL else "ewald.svg")
    png = cairosvg.svg2png(url=str(src), output_width=size, output_height=size)
    return Image.open(io.BytesIO(png)).convert("RGBA")


def main():
    images = {s: render(s) for s in SIZES}
    for s, im in images.items():
        im.save(HERE / f"ewald-{s}.png", optimize=True)
    ico = [16, 24, 32, 48, 64, 128, 256]
    images[256].save(HERE / "ewald.ico", sizes=[(s, s) for s in ico],
                     append_images=[images[s] for s in ico if s != 256])
    images[1024].save(HERE / "ewald.icns")


if __name__ == "__main__":
    main()
