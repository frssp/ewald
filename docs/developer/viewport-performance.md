# Large structures in the 3D view

The goal (D73) is a 3D view that turns a 1,000,000-atom structure at 30 frames per second and
switches trajectory frames in about 0.1 s, on the owner's Intel Arc laptop GPU (OpenGL 3.2 core
profile through VTK) as well as on NVIDIA and Apple GPUs. The interactive view puts speed first;
image export (render presets, ray tracer) is not part of this work.

## Measuring

```bash
python benchmarks/viewport.py                                   # 108k and 1M atoms, bonds off and on
python benchmarks/viewport.py --atoms 108000 1000000 4000000 -o viewport.json
xvfb-run -a -s "-screen 0 1920x1200x24" python benchmarks/viewport.py --atoms 108000   # container
```

The script opens the GUI's own `Viewport` in a window and draws a jiggled FCC Cu crystal (Ball &
stick, bonds up to 2.9 A, 12 per atom), so every machine draws the same scene. It reports the
first build and render, frame times while the camera turns (full detail and the coarse level of
detail used while dragging), frame switches of the same atoms (file reading not included; see
[File readers](readers.md)), memory and the OpenGL driver. Each render waits for the GPU
(`vtkRenderWindow.WaitForCompletion`), so the frame times are real. `-o` writes JSON; keep the
files from real GPUs next to the numbers below. On a laptop with two GPUs, `ewald gpu` says
which one Python uses ([GPUs](../user/gpu.md)).

The cloud container renders with Mesa llvmpipe, a CPU rasterizer: its render times are many
times those of a real GPU and only show whether the CPU side (bond search, scene building,
uploads) got faster.

## Where the time goes

Measured on 2026-10-26 in the container (4 cores, llvmpipe, 1280x800), before the work below:

| step | 108k atoms (1.3M half-bonds) | 1M atoms (12M half-bonds) |
| --- | --- | --- |
| bond search, new frame | 0.3 s | 2.3 s |
| `scene.build` (half-bonds, colors, caps) | 0.5 s | 3.8-6.7 s |
| `draw_scene` (`stick_data`, actors) | 0.17 s | 3.8 s |
| first render (upload) | 2.6 s | 15 s |
| one rotated frame with bonds (llvmpipe) | 1.9 s | 17 s |
| one rotated frame, atoms only (llvmpipe) | 0.26 s | 1.0 s |
| frame switch with bonds, no file read | 1.0 s | 17 s |
| frame switch, atoms only, no file read | 0.25 s | 1.9 s |

- Atoms are GPU sphere impostors (one vertex each) and bonds GPU stick impostors (D43), so
  atoms alone are close to the goal. Bonds are not: every bond is two half-bonds, and VTK's
  stick mapper expands each into a box of vertices on the CPU, about 580 bytes of buffer per
  half-bond (7 GB at 1M atoms).
- `Viewport.redraw` removes and rebuilds every actor on each frame, and the bond search of a
  new frame runs on the GUI thread.
- Reading a 1M-atom LAMMPS dump frame takes 0.45 s, so a jump to an arbitrary frame cannot
  take 0.1 s; playback and scrubbing can read ahead.

After phases 1-2, `python benchmarks/viewport.py --atoms 108000 1000000` in the same container:

| case | before | after |
| --- | --- | --- |
| 108k atoms with bonds: sticks | 1.3M | 0.67M |
| 108k with bonds: frame switch on the GUI thread | 2.7 s | 1.7 s (0.3 s search on the worker) |
| 108k with bonds: one rotated frame (llvmpipe) | 1.8 s | 1.2 s |
| 1M atoms only: `set_data` of a new frame | 0.36 s | 0.08 s |
| 1M with bonds: sticks | 12M | 6.1M |
| 1M with bonds: peak memory | 8.6 GB | 2.4 GB |
| 1M with bonds: frame switch on the GUI thread | 17 s | 13 s (4.4 s `set_data` + 8.5 s llvmpipe render; 4 s search on the worker) |

The 4.4 s of `set_data` at 1M atoms with bonds is the scene build and stick input on the GUI
thread; building the `Scene` on the compute worker as well (with the view's settings taken
when the frame is requested) would leave only the actors and the upload there.

## Plan

| phase | what | state |
| --- | --- | --- |
| 0 | `benchmarks/viewport.py` | done |
| 1 | bond search and reading ahead off the GUI thread; numba kernels release the GIL; faster color legend | done |
| 2 | lighter bonds: half-bond geometry, caps and stick input in numba, one stick for a bond whose atoms share a color, a GPU memory budget instead of an out-of-memory driver | done |
| 3 | an own GPU bond renderer: positions in a texture buffer, bonds as atom index pairs, both colors in the shader | after the Arc numbers |
| 4 | culling (blocks off screen, hidden interior atoms of opaque bulk) and distance level of detail | after the Arc numbers |

Phase 1 also measured whether frame switches should keep the actors and replace only their
arrays: for 1M atoms without bonds, `Viewport.set_data` (scene and actors) takes 0.08 s once the
color legend no longer loops over the atoms in Python, against 0.9 s of llvmpipe render, and a
reused actor uploads the same buffers. So actors are still rebuilt; the time with bonds is in
the half-bond geometry and stick input, which phase 2 takes on.

The main window's compute worker (`gui/compute.py`) now also searches the bonds the 3D view will
draw (`MainWindow._view_prepare`), so the GUI thread finds them in `bonds.cached_bonds`, and during
playback it computes the next frame (`Timeline.next_frame`) into the pipeline's cache while the
current one is drawn. The neighbor-list kernels are compiled with `nogil=True`, so the window keeps
painting while the worker searches.

Phase 2 builds the half-bonds (`bonds.bond_segments`, `bonds.halves_meet`) and the stick input
(`render.stick_data`) in parallel numba kernels and skips the copies when every half is kept.
`Cylinders.partner` pairs the two halves of a bond that meet, and the viewport draws a pair of
the same color, radius and opacity as one stick from atom to atom: the same picture (the
`impostor_bonds` visual reference did not change), and half the sticks for one element (12M
half-bonds of 1M Cu atoms become 6.1M sticks). Above `viewport.max_bond_sticks` (8M by default,
about 4.6 GB) the view leaves the bonds out before the upload and the window says so.

Constraints for every phase: OpenGL 3.2 core features only (texture buffers, instancing and
geometry shaders, no compute shaders or SSBOs; macOS stops at 4.1); bond ends look the same in
every renderer (D50, `tests/test_visual.py`); the OpenGL 1.x viewport for `ssh -X` keeps working
but gets none of this; `DataCollection` arrays are still never changed in place (only the VTK
arrays are).
