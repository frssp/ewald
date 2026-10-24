# Design decisions

The decisions that shaped EWALD, who made them and why. Most of EWALD's code is written by an AI
coding assistant (Claude Code) working from these decisions; the decisions themselves, the
scope and the acceptance of each change are the maintainer's (Sunghyun Kim, "the owner" below).
This page is the record of that human design work, newest last within each section.

Each entry gives the date, the decision, why, what else was considered and where it landed.
Quotes are the owner's own words (often Korean) from the project discussion. Where the owner's
reason was not written down, "Why" is left out rather than guessed. How to add an entry is at
the end.

Pull request numbers up to #242 refer to the pre-publication repository, which is kept private; the public repository's numbering starts again at #1 (decision D32), so its pull requests are written "public PR #N" here.

## Product

### D1. Target users and scope (2026-10-18, 2026-10-19)

- **Decision:** a structure and trajectory workbench for computational materials researchers
  who run VASP and LAMMPS (often on a cluster), at the analysis and figure level of OVITO Pro.
  Not a biomolecular viewer, not a workflow engine, not a web service.
- **Why:** the owner's own workflow: raw output, analysis and a paper figure today need three
  tools and leave no record of what was clicked.
- **Landed:** README "The idea" (PR #100).

### D2. The notebook is the identity (2026-10-19)

- **Decision:** the working unit is a Jupyter-like notebook ("scripts + Python visualization is
  our identity"): cells of Python next to a live 3D view, saved as a percent-format `.py` that
  also runs as a script; several notebooks as top tabs instead of modes; plugins loaded from
  cells.
- **Why:** the first GUI looked and worked like OVITO; a notebook makes every analysis
  reproducible and readable, and gives the GUI, scripts and the AI one shared API.
- **Considered:** five layout proposals (Stage, Notebook, Workbench and two hybrids); the owner
  picked Notebook + Workbench, then made the notebook the identity.
- **Landed:** PR #63 (notebook layout), `.ewd` bundle format (owner's spec: notebook, manifest,
  figures, data, plugins; Jupyter round trip).

### D3. No OVITO-style interface (2026-10-20)

- **Decision:** remove the "classic" layout (pipeline list + properties panel) and never bring
  it back ("classic 지우고 배제하자"). Display settings live in the View tab, source details in
  the Data inspector.
- **Landed:** PR #115.

### D4. AI first, and only through notebook cells (2026-10-19, 2026-10-20)

- **Decision:** an assistant is built in and drives the same notebook API as the GUI and
  scripts. It may change structures, views or results only by inserting, editing and running
  notebook cells; read-only tools are fine ("셀로 바꾸기").
- **Why:** the notebook stays the single, reproducible record of what the AI did, and the user
  can read and undo every change.
- **Considered:** direct pipeline/view edits by tools, and a free `run_python` tool (both
  rejected for the reason above).
- **Landed:** PR #161 (`CELL_WRITERS`, enforced by `tests/test_llm.py`).

### D40. Plugin tools that change results outside cells are undone (2026-10-24)

- **Decision:** after every AI plugin tool, ewald undoes steps and view settings it changed
  without a cell and fails the call; `export_file` makes missing folders inside the cell it
  runs. The other places that act outside cells (`render_image` files in the AI's work dir,
  `set_frame`, Forge plugins) stay, documented in docs/user/ai-permissions.md.
- **Why:** after the list of what the AI does outside cells, the owner asked what had to be
  blocked ("뭘 막아야하지? 위험?") and, of the five, approved fixing the plugin tools and
  `export_file` ("진행하고 문서 정리").
- **Considered:** only documenting the gaps; a warning without undoing the change.
- **Landed:** public PR #21.

### D31. The AI's API reference is complete in one read (2026-10-23)

- **Decision:** `python_api` (and `llms.txt`, the same text) gives the whole reference in one
  call, about 60 KB: notebook cells first, tested recipes, generated function signatures, and
  every modifier with the outputs and units from its docstring, grouped by family.
- **Why:** the owner asked for a better llms.txt ("llm.txt 를 잘 써야겠네") and, offered three
  sizes, picked the full reference in one read.
- **Considered:** splitting it by topic (`python_api(topic=...)`), cheaper per call but the
  model may skip the part it needs; keeping the old size (about 36 KB) with only a reordering.
- **Landed:** PR #242.

### D5. Any AI provider, named "Ask AI" (2026-10-20)

- **Decision:** not Claude-only: Anthropic by default, OpenAI, Gemini, OpenAI-compatible local
  servers, and plugin providers for API gateways; keys in the OS keychain, never in settings or
  the environment.
- **Why:** the owner needed to reach models through an institutional API gateway, not only
  public services.
- **Landed:** PR #116.

### D21. AI answers link the cells they changed (2026-10-23)

- **Decision:** when the AI writes, edits or runs a cell, its answer in the chat links to that
  cell (a click shows the notebook tab and lights the cell up). The links come from the tools'
  results, not from the model's wording, and cells are named as the user counts them (from 1).
- **Why:** the owner saw an answer saying it changed "the view cell (index 5)" and noted
  "어느 셀을 수정했는지 링크가 없다" (there is no link to which cell was changed).
- **Landed:** PR #228 (`notebook_tools.touched_cells`, the chat's cell links).

### D35. The notebook follows the AI's cells until you scroll (2026-10-23)

- **Decision:** while the AI chat works, reading cells never scrolls the notebook; a cell it
  writes or runs is scrolled to only when it is off screen, and once the user scrolls the
  notebook during that answer it stops following and a pill at the notebook's edge ("AI is on
  cell 7") jumps there instead. Scrolling once to the last cell at the end, or never, are
  Preferences › AI › Follow the AI's cells (`llm.follow_cells`).
- **Why:** the owner found the notebook "너무 왔다갔다" while the AI read and wrote cells and
  asked to weigh not moving at all, moving less, or going to the last edited cell at the end
  ("또는 다른 방식일지 고민해줘"); from these options the owner picked following while
  respecting the user's scroll. (The jumping itself was a bug: rebuilt cells passed the
  keyboard focus from cell to cell and the scroll area followed each.)
- **Considered:** never scrolling (cell links and the pill only); scrolling once to the last
  edited cell when the answer is done. Both stay as settings.
- **Landed:** public PR #5.

### D6. Structure editing is deferred (2026-10-19)

- **Decision:** no supercell, slab or defect builder until the owner asks.

### D36. Activity bar: work on top, management at the bottom (2026-10-24)

- **Decision:** the activity bar lists the side bars you work in on top (Notebook, AI chat,
  plugin side bars such as Workflows, Remotes) and Plugins plus a new Preferences button at the
  bottom; the Window menu follows the same order and every tooltip names its shortcut. New
  line icons: a notebook page with a `>_` cell, a chat bubble with a sparkle for AI (it looked
  like a loading spinner and like the settings icon), a server for Remotes (a monitor read as
  "display"), a flow chart for Workflows (instead of the layers icon of Replicate cell),
  blocks for Plugins and a gear for Preferences.
- **Why:** the owner asked to review the bar's icons ("왼쪽 바에 아이콘들 재검토"); from a sheet
  of options in both themes the owner picked the regrouped set B with the notebook-cells icon
  ("B + notebook cells") and the blocks icon for Plugins ("Plugin도 blocks").
- **Considered:** fixing only the misread icons and keeping the order (A); a sparkle-only AI
  icon (C); keeping the page-with-lines notebook icon and the puzzle piece.
- **Landed:** public PR #11.

### D38. A full activity bar ends in a "More side bars" menu (2026-10-24)

- **Decision:** when the activity bar's top group does not fit the window's height, its last
  slot becomes a "More side bars" (…) button listing the rest; an open side bar keeps its
  button, and Plugins and Preferences at the bottom always stay. A right-click on the bar hides
  or shows buttons (saved as `general.activity_hidden`); hidden side bars stay in the Window
  menu and the command palette.
- **Why:** the owner asked what happens when side bar icons become many or the window is short
  ("사이드바 아이콘이 아주 많아지면 어쩌지? 창이 작아서 밑으로 밀려내려가면?"); Qt cut the bottom
  group first and hid the rest behind a nearly invisible arrow. From three options the owner
  picked the VS Code-style overflow with a fixed bottom group and hiding.
- **Considered:** the overflow menu and fixed bottom group without hiding; scrolling the bar.
- **Landed:** public PR #15.

### D44. One icon set for the whole GUI (2026-10-24)

- **Decision:** the Preferences page list draws its icons from the same set as the activity
  bar, toolbars and menus (`gui/icons.PATHS`); the dialog's own icon table is gone, so Notebooks,
  AI and Plugins look the same in the bar and in Preferences.
- **Why:** the owner asked "왼쪽 바 아이콘과 설정 안에 있는 아이콘 동일하게. 같은 이미지 사용하나?";
  they did not (Preferences had its own book, sparkle and plug icons).
- **Landed:** public PR #29.

## Quality and platforms

### D7. Quality for files, speed for the screen (2026-10-19)

- **Decision:** rendered images, vector figures and movies put quality first (Draft, Standard,
  Publication presets; Publication by default); the interactive viewport puts speed first (level
  of detail, target frame rate, cache budget). Both are settings.

### D8. Windows, macOS and Linux behave the same (2026-10-21)

- **Decision:** EWALD must behave nearly identically on all three; tests simulate the other
  platforms where they can.
- **Why:** the owner runs EWALD on Windows while development containers are Linux; a Windows-only
  path bug (NEB reader splitting on "/") reached the owner.
- **Landed:** `docs/developer/portability.md`, `tests/test_portability.py`.

### D9. Notebooks do not run when opened (2026-10-21)

- **Decision:** opening a notebook shows its cells and saved outputs without running them
  ("노트북 열 때 무조건 바로 실행하지 말기"); running on open is an opt-in setting.
- **Landed:** PR #190 (`python.run_notebooks`, default "never").

### D10. Plots inline, linked plots in a panel (2026-10-20)

- **Decision:** plots appear under their cell like Jupyter; plots that follow the frame or are
  linked to the 3D view (histograms, g(r), time series, NEB, DOS/bands, phonons) open as Plots
  panel tabs with a thumbnail under the cell.
- **Landed:** PRs #183, #184.

### D25. Menus grouped by role, one word for one thing (2026-10-23)

- **Decision:** the menu bar follows roles: File opens (Open recent, Open example, Open from ›
  Remote file, Location, NEB folder, Workflow folder) and exports (Export › Image, Data, Movie,
  Session); Edit edits cells; View is the camera and what is drawn (Look along, Show, Structure
  theme, Layout, Full screen); Insert adds cells (Step with its Plugins, Plot, Quick analysis);
  Notebook runs them; Tools are the structure view's tools and the command palette; Window ticks
  the side bars (plugin side bars with the built-in ones) and lists the tabs; Help adds Tutorials
  and Keyboard shortcuts. Names say one thing each: plot cells are "Plot" (not "View"), layout
  moves act on a "tab", the structure color preset is "Structure theme" (the UI theme stays in
  Preferences), and the 3D view is the "structure view". Preferences pages become General, 3D view,
  Atoms & bonds, Rendering, Plots, Notebooks, Compute, Python, AI and Plugins, with plugin pages after
  Plugins and Edit › Plugin settings removed. Shortcuts in menus and tooltips are written the
  platform's way (⌘ on macOS), every menu entry has a mnemonic, Ctrl+N joins Ctrl+T for a new
  notebook. Existing shortcuts stay, except Ctrl+Shift+O of the legacy session file.
- **Why:** asked for a review of every menu "UX/UI디자이너의 관점으로" (from a UX/UI designer's
  point of view); from the review's options the owner picked applying all of them (A–F) over only
  the menu bar and naming, only platform conventions and labels, or only the small fixes.
- **Considered:** keeping the menu structure and fixing labels only; leaving the reorganization
  in the TODO list. The View tab keeps its name (the owner chose it in PR #127).
- **Landed:** PR #227 (small fixes found by the review), PR #235 (the reorganization); review
  report with screenshots in the project files (`ewald-menu-review/`).

## Architecture

### D11. A small core, everything else a plugin (2026-10-19, 2026-10-20)

- **Decision:** VASP, LAMMPS and the common formats are built in but registered as plugins that
  can be switched off; domain packages (phonopy, PLUMED, AiiDA) live in frssp/ewald-plugins;
  plugins declare themselves in a manifest (`ewald-plugin.toml`), as in VS Code.
- **Landed:** PRs #14, #22 and later; plugin index in frssp/ewald-plugins.

### D26. Recommend a restart when a plugin change needs one (2026-10-23)

- **Decision:** when installing, updating, switching on or removing a plugin only takes effect
  after a restart, ewald says so and offers *Restart now* (which reopens the saved notebooks);
  everything that can take effect at once does (removing a single-file plugin no longer waits
  for a restart).
- **Why:** "플러그인 중 재시작이 필요한 것이 있으면 재시작을 추천해줘" (if a plugin needs a
  restart, recommend restarting).
- **Landed:** PR #229 (Plugins side bar restart bar, `plugin_catalog.restart_reason`).

### D27. VASP phonons back in the core (2026-10-23)

- **Decision:** the part of phonon support that needs no phonopy (VASP OUTCAR of IBRION=5-8,
  phonopy band.yaml/mesh.yaml/qpoints.yaml, the `PhononMode` animation) moves back into the
  core as the built-in plugin group "Phonons"; the ewald-phonopy plugin keeps what needs the
  phonopy package (phonopy.yaml with forces or force constants, `from_phonopy`). This partly
  reverses D11 for phonons.
- **Why:** the owner picked "VASP 부분만 core로" (only the VASP part into the core) from a
  review of the plugins; the recommendation it answered: VASP is a core format, and this part is
  about 500 lines with no dependency beyond numpy, so VASP users should not need a plugin for it.
- **Considered:** keeping all phonon support in the plugin (as decided on 2026-10-19).
- **Landed:** PR #233; ewald-phonopy 0.3.0 in frssp/ewald-plugins.

### D12. Reuse before writing (2026-10-18)

- **Decision:** port verified analyses (CNA, PTM, Voronoi, Wigner-Seitz, grain segmentation) from
  OVITO's source under its MIT option and test them against OVITO and freud results; use
  OVITO's property names so OVITO users feel at home; DXA through the optional `ovito` package
  instead of a new implementation. VMD code is not reused (license).

### D13. One compute code for CPU and GPU (2026-10-18)

- **Decision:** kernels are written once against NumPy/CuPy; NVIDIA CUDA first, the backend
  pluggable for Apple Metal later; CPU fallback on macOS.
- **Why:** tests without a GPU still cover the GPU code path.

### D14. Nothing runs until trusted (2026-10-19)

- **Decision:** a notebook, `.ewd` bundle, session or text inside a structure file (prompt
  injection) cannot run code or steer the AI into running code until the user trusts it.
- **Landed:** PR #46 and later; `src/ewald/trust.py`, `tests/test_security.py`.

### D17. Cells run in IPython, not a hand-written engine (2026-10-22)

- **Decision:** notebook cells run through IPython's shell (one per notebook), so magics,
  `!commands`, `display()` and rich outputs (HTML, Markdown, LaTeX, SVG) work as in Jupyter, and
  the console shares the notebook's namespace; the percent-format `.py` reader and writer stay
  ewald's own (jupytext-compatible, tested against jupytext) and cells keep running in the app's
  Python by default. A running cell shows its output while it runs, and long work (downloads of
  example data) shows a progress bar under the cell (tqdm-based `ewald.progress`).
- **Why:** "주피터를 의존성으로 안가지면 바퀴 재발명 아니야?" (isn't it reinventing the wheel
  without Jupyter as a dependency?); for downloads, "언제 끝날지 몰라 멈춘 것 처럼 보여" (you
  can't tell when it ends, so it looks frozen).
- **Considered:** adding only rich outputs on the existing engine; leaving it as it was. Running
  every notebook in a separate Jupyter kernel process by default was not proposed, since the 3D
  view shares the pipeline's arrays with the cells.
- **Landed:** PR #205 (`src/ewald/shell.py`, `src/ewald/progress.py`).

### D18. Analyses of a workflow are notebook cells from its plugin (2026-10-23)

- **Decision:** a workflow source can offer analyses (`Analysis`: notebook cells plus the
  variable they leave); the opened notebook includes them and the AI adds them with
  `add_analysis`. The first is the potential lineup of sim-agent's band-offset workflow, computed
  in EWALD from the retrieved `locpot_planar_avg.json` and `vasprun.xml` of the interface and the
  two bulk references (bulk VBM/CBM moved by the plateau shift, E_F from the interface), with
  the kernels in core (`ewald.potential`) so other sources can reuse them, and without changing
  sim-agent. Recipes are per template, so work function, EOS and formation energies follow the
  same way.
- **Why:** "ewald에서 notebook으로 처리해야지" (handle it in an EWALD notebook); "sim-agent 수정은
  안 하는게 좋아." (better not to modify sim-agent); "이걸 다른 계산에도 적용할 수 있도록 생각하면서
  일해." (work so that this applies to other calculations too); and the planar average json is
  enough, LOCPOT is not needed.
- **Considered:** having sim-agent store its finalize result and serve it through the results
  API; the AI computing the alignment itself from the numbers; reading LOCPOT over SSH.
- **Landed:** PR #217 (`src/ewald/potential.py`, `ewald.workflow.Analysis`, `add_analysis`), with
  the recipe in frssp/ewald-simagent.

### D19. Restarts inside one dump: a default rule the user can change (2026-10-23)

- **Decision:** when several files or one appended LAMMPS dump are joined on Timestep, a step
  that goes back inside a file is a restart (the later run wins) unless it goes back to or below
  the first step of the run before it, which counts as `reset_timestep` and cuts nothing. That
  rule stays the default, and the user can change it: `join="restart"` (every step going back is
  a restart), `join="files"` (join only between files) or `join="concat"`, in `load(...)` and in
  the dialog that opens several files.
- **Why:** "기본 동작은 이대로 하고 사용자가 조절할 수 있게 하면 되지 않을까?" (keep this as the
  default and let the user adjust it).
- **Considered:** one fixed rule with no option.
- **Landed:** PR #212 (default rule, join dialog), PR #218 (the `join` options).

### D20. Plugin settings dialogs stay short (2026-10-23)

- **Decision:** in the plugin settings dialog a field's help is its tooltip, not a line under it,
  and fields a plugin marks `advanced` (CA file, host aliases, connection name) fold under
  **Advanced**, which says how many of them are set. Preferences pages keep their help lines
  and fold the same advanced fields.
- **Why:** "메뉴가 너무 긴데, 좀 위아래로 줄이는 레이아웃 구성 가능해?" (the menu is too long;
  can the layout be shorter top to bottom?), on the sim-agent settings dialog with a diagnosis
  report under it.
- **Considered:** two columns of fields (wide, and the help still needs room); a dialog that
  scrolls (as long, only hidden).
- **Landed:** PR #222 (`gui/plugin_settings.py`, `gui/preferences.py`, `SettingsField.advanced`).

### D28. Computed properties as plugins, with their knowledge as data (2026-10-23)

- **Decision:** material properties beyond the shared primitives become plugins of their own
  kind, each with a manifest (inputs, outputs and units, method, limits, references, checks, a
  reference case) that the AI reads to list and explain them; the knowledge part is a milestone
  for a general plugin about computed properties that other tools can read. Band structure and
  DOS level stays in core.
- **Why:** "band정도면 코어에 넣겠는데 좀 더 복잡한 물성이면 나중에 밖에서 유저가 추가해도 좋겠다"
  (band-level goes into core, more complex properties users may add from outside later), with
  "ai와 함께쓴다는 걸 고려해서"; the knowledge sharing "일반적인 계산 물성에 대한 지식 플러그인으로
  마일스톤에 두자" (as a milestone: a knowledge plugin for computed properties in general).
- **Considered:** keeping per-source recipes with the kernels in core; knowledge as data with the
  AI writing the code each time; sharing the computing code with other tools (only the knowledge
  is shared).
- **Landed:** milestone in docs/ROADMAP.md (Properties).
### D23. Color bars on translucent cards, drawn once for screen and file (2026-10-23)

- **Decision:** a color bar (ColorCoding, `color_by`, trajectory maps, grid slices) is a thin
  rounded ramp with round-number ticks on a small translucent card in the top right corner;
  several bars stand side by side. The 3D view draws the same overlay items as the pictures
  (`ewald.colorbars`), no longer VTK's scalar bar.
- **Why:** on the old viewport bar, "안 예쁘다. 디자인 다시해라" (it is not pretty, redesign it).
  From four options rendered on the Fe vacancy NEB the owner picked the card once it was clear
  it costs nothing: "성능에 영향없으면 B가 좋아" (if it does not affect performance, B is good).
- **Considered:** a slim bar without a card at the right edge (recommended), a horizontal bar in
  the bottom right corner, a framed matplotlib-style bar.
- **Landed:** PR #231 (`colorbars.py`, `overlays.ColorRamp`, `gui/viewport.py`).

### D22. Generic AI tools in the core, source-specific ones in plugins (2026-10-23)

- **Decision:** AI tools that work on any file, pipeline or workflow source belong to the core;
  a plugin adds only what is specific to its source. Searching workflows by formula, status
  and day and opening a step of one became core tools (`list_workflows` filters,
  `open_workflow`) instead of sim-agent plugin tools.
- **Why:** "ewald측에서 일반적인 파일이나 파이프라인 (sim-agent특화 아님) 툴 정도는 갖춰도 좋아"
  (the EWALD side may well have general file or pipeline tools, not specific to sim-agent).
- **Considered:** `simagent_find` / `simagent_open` inside ewald-simagent, which would have
  left local folders and AiiDA without them.
- **Landed:** PR #232 (`src/ewald/workflow.py` `find`/`target`, `src/ewald/llm/tools.py`).


### D24. Python steps in the tutorials use the decorators (2026-10-23)

- **Decision:** the tutorial notebooks write their Python steps as functions under `@pipe.add`
  marked `@each_frame` or `@all_frames`, and use them where a step works per frame or over the
  whole run (surface atoms over the run, energy and forces in the MD tutorial). New notebooks
  import `each_frame` and `all_frames`.
- **Why:** asked whether the examples used a decorator, the owner answered "Each frame 데코레이터
  적용" (apply the each-frame decorator) and asked for other places to use them, with more plots
  where useful.
- **Considered:** `# %% name [each frame]` cell tags, which read more like a notebook in the GUI
  but do not run as a plain `python` script.
- **Landed:** PR #236.

## Project

### D15. Name, license and openness (2026-10-19, 2026-10-20)

- **Decision:** the name "EWALD: Extensible Workbench for Atomic Lattices & Dynamics" (package
  `ewald`); MIT license, copyright Sunghyun Kim and the EWALD contributors; a public repository.

### D16. How the project is developed (2026-10-19)

- **Decision:** an AI coding assistant writes most changes in pull requests; local test runs are
  the merge gate (full suite under a virtual display for GUI and rendering changes, visual
  regression for renderers); each feature PR updates the feature inventory and the TODO list.

### D29. "AI-native", not "Claude built in" (2026-10-23)

- **Decision:** the README and the manual describe EWALD as an AI-native notebook; Claude is the
  default provider of Ask AI, not the product's identity, and the text names the other providers
  (OpenAI, Gemini, local models) where it matters.
- **Why:** the owner asked to replace "with Claude built in" in the README introduction: "claude
  built in 아니고 ai native? 이정도로" (not "Claude built in", something like "AI-native").
- **Landed:** PR #240.

### D30. The README's lead picture is an Ask AI recording of moving runs (2026-10-23)

- **Decision:** the first picture of the README and the manual's home page is
  `docs/images/ai-demo.gif`: Ask AI analysing three runs that move (Li paths and D* in a
  Li6PS5Cl AIMD run, sp/sp2/sp3 carbon through a melt-quench MD run, the barrier of a Fe vacancy
  NEB) by writing and running cells. The recording zooms like a screen recording, not into the
  3D view, and moves little: in on the chat input while the question is typed, then the chat and
  the notebook held still side by side while the AI works, then over to the structure view.
- **Why:** the owner asked for "AI 사용해서 예제 노트북 수정하는 좋은 케이스" as a GIF in the
  README, then: "줌인줌아웃이 화면에서 타이핑하는 또는 노트북 글자 잘보이게 하라는 거였어. 뷰포트가
  아니라" (zoom so the typing and the notebook text read, not the viewport), "대표 그림도 이걸로
  해도 좋아" (use it as the lead picture), and "나노파티클 보다 뭔가 움직이는 것, neb, md 등이
  관심있어" (something that moves, NEB or MD, rather than a nanoparticle); of the options shown
  (Li AIMD, carbon quench, Fe NEB, or joined) the owner picked all of them joined, "AI 써서 분석하는
  케이스로" (as cases of analysing with AI). On the first cut, which panned between the chat and each cell:
  "지금은 화면이 너무 채팅창-노트북 사이를 빨리 왔다갔다하던데?" (it flips between the chat and the
  notebook too fast), after "화면 전환이 너무 많거나 빠르거나 그러면 어지러워. 그럴 경우 화면을 좀 넓게
  써도 좋아" (too many or fast moves make one dizzy; a wider picture is fine). Then "카메라 좋아. 그리고 neb에 traj. color code
  해줘. viridis" (the camera is good; color the NEB trajectory with viridis).
- **Alternatives:** a Pt nanoparticle site classification with a 3D-view zoom (PR #239, replaced);
  a single run.
- **Landed:** PR #241 (pre-publication repository); the calm cut and the viridis NEB path in public PR #2.
- **Update (2026-10-23): a live model instead of the scripted one.** The owner put an Anthropic key
  into the recording environment ("환경변수 추가 완료") so the GIF could be recorded with a real
  model; `make_ai_demo.py --live` does it with the same three scenes and the same calm camera
  (the work time-lapsed, at most 20 s a scene, since a live model works for a minute or two). Shown
  a take with correct numbers but long answers and a failed cell on screen, the owner chose
  "live로 한 번 더 녹화" (record another live take) over using it or keeping the scripted GIF; the
  next take is the one committed. Alternatives: keep the scripted recording; commit the take
  shown. Landed in public PR #9 (model names taken out of the caption and transcript in public PR #10).

### D32. Publish from a fresh repository (2026-10-23)

- **Decision:** before going public, the history was scrubbed (author name unified to Sunghyun
  Kim, session links and the institution's API gateway identifiers removed with `git
  filter-repo`), and the clean `main` was pushed to a new `frssp/ewald`; the old repository was
  kept private as an archive. The commits stay; pull requests #1-#242 and their reviews do not
  carry over.
- **Why:** GitHub keeps pull request refs pointing at the old commits, which a force push cannot
  remove. Asked whether to delete the repository and create it again, the owner went ahead with
  it ("지금 github에서 저장소를 지우고 다시 ewald를 만들면 되나?").
- **Alternatives:** asking GitHub Support to purge the pull request refs (keeps the pull
  requests, delays publication); publishing with the refs as they were.

### D33. The app icon is an ammonia molecule over an orange chat bubble (2026-10-23)

- **Decision:** the app icon is NH3 (an Azure N, three H and a glass lone-pair lobe) over an
  orange chat bubble on a Graphite tile; the molecule is larger than the bubble and overlaps it.
  Sizes up to 32 px use a variant with heavier bonds. It replaces the teal Ewald sphere of PR #54
  (old repository).
- **Why:** the owner disliked the teal icon ("아이콘이 맘에 안 들어") and asked to explore widely
  first: "너무 완성작을 만들지 말고 한 15가지 정도 디자인 컨샙만 보여주고 거기서 선택한 다음
  다듬어 완성하자". From 15 rough concepts the owner kept a code prompt with an atom, a chat bubble
  with a molecule, a phonon wave and a periodic box, then narrowed to the prompt and the chat
  bubble, asked for "NH3 lone pair" with a large teardrop lobe drawn like a textbook sp3 orbital
  and no electron dots, let the molecule overflow the bubble ("분자가 메세지 창이 넘어가도
  괜찮은데", 1.4 times larger), asked for a clearer and then a bolder bubble colour and picked
  orange, and picked a glass lobe ("L1 glass") over Azure, nested shells and a stippled lobe.
- **Considered:** a code prompt `>` next to an atom on a baseline; the four sp3 lobes alone;
  slate, white, steel blue, violet and raspberry bubbles; a lit atom, a unit cell, a diffraction
  pattern, lattice planes and a 3D Ewald sphere (drafts in the project files `ewald-icons-*`).
- **Landed:** public PR #3.

### D34. The first start after an install or update shows a notice (2026-10-23)

- **Decision:** `ewald gui` shows a small "Preparing EWALD" notice on the first start after
  EWALD, Python or one of its big libraries was installed or updated, from right after PySide6
  loads until the main window is on screen. Later starts show nothing.
- **Why:** on the owner's Mac the first start after an update was very slow and the next was
  fast ("아니다 지금 다시 여니 빠른데?!"); the owner asked "첫 실행은 느리다고 뭔가 노티스를
  띄울 수 있어?".
- **Considered:** leaving the first start silent; a splash screen on every start.
- **Landed:** public PR #7.

### D37. A workflow row shows the group it came from (2026-10-24)

- **Decision:** a workflow source may give each workflow a `group` (the session, project or
  folder it was submitted from, such as the name of the chat session that submitted a
  calculation), and the workflow side bar draws it dimmed under the row's title, on every row.
- **Why:** the owner asked whether the plugin side bar could show the chat session name of each
  calculation ("플러그인에서 채팅 세션 이름을 보여줄 수 있어?") and picked a name per row.
- **Considered:** grouping the side bar under one header per session; showing the name only in
  the row's tooltip; the name after the title on the same line (cut off in a narrow side bar).
- **Landed:** public PR #12.

### D39. Plot settings: a matplotlib style file first, a form over it (2026-10-24)

- **Decision:** the user's plot settings are one matplotlib `.mplstyle` file
  (`<config dir>/plots/user.mplstyle`) drawn over the publication style. Preferences › Plots
  shows its common keys as a form that loads the file when it opens and writes back only the
  rows that changed; *Open as text* edits the whole file in the OS text editor. The Plots panel
  shares the file with notebook cells, kernels and the console.
- **Why:** the owner asked "Matplotlib 을 포함한 plot설정. 설정에 폼을 쓸까? 아님 (기본) 설정
  파일을 텍스트 에디터로 열어서 수정?", picked the hybrid, and set the rule "텍스트 파일 우선으로
  적용. 폼은 열때 파일 로드. 폼 저장하면 파일 업데이트."; and asked "Plot panel도 matplotlib rc 설정
  공유가능?".
- **Considered:** a form only (most rcParams out of reach); a text file only (no help for the
  common keys); TOML or settings.json instead of `.mplstyle` (not usable by `plt.style.use` in
  scripts outside EWALD).
- **Landed:** public PR #19.

### D42. Plot style presets named after cities, a short form (2026-10-24)

- **Decision:** the style file ships seven presets on one slide scale (14 pt type, 2.5 pt lines)
  that differ in look, named after cities and never after journals: Seoul (open axes, vivid
  colorblind-safe colors; what a new style file starts from), London (serif in a box, a line
  style per series), Barcelona (sand plot area, white grid, no frame lines), Boston (dashed
  grid, framed legend), Helsinki (thin gray axes, cool blues), Dongtan (dark background, neon
  lines) and Samlangjin (ticks inside on four sides with minor ticks, a dotted row grid, 3.5 pt lines). Your own presets can be saved from the form (`plots/<name>.mplstyle`) and are listed
  after the built-in ones; the Preferences form shows the preset and four keys (font, font
  size, line width, color cycle), everything else is in the file, which lists its keys by
  topic. The separate "Saved plot font size" preference is retired (the file's `font.size`).
- **Why:** the owner asked "기본 설정파일도 좀 꾸며줘 니가 보기 좋게. 그리고 프리셋 넣고. 그리고
  폼에서 설정할 수 있는 파람이 넘 많은 듯." and, on journal-named presets, "저널이름 쓰지마.
  특징으로 아니면 적당한 이름으로"; the first set (sizes and fonts only) was sent back with
  "Distinction 이 부족. 논문 퀄리티. 스타일 참고해서 다시", and the second with "Publication 이름
  별로 그냥 다른 이름. Slides를 기본으로. 모노크롬 별로 삭제. 다른 것 하나 만드렁봐", "선 굷기랑 글자
  크기를 좀 키우면 어때?" and "프리셋 저장도 되지?". The owner then set the final set: "슬라이드
  기준으로 도시이름 베리에이션 프리셋 / Seoul London Barcelona Boston Helsinki Dongtan", and after seeing them "Good Samlangjin 추가", then "색 팔레트 변경" and "선도 두껍게" for it.
- **Considered:** presets named after journals; presets that differ only in column width and
  type size; presets named by their look (Slides, Boxed, Classic serif, Grid, Gray panel), each
  at its own scale; Monochrome and Minimal presets (dropped: "모노크롬 별로 삭제"; Minimal looked
  like a small Slides, then "아니 제거"); eight form rows (tick direction, minor ticks, legend
  frame, axes line width).
- **Landed:** public PR #20.

### D41. Analysis plots draw their levels; one value per surface is a bar (2026-10-24)

- **Decision:** a table's energy levels, plateaus and barriers (VBM, CBM, E_F, E_vac, Φ) are
  drawn over its curves, and a table with one value per named row (one work function per
  surface) is drawn as one bar per row, not as a line through the row numbers.
- **Why:** looking at the mockups, the owner pointed out that the work-function summary drawn
  as a line was wrong ("저게 선으로 나올 수 있나? 표면 인덱스당 하나 아니고?"), then approved
  the mockups ("적용").
- **Landed:** public PR #18.

### D43. Performance order: bonds as impostors first, then adaptive sampling (2026-10-24)

- **Decision:** of nine measured candidates (bond cylinder impostors, lighter frame switches,
  adaptive sampling that knows the low-discrepancy AO, a CUDA ray tracer, viewport per-atom AO
  and outlines, GUI startup, the cluster notebook kernel, a Jupyter widget, harder AI eval
  tasks), bonds drawn as GPU impostors together with lighter frame switches come first and the
  ray tracer's adaptive sampling second; the rest stays in docs/todo.md.
- **Why:** the owner took the recommended order ("그렇게 진행해"), based on the measurements:
  a ball-and-stick frame took 1.3 s to build at 11k atoms and 11 s at 108k, almost all of it
  bond tubes, and a Publication picture of SrTiO3 at 1600 x 1200 took 12.6 s with adaptive
  sampling saving only 1.0-1.5x.
- **Considered:** the CUDA ray tracer first (largest gain, but it cannot be checked without a
  GPU in the cloud container); analysis kernels and readers (already fast: PTM 0.74 s, Voronoi
  1.4 s at 108k atoms).
- **Landed:** public PR #25 (bond impostors).

### D45. A CUDA ray tracer before a denoiser (2026-10-24)

- **Decision:** to make Publication pictures faster, the ray tracer gets a CUDA path
  (`device="cuda"`, single precision, the same rays as the CPU tracer) rather than an
  edge-aware denoiser over fewer samples or no further work.
- **Why:** the owner picked the CUDA ray tracer on the decision card. The owner then added that
  their own PC has no NVIDIA GPU ("앗 근데 나는 nvidia gpu가 없네", Intel Arc), so whether an
  OpenCL build of the same kernel follows is open.
- **Considered:** an edge-aware denoiser from the ray tracer's normal and depth buffers
  (recommended, since it runs on every machine and could be checked here); stopping after
  public PR #27 showed that adaptive sampling cannot gain at equal quality.
- **Landed:** public PR #33.

### D46. Viewport color bars fold to chips (2026-10-24)

- **Decision:** the viewport's color bar cards (D23) fold to one-line chips (a short horizontal
  ramp and the title, a down chevron) and unfold again with a click on a card or chip; an up
  chevron on each card shows that it folds. The state is the view setting
  `colorbar_collapsed`, written into the notebook's view cell and the session, so code and the
  AI set it like any other view property. Pictures ignore it and always draw the whole bars.
- **Why:** the owner asked for it ("뷰포트 컬러바 접을 수 있게."); the cards take room in the top
  right corner of the view.
- **Considered:** hiding the bars (already `legend=False`, which also drops them from pictures);
  folding each bar on its own (one switch for all keeps the view cell to one setting).
- **Landed:** public PR #31.

## Adding an entry

Add an entry in the same pull request that implements a decision, whenever the owner chooses
between real alternatives, sets a rule, or rejects an approach. Write the date, the decision,
the owner's reason (quote the owner where possible), what else was considered and the PR. Do
not record implementation details that the code already shows, and do not invent reasons: if
the reason was not stated, leave "Why" out. Name the pull request by its number ("public PR #N"): open the
pull request first and write the number in before merging, since "this PR" means nothing once
it has merged (`tests/test_docs.py` checks). Take the next free number (the highest D-number plus one), even when the
entry goes into an earlier section; `tests/test_docs.py` fails on a number used twice. A decision that is later reversed keeps its entry,
with a line saying what replaced it and when.
