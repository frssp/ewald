# Design decisions

The decisions that shaped EWALD, who made them and why. Most of EWALD's code is written by an AI
coding assistant (Claude Code) working from these decisions; the decisions themselves, the
scope and the acceptance of each change are the maintainer's (Sunghyun Kim, "the owner" below).
This page is the record of that human design work, newest last within each section.

Each entry gives the date, the decision, why, what else was considered and where it landed.
Quotes are the owner's own words (often Korean) from the project discussion. Where the owner's
reason was not written down, "Why" is left out rather than guessed. How to add an entry is at
the end.

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
- **Landed:** this PR (`notebook_tools.touched_cells`, the chat's cell links).

### D6. Structure editing is deferred (2026-10-19)

- **Decision:** no supercell, slab or defect builder until the owner asks.

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

## Architecture

### D11. A small core, everything else a plugin (2026-10-19, 2026-10-20)

- **Decision:** VASP, LAMMPS and the common formats are built in but registered as plugins that
  can be switched off; domain packages (phonopy, PLUMED, AiiDA) live in frssp/ewald-plugins;
  plugins declare themselves in a manifest (`ewald-plugin.toml`), as in VS Code.
- **Landed:** PRs #14, #22 and later; plugin index in frssp/ewald-plugins.

### D18. Recommend a restart when a plugin change needs one (2026-10-23)

- **Decision:** when installing, updating, switching on or removing a plugin only takes effect
  after a restart, ewald says so and offers *Restart now* (which reopens the saved notebooks);
  everything that can take effect at once does (removing a single-file plugin no longer waits
  for a restart).
- **Why:** "플러그인 중 재시작이 필요한 것이 있으면 재시작을 추천해줘" (if a plugin needs a
  restart, recommend restarting).
- **Landed:** PR #229 (Plugins side bar restart bar, `plugin_catalog.restart_reason`).

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
- **Landed:** this PR (`src/ewald/potential.py`, `ewald.workflow.Analysis`, `add_analysis`), with
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

### D21. Computed properties as plugins, with their knowledge as data (2026-10-23)

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

## Project

### D15. Name, license and openness (2026-10-19, 2026-10-20)

- **Decision:** the name "EWALD: Extensible Workbench for Atomic Lattices & Dynamics" (package
  `ewald`); MIT license, copyright Sunghyun Kim and the EWALD contributors; a public repository.

### D16. How the project is developed (2026-10-19)

- **Decision:** an AI coding assistant writes most changes in pull requests; local test runs are
  the merge gate (full suite under a virtual display for GUI and rendering changes, visual
  regression for renderers); each feature PR updates the feature inventory and the TODO list.

## Adding an entry

Add an entry in the same pull request that implements a decision, whenever the owner chooses
between real alternatives, sets a rule, or rejects an approach. Write the date, the decision,
the owner's reason (quote the owner where possible), what else was considered and the PR. Do
not record implementation details that the code already shows, and do not invent reasons: if
the reason was not stated, leave "Why" out. A decision that is later reversed keeps its entry,
with a line saying what replaced it and when.
