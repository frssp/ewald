# AI eval results

`ewald eval` ([AI integration](../user/claude.md#scoring-a-model-ai-eval-suite)) scores a model on
tasks with known answers. This page keeps the runs made with real models and what they found;
the saved runs are in `benchmarks/ai_eval/` (answers, the notebooks the models left, tool calls,
tokens and cost per task), so a later prompt or tool change can be compared with
`ewald eval --compare benchmarks/ai_eval/2026-10-24-opus.json new.json`.

## 2026-10-29: the atoms with the largest MSD

The owner asked Ask AI, in the Li6PS5Cl example, to "draw the trajectories of the 5 Li atoms
whose MSDs are largest" and it could not. EWALD had an MSD per species but none per atom, so the
models either wrote 30 lines of their own unwrapping and MSD code or chained
`UnwrapTrajectories`, `CalculateDisplacements`, `ComputeProperty` and `TimeAveraging` (which gives
the mean squared displacement from frame 0, not an MSD, and picked a different set of atoms), and
then could not use the averaged property in an expression: "SqDisp (Average)" was reachable only
as `SqDisp(Average)`, which parses as a call. The ids they found ended up hand-written in an
`ExpressionSelection`.

Now `MeanSquaredDisplacement` writes the particle property 'MSD' (each atom's own time-averaged MSD
over the lags of `fit_range`), `ExpressionSelection(expression, top=n, by=...)` keeps the `n`
particles with the largest `by`, property names drop every character a name cannot hold
(`SqDispAverage`) and are also understood as written inside an expression (`Net Displacement > 6`),
and python_api has the three-step recipe. The new task `li-top-msd-paths` (category `paper`, the
notebook already has the paper's MSD step) passes when the traced atoms are the top five under any
of the readings an analyst may take (squared displacement first to last frame, time-averaged MSD
over all lags or over a fit window).

Runs with `--tasks li-top-msd-paths --repeat 2`, effort high, first-party API, run cells off.
"Before" is main at 68c5cda with only the task and its grader copied in.

| Model | Passed before | Passed after | Tool errors | Tool calls per task | Cost (2 runs) |
|---|---|---|---|---|---|
| Claude Opus 5.5 | 1/2 | 2/2 | 5 → 0 | 21.5 → 9.0 | $1.16 → $0.54 |
| Claude Sonnet 5.5 | 2/2 | 2/2 | 6 → 0 | 23.0 → 8.0 | $0.65 → $0.30 |
| Claude Haiku 5.5 | 1/2 | 2/2 | 3 → 0 | 19.5 → 8.5 | $0.036 → $0.020 |

Before, Opus once traced a different five (after ranking by hand from properties it built) and
Haiku once drew no trajectory lines at all. After, every run added the selection and the
trajectory lines in two step cells. Saved runs: `benchmarks/ai_eval/2026-10-29-top-msd-{before,after}-{opus,sonnet,haiku}.json`.

## 2026-10-26: whole studies from one sentence (D70)

The owner's next goal (D70) is the AI doing a whole analysis alone: "find the activation energy
of Li diffusion from these runs" should end in a notebook that compares the runs, fits, plots and,
when asked, holds the paragraph for the paper. The 53 tasks above did not separate models, so the
new `study` category asks for whole studies. Nine of its 11 tasks run on synthetic VASP MD series
written by `ewald.llm.eval_data` (Li hopping between the empty sites of a grid in the ab plane of a
static S host, Ea 0.30 eV, seeded, so the answer is known), and one on the LLZO paper data:

| Task | What the model has to see |
|---|---|
| `study-arrhenius` | five runs next to the one opened; Li only; Ea and its plot |
| `study-paper` | the same, plus the 2D path, D in cm²/s and a results paragraph quoting the cells' numbers |
| `study-hot-start` | T500 starts at 2000 K for 120 frames (check_run flags it): cut it, or Ea is 0.03-0.18 eV instead of 0.29 |
| `study-mislabel` | folder T900 holds a 600 K run (its INCAR): using folder names gives 0.19 eV instead of 0.31 |
| `study-scf` | T600 hit NELM at 80 % of its steps and its D is off the trend: leave it out |
| `study-timestep` | the runs store frames 10, 10, 10, 20 and 50 fs apart: one time step for all gives 0.40 eV instead of 0.27 |
| `study-path` | how Li moves and how long one hop is (2.0 Å, in the ab plane), shown in the 3D view |
| `study-redo` | a follow-up, "leave the 500 K run out and redo the fit and the plot": edit the fit cell |
| `study-llzo` | LLZO at 500-800 K (GROMACS XTC, 50.8 fs frames, diffusive from 10 ps): Ea 0.118 eV |
| `study-one-temperature` | four runs, all at 600 K (a cutoff test): say no Ea can come from them |
| `study-no-diffusion` | Li only vibrates: say there is no D to fit |

Every Arrhenius task also needs the value printed by a cell (so the notebook makes it), the plot
under a cell, no failed cell left, and the same result when the notebook is re-run from the top.

Runs with `--category study --allow-code --repeat 2` (22 runs per model), effort high, first-party
API. "Before" is main at d926ff5 with only the new tasks and graders copied in; "after" is public
PR #73.

| Model | Passed before | Passed after | Reproducible after | Tool errors | Tool calls per task | Cost after |
|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 21/22 | 22/22 | 100% | 11 → 5 | 10.1 → 8.1 | $6.86 |
| Claude Sonnet 5.5 | 17/22 | 21/22 | 94% | 33 → 3 | 10.0 → 5.8 | $3.20 |
| Claude Haiku 5.5 | 17/22 | 21/22 | 100% | 16 → 5 | 12.9 → 9.8 | $0.25 |

The saved runs are `benchmarks/ai_eval/2026-10-26-study-{before,after}-{opus,sonnet,haiku}.json`.
The traps were caught more often than expected: in all but one run (Sonnet, `study-hot-start`
before, Ea 0.35 eV) the models read the INCARs, ran check_run and said which run they trimmed or
left out, and no run gave a number for the two wrong premises. What
failed was mostly the last mile:

- Cells that called a function the notebook had not imported (`load_runs`, `msd`, `check_run`:
  NameError, the most common tool error) and stayed in the notebook as failed cells. New notebooks
  now import the verbs and the run functions.
- `fit.plot()` (no such method; then, once added, it drew nothing under a cell because
  `plt.show()` of a plotstyle figure is lost there), `runs["500"]` for the run labelled "T500",
  `runs[run]`, `runs.keys()`, `arrhenius({T: D})`: all work now; `arrhenius` names the runs it got
  when the fit cannot be made, and `RunTables.attribute` lists the names there are.
- Not seeing that Li moves in the ab plane only (Haiku and Sonnet in `study-paper` and
  `study-path`): `MeanSquaredDisplacement(by_axis=True)` gives the MSD along x, y and z and 1D
  coefficients, and `compute_msd` reports how the MSD splits over the axes.
- Failed cells left behind and numbers quoted that no cell printed: the agent's system prompt now
  says that a study is checked run by run, that the notebook must run without errors before the
  answer and that every quoted number must come from a cell.
- A cell that drew a figure and then failed sent its image in an error tool result, which the
  Messages API refuses (a 400 that ended an Opus run); error results now carry text only.
- After the changes, the remaining failures were a failed cell Sonnet left in `study-paper` and
  Haiku putting a diffusion coefficient where `study-paper` asks for Ea.

The whole suite after the change (68 tasks, one run, `--allow-code`): Sonnet 5.5 68/68 ($4.57) and
Haiku 5.5 68/68 ($0.28, one task not reproducible), so the new import cell and prompt cost nothing
on the single-step tasks. The suite again nearly saturates for these tasks (Opus 22/22). Two runs per model are a small
sample; harder judgement calls (a run that looks fine but is too short, several traps at once,
the paragraph's writing judged by a second model) are in [todo.md](../todo.md).

## 2026-10-25: editing cells instead of adding copies (D68)

The owner saw the AI add a new cell, with the option changed, each time an option of an existing
cell should change. Multi-turn conversations in the GUI chat's prompts (a script playing the side
bar: notebook in each question, code blocks of answers into the notebook), on `Cu_md.extxyz`,
before and after public PR #71:

- "RDF 그려줘 → cutoff 8 Å → bin 200 → x축 2~6 Å → coordination 색칠 → coordination cutoff 3.0".
  Before: Haiku 5.5 answered the last question with a new scipy cell computing coordination by
  hand; Sonnet 5.5 added a second `CoordinationAnalysis(cutoff=3.0)` that overwrote the first one's
  `coordination-rdf` table, so the RDF plot silently changed. After: Haiku edited the step and
  plot cells all the way through (5 cells, no copy); Opus 5.5 added the second step with
  `again=true` and saved the short-cutoff coordination under its own name first, so both results
  stay right.
- A notebook with CoordinationAnalysis, ColorCoding and an RDF plot, then "coolwarm 으로 → RDF 6 Å
  까지 → 그래프 제목": the color map was an edit before and after. "Give the plot a title" was a new
  `figure()` cell next to the old plot for Haiku with running off, before and after, because
  `show()` has no `title` (open in docs/todo.md); Opus edited the plot cell. "RDF up to 6 Å" made
  Opus write its own RDF cell and Haiku raise the coordination cutoff to 6 Å, before and after:
  one cutoff sets both results (open in docs/todo.md).

The new `nb-plot-range`, `nb-rdf-bins`, `nb-redo-cutoff` and `nb-recolor` tasks (`ewald eval --tasks
'nb-*' --allow-code --repeat 2`) passed every run with Opus, Sonnet and Haiku 5.5, before the change
(only the tasks copied in) and after: a single question about a notebook that already has the cell
does not separate them, the copies came in longer conversations. Not saved to
`benchmarks/ai_eval/` (a subset of the suite).

## 2026-10-24: first runs with real models

All 53 tasks, `--allow-code`, asked as the AI chat side bar asks (`--context gui`), effort high,
first-party Anthropic API, one run per model (`--repeat 1`), at commit c77caf1.

| Model | Passed | Reproducible | Tool calls per task | Tool errors | Cost | Time |
|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 53/53 | 100% | 2.58 | 0 | $4.39 | 8.1 min |
| Claude Sonnet 5.5 | 53/53 | 100% | 2.60 | 1 | $3.40 | 5.7 min |
| Claude Haiku 5.5 | 53/53 | 100% | 3.30 | 3 | $0.26 | 5.8 min |

Before the fixes below, Opus 5.5 with the bare agent (`--context plain`, the 45 tasks of the time)
passed 44/45 with 7 tool errors and 94% reproducible; the one failure was the grader's (it wanted a
`ColorCoding` step where `view(pipe, color_by="Coordination")` does the same). The first runs in the
GUI context, between the fixes, gave Opus 53/53, Sonnet 52/53 (`view(pipe, frame=-1)` for "the last
frame", which the GUI then showed as frame 0) and Haiku 53/53.

### What the models tripped over, and what changed

- A cutoff beyond half the cell width (`compute_rdf(cutoff=6)`, `CoordinationAnalysis(cutoff=6)` on the
  108-atom Cu box or the 8-atom Si cell) was an error in every run of every model, until the
  model lowered it. The neighbor search now searches the periodic images (`neighbors._image_pairs`),
  as OVITO and ASE do.
- A cell written as python_api shows it, `# %% hide fcc [each frame]` first, kept the marker in its
  code and ran once without `data` (NameError). `insert_cell`, `edit_cell` and the GUI's inserted
  code now read a leading marker as the cell's title and frame tag (`notebook.cell_from_source`).
- `np` was not defined in the notebook of an opened file (and the AI's session), although
  python_api says notebooks import numpy as np; the import cell now does.
- `view(pipe, frame=-1)` showed frame 0 in the GUI; it shows the last frame, as `pipe.compute(-1)`.
- `ParticleType == 2` (OVITO's type numbers) matched nothing when the types have names; it now
  matches types named "2" and is an error naming the types otherwise. `evaluate_expression` of a
  name property (`ParticleType`) counts its values instead of failing.
- `describe_trajectory` with `cutoff="3.0"` (a string) failed; `get_view`'s description listed
  "Groups:" and a model passed `groups=`.

### What the run does not show

Every model passed everything, so the suite no longer separates them: it shows that the tools,
the notebook cells and python_api work end to end with real models, not how far they reach.
Harder tasks (multi-turn questions, paper datasets with a judgement call, picks and Ask about
selection, a wrong premise in the question) are in [todo.md](../todo.md). The answers' explanations
are not graded, a single run per model gives no spread (`--repeat 3`), and no other provider or
local model has been run yet.
