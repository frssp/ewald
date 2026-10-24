# AI eval results

`ewald eval` ([AI integration](../user/claude.md#scoring-a-model-ai-eval-suite)) scores a model on
tasks with known answers. This page keeps the runs made with real models and what they found;
the saved runs are in `benchmarks/ai_eval/` (answers, the notebooks the models left, tool calls,
tokens and cost per task), so a later prompt or tool change can be compared with
`ewald eval --compare benchmarks/ai_eval/2026-10-24-opus.json new.json`.

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
