# AI eval results

`ewald eval` ([AI integration](../user/claude.md#scoring-a-model-ai-eval-suite)) scores a model on
tasks with known answers. This page keeps the runs made with real models and what they found;
the saved runs are in `benchmarks/ai_eval/` (answers, the notebooks the models left, tool calls,
tokens and cost per task), so a later prompt or tool change can be compared with
`ewald eval --compare benchmarks/ai_eval/2026-10-24-opus.json new.json`.

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
