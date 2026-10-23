# What the AI may do and what it sees

This page lists, for Ask AI (the AI chat side bar, `ewald ask`, `ewald mcp` and
`ewald.llm.Agent`), every tool the model can call, what each tool may change, the settings that
widen or narrow that, and everything ewald sends to the AI provider with a question. It is
written from the code in `src/ewald/llm/` and `src/ewald/gui/ask_panel.py`;
`tests/test_docs.py` fails when a built-in tool or an `llm.*` setting is missing here. How to set
up a provider and a key is in [AI integration](claude.md).

In short: the model chooses what to compute and reads compact results; ewald computes every
number. It changes the structure, the view and every other result only by writing notebook
cells, so the notebook is the full record of what it did. It cannot run Python of its own unless
you allow it, it never sees your API key, and the built-in tools never hand it raw coordinate
arrays.

## Tools and what they may change

Every tool belongs to one of four groups. `tests/test_llm.py::test_every_tool_is_classified`
fails when a new built-in tool is not put into one of them, and
`test_read_only_tools_change_nothing` checks that the read-only ones leave the notebook, the
pipeline and the view as they were.

**Read only.** These compute and describe, and change nothing (`set_frame` only picks the frame
the next tools look at).

| Tool | What it returns |
| --- | --- |
| `describe_frame` | composition, cell, density, minimum distance, coordination histograms, g(r) peaks, property ranges, sanity warnings, `atom_colors` |
| `describe_selection` | the picked atoms and their surroundings (see [Selection](#a-question-about-picked-atoms)) |
| `describe_trajectory` | descriptor time series over sampled frames and the frames where one jumps (events) |
| `event_details` | the descriptors before and after one event |
| `list_modifiers` | every modifier the model may add, with parameters and defaults |
| `show_pipeline` | source, step list, current frame |
| `set_frame` | picks the frame later tools use |
| `evaluate_expression` | count and summary of a selection expression (arithmetic, comparisons and NumPy functions only) |
| `compute_rdf`, `compute_msd`, `compute_diffraction` | curves, downsampled to at most 60 rows at 4 significant digits |
| `analyze_symmetry` | space group and symmetry data |
| `check_run` | what the run is and whether it converged, from its input and log files ([Runs](runs.md)) |
| `render_image` | a PNG drawn with the current view settings |
| `get_view` | every view setting and `atom_colors` |
| `list_visuals`, `list_overlays`, `list_styles` | what can be drawn, and the themes and color maps |
| `get_script` | the notebook as a script |
| `python_api` | the API reference: `llms.txt` plus the modifier plugins installed here |
| `list_cells`, `read_cell` | cell summaries; one cell's source and outputs (plots as images) |
| `list_workflows`, `describe_workflow` | workflows of a workflow source; offered only when one is installed ([Workflows](workflows.md)) |

**Cell writers** (`CELL_WRITERS` in `llm/tools.py`). Each writes one cell from a name the tool
checked and JSON values, never from code the model wrote, and runs it. That is why they work
without permission to run code. A cell that fails is removed again and its error goes back to
the model.

| Tool | Cell it writes |
| --- | --- |
| `load_structure` | `pipe = load('XDATCAR')` |
| `open_workflow` | `pipe = load('simagent:lab/3f9c2e1a/bands')`: a workflow, a step or an output of it, by name; offered only with a workflow source |
| `add_modifier` | a step cell `pipe.add(CoordinationAnalysis(cutoff=3.0))` |
| `remove_modifier` | deletes that step cell |
| `set_view`, `add_visual`, `remove_visual`, `add_overlay`, `remove_overlay` | the view cell `view(pipe, ...)` |
| `export_file` | `export_file(pipe, 'out.extxyz', frames=[0])` |
| `add_analysis` | the cells of a workflow analysis (from `describe_workflow`), written by the workflow's plugin, not by the model; offered only with a workflow source |

`add_modifier` offers every modifier in `ewald.modifiers` and the modifier plugins, except
`PythonScript` (only with `allow_code`) and `OvitoModifier` (never; it takes a Python object).
`export_file` replaces files only inside its work dir and never writes into hidden folders such
as `~/.ssh`.

**Cell editors** (`llm/notebook_tools.py`). These change the notebook's cells directly:
`insert_cell`, `edit_cell`, `move_cell`, `delete_cell` and `run_cell`. A cell the model inserts
or edits holds code the model wrote, so it is marked not run (or out of date) until it is run;
when you let the AI run cells (below), `insert_cell` and `edit_cell` run the code cell they wrote
at once and return its outputs (`run=false` only writes it).
`edit_cell` with `params` is the exception: it rewrites a step's literal call with new values
and runs it like a cell writer, and refuses code steps (`PythonScript`, Python steps). In the GUI
each edit is one undo step ("AI: insert cell", Ctrl+Z).

**`run_cell`** runs any cell, so it is offered only when you allow it: in the GUI with
Preferences › AI › **Let the AI run cells** (`llm.run_cells`, off by default; read again before
every question), in Python and on the command line with `allow_code=True` / `--allow-code`.
While it is not offered, a call to it is refused with "the user has not allowed it". The same
permission makes `insert_cell` and `edit_cell` run what they wrote, tells the model in its
system prompt that it may run cells, and runs the ```python blocks of an answer that go into
the notebook.

**Plugin tools.** Installed plugins add tools with `@ai_tool` ([AI tool plugins](../plugins/tools.md));
ewald-plumed adds `compute_plumed_cvs` and `read_plumed_output`. A plugin tool cannot replace a
built-in one of the same name. Plugins run in-process with your rights, like any other Python
package you install, so only install plugins you trust; the rules above ask them to change
results through `toolset.call("add_modifier", ...)` and to call `toolset.check_read(path)`
before opening a path the model gave them.

There is no tool that runs Python, shell commands or arbitrary file writes (the old `run_python`
was removed when the AI was limited to notebook cells).

## Untrusted notebooks

An `.ewd` file opens restricted until you trust it (the **Trust** button on the notebook's banner). In a restricted notebook
no tool writes, edits or runs a cell: cell writers, cell editors and `run_cell` all refuse, and
`list_cells` says the notebook is not trusted. The chat saved in that `.ewd` is shown but not
sent to the model until the notebook is trusted, since its text came from someone else.

A model can act on text inside the files it reads (a comment line or an atom type name can carry
instructions). That is why the always-on tools are limited to what such text cannot abuse:
expressions accept only arithmetic, comparisons and NumPy array functions, `export_file` stays in
its work dir, and `compute_plumed_cvs` refuses `LOAD` (native code) and absolute or `..` paths.

## Files and hosts the AI may open

The files the model opens itself (`load_structure`, `describe_workflow`, `open_workflow`, `add_analysis`, plugin tools such as
`read_plumed_output`) pass `ReadAccess` (`llm/access.py`):

| Setting | Default | Effect |
| --- | --- | --- |
| `llm.read_folders` (Folders the AI may read) | empty | folders separated by `;`; local paths must lie inside one of them after `~`, `..`, links and wildcards are resolved (case-insensitive on Windows). Empty allows any folder. |
| `llm.confirm_hosts` (Ask before the AI connects to a host) | on | before the first connection to an SSH host the model named, the GUI asks you; the answer holds for that chat. Off: `~/.ssh/config` and known_hosts alone decide. |

Paths with a plugin scheme (`aiida:1234`) are not files and go to their plugin. Cells you run
yourself are not limited, and neither is a cell the model wrote once it runs (with run permission
it is ordinary Python), so `llm.run_cells` is the setting that matters most. In Python:
`toolset.read_access = ReadAccess(folders, confirm_host)`.

## What is sent with a question

Everything below goes to the provider chosen in Preferences › AI (Anthropic by default, or
OpenAI, Gemini, an OpenAI-compatible server or a plugin's). With a local server (Ollama, LM
Studio) nothing leaves your machine.

**Once per conversation (every request):**

- The system prompt: `SYSTEM_PROMPT` in `llm/agent.py` (what ewald is, that numbers must come
  from tools, that changes go through cells, to read `python_api` before writing code). In the
  GUI it is followed by the notebook note in `gui/ask_panel.py` (`NOTEBOOK_NOTE`: the cell kinds,
  the cell tools, and how ```python blocks in answers are handled, `llm.code_cells`).
- The tool definitions: name, description and JSON schema of every offered tool. The
  `load_structure` description names the installed readers.
- The conversation so far: questions, answers, tool calls and tool results.

**With each question, in the GUI** (`AskPanel._notebook_prompt`), each part only when it changed
since the last question of that chat:

- The current notebook's code (percent format), cut at 20,000 characters. It includes file paths
  and anything typed into the cells, but not cell outputs.
- The errors of cells whose last run failed (`failed_cells`: the last 25 lines of each, at most
  6,000 characters), or a note that there are none any more.
- How the 3D view colors the atoms (the color rule: which step or setting, which property,
  the colors or color map and range).
- Once, after the agent was made again (a new `pipe`) or for a chat saved in the notebook: the
  earlier transcript of that tab's chat (questions and answers as text, at most 20,000
  characters), marked as a record, not instructions. Each notebook tab has its own chat, saved in
  its `.ewd` (`Notebook.chat`).

**Only when the model asks**, as tool results: the descriptors and tables listed above, rendered
images (`render_image`, `read_cell` plots, the selection close-up), the API reference
(`python_api`), cell sources and outputs (`read_cell`, outputs cut in the middle when long). A
provider that does not take images (the OpenAI-compatible one) gets no images.

### A question about picked atoms

With atoms picked in the viewport, the question also carries `describe_selection` for them (each
atom's neighbors with distances, coordination, structure type and property values next to the
frame averages, distances and angles between up to six picked atoms, the nearest atom of another
structure type) and a close-up render with the picked atoms in yellow
(`llm/selection.py`).

### What is never sent

- API keys. Each provider's key lives in the system keychain (or a `credentials.json` readable
  only by you where no keychain works), is handed to the client object only, and never goes into
  `settings.json`, the environment, session files, logs or a prompt. The Modifier Forge test
  subprocess cannot read it (null keyring backend). Extra headers (`llm.headers`) are stored in
  `settings.json` and sent to the server with every request, so do not put secrets there.
- Raw coordinate or property arrays from the built-in tools: they return summaries, histograms,
  peaks and downsampled tables. The model can still read whatever a cell prints (`read_cell`) and
  the source of every cell, so a cell that prints positions shows them to the model.
- Files the model did not open through a tool, your settings, other notebooks' code (only the
  current tab's), and other tabs' chats.

## Settings

Every AI setting lives in Edit › Preferences › AI; the key is not a setting (it is in the
keychain).

| Setting | Controls |
| --- | --- |
| `llm.provider`, `llm.model`, `llm.base_url`, `llm.route_prefix`, `llm.key_header`, `llm.headers`, `llm.timeout`, `llm.profiles` | who answers and how ewald connects ([Providers](claude.md#providers)) |
| `llm.effort`, `llm.stream` | how hard the model thinks; answers streamed as they arrive |
| `llm.code_cells` | ```python blocks in answers: inserted as cells (`insert`, default) or offered with a button (`button`); run only with `llm.run_cells` |
| `llm.run_cells` | whether the AI runs cells: `run_cell` offered, cells written by `insert_cell`/`edit_cell` and inserted code blocks run at once (off) |
| `llm.read_folders`, `llm.confirm_hosts` | which folders and hosts the AI may open |

## Other AI features

- `ewald mcp` serves the same tools to Claude Code or another MCP client, with the same rules;
  `--allow-code` offers `run_cell` and `PythonScript`.
- `ewald narrate` sends `describe_trajectory`'s events (and renders of the event frames) to the
  model and asks for a timeline; it has no tools.
- The Modifier Forge (`ewald forge`) asks the model to write a modifier plugin and its tests. The
  code runs only in a subprocess with an allow-listed environment and a scratch home to test it,
  and is saved to the plugin folder only after you confirm.
- `ewald eval` runs the normal loop on fresh notebooks; its `code` tasks need `--allow-code`.
