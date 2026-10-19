# Claude integration

`ewald.llm` exposes the pipeline to Claude as tools, so an analysis can be asked for in words
and still ends up as an ordinary, reproducible pipeline.

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...          # or: ant auth login
```

```python
from ewald.llm import Agent

agent = Agent.from_file("XDATCAR")
print(agent.ask("What happens in this run? Render the frame where it changes."))
print(agent.toolset.to_script())      # the analysis as a standalone Python script
```

The model never reads raw coordinates. The tools return compact descriptors (composition, cell,
density, minimum distance, coordination histograms, g(r) peaks, sanity warnings), and
`describe_trajectory` flags the frames where a descriptor jumps. Renders come back as images
the model can look at. The tools are: `load_structure`, `describe_frame`,
`describe_trajectory`, `event_details`, `list_modifiers`, `add_modifier`, `remove_modifier`,
`show_pipeline`, `set_frame`, `evaluate_expression`, `compute_rdf`, `analyze_symmetry`,
`compute_diffraction`, `compute_msd`,
`describe_selection`, `compute_plumed_cvs`, `read_plumed_output` ([PLUMED](plumed.md)),
`render_image` (with an optional `theme`), `list_visuals`, `add_visual`, `remove_visual` (visual
element plugins drawn by later renders), `list_overlays`, `add_overlay`, `remove_overlay`
(overlays such as the coordinate tripod), `list_styles` (themes and color maps, each marked
whether it reads in grayscale) and `get_script`.

`run_python`, which executes model-written code in your Python process, is off unless you pass
`allow_code=True` (or `--allow-code` on the command line).

## Asking about a selection

A question can be about a few atoms rather than the whole structure: atoms picked in the GUI
(see [GUI](gui.md)), a selection expression, or Particle Identifiers.

```bash
ewald ask dump.lammpstrj "What are these atoms?" --select "StructureType == 0" --frame 40
ewald ask POSCAR "Why is this Li under-coordinated?" --ids 17
```

```python
agent = Agent.from_pipeline(pipe)
agent.ask_about_selection("What are these atoms?", indices=[17, 18, 25], frame=40)
```

ewald sends the question together with a description of the selection and of the atoms within
6 Å of it (`ewald.llm.selection.describe_selection`, which needs no LLM). The description holds
each selected atom's neighbors with distances, its coordination number, structure type and
property values. It also gives the same averages over the surroundings and over the whole
frame, the distances and angle between up to six selected atoms, and the nearest atom whose
structure type differs from the bulk. Structure types are included only when a CNA or PTM
modifier ran upstream; otherwise the description suggests one. A close-up render
(`selection_scene`) shows the region unwrapped across periodic boundaries, with the selected
atoms in yellow. Claude can call `describe_selection` again with another radius or cutoff,
or add a modifier that tests its reading.

## Without an LLM

The descriptors work on their own:

```bash
ewald describe dump.lammpstrj --trajectory --type-map 1=Cu   # JSON summary and detected events
ewald narrate dump.lammpstrj --type-map 1=Cu --no-llm        # event timeline, rule-based titles
```

## From Claude Code and other MCP clients

```bash
ewald mcp                                   # the same tools over MCP (stdio)
claude mcp add ewald -- ewald mcp           # register them in Claude Code
```

## For agents that write ewald code

`llms.txt` in the repository root is a compact API reference for LLMs: the data model and its
rules, expression syntax, the differences from OVITO's API (where models trained on OVITO code
go wrong), and every modifier's signature. `ewald api` prints the current version.
`data.summary()` shows a frame's properties and ranges, and OVITO modifier names resolve to
the ewald classes, so OVITO-style code usually runs as written.
