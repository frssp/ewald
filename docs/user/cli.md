# Command line

`ewald` (or `python -m ewald`) has one subcommand per task. `--type-map 1=Cu,2=Zr` names
LAMMPS numeric types wherever a file is read.

| Command | Does |
|---|---|
| `ewald info FILE` | frame count, particle count, properties and cell of frame 0 |
| `ewald render FILE OUT.png [--frame N] [--size W H] [--direction iso] [--bonds 2.8] [--color-by EXPR] [--cmap NAME] [--ao]` | render one frame |
| `ewald gui [FILE] [--device cuda]` | open the desktop viewer (also `ewald-gui`) |
| `ewald describe FILE [--frame N] [--trajectory]` | JSON summary of a frame, or of the run with detected events |
| `ewald narrate FILE [--no-llm] [--render] [--json]` | event timeline of a trajectory |
| `ewald ask FILE "question" [--allow-code] [--script out.py] [--model ID]` | ask Claude about a structure; `--script` saves the resulting pipeline |
| `ewald mcp [FILE] [--allow-code]` | serve the analysis tools over MCP (stdio) |
| `ewald api` | print the LLM API reference (`llms.txt`) |

Examples:

```bash
ewald info XDATCAR
ewald render dump.lammpstrj frame5.png --frame 5 --type-map 1=Cu,2=Zr --color-by "Position.Z"
ewald ask OUTCAR "Is the final structure reasonable?" --script analysis.py
```
