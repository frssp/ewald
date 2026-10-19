# Command line

`ewald` (or `python -m ewald`) has one subcommand per task. `--type-map 1=Cu,2=Zr` names
LAMMPS numeric types wherever a file is read.

| Command | Does |
|---|---|
| `ewald examples [--copy DIR]` | list the shipped [example files](quickstart.md#example-files) with their paths, or copy them into DIR |
| `ewald examples --papers`, `--fetch NAME` | list the [published datasets](papers.md) or download one (checksummed, cached) |
| `ewald info FILE` | frame count, particle count, properties and cell of frame 0 |
| `ewald render FILE OUT.png [--frame N] [--size W H] [--direction iso] [--bonds 2.8] [--color-by EXPR] [--cmap NAME] [--ao]` | render one frame |
| `ewald gui [FILE] [--device cuda]` | open the desktop viewer (also `ewald-gui`) |
| `ewald describe FILE [--frame N] [--trajectory]` | JSON summary of a frame, or of the run with detected events |
| `ewald narrate FILE [--no-llm] [--render] [--json]` | event timeline of a trajectory |
| `ewald ask FILE "question" [--select EXPR \| --ids 1,2] [--frame N] [--allow-code] [--script out.py] [--model ID]` | ask Claude about a structure, or about the selected particles; `--script` saves the resulting pipeline |
| `ewald mcp [FILE] [--allow-code]` | serve the analysis tools over MCP (stdio) |
| `ewald remote list\|add\|remove\|test\|ls\|get` | [saved remote computers](formats.md#saved-remote-computers): `add NAME --host H --user U [--key FILE] [--password] [--passphrase] [--jump J] [--folder DIR]` (secrets asked and saved in the OS keychain), `ls NAME[:DIR]`, `get NAME:/file [DEST]` |
| `ewald api` | print the LLM API reference (`llms.txt`) |
| `ewald inventory [-o docs/features.md] [--gui]` | print or rewrite the [feature inventory](../features.md); `--gui` reads the menus from a live window |

Examples:

```bash
ewald info XDATCAR
ewald render dump.lammpstrj frame5.png --frame 5 --type-map 1=Cu,2=Zr --color-by "Position.Z"
ewald render POSCAR fig.pdf --renderer vector --theme VESTA --overlay "Coordinate tripod"
ewald ask OUTCAR "Is the final structure reasonable?" --script analysis.py
```
