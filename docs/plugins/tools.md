# Claude tool plugins

A Claude tool plugin gives Claude (the Ask panel, `ewald ask` and the MCP server) a new action:
a function it can call with JSON arguments while it works on your pipeline. Write one when an
analysis needs more than adding a modifier, for example a summary over a whole trajectory or a
file format only your group uses.

```python
# <config dir>/plugins/contacts.py
from ewald.plugins import claude_tool


@claude_tool(schema={"type": "object", "properties": {"cutoff": {"type": "number"}}})
def count_short_contacts(toolset, cutoff=2.0):
    """Pairs of atoms closer than cutoff (Angstrom) in the current frame."""
    from ewald.neighbors import find_pairs

    data = toolset.pipeline.compute(toolset.frame)
    i, j, _, _ = find_pairs(data.positions, data.cell, cutoff, half=True)
    return {"pairs": int(len(i)), "frame": toolset.frame}
```

The function receives the `Toolset` first: `toolset.pipeline` is the loaded pipeline (or None),
`toolset.frame` the current frame, `toolset.workdir` a folder for files and `toolset.figures`
the images made so far. `toolset._need_pipeline()` raises the usual "no structure loaded" error.
Return JSON-ready data (numbers, strings, lists, dicts; NumPy values are converted), or a
`ewald.llm.tools.ToolResult` to send images back as well. `ewald.llm.tools.downsample(table)`
shortens a column table to 60 rows for the model.

| Argument | Meaning |
|---|---|
| `name` | tool name (default: the function name). A built-in tool of the same name wins |
| `description` | what Claude reads to decide when to call it (default: the docstring) |
| `schema` | JSON schema of the arguments (default: none) |

Tools run in your Python process with your permissions, so treat inputs as you would any text
in a file Claude reads: check paths and refuse anything that loads code (the ewald-plumed plugin's
`compute_plumed_cvs` refuses PLUMED `LOAD`).

In a package, register the function under the `ewald.tools` entry point group:

```toml
[project.entry-points."ewald.tools"]
count_short_contacts = "my_plugin.tools:count_short_contacts"
```

`ewald plugins` lists the tools of each plugin, and switching a plugin off removes them.
