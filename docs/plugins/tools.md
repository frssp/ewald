# AI tool plugins

An AI tool plugin gives the AI model (Ask AI, the AI chat, `ewald ask` and the MCP server), whichever
provider answers, a new action:
a function it can call with JSON arguments while it works on your pipeline. Write one when an
analysis needs more than adding a modifier, for example a summary over a whole trajectory or a
file format only your group uses.

```python
# <config dir>/plugins/contacts.py
from ewald.plugin_api import ai_tool


@ai_tool(schema={"type": "object", "properties": {"cutoff": {"type": "number"}}})
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
| `description` | what the model reads to decide when to call it (default: the docstring) |
| `schema` | JSON schema of the arguments (default: none) |

Tools run in your Python process with your permissions, so treat inputs as you would any text
in a file the model reads: check paths and refuse anything that loads code (the ewald-plumed plugin's
`compute_plumed_cvs` refuses PLUMED `LOAD`).

In a package, register the function under the `ewald.tools` entry point group:

```toml
[project.entry-points."ewald.tools"]
count_short_contacts = "my_plugin.tools:count_short_contacts"
```

`ewald plugins` lists the tools of each plugin, and switching a plugin off removes them.

## AI providers

An AI provider plugin adds a service to Preferences › AI › Provider (and to `--provider`,
`provider=`). Claude, OpenAI, Google Gemini and any OpenAI-compatible server are built in. Most
other services speak the OpenAI Chat Completions API, so a `Provider` with their address is all
it takes:

```python
# <config dir>/plugins/mistral.py
from ewald.plugin_api import ai_provider
from ewald.llm.providers import Provider


@ai_provider
def mistral():
    """Mistral models through Mistral's OpenAI-compatible API."""
    return Provider("mistral", "Mistral", assistant="Mistral",
                    models={"mistral-large-latest": "Mistral Large"},
                    base_url="https://api.mistral.ai/v1",
                    key_name="mistral-api-key", env_var="MISTRAL_API_KEY")
```

| Field | Meaning |
|---|---|
| `name`, `label` | the setting value and what Preferences shows |
| `models` | suggested model ids and labels; the first is the default |
| `assistant` | how the chat names the answers |
| `base_url` | the service's address |
| `key_name`, `env_var` | its keychain entry and environment variable (`key_name=""`: no key) |
| `vision` | whether the model gets images (renders, close-ups); default true |
| `chat_class` | `ewald.llm.chat.OpenAIChat` (default) or `AnthropicChat`; a subclass of `ewald.llm.chat.Chat` for an API that is neither |
| `make_client` | `(api_key, base_url, **kwargs) -> client` when the client is not `openai.OpenAI` |

A `Chat` subclass implements `user(text, images)`, `send(tool_choice=None) -> Reply` and
`tool_results(results)`; the agent loop, Modifier Forge and the narrator use nothing else. A
built-in name cannot be taken. In a package, use the `ewald.ai_providers` entry point group.
The key is saved like the built-in ones (keychain, never `settings.json` or the environment).
