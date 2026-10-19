# Modifier Forge

Modifier Forge has Claude write a [modifier plugin](modifiers.md) from a description. ewald
then tests the plugin and saves it only once every check passes:

```bash
ewald forge "fraction of O neighbours within 2.5 A around each Li atom, as a property"
```

```python
from ewald.llm.forge import forge

result = forge("fraction of O neighbours within 2.5 A around each Li atom", run_code=True)
print(result.summary())
```

Claude submits two files: the plugin module, a `Modifier` subclass registered with
`@modifier`, and a test module. The test module's `test_*` functions build structures with a
known answer (a perfect and a perturbed crystal, periodic boundaries) and assert that answer.
ewald runs both in a separate Python process, with a time limit, a temporary config folder and
no other plugins loaded, and checks:

| Check | Fails when |
|---|---|
| import | the module raises on import, or its class name clashes with a built-in modifier |
| class registered | the class is not a `Modifier` subclass decorated with `@modifier` |
| parameters | a constructor argument has no default, or is not stored as an attribute of the same name |
| no in-place writes | `compute` changed an array of the input frame (a rattled two-species fcc crystal) |
| GPU code path | the result on a NumPy-backed fake CUDA backend differs from the NumPy result, or is not a NumPy array |
| Claude's tests | any `test_*` function raises |

Every failure goes back to Claude with its traceback, and Claude fixes the code or the test.
This repeats up to four times (`--rounds`, `max_rounds=`). A version that passes is written to
`<config dir>/plugins/<module>.py`, with its tests in `<config dir>/plugins/_forge_tests/`.
From then on it appears in the GUI menu, in the Claude tools and as
`from ewald.modifiers import <Class>`. A failing version is never saved.

## Running generated code

The generated module and tests run on your machine. The subprocess limits how long they run
and keeps them away from your plugin folder, but it is not a security sandbox. So
`forge(...)` runs nothing unless you pass `run_code=True`, and `ewald forge` asks before it
starts (`--yes` skips the question). `ewald forge --draft` and `forge(..., run_code=False)`
return Claude's first draft without running or saving it.

The subprocess gets a minimal environment: `PATH`, locale and library paths, and a scratch
`HOME`, but none of your API keys, tokens, cloud credentials or SSH agent. A version that
passes is shown and saved only after you agree, because a saved plugin is imported every time
ewald starts (`--yes` saves without asking; from Python, `forge(..., confirm=fn)` asks
`fn(result)` and saves when it returns True).

## Options

| CLI | Python | Meaning |
|---|---|---|
| `--yes` | `run_code=True` | run the checks and save a passing plugin without asking |
| `--draft` | `run_code=False` | print the first draft only |
| `--overwrite` | `overwrite=True` | replace a plugin file of the same name |
| `--rounds N` | `max_rounds=N` | attempts to fix failing checks |
| `--model` | `model=` | the Claude model (needs `pip install ewald[llm]` and an API key) |

`forge` also takes `client=`, any object with `messages.create` (the tests use a scripted
fake), `save=False` to verify without saving, and `on_event(kind, result)`, called with
`"draft"` and `"report"` in each round. The saved tests are ordinary modules, so you can run
them yourself with `python -m pytest <config dir>/plugins/_forge_tests`.

Read the generated module before you rely on it. The checks prove that it follows the plugin
contract and that it passes tests Claude wrote. They do not prove that those tests cover the
physics you had in mind, so the class docstring states the conventions Claude chose (cutoffs,
normalisation, units).
