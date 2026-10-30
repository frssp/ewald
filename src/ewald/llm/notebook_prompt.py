"""What the AI chat side bar tells the model about the notebook beside it (Qt-free).

The GUI's chat (:mod:`ewald.gui.ask_panel`) appends :data:`NOTEBOOK_NOTE` to the agent's system
prompt and starts each question with the notebook (code, cell numbers and outputs), the errors of
its failed cells and what the 3D view shows; the Python blocks of an answer become cells, or
replace the cell doing the same job (:func:`code_blocks`, :func:`ewald.llm.notebook_tools.place_code`).
The AI eval suite (:mod:`ewald.llm.evals`) builds the same prompts, so a score measures what the
GUI user gets.
"""
from __future__ import annotations

import re

NOTEBOOK_NOTE = """

You are the chat side bar of ewald's notebook window. The user's notebook is a percent-format \
Python file (`# %%` cells, one namespace) where `pipe` is the pipeline: step cells are one \
`pipe.add(Modifier(...))` (every modifier of `ewald.modifiers` is imported by name), \
view cells are one `show(pipe, ...)` (a plot) or one `view(pipe, ...)` (the 3D view), and `pipe = load(path)` \
opens a file; python_api describes these cells, show, view and paint. When the user's message \
starts with the notebook, that is its current state: each cell comes after a `# [cell N · index k · \
kind · state]` line (N as the user counts, k for the cell tools) and before a `# [output: ...]` line \
with what it printed or drew; these lines are not part of the cells. Work on the notebook as a whole: \
before you write a cell, look for one that already does the job. When the user wants to change \
something the notebook already does (a step's option, a plot's table or range, the view, a fix to \
code), edit that cell and run it again; do not add a new cell that repeats it with other options, \
since the old cell would stay and a second step of the same modifier overwrites the first one's \
results. Add a cell only for something the notebook does not do yet, or when the user asks to keep \
both (then pass again=true). {code_cells} Write each ```python \
block as one complete cell that runs in that namespace; prefer `pipe.add(...)` steps over ad-hoc \
loops. When the user asks you to \
change the notebook itself, use the cell tools: list_cells and read_cell (sources and outputs), \
insert_cell, edit_cell (step parameters through params), move_cell and delete_cell. {run_cells} \
Tool names are not Python: never put a tool call in a ```python block or a cell; write the cell's own \
code (view(pipe, ...), pipe.add(Modifier(...)), pipe = load(path)). Say which \
cells you changed, and when something is already drawn, say which cell does it; call a cell "cell N" \
with N counted from 1 as the user sees them (index + 1), never by its index (the chat also links the \
cells your tools wrote)."""

#: What the note above says about running cells (Preferences › AI › Let the AI run cells).
RUN_CELLS_NOTE = {
    True: "The user lets you run cells: insert_cell and edit_cell run the code cell they write at once "
          "and return its outputs, and run_cell runs any cell, so run the cells you write, read their "
          "outputs and fix errors yourself instead of asking the user to run them.",
    False: "The user has not let you run cells: cells you write wait, marked not run, until the user "
           "runs them, so tell the user which cells to run. add_modifier, set_view, load_structure and "
           "export_file write their own cell and run it, and the compute and describe tools read results "
           "without a cell.",
}

#: How the note above describes the ```python blocks of an answer (Preferences › AI › Code in answers).
CODE_CELLS_NOTE = {
    "insert": "Every ```python block of your answer is added to the notebook ({run}), in order: as a new "
              "cell, or in place of the one cell that does the same job (the same modifier's step, a plot "
              "of the same table, the same view tab). So put only code meant for the notebook in ```python "
              "blocks (other code, such as a one-line illustration, in a ```text block), do not also "
              "insert_cell the same code, and change existing cells with edit_cell rather than with a block.",
    "button": "Every ```python block of your answer gets an \"Insert as notebook cell\" button.",
}

MAX_NOTEBOOK = 24000  # characters of the notebook sent with a question
_CODE_BLOCK = re.compile(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", re.S)


def notebook_system(base: str, run_cells: bool = False, code_cells: str = "insert") -> str:
    """``base`` (the agent's system prompt) plus the notebook note for these Preferences › AI
    settings: ``run_cells`` (Let the AI run cells) and ``code_cells`` ('insert' or 'button')."""
    mode = code_cells if code_cells in CODE_CELLS_NOTE else "insert"
    code = CODE_CELLS_NOTE[mode].format(run="and run" if run_cells else "not run")
    return base + NOTEBOOK_NOTE.format(code_cells=code, run_cells=RUN_CELLS_NOTE[bool(run_cells)])


def notebook_part(code: str) -> str:
    """The notebook as a question carries it (:func:`ewald.llm.notebook_tools.notebook_context`: its
    code with cell numbers and outputs), cut after :data:`MAX_NOTEBOOK` characters."""
    if len(code) > MAX_NOTEBOOK:
        code = code[:MAX_NOTEBOOK] + "\n# … (notebook truncated)\n"
    return f"The notebook now reads:\n```python\n{code}```"


def errors_part(errors: str) -> str:
    """The failed cells' errors as a question carries them ('' when they are gone)."""
    return (f"These notebook cells failed when they last ran:\n```text\n{errors}```" if errors
            else "The notebook's cells no longer show errors.")


def view_part(view: str) -> str:
    return f"The 3D view now shows: {view}"


_PATHISH = re.compile(r"\S*[/\\]\S*|\S+\.(?:gz|xyz|extxyz|cif|dat|yaml|csv|lammpstrj|h5)\b", re.I)


def related_part(question: str, limit: int = 4) -> str:
    """The steps and functions of the API reference that match ``question``'s words (paths and file
    names left out), one line each, as a question carries them ('' when nothing matches)."""
    from .reference import FUNCTIONS, _documented, _first_sentence, _doc, _signature, search_reference

    from .. import modifiers
    from ..notebook import IMPORTS

    imported = set(re.findall(r"\w+", IMPORTS)) | set(modifiers.__all__)
    home = {name: module for module, names in FUNCTIONS for name in names}
    lines = []
    hits = search_reference(_PATHISH.sub(" ", question), 3 * limit, with_plugins=True)
    for hit in hits:
        if hit["score"] < max(6.0, 0.5 * hits[0]["score"]):   # weak matches are noise ("again")
            break
        m = re.match(r"- `(\w+)\(", hit["text"])
        obj = _documented(m.group(1)) if m else None
        if obj is None:
            continue
        doc = _first_sentence(_doc(obj))
        where = "" if m.group(1) in imported else f" (from {home.get(m.group(1), obj.__module__)} import {m.group(1)})"
        lines.append(f"- `{m.group(1)}{_signature(obj)}`{where}" + (f": {doc}" if doc else ""))
        if len(lines) == limit:
            break
    if not lines:
        return ""
    return ("Steps and functions of ewald that may do this (search_api returns their whole docstrings; use "
            "them rather than your own code where they fit):\n" + "\n".join(lines))


def code_blocks(text: str) -> list[str]:
    """The Python code blocks of a Markdown answer (fenced as python, py or untagged)."""
    return [m.group(1).rstrip() + "\n" for m in _CODE_BLOCK.finditer(text) if m.group(1).strip()]


__all__ = ["CODE_CELLS_NOTE", "MAX_NOTEBOOK", "NOTEBOOK_NOTE", "RUN_CELLS_NOTE", "code_blocks", "errors_part",
           "notebook_part", "notebook_system", "related_part", "view_part"]
