# Writing docs

The manual is `docs/`, built with MkDocs Material from `mkdocs.yml`. The same Markdown files
read correctly on GitHub, which is where the GUI's Help menu points by default, so pages use
plain Markdown: no MkDocs-only syntax outside `reference/api.md`.

```bash
pip install mkdocs-material "mkdocstrings[python]"
mkdocs serve              # live preview at http://127.0.0.1:8000
mkdocs build --strict     # what CI runs; fails on broken links and missing pages
```

`reference/api.md` is generated from docstrings by mkdocstrings, which reads the source
statically, so building the docs does not need the C++ extension.

## Where things go

- `docs/user/`: how to use ewald. Task-oriented, with runnable examples.
- `docs/plugins/`: how to extend ewald without changing it. Every extension point has a page.
- `docs/developer/`: how ewald works inside. `CLAUDE.md` is the short version for agents and
  should agree with these pages.
- `docs/ROADMAP.md`: plan and state.
- `docs/design/`, `docs/research/`: design notes and research, kept for history and not in the
  site navigation.

## When to update

- A new or renamed modifier: add it to `docs/user/modifiers.md` (`tests/test_docs.py` checks),
  to README's list, to `__all__`, and regenerate `llms.txt`.
- A new reader, option, CLI command or GUI feature: the matching page under `docs/user/`.
- A new extension point or plugin API change: `docs/plugins/`, and the state column on
  `docs/plugins/index.md`.
- New pages go into `nav` in `mkdocs.yml`; `mkdocs build --strict` reports pages that are not.

## Style

English, full sentences, present tense. Lead with what the reader does, then the example, then
details. Use OVITO's property names in examples, and examples that run as written.

## Hosting

The repository is private, so the built site is not published anywhere yet. CI uploads it as
the `docs-site` artifact of each run; Preferences › General › Documentation URL points the GUI's
Help menu at a hosted copy once there is one.
