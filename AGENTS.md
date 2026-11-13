# Agent Instructions

This project uses Nora for research project state tracking.

## Git

Agents may commit and push to this repository. Follow the existing commit style (`type: summary`, then a bullet body), run `uv run pytest` before committing, and `git fetch` first; stop if the branch is behind origin.

## Startup protocol

Before working on this project:

1. Run `nora root` to resolve the active workspace (the nearest ancestor with `.nora/`). Do not assume `./.nora` — you may be in a subdirectory, and sibling directories may have their own `.nora`.
2. From the resolved workspace root, read:
   - `.nora/CONTEXT_BRIEF.md`
   - `.nora/PROJECT_STATE.yaml`
   - `.nora/NEXT_ACTIONS.md`
   - `.nora/OPEN_LOOPS.md`

## Operating rules

1. Prefer Nora CLI commands for state-changing operations; do not hand-build what a `nora` command already does.
2. Work in exactly one workspace at a time. Sibling `.nora` directories are other workspaces of the same research project, not conflicts — never read or merge their state implicitly.
3. Read compact state files and reports before raw logs or long files.
4. Do not manually edit structured state when CLI support exists; propose changes when it does not.
5. Do not create `.nora/` directories yourself — that is `nora new` / `nora <module> init`'s job, and nested `.nora` directories are conflicts.

## Decision gate

`.nora/decisions/decisions.yaml` is the decision gate. When your work produces something that changes research state — a paper worth reading, a citation to add, a substantive writing change, an experiment idea — append a proposal entry with `status: pending` instead of acting on it.

- Only the user flips a proposal to `approved`/`rejected` (or explicitly tells you to).
- Never act on a `pending` proposal as if it were approved.
- Never promote papers, citations, writing changes, or research decisions to final states without an approved entry.

## What requires explicit user approval

- Promoting any `pending` decision.
- Running `nora new` or any `nora <module> init`.
- Overwriting or restructuring Nora state files (appending session updates is fine when asked).
- Any edit to `.bib`/`.tex` beyond what the user asked for in this session.
- Introducing a new LaTeX package or macro (including formalizing placeholder commands like `\work[...]` — those are removed, never defined).

## Preferred skills

Use `nora-project-manager` for project state and session continuity. Additional Nora skills cover optional, separately-initialized modules — use them when the relevant work comes up, not by default:

- `nora-project-manager` — project state, lifecycle, and session continuity (this is the one to use at the start/end of every session)
- `nora-citation-auditor` — BibTeX hygiene, LaTeX citation checks, and claim-citation support review (`.nora/citation/`, enabled via `nora citation init`)
- `nora-literature-manager` — literature search, triage, reading queues, and related-work maps (`.nora/literature/`, enabled via `nora literature init`)
- `nora-writing-assistant` — polishing, restructuring, and overclaim checks for manuscript prose (`.nora/writing/`, enabled via `nora writing init`; also works standalone without any `.nora/` project)

Do not assume the citation, literature, or writing modules are initialized just because this project uses Nora — check for `.nora/citation/`, `.nora/literature/`, `.nora/writing/` (or run `nora doctor`) before using them, and ask before running the relevant `nora <module> init` if one is missing.

## Session behavior

At session start:

- Summarize current project status.
- Identify the active next action.
- Avoid expanding scope unless requested.

At session end:

- Propose updates to `.nora/SESSION_LOG.md`.
- Propose updates to `.nora/NEXT_ACTIONS.md`.
- Propose updates to `.nora/PROJECT_STATE.yaml`.
- Propose updates to `.nora/OPEN_LOOPS.md`.

## Human review policy

Do not silently overwrite Nora state files unless explicitly asked.

For uncertain or high-impact changes, propose updates first.
