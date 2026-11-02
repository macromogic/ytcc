import os
from collections.abc import Callable
from pathlib import Path


def default_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(config_home) / "ytcc" / "glossary.md"


def load_entries(path: Path) -> list[str]:
    """Read the "- " bullet lines of a Markdown glossary, notes included."""
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ")) and line[2:].strip():
            entries.append(line[2:].strip())
    return entries


def load_terms(path: Path) -> list[str]:
    """
    Read terms from a Markdown glossary: one term per "- " bullet line.

    Text after " — " or " -- " on a bullet is a note for humans and is not
    sent to Whisper. Headings and other lines are ignored.
    """
    terms = []
    for entry in load_entries(path):
        for sep in (" — ", " -- "):
            entry = entry.split(sep, 1)[0]
        term = entry.strip()
        if term and term not in terms:
            terms.append(term)
    return terms


def build_prompt(terms: list[str], extra: str | None = None) -> str | None:
    """Join an extra prompt and glossary terms into one Whisper prompt, or None."""
    parts = [p for p in (extra, ", ".join(terms)) if p]
    if not parts:
        return None
    return "; ".join(parts)


def fit_prompt(
    terms: list[str],
    extra: str | None,
    count_tokens: Callable[[str], int],
    limit: int,
) -> tuple[str | None, list[str]]:
    """
    Build the longest prompt within limit tokens.

    Terms are kept in glossary order, so put the important ones first. Returns
    the prompt and the terms left out. Raises ValueError when extra alone is
    too long.
    """
    if extra and count_tokens(extra) > limit:
        raise ValueError(f"--prompt is {count_tokens(extra)} tokens; Whisper takes at most {limit}.")
    kept = []
    for i, term in enumerate(terms):
        if count_tokens(build_prompt(kept + [term], extra)) > limit:
            return build_prompt(kept, extra), terms[i:]
        kept.append(term)
    return build_prompt(kept, extra), []
