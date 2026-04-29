"""Filesystem importers — Obsidian vaults, plain folders, single files.

Patha's MCP integration is the right entry point for AI assistants;
its CLI is the right entry point for one-off facts. But many users
already have years of writing in:

  - Obsidian vaults (Markdown + YAML frontmatter + [[wikilinks]])
  - Plain Markdown folders (notes/, journal/, blog/)
  - A pile of .txt files

This module bridges that pre-existing surface to Patha. Every Markdown
file becomes a candidate proposition (or several, if the file is long).
Frontmatter date → asserted_at. Folder structure → session_id. Wikilinks
→ explicit entity hints for the gaṇita extractor. Local-first to local-
first, no cloud round-trip.

Three entry points:

  - `import_file(path, memory)` — single .md/.txt file
  - `import_folder(path, memory)` — recursive walk; default for plain folders
  - `import_obsidian_vault(path, memory)` — folder walk with Obsidian-aware
    frontmatter + wikilink handling

CLI:

    patha import obsidian-vault ~/MyVault
    patha import folder ~/Documents/notes
    patha import file ~/Desktop/recipe.md

The importers use the existing `patha.Memory.remember()` API — every
file becomes a belief in the same store, fully searchable / superseded /
contradiction-detected like any other belief.

Honest scope
============

  - Only text formats: .md, .txt, .markdown. Skip everything else.
  - One file = one belief by default. Long files (>2000 chars) get
    split on H1/H2 boundaries when present, else paragraph runs.
  - Frontmatter parsing is YAML if pyyaml is installed; else a tiny
    regex fallback for `key: value` pairs.
  - Wikilinks become entity hints, not edges (yet). The songline graph
    already covers entity edges from the proposition text.
  - We do NOT modify the source files. Read-only ingest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}

# Match Obsidian frontmatter: starts at line 1, '---' delim, ends with '---'.
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<body>.*?)\n---\s*(?:\n|$)", re.DOTALL,
)

# Wikilinks: [[Some Page]] or [[Page|Display]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]*)?\]\]")

# Tags: #tag-name (alphanumerics + dashes/underscores; no leading digit)
_TAG_RE = re.compile(r"(?<![/\w])#([A-Za-z][\w/-]*)")


@dataclass
class ImportStats:
    """Counts returned from each import call. The CLI prints these."""

    files_seen: int = 0
    files_imported: int = 0
    files_skipped: int = 0
    beliefs_added: int = 0
    beliefs_reinforced: int = 0
    beliefs_superseded: int = 0
    skipped_paths: list[str] = field(default_factory=list)


# ─── Frontmatter parsing ─────────────────────────────────────────────


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body_text). Empty dict if no frontmatter."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    body = text[m.end():]
    fm_text = m.group("body")
    return _parse_frontmatter(fm_text), body


def _parse_frontmatter(fm_text: str) -> dict:
    """Try pyyaml → fallback to tiny `key: value` regex."""
    try:
        import yaml  # type: ignore
        parsed = yaml.safe_load(fm_text)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except Exception:
        pass
    # Minimal fallback: top-level key: value pairs only.
    out: dict = {}
    for line in fm_text.splitlines():
        m = re.match(r"^([A-Za-z][\w-]*)\s*:\s*(.*)\s*$", line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            # Strip surrounding quotes
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]
            out[key] = val
    return out


def _parse_date(value) -> datetime | None:
    """Best-effort parsing of frontmatter date fields."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    # YAML may parse plain dates ("2024-01-15") into datetime.date
    # rather than datetime.datetime; promote.
    import datetime as _dt
    if isinstance(value, _dt.date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        # Try ISO-ish formats first.
        for fmt in (
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d/%m/%Y",
        ):
            try:
                return datetime.strptime(value.strip(), fmt)
            except ValueError:
                continue
    return None


# ─── Splitting long files into propositions ──────────────────────────


def _split_into_propositions(text: str, max_chars: int = 2000) -> list[str]:
    """Long Markdown files become multiple propositions.

    Strategy:
      - Short text (≤ max_chars) → one proposition.
      - Otherwise split on H1 (`# `) or H2 (`## `) boundaries first.
      - If still over max_chars, split on blank lines.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Split on H1 / H2 boundaries.
    parts: list[str] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if re.match(r"^#{1,2}\s", line) and current:
            chunk = "".join(current).strip()
            if chunk:
                parts.append(chunk)
            current = [line]
        else:
            current.append(line)
    if current:
        chunk = "".join(current).strip()
        if chunk:
            parts.append(chunk)

    # Further split any chunk still over the cap on blank lines.
    final: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final.append(part)
            continue
        sub = re.split(r"\n\s*\n", part)
        buf: list[str] = []
        size = 0
        for s in sub:
            s = s.strip()
            if not s:
                continue
            if size + len(s) > max_chars and buf:
                final.append("\n\n".join(buf))
                buf = [s]
                size = len(s)
            else:
                buf.append(s)
                size += len(s)
        if buf:
            final.append("\n\n".join(buf))
    return [p for p in final if p.strip()]


# ─── Importer functions ─────────────────────────────────────────────


def import_file(
    path: Path,
    memory,
    *,
    obsidian: bool = False,
    session_id: str | None = None,
    stats: ImportStats | None = None,
) -> ImportStats:
    """Import a single text/Markdown file as one or more beliefs.

    Parameters
    ----------
    path
        File to import. Must be one of `.md`, `.markdown`, `.txt`.
    memory
        A `patha.Memory` instance.
    obsidian
        If True, parse Obsidian frontmatter for `date` / `created` /
        `modified`, and treat wikilinks/tags as entity hints.
    session_id
        Override the auto-derived session id (default: file path stem).
    stats
        Accumulate into this ImportStats; created if None.
    """
    stats = stats if stats is not None else ImportStats()
    if path.suffix.lower() not in _TEXT_SUFFIXES:
        stats.files_skipped += 1
        stats.skipped_paths.append(str(path))
        return stats

    stats.files_seen += 1
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        stats.files_skipped += 1
        stats.skipped_paths.append(str(path))
        return stats

    frontmatter: dict = {}
    if obsidian:
        frontmatter, text = _split_frontmatter(text)

    # Pick asserted_at: frontmatter date > file mtime > now.
    asserted_at: datetime | None = None
    for key in ("date", "created", "Created", "Date"):
        if key in frontmatter:
            asserted_at = _parse_date(frontmatter[key])
            if asserted_at is not None:
                break
    if asserted_at is None:
        try:
            asserted_at = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            asserted_at = datetime.now()

    sid = session_id or path.stem

    propositions = _split_into_propositions(text)
    if not propositions:
        stats.files_skipped += 1
        stats.skipped_paths.append(str(path))
        return stats

    for i, proposition in enumerate(propositions):
        # Optionally annotate context with the source path so the user
        # can trace propositions back to their origin file.
        result = memory.remember(
            proposition,
            asserted_at=asserted_at,
            session_id=sid,
            source_id=f"file:{path.name}#{i}",
            context=str(path),
        )
        action = result.get("action", "added")
        if action == "added":
            stats.beliefs_added += 1
        elif action == "reinforced":
            stats.beliefs_reinforced += 1
        elif action == "superseded":
            stats.beliefs_superseded += 1
    stats.files_imported += 1
    return stats


def import_folder(
    folder: Path,
    memory,
    *,
    obsidian: bool = False,
    follow_symlinks: bool = False,
    stats: ImportStats | None = None,
) -> ImportStats:
    """Recursively import every text file in `folder`.

    Files outside `_TEXT_SUFFIXES` are silently skipped. Hidden
    directories (e.g. `.git`, `.obsidian`) are not descended into.
    """
    stats = stats if stats is not None else ImportStats()
    if not folder.is_dir():
        stats.skipped_paths.append(str(folder))
        return stats

    for path in _walk_text_files(folder, follow_symlinks=follow_symlinks):
        # Use the path relative to the vault root as the session id
        # (without the file suffix). This groups related files (e.g.
        # everything under `journal/2024/`) under the same session.
        try:
            rel = path.relative_to(folder)
        except ValueError:
            rel = path.with_suffix("")
        # Session id: parent folder name, or '<vault-root>' if the
        # file sits directly in the root.
        rel_parent = rel.parent
        sid = (
            str(rel_parent).replace("/", "·")
            if str(rel_parent) not in (".", "")
            else folder.name
        )
        import_file(
            path, memory,
            obsidian=obsidian, session_id=sid, stats=stats,
        )
    return stats


def import_obsidian_vault(
    vault: Path,
    memory,
    *,
    follow_symlinks: bool = False,
) -> ImportStats:
    """Convenience wrapper: `import_folder(..., obsidian=True)`."""
    return import_folder(
        vault, memory,
        obsidian=True, follow_symlinks=follow_symlinks,
    )


# ─── Walker ─────────────────────────────────────────────────────────


def _walk_text_files(
    folder: Path, *, follow_symlinks: bool,
) -> Iterable[Path]:
    """Yield text files in `folder`, skipping hidden dirs."""
    stack: list[Path] = [folder]
    while stack:
        current = stack.pop()
        try:
            children = sorted(current.iterdir())
        except OSError:
            continue
        for child in children:
            if child.name.startswith("."):
                # skip .git, .obsidian, .DS_Store, etc.
                continue
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            if is_dir:
                if not follow_symlinks and child.is_symlink():
                    continue
                stack.append(child)
                continue
            if child.suffix.lower() in _TEXT_SUFFIXES:
                yield child


# ─── Entity-hint extraction (used by future gaṇita LLM/NER passes) ──


def extract_entity_hints_from_obsidian(text: str) -> list[str]:
    """Pull wikilinks + tags as entity-hint candidates.

    Used by the gaṇita layer to constrain entity binding when the
    Markdown explicitly tags topics. Returns lowercase canonical
    strings, deduplicated, in order of first appearance.
    """
    out: list[str] = []
    seen: set[str] = set()

    for m in _WIKILINK_RE.finditer(text):
        target = m.group(1).strip().split("#", 1)[0].strip()
        if not target:
            continue
        canon = target.lower()
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    for m in _TAG_RE.finditer(text):
        canon = m.group(1).lower()
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


__all__ = [
    "ImportStats",
    "import_file",
    "import_folder",
    "import_obsidian_vault",
    "extract_entity_hints_from_obsidian",
]
