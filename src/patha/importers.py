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
from typing import Any, Iterable


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


# ─── Claude conversation-history export ─────────────────────────────


# Heuristic-only filtering. We don't try to "extract facts" with an LLM
# at import time — Patha's existing karaṇa extractor handles tuple
# extraction at ingest. Here we just decide which user messages are
# worth ingesting at all.
_QUESTION_STARTS = (
    "what", "why", "how", "when", "where", "who", "which", "can ",
    "could ", "do ", "does ", "did ", "is ", "are ", "was ", "were ",
    "should ", "would ", "will ", "may ", "might ",
)


def _looks_like_question_or_command(text: str) -> bool:
    """Skip messages that are questions to Claude or imperatives, not
    statements of fact about the user. Conservative heuristic."""
    s = text.strip().lower()
    if not s:
        return True
    if s.endswith("?"):
        return True
    # First word is a question word OR an imperative verb
    first_word = s.split(None, 1)[0] if s.split() else ""
    if first_word in {
        "what", "why", "how", "when", "where", "who", "which",
        "explain", "tell", "show", "give", "list", "create", "make",
        "write", "draft", "generate", "fix", "change", "update",
        "rewrite", "rephrase", "translate", "summarize", "summarise",
        "review", "check", "help", "let's", "lets",
    }:
        return True
    return False


def _split_user_message_into_propositions(text: str) -> list[str]:
    """Split a user's chat message into candidate belief propositions.

    Strategy:
      - Strip code blocks (text wrapped in ```...```) — those are
        snippets, not facts.
      - Split on sentence boundaries (`.`, `!`, `\\n\\n`).
      - Drop fragments shorter than 12 chars (artifacts of splitting).
      - Drop fragments that look like questions or commands.
    """
    # Strip fenced code blocks; they're never user beliefs.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    # Strip inline code spans.
    text = re.sub(r"`[^`]*`", " ", text)
    # Split into sentence-ish chunks.
    chunks = re.split(r"(?<=[.!])\s+|\n{2,}", text)
    out: list[str] = []
    for chunk in chunks:
        s = chunk.strip().strip("-*•").strip()
        if len(s) < 12:
            continue
        if _looks_like_question_or_command(s):
            continue
        out.append(s)
    return out


def _load_claude_conversations(path: Path) -> list[dict]:
    """Load conversations from a Claude data export.

    Accepts:
      - A `.zip` file (Claude's standard export format)
      - A directory containing the unzipped contents
      - A `conversations.json` file directly

    Returns a list of conversation dicts in Claude's export schema:
      {uuid, name, created_at, chat_messages: [...]}
    """
    import json
    import zipfile

    convos: list[dict] = []

    def _maybe_load(raw: bytes | str) -> None:
        try:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            data = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "chat_messages" in item:
                    convos.append(item)
        elif isinstance(data, dict):
            if "chat_messages" in data:
                convos.append(data)
            # Some exports wrap in {"conversations": [...]}
            elif "conversations" in data and isinstance(data["conversations"], list):
                convos.extend(c for c in data["conversations"] if isinstance(c, dict))

    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                # Look for the conversations dump specifically; many exports
                # also include `users.json` and `projects.json` we should skip.
                lower = name.lower()
                if "conversation" not in lower and "messages" not in lower:
                    continue
                with zf.open(name) as f:
                    _maybe_load(f.read())
    elif path.is_dir():
        for jf in sorted(path.rglob("*.json")):
            lower = jf.name.lower()
            if "conversation" not in lower and "messages" not in lower:
                continue
            _maybe_load(jf.read_bytes())
    elif path.is_file() and path.suffix.lower() == ".json":
        _maybe_load(path.read_bytes())

    return convos


def import_claude_export(
    path: Path,
    memory: Any,
    *,
    min_message_chars: int = 30,
    max_per_message: int = 10,
    sentence_split: bool = True,
    stats: ImportStats | None = None,
    verbose: bool = False,
) -> ImportStats:
    """Import Claude's conversation-history export into a Patha store.

    Anthropic lets you download all your Claude conversations as a ZIP
    from Settings → Privacy → Export data. This function reads that
    export and ingests every user-side message (or every sentence, if
    `sentence_split=True`) as a Patha belief, with the original
    timestamp preserved.

    Only **user messages** are ingested. Assistant replies are skipped
    — those are Claude's outputs, not the user's beliefs.

    Filtering applied per message:
      - Skip if shorter than `min_message_chars`
      - Skip if it looks like a question or command to Claude
      - Skip code blocks (stripped before sentence-splitting)

    Parameters
    ----------
    path
        Path to a `.zip`, a directory of unzipped JSON, or a single
        `conversations.json` file.
    memory
        A `patha.Memory` instance.
    min_message_chars
        Minimum length in characters for a user message to be considered
        for ingestion. Default 30 — short messages ("yes", "thanks") are
        almost never beliefs.
    max_per_message
        Cap on propositions extracted per message after sentence
        splitting. Prevents a single 5,000-word brain-dump from drowning
        the store. Default 10.
    sentence_split
        If True (default), split each user message into sentence-level
        propositions and ingest each one separately. If False, ingest
        the whole message as a single belief.
    stats
        Optional pre-existing ImportStats to accumulate into.
    verbose
        Print a line per ingested conversation.
    """
    stats = stats or ImportStats()

    convos = _load_claude_conversations(path)
    if not convos:
        return stats

    for convo in convos:
        msgs = convo.get("chat_messages") or []
        if not msgs:
            continue
        stats.files_seen += 1
        convo_uuid = convo.get("uuid") or ""
        convo_name = convo.get("name") or "(untitled)"
        ingested_in_convo = 0

        for msg in msgs:
            sender = (msg.get("sender") or "").lower()
            if sender != "human":
                continue
            text = msg.get("text") or ""
            if len(text) < min_message_chars:
                continue
            if _looks_like_question_or_command(text):
                continue
            if sentence_split:
                props = _split_user_message_into_propositions(text)[:max_per_message]
            else:
                props = [text.strip()]
            if not props:
                continue

            ts = msg.get("created_at") or convo.get("created_at")
            asserted_at = _parse_claude_timestamp(ts)

            for j, prop in enumerate(props):
                try:
                    ev = memory.remember(
                        prop,
                        asserted_at=asserted_at,
                        session_id=convo_uuid or None,
                        source_id=f"claude-export:{convo_uuid}:{msg.get('uuid','?')}:{j}",
                    )
                    action = (
                        ev.get("action") if isinstance(ev, dict)
                        else getattr(ev, "action", "?")
                    )
                    if action == "added":
                        stats.beliefs_added += 1
                    elif action == "reinforced":
                        stats.beliefs_reinforced += 1
                    elif action == "superseded":
                        stats.beliefs_superseded += 1
                    ingested_in_convo += 1
                except Exception:
                    # Never fail the whole import on one message
                    continue

        if ingested_in_convo:
            stats.files_imported += 1
            if verbose:
                print(f"  + {ingested_in_convo:>4d} props from: {convo_name[:60]}")
        else:
            stats.files_skipped += 1

    return stats


def _parse_claude_timestamp(ts: str | None) -> datetime | None:
    """Parse Claude export's ISO timestamps. They look like
    '2024-08-15T14:23:01.123456+00:00' or with a trailing 'Z'."""
    if not ts:
        return None
    try:
        # Python's fromisoformat handles most variants in 3.11+
        s = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


__all__ = [
    "ImportStats",
    "import_file",
    "import_folder",
    "import_obsidian_vault",
    "import_claude_export",
    "extract_entity_hints_from_obsidian",
]
