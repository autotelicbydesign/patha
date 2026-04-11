"""Deterministic rule-based propositionizer.

Splits raw conversation turns into atomic propositions — the unit of indexing
for Patha. No LLM is invoked. The splitter cascades three rules:

1. List-item enumeration (when the turn is a bulleted or numbered list).
2. Sentence segmentation on terminal punctuation, with a small abbreviation
   blocklist to prevent false splits on "Dr.", "e.g.", decimal numbers, etc.
3. Within-sentence split on semicolons and independent-clause conjunctions
   (conjunctions only split when preceded by a comma, as a cheap heuristic
   for "clearly independent clause").

Each rule is a pure function and the cascade is deterministic — given the
same input, the same propositions come out, so downstream views and indexes
are reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Proposition:
    """A single atomic claim extracted from a conversation turn."""

    text: str
    session_id: str
    turn_idx: int
    prop_idx: int
    speaker: str | None = None
    timestamp: str | None = None

    @property
    def chunk_id(self) -> str:
        """Stable identifier used as the primary key across all indexes."""
        return f"{self.session_id}#t{self.turn_idx}#p{self.prop_idx}"


# Abbreviations we refuse to split sentences on. Small and conservative —
# false splits are worse than false merges because neighbor views (v2, v3, v4)
# depend on proposition ordering.
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "rev",
        "vs", "etc", "e.g", "i.e", "cf", "fig", "no", "approx",
        "inc", "ltd", "co", "corp",
    }
)

_ABBR_PATTERNS = tuple(
    re.compile(rf"\b{re.escape(a)}\.", re.IGNORECASE) for a in _ABBREVIATIONS
)

# Terminal sentence punctuation. Matches . ! ? followed by whitespace and an
# uppercase letter (or opening quote / paren). Explicitly rejects decimal
# numbers via the `(?<!\d\.)` lookbehind.
_SENT_BOUNDARY = re.compile(
    r"""
    (?<=[.!?])
    (?<!\d\.)
    \s+
    (?=[A-Z"'(\[])
    """,
    re.VERBOSE,
)

_SEMICOLON = re.compile(r"\s*;\s*")

# Split on "<comma> <conj> <lowercase>" — the comma is our cheap signal that
# what follows is an independent clause rather than a coordinate phrase.
_CONJ_SPLIT = re.compile(
    r",\s+(?=(?:and|but|so|yet|or|nor|for)\s+[a-z])",
    re.IGNORECASE,
)

# Leading conjunction stripped from the second-and-later pieces of a
# conjunction split — so "I went home, and I slept" yields
# ["I went home", "I slept"] rather than [..., "and I slept"].
_CONJ_LEADING = re.compile(r"^(?:and|but|so|yet|or|nor|for)\s+", re.IGNORECASE)

# List item markers at line start: "1.", "1)", "a.", "a)", "-", "*", "•".
_LIST_ITEM = re.compile(
    r"""
    ^\s*
    (?:
        \d+[.)]\s+
      | [a-zA-Z][.)]\s+
      | [-*\u2022]\s+
    )
    """,
    re.VERBOSE | re.MULTILINE,
)


def _split_sentences(text: str) -> list[str]:
    """First pass: split on sentence-terminal punctuation.

    Abbreviations are temporarily rewritten with a NUL sentinel so the
    boundary regex does not fire on them; sentinels are restored afterwards.
    """
    protected = text
    for pattern in _ABBR_PATTERNS:
        protected = pattern.sub(lambda m: m.group(0)[:-1] + "\x00", protected)

    parts = _SENT_BOUNDARY.split(protected)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def _split_clauses(sentence: str) -> list[str]:
    """Second pass: split a sentence on semicolons and independent-clause conjunctions."""
    out: list[str] = []
    for piece in _SEMICOLON.split(sentence):
        piece = piece.strip()
        if not piece:
            continue
        sub = _CONJ_SPLIT.split(piece)
        for i, s in enumerate(sub):
            s = s.strip().rstrip(",")
            if i > 0:
                s = _CONJ_LEADING.sub("", s)
            if s:
                out.append(s)
    return out


def _split_list_items(text: str) -> list[str] | None:
    """Third pass: if the turn is a list, split per item.

    Returns None if fewer than two list markers are detected, so callers can
    fall back to the sentence/clause cascade on the full text.
    """
    matches = list(_LIST_ITEM.finditer(text))
    if len(matches) < 2:
        return None

    items: list[str] = []
    if matches[0].start() > 0:
        lead = text[: matches[0].start()].strip()
        if lead:
            items.append(lead)
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        item = text[start:end].strip()
        if item:
            items.append(item)
    return items


def propositionize(
    text: str,
    *,
    session_id: str,
    turn_idx: int,
    speaker: str | None = None,
    timestamp: str | None = None,
) -> list[Proposition]:
    """Split a conversation turn into atomic propositions.

    Cascade: list-items (if the turn is a list) -> sentences -> clauses.
    Pure function of the input; no LLM invoked.
    """
    text = text.strip()
    if not text:
        return []

    units = _split_list_items(text) or [text]

    props: list[str] = []
    for unit in units:
        for sentence in _split_sentences(unit):
            for clause in _split_clauses(sentence):
                props.append(clause)

    return [
        Proposition(
            text=p,
            session_id=session_id,
            turn_idx=turn_idx,
            prop_idx=i,
            speaker=speaker,
            timestamp=timestamp,
        )
        for i, p in enumerate(props)
    ]
