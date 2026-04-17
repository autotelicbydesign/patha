"""Abhāva — four-fold epistemology of negation (Nyāya).

The Nyāya school treats absence (*abhāva*) as a first-class category
of knowledge. Four kinds:

  PRAGABHAVA       — prior absence. 'The pot doesn't exist YET.'
                     A negation of a state before its cause has
                     produced it. In Patha: 'I haven't started the
                     marathon training yet' (future positive state).

  PRADHVAMSABHAVA  — destructive / posterior absence. 'The pot IS
                     BROKEN now.' A negation after the destruction of
                     a prior state. In Patha: 'I no longer eat meat'
                     (negation linked to a destroyed prior belief).

  ANYONYABHAVA     — mutual / reciprocal absence. 'A is not B.'
                     Category-level distinction. In Patha: 'I am not
                     a nurse' (identity negation, no temporal anchor).

  ATYANTABHAVA     — absolute absence. 'A sky-flower never existed
                     and never will.' Strong negation invalidating
                     the entire class of claim across time. In Patha:
                     'I have never lived in Paris' — not merely
                     'not currently' but 'not in any known timeframe.'

Classifier is rule-based and conservative. Returns UNKNOWN when no
pattern fires so callers fall back to existing supersession logic
(validity-decay as weak abhāva, etc.). This module does NOT modify
how beliefs are stored — it adds a classification layer that downstream
code (render_summary, direct_answer) can consume.

Why it matters: treating "I no longer eat meat" (PRADHVAMSABHAVA) and
"I am not a nurse" (ANYONYABHAVA) and "I have never done X"
(ATYANTABHAVA) identically — as generic propositions — loses
important structure. The former should link to a prior positive
belief being destroyed; the latter should invalidate the entire
proposition-space without any prior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class AbhavaKind(str, Enum):
    """Classical four-fold absence taxonomy + UNKNOWN fallback."""

    PRAGABHAVA = "pragabhava"           # prior absence
    PRADHVAMSABHAVA = "pradhvamsabhava"  # destructive absence
    ANYONYABHAVA = "anyonyabhava"       # mutual (category) absence
    ATYANTABHAVA = "atyantabhava"       # absolute absence
    NONE = "none"                        # not an abhāva claim
    UNKNOWN = "unknown"                  # abhāva but kind unclear


@dataclass(frozen=True)
class AbhavaInference:
    """Output of abhāva classification."""

    kind: AbhavaKind
    cue: str                    # matched phrase, for debugging
    referenced_state: str | None # the positive state being negated (if any)


# ─── Patterns ──────────────────────────────────────────────────────

# ATYANTABHAVA: absolute / never-in-any-timeframe negation
_ATYANTABHAVA_PATTERNS: list[re.Pattern] = [
    # 'I have never', 'I've never', 'we have never', 'we've never',
    # 'I never have', 'we never have'
    re.compile(
        r"\b(?:i|we)(?:\s+have|\s*'ve|\s+never\s+have)\s+never\b",
        re.IGNORECASE,
    ),
    # Catch "I've never" separately (apostrophe makes 'I' look attached to 've')
    re.compile(
        r"\b(?:i|we)'ve\s+never\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:not\s+once|in\s+my\s+life\s+i\s+have\s+never)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:there\s+has\s+never\s+been|there\s+never\s+was)\b",
        re.IGNORECASE,
    ),
]

# PRADHVAMSABHAVA: destructive — 'no longer', 'stopped', 'used to be but not anymore'
_PRADHVAMSABHAVA_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\bno\s+longer\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i|we)\s+(?:stopped|quit|gave\s+up|ended|finished|dropped)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i|we)\s+used\s+to\s+\w+\s+(?:but|however)\b",
        re.IGNORECASE,
    ),
    # "I don't X Y Z anymore" — allow multiple words between don't and
    # the 'anymore' suffix
    re.compile(
        r"\b(?:i|we)\s+don'?t\s+\w+(?:\s+\w+){0,5}\s+(?:anymore|any\s+more|any\s+longer)\b",
        re.IGNORECASE,
    ),
]

# PRAGABHAVA: prior — 'haven't X yet', 'not started', 'before'
_PRAGABHAVA_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(?:i|we)\s+(?:haven'?t|have\s+not|hasn'?t|has\s+not)\s+\w+.{0,30}\byet\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i|we)\s+(?:hasn'?t|haven'?t|have\s+not)\s+(?:started|begun)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bnot\s+(?:yet|still\s+not)\b.{0,40}",
        re.IGNORECASE,
    ),
]

# ANYONYABHAVA: identity / category negation — 'I am not a X', 'X is not Y'
_ANYONYABHAVA_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(?:i|we|he|she|they)\s+(?:am|is|are)\s+not\s+(?:a|an|the)\s+\w+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\w+\s+is\s+not\s+(?:a|an|the)\s+\w+",
        re.IGNORECASE,
    ),
]


_REGISTRY: list[tuple[AbhavaKind, list[re.Pattern]]] = [
    # Order matters: most-specific first. ATYANTABHAVA has the most
    # distinctive cues; then PRADHVAMSABHAVA (explicit destruction);
    # then PRAGABHAVA (not-yet); then ANYONYABHAVA (category).
    (AbhavaKind.ATYANTABHAVA, _ATYANTABHAVA_PATTERNS),
    (AbhavaKind.PRADHVAMSABHAVA, _PRADHVAMSABHAVA_PATTERNS),
    (AbhavaKind.PRAGABHAVA, _PRAGABHAVA_PATTERNS),
    (AbhavaKind.ANYONYABHAVA, _ANYONYABHAVA_PATTERNS),
]


# ─── Public API ────────────────────────────────────────────────────

def classify_abhava(proposition: str) -> AbhavaInference:
    """Classify a proposition's negation kind.

    Returns AbhavaKind.NONE when the proposition isn't a negation at
    all (no negation cue detected). Returns AbhavaKind.UNKNOWN when
    negation is present but doesn't match any specific kind.
    """
    if not _has_negation_cue(proposition):
        return AbhavaInference(
            kind=AbhavaKind.NONE, cue="", referenced_state=None
        )

    for kind, patterns in _REGISTRY:
        for pat in patterns:
            m = pat.search(proposition)
            if m is not None:
                referenced = _extract_referenced_state(proposition, kind)
                return AbhavaInference(
                    kind=kind, cue=m.group(0), referenced_state=referenced
                )

    return AbhavaInference(
        kind=AbhavaKind.UNKNOWN,
        cue="<negation-present>",
        referenced_state=None,
    )


_NEGATION_CUES: frozenset[str] = frozenset({
    "not", "never", "no ", "don't", "didn't", "doesn't",
    "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't",
    "won't", "wouldn't", "shouldn't", "couldn't",
    "stopped", "quit", "gave up", "ended", "no longer",
    "used to",
})


def _has_negation_cue(text: str) -> bool:
    lower = text.lower()
    return any(cue in lower for cue in _NEGATION_CUES)


def _extract_referenced_state(text: str, kind: AbhavaKind) -> str | None:
    """Best-effort extraction of the positive state being negated.

    For PRADHVAMSABHAVA: 'I stopped smoking' → referenced = 'smoking'.
    For PRAGABHAVA: 'I haven't finished the report yet' → 'the report'.
    For ATYANTABHAVA: 'I have never lived in Paris' → 'lived in Paris'.
    For ANYONYABHAVA: no referenced state (identity negation).

    Returns None when extraction fails; this is a hint for downstream
    code, not a contract. Small regex-based extractor; good enough for
    v0.4 without hauling in a parser.
    """
    if kind == AbhavaKind.ANYONYABHAVA:
        return None

    if kind == AbhavaKind.PRADHVAMSABHAVA:
        # 'stopped X' / 'quit X' / 'no longer X' / "don't X anymore"
        for pat in [
            r"\bstopped\s+(\w+(?:\s+\w+){0,3})",
            r"\bquit\s+(\w+(?:\s+\w+){0,3})",
            r"\bgave\s+up\s+(\w+(?:\s+\w+){0,3})",
            r"\bno\s+longer\s+(\w+(?:\s+\w+){0,3})",
            r"\bdon'?t\s+(\w+(?:\s+\w+){0,3})\s+anymore",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".,;")

    if kind == AbhavaKind.PRAGABHAVA:
        for pat in [
            r"\bhaven'?t\s+(\w+(?:\s+\w+){0,5})\s+yet",
            r"\bhaven'?t\s+started\s+(\w+(?:\s+\w+){0,3})",
            r"\bnot\s+yet\s+(\w+(?:\s+\w+){0,3})",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".,;")

    if kind == AbhavaKind.ATYANTABHAVA:
        for pat in [
            r"\b(?:have\s+never|has\s+never|never\s+have)\s+(\w+(?:\s+\w+){0,5})",
            r"\bnever\s+(\w+(?:\s+\w+){0,5})",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".,;")

    return None
