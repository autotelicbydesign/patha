"""Pramāṇa detection — tagging beliefs with their source of valid knowledge.

Rule-based classifier that inspects the proposition text for linguistic
cues and returns a Pramana enum value. Conservative: when no confident
cue fires, returns Pramana.UNKNOWN rather than guessing.

Detection precedence (most-specific first):

  1. SHABDA        — "X told me / said / reports" constructions
  2. PRATYAKSA     — first-person perception verbs ("I saw / heard /
                      felt / noticed / observed")
  3. ANUPALABDHI   — absence-based: "I don't / no / not X anymore"
                     when paired with a perception verb
  4. ARTHAPATTI    — postulation: "must be / must have / therefore /
                     it follows / so he must..." (inference from
                     circumstance, distinct from simple deduction)
  5. ANUMANA       — inference cues: "I think / believe / probably /
                     seems / must be (without circumstance)"
  6. UPAMANA       — comparison / analogy: "like / similar to / as X is"

These are heuristics. A production pipeline would use a small classifier
fine-tuned on pramāṇa-annotated data (which doesn't exist yet — building
such a corpus is v0.4+ work).

Design rationale:
  - Following the same pattern as validity_extraction.py: narrow rules,
    silent when unsure, never overclaim.
  - Exposed list_patterns() for documentation and testing.
  - All successful detections come with a 'cue' string describing what
    matched, for debugging and eventual ML training data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from patha.belief.types import Pramana


@dataclass(frozen=True)
class PramanaInference:
    """Result of pramāṇa auto-detection on a proposition."""

    pramana: Pramana
    cue: str  # The phrase that matched, for debugging/training data


# ─── Pattern library ────────────────────────────────────────────────

# Shabda: testimony constructions
_SHABDA_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(?:my|the|a|an|her|his|their)?\s*\w*\s*(?:told\s+me|said\s+to\s+me|told\s+us|mentioned|informed\s+me|reported|reports)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\baccording\s+to\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i\s+was\s+told|we\s+were\s+told|i\s+heard\s+that|i\s+heard\s+from)\b",
        re.IGNORECASE,
    ),
]

# Pratyaksa: first-person direct perception
_PRATYAKSA_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(?:i|we)\s+(?:saw|see|watched|witnessed|observed|noticed|heard(?!\s+(?:that|from)))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i|we)\s+(?:felt|smelled|tasted|touched)\b",
        re.IGNORECASE,
    ),
]

# Anupalabdhi: absence-based inference
_ANUPALABDHI_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(?:i|we)\s+(?:don'?t|didn'?t|do\s+not|did\s+not)\s+(?:see|hear|notice|observe|find)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:there\s+is|was)\s+no\b.*\b(?:sign|trace|evidence)\b",
        re.IGNORECASE,
    ),
]

# Arthapatti: postulation / inference from circumstance
_ARTHAPATTI_PATTERNS: list[re.Pattern] = [
    # "must have been/be/have" and "must be X" where X is a state/location
    re.compile(
        r"\bmust\s+(?:have\s+been|be|have)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:since|because|given\s+that).{1,60}\b(?:therefore|so|hence|it\s+follows)\b",
        re.IGNORECASE | re.DOTALL,
    ),
]

# Anumana: general inference cues
_ANUMANA_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(?:i\s+think|i\s+believe|i\s+suspect|probably|seems\s+(?:like|that)|likely|apparently)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:if|when)\b.{1,50}\b(?:then|so)\b",
        re.IGNORECASE,
    ),
]

# Upamana: comparison / analogy
_UPAMANA_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(?:like|similar\s+to|resembles|as\s+if|much\s+like|akin\s+to)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:compared\s+to|in\s+comparison\s+with|as\s+\w+\s+is)\b",
        re.IGNORECASE,
    ),
]


# Evaluation order: shabda first (most specific), then pratyaksa,
# anupalabdhi, arthapatti, anumana, upamana. Upamana last because
# "like" is very common and often non-evidential.
_PATTERN_REGISTRY: list[tuple[Pramana, list[re.Pattern], str]] = [
    (Pramana.SHABDA, _SHABDA_PATTERNS, "shabda"),
    (Pramana.PRATYAKSA, _PRATYAKSA_PATTERNS, "pratyaksa"),
    (Pramana.ANUPALABDHI, _ANUPALABDHI_PATTERNS, "anupalabdhi"),
    (Pramana.ARTHAPATTI, _ARTHAPATTI_PATTERNS, "arthapatti"),
    (Pramana.ANUMANA, _ANUMANA_PATTERNS, "anumana"),
    (Pramana.UPAMANA, _UPAMANA_PATTERNS, "upamana"),
]


# ─── Public API ─────────────────────────────────────────────────────

def detect_pramana(proposition: str) -> PramanaInference:
    """Classify a proposition's source of valid knowledge.

    Returns Pramana.UNKNOWN with cue="" when no pattern fires.
    """
    for pramana, patterns, label in _PATTERN_REGISTRY:
        for pat in patterns:
            m = pat.search(proposition)
            if m is not None:
                return PramanaInference(pramana=pramana, cue=m.group(0))
    return PramanaInference(pramana=Pramana.UNKNOWN, cue="")


def list_patterns() -> dict[str, int]:
    """Return a dict of pramana -> pattern count, for docs/tests."""
    return {label: len(pats) for _, pats, label in _PATTERN_REGISTRY}
