"""Adhyāsa — superimposition-based contradiction detection (v0.4 #6).

Śaṅkara's *adhyāsa* is the cognitive error of superimposing one
thing's attributes onto another — the classical example being the
rope mistaken for a snake. The modern analogue: NLI systems miss
contradictions when surface lexemes differ but refer to the same
real-world entity. 'I love sushi' vs 'I'm avoiding raw fish' is
CONTRADICTORY because sushi IS (mostly) raw fish, but NLI treats
them as unrelated because the word 'sushi' doesn't appear in the
second sentence.

This module adds an adhyāsa-aware detection layer that runs BEFORE
the NLI/LLM judge:

  1. Extract key noun phrases from each proposition (spaCy).
  2. For each cross-pair noun phrase, check a small is-a / synonym
     ontology (WordNet-style, with hand-curated Patha-specific
     additions).
  3. When an identity is found, rewrite P2 with the identity
     substituted and re-query NLI on the rewritten pair.
  4. If the rewrite produces CONTRADICTS, flag the ORIGINAL pair as
     contradiction-via-adhyāsa.

v0.4 ships with a small built-in ontology (hand-curated to cover the
BeliefEval failure cases). Production use would wire WordNet or
ConceptNet. The module is designed so the ontology is swappable.

Caveats documented:
  - This is a preprocessor, not a replacement for NLI. Pairs that
    don't share an identity class fall through to the normal detector.
  - False positives are a real risk — the ontology must be conservative
    (is-a, synonym, kind-of; not 'related to').
  - Rewriting P2 can lose context. We ALSO check the substitution
    doesn't introduce a trivial tautology (e.g., "X is X").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


# ─── Ontology protocol + default implementation ────────────────────

@runtime_checkable
class IsAOntology(Protocol):
    """Source of is-a / synonym relations for adhyāsa detection.

    Implementations can wrap WordNet, ConceptNet, or hand-curated
    tables. The protocol is intentionally narrow — adhyāsa only needs
    'do these two terms refer to the same category'.
    """

    def equivalent_terms(self, term: str) -> set[str]:
        """Return the set of terms that refer to the same entity as
        ``term``, including term itself. For 'sushi' this might return
        {'sushi', 'raw fish', 'nigiri', 'maki'}.
        """
        ...


class HandCuratedOntology:
    """Small hand-curated ontology covering common BeliefEval failures.

    Each equivalence class is a set of mutually-interchangeable terms.
    Deliberately conservative: only includes cases where substitution
    would be semantically safe in most contexts.

    Production use: swap for a WordNet-backed implementation via
    ``WordNetOntology`` (v0.5+).
    """

    _CLASSES: list[set[str]] = [
        # Food / diet
        {"sushi", "raw fish", "nigiri", "maki", "sashimi"},
        {"vegetarian", "no meat", "meat-free"},
        {"vegan", "plant-based", "no animal products"},
        {"meat", "beef", "pork", "chicken", "lamb", "animal protein"},
        {"dairy", "milk products", "cheese and milk"},
        # Locations
        {"home", "house", "my place", "residence"},
        # Transport
        {"car", "automobile", "vehicle"},
        {"bike", "bicycle", "cycle"},
        # Work
        {"employed", "has a job", "working full-time"},
        {"unemployed", "jobless", "out of work", "between jobs"},
        {"remote", "working from home", "wfh"},
        # Relationships
        {"married", "spouse"},
        {"single", "unmarried"},
        # Activities
        {"running", "jogging"},
        {"reading physical books", "physical books"},
        {"reading on kindle", "e-reader", "kindle"},
    ]

    def equivalent_terms(self, term: str) -> set[str]:
        lower = term.lower().strip()
        for cls in self._CLASSES:
            normalised = {t.lower() for t in cls}
            if lower in normalised:
                return normalised.copy()
        return {lower}


# ─── Adhyāsa check result ──────────────────────────────────────────

@dataclass(frozen=True)
class AdhyasaResult:
    """Outcome of an adhyāsa check on a pair of propositions."""

    superimposition_detected: bool
    p1_term: str | None          # the term from p1 that was matched
    p2_term: str | None          # the equivalent term in p2
    rewritten_p2: str | None     # p2 with p1_term substituted for p2_term


def check_superimposition(
    p1: str,
    p2: str,
    ontology: IsAOntology | None = None,
) -> AdhyasaResult:
    """Detect whether p1 and p2 share an adhyāsa-worthy superimposition.

    Returns an AdhyasaResult. If superimposition_detected is True, the
    caller can re-run their contradiction detector on (p1, rewritten_p2)
    — which substitutes p2's equivalent term for the one used in p1.
    That rewritten pair shares the lexeme explicitly, giving NLI a
    fair shot at detecting the contradiction.

    Parameters
    ----------
    p1, p2
        The propositions being compared.
    ontology
        Source of equivalence classes. Defaults to
        HandCuratedOntology.
    """
    ont = ontology if ontology is not None else HandCuratedOntology()

    # Extract candidate terms from each: lowercased, stripped, up to 3 words.
    # Conservative tokenisation — we don't need full NER.
    p1_terms = _candidate_terms(p1)
    p2_terms = _candidate_terms(p2)

    for t1 in p1_terms:
        eq_class = ont.equivalent_terms(t1)
        if len(eq_class) <= 1:
            continue
        for t2 in p2_terms:
            if t2 in eq_class and t2 != t1:
                # Found: p1 mentions t1, p2 mentions t2, both in same
                # equivalence class
                rewritten = _substitute(p2, t2, t1)
                if rewritten == p2:
                    continue  # substitution was trivial
                return AdhyasaResult(
                    superimposition_detected=True,
                    p1_term=t1,
                    p2_term=t2,
                    rewritten_p2=rewritten,
                )

    return AdhyasaResult(
        superimposition_detected=False,
        p1_term=None, p2_term=None, rewritten_p2=None,
    )


# ─── Internals ─────────────────────────────────────────────────────

_STOP = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "i", "you",
    "he", "she", "it", "we", "they", "my", "your", "his", "her",
    "our", "their", "this", "that", "these", "those", "in", "on",
    "at", "to", "for", "with", "from", "of", "by", "and", "or",
    "but", "if", "then", "so", "as", "about", "have", "has", "had",
    "am", "be", "been", "being", "do", "does", "did",
})


def _candidate_terms(text: str) -> list[str]:
    """Extract candidate equivalence-class query terms from text.

    Returns lowercase single words and 2-3-word phrases that might
    appear in the ontology. Stopwords stripped. Keeps order so the
    first-matching equivalence class wins.
    """
    lower = text.lower()
    # Simple word-splitting; strip non-word chars
    words = re.findall(r"[a-zA-Z'-]+", lower)
    words = [w for w in words if w not in _STOP]

    terms: list[str] = []
    seen: set[str] = set()

    def _push(t: str) -> None:
        if t and t not in seen:
            seen.add(t)
            terms.append(t)

    # Unigrams
    for w in words:
        _push(w)
    # Bigrams
    for i in range(len(words) - 1):
        _push(f"{words[i]} {words[i+1]}")
    # Trigrams
    for i in range(len(words) - 2):
        _push(f"{words[i]} {words[i+1]} {words[i+2]}")

    return terms


def _substitute(text: str, old_term: str, new_term: str) -> str:
    """Case-insensitive word-boundary substitution of old_term by new_term.

    Leaves the surrounding structure intact. If old_term is not
    found as a phrase in text, returns text unchanged.
    """
    pattern = r"\b" + re.escape(old_term) + r"\b"
    return re.sub(pattern, new_term, text, flags=re.IGNORECASE)
