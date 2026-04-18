"""SequentialEventDetector — catches supersession that NLI misses.

NLI correctly reports NEUTRAL on propositions like:
    p1: "My laptop is a 2019 MacBook Pro"
    p2: "I upgraded to an M3 MacBook Pro this year"
    p1: "My dog Charlie is a rescue mutt"
    p2: "Charlie passed away and we adopted Maya"
    p1: "I cycled to commute"
    p2: "I now drive an EV to work"

These are *sequential events*, not logical contradictions. The first
proposition describes a state; the second describes a state change.
NLI's training data teaches it logical entailment, not temporal
supersession.

This module is a lightweight, principled approach:

  1. Detect a supersession marker in p2 ("now", "upgraded", "switched",
     "shut down", "passed away", "moved to", "instead", "no longer",
     "new [X]"). Markers are few and high-precision.
  2. Confirm topical overlap between p1 and p2 via sentence-embedding
     cosine similarity (>= configurable threshold). This is the
     "same subject" check — deterministic lexical overlap misses
     paraphrases; learned embeddings catch them.
  3. If both conditions fire, emit CONTRADICTS at confidence 0.85.

Deliberately NOT a regex pile:
  - Markers are semantic categories (state-change verbs, temporal
    shift adverbs) small enough to enumerate honestly.
  - Topic match is done by embeddings, not hand-written ontology
    entries. A paraphrase like "cycled to work" vs "drive an EV to
    work" will score high cosine similarity despite different tokens.
  - Confidence is capped at 0.85 (below adhyāsa/numerical) so higher-
    precision detectors take precedence when they fire.

Rationale vs. training a classifier:
  We considered training a small supersession classifier on the
  belief-eval scenarios, but (a) 300 self-authored scenarios is too
  little data, (b) the failure cases are semantically enumerable —
  "state change + same topic" — so a principled detector matches
  what a trained classifier would learn, without overfitting to our
  benchmark. An external benchmark (LongMemEval KU) will tell us if
  the principled detector generalises.
"""

from __future__ import annotations

import re
from typing import Callable

from patha.belief.contradiction import ContradictionDetector
from patha.belief.types import ContradictionLabel, ContradictionResult


# Supersession markers: compact, high-precision set of phrases in p2
# that indicate "my state has changed from a prior state."
_SUPERSESSION_MARKERS: tuple[re.Pattern, ...] = tuple(re.compile(pat, re.IGNORECASE) for pat in (
    # Temporal shift
    r"\bi\s+now\s+\w+",                    # "I now drive"
    r"\bnow\s+i\s+\w+",                    # "Now I drive"
    r"\bthese\s+days\b",                    # "These days I..."
    r"\bno\s+longer\b",                     # "No longer"
    r"\bnot\s+anymore\b",                   # "Not anymore"
    # State change verbs
    r"\bupgraded\s+(?:to|from)\b",          # "upgraded to M3"
    r"\bdowngraded\s+(?:to|from)\b",        # "downgraded to"
    r"\bswitched\s+(?:to|from)\b",          # "switched to"
    r"\bswapped\s+\w+\s+(?:to|for|with)\b", # "swapped X for Y"
    r"\bmoved\s+(?:to|from|into|out)\b",    # "moved to the suburbs"
    r"\brelocated\b",                       # "relocated"
    r"\breplaced\b",                         # "replaced with"
    r"\bshut\s+down\b",                     # "shut down the consultancy"
    r"\bwound\s+down\b",                    # "wound down"
    r"\bcancelled\b",
    r"\bquit\b",
    r"\bleft\s+(?:my|the|a)\s+\w+",        # "left my job"
    # Life events
    r"\bpassed\s+away\b",                   # "Charlie passed away"
    r"\bdied\b",
    r"\bpassed\b(?!\s+(?:the|an|my|out))",  # "passed" alone (not "passed the")
    r"\bretired\b",
    r"\bgraduated\b",
    r"\bgot\s+(?:a\s+)?(?:new|different)\b", # "got a new laptop"
    r"\bmy\s+new\s+\w+",                     # "my new landlord"
    r"\bis\s+new\s+(?:this|now|as\s+of)\b", # "is new this year"
    r"\badopted\b",                          # "adopted Maya"
    # Comparative / instead
    r"\binstead\s+of\b",                    # "instead of X"
    r"\binstead\b",
    # Acquired / divested
    r"\bsold\s+(?:my|the|our)\b",          # "sold my car"
    r"\bbought\s+(?:a\s+)?new\b",          # "bought a new house"
    r"\bstarted\s+\w+ing\b",                # "started cycling"
    r"\bstopped\s+\w+ing\b",                # "stopped eating"
))


def has_supersession_marker(text: str) -> bool:
    """Return True if text contains any supersession marker phrase."""
    return any(p.search(text) for p in _SUPERSESSION_MARKERS)


# Additive markers VETO supersession. If p2 contains "also", "too",
# "as well", "in addition", etc., the state is being *expanded*, not
# *replaced*. This is the single most common false-positive pattern:
#   "I now listen to classical music too" — additive, not replacing.
_ADDITIVE_MARKERS: tuple[re.Pattern, ...] = tuple(
    re.compile(pat, re.IGNORECASE) for pat in (
        r"\balso\b",
        r"\btoo\b(?!\s*(?:much|many|few|little))",  # "too" but not "too much"
        r"\bas\s+well\b",
        r"\bin\s+addition\b",
        r"\badditionally\b",
        r"\bon\s+top\s+of\b",
        r"\bplus\b",
        r"\banother\b",
        r"\bstill\b",  # "I still X" — state continues, plus maybe added
        r"\balong\s+with\b",
        r"\bas\s+well\s+as\b",
    )
)


def has_additive_marker(text: str) -> bool:
    """Return True if text carries an additive / expansion marker that
    VETOES supersession."""
    return any(p.search(text) for p in _ADDITIVE_MARKERS)


# Subject-overlap via embeddings with lazy caching.
# We intentionally depend on sentence-transformers (already a project
# dep) rather than a heavier NLI or LLM path, because we only need a
# cosine similarity on two short strings.

class _EmbeddingCache:
    """Caches embeddings for (text, model) pairs. Model loaded lazily."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._cache: dict[str, list[float]] = {}

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def similarity(self, a: str, b: str) -> float:
        self._ensure_model()
        import numpy as np
        to_encode = [t for t in (a, b) if t not in self._cache]
        if to_encode:
            vecs = self._model.encode(to_encode, normalize_embeddings=True)
            for t, v in zip(to_encode, vecs):
                self._cache[t] = list(v)
        va = np.array(self._cache[a])
        vb = np.array(self._cache[b])
        return float(np.dot(va, vb))


_DEFAULT_EMBED_CACHE = _EmbeddingCache()


class SequentialEventDetector:
    """Detects supersession via marker + topic-overlap.

    Usage:
        detector = SequentialEventDetector(inner=NLIContradictionDetector())
        # Now wraps NLI — fires CONTRADICTS on sequential-event cases
        # that NLI misses, delegates to inner for everything else.

    Parameters
    ----------
    inner
        The base contradiction detector (NLI, adhyāsa-wrapped, etc.).
    similarity_fn
        Optional callable (a, b) -> float in [-1, 1]. Defaults to a
        MiniLM sentence-transformer cosine similarity.
    similarity_threshold
        Minimum cosine similarity between p1 and p2 required to fire
        CONTRADICTS. Default 0.35. Tuned so paraphrastic same-topic
        pairs cross the threshold but unrelated pairs do not.
    confidence
        CONTRADICTS confidence when marker + similarity both fire.
        Default 0.85 — below numerical (0.9) and adhyāsa (up to 0.98),
        so specific detectors take precedence.
    """

    def __init__(
        self,
        inner: ContradictionDetector,
        *,
        similarity_fn: Callable[[str, str], float] | None = None,
        similarity_threshold: float = 0.35,
        confidence: float = 0.85,
    ) -> None:
        self._inner = inner
        self._sim = similarity_fn or _DEFAULT_EMBED_CACHE.similarity
        self._sim_threshold = similarity_threshold
        self._confidence = confidence
        # Metrics
        self.sequential_overrides = 0
        self.inner_calls = 0

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[ContradictionResult]:
        if not pairs:
            return []

        overrides: dict[int, ContradictionResult] = {}
        inner_pairs: list[tuple[int, tuple[str, str]]] = []

        for idx, (p1, p2) in enumerate(pairs):
            # Only fire if p2 carries a supersession marker, has NO
            # additive marker (expansion beats replacement), and is
            # topically similar to p1. Directional: new supersedes old.
            if (
                has_supersession_marker(p2)
                and not has_additive_marker(p2)
            ):
                sim = self._sim(p1, p2)
                if sim >= self._sim_threshold:
                    overrides[idx] = ContradictionResult(
                        label=ContradictionLabel.CONTRADICTS,
                        confidence=self._confidence,
                        rationale=(
                            f"sequential-event: marker + similarity={sim:.2f}"
                        ),
                    )
                    self.sequential_overrides += 1
                    continue
            inner_pairs.append((idx, (p1, p2)))

        if inner_pairs:
            batch = [pair for _, pair in inner_pairs]
            self.inner_calls += len(batch)
            inner_results = self._inner.detect_batch(batch)
            inner_by_idx = {
                inner_pairs[i][0]: inner_results[i]
                for i in range(len(inner_pairs))
            }
        else:
            inner_by_idx = {}

        return [
            overrides.get(i, inner_by_idx.get(i))
            for i in range(len(pairs))
        ]
