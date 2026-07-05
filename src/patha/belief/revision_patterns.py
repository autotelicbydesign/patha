"""RevisionPatternDetector — revision classes the v8 stack misses.

Built from the EvolutionEval held-out reveal (docs/benchmarks.md, "The
held-out reveal"): supersession generalization dropped 0.808 → 0.625,
and the misses decomposed into pattern families that neither NLI nor
the sequential-event markers cover:

  RESUMPTION   old: "I stopped drinking entirely…"
               new: "I drink again now but only socially…"
               The return-with-nuance never lexically contradicts the
               cessation, and "drink again now" matches no sequential
               marker ("I now X" / "now I X" require that word order).

  SETTLEMENT   old: "minimalism was making the flat feel like a
                     waiting room…"
               new: "I've landed on 'curated, not minimal'…"
               A settling-on-a-final-position statement supersedes the
               on-topic back-and-forth before it.

  ARRANGEMENT  old: "I play tennis with Dad every Saturday…"
               new: "Saturdays are now doubles with Dad coaching…"
               A schedule/structure rearrangement of the same activity.
               "are now" matches no sequential marker (the list covers
               "I now …"/"now I …" but not copular "is/are now").

Same architecture as SequentialEventDetector: additive wrapper that
only ESCALATES to CONTRADICTS (never suppresses), marker + embedding
topic-overlap, additive-marker veto, confidence just below sequential's
0.85 so more specific detectors take precedence.

Ships in `full-stack-v9`. v7/v8 are frozen for reproducibility of the
published EvolutionEval numbers — the held-out batch was spent on the
v8 report, and these fixes are validated on dev + a future held-out
batch 2, per the protocol note in docs/benchmarks.md.
"""

from __future__ import annotations

import re
from typing import Callable

from patha.belief.contradiction import ContradictionDetector
from patha.belief.sequential_detector import (
    _DEFAULT_EMBED_CACHE,
    has_additive_marker,
)
from patha.belief.types import ContradictionLabel, ContradictionResult

# ─── Family A: resumption ───────────────────────────────────────────
# Old belief must express cessation; new belief must express return.

_CESSATION_MARKERS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bquit\b",
        r"\bstopp?ed\b",
        r"\bgave\s+up\b",
        r"\bno\s+longer\b",
        r"\bnot\s+anymore\b",
        r"\bwent\s+(?:\w+[-\s])?free\b",       # "went caffeine-free"
        r"\bzero\s+\w+\b",                     # "zero alcohol"
        r"\bcut\s+(?:out|off)\b",
        r"\bdeleted\b",
        r"\bsold\s+(?:my|the|our)\b",
        r"\bcancell?ed\b",
        r"\bentirely\b",
        r"\bcompletely\b",
        r"\bfully\s+\w+\b",                    # "fully vegetarian"
        r"\b\w+\s+fast\b",                     # "news fast"
        r"\bnever\s+(?:again|owning?|drink)\b",
    )
)

_RESUMPTION_MARKERS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bback\s+on\b",                      # "back on coffee"
        r"\bback\s+to\b",                      # "back to the office"
        r"\b(?:i|i'm|im)\s+\w+\s+again\b",     # "I drink again"
        r"\bagain\s+now\b",                    # "… again now …"
        r"\bresumed\b",
        r"\breturned\s+to\b",
        r"\bre-?joined\b",
        r"\bstarted\s+\w+ing\s+again\b",       # "started keeping books again"
        r"\b\w+ing\s+again\b",                 # "eating fish again"
    )
)

# Negation immediately before a resumption phrase vetoes it:
# "I'm not back on twitter", "never going back to the office".
_NEGATED_RESUMPTION = re.compile(
    r"\b(?:not|never|no)\s+(?:\w+\s+){0,2}"
    r"(?:back\s+(?:on|to)|again|resum\w*|return\w*)\b",
    re.IGNORECASE,
)

# ─── Family B: settlement ───────────────────────────────────────────
# New belief announces a settled final position on the topic.

_SETTLEMENT_MARKERS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\blanded\s+on\b",                    # "I've landed on …"
        r"\bsettled\s+on\b",
        r"\bended\s+up\s+(?:with|at|on)\b",
        r"\bmy\s+rule\s+now\b",
        r"\bthe\s+rule\s+(?:now\s+)?is\b",
        r"\bit'?s\s+the\s+right\s+rule\b",
        r"\bon\s+my\s+own\s+terms\b",
    )
)

# ─── Family C: arrangement shift ────────────────────────────────────
# New belief restructures the schedule/shape of the same activity.

_ARRANGEMENT_MARKERS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\b(?:is|are)\s+now\b",               # "Saturdays are now doubles"
        r"\bwe\s+do\s+\w+(?:\s+\w+)?\s+now\b", # "we do video sessions … now"
        r"\b\w+\s+now\s+instead\b",            # "… weekly now instead"
        r"\bswitched\s+\w+\s+to\b",            # "switched the shift to"
        r"\bmoved\s+my\s+\w+\s+to\b",          # "moved my sessions to"
    )
)


def _any(markers: tuple[re.Pattern, ...], text: str) -> bool:
    return any(p.search(text) for p in markers)


class RevisionPatternDetector:
    """Additive detector for resumption / settlement / arrangement
    revisions. Wraps an inner detector; only escalates, never
    suppresses.

    Parameters mirror SequentialEventDetector: embedding topic-overlap
    with the same default similarity function and threshold, confidence
    just below sequential's (0.84 < 0.85) so when both could fire the
    more established detector wins the rationale.
    """

    def __init__(
        self,
        inner: ContradictionDetector,
        *,
        similarity_fn: Callable[[str, str], float] | None = None,
        similarity_threshold: float = 0.35,
        confidence: float = 0.84,
    ) -> None:
        self._inner = inner
        self._sim = similarity_fn or _DEFAULT_EMBED_CACHE.similarity
        self._sim_threshold = similarity_threshold
        self._confidence = confidence
        self.revision_overrides = 0
        self.inner_calls = 0

    # ── classification of a single (old, new) pair ──────────────────
    def _match_family(self, p1: str, p2: str) -> str | None:
        if has_additive_marker(p2):
            return None
        if _any(_RESUMPTION_MARKERS, p2) and not _NEGATED_RESUMPTION.search(p2):
            if _any(_CESSATION_MARKERS, p1):
                return "resumption"
        if _any(_SETTLEMENT_MARKERS, p2):
            return "settlement"
        if _any(_ARRANGEMENT_MARKERS, p2):
            return "arrangement"
        return None

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
            family = self._match_family(p1, p2)
            if family is not None:
                sim = self._sim(p1, p2)
                if sim >= self._sim_threshold:
                    overrides[idx] = ContradictionResult(
                        label=ContradictionLabel.CONTRADICTS,
                        confidence=self._confidence,
                        rationale=(
                            f"revision-pattern: {family} + "
                            f"similarity={sim:.2f}"
                        ),
                    )
                    self.revision_overrides += 1
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


__all__ = ["RevisionPatternDetector"]
