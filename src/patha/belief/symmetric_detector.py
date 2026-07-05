"""SymmetricContradictionDetector — bidirectional NLI check.

Belief-level contradiction is a symmetric relation: if "she was
protecting me from a reviewer bloodbath" contradicts "my supervisor
dismissed six months of work," it does so regardless of which belief
came first. NLI models are premise/hypothesis-ASYMMETRIC — the
EvolutionEval held-out reveal showed pairs scoring CONTRADICTS ≥ 0.94
in one direction and NEUTRAL ≥ 0.95 in the other. The belief layer's
convention (premise = older, hypothesis = newer) systematically misses
revisions where the negating content sits in the NEW statement —
reinterpretations and returns, exactly the classes the narrative path
cares about. Diagnosed pairs (held-out, layer direction → reverse):

    mf-therapy   NEUTRAL 0.617   →  CONTRADICTS 0.971
    ps-thesis    NEUTRAL 0.995   →  CONTRADICTS 0.938
    rb-alcohol   NEUTRAL 0.999   →  CONTRADICTS 0.770

This wrapper runs the inner detector in both directions. The forward
(old → new) result stands unless the REVERSE direction finds a
contradiction the forward missed at high confidence
(``reverse_min_confidence``, default 0.90 — conservative on purpose:
the reverse direction is auxiliary evidence, and a higher bar bounds
the false-positive surface the false-contradiction eval guards).

Wraps the NLI core ONLY (innermost in the stack). The outer detectors
(sequential, numerical, revision-pattern, learned) have direction-
DEPENDENT semantics — "I moved from X to Y" must read old→new — so
symmetrizing them would be wrong. Ships in ``full-stack-v9``; v7/v8
are frozen for reproducibility of published numbers.
"""

from __future__ import annotations

from typing import Callable

from patha.belief.contradiction import ContradictionDetector
from patha.belief.types import ContradictionLabel, ContradictionResult


class SymmetricContradictionDetector:
    """Run the inner detector both ways; adopt a high-confidence,
    TOPIC-OVERLAPPING reverse-direction contradiction the forward pass
    missed.

    The topic-overlap gate exists because NLI models can be confidently
    wrong on completely unrelated pairs — measured on the EvolutionEval
    dev set: DeBERTa scored "I finally fixed the squeaky hinge on the
    bathroom door" vs an on-theme critique reflection as CONTRADICTS
    0.992 in the reverse direction. A confidence bar alone cannot filter
    that; the model is *sure*. Requiring embedding similarity ≥ 0.35
    (the same gate the sequential and revision-pattern detectors use)
    encodes the actual constraint: a contradiction between two beliefs
    requires them to be about the same thing — in Nyāya terms, virodha
    presupposes a shared viṣaya (common locus). The genuine asymmetric
    pairs this wrapper exists for (reinterpretations, returns) are
    same-topic by nature and clear the gate comfortably.
    """

    def __init__(
        self,
        inner: ContradictionDetector,
        *,
        reverse_min_confidence: float = 0.90,
        similarity_fn: Callable[[str, str], float] | None = None,
        similarity_threshold: float = 0.35,
    ) -> None:
        self._inner = inner
        self._reverse_min = reverse_min_confidence
        if similarity_fn is None:
            from patha.belief.sequential_detector import _DEFAULT_EMBED_CACHE
            similarity_fn = _DEFAULT_EMBED_CACHE.similarity
        self._sim = similarity_fn
        self._sim_threshold = similarity_threshold
        self.reverse_adoptions = 0
        self.reverse_rejected_offtopic = 0

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[ContradictionResult]:
        if not pairs:
            return []
        fwd = self._inner.detect_batch(pairs)
        # Only pairs the forward pass did NOT flag need the reverse look.
        need_rev = [
            i for i, r in enumerate(fwd)
            if r.label != ContradictionLabel.CONTRADICTS
        ]
        if not need_rev:
            return fwd
        rev_results = self._inner.detect_batch(
            [(pairs[i][1], pairs[i][0]) for i in need_rev]
        )
        out = list(fwd)
        for i, rev in zip(need_rev, rev_results):
            if (
                rev.label == ContradictionLabel.CONTRADICTS
                and rev.confidence >= self._reverse_min
            ):
                # Topic-overlap gate: same-locus check before adoption.
                sim = self._sim(pairs[i][0], pairs[i][1])
                if sim < self._sim_threshold:
                    self.reverse_rejected_offtopic += 1
                    continue
                out[i] = ContradictionResult(
                    label=ContradictionLabel.CONTRADICTS,
                    confidence=rev.confidence,
                    rationale=(
                        f"symmetric-nli: reverse-direction contradiction "
                        f"(topic sim={sim:.2f})"
                    ),
                )
                self.reverse_adoptions += 1
        return out


__all__ = ["SymmetricContradictionDetector"]
