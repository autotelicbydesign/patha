"""Tests for RevisionPatternDetector + SymmetricContradictionDetector —
the two v9 detector fixes from the EvolutionEval held-out reveal.

Pure: fake inner detectors + fake similarity fn, no model loads. The
example pairs come from the held-out failure decomposition (that batch
is spent and folded into dev, per protocol — using its pairs as unit
fixtures is dev work; generalization claims await held-out batch 2).
"""

from __future__ import annotations

from patha.belief.revision_patterns import RevisionPatternDetector
from patha.belief.symmetric_detector import SymmetricContradictionDetector
from patha.belief.types import ContradictionLabel, ContradictionResult


class _NeutralInner:
    """Inner detector that always says NEUTRAL (mimics NLI missing)."""

    def __init__(self):
        self.calls: list = []

    def detect(self, p1, p2):
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(self, pairs):
        self.calls.append(list(pairs))
        return [
            ContradictionResult(
                label=ContradictionLabel.NEUTRAL, confidence=0.99,
            )
            for _ in pairs
        ]


def _high_sim(a: str, b: str) -> float:
    return 0.9


def _low_sim(a: str, b: str) -> float:
    return 0.1


# ─── RevisionPatternDetector ────────────────────────────────────────


class TestResumptionFamily:
    def test_cessation_then_resumption_fires(self):
        det = RevisionPatternDetector(_NeutralInner(), similarity_fn=_high_sim)
        r = det.detect(
            "I stopped drinking entirely for the marathon block — zero alcohol since February",
            "I drink again now but only socially, never at home alone",
        )
        assert r.label == ContradictionLabel.CONTRADICTS
        assert "resumption" in (r.rationale or "")

    def test_back_on_phrasing(self):
        det = RevisionPatternDetector(_NeutralInner(), similarity_fn=_high_sim)
        r = det.detect(
            "I deleted twitter off my phone; the noise was eating my focus",
            "I'm back on twitter but only Tuesdays, to share what I shipped",
        )
        assert r.label == ContradictionLabel.CONTRADICTS

    def test_resumption_without_prior_cessation_does_not_fire(self):
        # New says "again" but the old belief never expressed cessation —
        # nothing to resume FROM; delegate to inner.
        det = RevisionPatternDetector(_NeutralInner(), similarity_fn=_high_sim)
        r = det.detect(
            "I enjoy hiking on weekends",
            "I went hiking again now that spring is here",
        )
        assert r.label == ContradictionLabel.NEUTRAL

    def test_negated_resumption_vetoed(self):
        det = RevisionPatternDetector(_NeutralInner(), similarity_fn=_high_sim)
        r = det.detect(
            "I quit smoking two years ago",
            "I am never going back to smoking again",
        )
        assert r.label == ContradictionLabel.NEUTRAL

    def test_low_similarity_delegates(self):
        # Markers fire but topics don't overlap → inner.
        det = RevisionPatternDetector(_NeutralInner(), similarity_fn=_low_sim)
        r = det.detect(
            "I stopped drinking entirely",
            "I'm back on the night train to Vienna again",
        )
        assert r.label == ContradictionLabel.NEUTRAL


class TestSettlementFamily:
    def test_landed_on_fires(self):
        det = RevisionPatternDetector(_NeutralInner(), similarity_fn=_high_sim)
        r = det.detect(
            "minimalism was making the flat feel like a waiting room",
            "I've landed on 'curated, not minimal' — few things, but warm ones",
        )
        assert r.label == ContradictionLabel.CONTRADICTS
        assert "settlement" in (r.rationale or "")

    def test_own_terms_fires(self):
        det = RevisionPatternDetector(_NeutralInner(), similarity_fn=_high_sim)
        r = det.detect(
            "I sold the car; city living means I'll never own one again",
            "the hatchback stays, but it lives at Dad's — car ownership on my own terms this time",
        )
        assert r.label == ContradictionLabel.CONTRADICTS


class TestArrangementFamily:
    def test_are_now_fires(self):
        det = RevisionPatternDetector(_NeutralInner(), similarity_fn=_high_sim)
        r = det.detect(
            "I play tennis with Dad every Saturday at the club courts",
            "Saturdays are now doubles with Dad coaching from the bench",
        )
        assert r.label == ContradictionLabel.CONTRADICTS
        assert "arrangement" in (r.rationale or "")

    def test_we_do_now_fires(self):
        det = RevisionPatternDetector(_NeutralInner(), similarity_fn=_high_sim)
        r = det.detect(
            "I see my therapist in person every second Monday",
            "we do video sessions weekly now instead of in-person fortnightly",
        )
        assert r.label == ContradictionLabel.CONTRADICTS


class TestVetoAndDelegation:
    def test_additive_marker_vetoes(self):
        # "also" = expansion, not replacement — must NOT fire even with
        # markers + similarity.
        det = RevisionPatternDetector(_NeutralInner(), similarity_fn=_high_sim)
        r = det.detect(
            "I stopped lifting weights",
            "I'm back on weights again and I also kept the running habit",
        )
        assert r.label == ContradictionLabel.NEUTRAL

    def test_no_markers_delegates_to_inner(self):
        inner = _NeutralInner()
        det = RevisionPatternDetector(inner, similarity_fn=_high_sim)
        r = det.detect("I like tea", "I like coffee")
        assert r.label == ContradictionLabel.NEUTRAL
        assert inner.calls, "inner must have been consulted"

    def test_batch_mixes_overrides_and_inner(self):
        det = RevisionPatternDetector(_NeutralInner(), similarity_fn=_high_sim)
        results = det.detect_batch([
            ("I quit coffee entirely", "I'm back on coffee, one cup before noon"),
            ("I like tea", "I like coffee"),
        ])
        assert results[0].label == ContradictionLabel.CONTRADICTS
        assert results[1].label == ContradictionLabel.NEUTRAL


# ─── SymmetricContradictionDetector ─────────────────────────────────


class _DirectionalInner:
    """CONTRADICTS only when p1 starts with 'NEW:' — i.e. only in the
    reverse direction of a (old, new) layer call. Mimics the NLI
    asymmetry found in the held-out reveal."""

    def detect(self, p1, p2):
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(self, pairs):
        out = []
        for p1, _p2 in pairs:
            if p1.startswith("NEW:"):
                out.append(ContradictionResult(
                    label=ContradictionLabel.CONTRADICTS, confidence=0.95,
                ))
            else:
                out.append(ContradictionResult(
                    label=ContradictionLabel.NEUTRAL, confidence=0.99,
                ))
        return out


class _WeakDirectionalInner(_DirectionalInner):
    """Reverse contradiction below the adoption bar."""

    def detect_batch(self, pairs):
        out = []
        for p1, _p2 in pairs:
            if p1.startswith("NEW:"):
                out.append(ContradictionResult(
                    label=ContradictionLabel.CONTRADICTS, confidence=0.80,
                ))
            else:
                out.append(ContradictionResult(
                    label=ContradictionLabel.NEUTRAL, confidence=0.99,
                ))
        return out


class TestSymmetricDetector:
    def test_adopts_high_confidence_reverse(self):
        det = SymmetricContradictionDetector(
            _DirectionalInner(), similarity_fn=_high_sim,
        )
        r = det.detect("OLD: I was rejected", "NEW: she was protecting me")
        assert r.label == ContradictionLabel.CONTRADICTS
        assert "reverse-direction" in (r.rationale or "")
        assert det.reverse_adoptions == 1

    def test_reverse_below_bar_not_adopted(self):
        det = SymmetricContradictionDetector(
            _WeakDirectionalInner(), reverse_min_confidence=0.90,
            similarity_fn=_high_sim,
        )
        r = det.detect("OLD: I was rejected", "NEW: she was protecting me")
        assert r.label == ContradictionLabel.NEUTRAL
        assert det.reverse_adoptions == 0

    def test_offtopic_reverse_not_adopted(self):
        # The dev-measured failure: NLI confidently contradicts an
        # unrelated pair in reverse ("squeaky hinge" vs a critique
        # reflection @ 0.992). The topic-overlap gate must reject it
        # regardless of confidence.
        det = SymmetricContradictionDetector(
            _DirectionalInner(), similarity_fn=_low_sim,
        )
        r = det.detect(
            "OLD: I finally fixed the squeaky hinge",
            "NEW: that critique is why the redesign is my strongest work",
        )
        assert r.label == ContradictionLabel.NEUTRAL
        assert det.reverse_adoptions == 0
        assert det.reverse_rejected_offtopic == 1

    def test_forward_contradiction_needs_no_reverse(self):
        class _ForwardInner(_DirectionalInner):
            def detect_batch(self, pairs):
                return [
                    ContradictionResult(
                        label=ContradictionLabel.CONTRADICTS, confidence=0.9,
                    )
                    for _ in pairs
                ]

        det = SymmetricContradictionDetector(
            _ForwardInner(), similarity_fn=_high_sim,
        )
        r = det.detect("a", "b")
        assert r.label == ContradictionLabel.CONTRADICTS
        assert det.reverse_adoptions == 0

    def test_batch_only_reverses_non_contradictions(self):
        det = SymmetricContradictionDetector(
            _DirectionalInner(), similarity_fn=_high_sim,
        )
        results = det.detect_batch([
            ("OLD: quit entirely", "NEW: back on it"),
            ("OLD: I like tea", "plain neutral pair"),
        ])
        assert results[0].label == ContradictionLabel.CONTRADICTS
        assert results[1].label == ContradictionLabel.NEUTRAL
