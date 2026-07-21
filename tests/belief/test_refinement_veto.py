"""Tests for RefinementVetoDetector (full-stack-v10's precision lever).

Model-free: a fake inner detector emits CONTRADICTS for every pair and
similarity is injected, so each veto class and each KEEP override is
pinned in isolation. The quality measurement lives in EvolutionEval
rubric v2; this suite pins the mechanics.
"""

from __future__ import annotations

from patha.belief.refinement_veto import RefinementVetoDetector
from patha.belief.types import ContradictionLabel, ContradictionResult


class _AlwaysContradicts:
    def detect(self, p1, p2):
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(self, pairs):
        return [
            ContradictionResult(
                label=ContradictionLabel.CONTRADICTS,
                confidence=0.9,
                rationale="fake-inner",
            )
            for _ in pairs
        ]


class _NeverContradicts:
    def detect_batch(self, pairs):
        return [
            ContradictionResult(
                label=ContradictionLabel.NEUTRAL,
                confidence=0.9,
                rationale="fake-inner",
            )
            for _ in pairs
        ]


def _det(sim=0.9):
    return RefinementVetoDetector(
        inner=_AlwaysContradicts(), similarity_fn=lambda a, b: sim,
    )


class TestVetoes:
    def test_atomic_supersession_never_locus_vetoed(self):
        # the BeliefEval-guard regression, pinned: short atomic
        # supersessions with divergent surfaces must survive — there
        # is NO blanket similarity veto (embedding sim cannot separate
        # "Sydney"→"Sofia" from genuine distractors at atomic length)
        d = _det(sim=0.05)
        r = d.detect("I live in Sydney", "I moved to Sofia last month")
        assert r.label == ContradictionLabel.CONTRADICTS

    def test_v2_fulfilled_intention(self):
        d = _det()
        r = d.detect(
            "I've been thinking about making things with my hands lately",
            "bought a proper calligraphy pen set and filled thirty pages",
        )
        assert r.label == ContradictionLabel.NEUTRAL
        assert "intention" in r.rationale

    def test_v3_initiation_progress(self):
        d = _det()
        r = d.detect("I started running twice a week before work",
                     "I signed up for a 10k in the spring")
        assert r.label == ContradictionLabel.NEUTRAL
        assert "initiation" in r.rationale

    def test_v4_new_regime_facets(self):
        d = _det()
        r = d.detect(
            "sleep is now nine am to four pm behind blackout blinds",
            "training moved to five pm, before the shift starts",
        )
        assert r.label == ContradictionLabel.NEUTRAL
        assert "regime" in r.rationale


class TestKeepOverrides:
    def test_cessation_in_new_beats_regime_shape(self):
        # the naps case: arrangement-shaped old, but the new belief
        # carries cessation — KEEP wins over V4
        d = _det()
        r = d.detect(
            "twenty-minute naps are now part of my day and they're glorious",
            "dropped the naps — my chronotype banks it all at night",
        )
        assert r.label == ContradictionLabel.CONTRADICTS
        assert d.kept_by_reversal == 1
        assert all(v == 0 for v in d.vetoes.values())

    def test_resumption_kept(self):
        d = _det()
        r = d.detect(
            "cash is dead weight — I went fully cashless",
            "cashless was wrong for me — back on cash envelopes for everything",
        )
        assert r.label == ContradictionLabel.CONTRADICTS

    def test_explicit_correction_kept(self):
        d = _det()
        r = d.detect(
            "the audit feels like persecution",
            "I've stopped calling it persecution — the audit was tuition",
        )
        assert r.label == ContradictionLabel.CONTRADICTS

    def test_plain_reversal_negation_kept(self):
        d = _det()
        r = d.detect("I love the open-plan office",
                     "the open-plan office is not working for me at all")
        assert r.label == ContradictionLabel.CONTRADICTS


class TestPassthrough:
    def test_non_contradicts_untouched(self):
        d = RefinementVetoDetector(
            inner=_NeverContradicts(), similarity_fn=lambda a, b: 0.0,
        )
        r = d.detect("I've been thinking about pottery", "bought a wheel")
        assert r.label == ContradictionLabel.NEUTRAL
        assert r.rationale == "fake-inner"       # untouched, not re-labeled
        assert all(v == 0 for v in d.vetoes.values())

    def test_plain_contradiction_with_no_veto_shape_kept(self):
        d = _det()
        r = d.detect("I work from the office five days a week",
                     "I work fully remote these days")
        assert r.label == ContradictionLabel.CONTRADICTS
