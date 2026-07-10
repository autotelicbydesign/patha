"""Tests for anupalabdhi (absence queries — the fifth question class).

Covers: question detection (all scopes + the aggregation guards that
keep the route from stealing gaṇita's questions), locus extraction
conventions, the fixpoint stemmer, verdict logic per temporal scope
(actuality for 'ever' incl. presupposition projection, commitment for
'yet', latest-assertion for 'still', first-person assignment for
identity), and recall() routing integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from patha.belief.abhava import AbhavaKind
from patha.belief.anupalabdhi import (
    AbsenceQuestion,
    _extract_locus,
    _stem,
    answer_absence,
    detect_absence_question,
)


# ─── Detection ──────────────────────────────────────────────────────


class TestDetection:
    def test_scopes(self):
        cases = {
            "have I ever been to Japan?": ("ever", AbhavaKind.ATYANTABHAVA),
            "have we decided on a wedding venue yet?": ("yet", AbhavaKind.PRAGABHAVA),
            "do I still go to the Italian class?": ("still", AbhavaKind.PRADHVAMSABHAVA),
            "am I a homeowner?": ("identity", AbhavaKind.ANYONYABHAVA),
            "what have I not decided about the move?": ("open", AbhavaKind.PRAGABHAVA),
        }
        for q, (scope, kind) in cases.items():
            got = detect_absence_question(q)
            assert got is not None, q
            assert (got.scope, got.kind) == (scope, kind), q

    def test_identity_subject_form(self):
        got = detect_absence_question("is Maya my accountant?")
        assert got is not None and got.scope == "identity"
        assert got.subject == "maya" and got.role == "accountant"

    def test_aggregation_guards_decline(self):
        # absence-looking phrasings that belong to synthesis
        for q in (
            "have I ever spent more than 100 in one go on food?",
            "how much of the car loan haven't I paid off yet?",
            "how many books have I not finished?",
        ):
            assert detect_absence_question(q) is None, q

    def test_non_absence_phrasings_decline(self):
        for q in (
            "I can never remember — what's my physiotherapist's name?",
            "which foods did I say I don't like?",
            "what have I said about the deposit?",   # no negation → retrieval
            "how has my thinking about the move evolved?",
        ):
            assert detect_absence_question(q) is None, q


class TestLocusExtraction:
    def test_gold_conventions(self):
        cases = {
            "been to Japan": "japan",
            "tried acupuncture for my back pain": "acupuncture",
            "decided on a wedding venue": "wedding venue",
            "have the basketball season ticket": "season ticket",
            "chosen a builder for the kitchen": "builder",
            "picked a name for the puppy": "puppy name",
            "the team lead of the platform group": "team lead",
            "a certified first aider": "first aider",
            "take violin lessons": "violin lessons",  # scorer de-plurals
        }
        for phrase, expected in cases.items():
            assert _extract_locus(phrase) == expected, phrase

    def test_stem_fixpoint_convergence(self):
        # both sides of any comparison must reach the same root
        assert _stem("builders") == _stem("builder")
        assert _stem("wondering") == _stem("wonder")
        assert _stem("lessons") == _stem("lesson")


# ─── Verdict logic ──────────────────────────────────────────────────


@dataclass
class _B:
    id: str
    proposition: str
    asserted_at: datetime


@dataclass
class _Store:
    beliefs: list = field(default_factory=list)

    def current(self):
        return list(self.beliefs)

    def get(self, bid):
        return next((b for b in self.beliefs if b.id == bid), None)


def _q(scope, locus, **kw):
    kinds = {"ever": AbhavaKind.ATYANTABHAVA, "yet": AbhavaKind.PRAGABHAVA,
             "still": AbhavaKind.PRADHVAMSABHAVA,
             "identity": AbhavaKind.ANYONYABHAVA}
    return AbsenceQuestion(scope=scope, kind=kinds[scope], locus=locus,
                           cue=kw.pop("cue", f"q about {locus}"), **kw)


def _mk(*props):
    return _Store([
        _B(f"b{i}", text, datetime(2025, 1 + i, 1))
        for i, text in enumerate(props)
    ])


class TestVerdicts:
    def test_ever_present_on_actuality(self):
        store = _mk("finished my first sprint triathlon in Cascais")
        r = answer_absence(_q("ever", "triathlon"), store=store)
        assert r.verdict == "present" and r.searched_n == 1

    def test_ever_absent_when_only_plans(self):
        store = _mk("booked a motorcycle CBT taster session for September")
        r = answer_absence(_q("ever", "motorcycle"), store=store)
        assert r.verdict == "absent"
        # the plan is still cited as the locus lineage
        assert r.contrast_ids == ["b0"]

    def test_ever_presupposition_projects_through_irrealis(self):
        store = _mk("been wondering whether I'd ever move abroad again")
        r = answer_absence(_q("ever", "abroad"), store=store)
        assert r.verdict == "present"  # 'again' presupposes a prior stint

    def test_yet_requires_commitment_not_mention(self):
        store = _mk(
            "started the wedding venue search — longlist of nine places",
            "shortlisted three wedding venues",
        )
        r = answer_absence(_q("yet", "wedding venue"), store=store)
        assert r.verdict == "absent"
        store.beliefs.append(
            _B("b9", "we signed with the glasshouse wedding venue",
               datetime(2025, 6, 1)))
        r = answer_absence(_q("yet", "wedding venue"), store=store)
        assert r.verdict == "present"

    def test_still_latest_assertion_decides(self):
        store = _mk(
            "rented a storage unit on Mill Road",
            "cleared out and cancelled the storage unit",
        )
        r = answer_absence(_q("still", "storage unit"), store=store)
        assert r.verdict == "absent"
        # both ends of the lineage cited — the pair constitutes the absence
        assert set(r.contrast_ids) == {"b0", "b1"}

    def test_identity_third_person_assignment_rejected(self):
        store = _mk("Dario was announced as team lead for the platform group")
        r = answer_absence(_q("identity", "team lead", role="team lead"),
                           store=store)
        assert r.verdict == "absent"

    def test_identity_first_person_present(self):
        store = _mk("renewed my film society membership for the year")
        r = answer_absence(
            _q("identity", "film society", role="film society"), store=store)
        assert r.verdict == "present"

    def test_semantic_contrast_fallback_when_lineage_empty(self):
        store = _mk(
            "spent two weeks in South Korea last spring",
            "the landlord is repainting the stairwell",
        )
        sims = lambda q, texts: [0.9 if "Korea" in t else 0.1 for t in texts]
        r = answer_absence(_q("ever", "japan"), store=store,
                           similarity_fn=sims)
        assert r.verdict == "absent" and r.contrast_ids == ["b0"]


# ─── recall() routing integration ───────────────────────────────────


class TestRouting:
    def test_absence_routes_and_controls_do_not(self, tmp_path):
        import patha
        mem = patha.Memory(path=tmp_path / "b.jsonl", detector="stub",
                           enable_phase1=False)
        mem.remember("finished my first sprint triathlon in Cascais",
                     asserted_at=datetime(2025, 3, 1))
        r = mem.recall("have I ever done a triathlon?")
        assert r.strategy == "absence"
        assert r.absence is not None and r.absence.verdict == "present"
        assert r.answer == "yes"
        # zero LLM tokens on the absence path
        assert r.tokens == 0
        # control: plain retrieval phrasing stays off the absence route
        r2 = mem.recall("what did I say about the triathlon?")
        assert r2.strategy != "absence" and r2.absence is None
