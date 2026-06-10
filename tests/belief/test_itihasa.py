"""Tests for the narrative-synthesis intent layer (belief/itihasa.py).

Covers the detector (the three narrative operators + non-narrative
rejection + precedence), theme extraction, the result dataclasses, and
the deterministic through-line / timeline rendering. The walker itself
(retrieval/narrative_walk.py) is tested separately; this is the intent
gate + typed result, parallel to test_ganita.py's detector coverage.
"""

from __future__ import annotations

from datetime import datetime

from patha.belief.itihasa import (
    NarrativeBeat,
    NarrativeResult,
    detect_narrative,
    extract_theme,
    extract_themes,
    render_through_line,
)


# ─── Intent detection ───────────────────────────────────────────────


class TestDetectNarrative:
    def test_evolution_questions(self):
        for q in [
            "how has my thinking on agency evolved?",
            "how have my views on remote work changed over the last year?",
            "my evolving stance on the productivity stack",
            "how has my opinion on testing shifted?",
            "how did I change over the past months on this?",
        ]:
            assert detect_narrative(q) == "evolution", q

    def test_origin_questions(self):
        for q in [
            "when did I first start thinking about agency?",
            "the first time I mentioned Sofia",
            "how did I get into rock climbing?",
            "what got me into woodworking?",
            "when did I begin journaling?",
        ]:
            assert detect_narrative(q) == "origin", q

    def test_throughline_questions(self):
        for q in [
            "what's the through-line in my notes on writing?",
            "trace my thinking on consciousness",
            "what's the arc of my career decisions?",
            "patterns in how I think about money",
            "the thread running through my reflections on grief",
        ]:
            assert detect_narrative(q) == "throughline", q

    def test_non_narrative_rejected(self):
        # Plain retrieval and synthesis questions must NOT route narrative.
        for q in [
            "what did I say about the saddle?",
            "how much have I spent on bikes total?",
            "where do I live?",
            "what do I currently eat?",
            "how many books did I read?",
            "am I still vegetarian?",
        ]:
            assert detect_narrative(q) is None, q

    def test_origin_precedence_over_evolution(self):
        # "when did I first start" carries both origin and (loosely)
        # change semantics; origin must win because it's more specific.
        assert detect_narrative(
            "when did I first start changing my mind on agency?"
        ) == "origin"

    def test_empty_and_garbage(self):
        assert detect_narrative("") is None
        assert detect_narrative("asdf qwerty") is None


# ─── Theme extraction ───────────────────────────────────────────────


class TestExtractTheme:
    def test_pulls_content_noun(self):
        assert extract_theme("how has my thinking on agency evolved?") == "agency"

    def test_strips_narrative_scaffolding(self):
        # "thinking", "evolved", "views" etc. must not become the theme.
        theme = extract_theme("trace my evolving thinking on consciousness")
        assert theme == "consciousness"

    def test_origin_phrasing(self):
        assert extract_theme("when did I first start woodworking?") == "woodworking"

    def test_no_content_returns_none(self):
        # Pure scaffolding, no theme noun.
        assert extract_theme("how has it evolved over the last year?") is None

    def test_extract_themes_multiple(self):
        themes = extract_themes("trace my thinking on the productivity stack")
        assert "productivity" in themes
        assert "stack" in themes
        # scaffolding words excluded
        assert "thinking" not in themes
        assert "trace" not in themes


# ─── Result dataclasses + rendering ─────────────────────────────────


def _beat(bid, prop, date, status="current", **kw):
    return NarrativeBeat(
        belief_id=bid,
        proposition=prop,
        asserted_at=datetime.fromisoformat(date) if date else None,
        supersession_status=status,
        **kw,
    )


class TestNarrativeResult:
    def test_beat_count_and_to_dict(self):
        beats = [
            _beat("b1", "agency is about constraints", "2025-07-01", "origin"),
            _beat("b2", "agency is about leverage", "2025-12-01", "current"),
        ]
        res = NarrativeResult(
            operator="evolution",
            theme="agency",
            beats=beats,
            through_line="...",
            contributing_belief_ids=["b1", "b2"],
            anchors=["b2"],
        )
        assert res.beat_count == 2
        d = res.to_dict()
        assert d["theme"] == "agency"
        assert d["operator"] == "evolution"
        assert len(d["beats"]) == 2
        # asserted_at serialized to iso string
        assert d["beats"][0]["asserted_at"].startswith("2025-07-01")

    def test_as_timeline_orders_and_marks(self):
        beats = [
            _beat("b1", "loved sushi weekly", "2025-06-01", "revised-from"),
            _beat("b2", "avoiding raw fish now", "2025-12-01", "current"),
        ]
        res = NarrativeResult(
            operator="evolution", theme="sushi", beats=beats,
            through_line="View on sushi shifted.",
            contributing_belief_ids=["b1", "b2"],
        )
        tl = res.as_timeline()
        assert "Theme: sushi" in tl
        assert "2025-06-01" in tl
        assert "2025-12-01" in tl
        # supersession markers present
        assert "revised" in tl.lower()
        assert "current" in tl.lower()
        # earliest beat appears before the latest in the rendered string
        assert tl.index("2025-06-01") < tl.index("2025-12-01")


class TestRenderThroughLine:
    def test_origin(self):
        beats = [
            _beat("b1", "started caring about agency after a bad sprint",
                  "2025-07-01", "origin"),
            _beat("b2", "agency reflection two", "2025-09-01"),
        ]
        line = render_through_line("origin", "agency", beats)
        assert "First engaged with agency" in line
        assert "2025-07-01" in line
        assert "1 related reflection since" in line

    def test_evolution_with_revision(self):
        beats = [
            _beat("b1", "old view", "2025-06-01", "revised-from"),
            _beat("b2", "new view", "2025-12-01", "current"),
        ]
        line = render_through_line("evolution", "sushi", beats)
        assert "shifted" in line
        assert "1 revision" in line

    def test_evolution_no_revision(self):
        beats = [
            _beat("b1", "a", "2025-06-01"),
            _beat("b2", "b", "2025-09-01"),
            _beat("b3", "c", "2025-12-01"),
        ]
        line = render_through_line("evolution", "writing", beats)
        assert "continuous line" in line
        assert "3 reflections" in line

    def test_throughline(self):
        beats = [
            _beat("b1", "a", "2025-01-01"),
            _beat("b2", "b", "2025-12-01"),
        ]
        line = render_through_line("throughline", "money", beats)
        assert "Thread on money" in line
        assert "2025-01-01" in line
        assert "2025-12-01" in line

    def test_empty_beats(self):
        assert "No recorded reflections" in render_through_line(
            "evolution", "ghosts", []
        )
