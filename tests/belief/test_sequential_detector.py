"""Tests for SequentialEventDetector (Phase 2 v0.7 addition).

The detector wraps any inner ContradictionDetector. It overrides with
CONTRADICTS when p2 contains a supersession marker AND p1/p2 are
topically similar. Everything else passes through to the inner.

We exercise the marker regexes and stub the similarity function so
tests don't need sentence-transformers at import time.
"""

from __future__ import annotations

from patha.belief.sequential_detector import (
    SequentialEventDetector,
    has_additive_marker,
    has_supersession_marker,
)
from patha.belief.types import ContradictionLabel, ContradictionResult


class _SpyDetector:
    """Records the pairs it sees; always returns NEUTRAL."""
    def __init__(self):
        self.received: list[tuple[str, str]] = []

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(self, pairs):
        self.received.extend(pairs)
        return [
            ContradictionResult(
                label=ContradictionLabel.NEUTRAL, confidence=0.5
            )
            for _ in pairs
        ]


def _always_similar(a: str, b: str) -> float:
    return 0.9


def _never_similar(a: str, b: str) -> float:
    return 0.0


class TestMarkerDetection:
    def test_marker_present_in_state_change(self):
        assert has_supersession_marker(
            "I upgraded to an M3 MacBook Pro this year"
        )
        assert has_supersession_marker(
            "I now drive an EV to work"
        )
        assert has_supersession_marker(
            "Charlie passed away and we adopted Maya"
        )
        assert has_supersession_marker(
            "I shut down the consultancy to co-found a startup"
        )
        assert has_supersession_marker(
            "I switched from Python to Rust"
        )
        assert has_supersession_marker(
            "My landlord is new this year"
        )
        assert has_supersession_marker(
            "No longer a vegetarian"
        )
        assert has_supersession_marker(
            "I now sleep with the air-con on instead"
        )

    def test_no_marker_in_neutral_statement(self):
        assert not has_supersession_marker(
            "My laptop is a 2019 MacBook Pro"
        )
        assert not has_supersession_marker(
            "I live in Sofia"
        )
        assert not has_supersession_marker(
            "I love sushi"
        )
        assert not has_supersession_marker(
            "My dog Charlie is a rescue mutt"
        )


class TestAdditiveMarkers:
    def test_additive_detected(self):
        assert has_additive_marker("I also play piano")
        assert has_additive_marker("I play guitar too")
        assert has_additive_marker("I run in the mornings as well")
        assert has_additive_marker("In addition, I swim")
        assert has_additive_marker("I still go to the gym")
        assert has_additive_marker("We got another cat")

    def test_too_much_not_additive(self):
        """'too much' is not the additive 'too'."""
        assert not has_additive_marker("I eat too much chocolate")

    def test_absent(self):
        assert not has_additive_marker("I upgraded my laptop")
        assert not has_additive_marker("I moved to Berlin")


class TestAdditiveVeto:
    def test_additive_marker_blocks_supersession(self):
        """If p2 has both a supersession marker AND an additive marker,
        additive wins — the change is expansion not replacement."""
        spy = _SpyDetector()
        det = SequentialEventDetector(
            inner=spy, similarity_fn=_always_similar,
        )
        det.detect(
            "I love jazz",
            "I now listen to classical music too",  # 'now' + 'too'
        )
        assert det.sequential_overrides == 0
        assert det.inner_calls == 1


class TestSequentialDetector:
    def test_fires_on_marker_plus_similarity(self):
        spy = _SpyDetector()
        det = SequentialEventDetector(
            inner=spy, similarity_fn=_always_similar,
        )
        result = det.detect(
            "My laptop is a 2019 MacBook Pro",
            "I upgraded to an M3 MacBook Pro this year",
        )
        assert result.label == ContradictionLabel.CONTRADICTS
        assert result.confidence == 0.85
        assert det.sequential_overrides == 1
        assert det.inner_calls == 0
        # Inner never called when sequential fires
        assert spy.received == []

    def test_passes_through_when_no_marker(self):
        spy = _SpyDetector()
        det = SequentialEventDetector(
            inner=spy, similarity_fn=_always_similar,
        )
        det.detect(
            "My laptop is a 2019 MacBook Pro",
            "My laptop is a 2019 MacBook Pro",
        )
        assert det.sequential_overrides == 0
        assert det.inner_calls == 1
        assert len(spy.received) == 1

    def test_passes_through_when_similarity_low(self):
        """Marker present but different topic — don't fire."""
        spy = _SpyDetector()
        det = SequentialEventDetector(
            inner=spy, similarity_fn=_never_similar,
        )
        det.detect(
            "I live in Sofia",
            "I upgraded to an M3 MacBook Pro",  # marker, but unrelated topic
        )
        assert det.sequential_overrides == 0
        assert det.inner_calls == 1

    def test_batch_mixed(self):
        spy = _SpyDetector()
        det = SequentialEventDetector(
            inner=spy, similarity_fn=_always_similar,
        )
        pairs = [
            ("My laptop is a 2019 MacBook Pro",
             "I upgraded to an M3 MacBook Pro"),                # marker → fire
            ("I love sushi",
             "The weather is nice today"),                       # no marker → inner
            ("I cycled to commute",
             "I now drive an EV to work"),                       # marker → fire
        ]
        results = det.detect_batch(pairs)
        assert results[0].label == ContradictionLabel.CONTRADICTS
        assert results[1].label == ContradictionLabel.NEUTRAL
        assert results[2].label == ContradictionLabel.CONTRADICTS
        assert det.sequential_overrides == 2
        assert det.inner_calls == 1

    def test_directionality(self):
        """Supersession is directional: new supersedes old, not vice versa.
        Only fires when the marker is in p2 (the new), not p1."""
        spy = _SpyDetector()
        det = SequentialEventDetector(
            inner=spy, similarity_fn=_always_similar,
        )
        # p1 has the marker (wrong direction)
        det.detect(
            "I upgraded to an M3 MacBook Pro",   # marker in p1
            "My laptop is a 2019 MacBook Pro",   # no marker in p2
        )
        assert det.sequential_overrides == 0
        assert det.inner_calls == 1

    def test_empty_batch(self):
        det = SequentialEventDetector(inner=_SpyDetector())
        assert det.detect_batch([]) == []
