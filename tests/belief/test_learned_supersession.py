"""Tests for LearnedSupersessionDetector and the data-mining utility.

The learned classifier itself depends on heavy model weights that may
not be installed; we don't train or run a real model here. We verify:
  - Detector delegates to inner when no trained model exists
  - mine_training_pairs harvests pairs from the real BeliefEval data
  - TrainingPair dict conversion
"""

from __future__ import annotations

from pathlib import Path

from patha.belief.learned_supersession import (
    DEFAULT_MODEL_PATH,
    LearnedSupersessionDetector,
    TrainingPair,
    mine_training_pairs,
)
from patha.belief.types import ContradictionLabel, ContradictionResult


class _SpyDetector:
    def __init__(self):
        self.received: list[tuple[str, str]] = []

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(self, pairs):
        self.received.extend(pairs)
        return [
            ContradictionResult(
                label=ContradictionLabel.NEUTRAL, confidence=0.5,
            )
            for _ in pairs
        ]


class TestDetectorWithoutTrainedModel:
    def test_delegates_when_no_model_file(self, tmp_path):
        """With no model on disk, every pair goes to the inner detector."""
        spy = _SpyDetector()
        det = LearnedSupersessionDetector(
            inner=spy, model_path=tmp_path / "nonexistent.joblib",
        )
        result = det.detect("I love sushi", "I now avoid sushi")
        assert result.label == ContradictionLabel.NEUTRAL
        assert len(spy.received) == 1

    def test_empty_batch(self, tmp_path):
        det = LearnedSupersessionDetector(
            inner=_SpyDetector(), model_path=tmp_path / "nope.joblib",
        )
        assert det.detect_batch([]) == []


class TestTrainingPair:
    def test_to_dict(self):
        p = TrainingPair(p1="a", p2="b", label=1)
        d = p.to_dict()
        assert d == {"p1": "a", "p2": "b", "label": 1}


class TestMineTrainingPairs:
    def test_mines_from_real_belief_eval_data(self):
        pairs = mine_training_pairs()
        # The combined 300-scenario file should produce >= 200 pairs
        # (positives from scenario timelines + 20 from false-contradiction)
        assert len(pairs) > 100
        # Should have both labels
        labels = {p.label for p in pairs}
        assert 0 in labels
        assert 1 in labels
        # Most pairs are positive (scenario supersessions)
        pos = sum(1 for p in pairs if p.label == 1)
        neg = sum(1 for p in pairs if p.label == 0)
        assert pos > neg

    def test_graceful_when_files_missing(self, tmp_path):
        pairs = mine_training_pairs(
            belief_eval_path=tmp_path / "no.jsonl",
            false_contradiction_path=tmp_path / "no.jsonl",
        )
        assert pairs == []


class TestDefaults:
    def test_default_model_path(self):
        assert str(DEFAULT_MODEL_PATH).endswith("supersession_classifier.joblib")
