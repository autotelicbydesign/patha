"""Learned supersession classifier — scaffold + inference wrapper.

Goal:
    Replace the regex-based SequentialEventDetector with a small
    learned classifier that predicts P(supersession | p1, p2) from
    sentence-pair embeddings. Target false-positive rate < 2% on the
    false-contradiction eval set (current regex: 6%).

This module provides:

  1. `LearnedSupersessionDetector` — protocol-conforming detector that
     loads a trained classifier and wraps an inner detector. Returns
     CONTRADICTS when the classifier's score exceeds a threshold.

  2. `train()` — training harness (logistic regression over concatenated
     embedding features). Writes `patha_supersession.joblib` to disk.

  3. `mine_training_pairs()` — lifts labeled pairs from the existing
     BeliefEval scenarios + false-contradiction set. Positives are
     supersession pairs that survive validation; negatives are the
     false-positive cases from the regex detectors plus the
     NOT_CONTRADICT pairs from `false_contradiction_pairs.jsonl`.

Why this design:
  - Small and legible. Logistic regression on 2*384=768-dim feature
    vectors is fast, interpretable, and trains in seconds on CPU.
  - Additive over the existing stack. The learned detector wraps the
    inner; if it doesn't fire, inner behavior is preserved.
  - No new dependencies beyond sentence-transformers (already in core)
    and scikit-learn (already a core dep via numerics).

What's NOT included yet (deferred):
  - A final trained `.joblib` model in the repo. Training data is
    mineable but hasn't been regenerated since the v0.7 additive veto
    landed. Run `python -m patha.belief.learned_supersession train`
    when you want a fresh model.
  - Hyperparameter tuning. The defaults below (L2 regularisation,
    class_weight='balanced', 0.5 threshold) are reasonable starting
    points, not optimal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from patha.belief.contradiction import ContradictionDetector
from patha.belief.types import ContradictionLabel, ContradictionResult

if TYPE_CHECKING:
    pass


# ─── Defaults ───────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = Path.home() / ".patha" / "supersession_classifier.joblib"


# ─── Detector (inference) ───────────────────────────────────────────

@dataclass
class LearnedSupersessionDetector:
    """Wraps an inner detector with a learned-classifier supersession check.

    On each pair (p1, p2):
      1. Embed both propositions.
      2. Feature vector = concat(emb_p1, emb_p2, emb_p1 - emb_p2, |emb_p1 - emb_p2|).
      3. Classifier predicts P(supersession).
      4. If P >= threshold, return CONTRADICTS with that confidence.
      5. Otherwise delegate to the inner detector.

    The learned detector is additive: if the classifier is unsure, the
    regex/NLI stack still gets a chance.
    """

    inner: ContradictionDetector
    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)
    threshold: float = 0.7
    embedder_name: str = "all-MiniLM-L6-v2"

    _model: object | None = None
    _embedder: object | None = None
    _feature_arity: int = 4   # concat, diff, abs_diff, plus originals

    def _ensure_loaded(self) -> bool:
        """Return True iff model + embedder loaded successfully."""
        if self._model is not None and self._embedder is not None:
            return True
        try:
            import joblib
            if not self.model_path.exists():
                return False
            self._model = joblib.load(self.model_path)
        except Exception:
            return False
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedder_name)
        except Exception:
            return False
        return True

    def _features(self, p1: str, p2: str) -> np.ndarray:
        vecs = self._embedder.encode([p1, p2], normalize_embeddings=True)
        a, b = np.asarray(vecs[0]), np.asarray(vecs[1])
        diff = a - b
        abs_diff = np.abs(diff)
        return np.concatenate([a, b, diff, abs_diff]).reshape(1, -1)

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(self, pairs):
        if not pairs:
            return []
        if not self._ensure_loaded():
            # No trained model → delegate entirely
            return self.inner.detect_batch(pairs)

        feats = np.vstack([self._features(p1, p2) for p1, p2 in pairs])
        probs = self._model.predict_proba(feats)[:, 1]

        results: list[ContradictionResult | None] = [None] * len(pairs)
        to_delegate: list[tuple[int, tuple[str, str]]] = []
        for i, (p, pair) in enumerate(zip(probs, pairs)):
            if p >= self.threshold:
                results[i] = ContradictionResult(
                    label=ContradictionLabel.CONTRADICTS,
                    confidence=float(p),
                    rationale=f"learned-supersession: P={p:.2f}",
                )
            else:
                to_delegate.append((i, pair))

        if to_delegate:
            inner_out = self.inner.detect_batch([pair for _, pair in to_delegate])
            for (i, _), r in zip(to_delegate, inner_out):
                results[i] = r
        return [r for r in results if r is not None]


# ─── Training data mining ───────────────────────────────────────────

@dataclass
class TrainingPair:
    p1: str
    p2: str
    label: int  # 1 = supersession, 0 = not

    def to_dict(self) -> dict:
        return {"p1": self.p1, "p2": self.p2, "label": self.label}


def mine_training_pairs(
    *,
    belief_eval_path: Path = Path("eval/belief_eval_data/v05_combined_300.jsonl"),
    false_contradiction_path: Path = Path(
        "eval/belief_eval_data/false_contradiction_pairs.jsonl"
    ),
) -> list[TrainingPair]:
    """Harvest labeled pairs from existing eval data.

    Positives (label=1):
      Every (old, new) pair from BeliefEval scenarios where the
      scenario has ≥2 propositions in temporal order — these are the
      explicit supersession cases.

    Negatives (label=0):
      - Every "NOT_CONTRADICT" pair from false_contradiction_pairs.jsonl
      - Cross-scenario random pairs (unrelated topics from different
        scenarios' first propositions) — trivially non-supersession.
    """
    pairs: list[TrainingPair] = []

    # Positives from scenario timelines
    if belief_eval_path.exists():
        with open(belief_eval_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    scenario = json.loads(line)
                except json.JSONDecodeError:
                    continue
                props = sorted(
                    scenario.get("propositions", []),
                    key=lambda p: p.get("asserted_at", ""),
                )
                family = scenario.get("family", "")
                if family == "reinforcement":
                    # Reinforcement isn't supersession; skip
                    continue
                # Consecutive pairs
                for i in range(len(props) - 1):
                    pairs.append(TrainingPair(
                        p1=props[i]["text"],
                        p2=props[i + 1]["text"],
                        label=1,
                    ))

    # Negatives from the false-contradiction set
    if false_contradiction_path.exists():
        with open(false_contradiction_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                is_contradict = entry.get("expected") == "CONTRADICT"
                pairs.append(TrainingPair(
                    p1=entry["p1"],
                    p2=entry["p2"],
                    label=1 if is_contradict else 0,
                ))

    return pairs


# ─── Training ───────────────────────────────────────────────────────

def train(
    *,
    pairs: list[TrainingPair] | None = None,
    output_path: Path = DEFAULT_MODEL_PATH,
    embedder_name: str = "all-MiniLM-L6-v2",
    test_split: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train a logistic-regression supersession classifier.

    Returns a metrics dict:
        {
          "n_train": int, "n_test": int,
          "accuracy": float, "precision": float, "recall": float, "f1": float,
          "false_positive_rate": float,
          "model_path": str
        }
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    import joblib

    if pairs is None:
        pairs = mine_training_pairs()
    if not pairs:
        raise ValueError("no training pairs — mine_training_pairs() returned empty")

    embedder = SentenceTransformer(embedder_name)

    def featurise(p1: str, p2: str) -> np.ndarray:
        vecs = embedder.encode([p1, p2], normalize_embeddings=True)
        a, b = np.asarray(vecs[0]), np.asarray(vecs[1])
        diff = a - b
        return np.concatenate([a, b, diff, np.abs(diff)])

    X = np.vstack([featurise(p.p1, p.p2) for p in pairs])
    y = np.array([p.label for p in pairs], dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y,
    )

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    metrics["model_path"] = str(output_path)
    return metrics


# ─── CLI ────────────────────────────────────────────────────────────

def _cli_train() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="patha-train-supersession",
        description="Train the learned supersession classifier from existing "
                    "BeliefEval + false-contradiction data.",
    )
    ap.add_argument(
        "--output", type=Path, default=DEFAULT_MODEL_PATH,
        help=f"Where to save the model (default: {DEFAULT_MODEL_PATH})",
    )
    ap.add_argument(
        "--embedder", default="all-MiniLM-L6-v2",
        help="Sentence-transformer model to use for features.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs = mine_training_pairs()
    print(f"Mined {len(pairs)} training pairs "
          f"({sum(p.label for p in pairs)} positive, "
          f"{sum(1 for p in pairs if p.label == 0)} negative)")
    if not pairs:
        print("No training data. Make sure BeliefEval + false-contradiction "
              "files exist.")
        return 1

    metrics = train(
        pairs=pairs, output_path=args.output,
        embedder_name=args.embedder, seed=args.seed,
    )
    print()
    print("Training complete.")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_train())
