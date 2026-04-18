"""False-contradiction rate measurement.

The spec (§4.2) flags false-contradiction as the "dangerous failure
mode" — the belief layer should NOT supersede a belief that was never
contradicted. A false contradiction destroys current-state integrity.

This eval feeds hand-crafted pairs to the contradiction detector and
checks the label matches the expected verdict (CONTRADICT vs NOT).

Pairs cover:
  - reinforcement / parallel activities (should NOT contradict)
  - marker-present but different topic (should NOT contradict)
  - temporal past context (should NOT contradict)
  - legit supersession (SHOULD contradict — control group)

Metrics:
  - false_positive_rate = (NOT pairs labelled CONTRADICT) / (NOT pairs)
  - true_positive_rate  = (CONTRADICT pairs labelled CONTRADICT) / (CONTRADICT pairs)
  - precision / recall on CONTRADICT as the positive class
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patha.belief.types import ContradictionLabel

# Reuse the detector factory from belief_eval
from eval.belief_eval import _make_detector


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="False-contradiction rate eval"
    )
    ap.add_argument(
        "--pairs",
        default="eval/belief_eval_data/false_contradiction_pairs.jsonl",
    )
    ap.add_argument(
        "--detector", default="full-stack-v7",
        choices=[
            "stub", "nli", "hybrid",
            "adhyasa-nli", "adhyasa-hybrid",
            "full-stack", "full-stack-v7",
        ],
    )
    ap.add_argument("--output", default="runs/false_contradiction/results.json")
    args = ap.parse_args(argv)

    pairs: list[dict] = []
    with open(args.pairs) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    detector = _make_detector(args.detector)

    tp = fp = tn = fn = 0
    details: list[dict] = []
    for p in pairs:
        result = detector.detect(p["p1"], p["p2"])
        is_contradict = result.label == ContradictionLabel.CONTRADICTS
        expected_contradict = p["expected"] == "CONTRADICT"
        ok = is_contradict == expected_contradict
        if expected_contradict and is_contradict:
            tp += 1
        elif expected_contradict and not is_contradict:
            fn += 1
        elif not expected_contradict and is_contradict:
            fp += 1
        else:
            tn += 1
        details.append({
            "id": p["id"],
            "description": p["description"],
            "p1": p["p1"],
            "p2": p["p2"],
            "expected": p["expected"],
            "got_label": result.label.value,
            "got_confidence": result.confidence,
            "correct": ok,
        })

    total = tp + fp + tn + fn
    n_contradict = tp + fn
    n_not = tn + fp
    false_positive_rate = fp / max(n_not, 1)
    true_positive_rate = tp / max(n_contradict, 1)
    precision = tp / max(tp + fp, 1)
    accuracy = (tp + tn) / max(total, 1)

    print(f"False-contradiction eval ({args.detector})")
    print(f"  Pairs:         {total}")
    print(f"  Expected CONTRADICT: {n_contradict}")
    print(f"  Expected NOT:        {n_not}")
    print(f"  Accuracy:            {accuracy:.3f}")
    print(f"  Precision:           {precision:.3f}  (low = false positives)")
    print(f"  Recall (TPR):        {true_positive_rate:.3f}")
    print(f"  False-positive rate: {false_positive_rate:.3f}  (HIGH = DANGER)")
    print()
    print("Per-pair:")
    for d in details:
        tag = "PASS" if d["correct"] else "FAIL"
        print(f"  [{tag}] {d['id']} ({d['description']}): "
              f"got {d['got_label']}@{d['got_confidence']:.2f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "detector": args.detector,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": true_positive_rate,
                "false_positive_rate": false_positive_rate,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            },
            "details": details,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
