"""Belief layer demo — minimal end-to-end walkthrough.

Shows how to:
  1. Build a BeliefLayer with a detector and store
  2. Ingest a sequence of conversational propositions over time
  3. Query the current belief state
  4. See what was superseded and when
  5. Get a compact summary for downstream LLM consumption

Run this file directly:
    uv run python examples/belief_layer_demo.py                 # stub
    uv run python examples/belief_layer_demo.py --nli           # real NLI
"""

from __future__ import annotations

import argparse
from datetime import datetime

from patha.belief import (
    BeliefLayer,
    BeliefStore,
    NLIContradictionDetector,
    StubContradictionDetector,
)


def demo(use_nli: bool) -> None:
    detector = NLIContradictionDetector() if use_nli else StubContradictionDetector()
    layer = BeliefLayer(
        store=BeliefStore(),
        detector=detector,
        # Lower threshold for stub so the heuristic can fire in demo
        contradiction_threshold=0.75 if use_nli else 0.5,
    )

    # A preference-shift story told across three sessions
    timeline = [
        ("2023-03-15", "s1", "I love sushi and eat it every other week"),
        ("2023-09-10", "s2", "I tried a new ramen place this weekend"),
        ("2024-02-08", "s3", "I have been avoiding raw fish on my doctor's advice"),
        ("2024-04-01", "s4", "I am now fully vegetarian for ethical reasons"),
    ]

    print("=" * 60)
    print(f"Belief Layer Demo — {'NLI' if use_nli else 'Stub'} detector")
    print("=" * 60)
    print()
    print("Ingesting propositions in temporal order:")
    print()

    belief_ids = []
    for date_str, session, text in timeline:
        asserted_at = datetime.fromisoformat(date_str)
        ev = layer.ingest(
            proposition=text,
            asserted_at=asserted_at,
            asserted_in_session=session,
            source_proposition_id=f"{session}-p1",
        )
        belief_ids.append(ev.new_belief.id)
        affected = (
            f" → superseded {len(ev.affected_belief_ids)} older belief(s)"
            if ev.action == "superseded"
            else f" → reinforced existing belief"
            if ev.action == "reinforced"
            else ""
        )
        print(f"  [{date_str}] {ev.action.upper():11s} \"{text}\"{affected}")

    print()
    print("-" * 60)
    print("Query at 2024-05-01: current beliefs only")
    print("-" * 60)
    result = layer.query(
        belief_ids, at_time=datetime(2024, 5, 1), include_history=False
    )
    print(f"  {len(result.current)} current belief(s), "
          f"~{result.tokens_in_summary} tokens if sent to LLM")
    print()
    print("  Structured summary (what an LLM would see):")
    for line in layer.render_summary(result).split("\n"):
        print(f"    {line}")

    print()
    print("-" * 60)
    print("Query at 2024-05-01: include supersession history")
    print("-" * 60)
    result_history = layer.query(
        belief_ids, at_time=datetime(2024, 5, 1), include_history=True
    )
    print(
        f"  {len(result_history.current)} current + "
        f"{len(result_history.history)} historical, "
        f"~{result_history.tokens_in_summary} tokens if sent to LLM"
    )
    print()
    print("  Structured summary with history:")
    for line in layer.render_summary(
        result_history, include_history=True
    ).split("\n"):
        print(f"    {line}")

    print()
    print("-" * 60)
    print("Compression comparison")
    print("-" * 60)
    raw_tokens = sum(
        len(text.split()) for _, _, text in timeline
    ) + len(timeline) * 5  # plus per-chunk scaffolding
    print(f"  Raw propositions (naive RAG):  ~{raw_tokens} tokens")
    print(f"  Current-belief summary:        ~{result.tokens_in_summary} tokens "
          f"({raw_tokens / max(result.tokens_in_summary, 1):.1f}x compression)")
    print(f"  With history:                  ~{result_history.tokens_in_summary} tokens "
          f"({raw_tokens / max(result_history.tokens_in_summary, 1):.1f}x compression)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nli",
        action="store_true",
        help="Use real NLI model (downloads ~1.7 GB on first run)",
    )
    args = parser.parse_args()
    demo(use_nli=args.nli)


if __name__ == "__main__":
    main()
