"""End-to-end belief-layer demo (zero-download by default).

Wired into the CLI as `patha demo`. Also callable as:
    python -m patha.demo          # stub detector, ~10 seconds
    python -m patha.demo --nli    # real NLI (~1.7 GB first run)

The demo walks through a short preference-shift story across four
dates, ingests each proposition, shows which actions fire
(added / reinforced / superseded), and renders the final belief
summary both with and without history — demonstrating the core
value prop of the belief layer in under 15 seconds.
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


def demo(use_nli: bool = False) -> None:
    detector = (
        NLIContradictionDetector() if use_nli else StubContradictionDetector()
    )
    layer = BeliefLayer(
        store=BeliefStore(),
        detector=detector,
        # Lower threshold for stub so the heuristic can fire in demo
        contradiction_threshold=0.75 if use_nli else 0.5,
    )

    timeline = [
        ("2023-03-15", "s1", "I love sushi and eat it every other week"),
        ("2023-09-10", "s2", "I tried a new ramen place this weekend"),
        ("2024-02-08", "s3", "I have been avoiding raw fish on my doctor's advice"),
        ("2024-04-01", "s4", "I am now fully vegetarian for ethical reasons"),
    ]

    print("=" * 60)
    print(f"Patha belief-layer demo — {'NLI' if use_nli else 'Stub'} detector")
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
            else " → reinforced existing belief"
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
    ) + len(timeline) * 5
    print(f"  Raw propositions (naive RAG):  ~{raw_tokens} tokens")
    print(f"  Current-belief summary:        ~{result.tokens_in_summary} tokens "
          f"({raw_tokens / max(result.tokens_in_summary, 1):.1f}x compression)")
    print(f"  With history:                  ~{result_history.tokens_in_summary} tokens "
          f"({raw_tokens / max(result_history.tokens_in_summary, 1):.1f}x compression)")
    print()
    print("Next steps:")
    print("  patha ingest 'I am allergic to peanuts'")
    print("  patha ask 'what do I avoid eating?'")
    print("  patha viewer      # visual inspection")
    print("  patha mcp         # run as an MCP server for Claude Desktop")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="patha demo",
        description="End-to-end belief-layer demo",
    )
    parser.add_argument(
        "--nli", action="store_true",
        help="Use real NLI model (downloads ~1.7 GB on first run)",
    )
    args = parser.parse_args()
    demo(use_nli=args.nli)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
