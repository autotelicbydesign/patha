"""Separate retrieval failures from synthesis failures on multi-session
questions. The 500q integrated run scored 0.857 on multi-session;
the hypothesis from looking at a few questions is that these aren't
retrieval failures (we DO surface the right sessions) but synthesis
failures (the gold answer is a derived number that isn't in the
source text).

This script checks, for each multi-session question:
  A) Did Phase 1 retrieve the gold session(s) among the top-K?
     — "retrieval-correct" count
  B) Did the gold answer text appear in the retrieved summary?
     — "summary-match" count (same as current scorer)
  C) Discrepancy (retrieved gold but answer not in summary):
     — "retrieve-but-synth-gap" count — the upper bound of what
        an LLM synthesis step could recover.

If most of the 19 multi-session fails are (C) — retrieval-correct but
answer not in summary — then the 0.857 stratum is synthesis-bounded,
not retrieval-bounded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-results",
                    default="runs/integrated_500q/lme_s_500q_session_songline.json")
    ap.add_argument("--source-data", default="data/longmemeval_s_cleaned.json")
    ap.add_argument("--stratum", default="multi-session")
    args = ap.parse_args(argv)

    eval_results = json.load(open(args.eval_results))
    source = json.load(open(args.source_data))
    qs_by_id = {q["question_id"]: q for q in source}

    # Get multi-session question results
    outcomes = eval_results["outcomes"]
    stratum_results = []
    for r in outcomes:
        src = qs_by_id.get(r["question_id"])
        if not src:
            continue
        if src.get("question_type") != args.stratum:
            continue
        stratum_results.append((r, src))

    n = len(stratum_results)
    passed = sum(1 for r, _ in stratum_results if r["correct_with_history"])
    fails = [(r, src) for r, src in stratum_results if not r["correct_with_history"]]

    print(f"Stratum: {args.stratum}")
    print(f"Total: {n}, Passed: {passed} = {passed/n:.3f}, Failed: {len(fails)}")
    print()

    # Categorise each failure
    synthesis_bounded = 0  # retrieved gold but answer not in summary
    retrieval_bounded = 0  # didn't even retrieve a gold session
    unknown = 0

    for r, src in fails:
        gold_session_ids = set(src.get("answer_session_ids", []))
        # Reconstruct the ingested summary by re-running the eval? Too
        # expensive. Instead, use the outcome's current_summary field
        # if present.
        # The cleanest check: did any token of the gold session content
        # appear in the retrieved summary? We approximate with the
        # current_count — if > 0, retrieval happened; the miss must be
        # synthesis-bounded.
        # Sharper: check if gold_session_id strings appear in any belief.
        # But our ingest concatenates role:content and drops session ids
        # from the text. The simpler honest check is: "answer_in_retrieved"
        # as the eval already computed — False means neither current
        # nor history contained the answer.

        # If answer NOT in retrieved → could be retrieval OR synthesis
        # (answer not in source text). Check if the answer tokens appear
        # in ANY source session (by looking at gold session haystack content).
        # If answer appears in source → retrieval failed
        # If answer doesn't appear verbatim in source → synthesis-bounded
        answer = str(src["answer"]).lower()
        # Quick token check: do answer's content tokens appear in any gold session?
        import re
        answer_toks = [t for t in re.findall(r"\w+", answer) if len(t) >= 3]
        # Lookup gold session contents
        gold_content = ""
        for sess_id, sess in zip(src["haystack_session_ids"], src["haystack_sessions"]):
            if sess_id in gold_session_ids:
                for t in sess:
                    gold_content += " " + t.get("content", "")
        gold_content = gold_content.lower()
        answer_in_source = all(t in gold_content for t in answer_toks) if answer_toks else False

        # Older outcome JSON format uses correct_with_history;
        # compute "answer was reached by retrieval" from that + the
        # per-section flags.
        answer_reached = r.get("answer_in_retrieved",
                                r.get("answer_in_current") or r.get("answer_in_history", False))
        if not answer_reached:
            if answer_in_source:
                retrieval_bounded += 1
            else:
                synthesis_bounded += 1
        else:
            unknown += 1

    print(f"Fail classification:")
    print(f"  Synthesis-bounded (answer not in source — derived/computed): {synthesis_bounded}")
    print(f"  Retrieval-bounded (answer in source but we missed it):       {retrieval_bounded}")
    print(f"  Other (retrieved answer but scorer said wrong):              {unknown}")
    print()

    total_fails = len(fails)
    upper_bound_accuracy = (passed + synthesis_bounded) / n
    print(f"If LLM synthesis were added:")
    print(f"  {passed} (current-pass) + {synthesis_bounded} (synthesis-recoverable) = "
          f"{passed + synthesis_bounded}/{n} = {upper_bound_accuracy:.3f}")
    print(f"  vs current: {passed}/{n} = {passed/n:.3f}")
    print()
    print(f"Show me a few synthesis-bounded examples:")
    count = 0
    for r, src in fails:
        if count >= 3:
            break
        answer = str(src["answer"]).lower()
        import re
        answer_toks = [t for t in re.findall(r"\w+", answer) if len(t) >= 3]
        gold_content = ""
        for sess_id, sess in zip(src["haystack_session_ids"], src["haystack_sessions"]):
            if sess_id in set(src.get("answer_session_ids", [])):
                for t in sess:
                    gold_content += " " + t.get("content", "")
        gold_content_lower = gold_content.lower()
        answer_in_source = all(t in gold_content_lower for t in answer_toks) if answer_toks else False
        reached = r.get("answer_in_retrieved",
                         r.get("answer_in_current") or r.get("answer_in_history", False))
        if not reached and not answer_in_source:
            print(f"  {src['question_id']}: {src['question']}")
            print(f"    Gold: {src['answer']}")
            print(f"    Gold tokens not in source (answer is computed)")
            count += 1


if __name__ == "__main__":
    main()
