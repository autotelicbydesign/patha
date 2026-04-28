"""Smoke-test the gaṇita layer on the 18 synthesis-bounded multi-session
LongMemEval-S questions identified by eval/multisession_diagnosis.py.

For each: ingest the haystack into a fresh patha.Memory(), then call
recall(question). Check if Recall.ganita.value matches the gold answer
(numerical equality with tolerance).

Reports:
  - hits / total
  - per-question detail when ganita misses
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import patha
from patha.belief.ganita import extract_tuples


# Hand-curated list of synthesis-bounded multi-session questions
# from the diagnosis run, plus their gold answers.
SYNTHESIS_QUESTIONS = [
    # (question_id, gold_value_as_float, gold_unit_or_pattern)
    "gpt4_d84a3211",  # $185
    "2318644b",       # $270
    "gpt4_d12ceb0e",  # 59.6 (avg age)
    "d851d5ba",       # $3,750
    "gpt4_731e37d7",  # $720
    "e3038f8c",       # 99 items
    "2b8f3739",       # $495
    "gpt4_2f91af09",  # 23 pieces
]


def _parse_lme_date(s: str) -> datetime:
    s = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", s).strip()
    return datetime.strptime(s, "%Y/%m/%d %H:%M")


def _gold_value(answer: str) -> float | None:
    """Pull the numeric value from gold answer like '$185' or '23' or '59.6'."""
    answer = str(answer).replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", answer)
    if not m:
        return None
    return float(m.group(0))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/longmemeval_s_cleaned.json")
    ap.add_argument("--qids", nargs="*", default=SYNTHESIS_QUESTIONS)
    args = ap.parse_args(argv)

    qs = json.load(open(args.data))
    qs_by_id = {q["question_id"]: q for q in qs}

    print(f"Smoke-testing gaṇita on {len(args.qids)} synthesis-bounded questions...")
    print()

    hits = 0
    near_hits = 0
    fails = 0
    detail: list[dict] = []

    for qid in args.qids:
        q = qs_by_id.get(qid)
        if not q:
            print(f"  [{qid}]: NOT FOUND in dataset")
            continue

        gold_val = _gold_value(q["answer"])
        if gold_val is None:
            print(f"  [{qid}]: gold value unparseable ({q['answer']})")
            continue

        # Build a fresh in-memory Patha — NO Phase 1 (we don't need
        # retrieval for ganita; index reads all tuples regardless of
        # which beliefs were "retrieved").
        td = Path(tempfile.mkdtemp(prefix=f"patha-ganita-{qid}-"))
        m = patha.Memory(
            path=td / "beliefs.jsonl",
            detector="stub",
            enable_phase1=True,
            phase1_top_k=100,
            enable_ganita=True,
        )

        # Ingest entire haystack (both user + assistant turns)
        sess_dates = [_parse_lme_date(d) for d in q["haystack_dates"]]
        order = sorted(range(len(sess_dates)), key=lambda i: sess_dates[i])
        t0 = time.perf_counter()
        for idx in order:
            for turn in q["haystack_sessions"][idx]:
                content = turn.get("content", "").strip()
                if not content:
                    continue
                m.remember(content, asserted_at=sess_dates[idx],
                           session_id=q["haystack_session_ids"][idx])
        ingest_secs = time.perf_counter() - t0

        rec = m.recall(q["question"], at_time=_parse_lme_date(q["question_date"]))

        if rec.ganita is None:
            tag, predicted = "FAIL", None
            fails += 1
        else:
            predicted = rec.ganita.value
            # Tolerance: 5% of gold OR 1 absolute unit
            tol = max(abs(gold_val) * 0.05, 1.0)
            if abs(predicted - gold_val) <= tol:
                tag = "HIT"
                hits += 1
            elif abs(predicted - gold_val) <= max(abs(gold_val) * 0.20, 5.0):
                tag = "NEAR"
                near_hits += 1
            else:
                tag = "FAIL"
                fails += 1

        n_tuples = len(m._ganita_index)
        print(f"  [{qid}] {tag:>4}  gold={gold_val:>10}  "
              f"ganita={predicted}  "
              f"index={n_tuples}  ingest={ingest_secs:.0f}s")
        if rec.ganita and tag != "HIT":
            print(f"        explanation: {rec.ganita.explanation[:140]}")
        detail.append({
            "qid": qid, "tag": tag, "gold": gold_val, "predicted": predicted,
            "tuples": n_tuples,
        })

        import shutil
        shutil.rmtree(td, ignore_errors=True)

    n = len(detail)
    print()
    print(f"=== Summary ===")
    print(f"  Hits:      {hits}/{n}")
    print(f"  Near-hits: {near_hits}/{n} (within 20%)")
    print(f"  Fails:     {fails}/{n}")


if __name__ == "__main__":
    main()
