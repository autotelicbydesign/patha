"""Diagnostic: for each question where v0.7 fails but stub succeeds, was
the correct answer in v0.7's superseded store? If yes, the belief layer
retained the info but mis-routed it to history."""

from __future__ import annotations

import json
import re

from eval.belief_eval import _make_detector
from eval.longmemeval_belief import (
    _parse_lme_date,
    _relevant_to_question,
    _score_contains,
    _tokens,
    make_detector,
)
from patha.belief.layer import BeliefLayer
from patha.belief.store import BeliefStore
from patha.chunking.propositionizer import propositionize


def diagnose(q: dict, detector) -> dict:
    """Re-run ingestion but return current + superseded separately."""
    keywords = set(_tokens(q["question"])) | set(_tokens(q["answer"]))
    keywords = {k for k in keywords if len(k) >= 3}

    layer = BeliefLayer(
        store=BeliefStore(),
        detector=detector,
        contradiction_threshold=0.7,
    )

    session_ids = q["haystack_session_ids"]
    session_dates = [_parse_lme_date(d) for d in q["haystack_dates"]]
    sessions = q["haystack_sessions"]
    order = sorted(range(len(session_ids)), key=lambda i: session_dates[i])

    belief_ids: list[str] = []
    for idx in order:
        sid = session_ids[idx]
        date = session_dates[idx]
        sess = sessions[idx]
        for turn_idx, turn in enumerate(sess):
            if turn.get("role") != "user":
                continue
            text = turn.get("content", "")
            if not text.strip():
                continue
            props = propositionize(text, session_id=sid, turn_idx=turn_idx)
            for p in props:
                if not _relevant_to_question(p.text, keywords):
                    continue
                ev = layer.ingest(
                    proposition=p.text,
                    asserted_at=date,
                    asserted_in_session=sid,
                    source_proposition_id=p.chunk_id,
                )
                belief_ids.append(ev.new_belief.id)

    q_date = _parse_lme_date(q["question_date"])
    qr = layer.query(
        belief_ids, at_time=q_date, include_history=True,
    )
    current_summary = " | ".join(b.proposition for b in qr.current)
    super_summary = " | ".join(b.proposition for b in qr.history)
    return {
        "question_id": q["question_id"],
        "answer": str(q["answer"]),
        "n_current": len(qr.current),
        "n_super": len(qr.history),
        "in_current": _score_contains(str(q["answer"]), current_summary),
        "in_super": _score_contains(str(q["answer"]), super_summary),
        "in_either": _score_contains(str(q["answer"]), current_summary + " | " + super_summary),
    }


def main():
    # Load questions that stub got right but v0.7 got wrong
    with open('runs/longmemeval_ku/full_78_stub.json') as f:
        stub = json.load(f)
    with open('runs/longmemeval_ku/full_78_v7.json') as f:
        v7 = json.load(f)
    stub_by_id = {o['question_id']: o for o in stub['outcomes']}
    v7_by_id = {o['question_id']: o for o in v7['outcomes']}
    stub_only_ids = [
        qid for qid in stub_by_id
        if stub_by_id[qid]['correct'] and not v7_by_id[qid]['correct']
    ]

    with open('data/longmemeval_s_cleaned.json') as f:
        data = json.load(f)
    qs_by_id = {q['question_id']: q for q in data}

    detector = make_detector('full-stack-v7')

    print(f"Diagnosing {len(stub_only_ids)} stub-only-wins...")
    results = []
    recoverable = 0
    for i, qid in enumerate(stub_only_ids, 1):
        q = qs_by_id[qid]
        d = diagnose(q, detector)
        results.append(d)
        if d['in_either']:
            recoverable += 1
        tag = "recoverable" if d['in_either'] else "truly lost"
        loc = "current" if d['in_current'] else ("superseded" if d['in_super'] else "neither")
        print(f"  [{i}/{len(stub_only_ids)}] {qid}: {tag} ({loc}) "
              f"cur={d['n_current']} super={d['n_super']}")

    print()
    print(f"Stub-only wins: {len(stub_only_ids)}")
    print(f"Answer in current or superseded (recoverable): {recoverable}")
    print(f"Truly lost (not in either): {len(stub_only_ids) - recoverable}")

    with open('runs/longmemeval_ku/diagnose_v7.json', 'w') as f:
        json.dump({
            "stub_only_count": len(stub_only_ids),
            "recoverable": recoverable,
            "truly_lost": len(stub_only_ids) - recoverable,
            "per_question": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
