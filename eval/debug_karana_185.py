"""Debug what karaṇa+gemma4 extracts on gpt4_d84a3211 — why $999?

Replays the ingest but dumps every tuple that ends up in the gaṇita
index. Then runs the question-time matching step by step so we can
see EXACTLY which tuples get pulled and why.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

import patha
from patha.belief.karana import OllamaKaranaExtractor
from patha.belief.ganita import (
    extract_entity_hints, detect_aggregation, _detect_attribute,
    _canonicalize_entity,
)


def _parse_lme_date(s):
    s = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", s).strip()
    return datetime.strptime(s, "%Y/%m/%d %H:%M")


def main():
    qs = json.load(open('data/longmemeval_s_cleaned.json'))
    q = next(x for x in qs if x['question_id'] == 'gpt4_d84a3211')

    karana = OllamaKaranaExtractor(model='gemma4:latest', timeout_s=60.0)
    td = Path(tempfile.mkdtemp(prefix='patha-debug-'))
    m = patha.Memory(
        path=td / "beliefs.jsonl",
        detector="stub",
        enable_phase1=False,  # don't need Phase 1 for this debug
        karana_extractor=karana,
    )

    # Ingest each session
    sess_dates = [_parse_lme_date(d) for d in q["haystack_dates"]]
    order = sorted(range(len(sess_dates)), key=lambda i: sess_dates[i])

    print(f"Ingesting {len(order)} sessions...", flush=True)
    t0 = time.perf_counter()
    for i, idx in enumerate(order, 1):
        sid = q["haystack_session_ids"][idx]
        date = sess_dates[idx]
        all_turns = [
            f"{t.get('role','?')}: {t.get('content','').strip()}"
            for t in q["haystack_sessions"][idx]
            if t.get("content", "").strip()
        ]
        if not all_turns:
            continue
        text = "\n\n".join(all_turns)
        m.remember(text, asserted_at=date, session_id=sid)
        if i % 5 == 0:
            print(f"  {i}/{len(order)} sessions ingested ({time.perf_counter() - t0:.0f}s)", flush=True)

    print(f"\nIngest complete. Index has {len(m._ganita_index)} tuples.", flush=True)
    print(f"Karaṇa stats: {karana.calls} calls, {karana.failures} failures, "
          f"{karana.total_latency_s:.0f}s total LLM time")
    print()

    # Question-side matching
    print("=" * 60)
    print("Question-side matching")
    print("=" * 60)
    question = q['question']
    print(f"Question: {question}")
    op = detect_aggregation(question)
    print(f"Detected operator: {op}")
    hints = extract_entity_hints(question)
    print(f"Hints: {hints}")
    hinted_attr = _detect_attribute(question)
    print(f"Hinted attribute: {hinted_attr}")
    print()

    # For each hint, show matching tuples
    for h in hints:
        matched = m._ganita_index.all_for(h)
        if matched:
            print(f"Tuples matching hint '{h}' ({len(matched)} total):")
            for t in matched[:30]:
                print(f"  ent={t.entity:20s} attr={t.attribute:12s} "
                      f"value={t.value:>8.2f} {t.unit:5s} "
                      f"aliases={list(t.entity_aliases)[:5]}")
            if len(matched) > 30:
                print(f"  ... ({len(matched) - 30} more)")
            print()

    # Show what survives the attribute filter
    print()
    print("Tuples after attribute filter (must be 'expense'):")
    candidates = []
    seen_ids = set()
    for h in hints:
        for c in m._ganita_index.all_for(h):
            if id(c) in seen_ids: continue
            seen_ids.add(id(c))
            if c.attribute == "expense":
                candidates.append(c)
    candidates.sort(key=lambda c: -c.value)
    for c in candidates[:30]:
        print(f"  ent={c.entity:20s} value={c.value:>8.2f} {c.unit:5s} "
              f"aliases={list(c.entity_aliases)[:5]}  bid={c.belief_id[:12]}")
    print(f"\nTotal: {len(candidates)} candidates, sum=${sum(c.value for c in candidates):.2f}")

    # Show all tuples with 'bike' anywhere in entity OR aliases
    print()
    print("All tuples mentioning 'bike' (entity or alias):")
    bike_tuples = []
    for ts in m._ganita_index._by_key.values():
        for t in ts:
            if t.entity == "bike" or "bike" in t.entity_aliases:
                bike_tuples.append(t)
    bike_tuples.sort(key=lambda c: -c.value)
    for t in bike_tuples:
        print(f"  ent={t.entity:20s} attr={t.attribute:12s} "
              f"value={t.value:>8.2f} {t.unit:5s} "
              f"aliases={list(t.entity_aliases)}")
    print(f"\nTotal bike-related tuples: {len(bike_tuples)}")
    print(f"Sum of bike-expense tuples: ${sum(t.value for t in bike_tuples if t.attribute=='expense'):.2f}")

    import shutil
    shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
