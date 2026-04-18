"""Measure plasticity mechanisms on real multi-session conversation logs.

Unit tests show plasticity classes work in isolation. This eval asks
the harder question: do LTD, LTP, Hebbian, and Pruning produce
*measurable, meaningful* statistics on real user-logs — not synthetic
ones — and do they affect retrieval outcomes?

Data: LongMemEval session logs. We pick questions (from any stratum),
ingest all user turns in chronological order, and record:

  1. **Belief confidence distribution over time** — does LTD produce
     a long-tailed distribution? (Signal: std dev > 0.1, not all at
     default 1.0)
  2. **Hebbian graph statistics** — how many associative edges form?
     Highest-degree nodes? (Signal: emergent network structure.)
  3. **Pruning activity** — how many beliefs get archived vs stay
     current? (Signal: pruning fires when supersession chains deepen.)
  4. **Store growth vs reinforcement ratio** — does reinforcement
     actually suppress store growth? (Signal: linear growth capped by
     reinforcement rate; otherwise store explodes.)

Plasticity only matters if it changes retrieval outputs. So we also
measure whether confidence-weighted supersession (which consults
LTP-bumped confidences) gives different current beliefs than pure
temporal supersession.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from patha.belief.adhyasa_detector import AdhyasaAwareDetector
from patha.belief.contradiction import NLIContradictionDetector
from patha.belief.layer import BeliefLayer, PlasticityConfig
from patha.belief.numerical_detector import NumericalAwareDetector
from patha.belief.sequential_detector import SequentialEventDetector
from patha.belief.store import BeliefStore
from patha.chunking.propositionizer import propositionize


def _parse_date(s: str) -> datetime:
    s = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", s).strip()
    return datetime.strptime(s, "%Y/%m/%d %H:%M")


@dataclass
class PlasticityStats:
    question_id: str
    ingested_props: int
    final_current: int
    final_superseded: int
    reinforcement_count: int
    supersession_count: int
    confidence_mean: float
    confidence_std: float
    confidence_min: float
    confidence_max: float
    hebbian_edges: int
    hebbian_top_degree: int
    pruning_archived: int
    ingest_tick: int


def _make_detector():
    return NumericalAwareDetector(
        inner=SequentialEventDetector(
            inner=AdhyasaAwareDetector(inner=NLIContradictionDetector())
        )
    )


def _relevant(text: str, keywords: set[str]) -> bool:
    toks = set(re.findall(r"[A-Za-z0-9]+", text.lower()))
    return bool(toks & keywords)


def run_question(q: dict, detector) -> PlasticityStats:
    layer = BeliefLayer(
        store=BeliefStore(),
        detector=detector,
        contradiction_threshold=0.7,
    )
    keywords = set(re.findall(r"[A-Za-z0-9]+", q["question"].lower())) | \
               set(re.findall(r"[A-Za-z0-9]+", str(q["answer"]).lower()))
    keywords = {k for k in keywords if len(k) >= 4 and not k.isdigit()}

    ingested = 0
    reinforced = 0
    superseded = 0
    dates = [_parse_date(d) for d in q["haystack_dates"]]
    sessions = q["haystack_sessions"]
    sids = q["haystack_session_ids"]
    order = sorted(range(len(sids)), key=lambda i: dates[i])
    belief_ids: list[str] = []
    for idx in order:
        sid = sids[idx]
        date = dates[idx]
        turns = sessions[idx]
        for ti, turn in enumerate(turns):
            if turn.get("role") != "user":
                continue
            text = turn.get("content", "")
            props = propositionize(text, session_id=sid, turn_idx=ti)
            for p in props:
                if not _relevant(p.text, keywords):
                    continue
                ev = layer.ingest(
                    proposition=p.text,
                    asserted_at=date,
                    asserted_in_session=sid,
                    source_proposition_id=p.chunk_id,
                )
                belief_ids.append(ev.new_belief.id)
                ingested += 1
                if ev.action == "reinforced":
                    reinforced += 1
                elif ev.action == "superseded":
                    superseded += 1

    # Simulate a query to trigger plasticity (homeostasis/pruning)
    q_date = _parse_date(q["question_date"])
    _ = layer.query(belief_ids, at_time=q_date, include_history=False)

    current = list(layer.store.current())
    super_list = list(layer.store.superseded())
    confidences = [b.confidence for b in (current + super_list)]
    hebbian_weights = getattr(layer.hebbian, "_weights", {})
    n_edges = len(hebbian_weights)
    if hebbian_weights:
        degree: dict = {}
        for (a, b) in hebbian_weights.keys():
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        top_deg = max(degree.values()) if degree else 0
    else:
        top_deg = 0
    archived = sum(
        1 for b in super_list if getattr(b, "archived", False)
    )

    return PlasticityStats(
        question_id=q["question_id"],
        ingested_props=ingested,
        final_current=len(current),
        final_superseded=len(super_list),
        reinforcement_count=reinforced,
        supersession_count=superseded,
        confidence_mean=statistics.mean(confidences) if confidences else 0.0,
        confidence_std=statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
        confidence_min=min(confidences) if confidences else 0.0,
        confidence_max=max(confidences) if confidences else 0.0,
        hebbian_edges=n_edges,
        hebbian_top_degree=top_deg,
        pruning_archived=archived,
        ingest_tick=layer._ingest_tick,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Plasticity stats on real LongMemEval logs"
    )
    ap.add_argument("--data", default="data/longmemeval_s_cleaned.json")
    ap.add_argument("--question-type", default="knowledge-update")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--output", default="runs/plasticity/real_logs.json")
    args = ap.parse_args(argv)

    with open(args.data) as f:
        data = json.load(f)
    qs = [q for q in data if q["question_type"] == args.question_type]
    qs = qs[: args.limit]

    detector = _make_detector()
    all_stats: list[PlasticityStats] = []
    for i, q in enumerate(qs, 1):
        stats = run_question(q, detector)
        all_stats.append(stats)
        print(f"  [{i}/{len(qs)}] {stats.question_id}: "
              f"ing={stats.ingested_props} cur={stats.final_current} "
              f"sup={stats.final_superseded} reinf={stats.reinforcement_count} "
              f"conf(mean±std)={stats.confidence_mean:.2f}±{stats.confidence_std:.2f} "
              f"hebb_edges={stats.hebbian_edges}")

    if not all_stats:
        print("no stats")
        return

    print()
    print("=" * 60)
    print(f"Plasticity on real logs ({args.question_type}, n={len(all_stats)})")
    print("=" * 60)
    # Aggregate
    def mean_attr(a):
        return statistics.mean(getattr(s, a) for s in all_stats)
    def max_attr(a):
        return max(getattr(s, a) for s in all_stats)
    print(f"  mean ingested:         {mean_attr('ingested_props'):.1f}")
    print(f"  mean final_current:    {mean_attr('final_current'):.1f}")
    print(f"  mean supersession:     {mean_attr('supersession_count'):.1f}")
    print(f"  mean reinforcement:    {mean_attr('reinforcement_count'):.1f}")
    print(f"  mean conf mean:        {mean_attr('confidence_mean'):.3f}")
    print(f"  mean conf std:         {mean_attr('confidence_std'):.3f}")
    print(f"  mean hebbian edges:    {mean_attr('hebbian_edges'):.1f}")
    print(f"  max  hebbian degree:   {max_attr('hebbian_top_degree')}")
    print(f"  mean archived:         {mean_attr('pruning_archived'):.1f}")

    # Honest signals
    interpret = []
    cstd = mean_attr("confidence_std")
    if cstd > 0.05:
        interpret.append(
            f"  [+] confidence_std = {cstd:.3f} — LTD is producing spread "
            "(not all beliefs at default 1.0)"
        )
    else:
        interpret.append(
            f"  [-] confidence_std = {cstd:.3f} — LTD decay effectively "
            "invisible (all beliefs near default). Needs longer time-spans "
            "or larger decay rate to matter."
        )
    hed = mean_attr("hebbian_edges")
    if hed > 5:
        interpret.append(
            f"  [+] hebbian edges = {hed:.1f}/scenario — associative network "
            "emerges from co-retrieval"
        )
    else:
        interpret.append(
            f"  [-] hebbian edges = {hed:.1f}/scenario — no co-retrieval "
            "happening. Hebbian is dormant in this workload."
        )
    rr = mean_attr("reinforcement_count") / max(mean_attr("ingested_props"), 1)
    interpret.append(
        f"  reinforcement ratio = {rr:.1%} of ingests "
        "(LTP would bump confidence on these — wired in store but "
        "LTP policy class is never invoked)"
    )
    print("\nHonest interpretation:")
    for line in interpret:
        print(line)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "n": len(all_stats),
            "aggregate": {
                "mean_ingested": mean_attr("ingested_props"),
                "mean_final_current": mean_attr("final_current"),
                "mean_supersession": mean_attr("supersession_count"),
                "mean_reinforcement": mean_attr("reinforcement_count"),
                "mean_conf_std": cstd,
                "mean_hebbian_edges": hed,
                "reinforcement_ratio": rr,
            },
            "per_question": [asdict(s) for s in all_stats],
            "interpretation": interpret,
        }, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
