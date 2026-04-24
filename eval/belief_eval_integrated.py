"""BeliefEval end-to-end through the unified `patha.Memory` API.

BeliefEval historically runs against the BeliefLayer directly — it
passes belief_ids and bypasses any retrieval. That tests the belief
layer's supersession logic in isolation, but doesn't answer: "If a
developer uses `patha.Memory`, does Phase 1 retrieval surface the
right beliefs so Phase 2 supersession can act on them?"

This script runs the SAME 300 scenarios but through the full Memory
pipeline:

  for each scenario:
    memory = patha.Memory(detector=...)
    for each proposition (in temporal order):
        memory.remember(prop.text, asserted_at=prop.asserted_at, ...)
    for each question:
        rec = memory.recall(question.q, at_time=question.at_time,
                            include_history=True)
        score rec.summary against expected_current + expected_superseded

The difference vs belief_eval.py: Phase 1 is in the loop. A scenario
might have 5 propositions; retrieval picks the most relevant few; if
the retrieval misses the superseded-old-belief pair, supersession
never fires — and that's what we want to measure.

This is the real "merged system" test.

Usage:
    uv run python -m eval.belief_eval_integrated \\
        --scenarios eval/belief_eval_data/v05_combined_300.jsonl \\
        --detector full-stack-v8 \\
        --output runs/integrated_beliefeval/v08_300.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import patha

# Reuse the scorer + scenario loader from the existing belief_eval
from eval.belief_eval import (
    Question,
    Scenario,
    _score_current_belief,
    _score_validity_at_time,
    load_scenarios,
)


@dataclass
class IntegratedQResult:
    scenario_id: str
    family: str
    question: str
    type: str
    correct: bool
    answer_in_retrieved: bool
    answer_in_current: bool
    answer_in_history: bool
    details: dict = field(default_factory=dict)


def run_scenario_integrated(
    scenario: Scenario,
    *,
    detector: str,
    phase1_top_k: int = 50,
    tmp_root: Path | None = None,
) -> list[IntegratedQResult]:
    """Run one scenario through the full patha.Memory pipeline."""
    # Learning: don't use /tmp for long-lived artifacts — macOS /tmp
    # cleanup can wipe work mid-run. Per-scenario stores are short-lived
    # (deleted at end of the scenario) so /tmp is acceptable here, but
    # allow overriding for CI or sandboxed environments.
    tmp_root = tmp_root or Path("/tmp")
    tmp_path = tmp_root / f"patha-beliefeval-{scenario.id}.jsonl"
    tmp_path.unlink(missing_ok=True)

    memory = patha.Memory(
        path=tmp_path,
        detector=detector,
        enable_phase1=True,
        phase1_top_k=phase1_top_k,
    )

    # Ingest in temporal order
    props_sorted = sorted(scenario.propositions, key=lambda p: p.asserted_at)
    for i, p in enumerate(props_sorted):
        memory.remember(
            p.text,
            asserted_at=p.asserted_at,
            session_id=p.session,
            source_id=f"{scenario.id}-p{i}",
            context=p.context,
        )

    results: list[IntegratedQResult] = []
    for q in scenario.questions:
        at_time = q.at_time if q.at_time is not None else datetime(2030, 1, 1)
        rec = memory.recall(q.q, at_time=at_time, include_history=True)

        current_props = [c["proposition"] for c in rec.current]
        superseded_props = [h["proposition"] for h in rec.history]

        if q.type == "current_belief":
            correct, details = _score_current_belief(
                q, current_props, superseded_props,
            )
        elif q.type == "validity_at_time":
            valid = len(rec.current) > 0
            correct, details = _score_validity_at_time(q, valid)
        else:
            correct = False
            details = {"error": f"unknown question type: {q.type}"}

        cur_text = " | ".join(current_props).lower()
        his_text = " | ".join(superseded_props).lower()
        answer_terms = [t.lower() for t in q.expected_current_contains]
        answer_in_current = all(t in cur_text for t in answer_terms) if answer_terms else True
        answer_in_history = any(t in his_text for t in answer_terms) if answer_terms else False

        results.append(IntegratedQResult(
            scenario_id=scenario.id,
            family=scenario.family,
            question=q.q,
            type=q.type,
            correct=correct,
            answer_in_retrieved=answer_in_current or answer_in_history,
            answer_in_current=answer_in_current,
            answer_in_history=answer_in_history,
            details=details,
        ))

    tmp_path.unlink(missing_ok=True)
    return results


def _flush(msg: str) -> None:
    """Learning: Python buffers stdout over pipes. Force-flush every
    progress line so background runs show up in `tail -f`."""
    print(msg, flush=True)
    sys.stdout.flush()


def _save_checkpoint(path: Path, results: list[IntegratedQResult], meta: dict) -> None:
    """Learning: write atomic checkpoints every N scenarios so a kill at
    scenario 180/300 doesn't lose the first 179."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {"meta": meta, "results": [asdict(r) for r in results]}
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _load_checkpoint(path: Path) -> tuple[list[IntegratedQResult], set[str]]:
    """Resume: return (results, completed_scenario_ids)."""
    if not path.exists():
        return [], set()
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return [], set()
    results = [
        IntegratedQResult(**{k: v for k, v in r.items() if k in IntegratedQResult.__annotations__})
        for r in data.get("results", [])
    ]
    scenario_ids = {r.scenario_id for r in results}
    return results, scenario_ids


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="BeliefEval end-to-end through patha.Memory",
    )
    ap.add_argument(
        "--scenarios",
        default="eval/belief_eval_data/v05_combined_300.jsonl",
    )
    ap.add_argument(
        "--detector", default="stub",
        choices=["stub", "nli", "adhyasa-nli", "full-stack",
                 "full-stack-v7", "full-stack-v8"],
    )
    ap.add_argument("--phase1-top-k", type=int, default=50)
    ap.add_argument("--output", default="runs/integrated_beliefeval/results.json")
    ap.add_argument(
        "--checkpoint", default=None,
        help="Checkpoint path (default: <output>.checkpoint.json). "
             "Resume from here if run is interrupted.",
    )
    ap.add_argument(
        "--checkpoint-every", type=int, default=5,
        help="Save a checkpoint every N scenarios (default 5).",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--fresh", action="store_true",
        help="Ignore any existing checkpoint; start from scenario 0.",
    )
    args = ap.parse_args(argv)

    output_path = Path(args.output)
    ckpt_path = (
        Path(args.checkpoint) if args.checkpoint
        else output_path.with_suffix(".checkpoint.json")
    )

    scenarios = load_scenarios(args.scenarios)
    if args.limit:
        scenarios = scenarios[: args.limit]

    # Resume from checkpoint unless --fresh
    if args.fresh:
        ckpt_path.unlink(missing_ok=True)
        all_results: list[IntegratedQResult] = []
        done_ids: set[str] = set()
    else:
        all_results, done_ids = _load_checkpoint(ckpt_path)
        if all_results:
            _flush(f"Resuming from {ckpt_path}: {len(done_ids)} scenarios "
                   f"already completed, {len(all_results)} question results cached.")

    _flush(
        f"Running {len(scenarios)} scenarios through patha.Memory "
        f"(detector={args.detector}, phase1_top_k={args.phase1_top_k}, "
        f"checkpoint={ckpt_path})"
    )

    meta = {
        "scenarios_path": str(args.scenarios),
        "detector": args.detector,
        "phase1_top_k": args.phase1_top_k,
        "started_at": datetime.now().isoformat(),
    }

    t_total = time.perf_counter()
    skipped = 0
    for i, sc in enumerate(scenarios, 1):
        if sc.id in done_ids:
            skipped += 1
            continue
        t_sc = time.perf_counter()
        try:
            new_results = run_scenario_integrated(
                sc, detector=args.detector, phase1_top_k=args.phase1_top_k,
            )
            all_results.extend(new_results)
            done_ids.add(sc.id)
        except Exception as e:
            _flush(f"  [{i}/{len(scenarios)}] ERROR on {sc.id}: {type(e).__name__}: {e}")
            # Mark as done so we don't retry forever; record a failure
            # row per question for transparency.
            for q in sc.questions:
                all_results.append(IntegratedQResult(
                    scenario_id=sc.id, family=sc.family, question=q.q, type=q.type,
                    correct=False, answer_in_retrieved=False,
                    answer_in_current=False, answer_in_history=False,
                    details={"error": str(e)},
                ))
            done_ids.add(sc.id)

        # Progress every scenario (flushes)
        elapsed = time.perf_counter() - t_sc
        passed = sum(1 for r in all_results if r.correct)
        _flush(f"  [{i}/{len(scenarios)}] {sc.id} ({sc.family}) — "
               f"{elapsed:.1f}s — running {passed}/{len(all_results)} correct")

        # Checkpoint every N scenarios
        if (i - skipped) % args.checkpoint_every == 0:
            _save_checkpoint(ckpt_path, all_results, meta)

    # Final checkpoint
    _save_checkpoint(ckpt_path, all_results, meta)
    meta["completed_at"] = datetime.now().isoformat()
    meta["total_seconds"] = time.perf_counter() - t_total

    # Summary
    total = len(all_results)
    correct = sum(1 for r in all_results if r.correct)
    in_retrieved = sum(1 for r in all_results if r.answer_in_retrieved)

    by_family: dict[str, list[int]] = {}
    for r in all_results:
        by_family.setdefault(r.family, [0, 0])
        by_family[r.family][1] += 1
        if r.correct:
            by_family[r.family][0] += 1

    _flush("")
    _flush("=" * 60)
    _flush(f"BeliefEval-integrated ({args.detector}, top_k={args.phase1_top_k})")
    _flush("=" * 60)
    _flush(f"  Questions:                 {total}")
    _flush(f"  Correct:                   {correct}/{total} = {correct/max(total,1):.3f}")
    _flush(f"  Answer reached retrieved:  {in_retrieved}/{total} = {in_retrieved/max(total,1):.3f}")
    _flush(f"  Total time:                {meta.get('total_seconds', 0):.0f}s")
    _flush("")
    _flush("  By family:")
    for fam, (c, t) in sorted(by_family.items()):
        _flush(f"    {fam:25s}: {c}/{t} = {c/t:.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "meta": meta,
            "summary": {
                "n_questions": total,
                "n_correct": correct,
                "accuracy": correct / max(total, 1),
                "answer_reached_retrieval": in_retrieved / max(total, 1),
                "by_family": {
                    fam: {"correct": c, "total": t, "accuracy": c / t}
                    for fam, (c, t) in by_family.items()
                },
            },
            "results": [asdict(r) for r in all_results],
        }, f, indent=2, default=str)
    _flush(f"\nSaved to {output_path}")
    _flush(f"Checkpoint kept at {ckpt_path} — delete with --fresh next time.")


if __name__ == "__main__":
    main()
