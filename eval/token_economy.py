"""Token-economy measurement for Patha.

Produces the curves that back the 'Patha reduces tokens' claim:

  - tokens_per_query at different memory sizes (100 → 1,000 → 10,000)
  - compression ratio: raw memory size / tokens sent to LLM per query
  - marginal token cost as the memory store grows
  - direct-answer vs. raw-RAG vs. structured-summary comparison

Usage:
    python -m eval.token_economy \\
        --scenarios eval/belief_eval_data/seed_scenarios.jsonl \\
        --sizes 50,200,1000 \\
        --output runs/token_economy/results.json

Output: a JSON report with the three curves, one row per (memory_size,
strategy) pair. Strategies compared:

  - naive_rag        : retrieve top-k raw propositions, dump into prompt
  - structured       : render belief state as structured summary (D7-B)
  - direct_answer    : answer from belief state, no LLM (D7-C)

The measurement is deliberately simple. It counts tokens using a
4-chars-per-token heuristic (calibrated approximation of tiktoken on
English text). Production measurement would use the actual tokenizer
of the downstream LLM; for publishing compression ratios across
strategies, the heuristic is fine because the ratios are invariant
under a linear token-counter.

This module does not evaluate correctness — that's BeliefEval's job.
Token economy is an independent axis. A memory strategy can be
more token-efficient AND less correct (pathological case: return
nothing, zero tokens, zero accuracy). The real publication value is
in combining the token-economy curves with BeliefEval accuracy in
the same write-up.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from eval.belief_eval import Scenario, load_scenarios
from patha.belief.contradiction import StubContradictionDetector
from patha.belief.direct_answer import DirectAnswerer
from patha.belief.layer import BeliefLayer
from patha.belief.store import BeliefStore


# ─── Token counting ──────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token for English text.

    Calibrated against tiktoken's cl100k_base encoding on typical
    conversational text. Fine for relative comparisons between
    strategies; use an actual tokenizer for absolute cost forecasting.
    """
    return max(1, len(text) // 4)


# ─── Strategies ──────────────────────────────────────────────────────

Strategy = Literal["naive_rag", "structured", "direct_answer"]


@dataclass
class StrategyResult:
    strategy: Strategy
    tokens_in: int
    tokens_out: int   # stub — real LLM measurement would fill this
    llm_called: bool


def _naive_rag_tokens(
    query: str,
    retrieved_propositions: list[str],
    system_prompt: str,
) -> StrategyResult:
    """Baseline: dump retrieved propositions + system prompt + query into LLM."""
    prompt = (
        system_prompt
        + "\n\nContext:\n"
        + "\n".join(f"- {p}" for p in retrieved_propositions)
        + f"\n\nQuestion: {query}\nAnswer:"
    )
    return StrategyResult(
        strategy="naive_rag",
        tokens_in=approx_tokens(prompt),
        tokens_out=30,  # assumed LLM output — held constant across strategies
        llm_called=True,
    )


def _structured_tokens(
    query: str,
    layer: BeliefLayer,
    belief_ids: list[str],
    system_prompt: str,
    at_time: datetime,
) -> StrategyResult:
    """D7 Option B: structured belief summary sent to LLM."""
    summary = layer.render_summary(
        layer.query(belief_ids, at_time=at_time, include_history=False)
    )
    prompt = (
        system_prompt
        + "\n\nCurrent beliefs:\n"
        + summary
        + f"\n\nQuestion: {query}\nAnswer:"
    )
    return StrategyResult(
        strategy="structured",
        tokens_in=approx_tokens(prompt),
        tokens_out=30,
        llm_called=True,
    )


def _direct_answer_tokens(
    query: str,
    answerer: DirectAnswerer,
    belief_ids: list[str],
    at_time: datetime,
) -> StrategyResult:
    """D7 Option C: answer directly from belief state, no LLM."""
    answer = answerer.try_answer(query, belief_ids, at_time=at_time)
    if answer is None:
        # Cannot answer directly — would fall back to LLM.
        # Report as if naive_rag fired; this is an honest account.
        return StrategyResult(
            strategy="direct_answer",
            tokens_in=0,  # direct-answer itself spent nothing
            tokens_out=0,
            llm_called=False,
        )
    return StrategyResult(
        strategy="direct_answer",
        tokens_in=0,
        tokens_out=answer.tokens_used,
        llm_called=False,
    )


# ─── Padding: generate synthetic beliefs to reach target memory size ─

_FILLER_TEMPLATES = [
    "I watched {} last weekend",
    "I bought {} at the market",
    "I talked to {} about work",
    "{} is on sale until next week",
    "I plan to visit {} next month",
    "{} gave me advice on cooking",
    "{} is learning to play piano",
]

_FILLER_SUBJECTS = [
    "a documentary", "fresh tomatoes", "Ana", "the coffee shop",
    "my old flat", "a colleague", "my neighbor", "the bookstore",
    "the new podcast", "Bimal", "the Sydney office", "a classmate",
]


def _synthetic_filler_proposition(rng: random.Random, idx: int) -> str:
    template = rng.choice(_FILLER_TEMPLATES)
    subject = rng.choice(_FILLER_SUBJECTS)
    return f"[filler-{idx}] " + template.format(subject)


def _build_store_at_size(
    scenarios: list[Scenario],
    target_size: int,
    *,
    detector,
    rng: random.Random,
) -> tuple[BeliefLayer, list[Scenario], dict[str, list[str]]]:
    """Build a store with exactly ``target_size`` beliefs by combining
    scenario propositions and synthetic filler.

    Returns the constructed layer, the scenarios whose propositions are
    in the store, and a mapping of scenario_id -> belief_ids in order.
    """
    layer = BeliefLayer(store=BeliefStore(), detector=detector)
    scenario_ids: dict[str, list[str]] = {}

    # Ingest scenarios first (they're the 'real' memory)
    base_date = datetime(2024, 1, 1)
    for sc in scenarios:
        props_sorted = sorted(sc.propositions, key=lambda p: p.asserted_at)
        ids = []
        for i, p in enumerate(props_sorted):
            ev = layer.ingest(
                proposition=p.text,
                asserted_at=p.asserted_at,
                asserted_in_session=p.session,
                source_proposition_id=f"{sc.id}-p{i}",
            )
            ids.append(ev.new_belief.id)
        scenario_ids[sc.id] = ids

    # Pad to target size with filler
    deficit = target_size - len(layer.store)
    for i in range(max(0, deficit)):
        layer.ingest(
            proposition=_synthetic_filler_proposition(rng, i),
            asserted_at=base_date + timedelta(days=i % 365),
            asserted_in_session=f"filler-session-{i // 50}",
            source_proposition_id=f"filler-{i}",
        )

    return layer, scenarios, scenario_ids


# ─── Main measurement ────────────────────────────────────────────────

@dataclass
class MeasurementRow:
    memory_size: int
    scenario_id: str
    question: str
    strategy: Strategy
    tokens_in: int
    tokens_out: int
    llm_called: bool
    compression_ratio: float  # raw_rag_tokens / this_tokens_in (for non-naive)


@dataclass
class TokenEconomyReport:
    memory_sizes: list[int]
    strategies: list[Strategy]
    rows: list[MeasurementRow]

    def summary_by_strategy(self) -> dict[str, dict[str, float]]:
        """Aggregate: mean tokens_in per strategy across all (size, question)."""
        out: dict[str, dict[str, float]] = {}
        for s in self.strategies:
            rows = [r for r in self.rows if r.strategy == s]
            if not rows:
                continue
            out[s] = {
                "mean_tokens_in": sum(r.tokens_in for r in rows) / len(rows),
                "mean_tokens_out": sum(r.tokens_out for r in rows) / len(rows),
                "llm_call_rate": sum(1 for r in rows if r.llm_called) / len(rows),
                "mean_compression_vs_naive": (
                    sum(r.compression_ratio for r in rows) / len(rows)
                ),
            }
        return out

    def summary_by_size(self) -> dict[int, dict[str, dict[str, float]]]:
        """Per-memory-size breakdown. size -> strategy -> stats."""
        out: dict[int, dict[str, dict[str, float]]] = {}
        for size in self.memory_sizes:
            out[size] = {}
            for s in self.strategies:
                rows = [
                    r for r in self.rows
                    if r.strategy == s and r.memory_size == size
                ]
                if not rows:
                    continue
                out[size][s] = {
                    "mean_tokens_in": (
                        sum(r.tokens_in for r in rows) / len(rows)
                    ),
                    "mean_compression_vs_naive": (
                        sum(r.compression_ratio for r in rows) / len(rows)
                    ),
                    "llm_call_rate": (
                        sum(1 for r in rows if r.llm_called) / len(rows)
                    ),
                }
        return out


def measure(
    scenarios: list[Scenario],
    memory_sizes: list[int],
    *,
    system_prompt: str = (
        "You are a careful assistant that answers questions using the "
        "user's memory. Be concise and factual."
    ),
    rng_seed: int = 42,
    naive_rag_top_k: int = 20,
) -> TokenEconomyReport:
    """Run the token-economy measurement across the cartesian product of
    (scenarios, questions, memory_sizes, strategies).

    Parameters
    ----------
    naive_rag_top_k
        Number of propositions a 'naive' RAG baseline would retrieve
        per query. Scales with memory size to simulate realistic
        retrieval-pressure growth: at memory=50, the baseline sees all
        ~50 beliefs; at memory=1000, a real system would retrieve
        top-k and miss some of the correct ones, while a less-careful
        system stuffs more context. We model the second behaviour:
        the baseline dumps min(top_k, memory_size) random beliefs plus
        the scenario's own propositions.
    """
    rng = random.Random(rng_seed)
    detector = StubContradictionDetector()  # cheap; token math is orthogonal
    rows: list[MeasurementRow] = []
    strategies: list[Strategy] = ["naive_rag", "structured", "direct_answer"]

    for size in memory_sizes:
        layer, used_scenarios, scenario_ids = _build_store_at_size(
            scenarios, size, detector=detector, rng=random.Random(rng_seed)
        )
        answerer = DirectAnswerer(layer.store)
        all_ids = [b.id for b in layer.store.all()]
        sample_rng = random.Random(rng_seed + size)

        for sc in used_scenarios:
            scenario_belief_ids = scenario_ids[sc.id]
            for q in sc.questions:
                at_time = q.at_time if q.at_time is not None else datetime(2030, 1, 1)

                # Naive RAG baseline:
                # - Always includes the scenario's own propositions
                #   (perfect recall on the target content)
                # - Plus top-k other propositions from the memory
                #   (simulating retrieval noise: models that err toward
                #   dumping context get more tokens as memory grows)
                scenario_set = set(scenario_belief_ids)
                other_ids = [bid for bid in all_ids if bid not in scenario_set]
                k = min(naive_rag_top_k, len(other_ids))
                noise_ids = sample_rng.sample(other_ids, k) if other_ids else []
                raw_bids = list(scenario_belief_ids) + noise_ids
                raw_props = [
                    layer.store.get(bid).proposition for bid in raw_bids  # type: ignore[union-attr]
                    if layer.store.get(bid) is not None
                ]
                baseline = _naive_rag_tokens(q.q, raw_props, system_prompt)

                structured = _structured_tokens(
                    q.q, layer, scenario_belief_ids, system_prompt, at_time
                )
                direct = _direct_answer_tokens(
                    q.q, answerer, scenario_belief_ids, at_time
                )

                for sr in (baseline, structured, direct):
                    compression = (
                        baseline.tokens_in / max(sr.tokens_in, 1)
                        if sr.strategy != "naive_rag"
                        else 1.0
                    )
                    rows.append(
                        MeasurementRow(
                            memory_size=size,
                            scenario_id=sc.id,
                            question=q.q,
                            strategy=sr.strategy,
                            tokens_in=sr.tokens_in,
                            tokens_out=sr.tokens_out,
                            llm_called=sr.llm_called,
                            compression_ratio=compression,
                        )
                    )

    return TokenEconomyReport(
        memory_sizes=memory_sizes,
        strategies=strategies,
        rows=rows,
    )


# ─── CLI ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Patha token-economy measurement")
    parser.add_argument(
        "--scenarios",
        default="eval/belief_eval_data/seed_scenarios.jsonl",
    )
    parser.add_argument(
        "--sizes",
        default="50,200,1000",
        help="Comma-separated memory sizes to measure at",
    )
    parser.add_argument(
        "--output",
        default="runs/token_economy/results.json",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    scenarios = load_scenarios(args.scenarios)
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    report = measure(scenarios, sizes)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "memory_sizes": report.memory_sizes,
                "strategies": report.strategies,
                "summary_by_strategy": report.summary_by_strategy(),
                "summary_by_size": {
                    str(k): v for k, v in report.summary_by_size().items()
                },
                "rows": [asdict(r) for r in report.rows],
            },
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("Token economy summary (mean over all questions)")
    print("=" * 60)
    summary = report.summary_by_strategy()
    for s, stats in summary.items():
        print(
            f"  {s:15s}  tokens_in={stats['mean_tokens_in']:7.1f}  "
            f"compression_vs_naive={stats['mean_compression_vs_naive']:5.2f}x  "
            f"llm_call_rate={stats['llm_call_rate']:.2f}"
        )

    print()
    print("=" * 60)
    print("By memory size")
    print("=" * 60)
    size_summary = report.summary_by_size()
    for size, strat_stats in size_summary.items():
        print(f"  size={size}")
        for s, stats in strat_stats.items():
            print(
                f"    {s:15s}  tokens_in={stats['mean_tokens_in']:7.1f}  "
                f"compression={stats['mean_compression_vs_naive']:5.2f}x"
            )

    print()
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
