"""AnalogyEval — the upamāna instrument: scorers + harness + CLI runner.

Measures analogical recall ("what does this remind me of?", "have I
been in a situation like X before?"): given a NEW situation described
in the question, does the system return the structurally-matching past
SITUATION (a multi-proposition episode, identified by session id) —
and can it name the shared abstract structure?

Built instrument-first per docs/roadmap.md section 5 (upamāna): no
production analogy route exists yet, so the harness scores any
*answerer* — a callable that receives the scenario context and the
question and returns a routing claim + ranked analogue sessions +
free-text structure naming. The committed floor is a deterministic
random-session stub. When upamāna ships, its answerer registers in
``ANSWERERS`` and is scored by the same frozen rubric.

Scenario format + the frozen rubric are documented in
eval/analogy_data/README.md. Rubric changes require a version bump and
a re-report — never silent edits.

Rubric history:
- v1 (frozen 2026-07-08): routed / analogue_hit_1 / analogue_hit_2 /
  trap_resistance / structure_overlap.

Design notes:
- The defining property of the gold sets is CONTENT-WORD DISJOINTNESS:
  the new situation and its gold analogue share structure but minimal
  vocabulary (that is what makes it analogy, not retrieval). The
  invariant is measured, not aspirational: tests enforce
  |content_tokens(question) ∩ content_tokens(gold session)| ≤ 2, and
  surface-trap scenarios must lexically favour the WRONG session.
- Scoring is session-level by exact session-id match. No fuzzy text
  matching anywhere except structure_overlap, which is explicitly the
  weakest scorer (see its docstring).
- Scorers are pure functions over (claimed_routed, returned_sessions,
  structure_text, gold) — unit-testable without any model. The harness
  only touches patha.Memory when the answerer declares
  ``needs_memory = True``; the stub floor runs model-free.
- Artifacts (routing claim + ranked sessions + structure text per
  question) are persisted in the results JSON so runs can be re-scored
  under future rubric versions without re-running (--rescore).
- Deterministic run-to-run: the stub seeds from md5(scenario id +
  question), never from Python's salted hash().

Usage:
    uv run python -m eval.analogy_eval \\
        --data eval/analogy_data/dev_scenarios.jsonl \\
        --answerer stub \\
        --output runs/analogy/dev-stub-floor.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

RUBRIC_VERSION = "v1"

_VALID_ROUTES = ("analogy", "ganita", "narrative", "retrieval")


# ─── Content tokens (shared by the weakest scorer and the data tests) ─

# Function words + question-frame vocabulary ("does this remind me of",
# "a situation like this") + light verbs. Frame words are stopworded so
# the disjointness invariant measures *situation* vocabulary, not the
# asking formula every analogy question shares.
_STOPWORDS = frozenset("""
a an the and or but if then than so to of in on at for with by from as
is are was were be been being am do does did done have has had having
will would can could should shall may might must not no nor
it its itself i me my mine myself we us our ours you your yours he him
his she her hers they them their theirs this that these those there
here what which who whom whose when where why how
all any both each few more most other some such only own same too very
just many much about into over under again once during between through
above below up down out off until while since after before ever never
always still also even now
like situation situations remind reminds reminded anything something
someone else way ways one two
get got gets getting go goes went going make makes made making take
takes took taking give gives gave given keep keeps kept let lets say
says said see sees saw seen
actually anyway every new thing things
""".split())


def content_tokens(text: str) -> set[str]:
    """Lowercased alphanumeric tokens, stopwords and <3-char tokens
    dropped, naive plural-s stripped (runs→run, plants→plant). This is
    the ONLY text normalisation in the instrument; both the
    structure_overlap scorer and the disjointness data-tests use it, so
    the invariant and the scorer can't drift apart."""
    out: set[str] = set()
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in _STOPWORDS or len(t) < 3:
            continue
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        out.add(t)
    return out


# ─── Scorers (frozen rubric — see analogy_data/README.md) ───────────


def score_routed(claimed_routed: bool, expected_route: str) -> float:
    """1.0 iff the routing claim matches the gold route. Only
    analogy-vs-not is scored: expected_route values other than
    "analogy" (ganita/narrative/retrieval) are advisory labels for
    routing-confusion analysis, not separately graded."""
    return 1.0 if claimed_routed == (expected_route == "analogy") else 0.0


def score_analogue_hit(
    returned_sessions: list[str],
    gold_sessions: list[str],
    k: int,
) -> float | None:
    """1.0 iff any of the top-k returned sessions is a gold analogue.
    Standard hit@k over the gold SET — the gold ranking is advisory
    (primary analogue first) and not position-graded in v1. 0.0 when
    the route was claimed but nothing gold surfaced in the top k
    (including empty returns: claiming the route owns the miss). None
    when there is no gold analogue (routing-negative controls)."""
    if not gold_sessions:
        return None
    g = set(gold_sessions)
    return 1.0 if any(s in g for s in returned_sessions[:k]) else 0.0


def score_trap_resistance(
    returned_sessions: list[str],
    gold_sessions: list[str],
    trap_session: str | None,
) -> float | None:
    """Surface-trap controls only (None elsewhere): hit@1 on a scenario
    whose vocabulary points at the WRONG session. 1.0 iff the top
    returned session is the structurally-right gold analogue — picking
    the lexically-seductive trap (or anything else) scores 0.0.
    Reported separately from analogue_hit_1 so trap scenarios can't
    hide inside the overall hit rate."""
    if trap_session is None:
        return None
    return score_analogue_hit(returned_sessions, gold_sessions, 1)


def score_structure_overlap(
    structure_text: str,
    shared_structure: list[str],
) -> float | None:
    """THE WEAKEST SCORER — documented as such. Fraction of gold
    shared-structure phrases the answer "names", where a phrase counts
    as named if ≥ half of its content tokens appear in the answer's
    structure text. A lexical proxy for a semantic property: it misses
    paraphrase ("time limit" does not name "external deadline") and it
    can be gamed by keyword-stuffing. Use it for coarse comparison
    between answerers, never as a headline number. None when the
    question has no gold structure (routing-negative controls); 0.0
    when the route was claimed but nothing was named."""
    if not shared_structure:
        return None
    answer_toks = content_tokens(structure_text)
    named = 0
    scored = 0
    for phrase in shared_structure:
        ptoks = content_tokens(phrase)
        if not ptoks:
            continue
        scored += 1
        if sum(1 for t in ptoks if t in answer_toks) / len(ptoks) >= 0.5:
            named += 1
    return (named / scored) if scored else None


def score_question(
    claimed_routed: bool,
    returned_sessions: list[str],
    structure_text: str,
    question: dict,
) -> dict:
    """Apply the full frozen rubric to one question's outcome. Routing
    failure on an analogy question gates everything else to None (the
    routed mean owns the failure) — same convention as EvolutionEval."""
    routed = score_routed(claimed_routed, question["expected_route"])
    if not claimed_routed:
        return {
            "routed": routed, "analogue_hit_1": None, "analogue_hit_2": None,
            "trap_resistance": None, "structure_overlap": None,
        }
    gold = question.get("gold_analogue_sessions", [])
    return {
        "routed": routed,
        "analogue_hit_1": score_analogue_hit(returned_sessions, gold, 1),
        "analogue_hit_2": score_analogue_hit(returned_sessions, gold, 2),
        "trap_resistance": score_trap_resistance(
            returned_sessions, gold, question.get("surface_trap_session"),
        ),
        "structure_overlap": score_structure_overlap(
            structure_text, question.get("shared_structure", []),
        ),
    }


_SCORER_NAMES = [
    "routed", "analogue_hit_1", "analogue_hit_2", "trap_resistance",
    "structure_overlap",
]


def aggregate(rows: list[dict]) -> dict:
    """Mean per scorer, None-excluded; count of contributing questions."""
    out = {}
    for name in _SCORER_NAMES:
        vals = [r["scores"][name] for r in rows if r["scores"][name] is not None]
        out[name] = {
            "mean": (sum(vals) / len(vals)) if vals else None,
            "n": len(vals),
        }
    return out


# ─── Answerers ──────────────────────────────────────────────────────


@dataclass
class AnalogyAnswer:
    """What an answerer must produce for one question."""

    routed: bool                       # claims the analogy route?
    analogue_sessions: list[str] = field(default_factory=list)  # ranked, best first
    structure_text: str = ""           # free text naming the shared structure


@dataclass
class ScenarioContext:
    """Everything an answerer may consult. ``memory`` is a live
    patha.Memory with the scenario ingested — populated only when the
    answerer declares ``needs_memory = True`` (the stub floor never
    pays the model cost)."""

    scenario: dict
    candidate_sessions: list[str]
    memory: Any | None = None


class Answerer(Protocol):
    """Callable scored by the harness. Set a class attribute
    ``needs_memory = True`` to receive an ingested patha.Memory in the
    context (a future upamāna answerer will; the stub doesn't)."""

    def __call__(self, ctx: ScenarioContext, question: str) -> AnalogyAnswer: ...


class RandomSessionAnswerer:
    """The floor: always claims the analogy route and returns two
    uniformly-drawn distinct sessions (no structure named). Seeded from
    md5(scenario id | question) so runs are byte-identical. Floors it
    establishes: routed = fraction of analogy questions (it never
    declines the route, so every routing-negative control scores 0);
    hit@k = chance of k draws landing in the gold set; trap_resistance
    = chance; structure_overlap = 0."""

    needs_memory = False

    def __call__(self, ctx: ScenarioContext, question: str) -> AnalogyAnswer:
        seed_src = f"{ctx.scenario['id']}|{question}".encode()
        seed = int.from_bytes(hashlib.md5(seed_src).digest()[:8], "big")
        rng = random.Random(seed)
        sessions = list(ctx.candidate_sessions)
        rng.shuffle(sessions)
        return AnalogyAnswer(
            routed=True,
            analogue_sessions=sessions[:2],
            structure_text="",
        )


ANSWERERS: dict[str, type] = {
    "stub": RandomSessionAnswerer,
}


# ─── Harness ────────────────────────────────────────────────────────


def candidate_sessions(scenario: dict) -> list[str]:
    """Distinct session ids in first-appearance (ingest) order."""
    seen: list[str] = []
    for prop in scenario["propositions"]:
        s = prop.get("session")
        if s and s not in seen:
            seen.append(s)
    return seen


def _ask_questions(scenario: dict, answerer, ctx: ScenarioContext) -> list[dict]:
    rows: list[dict] = []
    for q in scenario["questions"]:
        t0 = time.time()
        ans = answerer(ctx, q["q"])
        dt = time.time() - t0
        trap = q.get("surface_trap_session")
        rows.append({
            "scenario_id": scenario["id"],
            "family": scenario["family"],
            "question": q["q"],
            "expected_route": q["expected_route"],
            "claimed_routed": ans.routed,
            "returned_sessions": list(ans.analogue_sessions),
            "structure_text": ans.structure_text,
            "trap_taken": (
                bool(ans.analogue_sessions)
                and ans.analogue_sessions[0] == trap
            ) if trap else None,
            "seconds": round(dt, 2),
            "scores": score_question(
                ans.routed, list(ans.analogue_sessions),
                ans.structure_text, q,
            ),
        })
    return rows


def run_scenario(
    scenario: dict,
    answerer,
    *,
    detector: str = "stub",
    shared_embedder=None,
) -> list[dict]:
    """Score one scenario's questions with `answerer`. Builds a fresh
    ingested Memory only for answerers that declare ``needs_memory``;
    otherwise the run is model-free (the stub floor is CI-able).
    Returns one row per question (with artifacts for re-scoring)."""
    sessions = candidate_sessions(scenario)
    if not getattr(answerer, "needs_memory", False):
        ctx = ScenarioContext(scenario=scenario, candidate_sessions=sessions)
        return _ask_questions(scenario, answerer, ctx)

    import patha
    from datetime import datetime

    with tempfile.TemporaryDirectory(prefix="analogy-eval-") as td:
        mem = patha.Memory(
            path=Path(td) / "beliefs.jsonl",
            detector=detector,
            enable_phase1=True,
        )
        # Same shared-embedder injection point as evolution_eval:
        # the retriever lazily creates its embedder only if None.
        if shared_embedder is not None and mem._phase1_retriever is not None:
            mem._phase1_retriever._embedder = shared_embedder
        for i, prop in enumerate(scenario["propositions"]):
            mem.remember(
                prop["text"],
                asserted_at=datetime.fromisoformat(prop["asserted_at"]),
                session_id=prop.get("session"),
                source_id=f"analogy:{scenario['id']}#{i}",
            )
        ctx = ScenarioContext(
            scenario=scenario, candidate_sessions=sessions, memory=mem,
        )
        return _ask_questions(scenario, answerer, ctx)


def rescore_rows(rows: list[dict], scenarios_by_id: dict[str, dict]) -> list[dict]:
    """Re-apply the current rubric to persisted artifacts (no re-run)."""
    out = []
    for r in rows:
        scenario = scenarios_by_id[r["scenario_id"]]
        q = next(
            qq for qq in scenario["questions"] if qq["q"] == r["question"]
        )
        out.append({
            **r,
            "scores": score_question(
                bool(r["claimed_routed"]),
                r["returned_sessions"],
                r.get("structure_text", ""),
                q,
            ),
        })
    return out


# ─── CLI ────────────────────────────────────────────────────────────


def _load_scenarios(path: Path, include_heldout: bool) -> list[dict]:
    scenarios = []
    sealed = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("heldout") and not include_heldout:
                sealed += 1
                continue
            scenarios.append(s)
    if sealed:
        print(
            f"refused {sealed} sealed held-out scenarios "
            f"(pass --include-heldout for a RELEASE REPORT run only — "
            f"never between tuning iterations)",
            file=sys.stderr,
        )
    return scenarios


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AnalogyEval runner")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--answerer", default="stub",
        help="Which answerer to score: a registry name ('stub', the "
             "random-session floor) or a 'module:attr' dotted path to "
             "an Answerer instance/class (absence_eval's pattern).",
    )
    p.add_argument(
        "--detector", default="stub",
        help="Detector stack for Memory ingest (only used when the "
             "answerer declares needs_memory).",
    )
    p.add_argument(
        "--include-heldout", action="store_true",
        help="Unseal held-out scenarios. Release reports ONLY.",
    )
    p.add_argument(
        "--rescore", type=Path, default=None,
        help="Re-apply the current rubric to a prior results JSON "
             "(reads artifacts; does not re-run the answerer).",
    )
    p.add_argument("--max-scenarios", type=int, default=None)
    args = p.parse_args(argv)

    scenarios = _load_scenarios(args.data, args.include_heldout)
    if args.max_scenarios:
        scenarios = scenarios[: args.max_scenarios]

    if args.rescore:
        prior = json.loads(args.rescore.read_text())
        by_id = {s["id"]: s for s in scenarios}
        rows = rescore_rows(prior["rows"], by_id)
    else:
        if args.answerer in ANSWERERS:
            answerer = ANSWERERS[args.answerer]()
        else:
            import importlib
            module, attr = args.answerer.rsplit(":", 1)
            obj = getattr(importlib.import_module(module), attr)
            answerer = obj() if isinstance(obj, type) else obj
        shared = None
        if getattr(answerer, "needs_memory", False):
            from patha.models.embedder_st import SentenceTransformerEmbedder
            shared = SentenceTransformerEmbedder()
        rows = []
        t0 = time.time()
        for i, s in enumerate(scenarios, 1):
            rows.extend(run_scenario(
                s, answerer, detector=args.detector, shared_embedder=shared,
            ))
            print(
                f"  [{i}/{len(scenarios)}] {s['id']} "
                f"({s['family']})", file=sys.stderr,
            )
        print(f"ran {len(scenarios)} scenarios in {time.time()-t0:.0f}s",
              file=sys.stderr)

    overall = aggregate(rows)
    families = sorted({r["family"] for r in rows})
    by_family = {
        fam: aggregate([r for r in rows if r["family"] == fam])
        for fam in families
    }

    def fmt(a: dict) -> str:
        return "  ".join(
            f"{name}={a[name]['mean']:.3f}({a[name]['n']})"
            if a[name]["mean"] is not None else f"{name}=—(0)"
            for name in _SCORER_NAMES
        )

    print(f"\nAnalogyEval rubric {RUBRIC_VERSION} — {len(rows)} questions")
    print(f"overall: {fmt(overall)}")
    for fam in families:
        print(f"  {fam:24s} {fmt(by_family[fam])}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "rubric_version": RUBRIC_VERSION,
            "data": str(args.data),
            "answerer": args.answerer if not args.rescore else None,
            "detector": args.detector,
            "n_questions": len(rows),
            "overall": overall,
            "by_family": by_family,
            "rows": rows,
        }, indent=2))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
