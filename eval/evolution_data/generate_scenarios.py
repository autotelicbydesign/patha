"""EvolutionEval scenario generator — the dev set.

EvolutionEval measures whether a memory system can reconstruct HOW A
THEME EVOLVED over time: the ordered beats, the origin, the revisions,
and the exclusion of interleaved off-theme distractors. No external
benchmark measures this (verified against LongMemEval 2026-07: its
temporal-reasoning stratum is timestamp arithmetic; knowledge-update is
single-fact changes). EvolutionEval is authored to define the category.

Four scenario families, each grounded in an OBSERVED failure mode from
the Phase-4 dogfood (docs/phase_4_dogfood.md):

  progressive_revelation   Theme starts vague, becomes specific.
                           Targets F6: origin identification.
  multi_factor_change      A change in one life-factor causes a change
                           in the theme. The causal beat is off-theme by
                           substring — documented HEADROOM for the
                           current walker (composition precursor).
  perspective_shift        Same events, reinterpreted later — a reversal
                           WITHOUT lexical contradiction. Targets the
                           N1 finding: supersession detectors don't fire
                           on reinterpretation, so through-lines said
                           "no reversals" when the reversal was the story.
  reversed_belief_chain    X → Y → back to X′ (nonmonotonic). Tests that
                           the middle beat survives and order holds.

Determinism: NO randomness. Templates × fixed slot tables, iterated in
order. Running this script twice produces byte-identical output — the
dev set is regenerable and diffable.

Overfit guards (docs/phase_4_dogfood.md, "Discipline stop"):
  - This generator produces the DEV set only. The held-out set
    (heldout_scenarios.jsonl) is hand-written in different surface
    domains and is SEALED — never tune walker knobs against it.
  - The walker was frozen (commit 5ca7738) BEFORE these scenarios were
    authored.
  - Scoring rubric is frozen in eval/evolution_eval.py before the first
    reported run.

Usage:
    uv run python -m eval.evolution_data.generate_scenarios \\
        --output eval/evolution_data/dev_scenarios.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ─── Slot tables (fixed, ordered — determinism depends on this) ─────
#
# Each entry: (theme_token, vague, concrete_1, concrete_2, deepening)
# The theme_token must appear in SOME beats and be paraphrased away in
# others — that mix is what exercises the topic-cluster gate.

_PROGRESSIVE_THEMES = [
    ("woodworking",
     "I've been thinking about making things with my hands lately",
     "I took an intro woodworking class at the community shop",
     "I bought my first set of chisels and a dovetail saw",
     "I finished a walnut box with hand-cut dovetails — my first real piece"),
    ("running",
     "I feel like I need something physical in my mornings",
     "I started running twice a week before work",
     "I signed up for a 10k in the spring",
     "I now run four times a week and the 10k felt easy"),
    ("meditation",
     "my mind has felt scattered and loud for months",
     "I tried a guided meditation app for ten minutes a day",
     "I sit for twenty minutes every morning without the app",
     "meditation is now the anchor of my day; I notice when I skip it"),
    ("photography",
     "I keep noticing light on buildings and wishing I could hold onto it",
     "I started taking photos on walks with my phone",
     "I bought a used mirrorless camera and a 35mm lens",
     "I shot a friend's engagement and they printed three of my photos"),
    ("gardening",
     "I want something slow and alive in my week",
     "I planted herbs in pots on the balcony",
     "I built two raised beds and started tomatoes from seed",
     "the garden feeds us most of the summer now"),
    ("piano",
     "I miss having music I make myself, not just listen to",
     "I started piano lessons as a complete beginner",
     "I can play two pieces from memory now",
     "I performed at the school recital and didn't freeze"),
    ("spanish",
     "I feel embarrassed ordering food when we visit my partner's family",
     "I started Spanish on a language app during commutes",
     "I have a weekly conversation exchange with a tutor",
     "I got through a whole family dinner in Spanish without switching"),
    ("climbing",
     "I want a sport that feels like solving puzzles",
     "I tried the bouldering gym with a friend",
     "I climb three times a week and finished my first V4",
     "I led my first outdoor route this weekend"),
    ("journaling",
     "my thoughts evaporate before I can use them",
     "I started writing three sentences every night",
     "the journal is now two pages every morning",
     "reading January's journal showed me how much my thinking has changed"),
    ("chess",
     "I want a game I can get better at for years",
     "I started playing chess online in the evenings",
     "I study one opening a week and do puzzles daily",
     "I won my first rated tournament game"),
]

# (theme_token, initial_state, cause_offtheme, resulting_change)
_MULTIFACTOR_THEMES = [
    ("gym",
     "I lift at the gym every morning before work",
     "my new job starts daily standups at 8am sharp",
     "I moved my gym sessions to the evenings after work"),
    ("cooking",
     "I cook elaborate dinners most weeknights",
     "the baby arrived in March and evenings are chaos",
     "I batch-cook simple meals on Sundays now"),
    ("cycling",
     "I cycle to the office every day along the river",
     "we moved to the hills outside the city in autumn",
     "I take the train now and only ride on weekends"),
    ("reading",
     "I read a novel a week, mostly at night",
     "my eyes started straining badly after the screen redesign project",
     "I switched almost entirely to audiobooks on walks"),
    ("budgeting",
     "I track every expense in a spreadsheet weekly",
     "my freelance income became three retainer clients",
     "I moved to a quarterly budget review since income is stable now"),
    ("coffee",
     "I drink three espressos a day, first one at 6am",
     "the doctor flagged my blood pressure at the annual checkup",
     "I'm down to one coffee before noon"),
    ("painting",
     "I paint landscapes in oils every Sunday in the spare room",
     "my sister moved in with us for a while after her surgery",
     "I switched to a small watercolor kit I can use at the kitchen table"),
    ("swimming",
     "I swim laps at the outdoor pool three mornings a week",
     "the council closed the outdoor pool for renovation until next year",
     "I joined the indoor pool across town and go twice a week"),
    ("volunteering",
     "I volunteer at the food bank every Saturday morning",
     "my mother now needs weekend care visits",
     "I switched to the food bank's Wednesday evening shift"),
    ("guitar",
     "I practice guitar an hour every evening",
     "the new apartment has paper-thin walls and a grumpy neighbor",
     "I bought a silent travel guitar and practice with headphones"),
]

# (theme_token, event_and_first_read, deepening_first_read, reinterpretation, settled_view)
_PERSPECTIVE_THEMES = [
    ("marathon",
     "I dropped out of the marathon at km 30 and I feel like a failure",
     "everyone finished except me; I keep replaying the marathon",
     "looking back, dropping out of the marathon was the smartest thing my body ever made me do",
     "the marathon DNF taught me more about pacing than any finish line has"),
    ("startup",
     "the startup shut down today; two years of my life wasted",
     "I can't even look at the startup's logo without feeling sick",
     "with distance, the startup was the best education I never paid for",
     "I talk about the startup in interviews now as my proudest chapter"),
    ("breakup",
     "the breakup blindsided me; I keep asking what I did wrong",
     "three months of replaying every conversation from before the breakup",
     "honestly the breakup freed us both; we wanted different lives",
     "we had coffee last week and I felt only warmth about how the breakup went"),
    ("layoff",
     "I got laid off this morning; I feel discarded",
     "the layoff has me doubting whether I was ever any good",
     "the layoff pushed me to freelance, which I'd never have dared myself",
     "my income and my mornings are both better since the layoff"),
    ("critique",
     "the design critique shredded my portfolio piece; I'm gutted",
     "I haven't opened the portfolio file since the critique",
     "rereading the critique notes, almost every point was a gift",
     "that critique is why the redesign is the strongest thing I've shipped"),
    ("injury",
     "the knee injury ended my season and maybe my running",
     "watching the team play without me after the injury is torture",
     "the injury forced me into coaching, and I love it more than playing",
     "if the injury hadn't happened I'd never have found coaching"),
    ("rejection",
     "the manuscript was rejected by all five publishers; maybe I'm not a writer",
     "I put the manuscript in a drawer after the rejections",
     "rereading it a year on, the rejections were right — and fixable",
     "the rewritten manuscript just got an agent; the rejections built it"),
    ("failure",
     "I failed the licensing exam by two points",
     "everyone at work knows I failed the exam; the shame is loud",
     "failing the exam exposed exactly which two topics I'd faked knowing",
     "I passed with the highest score in my cohort; the failure was the syllabus"),
]

# (theme_token, initial, reversal, return_with_nuance)
_REVERSED_THEMES = [
    ("coffee",
     "I drink coffee every single morning; it's non-negotiable",
     "I quit coffee entirely — two weeks caffeine-free now",
     "I'm back on coffee, but only one cup and never after noon"),
    ("twitter",
     "I post on twitter daily; it's where my whole field talks",
     "I deleted twitter off my phone; the noise was eating my focus",
     "I'm back on twitter but only Tuesdays, to share what I shipped"),
    ("meat",
     "I eat meat with most meals; always have",
     "I went fully vegetarian in January",
     "I eat fish again now, but still no red meat"),
    ("city",
     "I love living in the city center; the energy feeds me",
     "we moved to the countryside for the quiet",
     "we're moving back to the city, but to a leafy edge neighborhood"),
    ("news",
     "I read the news first thing every morning",
     "I went on a total news fast for a month",
     "I read a weekly digest now instead of the daily churn"),
    ("openplan",
     "I do my best work in the open-plan office buzz",
     "I switched to working from home full-time for deep focus",
     "I do two office days for people, three home days for focus"),
    ("phone",
     "my phone is always within reach; I answer everything instantly",
     "I started leaving the phone in a drawer all evening",
     "the phone lives in the hallway now; I check it three times a day"),
    ("sugar",
     "I bake something sweet every weekend and eat dessert daily",
     "I cut out sugar completely for three months",
     "I bake again on birthdays only, and dessert is a weekend thing"),
]

# Off-theme distractor pool — interleaved into every scenario so
# precision (distractor exclusion) is measurable. Fixed order.
_DISTRACTORS = [
    "the dentist moved my appointment to Thursday",
    "my sister's flight lands at 6pm on Saturday",
    "the car needs new brake pads before winter",
    "I renewed my passport; it took nine days",
    "the wifi router died and the replacement arrives Monday",
    "I finally fixed the squeaky hinge on the bathroom door",
]


def _iso(month: int, day: int, hour: int = 9) -> str:
    return f"2025-{month:02d}-{day:02d}T{hour:02d}:00:00"


def _props(on_theme: list[tuple[str, str]], distractors: list[tuple[str, str]]):
    """Merge (iso, text) beats + distractors, sorted by date — the
    benchmark ingests chronologically, like a real life would."""
    merged = sorted(on_theme + distractors, key=lambda x: x[0])
    props = [
        {"text": text, "asserted_at": iso, "session": f"s{i+1}"}
        for i, (iso, text) in enumerate(merged)
    ]
    # gold indices = positions of on-theme beats after the merge,
    # in chronological order
    on_set = {id(t) for t in on_theme}
    gold = [i for i, item in enumerate(merged) if id(item) in on_set]
    distractor_idx = [i for i in range(len(merged)) if i not in gold]
    return props, gold, distractor_idx


def gen_progressive_revelation() -> list[dict]:
    out = []
    for n, (theme, vague, c1, c2, deep) in enumerate(_PROGRESSIVE_THEMES):
        beats = [
            (_iso(1, 5), vague),
            (_iso(3, 10), c1),
            (_iso(6, 15), c2),
            (_iso(10, 20), deep),
        ]
        distractors = [
            (_iso(2, 7), _DISTRACTORS[n % len(_DISTRACTORS)]),
            (_iso(7, 12), _DISTRACTORS[(n + 3) % len(_DISTRACTORS)]),
        ]
        props, gold, dist = _props(beats, distractors)
        out.append({
            "id": f"pr-{n+1:03d}",
            "family": "progressive_revelation",
            "propositions": props,
            "questions": [{
                "q": f"how has my thinking about {theme} evolved?",
                "type": "evolution",
                "expected_theme": theme,
                "expected_beat_order": gold,
                "expected_origin": gold[0],
                "distractor_indices": dist,
                "expected_supersessions": [],
            }],
        })
    return out


def gen_multi_factor_change() -> list[dict]:
    out = []
    for n, (theme, initial, cause, change) in enumerate(_MULTIFACTOR_THEMES):
        beats = [
            (_iso(1, 8), initial),
            (_iso(5, 14), change),
        ]
        # The CAUSE is gold but off-theme by substring — the documented
        # headroom case. It sits between initial and change.
        cause_beat = [(_iso(4, 2), cause)]
        distractors = [
            (_iso(2, 20), _DISTRACTORS[n % len(_DISTRACTORS)]),
            (_iso(8, 5), _DISTRACTORS[(n + 2) % len(_DISTRACTORS)]),
        ]
        props, gold, dist = _props(beats + cause_beat, distractors)
        out.append({
            "id": f"mf-{n+1:03d}",
            "family": "multi_factor_change",
            "propositions": props,
            "questions": [{
                "q": f"how has my {theme} routine changed and why?",
                "type": "evolution",
                "expected_theme": theme,
                "expected_beat_order": gold,
                "expected_origin": gold[0],
                "distractor_indices": dist,
                "expected_supersessions": [[gold[0], gold[-1]]],
            }],
        })
    return out


def gen_perspective_shift() -> list[dict]:
    out = []
    for n, (theme, first, deepen, reinterpret, settled) in enumerate(
        _PERSPECTIVE_THEMES
    ):
        beats = [
            (_iso(1, 12), first),
            (_iso(2, 18), deepen),
            (_iso(7, 9), reinterpret),
            (_iso(11, 3), settled),
        ]
        distractors = [
            (_iso(4, 22), _DISTRACTORS[n % len(_DISTRACTORS)]),
            (_iso(9, 16), _DISTRACTORS[(n + 1) % len(_DISTRACTORS)]),
        ]
        props, gold, dist = _props(beats, distractors)
        out.append({
            "id": f"ps-{n+1:03d}",
            "family": "perspective_shift",
            "propositions": props,
            "questions": [{
                "q": f"how has my view of the {theme} changed over time?",
                "type": "evolution",
                "expected_theme": theme,
                "expected_beat_order": gold,
                "expected_origin": gold[0],
                "distractor_indices": dist,
                # The reversal is REINTERPRETIVE, not lexical — the
                # first-read beat should end up marked as revised once
                # detectors can catch reinterpretation. Documented
                # headroom for stub + current NLI detectors.
                "expected_supersessions": [[gold[0], gold[2]]],
            }],
        })
    return out


def gen_reversed_belief_chain() -> list[dict]:
    out = []
    for n, (theme, initial, reversal, ret) in enumerate(_REVERSED_THEMES):
        beats = [
            (_iso(1, 15), initial),
            (_iso(5, 20), reversal),
            (_iso(10, 10), ret),
        ]
        distractors = [
            (_iso(3, 11), _DISTRACTORS[n % len(_DISTRACTORS)]),
            (_iso(7, 25), _DISTRACTORS[(n + 4) % len(_DISTRACTORS)]),
        ]
        props, gold, dist = _props(beats, distractors)
        out.append({
            "id": f"rb-{n+1:03d}",
            "family": "reversed_belief_chain",
            "propositions": props,
            "questions": [{
                "q": f"trace how my relationship with {theme} has gone back and forth",
                "type": "evolution",
                "expected_theme": theme,
                "expected_beat_order": gold,
                "expected_origin": gold[0],
                "distractor_indices": dist,
                "expected_supersessions": [
                    [gold[0], gold[1]], [gold[1], gold[2]],
                ],
            }],
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate EvolutionEval dev set")
    p.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "dev_scenarios.jsonl",
    )
    args = p.parse_args()

    scenarios = (
        gen_progressive_revelation()
        + gen_multi_factor_change()
        + gen_perspective_shift()
        + gen_reversed_belief_chain()
    )
    with open(args.output, "w") as f:
        for s in scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    fam = {}
    for s in scenarios:
        fam[s["family"]] = fam.get(s["family"], 0) + 1
    print(f"wrote {len(scenarios)} scenarios → {args.output}")
    for k, v in fam.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
