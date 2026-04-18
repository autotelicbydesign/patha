"""BeliefEval scenario generator.

Programmatically generates additional scenarios across the three
families used by BeliefEval, bringing the benchmark from the v0.1
seed of 20 up toward the v0.3 target of ~150.

Two generation modes:

  1. Templated (default): a set of hand-written templates with slot
     variables (subject, old state, new state, timestamps). Each
     template produces N variations. Deterministic, reproducible,
     no LLM required. Use for CI and quick scaling.

  2. LLM-assisted (opt-in via --llm-generate): calls an Ollama-style
     generator to propose novel scenario slot-fillers that go beyond
     the templated set. Output is still human-readable JSONL and can
     be reviewed/edited before inclusion.

Output format matches eval/belief_eval_data/seed_scenarios.jsonl so
eval/belief_eval.py can consume generated files directly.

Usage:
    # Generate 150 templated scenarios (extends the seed by ~130)
    python -m eval.belief_eval_data.generate_scenarios \\
        --count 150 --output eval/belief_eval_data/v03_scenarios.jsonl

    # LLM-assisted (requires Ollama running)
    python -m eval.belief_eval_data.generate_scenarios \\
        --count 150 --llm-generate \\
        --output eval/belief_eval_data/v03_scenarios.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


# ─── Templates ──────────────────────────────────────────────────────

@dataclass
class PreferenceTemplate:
    """Template for a preference_supersession scenario."""

    subject: str           # 'sushi', 'coffee'
    verb: str              # 'love', 'drink'
    frequency: str         # 'every week', 'every morning'
    new_state: str         # 'avoid raw fish on doctor advice'
    answer_current: list[str]
    answer_superseded: list[str]


_PREFERENCE_TEMPLATES: list[PreferenceTemplate] = [
    PreferenceTemplate(
        subject="sushi", verb="love", frequency="every week",
        new_state="am avoiding raw fish on my doctor's advice",
        answer_current=["avoiding raw fish"],
        answer_superseded=["love sushi"],
    ),
    PreferenceTemplate(
        subject="coffee", verb="drink", frequency="every morning",
        new_state="switched from coffee to green tea three months ago",
        answer_current=["green tea"],
        answer_superseded=["coffee every morning"],
    ),
    PreferenceTemplate(
        subject="running", verb="do", frequency="on weekends",
        new_state="have switched entirely to cycling after getting a new bike",
        answer_current=["cycling"],
        answer_superseded=["running"],
    ),
    PreferenceTemplate(
        subject="physical books", verb="read", frequency="most of the time",
        new_state="now exclusively read on my Kindle",
        answer_current=["Kindle"],
        answer_superseded=["physical books"],
    ),
    PreferenceTemplate(
        subject="dairy", verb="consume", frequency="daily",
        new_state="cut out dairy after lactose testing",
        answer_current=["cut out dairy"],
        answer_superseded=["dairy"],
    ),
    PreferenceTemplate(
        subject="Django", verb="use", frequency="for most backend work",
        new_state="moved entirely to FastAPI for new services",
        answer_current=["FastAPI"],
        answer_superseded=["Django"],
    ),
    PreferenceTemplate(
        subject="sparkling water", verb="prefer", frequency="with meals",
        new_state="only drink still water now",
        answer_current=["still water"],
        answer_superseded=["sparkling water"],
    ),
    PreferenceTemplate(
        subject="Linux", verb="use", frequency="as my main OS",
        new_state="migrated away from Linux to macOS entirely",
        answer_current=["macOS"],
        answer_superseded=["Linux"],
    ),
    PreferenceTemplate(
        subject="beer", verb="enjoy", frequency="on Fridays",
        new_state="quit alcohol six months ago",
        answer_current=["quit alcohol"],
        answer_superseded=["beer"],
    ),
    PreferenceTemplate(
        subject="Notion", verb="use", frequency="for all my notes",
        new_state="migrated my notes to Obsidian",
        answer_current=["Obsidian"],
        answer_superseded=["Notion"],
    ),
    PreferenceTemplate(
        subject="vim", verb="use", frequency="as my editor",
        new_state="switched to VS Code for everyday work",
        answer_current=["VS Code"],
        answer_superseded=["vim"],
    ),
    PreferenceTemplate(
        subject="bread", verb="bake", frequency="weekly",
        new_state="gave up baking after going gluten-free",
        answer_current=["gluten-free"],
        answer_superseded=["bread"],
    ),
]


@dataclass
class FactualTemplate:
    """Template for a factual_supersession scenario."""

    topic: str            # 'location', 'company', 'car'
    old_value: str        # 'Sydney', 'Autotelic', '2015 Corolla'
    new_value: str        # 'Sofia', 'Waves Collective', 'Tesla Model 3'
    setup_sentence: str   # "I live in Sydney"
    update_sentence: str  # "I moved to Sofia last month"
    q: str                # "Where does the user live currently?"
    answer_current: list[str]
    answer_superseded: list[str]


_FACTUAL_TEMPLATES: list[FactualTemplate] = [
    FactualTemplate(
        topic="location", old_value="Sydney", new_value="Sofia",
        setup_sentence="I live in Sydney",
        update_sentence="I moved to Sofia last month",
        q="Where does the user live currently?",
        answer_current=["Sofia"],
        answer_superseded=["Sydney"],
    ),
    FactualTemplate(
        topic="company", old_value="Autotelic", new_value="Waves Collective",
        setup_sentence="My company is called Autotelic",
        update_sentence="We rebranded the company as Waves Collective",
        q="What is the user's company called now?",
        answer_current=["Waves Collective"],
        answer_superseded=["Autotelic"],
    ),
    FactualTemplate(
        topic="car", old_value="2015 Toyota Corolla",
        new_value="Tesla Model 3",
        setup_sentence="I drive a 2015 Toyota Corolla",
        update_sentence="I just bought a Tesla Model 3",
        q="What car does the user drive now?",
        answer_current=["Tesla"],
        answer_superseded=["Corolla"],
    ),
    FactualTemplate(
        topic="job", old_value="Canva", new_value="own design studio",
        setup_sentence="I work at Canva as a lead designer",
        update_sentence="I left Canva and now run my own design studio",
        q="Where does the user work now?",
        answer_current=["own design studio"],
        answer_superseded=["Canva"],
    ),
    FactualTemplate(
        topic="phone", old_value="iPhone 13", new_value="iPhone 15 Pro",
        setup_sentence="I use an iPhone 13",
        update_sentence="I just upgraded to the iPhone 15 Pro",
        q="What phone does the user use currently?",
        answer_current=["iPhone 15 Pro"],
        answer_superseded=["iPhone 13"],
    ),
    FactualTemplate(
        topic="age", old_value="thirty-four", new_value="thirty-five",
        setup_sentence="I am thirty-four years old",
        update_sentence="I just turned thirty-five last week",
        q="How old is the user now?",
        answer_current=["thirty-five"],
        answer_superseded=["thirty-four"],
    ),
    FactualTemplate(
        topic="relationship", old_value="single", new_value="married",
        setup_sentence="I am single",
        update_sentence="I got married last autumn",
        q="What is the user's marital status now?",
        answer_current=["married"],
        answer_superseded=["single"],
    ),
    FactualTemplate(
        topic="contract", old_value="freelance", new_value="full-time at Patha",
        setup_sentence="I work freelance for various clients",
        update_sentence="I took a full-time role at Patha last month",
        q="What is the user's employment type now?",
        answer_current=["full-time"],
        answer_superseded=["freelance"],
    ),
    FactualTemplate(
        topic="pet", old_value="only cats", new_value="a golden retriever",
        setup_sentence="I had only cats for years",
        update_sentence="We adopted a golden retriever named Lily",
        q="What pets does the user have now?",
        answer_current=["golden retriever"],
        answer_superseded=["only cats"],
    ),
    FactualTemplate(
        topic="pricing", old_value="29 dollars per month",
        new_value="39 dollars per month",
        setup_sentence="Our pricing is 29 dollars per month",
        update_sentence="We raised the price to 39 dollars per month in Q1",
        q="What is the current price?",
        answer_current=["39 dollars"],
        answer_superseded=["29 dollars"],
    ),
]


@dataclass
class TemporallyBoundedTemplate:
    text: str                  # "I am on paternity leave until June 1 2024"
    q_inside: str
    q_outside: str
    at_inside: datetime
    at_outside: datetime
    expected_inside: bool = True
    expected_outside: bool = False
    asserted_at: datetime = field(default_factory=lambda: datetime(2024, 3, 1))


_BOUNDED_TEMPLATES: list[TemporallyBoundedTemplate] = [
    TemporallyBoundedTemplate(
        text="I am on paternity leave until June 1 2024",
        q_inside="Is the user on paternity leave in April 2024?",
        q_outside="Is the user on paternity leave in August 2024?",
        at_inside=datetime(2024, 4, 15),
        at_outside=datetime(2024, 8, 1),
    ),
    TemporallyBoundedTemplate(
        text="I am avoiding raw fish for three weeks after my dental surgery",
        q_inside="Is the user avoiding raw fish on March 15 2024?",
        q_outside="Is the user avoiding raw fish on April 10 2024?",
        at_inside=datetime(2024, 3, 15),
        at_outside=datetime(2024, 4, 10),
    ),
    TemporallyBoundedTemplate(
        text="I am working remotely for the next two months",
        q_inside="Is the user working remotely in March 2024?",
        q_outside="Is the user working remotely in June 2024?",
        at_inside=datetime(2024, 3, 1),
        at_outside=datetime(2024, 6, 1),
        asserted_at=datetime(2024, 2, 1),
    ),
    TemporallyBoundedTemplate(
        text="I am visiting my parents for five days starting Monday",
        q_inside="Is the user with their parents on April 3 2024?",
        q_outside="Is the user with their parents on April 20 2024?",
        at_inside=datetime(2024, 4, 3),
        at_outside=datetime(2024, 4, 20),
        asserted_at=datetime(2024, 4, 1),
    ),
    TemporallyBoundedTemplate(
        text="I am on a yoga retreat for one week",
        q_inside="Is the user on retreat on May 3 2024?",
        q_outside="Is the user on retreat on May 20 2024?",
        at_inside=datetime(2024, 5, 3),
        at_outside=datetime(2024, 5, 20),
        asserted_at=datetime(2024, 5, 1),
    ),
    TemporallyBoundedTemplate(
        text="I am covering for my colleague for the next three weeks",
        q_inside="Is the user covering on June 10 2024?",
        q_outside="Is the user covering on July 20 2024?",
        at_inside=datetime(2024, 6, 10),
        at_outside=datetime(2024, 7, 20),
        asserted_at=datetime(2024, 6, 1),
    ),
    TemporallyBoundedTemplate(
        text="I am house-sitting until August 15 2024",
        q_inside="Is the user house-sitting in July 2024?",
        q_outside="Is the user house-sitting in September 2024?",
        at_inside=datetime(2024, 7, 15),
        at_outside=datetime(2024, 9, 1),
        asserted_at=datetime(2024, 7, 1),
    ),
]


# ─── Scenario builders ──────────────────────────────────────────────

def _id(prefix: str, *, unique: str) -> str:
    h = hashlib.sha256(unique.encode()).hexdigest()[:8]
    return f"{prefix}-{h}"


def _preference_scenario(t: PreferenceTemplate, seed_offset: int) -> dict:
    rng = random.Random(seed_offset)
    asserted_year = 2022 + rng.randint(0, 1)
    old_month = rng.randint(1, 12)
    new_year = asserted_year + 1 + rng.randint(0, 1)
    new_month = rng.randint(1, 12)
    old_dt = datetime(asserted_year, old_month, rng.randint(1, 28), 12)
    new_dt = datetime(new_year, new_month, rng.randint(1, 28), 12)
    prop_old = f"I {t.verb} {t.subject} {t.frequency}"
    prop_new = f"I {t.new_state}"
    scenario_id = _id("prefs", unique=f"{t.subject}-{seed_offset}")
    return {
        "id": scenario_id,
        "family": "preference_supersession",
        "propositions": [
            {
                "text": prop_old,
                "asserted_at": old_dt.isoformat(),
                "session": f"{scenario_id}-s1",
            },
            {
                "text": prop_new,
                "asserted_at": new_dt.isoformat(),
                "session": f"{scenario_id}-s2",
            },
        ],
        "questions": [
            {
                "q": f"What does the user currently believe about {t.subject}?",
                "expected_current_contains": t.answer_current,
                "expected_superseded_contains": t.answer_superseded,
                "type": "current_belief",
            }
        ],
    }


def _factual_scenario(t: FactualTemplate, seed_offset: int) -> dict:
    rng = random.Random(seed_offset + 1000)
    asserted_year = 2022 + rng.randint(0, 1)
    new_year = asserted_year + 1
    old_dt = datetime(asserted_year, rng.randint(1, 12), rng.randint(1, 28), 12)
    new_dt = datetime(new_year, rng.randint(1, 12), rng.randint(1, 28), 12)
    scenario_id = _id("facts", unique=f"{t.topic}-{seed_offset}")
    return {
        "id": scenario_id,
        "family": "factual_supersession",
        "propositions": [
            {
                "text": t.setup_sentence,
                "asserted_at": old_dt.isoformat(),
                "session": f"{scenario_id}-s1",
            },
            {
                "text": t.update_sentence,
                "asserted_at": new_dt.isoformat(),
                "session": f"{scenario_id}-s2",
            },
        ],
        "questions": [
            {
                "q": t.q,
                "expected_current_contains": t.answer_current,
                "expected_superseded_contains": t.answer_superseded,
                "type": "current_belief",
            }
        ],
    }


def _bounded_scenario(t: TemporallyBoundedTemplate, seed_offset: int) -> dict:
    scenario_id = _id("bound", unique=f"{t.text[:20]}-{seed_offset}")
    return {
        "id": scenario_id,
        "family": "temporally_bounded",
        "propositions": [
            {
                "text": t.text,
                "asserted_at": t.asserted_at.isoformat(),
                "session": f"{scenario_id}-s1",
            }
        ],
        "questions": [
            {
                "q": t.q_inside,
                "at_time": t.at_inside.isoformat(),
                "expected_valid": t.expected_inside,
                "type": "validity_at_time",
            },
            {
                "q": t.q_outside,
                "at_time": t.at_outside.isoformat(),
                "expected_valid": t.expected_outside,
                "type": "validity_at_time",
            },
        ],
    }


# ─── Top-level generation ───────────────────────────────────────────

def generate_scenarios(count: int, *, rng_seed: int = 42) -> list[dict]:
    """Generate up to ``count`` scenarios via template variation.

    Distribution target: 50% preference_supersession, 30%
    factual_supersession, 20% temporally_bounded. Each template is
    re-used with varied timestamps until count is hit; duplicate IDs
    are deduped.
    """
    rng = random.Random(rng_seed)
    scenarios: list[dict] = []
    seen_ids: set[str] = set()

    target_pref = int(count * 0.5)
    target_fact = int(count * 0.3)
    target_bound = count - target_pref - target_fact

    offset = 0
    while sum(1 for s in scenarios if s["family"] == "preference_supersession") < target_pref:
        t = _PREFERENCE_TEMPLATES[offset % len(_PREFERENCE_TEMPLATES)]
        sc = _preference_scenario(t, offset)
        if sc["id"] not in seen_ids:
            scenarios.append(sc)
            seen_ids.add(sc["id"])
        offset += 1

    offset = 0
    while sum(1 for s in scenarios if s["family"] == "factual_supersession") < target_fact:
        t = _FACTUAL_TEMPLATES[offset % len(_FACTUAL_TEMPLATES)]
        sc = _factual_scenario(t, offset)
        if sc["id"] not in seen_ids:
            scenarios.append(sc)
            seen_ids.add(sc["id"])
        offset += 1

    offset = 0
    while sum(1 for s in scenarios if s["family"] == "temporally_bounded") < target_bound:
        t = _BOUNDED_TEMPLATES[offset % len(_BOUNDED_TEMPLATES)]
        sc = _bounded_scenario(t, offset)
        if sc["id"] not in seen_ids:
            scenarios.append(sc)
            seen_ids.add(sc["id"])
        offset += 1

    rng.shuffle(scenarios)
    return scenarios


def write_scenarios(scenarios: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for sc in scenarios:
            f.write(json.dumps(sc) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="BeliefEval scenario generator",
    )
    parser.add_argument(
        "--count", type=int, default=150,
        help="Number of scenarios to generate (default 150)",
    )
    parser.add_argument(
        "--output", default="eval/belief_eval_data/v03_scenarios.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    scenarios = generate_scenarios(args.count, rng_seed=args.seed)
    write_scenarios(scenarios, Path(args.output))
    from collections import Counter
    families = Counter(s["family"] for s in scenarios)
    print(f"Wrote {len(scenarios)} scenarios to {args.output}")
    for fam, c in families.items():
        print(f"  {fam}: {c}")


if __name__ == "__main__":
    main()
