"""One-shot authoring script for the AbsenceEval DEV set (kept for provenance).

Hand-written scenarios, transcribed here so the integrity invariants are
asserted at write time. This is NOT a template generator — the texts are
authored, the script only serializes and validates. Regeneration must be
byte-identical (tests/test_absence_eval.py pins it).

DEV ONLY: tuning against this file is allowed. No held-out split exists
yet; a sealed batch gets authored AFTER the absence recall path freezes
(working-protocol rule 5).

Gold-contrast rule (what expected_contrast_ids means):
- absent verdicts — the present beliefs constitutive of the absence
  claim: pradhvaṃsābhāva cites the prior positive + its cessation;
  prāgabhāva cites the current progress/intent beliefs; anyonyābhāva
  cites the true-identity beliefs; atyantābhāva cites the
  nearest-domain present beliefs ("what you HAVE tried/done").
- present verdicts (traps) — the minimal evidence set proving presence.
- routing controls — null (no absence answer exists to cite).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from eval.absence_eval import ROUTES, TAXONOMY, canonicalize_locus

_SCOPE_FOR_KIND = {
    "atyantabhava": "ever",
    "pragabhava": "yet",
    "pradhvamsabhava": "still",
    "anyonyabhava": "identity",
}

# (id, family, props [(text, iso)], q, scope, gold)
# gold = (expected_route, expected_kind, expected_verdict,
#         expected_locus, expected_contrast_ids)
# sessions auto-assigned s1..sN; dates strictly increasing.
SCENARIOS = [
    # ─── absent_atyantabhava: "have I ever…" and the answer is a
    # qualified never — contrast is what the user HAS done ───────────
    ("ab-aty-001", "absent_atyantabhava",
     [("spent two weeks in South Korea last spring — Seoul and Busan", "2025-03-30T18:00:00"),
      ("the landlord is repainting the stairwell next week", "2025-04-14T09:00:00"),
      ("booked flights to Taipei for the October food trip", "2025-08-02T20:00:00"),
      ("finally typed up my travel journal from the Busan trip", "2025-09-10T21:00:00")],
     "have I ever been to Japan?", "ever",
     ("absence", "atyantabhava", "absent", "japan", [0, 2])),
    ("ab-aty-002", "absent_atyantabhava",
     [("the physio gave me a daily stretching routine for the lower back pain", "2025-02-04T10:00:00"),
      ("started a standing-desk trial at work for my back", "2025-03-11T09:30:00"),
      ("the pharmacist recommended ibuprofen gel for the back flare-ups", "2025-05-20T17:00:00"),
      ("Priya's birthday picnic is on the 14th", "2025-06-01T12:00:00")],
     "have I ever tried acupuncture for my back pain?", "ever",
     ("absence", "atyantabhava", "absent", "acupuncture", [0, 1, 2])),
    ("ab-aty-003", "absent_atyantabhava",
     [("renewed my regular driving licence online", "2025-01-22T08:30:00"),
      ("booked a motorcycle CBT taster session for late September", "2025-06-30T19:00:00"),
      ("the dentist moved my cleaning to Friday", "2025-07-08T11:00:00"),
      ("cycled the coast path with Ana on Saturday", "2025-07-19T16:00:00")],
     "have I ever ridden a motorcycle?", "ever",
     ("absence", "atyantabhava", "absent", "motorcycle", [1])),

    # ─── absent_pragabhava: "…yet?" — not-yet, with the search/intent
    # state as contrast ("you HAVE decided the budget cap") ──────────
    ("ab-prag-001", "absent_pragabhava",
     [("we started the wedding venue search — longlist of nine places", "2025-01-12T10:00:00"),
      ("shortlisted three wedding venues: the barn, the glasshouse, the town hall", "2025-02-09T11:00:00"),
      ("the barn quoted 6k for a Friday in May", "2025-03-01T14:00:00"),
      ("my brother finally fixed his bike", "2025-03-15T13:00:00")],
     "have we decided on a wedding venue yet?", "yet",
     ("absence", "pragabhava", "absent", "wedding venue", [1, 2])),
    ("ab-prag-002", "absent_pragabhava",
     [("I want my first tattoo to be about the sea — that much is settled", "2025-02-18T20:00:00"),
      ("collected a folder of wave and compass tattoo references", "2025-04-02T21:30:00"),
      ("booked a consult with the tattoo artist Mara for August", "2025-06-21T15:00:00"),
      ("the gym changed its opening hours again", "2025-06-28T08:00:00")],
     "have I decided on a tattoo design yet?", "yet",
     ("absence", "pragabhava", "absent", "tattoo design", [0, 1, 2])),
    ("ab-prag-003", "absent_pragabhava",
     [("the Lisbon relocation is confirmed for next spring", "2025-05-06T09:00:00"),
      ("gathered the document checklist for the Portugal visa application", "2025-06-14T10:00:00"),
      ("my employer's HR sent the employment letter for the visa file", "2025-07-03T16:00:00"),
      ("found a great taco place near the office", "2025-07-11T13:00:00")],
     "have I submitted the visa application yet?", "yet",
     ("absence", "pragabhava", "absent", "visa application", [1, 2])),

    # ─── absent_pradhvamsabhava: "do I still…" — no-longer; contrast
    # is the destroyed prior positive + its cessation ────────────────
    ("ab-pradh-001", "absent_pradhvamsabhava",
     [("enrolled in the Tuesday evening Italian class at the language centre", "2025-01-15T18:00:00"),
      ("the building's water will be off on Thursday morning", "2025-03-04T08:00:00"),
      ("I stopped going to the Italian class when the project crunch started", "2025-05-27T19:00:00"),
      ("the balcony tomatoes are ripening early this year", "2025-06-10T18:30:00")],
     "do I still go to the Italian class?", "still",
     ("absence", "pradhvamsabhava", "absent", "italian class", [0, 2])),
    ("ab-pradh-002", "absent_pradhvamsabhava",
     [("rented a storage unit on Mill Road for the leftover furniture", "2025-02-01T10:00:00"),
      ("Talia's housewarming moved to the last weekend of March", "2025-03-08T12:00:00"),
      ("cleared out and cancelled the storage unit — sold the last of the furniture", "2025-08-09T15:00:00"),
      ("the neighbours' renovation starts Monday", "2025-08-15T09:00:00")],
     "do I still have the storage unit?", "still",
     ("absence", "pradhvamsabhava", "absent", "storage unit", [0, 2])),
    ("ab-pradh-003", "absent_pradhvamsabhava",
     [("signed up for the weekly veg box delivery from the farm co-op", "2025-01-20T09:00:00"),
      ("cancelled the veg box for good — we just don't cook enough to keep up", "2025-04-18T18:00:00"),
      ("Talia lent me her pasta maker", "2025-05-02T19:00:00")],
     "do we still get the veg box?", "still",
     ("absence", "pradhvamsabhava", "absent", "veg box", [0, 1])),

    # ─── absent_anyonyabhava: "is A a B?" — category/identity
    # negation; contrast is the true identity ────────────────────────
    ("ab-anyo-001", "absent_anyonyabhava",
     [("Maya is my physiotherapist — sessions every other Friday", "2025-01-10T09:00:00"),
      ("my accountant is Priya from Ledger & Lane", "2025-02-14T11:00:00"),
      ("the car needs new brake pads before winter", "2025-10-03T17:00:00")],
     "is Maya my accountant?", "identity",
     ("absence", "anyonyabhava", "absent", "maya", [0, 1])),
    ("ab-anyo-002", "absent_anyonyabhava",
     [("we renewed the lease on the Elm Street flat for two more years", "2025-03-01T10:00:00"),
      ("started a house-deposit savings pot this month", "2025-03-20T09:00:00"),
      ("the ficus finally recovered after repotting", "2025-04-04T18:00:00")],
     "am I a homeowner?", "identity",
     ("absence", "anyonyabhava", "absent", "homeowner", [0, 1])),
    ("ab-anyo-003", "absent_anyonyabhava",
     [("Dario was announced as team lead for the platform group — I'm staying senior engineer", "2025-02-06T10:00:00"),
      ("I run the Tuesday architecture guild session, which I enjoy", "2025-02-20T14:00:00"),
      ("the lunch trucks moved to the north lot", "2025-03-05T12:30:00")],
     "am I the team lead of the platform group?", "identity",
     ("absence", "anyonyabhava", "absent", "team lead", [0, 1])),

    # ─── trap_present: the question SOUNDS like absence but a positive
    # belief EXISTS — the system must find it, not claim absence.
    # A wrong "you never did" here is the catastrophic failure. ──────
    ("ab-trap-ever-001", "trap_present",
     [("finished my first sprint triathlon in Cascais — 1:41 and grinning", "2025-06-08T14:00:00"),
      ("the library extended its weekend hours", "2025-06-15T10:00:00"),
      ("signed up for an olympic-distance triathlon next season", "2025-09-02T19:00:00")],
     "have I ever done a triathlon?", "ever",
     ("absence", None, "present", "triathlon", [0])),
    ("ab-trap-ever-002", "trap_present",
     [("I lived in Montreal for two years during the exchange program", "2025-01-30T20:00:00"),
      ("picked up my new glasses on Saturday", "2025-02-12T11:00:00"),
      ("been wondering lately whether I'd ever move abroad again", "2025-05-22T22:00:00")],
     "have I ever lived abroad?", "ever",
     ("absence", None, "present", "abroad", [0])),
    ("ab-trap-yet-001", "trap_present",
     [("we're bringing the puppy home in three weeks and can't agree on a name", "2025-04-01T19:00:00"),
      ("we settled on the name Miso for the puppy — it just fits", "2025-04-19T20:00:00"),
      ("the balcony door lock jammed again", "2025-04-25T08:00:00")],
     "have we picked a name for the puppy yet?", "yet",
     ("absence", None, "present", "puppy name", [1])),
    ("ab-trap-yet-002", "trap_present",
     [("comparing three quotes for the kitchen renovation", "2025-02-10T09:00:00"),
      ("we signed with Fenwick Builders for the kitchen renovation — starting in June", "2025-03-27T17:00:00"),
      ("borrowed a wallpaper steamer from next door", "2025-04-08T10:00:00")],
     "have we chosen a builder for the kitchen yet?", "yet",
     ("absence", None, "present", "builder", [1])),
    ("ab-trap-still-001", "trap_present",
     [("started violin lessons with an online tutor in January", "2025-01-19T17:00:00"),
      ("moved my violin lesson to Thursday evenings this term", "2025-09-08T18:00:00"),
      ("the recycling schedule changed to alternate weeks", "2025-09-15T08:00:00")],
     "do I still take violin lessons?", "still",
     ("absence", None, "present", "violin lesson", [1])),
    ("ab-trap-still-002", "trap_present",
     [("renewed my season ticket for the basketball club for the 25/26 season", "2025-08-30T12:00:00"),
      ("the elevator inspection is next Wednesday", "2025-09-04T09:00:00"),
      ("went to the basketball season opener with Leo", "2025-10-12T19:30:00")],
     "do I still have the basketball season ticket?", "still",
     ("absence", None, "present", "season ticket", [0])),
    ("ab-trap-id-001", "trap_present",
     [("the film society's winter program looks great this year", "2025-11-20T21:00:00"),
      ("renewed my film society membership for 2026", "2025-12-02T20:00:00"),
      ("the corner bakery now opens on Sundays", "2025-12-09T10:00:00")],
     "am I a member of the film society?", "identity",
     ("absence", None, "present", "film society", [1])),
    ("ab-trap-id-002", "trap_present",
     [("passed the workplace first aid certification — valid for three years", "2025-05-16T15:00:00"),
      ("our floor's fire drill is moving to Fridays", "2025-05-23T10:00:00"),
      ("restocked the office first aid kit", "2025-07-01T14:00:00")],
     "am I a certified first aider?", "identity",
     ("absence", None, "present", "first aider", [0])),

    # ─── routing_control: absence-ish surface, but the true route is
    # retrieval or synthesis — the absence gate must NOT steal these ─
    ("ab-ctrl-ret-001", "routing_control",
     [("my physiotherapist is Dana Petrova at the Riverside clinic", "2025-01-09T09:00:00"),
      ("physio homework: bridges and dead bugs, ten minutes nightly", "2025-01-16T09:30:00"),
      ("the blender finally died", "2025-02-02T18:00:00")],
     "I can never remember — what's my physiotherapist's name?", None,
     ("retrieval", None, None, None, None)),
    ("ab-ctrl-ret-002", "routing_control",
     [("I don't like cilantro — it tastes like soap to me", "2025-02-07T19:00:00"),
      ("I can't stand blue cheese", "2025-03-12T20:00:00"),
      ("made a great lentil soup on Sunday", "2025-03-16T14:00:00")],
     "which foods did I say I don't like?", None,
     ("retrieval", None, None, None, None)),
    ("ab-ctrl-syn-001", "routing_control",
     [("spent 40 at the farmers market on the 3rd", "2025-05-03T11:00:00"),
      ("spent 65 at the fish counter for Ana's dinner", "2025-05-17T12:00:00"),
      ("spent 120 on the birthday dinner out", "2025-05-24T21:00:00")],
     "have I ever spent more than 100 in one go on food?", None,
     ("synthesis", None, None, None, None)),
    ("ab-ctrl-syn-002", "routing_control",
     [("the car loan started at 12000", "2025-01-05T10:00:00"),
      ("I've paid off 4500 of the car loan so far", "2025-06-30T09:00:00"),
      ("the parking permit renews in November", "2025-07-07T08:00:00")],
     "how much of the car loan haven't I paid off yet?", None,
     ("synthesis", None, None, None, None)),
]


def build() -> list[dict]:
    scenarios = []
    for sid, family, props, q, scope, gold in SCENARIOS:
        route, kind, verdict, locus, contrast = gold
        scenarios.append({
            "id": sid,
            "family": family,
            "split": "dev",
            "propositions": [
                {"text": text, "asserted_at": iso, "session": f"s{i + 1}"}
                for i, (text, iso) in enumerate(props)
            ],
            "questions": [{
                "q": q,
                "type": "absence" if route == "absence" else "control",
                "scope": scope,
                "gold": {
                    "expected_route": route,
                    "expected_kind": kind,
                    "expected_verdict": verdict,
                    "expected_locus": locus,
                    "expected_contrast_ids": contrast,
                },
            }],
        })
    return scenarios


def validate(scenarios: list[dict]) -> None:
    ids = [s["id"] for s in scenarios]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for s in scenarios:
        assert s["split"] == "dev"
        dates = [p["asserted_at"] for p in s["propositions"]]
        parsed = [datetime.fromisoformat(d) for d in dates]
        assert parsed == sorted(parsed) and len(set(parsed)) == len(parsed), (
            f"{s['id']}: dates must be strictly increasing"
        )
        n = len(s["propositions"])
        for q in s["questions"]:
            g = q["gold"]
            assert g["expected_route"] in ROUTES, s["id"]
            if g["expected_route"] != "absence":
                assert q["type"] == "control" and q["scope"] is None, s["id"]
                assert g["expected_kind"] is None, s["id"]
                assert g["expected_verdict"] is None, s["id"]
                assert g["expected_locus"] is None, s["id"]
                assert g["expected_contrast_ids"] is None, s["id"]
                continue
            assert g["expected_verdict"] in ("absent", "present"), s["id"]
            if g["expected_verdict"] == "absent":
                assert g["expected_kind"] in TAXONOMY, s["id"]
                assert q["scope"] == _SCOPE_FOR_KIND[g["expected_kind"]], s["id"]
            else:
                # traps carry no kind — we do not assert what kind a
                # "present" answer has
                assert g["expected_kind"] is None, s["id"]
                assert q["scope"] in _SCOPE_FOR_KIND.values(), s["id"]
            locus = g["expected_locus"]
            assert locus and canonicalize_locus(locus) == locus, (
                f"{s['id']}: locus must be a canonical fixed point"
            )
            contrast = g["expected_contrast_ids"]
            assert contrast, f"{s['id']}: absence gold needs contrast ids"
            assert all(0 <= i < n for i in contrast), s["id"]
            assert len(set(contrast)) == len(contrast), s["id"]


def main() -> None:
    p = argparse.ArgumentParser(description="Serialize the AbsenceEval dev set")
    p.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "dev_scenarios.jsonl",
    )
    args = p.parse_args()
    scenarios = build()
    validate(scenarios)
    with open(args.output, "w") as f:
        for s in scenarios:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(scenarios)} scenarios to {args.output}")


if __name__ == "__main__":
    main()
