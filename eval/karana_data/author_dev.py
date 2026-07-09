"""One-shot authoring script for the KaranaEval dev gold set (provenance).

25 hand-labeled dense-conversation cases across the extraction failure
taxonomy (docs/roadmap.md §1). Serialized with invariants asserted at
write time. Not a generator — texts and labels are authored; the script
validates and serializes.

Gold rules (v1, encoded in eval/karana_eval.py and re-checked by
tests/test_karana_eval.py):
- GOLD tuples are numerical facts a user could later aggregate under
  gaṇita semantics: spend (money OUT), counts, distances, durations,
  ages. Stated exactly (no inference beyond the sentence).
- MONEY IN is forbidden as a spend tuple (refunds, sales, discounts) —
  mirrors gaṇita's _NEGATIVE veto contract at aggregation.
- RANGES, HYPOTHETICALS, COLLOQUIAL quantities ("a couple hundred") are
  forbidden: no exact value is asserted, so extracting one fabricates
  precision.
- NUMERIC DISTRACTORS (clock times, temperatures, versions) are
  forbidden as money.
- `acceptable` lists every entity token that legitimately identifies
  the fact (part→whole aliases included, e.g. chain→bike); matching is
  canonical via ganita's _canonicalize_entity.
- Time is NOT scored in rubric v1 (extractors don't label it yet);
  gold `time` fields are recorded where present for a future v2.

Dev-only; no held-out split yet.
"""

from __future__ import annotations

import json
from pathlib import Path

# (id, family, text, golds, forbidden)
# golds: (entity, acceptable[], value, unit)
# forbidden: (value, reason)
CASES = [
    ("ka-ms-01", "multi_amount",
     "ended up spending $40 on the pump and another $85 on the saddle bag, both for the commuter bike",
     [("pump", ["pump", "bike", "cycling"], 40.0, "USD"),
      ("saddle bag", ["saddle bag", "saddle", "bag", "bike", "cycling"], 85.0, "USD")],
     []),
    ("ka-ms-02", "multi_amount",
     "the vet visit was $120, the meds came to $35.50, and parking ate another $8",
     [("vet visit", ["vet visit", "vet", "cat", "pet"], 120.0, "USD"),
      ("meds", ["meds", "medication", "vet", "pet"], 35.5, "USD"),
      ("parking", ["parking"], 8.0, "USD")],
     []),
    ("ka-far-01", "amount_far",
     "the quote for repainting the hallway — after all the back and forth with the contractor about scheduling, scope, and the mess from last time — landed at $650",
     [("repainting", ["repainting", "painting", "hallway", "renovation"], 650.0, "USD")],
     []),
    ("ka-far-02", "amount_far",
     "my share, once Priya totted everything up from the cabin weekend and sent round the breakdown, came to $86",
     [("share", ["share", "cabin", "weekend", "trip"], 86.0, "USD")],
     []),
    ("ka-rng-01", "range_forbidden",
     "the new laptop will probably run me somewhere between $1,200 and $1,500",
     [],
     [(1200.0, "range endpoint"), (1500.0, "range endpoint")]),
    ("ka-rng-02", "range_forbidden",
     "rent in that neighbourhood goes for $1,800 to $2,200 a month",
     [],
     [(1800.0, "range endpoint"), (2200.0, "range endpoint")]),
    ("ka-hyp-01", "hypothetical_forbidden",
     "if I took the Berlin job I'd be looking at about $95,000",
     [],
     [(95000.0, "hypothetical")]),
    ("ka-hyp-02", "hypothetical_forbidden",
     "imagine dropping $300 on a single dinner — not happening",
     [],
     [(300.0, "hypothetical")]),
    ("ka-in-01", "money_in_forbidden",
     "the airline finally refunded the $240 for the cancelled flight",
     [],
     [(240.0, "refund is money in, not spend")]),
    ("ka-in-02", "money_in_forbidden",
     "turns out I was overcharged; they knocked $60 off the bill",
     [],
     [(60.0, "discount is money in, not spend")]),
    ("ka-cur-01", "currency",
     "paid 220 euros for the hotel in Berlin",
     [("hotel", ["hotel", "berlin", "trip"], 220.0, "EUR")],
     []),
    ("ka-cur-02", "currency",
     "the workshop fee was £150, plus $45 for materials",
     [("workshop", ["workshop", "fee", "course"], 150.0, "GBP"),
      ("materials", ["materials", "workshop"], 45.0, "USD")],
     []),
    ("ka-col-01", "colloquial_forbidden",
     "the brakes are going to cost a couple hundred to sort out",
     [],
     [(200.0, "colloquial, no exact value asserted"),
      (2.0, "'couple' is not a count")]),
    ("ka-col-02", "colloquial_forbidden",
     "we must have spent a few thousand on the garden over the years",
     [],
     [(1000.0, "colloquial"), (3000.0, "colloquial"),
      (3.0, "'few' is not a count")]),
    ("ka-imp-01", "implicit_count",
     "picked up three more succulents for the office shelf",
     [("succulents", ["succulents", "succulent", "plant", "plants"], 3.0, "item")],
     []),
    ("ka-imp-02", "implicit_count",
     "adopted two kittens from the shelter on Saturday",
     [("kittens", ["kittens", "kitten", "cat", "cats", "pet"], 2.0, "item")],
     []),
    ("ka-date-01", "dated_amount",
     "on March 3rd I put $500 into the emergency fund",
     [("emergency fund", ["emergency fund", "fund", "savings"], 500.0, "USD")],
     []),
    ("ka-date-02", "dated_amount",
     "last Tuesday's grocery run came to $92.40",
     [("grocery run", ["grocery run", "grocery", "groceries", "food"], 92.4, "USD")],
     []),
    ("ka-den-01", "dense_paragraph",
     "big month: gym renewal $89, one physio session $70, and new running shoes $130",
     [("gym renewal", ["gym renewal", "gym"], 89.0, "USD"),
      ("physio session", ["physio session", "physio"], 70.0, "USD"),
      ("running shoes", ["running shoes", "shoes", "running"], 130.0, "USD")],
     []),
    ("ka-den-02", "dense_paragraph",
     "trip damage: flights $410, the airbnb $380, and the cat sitter charged $150 total",
     [("flights", ["flights", "flight", "trip", "travel"], 410.0, "USD"),
      ("airbnb", ["airbnb", "accommodation", "trip"], 380.0, "USD"),
      ("cat sitter", ["cat sitter", "sitter", "cat"], 150.0, "USD")],
     []),
    ("ka-dis-01", "numeric_distractor",
     "meeting moved to 3pm; the build takes 45 minutes now and we're on version 2.7",
     [],
     [(3.0, "clock time as money"), (45.0, "process duration as money"),
      (2.7, "version number as money")]),
    ("ka-dis-02", "numeric_distractor",
     "it was 34 degrees out; ran 8k anyway",
     [("ran", ["ran", "run", "running"], 8.0, "km")],
     [(34.0, "temperature as money")]),
    ("ka-al-01", "part_alias",
     "replaced the chain for $28 — the bike deserves it",
     [("chain", ["chain", "bike", "cycling"], 28.0, "USD")],
     []),
    ("ka-age-01", "age_series",
     "my three nephews are 4, 7, and 12",
     [("nephews", ["nephews", "nephew"], 4.0, "years"),
      ("nephews", ["nephews", "nephew"], 7.0, "years"),
      ("nephews", ["nephews", "nephew"], 12.0, "years")],
     [(3.0, "'three nephews' is context, not an aggregable count fact "
            "distinct from the ages")]),
    ("ka-ord-01", "ordinal_count",
     "finished my ninth book of the year last night",
     [("book", ["book", "books", "reading"], 9.0, "item")],
     []),
]


def build() -> list[dict]:
    out = []
    seen_ids: set[str] = set()
    for cid, family, text, golds, forbidden in CASES:
        assert cid not in seen_ids, f"duplicate id {cid}"
        seen_ids.add(cid)
        gold_values = {(g[2], g[3]) for g in golds}
        for v, _reason in forbidden:
            # a forbidden value may share a number with a gold only if
            # the unit differs; same (value) with no gold at all is the
            # common case
            assert all(v != gv for gv, _u in gold_values), (
                f"{cid}: forbidden value {v} collides with a gold value"
            )
        for entity, acceptable, value, unit in golds:
            assert entity in acceptable, (cid, entity)
            assert value > 0, (cid, value)
            assert unit in ("USD", "EUR", "GBP", "item", "km", "hours",
                            "years"), (cid, unit)
        out.append({
            "id": cid,
            "family": family,
            "text": text,
            "gold_tuples": [
                {"entity": e, "acceptable": a, "value": v, "unit": u}
                for e, a, v, u in golds
            ],
            "forbidden_tuples": [
                {"value": v, "reason": r} for v, r in forbidden
            ],
        })
    return out


if __name__ == "__main__":
    cases = build()
    path = Path(__file__).parent / "gold_cases.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in cases))
    fams: dict[str, int] = {}
    for c in cases:
        fams[c["family"]] = fams.get(c["family"], 0) + 1
    n_gold = sum(len(c["gold_tuples"]) for c in cases)
    n_forb = sum(len(c["forbidden_tuples"]) for c in cases)
    print(f"wrote {len(cases)} cases ({n_gold} gold tuples, "
          f"{n_forb} forbidden) to {path}")
    print(f"families: {fams}")
