"""One-shot authoring script for the AnalogyEval dev set (provenance).

Hand-written scenarios, serialized with the instrument's own invariants
asserted at write time (imported from eval.analogy_eval so the data
rules and the scorer can never drift apart):

- DISJOINTNESS: an analogy question shares ≤ 2 content tokens with the
  full text of each gold analogue session — structure is shared,
  vocabulary is not; that is what makes it analogy rather than
  retrieval.
- TRAP DOMINANCE: in surface_trap scenarios the trap session shares
  ≥ 3 content tokens with the question and strictly more than the gold
  does — the lexically-seductive answer must be the wrong one.
- Session references valid; dates ascend within a scenario; negatives
  carry no gold analogues.

Not a generator — texts are authored; the script only validates and
serializes. Dev-only (no held-out split yet).
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.analogy_eval import content_tokens

# (id, family, props, questions)
# props: (session, text, iso_date) in ascending date order.
# questions: dict per runner schema.
SCENARIOS = [
    # ─── core_analogy: structure shared, vocabulary disjoint ────────
    ("an-core-01", "core_analogy",
     [("s-joboffer", "the Meridian offer came with a 48-hour expiry and my stomach dropped", "2024-03-11T09:00:00"),
      ("s-joboffer", "kept flip-flopping all night about the offer; wrote two resignation drafts and binned both", "2024-03-11T23:30:00"),
      ("s-joboffer", "phoned my old mentor Priya at midnight; she asked one question about mornings", "2024-03-12T00:10:00"),
      ("s-joboffer", "accepted the Meridian offer an hour before the expiry and felt instantly calm", "2024-03-12T16:00:00"),
      ("s-kitchen", "picked the quartz counters for the kitchen after months of browsing samples", "2024-06-02T10:00:00"),
      ("s-kitchen", "the kitchen contractor starts the first week of May", "2024-06-14T09:00:00"),
      ("s-noise", "the library extended weekend opening hours", "2024-07-01T12:00:00")],
     [{"q": "the landlord gave us until Friday to sign the lease renewal and I keep going back and forth — have I been in a situation like this before?",
       "type": "analogy", "expected_route": "analogy",
       "gold_analogue_sessions": ["s-joboffer"],
       "shared_structure": ["hard time limit forcing a choice",
                            "prolonged indecision",
                            "advice from a trusted person",
                            "calm after committing"]}]),
    ("an-core-02", "core_analogy",
     [("s-dana", "realized I'd driven Dana to the airport four times this year and she's never once offered petrol money", "2024-02-08T18:00:00"),
      ("s-dana", "the resentment about Dana crept up so slowly I didn't notice until I dreaded her name on my phone", "2024-04-19T21:00:00"),
      ("s-dana", "finally told Dana plainly what I could and couldn't do; shaky voice, held firm", "2024-05-30T19:30:00"),
      ("s-dana", "Dana and I are lighter with each other since; the friendship survived the honesty", "2024-08-12T14:00:00"),
      ("s-marathon", "long run Sundays are sacred now; eighteen kilometres this morning", "2024-09-01T08:00:00"),
      ("s-marathon", "carb-loading strategy sorted for race week", "2024-09-20T13:00:00"),
      ("s-noise", "the corner bakery changed hands again", "2024-10-01T09:00:00")],
     [{"q": "my coworker keeps offloading his tickets onto my queue and I'm rehearsing how to raise it without blowing up — does this remind me of anything?",
       "type": "analogy", "expected_route": "analogy",
       "gold_analogue_sessions": ["s-dana"],
       "shared_structure": ["one-sided arrangement",
                            "resentment building slowly",
                            "direct boundary conversation",
                            "relationship surviving honesty"]}]),
    ("an-core-03", "core_analogy",
     [("s-recital", "practiced the nocturne four hours a day for the recital; my teacher says it's over-polished", "2024-01-15T17:00:00"),
      ("s-recital", "froze eight bars into the recital piece; the room went white", "2024-02-03T19:00:00"),
      ("s-recital", "restarted the nocturne at half tempo, dropped the ornaments, and the music came back", "2024-02-03T19:05:00"),
      ("s-recital", "my teacher's note afterwards: the simple version moved people; the polish was for me", "2024-02-10T16:00:00"),
      ("s-tax", "the accountant wants every receipt from the studio conversion", "2024-03-05T10:00:00"),
      ("s-tax", "filed the extension paperwork for the studio's depreciation claim", "2024-03-22T11:00:00"),
      ("s-noise", "new neighbours moved into the blue house", "2024-04-02T15:00:00")],
     [{"q": "I over-rehearsed the investor demo, blanked on slide two, and I'm dreading the next pitch — have I been somewhere like this?",
       "type": "analogy", "expected_route": "analogy",
       "gold_analogue_sessions": ["s-recital"],
       "shared_structure": ["over-preparation",
                            "freezing under observation",
                            "recovering by simplifying",
                            "audiences prefer the simple version"]}]),
    ("an-core-04", "core_analogy",
     [("s-camper", "year two of the camper conversion and the electrics still aren't in; I dread the garage", "2024-05-04T11:00:00"),
      ("s-camper", "admitted the camper dream belonged to a version of me from 2022", "2024-06-15T20:00:00"),
      ("s-camper", "sold the camper shell and the solar kit to a couple who cried with excitement", "2024-07-06T13:00:00"),
      ("s-camper", "the empty garage feels like a held breath finally let out", "2024-07-07T09:00:00"),
      ("s-choir", "joined the Tuesday choir; we're butchering Brahms beautifully", "2024-08-13T19:00:00"),
      ("s-choir", "choir concert booked for December at St Anne's", "2024-09-24T18:00:00"),
      ("s-noise", "council resurfaced our street at last", "2024-10-11T08:00:00")],
     [{"q": "four drafts into the novel and I'm wondering if shelving the manuscript would be failure or freedom — anything in my past like this?",
       "type": "analogy", "expected_route": "analogy",
       "gold_analogue_sessions": ["s-camper"],
       "shared_structure": ["sunk cost in a long project",
                            "outgrowing an old ambition",
                            "walking away deliberately",
                            "relief outweighing loss"]}]),
    ("an-core-05", "core_analogy",
     [("s-herbs", "three basil pots on the fire escape — my entire agricultural empire", "2024-04-01T09:00:00"),
      ("s-herbs", "the basil thrived so the fire escape now hosts eight pots and a chilli experiment", "2024-05-20T18:00:00"),
      ("s-herbs", "signed for the community plot; the fire-escape trial earned it", "2024-07-08T10:00:00"),
      ("s-herbs", "first full bed planted at the plot; scaling what already worked, nothing speculative", "2024-08-02T17:00:00"),
      ("s-jury", "jury duty summons for the last week of September", "2024-08-20T12:00:00"),
      ("s-jury", "the trial wrapped early; civic duty complete", "2024-09-26T16:00:00"),
      ("s-noise", "my cousin's boat finally sold", "2024-10-05T14:00:00")],
     [{"q": "instead of committing to the shop lease I could trial a Saturday market stall for a month first — have I approached something this way before?",
       "type": "analogy", "expected_route": "analogy",
       "gold_analogue_sessions": ["s-herbs"],
       "shared_structure": ["small pilot before commitment",
                            "scaling only after the trial works",
                            "low-stakes proof first"]}]),
    ("an-core-06", "core_analogy",
     [("s-photo", "the photography account crossed ten thousand followers and I felt nothing", "2024-02-14T21:00:00"),
      ("s-photo", "caught myself shooting for the algorithm — golden hour, faces, whatever performs", "2024-03-30T19:00:00"),
      ("s-photo", "quit checking the follower graph; started selling prints of the harbour series instead", "2024-05-11T10:00:00"),
      ("s-photo", "a stranger emailed that the harbour print hangs over her desk; that one email beat the whole graph", "2024-06-21T09:00:00"),
      ("s-visa", "the visa paperwork for mum's visit needs three more documents", "2024-07-15T11:00:00"),
      ("s-visa", "mum's visa approved; she lands in October", "2024-08-30T15:00:00"),
      ("s-noise", "the gym replaced all the treadmills", "2024-09-10T07:00:00")],
     [{"q": "I keep chasing personal records at the gym and enjoying training less every week — where have I seen this pattern in my own life?",
       "type": "analogy", "expected_route": "analogy",
       "gold_analogue_sessions": ["s-photo"],
       "shared_structure": ["optimizing a proxy metric",
                            "joy draining from measurement",
                            "switching to what actually matters"]}]),

    # ─── surface_trap: vocabulary points at the WRONG session ───────
    ("an-trap-01", "surface_trap",
     [("s-guitar", "wrist strain from guitar practice; the physio says six weeks of restraint", "2024-03-02T10:00:00"),
      ("s-guitar", "cut guitar sessions to fifteen careful minutes with a timer; hated every restriction", "2024-03-10T18:00:00"),
      ("s-guitar", "the shortened guitar practice made me precise; the strain healed ahead of schedule", "2024-04-25T17:00:00"),
      ("s-rungear", "bought new running shoes for marathon training, the neon ones, plus a running vest", "2024-05-15T12:00:00"),
      ("s-rungear", "the running club discount got me a knee sleeve and a second pair of running shoes for training volume", "2024-06-01T13:00:00"),
      ("s-noise", "the office espresso machine died again", "2024-06-20T09:00:00")],
     [{"q": "my marathon training hit an injury — the knee flared up mid-run and I have to pull my running volume way back — when have I handled something like this?",
       "type": "analogy", "expected_route": "analogy",
       "gold_analogue_sessions": ["s-guitar"],
       "surface_trap_session": "s-rungear",
       "shared_structure": ["physical setback interrupting practice",
                            "forced reduction of volume",
                            "constraint improving quality"]}]),
    ("an-trap-02", "surface_trap",
     [("s-speech", "best-man speech drafted to the minute: eleven pages, colour-coded cue cards", "2024-04-05T20:00:00"),
      ("s-speech", "binned the eleven pages; kept one card with three beats and a joke", "2024-05-18T22:00:00"),
      ("s-speech", "the speech landed — laughter in the right places precisely because I could breathe and improvise", "2024-06-01T21:00:00"),
      ("s-wedtravel", "booked flights to Lisbon for my sister's wedding and planned the whole trip itinerary; aisle seats both ways", "2024-04-20T11:00:00"),
      ("s-wedtravel", "the wedding hotel is walkable from the venue; trip logistics fully planned and sorted", "2024-05-02T15:00:00"),
      ("s-noise", "renewed my passport two years early", "2024-05-25T10:00:00")],
     [{"q": "I've over-planned this Berlin trip to a fifteen-tab itinerary and it's stressing me out — I want to scrap it for a loose plan — have I learned this lesson somewhere before?",
       "type": "analogy", "expected_route": "analogy",
       "gold_analogue_sessions": ["s-speech"],
       "surface_trap_session": "s-wedtravel",
       "shared_structure": ["overplanning paralysis",
                            "throwing away the detailed plan",
                            "loose structure performing better"]}]),
    ("an-trap-03", "surface_trap",
     [("s-intern", "the new intern missed both deadlines his first fortnight; the team wanted him reassigned", "2024-02-12T14:00:00"),
      ("s-intern", "set up the same fifteen-minute check-in with the intern every morning, no judgement, same questions", "2024-02-19T09:00:00"),
      ("s-intern", "the intern shipped the reporting feature solo; the morning rhythm did what pressure couldn't", "2024-05-10T16:00:00"),
      ("s-dogsit", "dog-sitting Rufus this weekend; the dog's routine demands three walks a day", "2024-06-08T08:00:00"),
      ("s-dogsit", "returned Rufus the dog to the neighbours with a report on his walk routine and park behaviour", "2024-06-10T18:00:00"),
      ("s-noise", "the balcony door finally got fixed", "2024-06-30T12:00:00")],
     [{"q": "the rescue dog we adopted has behaviour problems and everyone says give him up, but I think a steady dog routine could turn him around — when have I bet on patience like this?",
       "type": "analogy", "expected_route": "analogy",
       "gold_analogue_sessions": ["s-intern"],
       "surface_trap_session": "s-dogsit",
       "shared_structure": ["rough start others gave up on",
                            "steady low-pressure routine",
                            "trust built through consistency",
                            "gradual turnaround"]}]),
    ("an-trap-04", "surface_trap",
     [("s-carrier", "the phone carrier billed the cancelled add-on a third time; started a dated log of every call", "2024-03-08T10:00:00"),
      ("s-carrier", "escalated to the carrier's complaints team with the full log attached; suddenly everyone was helpful", "2024-04-02T11:00:00"),
      ("s-carrier", "ported my number out anyway — the refund didn't buy back the trust", "2024-04-29T15:00:00"),
      ("s-savings", "opened the high-interest savings account for the house deposit at the new bank branch — no monthly account fee", "2024-05-12T10:00:00"),
      ("s-savings", "automated the monthly transfer into the bank savings account; deposit fund growing", "2024-06-01T09:00:00"),
      ("s-noise", "recycling day moved to Thursdays", "2024-06-15T08:00:00")],
     [{"q": "the bank keeps charging that account fee they promised to waive and support loops me endlessly — I'm ready to move banks — have I run this playbook before?",
       "type": "analogy", "expected_route": "analogy",
       "gold_analogue_sessions": ["s-carrier"],
       "surface_trap_session": "s-savings",
       "shared_structure": ["repeated service failure",
                            "documenting every interaction",
                            "escalation with evidence",
                            "leaving despite the fix"]}]),

    # ─── multi_candidate: rank the structurally-closest first ───────
    ("an-multi-01", "multi_candidate",
     [("s-band", "the band split the wedding-gig fee unevenly and nobody would say so out loud", "2024-01-20T22:00:00"),
      ("s-band", "our drummer quit the band over the money silence, not the money", "2024-02-14T19:00:00"),
      ("s-band", "post-mortem with the band: we agreed the unspoken split killed it, not the amounts", "2024-03-01T20:00:00"),
      ("s-holiday", "the group holiday kitty caused sulking until Ines built the shared spreadsheet", "2024-06-10T18:00:00"),
      ("s-holiday", "spreadsheet transparency fixed the holiday kitty mood overnight", "2024-06-12T09:00:00"),
      ("s-pottery", "wheel-throwing class booked for spring; solo hobby, my clay, my mess", "2024-07-01T17:00:00"),
      ("s-noise", "the ferry timetable changed for winter", "2024-08-01T08:00:00")],
     [{"q": "my friend and I started a little app side-business and the revenue conversation is getting awkward and avoided — what past experience maps onto this?",
       "type": "analogy", "expected_route": "analogy",
       "gold_analogue_sessions": ["s-band", "s-holiday"],
       "shared_structure": ["shared venture with a friend",
                            "unspoken money tension",
                            "avoidance doing the damage",
                            "transparency as the cure"]}]),
    ("an-multi-02", "multi_candidate",
     [("s-burnout", "back from the burnout leave with hard rules: no email past six, one project at a time", "2024-02-01T09:00:00"),
      ("s-burnout", "caught myself sneaking a second project three weeks in; the old pattern knocking", "2024-02-22T20:00:00"),
      ("s-burnout", "held the one-project rule for the full quarter; energy actually compounding now", "2024-05-06T10:00:00"),
      ("s-leash", "the trainer's plan for Milo is painfully gradual: one calm street corner at a time", "2024-06-15T08:00:00"),
      ("s-leash", "skipped ahead to the busy market with Milo once and paid for it with a week of regression", "2024-07-02T09:00:00"),
      ("s-hikes", "the ridge trail hike was all fog and no view; glorious anyway", "2024-08-10T14:00:00"),
      ("s-noise", "the streetlight outside finally got repaired", "2024-09-01T21:00:00")],
     [{"q": "post-surgery rehab says weeks of tiny exercises and I'm itching to rush the recovery — where have I met this temptation before?",
       "type": "analogy", "expected_route": "analogy",
       "gold_analogue_sessions": ["s-burnout", "s-leash"],
       "shared_structure": ["recovery demanding restraint",
                            "temptation to rush",
                            "regression after skipping ahead",
                            "slow compounding payoff"]}]),

    # ─── routing_negative: near-analogy phrasings that belong to
    # other pramāṇa (gold: do NOT claim the analogy route) ───────────
    ("an-neg-01", "routing_negative",
     [("s-bath", "spent $600 on the bathroom tiles", "2024-03-01T10:00:00"),
      ("s-bath", "spent $450 on the bathroom vanity", "2024-04-11T15:00:00"),
      ("s-bath", "paid the plumber $800 for the bathroom rough-in", "2024-05-20T09:00:00"),
      ("s-noise", "the ficus needs repotting", "2024-06-01T12:00:00")],
     [{"q": "how much have I spent on the bathroom in total?",
       "type": "synthesis", "expected_route": "ganita",
       "gold_analogue_sessions": [], "shared_structure": []}]),
    ("an-neg-02", "routing_negative",
     [("s-bath2", "the tiler quoted $1,200 for the bathroom walls and floor", "2024-03-05T11:00:00"),
      ("s-bath2", "the tiler can start the first week of June", "2024-04-15T14:00:00"),
      ("s-noise", "borrowed the wet saw from Theo", "2024-05-01T16:00:00")],
     [{"q": "what did I say about the tiler's quote?",
       "type": "retrieval", "expected_route": "retrieval",
       "gold_analogue_sessions": [], "shared_structure": []}]),
    ("an-neg-03", "routing_negative",
     [("s-budget", "the renovation budget felt limitless in January; famous last words", "2024-01-10T09:00:00"),
      ("s-budget", "halfway through and the renovation budget spreadsheet has a red tab now", "2024-04-20T19:00:00"),
      ("s-budget", "we finish the renovation under a revised budget I actually believe", "2024-08-30T18:00:00"),
      ("s-noise", "market on Saturday had the good peaches", "2024-09-05T10:00:00")],
     [{"q": "how has my thinking about the renovation budget evolved?",
       "type": "narrative", "expected_route": "narrative",
       "gold_analogue_sessions": [], "shared_structure": []}]),
    ("an-neg-04", "routing_negative",
     [("s-quotes", "first renovation quote came in and I laughed, then cried", "2024-02-01T13:00:00"),
      ("s-quotes", "third renovation quote finally within reach; shortlisted two builders", "2024-03-15T17:00:00"),
      ("s-noise", "the car passed inspection first try", "2024-04-01T09:00:00")],
     [{"q": "when did I first get quotes for the renovation?",
       "type": "narrative", "expected_route": "narrative",
       "gold_analogue_sessions": [], "shared_structure": []}]),
]


# Bland filler episodes injected into every analogy scenario (two extra
# sessions each) to lower the random-stub chance floor on hit@k: with
# ~3 candidate sessions a 2-draw stub hits gold at ~5/6, which is too
# high for the instrument to discriminate. Fillers are deliberately
# vocabulary-bland; the validator asserts each shares ≤1 content token
# with the question so they can never become accidental lexical
# attractors.
_FILLERS = [
    "the dentist confirmed the checkup for the ninth",
    "watered the ficus and rotated it toward the window",
    "the parcel locker code changed again",
    "picked up stamps and posted grandma's birthday card",
    "the stairwell lightbulb needed replacing",
    "printed the insurance forms for the glovebox",
    "the supermarket moved the oats aisle again",
    "charged the spare batteries for the smoke alarm",
    "booked the annual boiler service",
    "returned the borrowed ladder to Sam",
    "the pharmacy switched to earlier closing hours",
    "descaled the kettle at last",
    "swapped the winter duvet in",
    "the bins schedule shifted for the public holiday",
    "topped up the transit card online",
    "wiped down the balcony furniture for spring",
    "the wifi router got its firmware update",
    "bought a new umbrella after the old one inverted",
    "labelled the freezer containers properly",
    "the car park barcode scanner works again",
    "renewed the library card for two more years",
    "vacuumed behind the sofa, found two pens",
    "the office badge photo finally got retaken",
    "recycled the box mountain in the hallway",
]


def build() -> list[dict]:
    out = []
    for idx, (sid, family, props, questions) in enumerate(SCENARIOS):
        # serialize in chronological ingest order regardless of how the
        # authoring groups sessions above
        props = sorted(props, key=lambda p: p[2])
        if family != "routing_negative":
            from datetime import datetime, timedelta
            last = datetime.fromisoformat(props[-1][2])
            for j in range(2):
                text = _FILLERS[(2 * idx + j) % len(_FILLERS)]
                props.append((
                    f"s-fill{j + 1}",
                    text,
                    (last + timedelta(days=j + 1)).isoformat(),
                ))
        sessions = {p[0] for p in props}
        session_text = {
            s: " ".join(p[1] for p in props if p[0] == s) for s in sessions
        }
        for q in questions:
            qt = content_tokens(q["q"])
            for g in q["gold_analogue_sessions"]:
                assert g in sessions, (sid, g)
                overlap = qt & content_tokens(session_text[g])
                assert len(overlap) <= 2, (
                    f"{sid}: disjointness violated vs {g}: {sorted(overlap)}"
                )
            for fs in ("s-fill1", "s-fill2"):
                if fs in sessions:
                    f_overlap = qt & content_tokens(session_text[fs])
                    assert len(f_overlap) <= 1, (
                        f"{sid}: filler {fs} lexically attracts: "
                        f"{sorted(f_overlap)}"
                    )
            trap = q.get("surface_trap_session")
            if trap:
                assert trap in sessions and trap not in q["gold_analogue_sessions"], sid
                trap_overlap = qt & content_tokens(session_text[trap])
                gold_overlap = qt & content_tokens(
                    session_text[q["gold_analogue_sessions"][0]]
                )
                assert len(trap_overlap) >= 3, (
                    f"{sid}: trap not seductive enough: {sorted(trap_overlap)}"
                )
                assert len(trap_overlap) > len(gold_overlap), (
                    f"{sid}: trap must out-overlap gold "
                    f"({sorted(trap_overlap)} vs {sorted(gold_overlap)})"
                )
            if q["expected_route"] != "analogy":
                assert not q["gold_analogue_sessions"], sid
        out.append({
            "id": sid,
            "family": family,
            "propositions": [
                {"session": s, "text": t, "asserted_at": d}
                for s, t, d in props
            ],
            "questions": questions,
        })
    return out


if __name__ == "__main__":
    scenarios = build()
    path = Path(__file__).parent / "dev_scenarios.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in scenarios))
    fams: dict[str, int] = {}
    for s in scenarios:
        fams[s["family"]] = fams.get(s["family"], 0) + 1
    print(f"wrote {len(scenarios)} scenarios to {path}")
    print(f"families: {fams}")
