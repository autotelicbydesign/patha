"""One-shot authoring script for held-out batch 2 (kept for provenance).

Hand-written scenarios, transcribed here so the integrity invariants are
asserted at write time. This is NOT a generator like generate_scenarios.py
— the texts are authored, the script only serializes and validates.

Sealed protocol: committed before any run; run once per reported config
under rubric v2; results publish as-run.
"""

from __future__ import annotations

import json
from pathlib import Path

# (id, family, question, qtype, theme, props, gold, pairs)
# props: (text, iso_date) — index order, dates strictly increasing.
# distractors = indices not in gold. sessions auto-assigned s1..sN.
SCENARIOS = [
    # ─── progressive_revelation: refinement arcs, ZERO expected pairs
    # (the supersession-precision probes) ───────────────────────────
    ("ho2-pr-001", "progressive_revelation",
     "how has my interest in astronomy evolved?", "evolution", "astronomy",
     [("I keep stopping on the roof to stare at the sky longer than I mean to", "2025-01-14T21:30:00"),
      ("the building manager finally fixed the lobby intercom", "2025-02-02T09:00:00"),
      ("borrowed binoculars and found Jupiter's moons from the balcony — actual astronomy from a city roof", "2025-02-21T22:00:00"),
      ("joined the astronomy society's dark-sky trip; first time through a proper telescope", "2025-04-12T23:30:00"),
      ("my cousin's flight got moved to Sunday evening", "2025-05-03T17:00:00"),
      ("my own dobsonian lives by the door now; I sketch at the eyepiece instead of photographing", "2025-07-19T22:15:00"),
      ("gave the beginners' talk at the astronomy society — how to start with just binoculars", "2025-11-08T19:00:00")],
     [0, 2, 3, 5, 6], []),
    ("ho2-pr-002", "progressive_revelation",
     "trace my thinking on beekeeping", "evolution", "beekeeping",
     [("read one article about colony collapse and couldn't stop thinking about it", "2025-01-20T08:00:00"),
      ("the council changed the parking zones on our street", "2025-02-10T10:00:00"),
      ("signed up for the weekend beekeeping taster at the community farm", "2025-03-08T09:30:00"),
      ("the beekeeping association matched me with a mentor; suited up for my first hive inspection", "2025-04-26T11:00:00"),
      ("finally returned the drill I borrowed from Marco", "2025-05-30T18:00:00"),
      ("my own hive made it through winter — checked the frames this morning and the queen is laying", "2025-08-16T08:30:00"),
      ("harvested twelve jars and I'm mentoring the new beekeeping cohort this spring", "2025-12-13T12:00:00")],
     [0, 2, 3, 5, 6], []),
    ("ho2-pr-003", "progressive_revelation",
     "when did I first start calligraphy?", "origin", "calligraphy",
     [("I keep pausing over hand-lettered shop signs; something in the strokes", "2025-02-05T13:00:00"),
      ("the boiler service is booked for the ninth", "2025-02-26T08:00:00"),
      ("bought a proper calligraphy pen set and ruined thirty pages learning the italic hand", "2025-03-15T16:00:00"),
      ("calligraphy practice is a nightly thing now — one page of letterforms before bed", "2025-05-22T22:00:00"),
      ("Raj started a fantasy football league at work", "2025-08-04T12:30:00"),
      ("addressed all sixty envelopes for Priya's wedding in copperplate calligraphy", "2025-09-27T14:00:00"),
      ("took my first paid calligraphy commission — a poem for someone's anniversary", "2025-12-01T10:00:00")],
     [0, 2, 3, 5, 6], []),
    ("ho2-pr-004", "progressive_revelation",
     "how has my kayaking evolved?", "evolution", "kayaking",
     [("watched the dawn kayakers slide under the bridge and felt weirdly jealous", "2025-03-02T07:15:00"),
      ("mum's birthday dinner is at the Lebanese place this year", "2025-03-20T19:00:00"),
      ("rented a kayak on the estuary — arms are jelly and the grin won't leave", "2025-04-13T10:00:00"),
      ("bought a used touring kayak from the club noticeboard; named her Heron", "2025-06-01T15:00:00"),
      ("the office moved to hot-desking", "2025-07-07T09:00:00"),
      ("kayaking before work now on flat mornings; the city is a different animal from the water", "2025-08-23T06:30:00"),
      ("qualified as a kayaking trip leader and took six beginners around the point", "2025-11-15T13:00:00")],
     [0, 2, 3, 5, 6], []),
    ("ho2-pr-005", "progressive_revelation",
     "how has my salsa dancing evolved?", "evolution", "salsa",
     [("the couple at Elena's party made dancing look like a conversation; I couldn't look away", "2025-01-25T23:00:00"),
      ("the pharmacy moved my prescription to the branch on Hale Street", "2025-02-14T11:00:00"),
      ("went to my first salsa class and stepped on every beat except the right one", "2025-03-06T20:00:00"),
      ("salsa Tuesdays are non-negotiable now; my basic step finally has weight in it", "2025-05-13T21:00:00"),
      ("our team got a new project tracker at work", "2025-06-25T10:00:00"),
      ("danced a whole social at the salsa congress without counting once", "2025-09-06T23:30:00"),
      ("I'm demoing for the beginners' salsa class and DJing the first hour of the social", "2025-12-09T20:30:00")],
     [0, 2, 3, 5, 6], []),

    # ─── multi_factor_change: one pivot cascades; arrangement
    # phrasings in unseen domains (v9 RevisionPatternDetector's first
    # held-out test) ────────────────────────────────────────────────
    ("ho2-mf-001", "multi_factor_change",
     "how has my routine with the dog evolved?", "evolution", "dog",
     [("evenings are mine: gym at seven, dinner at nine, nobody waiting", "2025-02-03T19:00:00"),
      ("the quarterly numbers meeting moved to Thursdays", "2025-02-19T09:00:00"),
      ("we adopted Biscuit from the shelter on Saturday — a two-year-old scruffy terrier mix of a dog", "2025-03-22T14:00:00"),
      ("mornings are now the long dog walk before any screen goes on", "2025-04-10T07:00:00"),
      ("moved my gym slot to lunchtime because the dog can't wait till eight", "2025-05-06T12:30:00"),
      ("my sister's kitchen renovation finally got its permit", "2025-06-11T16:00:00"),
      ("weekends we do the woods loop with Biscuit now — Sunday hikes are the new default", "2025-07-20T10:00:00"),
      ("turned down the Berlin conference; two weeks in kennels isn't happening for this dog", "2025-10-04T11:00:00")],
     [0, 2, 3, 4, 6, 7], [[0, 3], [0, 4]]),
    ("ho2-mf-002", "multi_factor_change",
     "how has my thinking about freelancing evolved?", "evolution", "freelancing",
     [("the nine-to-five pays fine but every interesting project this year happened after six pm", "2025-01-16T18:30:00"),
      ("the flat upstairs sold in a week", "2025-02-08T13:00:00"),
      ("handed in my notice — freelancing full time from March", "2025-02-27T17:00:00"),
      ("mornings are deep work now, calls only after two — freelancing rule one", "2025-04-03T08:00:00"),
      ("landed on a day rate instead of hourly, after two awkward freelancing negotiations", "2025-05-21T15:00:00"),
      ("Nadia's baby shower is on the 14th", "2025-06-02T12:00:00"),
      ("Tuesdays I work from the coworking space now, for the company of other humans", "2025-08-12T09:30:00")],
     [0, 2, 3, 4, 6], [[0, 2]]),
    ("ho2-mf-003", "multi_factor_change",
     "trace my thinking on the night shift", "evolution", "shift",
     [("my days have a fixed shape: train at six, dinner at eight, asleep by eleven", "2025-01-09T20:00:00"),
      ("the printer at work jams on anything double-sided", "2025-01-28T10:00:00"),
      ("took the night shift rotation — better pay, and the ward needs seniors on nights", "2025-02-15T14:00:00"),
      ("sleep is now nine am to four pm behind blackout blinds on shift weeks", "2025-03-09T09:00:00"),
      ("training moved to five pm, before the shift starts", "2025-04-01T17:00:00"),
      ("my brother finally sold his motorbike", "2025-05-17T13:00:00"),
      ("we do Sunday lunch as the family meal now because dinners are gone on shift weeks", "2025-06-29T13:30:00"),
      ("swapped my book group to the daytime shift-workers' one at the library", "2025-09-14T11:00:00")],
     [0, 2, 3, 4, 6, 7], [[0, 2], [0, 3]]),
    ("ho2-mf-004", "multi_factor_change",
     "how has my thinking about the sailing boat evolved?", "evolution", "sailing",
     [("Saturdays are for the flea market and slow breakfasts, and have been for years", "2025-02-22T10:00:00"),
      ("the lift in our building is out till Friday", "2025-03-12T08:00:00"),
      ("bought a quarter share in a 26-foot sailing boat with Tom and the Kerrs", "2025-04-05T16:00:00"),
      ("Saturdays are now sail days, tide permitting", "2025-05-10T09:00:00"),
      ("there's a boat budget line now: mooring, insurance, and a scary sinking fund", "2025-06-14T19:00:00"),
      ("Ana's quiz team needs a sub on Thursday", "2025-07-30T18:00:00"),
      ("winter weekends go to sanding and varnish at the yard — sailing owns the calendar even ashore", "2025-11-22T14:00:00")],
     [0, 2, 3, 4, 6], [[0, 3]]),
    ("ho2-mf-005", "multi_factor_change",
     "how has my thinking about sign language evolved?", "evolution", "sign",
     [("Thursday evenings have been five-a-side since university", "2025-01-23T19:30:00"),
      ("the landlord repainted the stairwell a violent green", "2025-02-11T09:00:00"),
      ("enrolled in the sign language beginners' course — my niece signs faster than she speaks now", "2025-03-04T18:00:00"),
      ("Thursdays are now sign language class; football moved to the Monday casual game", "2025-03-27T19:00:00"),
      ("my commute podcasts switched to Deaf culture interviews and sign language vlogs", "2025-05-15T08:15:00"),
      ("quarterly reviews moved online", "2025-06-20T15:00:00"),
      ("practice supper with Maya is Sunday now — she corrects my handshapes between bites", "2025-09-07T18:30:00")],
     [0, 2, 3, 4, 6], [[0, 3]]),

    # ─── perspective_shift: same event, reinterpreted over time;
    # pairs = (initial interpretation → reinterpretation) ───────────
    ("ho2-ps-001", "perspective_shift",
     "how has my thinking about the audit evolved?", "evolution", "audit",
     [("the letter says we're being audited for the last two tax years; I feel sick", "2025-01-31T09:30:00"),
      ("Priya's wedding invitations finally went out", "2025-02-18T12:00:00"),
      ("the audit feels like persecution — three years of receipts for a business that barely profits", "2025-03-05T22:00:00"),
      ("hired a proper bookkeeper to survive the audit; first time the accounts have ever been current", "2025-04-17T10:00:00"),
      ("the gym changed its opening hours again", "2025-05-28T07:00:00"),
      ("the audit closed with a tiny adjustment and an apology-shaped letter", "2025-07-11T14:00:00"),
      ("I've stopped calling it persecution — the audit was tuition; the books it forced on us got us the loan", "2025-10-23T16:00:00")],
     [0, 2, 3, 5, 6], [[2, 6]]),
    ("ho2-ps-002", "perspective_shift",
     "how has my thinking about the flood evolved?", "evolution", "flood",
     [("came home to four inches of water in the flat; the sofa is ruined and so is my week", "2025-02-09T18:45:00"),
      ("work rolled out yet another expense tool", "2025-02-25T11:00:00"),
      ("the flood took the photo albums — that's the part I can't talk about yet", "2025-03-14T21:00:00"),
      ("insurance came through; we replaced almost nothing — the empty living room feels strangely light", "2025-05-02T17:00:00"),
      ("Sam got tickets for the 22nd", "2025-06-10T13:00:00"),
      ("the flood introduced me to more neighbours in a month than five years of hallway nods", "2025-07-26T10:30:00"),
      ("I'd never choose the flood, but it shocked us into the decluttered life we kept meaning to pick", "2025-11-30T15:00:00")],
     [0, 2, 3, 5, 6], [[2, 6]]),
    ("ho2-ps-003", "perspective_shift",
     "how has my thinking about stepping down evolved?", "evolution", "stepping",
     [("I asked to step down from team lead; it goes into effect next month", "2025-01-27T16:00:00"),
      ("the canteen finally has decent vegetarian options", "2025-02-13T12:30:00"),
      ("everyone's being kind about the step-down, which somehow makes it worse — it reads as failure", "2025-03-01T20:00:00"),
      ("first month back as a senior engineer: shipped more than in any quarter of managing", "2025-04-08T18:00:00"),
      ("my passport renewal took nine days", "2025-05-19T09:00:00"),
      ("a grad asked why I 'gave up' the lead role and I heard myself defend stepping down without flinching", "2025-08-06T14:00:00"),
      ("stepping down wasn't failure — it was the most senior decision I've made; the craft was the career", "2025-12-04T17:30:00")],
     [0, 2, 3, 5, 6], [[2, 6]]),
    ("ho2-ps-004", "perspective_shift",
     "how has my thinking about the sabbatical evolved?", "evolution", "sabbatical",
     [("signed the sabbatical papers: six months unpaid, starting June", "2025-03-10T15:00:00"),
      ("the neighbour's cat has adopted our windowsill", "2025-04-02T08:30:00"),
      ("everyone at the sabbatical leaving drinks said 'brave' the way you'd say 'doomed' — career suicide with a smile", "2025-05-30T21:00:00"),
      ("month two of the sabbatical: I slept, walked, and read; the ambition noise is finally quieter", "2025-07-28T10:00:00"),
      ("council tax went up again", "2025-08-15T09:00:00"),
      ("took a tiny consulting gig mid-sabbatical from the old team — on my terms, three days, done", "2025-09-19T11:30:00"),
      ("the sabbatical wasn't career suicide, it was the service the career needed; I came back a different worker", "2025-12-15T18:00:00")],
     [0, 2, 3, 5, 6], [[2, 6]]),
    ("ho2-ps-005", "perspective_shift",
     "how has my thinking about the inheritance evolved?", "evolution", "inheritance",
     [("the solicitor confirmed it: Nana left me the house in Wexford", "2025-01-21T11:00:00"),
      ("my laptop hinge finally gave out", "2025-02-06T14:00:00"),
      ("the inheritance feels like a stone — a house I can't visit without crying and taxes I can't parse", "2025-02-28T22:30:00"),
      ("spent a week in Wexford sorting Nana's boxes; found the letters from her sister in Boston", "2025-04-21T16:00:00"),
      ("the pharmacy changed its opening hours", "2025-06-05T10:00:00"),
      ("we host the whole family in the Wexford house every August now", "2025-08-24T13:00:00"),
      ("the house stopped being a stone — the inheritance is the family's gravity, and I'm its keeper", "2025-11-16T19:00:00")],
     [0, 2, 3, 5, 6], [[2, 6]]),

    # ─── reversed_belief_chain: X → Y → back-to-X-with-nuance;
    # resumption/settlement phrasings on the return beats ───────────
    ("ho2-rb-001", "reversed_belief_chain",
     "how has my relationship with cash gone back and forth?", "throughline", "cash",
     [("cash is dead weight — I went fully cashless, cards and phone for everything", "2025-01-12T10:00:00"),
      ("the office moved to a four-day-week pilot", "2025-02-04T09:00:00"),
      ("the food budget vanished by the 19th and I couldn't say where a single tap went", "2025-02-20T21:00:00"),
      ("cashless was wrong for me — back on cash envelopes for everything, drawn Monday", "2025-03-18T08:30:00"),
      ("my library card expired mid-loan", "2025-05-09T12:00:00"),
      ("paying the plumber in creased twenties was ridiculous; some of this belongs on the card", "2025-06-27T17:00:00"),
      ("landed on the hybrid: cashless for bills and big stuff again, cash envelopes only for food and fun", "2025-09-21T10:30:00")],
     [0, 2, 3, 5, 6], [[0, 3], [3, 6]]),
    ("ho2-rb-002", "reversed_belief_chain",
     "trace my thinking on gaming", "throughline", "gaming",
     [("Friday co-op gaming with the guild is the best hour of my week, ten years running", "2025-01-17T22:00:00"),
      ("the recycling schedule changed to alternate weeks", "2025-02-12T08:00:00"),
      ("rage-quit for good — uninstalled everything; gaming is a slot machine wearing a story", "2025-03-07T23:30:00"),
      ("the group chat has gone quiet without me; Dev says the guild misses their healer", "2025-04-25T20:00:00"),
      ("Priya switched teams at work", "2025-06-03T11:00:00"),
      ("back in — gaming again, but only the Friday guild night, no solo grinding", "2025-07-18T21:30:00"),
      ("six months of Friday-only gaming and it holds; turns out it was the people, not the game", "2025-12-19T22:00:00")],
     [0, 2, 3, 5, 6], [[0, 2], [2, 5]]),
    ("ho2-rb-003", "reversed_belief_chain",
     "how has my thinking about naps evolved?", "evolution", "naps",
     [("naps are for toddlers — adults who nap are hiding from their day", "2025-02-01T15:00:00"),
      ("the team offsite is in Leeds this year", "2025-02-24T10:00:00"),
      ("the 3pm crash won; twenty-minute naps are now part of my day and they're glorious", "2025-03-19T15:30:00"),
      ("week three of naps: waking groggy more days than not, and nights are getting shallow", "2025-04-09T16:00:00"),
      ("my umbrella died spectacularly on the bridge", "2025-05-27T18:00:00"),
      ("dropped the naps — my sleep-doctor friend says my chronotype banks it all at night", "2025-06-30T14:00:00"),
      ("no naps, but a hard 9:30 shutdown instead — the crash was a bedtime problem all along", "2025-10-11T21:45:00")],
     [0, 2, 3, 5, 6], [[0, 2], [2, 5]]),
    ("ho2-rb-004", "reversed_belief_chain",
     "how has my relationship with flying gone back and forth?", "throughline", "flying",
     [("three flights this quarter alone — flying is just how my life fits together", "2025-01-08T07:00:00"),
      ("our floor's kitchen is being refitted", "2025-01-30T09:00:00"),
      ("read the aviation-carbon piece twice and did my own math; I'm done flying — trains or nothing", "2025-02-23T19:00:00"),
      ("the sleeper to Vienna was twelve hours of actual sleep and zero guilt about not flying", "2025-04-15T08:00:00"),
      ("Zoë lent me her label maker and it's life-changing", "2025-05-24T13:00:00"),
      ("missed Amir's wedding in Beirut — no train goes there, and sorry-by-video-call broke my heart", "2025-07-05T23:00:00"),
      ("flying again — one long-haul a year, family only, offset and owned; trains for everything else", "2025-10-19T12:00:00")],
     [0, 2, 3, 5, 6], [[0, 2], [2, 6]]),
    ("ho2-rb-005", "reversed_belief_chain",
     "how has my thinking about hosting evolved?", "evolution", "hosting",
     [("hosting dinners is my favourite thing — eight people, three courses, every other Friday", "2025-01-10T19:00:00"),
      ("the market moved to the school car park for winter", "2025-02-07T10:00:00"),
      ("hosted four times this quarter and spent two days recovering each time; cancelling the next one", "2025-03-28T23:00:00"),
      ("I've stopped hosting entirely — restaurants split eight ways are civilised", "2025-04-30T20:30:00"),
      ("my headphones only pair on the third try now", "2025-06-16T08:00:00"),
      ("restaurant dinners feel like meetings with wine; nobody lingers and nobody helps with the dishes", "2025-08-09T22:00:00"),
      ("hosting again — but potluck now, paper napkins, and we do it monthly instead of fortnightly", "2025-11-01T18:30:00")],
     [0, 2, 3, 5, 6], [[0, 3], [3, 6]]),
]


def build() -> list[dict]:
    out = []
    for sid, family, q, qtype, theme, props, gold, pairs in SCENARIOS:
        n = len(props)
        dates = [p[1] for p in props]
        assert dates == sorted(dates), f"{sid}: dates must increase in index order"
        assert n >= 3 and gold and all(0 <= i < n for i in gold), sid
        assert gold == sorted(gold), f"{sid}: gold must be index-ascending (dates ascend)"
        distractors = [i for i in range(n) if i not in gold]
        for old, new in pairs:
            assert old in gold and new in gold and gold.index(old) < gold.index(new), sid
        out.append({
            "id": sid,
            "heldout": True,
            "batch": 2,
            "family": family,
            "propositions": [
                {"text": t, "asserted_at": d, "session": f"s{i+1}"}
                for i, (t, d) in enumerate(props)
            ],
            "questions": [{
                "q": q,
                "type": qtype,
                "expected_theme": theme,
                "expected_beat_order": gold,
                "expected_origin": gold[0],
                "distractor_indices": distractors,
                "expected_supersessions": pairs,
            }],
        })
    return out


if __name__ == "__main__":
    scenarios = build()
    path = Path(__file__).parent / "heldout_batch2.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in scenarios))
    fams = {}
    for s in scenarios:
        fams[s["family"]] = fams.get(s["family"], 0) + 1
    print(f"wrote {len(scenarios)} scenarios to {path}")
    print(f"families: {fams}")
