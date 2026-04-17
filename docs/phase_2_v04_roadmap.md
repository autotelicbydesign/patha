# Phase 2 v0.4 — Deeper Cognitive Dimensions

**Status:** ROADMAP — design, not implementation
**Source traditions:** Nyāya-Vaiśeṣika / Advaita Vedānta (for the Vedic items) + quantum cognition (Busemeyer, Pothos, Bruza)
**Purpose:** extend Patha with cognitive-theory primitives that give it dimensions no other AI memory system has. Landing after v0.3 ships.

---

## Guiding principle

Every concept from Vedic tradition and quantum cognition must earn its place by mapping to an **operational software mechanism** that is demonstrably better than its secular equivalent. If the mechanism reduces to "good engineering with a philosophical name," we call it good engineering and skip the costume. Previously v0.2 concluded neuroplasticity framing is *mostly* decorative; we only kept what was mechanistic (plasticity mechanisms now wired into runtime in v0.3). Same bar applies here.

v0.3 already landed one item from this layer:
- **Pramāṇa-aware source tracking** (six sources of valid knowledge, reinforcement-diversity weighting).

v0.4 takes on the remaining six.

---

## 1. Vṛtti-state taxonomy (Yoga Sūtras, I.5–11)

**Classical concept.** Patañjali lists five *vṛttis* — modifications of mind:

- **Pramāṇa** — valid cognition (already a first-class axis in Patha)
- **Viparyaya** — erroneous cognition (mistaken belief)
- **Vikalpa** — imagined / verbal-only cognition (without corresponding reality)
- **Nidrā** — latency / dormancy
- **Smṛti** — recall of prior impression (memory proper)

**Software translation.** Extend `ResolutionStatus` with vṛtti-aligned states:

| Existing status | Vṛtti-aligned refinement |
|---|---|
| CURRENT | pramana-grounded current belief |
| DISPUTED | possibly-erroneous (viparyaya candidate) |
| AMBIGUOUS | vikalpa (asserted but unverified) |
| ARCHIVED | nidrā (latent; can be reactivated) |
| (new) SMRTI-LINKED | memory pointing at another belief rather than asserting one |

Each status would gain a vṛtti label and a retrieval-policy consequence: vikalpa beliefs don't surface by default in direct-answer paths; viparyaya-candidates always surface with a confidence caveat; nidrā beliefs reactivate under specific triggers (pattern match on new input).

**Effort:** small. 1-2 commits. Mostly enum extension + minor policy changes in the query path.

**Value:** genuine. Adds a cognitive vocabulary to belief states that the field doesn't have.

---

## 2. Order-sensitive belief operations (quantum cognition)

**Classical concept.** In quantum cognition (Busemeyer & Bruza 2012), mental operators are *non-commutative*: asking A then B yields different results than B then A. This is empirically well-supported in humans (order effects in survey research, Moore 2002).

**Software translation.** Surface the fact that Patha's final belief state depends on *order* of assertions, not just the set. Two APIs:

```python
# What do you currently believe, if you'd heard B before A?
counterfactual = layer.query_counterfactual(
    question="What does the user prefer?",
    reorder=[belief_b, belief_a],  # instead of the historical order
)

# By how much does the current state depend on order?
divergence = layer.order_sensitivity_score(belief_ids)
```

Internally this would replay the supersession/reinforcement decisions in the specified order and return the alternate belief state. Not destructive; counterfactual state is computed in a clone.

**Effort:** medium. 3-5 commits. The store already records ordered events (JSONL event log); replaying them in a different order is straightforward. The novel piece is the API surface and the divergence metric.

**Value:** high. **No other memory system exposes counterfactual belief formation.** This is a publishable primitive.

**Risk to watch:** quantum-cognition framing must not be conflated with "quantum mechanics in the brain" (Penrose-Hameroff territory). The framing is mathematical (non-commutative operators), not physical.

---

## 3. Saṁskāra → Vāsanā layered confidence

**Classical concept.** A *saṁskāra* is a mental impression left by an experience; repeated saṁskāras deepen into *vāsanās* (deep-set tendencies that shape future cognition without being explicitly recalled). The Yoga tradition is specific about this: vāsanās are what make repeated acts *habits* vs. isolated events.

**Software translation.** Add a second confidence layer beneath the surface one. Surface confidence moves quickly under LTP/LTD. Deep (vāsanā) confidence moves slowly and only after a threshold number of reinforcements have accumulated, and once established, resists decay.

```python
@dataclass
class LayeredConfidence:
    surface: float        # 0-1, fast-moving (current v0.3 confidence)
    deep: float           # 0-1, slow-moving; updated only after N reinforcements
    deep_established: bool  # True once deep crosses an establishment threshold
```

Mechanistically:
- Each reinforcement increments a *samskara counter* on the belief.
- When the counter crosses a threshold (e.g., 10), a portion of surface confidence crystallises into deep confidence.
- Deep confidence decays at 1/10th the surface LTD rate.
- Direct-answer responses cite deep confidence when available ("strong belief, held for years") vs surface confidence ("current position").

**Effort:** medium. 3-4 commits. Adds a new field, mutation logic in reinforce(), decay logic in LTD, surfacing in DirectAnswerer.

**Value:** high. Maps to how humans actually hold beliefs — some surface and revisable, some deeply ingrained. Enables *"the user has always believed X"* vs *"the user currently believes X"* distinction, which no competing system has.

---

## 4. Adhyāsa — superimposition-based contradiction detection

**Classical concept.** Śaṅkara's *adhyāsa* is the cognitive error of superimposing one thing's attributes onto another — mistaking a rope for a snake, to take the classical example. In contemporary terms: a near-identity relation that NLI models miss because the surface lexemes differ (sushi ≈ raw fish; vegetarian ≈ doesn't-eat-fish).

**Software translation.** Add an *adhyāsa detector* in the contradiction pipeline that runs *before* the NLI/LLM judge:

1. For each (P1, P2) pair, extract the key noun phrases.
2. For each pair of noun phrases (one from each proposition), query an entity-linking service (we already use spaCy; upgrade to an ontology-aware model): is there an is-a, part-of, or equivalence relation?
3. If yes, rewrite P2 with the superimposed identity substituted ("I am avoiding **sushi**" instead of "raw fish"), and re-run NLI.
4. If the rewrite causes CONTRADICTS, flag the original pair as contradiction via adhyāsa.

This is the exact failure class v0.2's BeliefEval exposed and v0.3's hybrid patches per-case. Adhyāsa detection is the principled fix.

**Effort:** large. 5-8 commits. Requires an ontology (WordNet / ConceptNet / Wikidata) + entity-linking upgrade + the rewrite-and-retest logic.

**Value:** very high. Closes the specific failure mode in v0.2 BeliefEval (prefs-01, prefs-03, prefs-04) without scripting LLM verdicts.

---

## 5. Abhāva — the epistemology of negation (Nyāya)

**Classical concept.** The Nyāya school treats absence (*abhāva*) as a first-class category of knowledge, with four kinds:

- **Prāgabhāva** — prior absence ("the pot doesn't exist yet")
- **Pradhvaṁsābhāva** — destructive absence ("the pot is broken now")
- **Anyonyābhāva** — mutual absence ("A is not B")
- **Atyantābhāva** — absolute absence ("a sky-flower never existed")

**Software translation.** A belief can refer to an absence. Current Patha treats "I no longer eat meat" as a proposition like any other, losing the *negation-of-prior-state* structure. Abhāva-aware beliefs would:

- Carry an `abhava_kind` field on the belief
- Link pradhvaṁsābhāva beliefs ("no longer X") to the destroyed prior belief ("X was asserted before")
- Treat atyantābhāva claims ("never X") as stronger-than-supersession — they invalidate not just the latest state but the entire history of that claim
- Surface abhāva-negation distinctly in direct-answer rendering ("The user no longer does X [since date]" vs "The user has never done X")

**Effort:** small-medium. 2-3 commits. Mostly metadata + a few rendering rules.

**Value:** medium-high. Currently our validity-decay handles one crude form of abhāva. Full abhāva handling gives us principled treatment of negation that competing systems lack.

---

## 6. Contextuality — session-scoped / context-dependent beliefs

**Classical concept (quantum cognition).** Same concept has different meaning in different contexts: *trust* in a professional relationship ≠ *trust* in an intimate one; *home* as physical location ≠ *home* as family.

**Software translation.** Beliefs get a `context` dimension alongside `session_id`. Currently we flatten across contexts. v0.4 would:

- Tag contexts at ingest time (user-supplied or auto-detected: "work", "health", "family", "finance")
- Scope queries by context: `layer.query(..., context="work")` returns only beliefs tagged to work (or context-independent ones)
- Allow same proposition-like claim to exist non-contradictorily in two contexts ("I'm available" in work context vs. personal context)
- Detect cross-context contradictions (apparent same-proposition claims that differ by context are *not* contradictions)

**Effort:** medium. 3-4 commits. New field, new filters, light changes to the contradiction path.

**Value:** high. Many BeliefEval-like failures come from cross-context confusion; handling context explicitly is both correct and demoable.

---

## 7. Probability interference (quantum cognition) — optional / defer

**Classical concept.** Classical probability: P(A and B) = P(A) × P(B | A). Quantum formulation admits interference terms: P(A and B) can be lower than either marginal. The *conjunction fallacy* (humans judging "Linda is a bank teller AND feminist" more likely than "Linda is a bank teller") has been modelled this way (Pothos & Busemeyer 2013).

**Software translation.** Use quantum-style probability when combining evidence from multiple beliefs on related claims. This would replace or supplement Bayesian joint-probability logic in the confidence-update step.

**Honest assessment.** This is mathematically rich but the **operational benefit for Patha is unclear**. We don't have a clear task where the classical-probability combination is observably wrong for Patha's use case. I'd defer unless v0.4 surfaces a specific weakness in the conjunction behaviour.

**Effort:** large. 5+ commits, plus a theoretical design doc.

**Value:** uncertain. Mark as "wait for a weak spot before implementing."

---

## Sequencing recommendation

v0.4 sprint order, highest-impact-per-hour first:

1. **Vṛtti-state taxonomy** (small, clean, narrative-rich)
2. **Abhāva-aware negation** (closes a real gap)
3. **Contextuality** (closes a real gap and is demoable)
4. **Saṁskāra → Vāsanā layered confidence** (medium effort, unique capability)
5. **Order-sensitive / counterfactual belief API** (high-leverage publication piece)
6. **Adhyāsa-based contradiction detection** (biggest lift but requires ontology infra)

Probability interference: only if v0.4 reveals a concrete weak spot.

Order items 1-4 don't depend on each other and could ship independently.

---

## What this layer is *not*

- It is not "AI that works like the brain." The brain claim is out of scope and would be false.
- It is not claiming quantum-mechanical processes in cognition. The quantum-cognition framing is mathematical (non-commutative operators in belief state transitions), not physical.
- It is not a Vedic apologia. Patha borrows operational insights that happen to be well-theorised in Indian philosophy; the same borrowings could be expressed in other traditions' vocabularies (e.g., Peirce on abductive inference for arthāpatti). We use Vedic terms because they're the ones with the richest and most specific operational commitments — and because doing so surfaces the cultural epistemology question the field otherwise avoids.

## Related files

- `docs/phase_2_spec.md` — the overall Phase 2 spec
- `docs/phase_2_literature_survey.md` — survey of competing memory systems
- `docs/phase_2_v01_results.md` — first BeliefEval run
- `docs/phase_2_v02_results.md` — v0.2 complete sprint results
- `src/patha/belief/` — current (v0.3) implementation
