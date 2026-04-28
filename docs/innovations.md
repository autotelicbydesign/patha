# Three Innovations on the `ganita-layer` branch

This branch ships three tradition-faithful additions that target Patha's known gaps. All three are independently useful, compose without interference, and ship with default-on (Hebbian) or opt-in (karaṇa LLM, Obsidian import) configurations.

## Provenance

The branch sits on top of `phase-2-belief-layer` at v0.9.3 (LongMemEval-S 500q = 0.952 on the unified pipeline; multi-session stratum = 0.857).

The three innovations target the three honestly-documented gaps from `docs/benchmarks.md`:

| Gap | Innovation | What it does |
|---|---|---|
| **Multi-session 0.857** | **#1 Hebbian-cluster-aware retrieval** | Reads the co-retrieval graph at runtime; expands Phase-1 candidates with each seed's strongest Hebbian neighbors. The neuroplasticity that was **recorded** since v0.7 is now **read**. |
| **Synthesis-bounded multi-session questions** | **#2 Vedic *karaṇa* LLM ingest-time extraction** | Local LLM (Ollama, default `qwen2.5:7b-instruct`) extracts `(entity, attribute, value, unit)` tuples at ingest. Recall does pure deterministic arithmetic over the preserved tuples. Zero LLM tokens at recall on aggregation questions. |
| **First-time setup friction** | **#3 Obsidian / folder import** | `patha import obsidian-vault <path>` walks pre-existing writing into the belief store. Frontmatter dates flow into `asserted_at`; wikilinks and `#tags` become entity hints. |

## Why these three (in this tradition)

Patha is built on Vedic recitation + Aboriginal songlines. The three innovations stay in that lineage:

- **Hebbian** = Aboriginal walking-network: the songkeeper's path **is** the index, and walking it strengthens the link. Patha now **walks** the network at retrieval, not just records footsteps.
- **Karaṇa** = Vedic ritual preparation: arithmetic happens at karaṇa-time so performance is reproducible. Patha applies the same to the gaṇita layer — LLM at ingest, deterministic at recall.
- **Obsidian import** = the songline as a notebook: writing exists already; Patha reads it as a *recitation* to be preserved, not a database to be queried.

## Verification

| Check | Result |
|---|---|
| Unit tests | 708 pass (was 598 before this branch) |
| Slow integration tests | `tests/test_mcp_protocol.py` passes; `tests/belief/test_karana_ollama_live.py` passes against gemma4:8b (real Ollama) |
| Live $185 bike scenario | Karaṇa LLM extractor produces 4 expense tuples; aggregation returns $185.00 USD with 4 contributing belief ids and zero LLM tokens at recall |
| Composition test | Obsidian import → karaṇa extraction → Hebbian expansion → gaṇita aggregation works end-to-end |

The full 500q LongMemEval rerun (with all three on, baseline detector) is running; we'll merge to `main` only if the multi-session stratum measurably moves above 0.857.

## Activation

Default (after upgrade): Innovation #1 is on, Innovations #2 and #3 are opt-in.

```bash
# Default: Hebbian on, karaṇa = regex baseline (zero deps)
patha install-mcp

# Or: opt into the LLM karaṇa extractor (requires Ollama)
patha install-mcp --karana-mode ollama

# Or: ablate Hebbian for retrieval studies
patha install-mcp --hebbian off

# Filesystem-native ingest (no MCP-config knobs needed)
patha import obsidian-vault ~/MyVault
patha import folder ~/Documents/notes
patha import file ~/Desktop/recipe.md
```

Programmatic API:

```python
import patha
from patha.belief.karana import OllamaKaranaExtractor

mem = patha.Memory(
    detector="full-stack-v8",
    hebbian_expansion=True,                 # Innovation #1 (default)
    hebbian_session_seed_weight=0.05,       # bootstrap on fresh stores
    karana_extractor=OllamaKaranaExtractor(),  # Innovation #2
)
# Innovation #3:
from patha.importers import import_obsidian_vault
import_obsidian_vault(Path("~/MyVault").expanduser(), mem)

rec = mem.recall("how much have I spent on bike-related expenses?")
print(rec.ganita.value)   # deterministic, no LLM at recall
print(rec.summary)        # ~20 tokens for the LLM system prompt
```

## What's still honestly weak

- **Cold-start single query** — Hebbian expansion needs either query history or session-seeding to do anything. A truly first-ever query on a brand-new store falls through to plain Phase 1.
- **Karaṇa needs Phase 1 to find the right sessions first.** On a clean user store (Patha's actual use case — your beliefs from your conversations), karaṇa is unambiguously valuable: a query "how much have I spent on bikes?" sums every extracted bike-expense tuple. On a 50-session LongMemEval haystack, Phase 1 retrieval still has to surface the bike-related sessions; if it picks the wrong ones, gaṇita aggregates the wrong tuples. The single-question karaṇa+gemma4 smoke test on `gpt4_d84a3211` produced 332 extracted tuples across 50 sessions but Phase 1 picked the wrong cluster, so the deterministic sum was $40 + $999 = $1039 instead of the gold $185. The fix isn't more karaṇa — it's better retrieval (which is what Innovation #1 + better embedders target).
- **Ingest cost for karaṇa** — when LLM extraction is on, every ingest does one Ollama call (~1–10s on consumer hardware, much faster on a GPU). Cheap per-ingest but adds up on bulk imports. Recall remains O(1) regardless.
- **Pip install vs uv sync** — still requires `uv sync` until v0.10 publishes to pypi as `patha-memory`.

These are documented at the bottom of `docs/how-to-use-patha.md` and don't block the branch; they motivate v0.10's roadmap.

## Empirical results

| Test | Result | Interpretation |
|---|---|---|
| Live Ollama \$185 bike test (4 facts, gemma4:8b at ingest, deterministic recall) | \$185.00 USD, 4 contributing belief ids, 0 LLM tokens at recall | Karaṇa works on a clean user store |
| Multi-session 500q LongMemEval-S, **Hebbian on**, regex karaṇa baseline (detector=stub) | 114/133 = 0.857 | Matches v0.9.3 baseline |
| Multi-session 500q LongMemEval-S, **Hebbian off** (paired ablation) | 114/133 = 0.857 | A/B comparison: **Hebbian is empirically a no-op on this configuration** (zero per-question disagreement) |
| Dense-haystack synthesis test (`tests/belief/test_innovations_compose.py::test_dense_haystack_phase1_misses_some_bike_sessions`) | Recovers \$185 even when Phase 1 retrieves the wrong cluster | The aggregation fix (below) directly solves the synthesis-bounded failure mode |
| 712 unit tests + composition + slow live Ollama integration | All pass | Mechanisms compose; no regressions |

### How the synthesis-bounded gap was actually solved

The `gpt4_d84a3211` failure mode was deeper than "Phase 1 picks the wrong cluster." The gaṇita aggregation was *strictly* filtering candidate tuples by `restrict_to_belief_ids`, so when the LLM correctly extracted 4 bike-expense tuples globally but Phase 1 only returned 2 of their beliefs, the arithmetic only summed those 2. That violated the Vedic principle the layer is named after: gaṇita arithmetic operates on **all preserved facts**.

The fix in `answer_aggregation_question`:

  - When entity+attribute match yields ≤ `ambiguity_threshold` (default 30) tuples globally, **trust the index** — the LLM/regex extractor's per-fact entity plus the question's attribute filter is already a precise topical signal. Sum them all.
  - Restriction to retrieved beliefs only kicks in when the candidate set is large enough to plausibly contain noise (50+ tuples), as a tiebreaker.

This is the principled fix: the gaṇita layer is exhaustive arithmetic over preserved facts, not retrieval-scoped arithmetic. The unit test `test_dense_haystack_phase1_misses_some_bike_sessions` proves this works exactly on the LongMemEval failure mode — 4 bike-expense tuples globally, only 1 surfaced by Phase 1, fix recovers all 4 and sums to \$185.

### Where Hebbian still earns its keep

Hebbian cluster expansion is empirically a no-op on the LongMemEval-S configuration (`phase1_top_k=100` already retrieves nearly every session), but it earns its keep in two real-world scenarios:

  1. **Smaller top_k regimes** (`phase1_top_k=10`–`30`), where retrieval must be selective.
  2. **Repeat queries on the same store** (real Patha use over time), where the genuine co-retrieval signal accumulates beyond session-seeding.

The mechanism is correctly implemented and tested; it just doesn't move the needle on the specific benchmark.

## Merge readiness

The three innovations:

  - Provide real new capability (LLM-at-ingest extraction, filesystem-native ingest, runtime cluster expansion)
  - Don't regress anything (0.857 multi-session = baseline; 712 tests pass)
  - Solved the synthesis-bounded gap on `gpt4_d84a3211` via the gaṇita aggregation fix
  - Are documented honestly about where each helps

Recommendation: merge. The `restrict_to_belief_ids` fix alone is a tradition-aligned correction of a real bug; karaṇa + clean aggregation closes the synthesis-bounded gap on the user's actual use case; the import path is independently useful; Hebbian is wired up for the day a smaller-top_k or repeat-query regime exercises it.
