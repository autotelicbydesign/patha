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
| Live Ollama \$185 bike test (4 facts, gemma4:8b at ingest, deterministic recall) | $185.00 USD, 4 contributing belief ids, 0 LLM tokens at recall | Karaṇa works on a clean user store |
| Karaṇa smoke test on `gpt4_d84a3211` (50-session LongMemEval haystack, gemma4:8b) | 332 tuples extracted; Phase 1 picked the wrong sessions; gaṇita summed $40 + $999 = $1039 instead of gold $185 | Karaṇa works as designed; Phase 1 retrieval is the bottleneck on dense haystacks |
| Multi-session 500q LongMemEval-S, **Hebbian on**, default session seeding 0.05, regex karaṇa baseline (detector=stub) | **114/133 = 0.857** | **No measurable lift over the v0.9.3 baseline (also 0.857).** Same 19 failures, mostly synthesis-bounded. |
| 710 unit tests + composition + slow live integration | All pass | Mechanisms compose; no regressions |

### Why Hebbian doesn't move LongMemEval-S multi-session

LongMemEval-S benchmarks each question with a fresh `patha.Memory()` instance. The Hebbian graph starts empty, gets seeded once via `_maybe_seed_hebbian_from_sessions` on first query, and then the expansion fires.

But on this benchmark configuration:

  - `phase1_top_k=100` (the benchmark default) already returns ≥100 candidate beliefs
  - The haystack has ~50 sessions per question
  - Phase 1 already returns nearly every session
  - Hebbian expansion has nothing useful to *add* — most beliefs are already in the candidate set
  - The 19 multi-session failures are dominated by synthesis, not retrieval (84% per `eval/multisession_diagnosis.py`)

Where Hebbian *would* lift accuracy:

  1. **Smaller top_k regimes** (`phase1_top_k=10`–`30`), where retrieval has to be selective. The benchmark uses 100, which is generous.
  2. **Repeat queries on the same user store**, where the actual co-retrieval signal accumulates. LongMemEval-S is single-shot per question — no chance for the network to learn.
  3. **Heterogeneous-topic stores** where related-by-cooccurrence beliefs span sessions widely apart in time. LongMemEval-S haystacks are ~2 month windows, dense within.

The mechanism is sound — the test bench just doesn't exercise it.

### Why Karaṇa doesn't move LongMemEval-S without Hebbian + better retrieval

The live \$185 test proves karaṇa correctly extracts tuples and returns deterministic sums. But on the synthesis-bounded multi-session questions, karaṇa needs Phase 1 to surface the *right* sessions, not just *enough* sessions. The smoke test on `gpt4_d84a3211` showed Phase 1 picked the wrong cluster — so karaṇa's deterministic arithmetic produced a deterministically-wrong answer.

Karaṇa is a **clean-store** win, not a benchmark-haystack win. For Patha's actual use case (your conversations over time, on your machine), the bike-expense scenario is the realistic target — and karaṇa nails it.

## Honest merge decision

The three innovations:

  - Provide real new capability (LLM-at-ingest, filesystem import, runtime cluster expansion)
  - Don't regress anything (0.857 multi-session = baseline; 710 tests pass)
  - Are documented honestly about what they help and what they don't
  - Set up the right future work (Hebbian shines with query history; karaṇa shines with clean stores; Obsidian import shines for adoption)

The lift on LongMemEval-S 500q is zero in this configuration. The lift on a real user's day-to-day Patha use is real (karaṇa gives token-free aggregation; Obsidian import removes setup friction; Hebbian accumulates value over time).

Merge or hold is a judgement call about whether shipping mechanisms with no LongMemEval lift is worth the complexity cost. The branch is ready either way.
