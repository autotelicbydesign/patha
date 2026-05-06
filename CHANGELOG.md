# Changelog

## v0.10.8 (2026-05-06) — Claude conversation import + CLI fixes (REPL, --version, PATHA_DETECTOR env var)

Driven by real user feedback on what the v0.10.7 surfaces actually felt like.

### New: `patha import claude-export <zip>`

Bulk-import your Claude conversation history into a Patha store. Anthropic lets you download all your conversations as a `.zip` from `claude.ai` → Settings → Privacy → Export data. Patha can now read that export and ingest each user-side message (or each user-side sentence, by default) as a belief, with the original timestamp preserved.

Only **user messages** are ingested. Assistant replies are skipped (those are Claude's outputs, not the user's beliefs). Filtering: messages shorter than 30 chars, questions, and obvious commands to Claude are skipped. Messages over a configurable cap (default 10 sentences) are truncated to prevent one brain-dump from flooding the store. Code blocks are stripped.

CLI:
```
patha import claude-export ~/Downloads/data-export.zip
patha import claude-export ~/Downloads/data-export.zip --whole-messages  # don't sentence-split
patha import claude-export ~/Downloads/data-export.zip --verbose         # one line per conversation
```

Why this matters: a major UX gap in v0.10.7 was that "your AI memory follows you across tools" oversold what Patha actually does — Patha is a memory store, not an observer. Live conversations only enter the store when Claude (or you) explicitly call `patha_ingest`. `patha import claude-export` closes the gap retroactively for everything you've already told Claude.

### CLI fixes

- **`patha shell` REPL** — drop into an interactive prompt; type sentences to remember, prefix `?` (or end with `?`) to ask. No more `patha ingest "…"` boilerplate.
- **`--version` flag** — `patha --version` now prints the package version. Basic CLI hygiene that was missing.
- **`PATHA_DETECTOR` env var** — `--detector` now reads the env var as its default. Previously the env var was silently ignored, which meant `export PATHA_DETECTOR=full-stack-v8` did nothing — the CLI kept running on the stub detector. Real bug, surfaced by user testing.
- **`patha install-mcp` honors the global `--detector` / `PATHA_DETECTOR`** — previously `patha install-mcp` always wrote `"PATHA_DETECTOR": "stub"` into the generated Claude Desktop config, regardless of the global flag or env var. Users had to know to use the (less obvious) `--install-detector` flag, or hand-edit the JSON afterwards. Now `patha --detector full-stack-v8 install-mcp` and `PATHA_DETECTOR=full-stack-v8 patha install-mcp` both bake `full-stack-v8` into the config. The dedicated `--install-detector` flag still overrides if explicitly passed.
- **CLI ganita wiring** (was already in v0.10.7) — `patha ingest` / `patha ask` go through the public `Memory` class, so the synthesis path works through the CLI exactly the way it does through the Python library and MCP server.

### Documentation honesty pass

The "Switch tools mid-project; your memory follows" line on the README was technically true (the store is portable) but implicitly oversold what passive observation Patha does (none — it's a store, not a transcript-scraper). Added an explicit clarification block: facts enter via CLI / REPL / file import / explicit `patha_ingest` call from the AI. There is no background magic.

No architectural changes to the Retrieval Layer, Belief Layer, or Articulation Bridge in this release.

## v0.10.7 (2026-05-05) — metric relabel: answer-recall vs end-to-end + first real-LLM Articulation Bridge measurement

A previous version labelled the LongMemEval-KU 1.000 (77/77) result as "end-to-end accuracy." That was a category error: the metric scores whether the gold answer (or a synonym) appears as a *substring* in Patha's emitted summary. **No LLM is involved in scoring.** It measures what Patha surfaces, not what an LLM does with it.

**Relabelled across README, `docs/benchmarks.md`, `docs/releases/v0.10.7.md`:**
- *was*: "End-to-end accuracy on KU: 1.000 (77/77)" (in v0.10.5/0.10.6 README)
- *is*: "Belief-Layer answer-recall on KU: 1.000 (77/77)" — explicitly described as "the gold answer appears in Patha's summary; no LLM in the scoring loop"
- Same correction applied to Claim B (0.987), Claim C (0.952 / 472/496), and the multi-session 0.857 line — all are answer-recall, not end-to-end.
- Added a "three different metrics, three different things" explainer block in the README so readers don't conflate retrieval R@5, answer-recall, and Articulation Bridge end-to-end.

**Added: first real-LLM Articulation Bridge measurement.** Ran the existing v0.10 scaffolding (`eval/run_answer_eval`) end-to-end on KU 78q with `qwen2.5:14b-instruct` via local Ollama:
- token-overlap ≥0.6 (LongMemEval-S official scorer): **24/78 = 0.308**
- numeric: 12/78 = 0.154
- embedding-cosine ≥0.55: 36/78 = 0.462
- per-strategy on token-overlap: ganita 5/41 (0.122), structured 19/37 (0.514)
- 12 minutes wall time; 10 s/question on warm GPU

This is **the first real number from an LLM in the loop.** It's an open-source 14B local model — a floor for "real LLMs," not a ceiling. Frontier-LLM measurement (Claude Sonnet 4 / GPT-4o) pending API access; will publish in v0.11.

**Other:**
- Bumped `OllamaLLM.timeout_s` default 60 → 240 (60s was too short for 14B models on long prompts; the first run hit a urllib socket timeout).

No code, test, or architecture changes vs v0.10.6 beyond the timeout bump and the metric relabel.

## v0.10.6 (2026-05-04) — license switched to Apache 2.0

License changed from MIT to Apache 2.0. Substantive reasons:

- **Explicit patent grant.** Apache 2.0 grants users an explicit license to any patents the author holds covering Patha's code; MIT is silent on patents. Patha implements genuinely novel architecture (synthesis-intent routing, gaṇita as a recall strategy, non-commutative belief evolution) — explicit patent terms matter more than for a generic library.
- **Patent retaliation clause.** Anyone who patent-trolls Patha automatically loses their right to use Patha's code under Apache 2.0. MIT has no such mechanism.
- **`NOTICE` file.** Apache 2.0 includes a structured attribution mechanism. `NOTICE` carries the project copyright, attribution to Stefi P. Krishnan, and the Claude Code pair-programming acknowledgment.

Practical changes:

- `LICENSE` file added (was missing — the v0.10.5 README badge linked to a 404).
- `NOTICE` file added.
- `pyproject.toml`: `license = "Apache-2.0"`; classifier `License :: OSI Approved :: Apache Software License`.
- README badge `License: MIT` → `License: Apache 2.0`.
- README "## License" section now points at LICENSE + NOTICE.

No code or test changes. License is wheel metadata, so a version bump is required to publish the change.

## v0.10.5 (2026-05-04) — first PyPI release

v0.10.2, v0.10.3, and v0.10.4 were all uploaded to TestPyPI but never to real PyPI. v0.10.5 is the accumulated fixes from three review rounds and is the first version published to real PyPI.

**Cumulative changes across the v0.10.2 → v0.10.5 review chain:**

- **Author metadata** — `authors = "Stefi P. Krishnan"` (was `"stefi"`). The wheel records the author and the rendered PyPI page shows it; bumping the version is the only clean way to fix this.
- **Public-facing architecture renamed.** `Phase 1` / `Phase 2` is release-sequencing language, which planted a misleading "is one superseded by the other?" reading. The architecture is now framed as **two layers and one bridge**:
  - **Retrieval Layer (Pratyakṣa)** — direct perception / lookup. Function: did the gold session surface in top-K?
  - **Belief Layer (Anumāna)** — inference / belief evolution. Function: reason over time — what do I currently believe? what changed?
  - **Articulation Bridge** *(not a runtime layer)* — connection between Patha's output and a user's LLM, plus the methodology for measuring how well it works. Function: given Patha's output, does the user's LLM articulate the right answer?

  The asymmetry is intentional: only the two layers execute on `.recall()`. The bridge runs in your application code (or our offline eval harness when we measure it). The first two layers are named after canonical Nyāya pramāṇas; the bridge stands on its English name (Sanskrit pairing left open — *śabda* was rejected because in Mīmāṃsā śabda is *infallible authoritative testimony*, opposite directionality from "candidate measured against gold"). Internal engineering filenames (`docs/phase_2_*`, `docs/phase_3_plan.md`) keep "Phase N" — release-history bookkeeping, not user-facing taxonomy.
- **README opening rewritten** — drops "The way. The recitation." / "Your AI memory, inspectable and portable." / "What makes it different from the memory your AI assistant already has." Replaces with a Vedic-recitation + Aboriginal-songline epistemology framing that surfaces the architectural claim above the fold.
- **Headline numbers above Quickstart** — R@5 = 1.000 on LongMemEval-KU, 6.5× token reduction, zero LLM tokens at recall, now visible in the first 200 words.
- **"Why Patha (vs Claude's built-in memory…)"** → **"Beyond default AI memory: where Patha fits in your stack"**. Same content, neutral framing.
- **Roadmap audit** — no residual "v0.10.x will publish R@5 vs MemPalace" deferral language.

No code or test changes vs the v0.10.1 git tag content; this is a metadata + README release across the whole v0.10.2/3/4/5 chain.

## v0.10.4 (2026-05-04) — TestPyPI only (superseded by v0.10.5)

Renamed third piece from "Answer Layer (Upamāna)" → "Articulation Layer" (no Sanskrit pairing — śabda's directionality doesn't fit). v0.10.5 then renamed it again to "Articulation Bridge" to reflect that it isn't a runtime layer.

## v0.10.3 (2026-05-04) — TestPyPI only (superseded by v0.10.4)

Author metadata fix + first pass of public layer rename. Used "Answer Layer (Upamāna)" for the third piece.

## v0.10.2 (2026-05-03) — TestPyPI only (superseded by v0.10.3)

Identical engineering to v0.10.1 (which was tagged but never published to PyPI), plus:

- **README rewritten to lead with the architectural distinction** (synthesis-intent routing, zero LLM tokens at recall) rather than benchmark tables. All implicit and explicit cross-system comparison framing removed. The numbers Patha publishes are Patha's own measured numbers; readers can compare them to whatever they want.
- **`writeups/update_08_compression.md` receipt updated** to match.
- **Version bump only**: 0.10.1 → 0.10.2. No code or test changes vs the v0.10.1 git tag content; this is a packaging release.

v0.10.1 remains in git history as a tag (commit `3a44572`) for reference; it was demoted to git-only when we decided the receipt's framing needed work before the first public PyPI publish.

## v0.10.1 (2026-04-30) — git tag only, never published to PyPI

Post-tag follow-ups on top of v0.10.0:

- **Language audit.** Replaced "synthesis bypasses Phase 1" with the precise claim across README, `docs/innovations.md`, CHANGELOG, source docstrings, demo, and the writeup receipt. Phase 1 still runs in parallel to populate retrieval context; only the synthesis *answer source* is independent of Phase 1's top-K.
- **Test rename.** `test_synthesis_intent_bypasses_phase1` → `test_synthesis_intent_independent_of_phase1`. Sweep applied; prior name removed from all references.
- **Receipt restructure.** README and `writeups/update_08_compression.md` now use a bulleted v0.10 receipt with exact `118,761 → 18,384` token numbers, the `¹ one question excluded` footnote on KU 77/77, and Option-A wording on the extraction-quality claim ("~84% of failures are synthesis-bounded; full benchmark with stronger extractors is future work" — measured rather than overclaimed).
- **Three regex extractor false-positive filters** (`src/patha/belief/ganita.py`):
  - **Range filter** — `$100 to $500`, `$50-$200`, `$80–$120` no longer extract two phantom purchases.
  - **Hypothetical filter** — `thinking about a $300 helmet`, `would cost $X`, `considering`, `if I bought` get suppressed within 50 chars.
  - **Negated-purchase filter** — `didn't buy the $X`, `couldn't afford`, `returned the $X helmet`, `decided against` get suppressed.
  - 12 new test cases; 750 unit tests pass.
- **Packaging hardening.** README install line now says `pip install patha-memory` (correct PyPI distribution name) with explicit Python-3.11+ note. Wheel metadata verified: `Name: patha-memory`, `Version: 0.10.1`, `Requires-Python: >=3.11`.


## v0.10.0 (2026-04-29)

### One coherent architectural claim

**Patha separates retrieval queries from synthesis queries.** No mainstream AI memory system makes this distinction; they all force every question through the same top-K funnel.

Two pramāṇa, two paths:

- **Retrieval** — *pratyakṣa*: "what did I say about the saddle?" → Phase 1 (7-view dense + BM25 + RRF + reranker + songlines) → Phase 2 (current-state filter) → direct-answer or structured summary.
- **Synthesis** — *anumāna*: "how much have I spent on bikes total?" → gaṇita queries the belief store directly. Phase 1 isn't the right primitive — top-K of N misses (N-K) of the inputs you need to sum. Pure deterministic arithmetic over preserved tuples. **Zero LLM tokens at recall.**

### What this changes in the codebase

**`patha.Memory.recall()`** — detects aggregation intent (existing `detect_aggregation`); on synthesis, queries the gaṇita index globally without `restrict_to_belief_ids`. Phase 1 runs in parallel to populate retrieval context. The architectural claim: the synthesis answer is independent of Phase 1's top-K — proven by `test_synthesis_intent_independent_of_phase1` (forces Phase 1 to return `[]` and gaṇita still computes the right answer).

**`answer_aggregation_question`** — bug fix: was strictly filtering by `restrict_to_belief_ids`. Violated the Vedic principle gaṇita is named for (arithmetic on **all** preserved facts). Now uses an `ambiguity_threshold` (default 30): trust the index when entity+attribute matches yield few enough tuples globally; restrict only on broad/ambiguous queries.

**`patha.belief.karana.HybridKaranaExtractor`** — new. Regex finds every `$X` in text (perfect recall on the easy task); LLM only labels each `(entity, aliases, attribute)` or marks `entity: "skip"` for ranges/hypotheticals. The architectural design splits the work along the abstraction line; recall isn't bounded by the LLM's free-form judgement. Generalises beyond karaṇa.

**`patha.belief.karana.OllamaKaranaExtractor`** — full LLM extraction; prompt asks for explicit broader-category aliases (`saddle → ["saddle", "bike", "cycling"]`).

**`patha.importers`** — new. `patha import obsidian-vault <path>` walks pre-existing writing into the belief store. Frontmatter dates → `asserted_at`; wikilinks/`#tags` → entity hints. Read-only.

### Hebbian retrieval

Kept on the branch with honest framing.

- No measurable lift on LongMemEval-S 500q multi-session (paired A/B = identical).
- No regression either.
- Real for repeat-query workloads — a store queried many times accumulates co-retrieval edges that Phase-1's static cosine never sees. LongMemEval is single-shot, so the recorded signal can't accumulate.

### Empirical caveat

The synthesis-intent routing reaches the right architectural answer; the gaṇita arithmetic is correct. But it depends on the karaṇa extractor's tagging quality. Tested with `gemma4:8b` (Q4) and `qwen2.5:7b-instruct`; both work on clean user stores. On dense LongMemEval haystacks, small-quantized models miss bike-relevant facts (mis-classify "$25 chain replacement" as a duration; fail to add "bike" to the helmet's alias list).

**Recommendation: ≥ 14B local model or hosted LLM for synthesis-heavy workloads.** The architecture doesn't change; only the karaṇa quality.

### Phase 3 plan

`docs/phase_3_plan.md` lays out the next milestone: end-to-end answer evaluation. Token-overlap-on-summary measures retrieval; the product question is whether the user's LLM, given Patha's output, produces the correct answer. Phase 3 ships that scorer.

### Verification

- 725 unit tests pass (was 598 before this branch).
- `tests/test_mcp_protocol.py::test_mcp_full_roundtrip` passes.
- `tests/belief/test_karana_ollama_live.py` passes against `gemma4:8b`.
- `test_synthesis_intent_independent_of_phase1` proves the architectural claim: the synthesis answer is independent of Phase 1's top-K — gaṇita computes the right answer even when Phase 1 is forced to return `[]`.

### Multi-session benchmark (500q LongMemEval-S, stub detector, regex karaṇa)

| Metric | Baseline (v0.9.3) | ganita-layer | Δ |
|---|:---:|:---:|:---:|
| Multi-session accuracy | 114/133 = 0.857 | 114/133 = 0.857 | 0 |
| Average tokens / summary | ~118,000 | ~18,400 | **−6.5×** |

**Accuracy is unchanged** because the regex karaṇa extractor can't reliably extract the relevant tuples from dense conversational text. The architecture is correct; the bottleneck is extraction quality. With `HybridKaranaExtractor` + ≥14B model the synthesis path delivers correct answers (verified end-to-end on the canonical \$185 bike scenario via `tests/belief/test_karana_ollama_live.py`).

**Token economy improves 6.5×** because synthesis questions now produce a compact gaṇita summary (`"sum: \$50 + \$75 + \$30 + \$30 = \$185.0"` plus contributing beliefs) instead of the full Phase 2 retrieval dump. Zero LLM tokens at recall on the synthesis path.
