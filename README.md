# Patha

[![PyPI version](https://img.shields.io/pypi/v/patha-memory.svg)](https://pypi.org/project/patha-memory/)
[![Python versions](https://img.shields.io/pypi/pyversions/patha-memory.svg)](https://pypi.org/project/patha-memory/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/autotelicbydesign/patha/tests.yml?branch=main&label=tests)](https://github.com/autotelicbydesign/patha/actions/workflows/tests.yml)

**Local-first AI memory designed from a different epistemology.**

Most AI memory systems treat memory as storage and retrieval — a warehouse with an index. Patha treats memory as architecture, drawn from two traditions:

- **Vedic recitation**, which preserved sacred texts across three thousand years through redundant encoding — the same proposition stored seven different ways, so a query that misses one view catches another.
- **Australian Aboriginal songlines**, the oldest continuously transmitted information on Earth, where the landscape itself is the index and retrieval is a walked path — not point lookup, but narrative traversal.

What that produces, mechanically:

- **Patha separates retrieval from synthesis.** Ask *"what did I say about X?"* — retrieval works the way every memory system works. Ask *"how much have I spent on Y total?"* — the system answers directly from a structured belief state, with **zero LLM tokens at recall** and a **6.5× compression** of the context other systems dump into your prompt.

- **Beliefs carry their cognitive status.** Drawing from the classical Indian philosophical schools (a third strand alongside the recitation tradition), every belief is tagged with how it was learned (*pramāṇa* — perception, inference, testimony, comparison, postulation, non-perception), what mode it surfaces in (*vṛtti* — direct, mistaken, imagined, dormant, remembered), and how deep it has crystallised (*saṃskāra* → *vāsanā*, surface vs. established). These are mechanisms, not metaphors. Each one is testable.

- **Old beliefs are never overwritten.** When you change your mind, Patha marks the old belief as superseded with a full lineage you can walk. You can ask *"what did I used to think about X?"* and get an answer.

The belief store is plain JSONL — `~/.patha/beliefs.jsonl` — that you can read, edit, version-control, grep, or copy to another machine. Nothing leaves your laptop. The same store feeds every MCP-compatible AI tool: Claude Desktop, Claude Code, Cursor, Zed, Goose.

> **What "feeds" means here, precisely.** Patha is a memory *store*, not an observer. There is no background process that scrapes your conversations. Facts enter the store when you put them in — via the CLI / REPL, by importing a file, or by your AI explicitly calling `patha_ingest` after you tell it *"remember that …"*. To bulk-import what you've already told Claude in past chats, see `patha import claude-export <export.zip>` below.

---

## See it run

### 1. AI memory that does math

Four purchases mentioned across four conversations → *"how much have I spent on bike-related expenses?"* → `$185.00`, computed directly with **zero LLM tokens at recall**.

![Patha synthesis-intent demo](assets/demo-synthesis.gif)

### 2. When you change your mind, AI memory should change too

Tell Patha you love sushi every week. Six months later, tell it you're avoiding raw fish. Ask *"what do I currently eat?"* — the new belief supersedes the old. Ask *"what did I used to think?"* — the old belief is filed under history, not deleted.

![Patha supersession demo](assets/demo-supersession.gif)

### 3. Your AI memory belongs to you

No cloud. No login. No SaaS account. Patha writes to a plain text file at `~/.patha/beliefs.jsonl` that you can grep, git-commit, copy to another machine. The same file feeds every MCP-compatible AI tool (Claude Desktop, Cursor, Zed) — your memory follows you.

![Patha portability demo](assets/demo-portability.gif)

---

## At a glance

- **R@5 = 1.000** on LongMemEval-KU (78q, public knowledge-update subset)
- **6.5× token reduction** on the LongMemEval-S multi-session stratum (118,761 → 18,384 tokens/summary)
- **Zero LLM tokens at recall** on synthesis questions (gaṇita queries a preserved tuple index)
- **95.8% non-commutative** — on 240 supersession scenarios, reversing ingest order produces a different final belief set
- **799 unit tests pass** (3 skipped on optional deps)

Methodology and full tables in [docs/benchmarks.md](docs/benchmarks.md). Caveats and metric definitions there too — these are Patha's own measured numbers; cross-system comparison is left to the reader on like-for-like terms.

---

## Two layers and one bridge

Patha has two internal layers that run inside `Memory.recall()`. The third piece — the Articulation Bridge — is *not* a runtime layer of Patha; it's the connection between Patha's output and your LLM, and the harness that measures it. Naming it a "bridge" rather than a third layer keeps the architecture honest: only two things execute when you call `.recall()`; the bridge runs in your application code (or in our offline eval harness when we measure it).

- **Retrieval Layer (Pratyakṣa)** — *"that which stands before the senses,"* direct perception. 7-view Vedic encoding (pada / krama / jaṭā / ghana / entity-anchored / reframed / temporally-anchored) + BM25 + RRF + cross-encoder reranker + songline graph traversal. Function: did the gold session surface in top-K?
- **Belief Layer (Anumāna)** — *"knowledge that follows from what is observed,"* inference. Contradiction detection (NLI + adhyāsa + numerical + sequential), non-destructive supersession, plasticity (LTP, LTD, Hebbian co-retrieval, homeostasis, pruning), validity, pramāṇa, vṛtti. Function: reason over time — what do I currently believe? what changed?
- **Articulation Bridge** *(not a runtime layer)* — the connection from Patha's memory output to a user's LLM, plus the methodology for measuring how well it works. Five scorers (exact / normalised / numeric / token-overlap / embedding-cosine / LLM-as-judge), three LLM adapters (Null / Claude / Ollama), one runner CLI. Function: given Patha's output, does the user's LLM articulate the right answer?

`Memory.recall()` routes by question intent:

- **Retrieval intent** — *"What did I say about the saddle?"* Retrieval Layer → Belief Layer's current-state filter → direct-answer or structured summary.
- **Synthesis intent** — *"How much have I spent on bikes total?"* The gaṇita component of the Belief Layer queries the preserved tuple index exhaustively. Pure deterministic arithmetic. **Zero LLM tokens at recall.** The Retrieval Layer still runs in parallel to populate retrieval context, but the synthesis answer is independent of its top-K.

Top-K retrieval is the wrong primitive for synthesis: top-100 of 1000 sessions misses 90% of the inputs you'd need to sum. Mainstream AI memory systems force every question through the same retrieval funnel and let an LLM clean up at recall — paying tokens per query, indefinitely. Patha doesn't.

## What ships in v0.10

- **Synthesis-intent routing** — `Memory.recall()` detects sum/count/avg/min/max/difference and routes to gaṇita. Verified by `test_synthesis_intent_independent_of_phase1`, which forces the Retrieval Layer to return `[]` and the gaṇita layer still recovers the canonical $185.
- **Retrieval Layer R@5: 1.000** on the LongMemEval-KU 78-question public subset.
- **Belief-Layer answer-recall on KU: 1.000 (77/77)** ¹ with synthesis-intent routing on (up from 0.987 baseline). *Answer-recall = the gold answer (or one of its synonyms) appears as a substring in Patha's emitted summary. This is a measurement of what Patha surfaces, **not** what an LLM does with it; see "Articulation Bridge" below for the end-to-end-through-an-LLM measurement.*
- **Average tokens/summary on the multi-session 500q stratum: 18,384** — a **6.5× reduction** from the 118,761 baseline, with **zero LLM tokens at recall** on the synthesis path.
- **Hybrid karaṇa extractor** — regex enumerates every `$X`, LLM only labels semantically. Recall preserved; LLM cost paid once at ingest, never at recall.
- **Three regex false-positive filters** — range, hypothetical ("thinking about"), negated-purchase ("didn't buy"). Documented in `tests/belief/test_ganita.py::TestFalsePositiveFilters`.
- **Filesystem-native ingest** — `patha import obsidian-vault <path>` walks pre-existing writing into the belief store.
- **Articulation Bridge scaffolding** — `eval/answer_eval.py` + `eval/run_answer_eval.py` ship the engine, three LLM adapters, six scorers, and a runner CLI. Measured baseline floor on KU 78q with NullTemplateLLM: 5/78 = 0.064 (the bar a real LLM should beat). **Real measurement on KU 78q (qwen2.5:14b local, token-overlap ≥0.6 — the LongMemEval-S official scorer): 0.308 (24/78). Frontier-LLM measurement pending.**

¹ One question excluded from answer-recall scoring due to a known datetime-tz edge case; scored over 77.

## Token economy

| Strategy | Tokens / query | vs naive RAG |
|---|:---:|:---:|
| Naive RAG (raw history dump) | 285.9 | 1.0× |
| Patha structured summary | 64.6 | **4.5× reduction** |
| Patha direct-answer (incl. gaṇita aggregation) | **0** | **∞ (no LLM call)** |

Full methodology in [docs/benchmarks.md](docs/benchmarks.md).

---

## Quickstart — 2 minutes

```bash
# 1. Install (Python 3.11+ and uv required)
git clone https://github.com/autotelicbydesign/patha.git
cd patha
uv sync

# 2. Check your environment
uv run patha verify

# 3. Run the end-to-end demo (no downloads, ~10 seconds)
uv run patha demo

# 4. Pick how you want to use it:
uv run patha-mcp     # run as MCP server on stdio (for Claude Desktop)
uv run patha ingest "I am vegetarian"
uv run patha ask "what do I eat?"
uv run patha viewer  # visual inspection in browser
```

That's it. State persists to `~/.patha/beliefs.jsonl` across all three modes.

---

## Three ways to use it

### 1. As an MCP server (Claude Desktop, Claude Code, Cursor)

**One command:**

```bash
make mcp-install            # detects OS, writes Claude Desktop config safely
make mcp-install-code       # for Claude Code instead
```

Quit + restart Claude Desktop. Four tools (`patha_ingest`, `patha_query`, `patha_history`, `patha_stats`) become available. See [docs/mcp.md](docs/mcp.md) and [docs/e2e-test-claude-desktop.md](docs/e2e-test-claude-desktop.md) for details and the post-install verification checklist.

**Or manually** — add one block to your MCP client's config:

```json
{
  "mcpServers": {
    "patha": {
      "command": "uv",
      "args": ["run", "--project", "/ABSOLUTE/PATH/TO/patha", "patha-mcp"],
      "env": { "PATHA_STORE_PATH": "/Users/YOU/.patha" }
    }
  }
}
```

Restart your client. Four tools become available: `patha_ingest`, `patha_query`, `patha_history`, `patha_stats`. Your AI assistant can now remember things across sessions, detect contradictions, and reason over a personal belief store. See [docs/mcp.md](docs/mcp.md) for the full install guide + Claude Desktop walkthrough.

### 2. As a CLI

```bash
patha ingest "I love sushi"
patha ingest "I am avoiding raw fish on my doctor's advice"
patha ask "what do I currently eat?"          # routes through supersession
patha history "sushi"                         # every mention, current + superseded
patha stats                                   # store counts + plasticity state

# Or skip the prefix entirely with the REPL:
patha shell                                   # type sentences naturally
                                              # prefix `?` to ask

# Bring an existing Obsidian vault, Markdown folder, or single file:
patha import obsidian-vault ~/MyVault
patha import folder ~/Documents/notes
patha import file ~/Desktop/recipe.md

# Bulk-import what you've already told Claude:
patha import claude-export ~/Downloads/data-export.zip
# (Get the export from claude.ai → Settings → Privacy → Export data.
#  Only your messages are imported; Claude's replies are skipped.)
```

Use `--detector full-stack-v7` to switch to the production NLI + adhyāsa + numerical + sequential detector (downloads ~1.7 GB on first run). Default is `stub` for instant startup.

### 3. As a Python library (for developers building LLM apps)

**Install** (Python 3.11+ required):

```bash
pip install patha-memory     # PyPI distribution name
# or:
uv pip install patha-memory
```

The import name is `patha`; the PyPI distribution is `patha-memory`. If you see a `thinc`/`spacy` build error during install, you're likely on Python ≤ 3.10 — upgrade to 3.11+.

`Memory()` defaults to a persistent store at `~/.patha/beliefs.jsonl`. For tests or smoke checks, pass `path=` explicitly (e.g. `Memory(path="/tmp/test.jsonl")`) so each run starts fresh and doesn't accumulate state across the host's other Patha invocations.

**Use (5 lines):**

```python
import patha

memory = patha.Memory(detector="full-stack-v8")
memory.remember("I live in Lisbon")
memory.remember("I am avoiding raw fish on my doctor's advice")

rec = memory.recall("where do I live?")
print(rec.summary)          # ~20-token string to drop into an LLM system prompt
print(rec.answer)            # direct answer (when the layer can produce one)
```

**Wire it into an Anthropic-API chatbot for 10–15× smaller memory context:**

```python
import anthropic, patha

client = anthropic.Anthropic()
memory = patha.Memory(detector="full-stack-v8")

def on_user_message(text: str) -> str:
    memory.remember(text)                      # auto-ingest user fact
    mem = memory.recall(text).summary          # ~20 tokens instead of ~280
    reply = client.messages.create(
        model="claude-sonnet-4",
        system=f"User memory:\n{mem}",
        messages=[{"role": "user", "content": text}],
        max_tokens=512,
    )
    return reply.content[0].text
```

**Why the library matters for developers:**

- **Token bill.** Naive conversation-history dumping is ~280–325 tokens per turn. Patha's structured summary is ~20 tokens on the same benchmark — a 10–15× cut. At $3–15 / 1M tokens × many users × many turns, that's real money.
- **Contradiction handling.** When a user changes their mind, `.remember()` resolves it via supersession. Your app doesn't overwrite facts silently.
- **Local-only by default.** No SaaS, no API keys, no rate limits. The belief store is a JSONL file in `~/.patha/` that your user owns.
- **Swap detectors.** Use `"stub"` in CI (instant, no models), `"full-stack-v8"` in prod (DeBERTa-large NLI + lexical + numerical + learned classifier, ~1.7 GB first-download).

**Power-user APIs:**

```python
memory.store          # underlying BeliefStore — raw event log
memory.belief_layer   # underlying BeliefLayer — plasticity, thresholds, etc.
memory.history("X")   # every mention of X, current + superseded
memory.stats()        # counts, plasticity state, data path
```

**For synthesis-heavy workloads** ("how much have I spent on bikes?", "how many books read this year?"), enable the karaṇa LLM extractor at ingest. ≥14B local model or hosted LLM recommended for dense conversational text:

```python
import patha
from patha.belief.karana import HybridKaranaExtractor

memory = patha.Memory(
    detector="full-stack-v8",
    karana_extractor=HybridKaranaExtractor(
        model="qwen2.5:14b-instruct",  # or your model
    ),
)
memory.remember("I bought a $50 saddle for the bike")
# ...
rec = memory.recall("how much have I spent on bike-related expenses?")
print(rec.ganita.value)  # 50.0 — deterministic, zero LLM tokens at recall
```

The synthesis answer is independent of the Retrieval Layer's top-K — gaṇita queries the preserved tuple index exhaustively (`docs/innovations.md` for the architectural explanation). The Retrieval Layer still runs in parallel to populate retrieval context; the answer just doesn't depend on it. Top-100 of 1000 sessions would otherwise miss 90% of inputs you'd need to sum.

See [`examples/developer_quickstart.py`](examples/developer_quickstart.py) for a runnable walkthrough, and [docs/benchmarks.md](docs/benchmarks.md) for the full benchmark methodology.

### Streamlit viewer

```bash
uv pip install 'patha[viewer]'
uv run patha viewer
```

Opens a browser dashboard over `~/.patha/beliefs.jsonl`:

- **Overview** — totals, confidence histogram, detector status
- **Timeline** — chronological ingest events (added / reinforced / superseded)
- **Current** — live belief table
- **History** — superseded beliefs with their successors
- **Non-commutative replay** — enter propositions, see how forward vs reversed ingest orders produce different final beliefs

---

## Beyond default AI memory: where Patha fits in your stack

Four architectural choices that distinguish Patha:

1. **You can see and edit your memory.** `~/.patha/beliefs.jsonl` is a plain text file. Open it in any editor. Commit it to git. Diff it between machines. Export it.

2. **Non-destructive supersession.** When new evidence contradicts an old belief, the old belief moves to *history* — it isn't overwritten. Queries can ask for current-only ("what do I think now?") or current+history ("what did I used to think?").

3. **Order-dependent evolution, measured.** On 240 supersession scenarios, reversing the ingest order produces a different final belief set 95.8% of the time (mean divergence 0.91). Reinforcement scenarios correctly come out 0% non-commutative. Patha has a principled theory of *when order matters* — and exposes an API to ask "what would I currently believe if I'd heard B before A?"

4. **Cross-tool, cross-process.** Every MCP-compatible AI tool reads the same belief store. Switch from Claude Desktop to Cursor mid-project, and Cursor sees what Claude Desktop saw.

Plus: plasticity mechanisms (time decay, Hebbian associations, homeostasis, LTP, pruning) that operate during normal use. On 10 real LongMemEval conversations the confidence distribution has std=0.106 (LTD is doing real work) with a mean of 150 Hebbian edges per conversation (an associative graph emerges from use).

---

## Benchmarks (highlights)

Full numbers with caveats, ablations, and methodology live in [docs/benchmarks.md](docs/benchmarks.md). The headlines:

| Benchmark | Result | Notes |
|---|---|---|
| **LongMemEval-KU R@5 (Retrieval Layer)** | 1.000 (78/78) | did Phase 1 surface the gold session in top-5? — perfect on the public KU subset |
| LongMemEval S 100q stratified R@5 | 0.989 | same retrieval-quality metric |
| **LongMemEval-KU answer-recall (Belief Layer)** | **1.000 (77/77)** ¹ | did the gold answer appear as a substring in Patha's summary? *(measures what Patha surfaces, not what an LLM does with it)* |
| **LongMemEval-KU end-to-end through LLM (Articulation Bridge)** | **0.308 (24/78)** | qwen2.5:14b local, token-overlap ≥0.6 (LongMemEval-S official scorer). **Frontier-LLM measurement pending.** |
| BeliefEval 300-scenario / 347q (Belief Layer) | 1.000 with full-stack-v7 | our own benchmark; see caveat |
| LongMemEval-KU answer-in-summary alternate scoring | 0.885 (69/78) | stub null baseline: 0.795 |
| Articulation Bridge baseline floor (KU 78q, NullTemplateLLM, numeric scorer) | 5/78 = 0.064 | the bar a real LLM should beat |
| Non-commutativity on 240 supersession scenarios | 95.8% | 0% on reinforcement |
| Test suite | 799 pass | 3 skip on optional deps |

¹ One question excluded from answer-recall scoring due to a known datetime-tz edge case; scored over 77.

**Three different metrics, three different things they measure:**
- **Retrieval R@5** — *did Phase 1 surface the gold session in the top-5?* (a Retrieval Layer quality measurement)
- **Belief-Layer answer-recall** — *does the gold answer string (or a synonym) appear in Patha's emitted summary?* (a Belief-Layer surface-quality measurement; we previously labelled this "end-to-end" — it isn't, since no LLM is involved in scoring)
- **Articulation Bridge end-to-end** — *given Patha's output as context, does the user's LLM produce an answer that matches the gold under a chosen scorer?* (the actual end-to-end answer accuracy with a real LLM in the loop)

**Caveat on the 1.000 BeliefEval:** the Belief Layer's detector was iteratively tuned on the exact scenarios in our benchmark, so 1.000 on that set should be read as "no known misses" rather than "generalises everywhere." The honest external number is 0.885 on LongMemEval-KU, measured on a benchmark we didn't write.

---

## Architecture

```
INGEST:  conversation turn
           │
           ▼
     Retrieval Layer — Pratyakṣa (Vedic 7-view + songline graph + BM25 + RRF)
           │
           ▼
     Belief Layer — Anumāna
           ├─ contradiction detection (NLI + adhyāsa + numerical + sequential)
           ├─ non-destructive supersession
           ├─ plasticity (LTD / LTP / Hebbian / homeostasis / pruning)
           └─ validity + pramāṇa + context
           │
           ▼
     BeliefStore  ──────►  ~/.patha/beliefs.jsonl   (append-only event log)
           │
           ▼
QUERY:    current-only  or  current + history
           │
           ▼
     strategy: direct-answer (no LLM) | structured summary | raw
           │
           └─────────────►  Patha output (ends here)
                                       │
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                                       ▼
ARTICULATION BRIDGE  (your application — or our eval harness):
     Patha output  +  your LLM  →  articulated answer
     (offline harness adds: scorer × gold answer → accuracy)
```

The Retrieval Layer is a self-contained retrieval pillar; the Belief Layer is a self-contained inference pillar. They can be used independently or wired together via `patha.integrated.IntegratedPatha`. The Articulation Bridge sits between Patha's output and a user's LLM — in production it lives in the application code that calls `memory.recall()` and then prompts an LLM; in evaluation, our offline harness in `eval/run_answer_eval.py` exercises the same shape against a benchmark JSON to measure how often the bridge produces correct answers.

Background reading: [docs/phase_2_spec.md](docs/phase_2_spec.md) (Belief Layer architecture spec), [docs/phase_2_v07_results.md](docs/phase_2_v07_results.md) (latest sprint results with honest caveats), [docs/phase_3_plan.md](docs/phase_3_plan.md) (Articulation Bridge plan), [docs/benchmarks.md](docs/benchmarks.md) (full benchmark tables).

---

## Project structure

```
src/patha/
  chunking/, indexing/, retrieval/, query/, models/   # Retrieval Layer
  belief/                                              # Belief Layer
    layer.py, store.py, types.py
    contradiction.py, adhyasa_detector.py, numerical_detector.py,
    sequential_detector.py, llm_judge.py, ollama_judge.py
    plasticity.py, counterfactual.py, validity_extraction.py
    pramana.py, vritti.py, abhava.py, direct_answer.py
    detector_factory.py                               # named-detector registry
  integrated.py                                        # Retrieval + Belief Layer
  cli.py                                               # patha verify/demo/ingest/ask/...
  mcp_server.py                                        # MCP stdio server
  viewer/                                              # Streamlit dashboard
    app.py
  demo.py                                              # patha demo

eval/
  runner.py, ablations.py, metrics.py                 # Retrieval Layer eval
  belief_eval.py, longmemeval_belief.py               # Belief Layer eval
  non_commutative_eval.py, plasticity_on_real_logs.py # Belief Layer novel metrics
  false_contradiction_eval.py                          # Belief Layer FP rate
  answer_eval.py, run_answer_eval.py                  # Articulation Bridge — eval engine + runner
  token_economy.py                                     # compression curves

examples/
  belief_layer_demo.py                                 # walkthrough story
  mcp_config_example.json                              # Claude Desktop template

docs/
  mcp.md                                               # MCP install guide
  benchmarks.md                                        # full numbers
  phase_2_spec.md, phase_2_v0{1..7}_results.md        # design + results
```

---

## Reproducing the numbers

```bash
# LongMemEval data (not in this repo — download from upstream)
# https://github.com/xiaowu0162/long-mem-eval → place at data/longmemeval_s_cleaned.json

# Retrieval Layer (R@5)
uv run python -m eval.runner --limit 100            # 100q stratified sample
uv run python -m eval.ablations                     # full ablation matrix

# Belief Layer (external)
uv run python -m eval.longmemeval_belief \
    --detector full-stack-v7 --include-history      # the 0.885 headline

# Belief Layer novelties
uv run python -m eval.non_commutative_eval          # 95.8% on 240 scenarios
uv run python -m eval.plasticity_on_real_logs      # LTD/Hebbian/LTP stats
uv run python -m eval.false_contradiction_eval     # 6% FP rate

# Articulation Bridge — offline measurement harness (end-to-end)
uv run python -m eval.run_answer_eval \
    --data data/longmemeval_ku_78.json \
    --llm null --scorer numeric                     # baseline floor: 5/78

# Full test suite
uv run pytest tests/ -q                             # 799 tests, ~75s
```

---

## Roadmap

**Shipped (v0.10):**
- Retrieval Layer (R@5 = 1.000 on LongMemEval-KU)
- Belief Layer with non-destructive supersession, validity, pramāṇa, plasticity, adhyāsa, abhāva, counterfactual replay, contextuality, raw archive
- Sequential-event supersession detector with additive-veto (6% FP rate)
- Non-commutative belief evolution: empirical benchmark + 95.8% measurement
- Synthesis-intent routing — gaṇita arithmetic at recall, zero LLM tokens
- 6.5× token reduction on the multi-session 500q stratum
- Hybrid karaṇa extractor + three regex false-positive filters
- Articulation Bridge scaffolding — engine, three LLM adapters, six scorers, runner CLI, KU baseline floor
- MCP server, CLI, Streamlit viewer, Python library
- Published to PyPI as `patha-memory`

**Near-term:**
- Belief Layer + Retrieval Layer integration inside the MCP server (retrieval-filtered supersession)
- Real LLM runs on the Articulation Bridge (Claude / Ollama on KU and BeliefEval)
- Karaṇa-quality correlation: how does the Articulation Bridge accuracy curve change as the karaṇa extractor moves from regex → ollama-7b → hybrid-14b?
- BeliefEval adapter for the Articulation Bridge runner (300 supersession scenarios via the same engine)

**Longer-term:**
- Multi-user belief attribution (whose belief is it?)
- Bayesian confidence propagation
- Adapters for LangChain / LlamaIndex
- Persistent index API for the Retrieval Layer, so the MCP server can run dense retrieval across sessions without re-embedding

---

## Acknowledgments

Designed by [Stefi P. Krishnan](https://www.linkedin.com/in/stefka-peykova/) and built with [Claude Code](https://claude.ai/code) as a pair-programming partner.

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
