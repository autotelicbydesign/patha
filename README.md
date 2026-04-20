# Patha

> *The way. The recitation.*

**Your AI memory, inspectable and portable.** One belief store shared across every AI tool you use — Claude Desktop, Claude Code, Cursor, Zed, Goose. Runs fully on your machine, zero cloud, zero API calls.

**What makes it different from the memory your AI assistant already has:**

- **You can see it.** Memory lives in a plain `~/.patha/beliefs.jsonl` file you can read, edit, version-control, or export. Open the Streamlit viewer for a visual dashboard.
- **Old beliefs aren't lost.** When you change your mind, Patha marks the old belief as *superseded* instead of overwriting it. You can ask "what did I used to think about X?" and get an answer.
- **Contradictions are explicit.** Every ingest is tagged `added` / `reinforced` / `superseded`, with the full supersession chain visible. No silent "remembered."
- **Cross-tool.** The same belief store feeds every MCP-compatible AI tool. Switch from Claude Desktop to Cursor mid-project — your memory follows you.
- **Portable.** Copy the JSONL file to another machine. Your memory moves with it.
- **Private.** Nothing leaves your laptop. No API keys, no OAuth, no Anthropic account required for the memory itself.

Under the hood: contradiction detection via NLI + lexical rewriting + sequential-event markers + numerical-change handling. Non-commutative belief evolution measured empirically (96% of supersession scenarios are order-dependent). Plasticity mechanisms (time decay, Hebbian associations, homeostasis) operate during normal use. Primitives drawn from two human memory traditions that lasted thousands of years: Vedic recitation (redundant multi-view encoding) and Aboriginal songlines (narrative graph traversal).

**Honest benchmark summary:**

- **Phase 1 retrieval on LongMemEval-KU:** 1.000 R@5 (78/78), beating Mem0 (0.934) and MemPalace (0.966). Session-level chunks, apples-to-apples with Mem0.
- **Unified `patha.Memory` on LongMemEval-KU end-to-end:** 0.455 at turn-granularity ingest (our public API). The 54pp gap vs the retrieval number is a real granularity mismatch — LongMemEval assumes session-level chunks. See [docs/benchmarks.md](docs/benchmarks.md) for the honest analysis.
- **BeliefEval (our own supersession benchmark):** 1.000 at turn-granularity.
- **Non-commutative belief evolution:** 96% of supersession scenarios are order-dependent — empirically measured.

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
```

Use `--detector full-stack-v7` to switch to the production NLI + adhyāsa + numerical + sequential detector (downloads ~1.7 GB on first run). Default is `stub` for instant startup.

### 3. As a Python library (for developers building LLM apps)

**Install:**

```bash
pip install patha
# or:
uv pip install patha
```

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

See [`examples/developer_quickstart.py`](examples/developer_quickstart.py) for a runnable walkthrough, and [docs/benchmarks.md](docs/benchmarks.md) for the head-to-head vs Mem0 and MemPalace.

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

## Why Patha (vs Claude's built-in memory, ChatGPT memory, etc.)

Most AI assistants now have some form of built-in memory. The value of Patha over those isn't that it "remembers things" — they all do that. It's what it does with the memory once it has it.

Four architectural choices that cloud-hosted memory systems don't offer:

1. **You can see and edit your memory.** `~/.patha/beliefs.jsonl` is a plain text file. Open it in any editor. Commit it to git. Diff it between machines. Export it. Anthropic's and OpenAI's memory features give you a settings panel with a toggle; Patha gives you the actual data.

2. **Non-destructive supersession.** When new evidence contradicts an old belief, the old belief moves to *history* — it isn't overwritten. Queries can ask for current-only ("what do I think now?") or current+history ("what did I used to think?"). The cloud-hosted systems make old beliefs disappear as if they were never asserted.

3. **Order-dependent evolution, measured.** On 240 supersession scenarios, reversing the ingest order produces a different final belief set 95.8% of the time (mean divergence 0.91). Reinforcement scenarios correctly come out 0% non-commutative. This means Patha has a principled theory of *when order matters* — and exposes an API to ask "what would I currently believe if I'd heard B before A?" No other memory system I'm aware of makes this explicit.

4. **Cross-tool, cross-process.** Every MCP-compatible AI tool reads the same belief store. Your memory isn't trapped inside one app's account. Switch from Claude Desktop to Cursor mid-project, and Cursor sees what Claude Desktop saw.

Plus: plasticity mechanisms (time decay, Hebbian associations, homeostasis, LTP, pruning) that operate during normal use. On 10 real LongMemEval conversations the confidence distribution has std=0.106 (LTD is doing real work) with a mean of 150 Hebbian edges per conversation (an associative graph emerges from use).

---

## Benchmarks (highlights)

Full numbers with caveats, ablations, and methodology live in [docs/benchmarks.md](docs/benchmarks.md). The headlines:

| Benchmark | Result | Comparison |
|---|---|---|
| **LongMemEval-KU R@5 (Phase 1 retrieval)** | 1.000 (78/78) | Mem0 ECAI 2025: 0.934 |
| LongMemEval S 100q stratified R@5 | 0.989 | — |
| BeliefEval 300-scenario / 347q (Phase 2) | 1.000 with full-stack-v7 | our own benchmark; see caveat |
| **LongMemEval-KU answer-in-summary (Phase 2 alone)** | **0.885 (69/78)** | stub null baseline: 0.795 |
| Non-commutativity on 240 supersession scenarios | 95.8% | 0% on reinforcement |
| Test suite | 602 pass | — |

**Caveat on the 1.000 BeliefEval:** the Phase 2 detector was iteratively tuned on the exact scenarios in our benchmark, so 1.000 on that set should be read as "no known misses" rather than "generalises everywhere." The honest external number is 0.885 on LongMemEval-KU, measured on a benchmark we didn't write.

---

## Architecture

```
INGEST:  conversation turn
           │
           ▼
     Phase 1 — Retrieval (Vedic 7-view + songline graph + BM25 + RRF)
           │
           ▼
     Phase 2 — Belief layer
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
```

Phase 1 is a self-contained retrieval pillar; Phase 2 is a self-contained belief layer. They can be used independently or wired together via `patha.integrated.IntegratedPatha`.

Background reading: [docs/phase_2_spec.md](docs/phase_2_spec.md) (architecture spec), [docs/phase_2_v07_results.md](docs/phase_2_v07_results.md) (latest sprint results with honest caveats), [docs/benchmarks.md](docs/benchmarks.md) (full benchmark tables).

---

## Project structure

```
src/patha/
  chunking/, indexing/, retrieval/, query/, models/   # Phase 1 — retrieval
  belief/                                              # Phase 2 — belief layer
    layer.py, store.py, types.py
    contradiction.py, adhyasa_detector.py, numerical_detector.py,
    sequential_detector.py, llm_judge.py, ollama_judge.py
    plasticity.py, counterfactual.py, validity_extraction.py
    pramana.py, vritti.py, abhava.py, direct_answer.py
    detector_factory.py                               # named-detector registry
  integrated.py                                        # Phase 1 + Phase 2
  cli.py                                               # patha verify/demo/ingest/ask/...
  mcp_server.py                                        # MCP stdio server
  viewer/                                              # Streamlit dashboard
    app.py
  demo.py                                              # patha demo

eval/
  runner.py, ablations.py, metrics.py                 # Phase 1 eval
  belief_eval.py, longmemeval_belief.py               # Phase 2 eval
  non_commutative_eval.py, plasticity_on_real_logs.py # Phase 2 novel metrics
  false_contradiction_eval.py                          # Phase 2 FP rate
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

# Phase 1 retrieval (R@5)
uv run python -m eval.runner --limit 100            # 100q stratified sample
uv run python -m eval.ablations                     # full ablation matrix

# Phase 2 belief layer (external)
uv run python -m eval.longmemeval_belief \
    --detector full-stack-v7 --include-history      # the 0.885 headline

# Phase 2 novelties
uv run python -m eval.non_commutative_eval          # 95.8% on 240 scenarios
uv run python -m eval.plasticity_on_real_logs      # LTD/Hebbian/LTP stats
uv run python -m eval.false_contradiction_eval     # 6% FP rate

# Full test suite
uv run pytest tests/ -q                             # 602 tests, ~10s
```

---

## Roadmap

**Shipped (v0.7, 2026-04):**
- Phase 1 retrieval pillar (R@5 = 1.000 on LongMemEval-KU)
- Phase 2 belief layer with non-destructive supersession, validity, pramāṇa, plasticity, adhyāsa, abhāva, counterfactual replay, contextuality, raw archive
- Sequential-event supersession detector with additive-veto (6% FP rate)
- Non-commutative belief evolution: empirical benchmark + 95.8% measurement
- LongMemEval-KU external eval (0.885 with current+history)
- MCP server, CLI, Streamlit viewer, Python library

**Next (v0.8):**
- Publish to pypi as `patha-memory`
- Phase 1 + Phase 2 integration inside the MCP server (retrieval-filtered supersession)
- Learned sequential-event classifier (target: FPR < 2%)
- LLM-in-the-loop scorer for external benchmarks

**Future:**
- Multi-user belief attribution (whose belief is it?)
- Bayesian confidence propagation
- Adapters for LangChain / LlamaIndex
- Full 500q LongMemEval S eval

---

## Acknowledgments

Built with [Claude Code](https://claude.ai/code) as a pair-programming partner. Architectural decisions, evaluation design, framing, and editorial honesty owned by the author. Code written in collaboration.

## License

MIT.
