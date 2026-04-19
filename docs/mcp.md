# Patha MCP server

Patha plugs into any MCP-compatible AI tool (Claude Desktop, Claude Code, Cursor, Zed, Goose) as a local memory server. Unlike built-in assistant memory (Claude's or ChatGPT's), the belief store is:

- **Inspectable** — a plain `~/.patha/beliefs.jsonl` file you can read, edit, grep, git-diff.
- **Non-destructive** — old beliefs become *superseded*, not overwritten; you can ask for history.
- **Explicit about contradictions** — every ingest is tagged `added` / `reinforced` / `superseded`.
- **Shared across tools** — the same belief store feeds Claude Desktop, Claude Code, Cursor, Zed, Goose simultaneously.
- **Local-only** — your memory never leaves your machine.

If Claude's built-in memory is enough for your use case, great — don't adopt this. Reach for Patha when you want to see the memory, own it, move it between machines, or share it across AI tools.

## Install

### Option A — from a local clone (today)

```bash
git clone https://github.com/autotelicbydesign/patha.git
cd patha
uv sync --extra mcp
uv run patha verify      # confirm env is good
uv run patha-mcp         # verify it starts (Ctrl-C to stop)
```

### Option B — from pypi (once published)

```bash
uvx patha-memory patha-mcp --help
```

## Wire up Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS (or the equivalent on Windows/Linux). Add:

```json
{
  "mcpServers": {
    "patha": {
      "command": "uv",
      "args": ["run", "--project", "/ABSOLUTE/PATH/TO/patha", "patha-mcp"],
      "env": {
        "PATHA_DETECTOR": "stub",
        "PATHA_STORE_PATH": "/Users/YOURNAME/.patha"
      }
    }
  }
}
```

Restart Claude Desktop. You should see a "patha" entry under "Connected MCP Servers" in settings, with four tools listed: `patha_ingest`, `patha_query`, `patha_history`, `patha_stats`.

## Wire up Claude Code

Edit `~/.claude/config.json` the same way. Same four tools become available inside every `claude` session.

## What the tools do

| Tool | Purpose |
|---|---|
| `patha_ingest` | Remember a new proposition. Automatically detects contradictions with what's already stored and routes to one of `added` / `reinforced` / `superseded`. |
| `patha_query` | Ask what the user currently believes about X. Returns a summary + structured list of current beliefs. Pass `include_history=true` to also see what they used to believe (superseded beliefs). |
| `patha_history` | Find every belief that mentions a term — current or superseded. Useful for "did I ever mention X?" queries. |
| `patha_stats` | Inspect the store: total beliefs, current vs superseded vs archived counts, Hebbian-edge count. |

## Example session

```
User: Remember that I'm vegetarian.
Claude: [calls patha_ingest("I am vegetarian")]  → added
Claude: Got it. I'll remember.

User: [new chat, next day] What am I again?
Claude: [calls patha_query("what is the user's diet?")]
         → current: ["I am vegetarian"]
Claude: You're vegetarian.

User: Actually I eat fish now, started last week.
Claude: [calls patha_ingest("I eat fish as of last week")]
         → superseded (belief 1 moved to history)
Claude: Noted — your previous vegetarian status is now in history.

User: What have I ever said about fish?
Claude: [calls patha_history("fish")]
         → matches: current="I eat fish as of last week",
                     superseded=[]  (vegetarian belief matched "fish" indirectly? no)
Claude: You said you eat fish as of last week.
```

## Configuration

Environment variables read by the MCP server:

| Var | Default | Purpose |
|---|---|---|
| `PATHA_STORE_PATH` | `~/.patha` | Where the belief-store JSONL lives. One folder per user. |
| `PATHA_DETECTOR` | `stub` | Contradiction detector. Set to `full-stack-v7` for production behavior (downloads a ~1.7 GB NLI model on first ingest). |

## Detector trade-offs

- **`stub`** (default) — heuristic-only. Fast startup, no downloads. Catches clear asymmetric-negation pairs ("I love X" vs "I don't love X") and exact-entity contradictions. Misses paraphrases ("sushi" vs "raw fish").
- **`full-stack-v7`** — production detector. NLI (DeBERTa-v3-large MNLI) + adhyāsa lexical rewriting + numerical-change detection + sequential-event detection with additive-veto. Catches the vast majority of real-world supersessions.

**Retrieval pipeline** (as of v0.9):

Every `patha_query` runs through Patha's full retrieval pipeline before supersession/summary:

1. **Phase 1 — Vedic 7-view retrieval.** The belief store is indexed across 7 overlapping views (pada, krama, jata, ghana, entity, reframed, temporal) plus a BM25 sparse index. Queries are dense-matched against each view, fused via Reciprocal Rank Fusion, diversified via MMR. This is the paraphrase-robust core of Patha: "do I eat raw fish?" retrieves "I love sushi" even without shared tokens.
2. **Phase 2 — supersession & contradiction resolution.** The retrieved candidates are filtered through the belief layer's current/superseded distinction, then either summarized (Option B) or direct-answered (Option C).

Phase 1 indexes are built **lazily** on the first query after MCP startup (~3 s per 100 beliefs on CPU). Subsequent queries in the same session are fast (<100 ms). After every ingest, the index is marked dirty; the next query rebuilds so new beliefs are findable.

**Scale notes**:
- To disable Phase 1 (e.g., for benchmarking or if your store is tiny and you want cosine-only): `PATHA_PHASE1=off` in the MCP config. A lightweight MiniLM cosine filter (`PATHA_SEMANTIC_FILTER=on`, default) then handles narrowing alone.
- The semantic filter and Phase 1 compose. When both are on, the semantic filter narrows first to the top-K relevant beliefs, then Phase 1 retrieves among those. This is the default and has no measured drawback at typical personal-memory scale.

**Learned classifier (experimental, not default)**: `src/patha/belief/learned_supersession.py` provides a scaffold for training a logistic-regression supersession classifier on top of sentence embeddings. The infrastructure works (train with `python -m patha.belief.learned_supersession`) but the current training set is skewed (245 positive vs 16 negative) and hasn't yielded a production-ready model. Wiring it into the MCP server would be a small change once the dataset is balanced.

For first-time users, start with `stub`. Once you're comfortable and want serious contradiction handling:

```bash
PATHA_DETECTOR=full-stack-v7 uv run patha verify --preload
```

This pre-downloads the NLI weights so Claude Desktop doesn't hang on first use.

## Storage

Every ingest appends to `~/.patha/beliefs.jsonl` (an event log). The store is:

- **Append-only** — no events are ever deleted; supersession marks them as history, doesn't destroy them.
- **Safe for concurrent access** — CLI and MCP server can both write; JSONL appends don't corrupt.
- **Portable** — copy the file to move your memory between machines.
- **Inspectable** — `uv run patha viewer` opens a Streamlit dashboard over it.

## End-to-end testing

For a step-by-step verification checklist (ingest through Claude, confirm persistence across restart, observe supersession, open the viewer), see [e2e-test-claude-desktop.md](e2e-test-claude-desktop.md).

## Troubleshooting

- **Claude Desktop shows "patha" as disconnected** — run `uv run patha-mcp` manually from a terminal; errors will surface. Most often: wrong `--project` path in the config.
- **Slow first ingest** — if `PATHA_DETECTOR=full-stack-v7`, the first call downloads 1.7 GB. Pre-download with `patha verify --preload`.
- **"Permission denied" on `~/.patha`** — the folder is created on first run. If you pre-created it with different permissions, `chmod 755 ~/.patha`.
- **Resetting the store** — delete or move `~/.patha/beliefs.jsonl`. Next ingest creates a fresh file.
