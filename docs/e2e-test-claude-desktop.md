# End-to-end test in Claude Desktop

10-minute checklist to verify Patha works through a real Claude Desktop session. Use this after `uv sync && make mcp-install` and after a fresh Claude Desktop restart.

## Live verification log (2026-04-20)

This guide was dogfooded end-to-end against a real Claude Desktop install. Results:

| Step | Result |
|---|---|
| `make mcp-install --install-detector stub -y` | ✓ config written to `~/Library/Application Support/Claude/claude_desktop_config.json` (existing keys preserved, old config backed up) |
| Claude Desktop loads the MCP server | ✓ four tools (`patha_ingest`, `patha_query`, `patha_history`, `patha_stats`) become callable in every MCP-aware Claude session |
| `patha_stats` on fresh store | ✓ returns `{total_beliefs: 0, detector: "stub", ...}` |
| `patha_ingest` × 4 (Sofia, Anthropic work, vegetarian, moved-to-Lisbon) | ✓ each returns `action: "added"` with a UUID belief_id |
| `patha_stats` after ingests | ✓ `total_beliefs: 4, current: 4, ingest_tick: 4` |
| `patha_query("what do I do for work?")` | ✓ `strategy: "structured"`, semantic filter narrowed 4 beliefs → 1 (only the Anthropic work belief surfaced) |
| `patha_history("Sofia")` | ✓ returns both Sofia beliefs (live + the moved-to-Lisbon mention) |
| `~/.patha/beliefs.jsonl` on disk | ✓ 5 lines, plain JSON, grep-able, editable |
| Supersession with `stub` detector | ✗ expected — stub is heuristic-only, doesn't catch paraphrastic contradictions. Upgrade to `full-stack-v7` or `full-stack-v8` (below) |
| `full-stack-v7` on same pair via direct API | ✓ `"I live in Sofia"` vs `"I just moved to Lisbon"` → CONTRADICTS @ 0.85 |

**Upgrading to production detector** — if the stub output is too limited, edit `~/Library/Application Support/Claude/claude_desktop_config.json` and change:

```json
"PATHA_DETECTOR": "full-stack-v8"
```

Pre-download the NLI weights (~1.7 GB) before restarting Claude Desktop so it doesn't appear to hang on first ingest:

```bash
PATHA_DETECTOR=full-stack-v8 uv run patha verify --preload
```

Full quit + reopen Claude Desktop. Subsequent ingests will catch paraphrastic supersessions, sequential events ("moved to", "passed away", "upgraded to"), numerical changes ("rent 1500 → 1800"), and fire the learned classifier on topically-similar supersession markers.

## What you're verifying

1. Claude Desktop sees the Patha MCP server in its tools list.
2. Ingesting a belief through Claude persists to `~/.patha/beliefs.jsonl`.
3. Querying through Claude returns current beliefs correctly.
4. Closing and reopening Claude Desktop preserves memory.
5. Supersession fires when a contradicting belief is ingested.
6. The Streamlit viewer shows the same state Claude Desktop sees.

## Steps

### 1. Install

```bash
cd /path/to/patha
uv sync
make mcp-install         # writes Claude Desktop config
```

Quit Claude Desktop completely (⌘Q on macOS; don't just close the window), then reopen it.

### 2. Confirm the server appears

In Claude Desktop → Settings → Developer → MCP Servers, you should see **patha** listed as "Running" with 4 tools: `patha_ingest`, `patha_query`, `patha_history`, `patha_stats`.

If it's not running:

- Open a terminal and run `uv run patha-mcp` from the Patha checkout. If that fails, fix the error there first.
- Check the Claude Desktop log: `~/Library/Logs/Claude/mcp*.log` on macOS.

### 3. Ingest a few beliefs through Claude

Open a new chat in Claude Desktop and say:

> Please remember that I live in Sofia, I work as an AI engineer, and I'm vegetarian. Use the `patha_ingest` tool for each fact.

Expected: Claude calls `patha_ingest` three times. Each call's output shows `action: added`.

### 4. Verify on disk

In your terminal:

```bash
cat ~/.patha/beliefs.jsonl | head
```

You should see three event-log entries — one per belief. This is the raw store; it's plain JSONL.

### 5. Query through Claude

Start a **new chat** (important — this verifies memory persists across sessions, not just within one chat):

> What do you currently know about me?

Expected: Claude calls `patha_query` and paraphrases the three beliefs back.

### 6. Fire a supersession

In the same or new chat:

> I actually just moved to Lisbon last week, not Sofia anymore. Please update.

Expected:
- Claude calls `patha_ingest` with the Lisbon claim.
- The `action` comes back as `superseded` (with the old Sofia belief ID in `affected_belief_ids`).
- Alternatively `added` if the `stub` detector didn't catch it — `stub` has limited pattern coverage. Upgrade to `full-stack-v7` for better contradiction detection (see below).

Verify:

```bash
uv run patha history Sofia
uv run patha history Lisbon
```

`Sofia` should appear as `superseded`; `Lisbon` as `current`.

### 7. Open the viewer

```bash
uv run patha viewer
```

Browser opens to localhost. You should see:
- **Overview**: totals + confidence histogram
- **Timeline**: ~4 events, with the Lisbon one flagged as `superseded`-fired
- **Current**: Lisbon, AI engineer, vegetarian
- **History**: Sofia belief with Lisbon as its successor

### 8. Restart test

Quit Claude Desktop completely. Reopen it. Start a new chat:

> Where do I live?

Expected: Claude calls `patha_query`, returns "Lisbon" (not Sofia).

### 9. Upgrade to production detector (optional)

If steps 1–8 all worked with the `stub` detector, you can upgrade to the full detector. It downloads a ~1.7 GB NLI model on first ingest.

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` and change:

```json
"PATHA_DETECTOR": "full-stack-v7"
```

Pre-download the weights so Claude Desktop doesn't hang on first ingest:

```bash
PATHA_DETECTOR=full-stack-v7 uv run patha verify --preload
```

Quit and restart Claude Desktop. Contradiction detection will now catch paraphrastic supersessions (e.g., "I love sushi" vs "I avoid raw fish") that the stub misses.

## What "should" look like

A healthy session produces a `~/.patha/beliefs.jsonl` file that looks like:

```json
{"type": "add", "belief": {"id": "...", "proposition": "I live in Sofia", "status": "superseded", ...}}
{"type": "add", "belief": {"id": "...", "proposition": "I work as an AI engineer", "status": "current", ...}}
{"type": "add", "belief": {"id": "...", "proposition": "I'm vegetarian", "status": "current", ...}}
{"type": "add", "belief": {"id": "...", "proposition": "I moved to Lisbon last week", "status": "current", ...}}
{"type": "supersede", "new_id": "...", "old_id": "..."}
```

## Failure modes to report

If any step doesn't work, capture these before filing an issue:

- `uv run patha verify` output
- `uv run patha-mcp` output from a terminal (without Claude Desktop)
- `cat ~/Library/Logs/Claude/mcp*.log | tail -50` (macOS)
- Content of `~/Library/Application Support/Claude/claude_desktop_config.json`
- Whether Claude Desktop's tools panel shows patha at all

## OS notes

- **macOS**: config at `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json` (Claude Desktop for Linux isn't official; `make mcp-install` configures this path for community builds)

For Claude Code (the CLI), run `make mcp-install-code` (or `uv run patha install-mcp --client claude-code`) instead — it writes to `~/.claude/config.json`.
