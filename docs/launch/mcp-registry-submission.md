# Submitting Patha to the MCP Registry

The MCP Registry (https://registry.modelcontextprotocol.io/) is the official "app store" of MCP servers — what Claude Desktop, Cursor, Zed, and other MCP clients browse to find servers. Listing Patha here is the fastest way for the existing MCP-using audience to discover it.

## Prerequisites — already satisfied

- ✅ **Package on PyPI**: `patha-memory==0.10.7` is live at https://pypi.org/project/patha-memory/0.10.7/
- ✅ **Public GitHub repo**: https://github.com/autotelicbydesign/patha
- ✅ **`server.json` in repo root**: drafted at `./server.json` (uses `io.github.autotelicbydesign/patha` namespace per GitHub-auth convention)
- ⏸ **PyPI ownership verification step** — the registry will check that the wheel actually belongs to you. The mechanism for PyPI is documented at https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/package-types.md (typically: include a `MCP-NAME` field in PyPI's `[project.urls]` or similar; see step 1 of the actual flow below)

## The flow — your hands, ~10 minutes

### 1. Install `mcp-publisher`

```bash
brew install mcp-publisher
# or:
curl -L "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_$(uname -s | tr '[:upper:]' '[:lower:]')_$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/').tar.gz" | tar xz mcp-publisher && sudo mv mcp-publisher /usr/local/bin/
```

Verify:

```bash
mcp-publisher --help
```

### 2. Authenticate with GitHub

```bash
mcp-publisher login
```

Opens a GitHub OAuth flow in your browser. Tied to the `autotelicbydesign` namespace because the `name` in `server.json` starts with `io.github.autotelicbydesign/`.

### 3. Verify PyPI ownership

The registry verifies the PyPI package is yours. For PyPI packages this is currently done via a verification field — the simplest approach is described in the published `package-types` doc:

```bash
# Check the current required mechanism (it's small + may evolve during preview)
open https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/package-types.md
```

If a metadata field needs to be added to `pyproject.toml` (e.g. an `mcpName` URL in `[project.urls]`), it'll be a one-line change → bump to v0.10.8 → republish to PyPI → then publish to registry.

If the verification works against the existing `Repository` URL in PyPI metadata (already correct), you can skip the bump and go straight to step 4.

### 4. Publish

From the repo root (where `server.json` lives):

```bash
mcp-publisher publish
```

Reads `server.json`, validates it against the schema at https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json, and posts it to the registry. The first publish for a given `name` claims that namespace for your authenticated GitHub account.

### 5. Verify

```bash
curl -s "https://registry.modelcontextprotocol.io/v0/servers" | python3 -c "import json,sys; d=json.load(sys.stdin); print([s for s in d.get('servers', []) if 'patha' in s.get('name','').lower()])"
```

Or open the registry's web UI (not yet GA but close): https://registry.modelcontextprotocol.io/

## What the listing carries

From `server.json` in this repo:

- **Name**: `io.github.autotelicbydesign/patha`
- **Description**: "Local-first AI memory with Vedic-recitation + Aboriginal-songline architecture. Synthesis-intent routing answers aggregation queries with zero LLM tokens at recall. Includes belief layer with non-destructive supersession, plasticity, and explicit cognitive-status tagging."
- **Repository**: github.com/autotelicbydesign/patha
- **Version**: 0.10.7 (Apache 2.0)
- **Package**: PyPI `patha-memory`
- **Transport**: stdio
- **Runtime entry**: `patha-mcp` (the script `pip install` makes available)
- **Optional env vars**: `PATHA_STORE_PATH`, `PATHA_PHASE1`, `PATHA_GANITA`

## After publishing — keep the listing fresh

Every release that changes the MCP server surface (new tools, env vars, transport) needs a registry republish. The flow each time:

1. Bump `version` in `server.json` to match the new PyPI version
2. `mcp-publisher publish`

For pure-doc / internal releases, no registry republish is needed.

## Notes

- The MCP Registry is currently in preview (API freeze v0.1 since 2025-10-24). The submission flow above is stable; the listing UI is still being built out.
- Listing in the registry doesn't auto-add Patha to anyone's MCP config — it just makes it discoverable. Users still need to add the config block (see `docs/mcp.md`).
