# Publishing Patha to PyPI

Status: wheel + sdist built locally at `dist/` as **v0.9.1**. Final publish requires your PyPI credentials, so it's a manual step — not something Claude Code can do for you.

## Pre-flight checks (all passing as of v0.9.1)

- [x] **Name available.** `pypi.org/simple/patha/` → 404 verified; `patha` is unclaimed.
- [x] **Build succeeds.** `uv build` produces:
  - `dist/patha-0.9.1.tar.gz` (sdist)
  - `dist/patha-0.9.1-py3-none-any.whl` (wheel)
- [x] **Bundled model included.** The wheel contains `patha/belief/_models/supersession_classifier.joblib` (13 KB trained on 421 pairs).
- [x] **Three console scripts registered.** `patha`, `patha-mcp`, `patha-viewer`.
- [x] **Top-level API.** `import patha; patha.Memory()` works out of the box.
- [x] **All 656 tests pass** (+ 1 slow MCP e2e test).

## One-time account setup

1. Register at https://pypi.org (create an account).
2. Register at https://test.pypi.org too (for rehearsal uploads).
3. Generate an API token at https://pypi.org/manage/account/token/ with scope `Entire account` for the first upload (you can narrow it to `patha` after the project exists).
4. Save the token. It starts with `pypi-` and looks like `pypi-AgEIcHlwaS5vcmc...`.

## One-command publish (after setup)

```bash
# TestPyPI rehearsal (recommended first time):
TEST_PYPI_TOKEN="pypi-AgEIcHlwaS5...your-token..." make publish-test

# Real PyPI:
PYPI_TOKEN="pypi-AgEIcHlwaS5...your-token..." make publish
```

`make build` is called automatically by both targets. The wheel at `dist/patha-0.9.2-py3-none-any.whl` includes the bundled supersession classifier, all 4 CLI scripts (`patha`, `patha-mcp`, `patha-viewer`), and the top-level `patha.Memory` developer API.

## Rehearsal on TestPyPI (recommended first)

```bash
cd /Users/stefi/Sandbox/Memory

# Use twine with the test repo
uv pip install twine
python -m twine upload --repository testpypi dist/*
# Prompted for credentials:
#   username: __token__
#   password: <your TestPyPI token>
```

Then in a scratch virtualenv:

```bash
python -m venv /tmp/patha-test && source /tmp/patha-test/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            patha
patha verify
patha demo
```

If verify + demo pass, you're good for the real upload.

## Real upload

```bash
cd /Users/stefi/Sandbox/Memory
python -m twine upload dist/*
# username: __token__
# password: <your PyPI token>
```

The project appears at `https://pypi.org/project/patha/` within ~1 minute.

## Smoke test from a fresh machine

```bash
uvx patha verify                     # should print env check + detectors
uvx patha demo                       # should run the belief walkthrough
uvx --from 'patha[mcp]' patha-mcp    # MCP server starts on stdio
uvx --from 'patha[viewer]' patha-viewer  # Streamlit viewer opens
```

Everyone can now install with:

```bash
uv pip install patha                      # core
uv pip install 'patha[mcp]'               # + MCP server
uv pip install 'patha[viewer]'            # + Streamlit viewer
uv pip install 'patha[mcp,viewer,dev]'    # everything
```

And configure Claude Desktop simply by:

```bash
uvx patha install-mcp --uvx -y
```

(The `--uvx` flag writes the config so Claude Desktop calls `uvx patha-mcp` instead of a local checkout — the user doesn't need to clone the repo at all.)

## Versioning

This release is `0.8.0`. Bump rules going forward:

- **Patch** (`0.8.x`) — bug fixes, doc changes, internal refactors.
- **Minor** (`0.x.0`) — new capabilities (new detectors, new tools, new plasticity mechanisms).
- **Major** (`1.0.0`) — first stable API, after 1–2 months of real-world MCP usage with no breaking changes.

Update `pyproject.toml`'s `version = "..."` before each release. Tag the commit (`git tag v0.8.0 && git push --tags`). Build again, upload again.

## Troubleshooting

- **"File already exists" on upload** — you can't overwrite an existing version on PyPI. Bump the version and rebuild. Use TestPyPI for iteration.
- **spaCy model missing** — the core package doesn't bundle spaCy models. Users run `uv run python -m spacy download en_core_web_sm` the first time they need NER. We document this in the README.
- **Big dependency install** — `torch` + `transformers` + `sentence-transformers` is ~2 GB. Unavoidable for NLI; the `stub` detector doesn't need any of it, but the transitive dep graph pulls them in. For future: consider splitting `patha-core` (no torch) and `patha[nli]` (with it).

## Post-publish checklist

- [ ] Tag the commit: `git tag v0.8.0 && git push --tags`
- [ ] Update README with the new `uvx patha install-mcp --uvx` one-liner
- [ ] Post a GitHub release at https://github.com/autotelicbydesign/patha/releases/new
- [ ] (Optional) Announce on the Anthropic Discord, r/LocalLLaMA, HN Show
