.PHONY: test eval eval-quick eval-32 lint typecheck clean setup-data \
        verify demo mcp mcp-install viewer \
        build publish-test publish

# ── Quickstart targets ───────────────────────────────────────────────

verify:
	uv run patha verify

demo:
	uv run patha demo

mcp:
	uv run patha-mcp

mcp-install:
	uv run patha install-mcp

mcp-install-dry:
	uv run patha install-mcp --dry-run

mcp-install-code:
	uv run patha install-mcp --client claude-code

# ── Release ──────────────────────────────────────────────────────────

# Build wheel + sdist at the current pyproject.toml version.
build:
	@rm -rf dist/
	@uv build
	@echo ""
	@echo "Built:"
	@ls dist/
	@echo ""
	@echo "Next: TEST_PYPI_TOKEN=<your-token> make publish-test"
	@echo "  or: PYPI_TOKEN=<your-token> make publish"

# Rehearse the upload on TestPyPI. Requires TEST_PYPI_TOKEN env var
# (get one from https://test.pypi.org/manage/account/token/).
publish-test: build
	@if [ -z "$$TEST_PYPI_TOKEN" ]; then \
	    echo "error: set TEST_PYPI_TOKEN (from https://test.pypi.org/manage/account/token/)"; \
	    exit 1; \
	fi
	@uv pip install twine
	@TWINE_USERNAME=__token__ TWINE_PASSWORD=$$TEST_PYPI_TOKEN \
	    uv run python -m twine upload --repository testpypi dist/*
	@echo ""
	@echo "TestPyPI upload done. Smoke-test with:"
	@echo "  pip install --index-url https://test.pypi.org/simple/ \\"
	@echo "              --extra-index-url https://pypi.org/simple/ patha"

# Publish to real PyPI. Requires PYPI_TOKEN env var (get one from
# https://pypi.org/manage/account/token/). This is irreversible — you
# cannot unpublish a version, only yank it.
publish: build
	@if [ -z "$$PYPI_TOKEN" ]; then \
	    echo "error: set PYPI_TOKEN (from https://pypi.org/manage/account/token/)"; \
	    exit 1; \
	fi
	@echo "About to publish $$(ls dist/*.whl) to https://pypi.org/project/patha/"
	@echo "Ctrl-C in 5s to abort..."
	@sleep 5
	@uv pip install twine
	@TWINE_USERNAME=__token__ TWINE_PASSWORD=$$PYPI_TOKEN \
	    uv run python -m twine upload dist/*
	@echo ""
	@echo "Published. Verify at https://pypi.org/project/patha/"
	@echo "Smoke-test: pip install patha  (may take ~1 min to propagate)"

viewer:
	uv run patha viewer

# ── Setup ────────────────────────────────────────────────────────────

setup-data:
	@mkdir -p data
	@echo "Download LongMemEval S dataset from:"
	@echo "  https://github.com/xiaowu0162/long-mem-eval"
	@echo ""
	@echo "Place the cleaned JSON at: data/longmemeval_s_cleaned.json"
	@echo ""
	@echo "Then download the spaCy model:"
	@echo "  uv run python -m spacy download en_core_web_sm"
	@test -f data/longmemeval_s_cleaned.json || (echo "ERROR: data/longmemeval_s_cleaned.json not found" && exit 1)

# ── Testing ──────────────────────────────────────────────────────────

test:
	uv run python -m pytest tests/ -q

test-verbose:
	uv run python -m pytest tests/ -v

# ── Evaluation ───────────────────────────────────────────────────────

# Smoke test: 10 questions, fast, no reranker
eval-quick:
	uv run python -m eval.runner \
		--data data/longmemeval_s_cleaned.json \
		--embedder minilm \
		--limit 10 \
		--output runs/quick/results.json \
		--verbose

# 32-question diverse sample with cross-encoder
eval-32:
	uv run python -m eval.runner \
		--data data/longmemeval_s_cleaned.json \
		--embedder minilm \
		--reranker ce-mini \
		--limit 32 \
		--output runs/eval_32/results.json \
		--verbose

# 100-question stratified sample with cross-encoder
eval-100:
	uv run python -m eval.runner \
		--data /tmp/patha_stratified_100.json \
		--embedder minilm \
		--reranker ce-mini \
		--output runs/eval_100/results.json \
		--verbose

# LongMemEval-KU 78-question subset — the head-to-head vs Mem0
eval-ku:
	uv run python -m eval.runner \
		--data data/longmemeval_ku_78.json \
		--embedder minilm \
		--reranker ce-mini \
		--output runs/eval_ku/results.json \
		--verbose

# Full 500-question eval with pre-warming + cross-encoder
eval:
	uv run python -m eval.runner \
		--data data/longmemeval_s_cleaned.json \
		--embedder minilm \
		--reranker ce-mini \
		--pre-warm \
		--output runs/full/results.json \
		--verbose

# ── Ablations ────────────────────────────────────────────────────────

ablation-no-reranker:
	uv run python -m eval.runner \
		--data /tmp/patha_stratified_100.json \
		--embedder minilm \
		--reranker none \
		--output runs/ablation_no_reranker/results.json \
		--verbose

ablation-no-songline:
	uv run python -m eval.runner \
		--data /tmp/patha_stratified_100.json \
		--embedder minilm \
		--reranker ce-mini \
		--no-songline \
		--output runs/ablation_no_songline/results.json \
		--verbose

# ── Code Quality ─────────────────────────────────────────────────────

lint:
	uv run ruff check src/ eval/ tests/

lint-fix:
	uv run ruff check --fix src/ eval/ tests/

typecheck:
	uv run mypy src/patha/ --ignore-missing-imports

# ── Cleanup ──────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache
