"""Patha CLI — minimal command-line interface to the integrated system.

    patha ingest "I love sushi"              # single proposition
    patha ingest --file notes.txt            # file (one per line)
    patha ask "what do I currently believe about sushi?"
    patha history "sushi"                    # current + supersession chain

State persists to ~/.patha/ by default. Override with --data-dir.

This CLI is deliberately minimal — a starting point for using Patha as
a personal memory system, not a production tool. It demonstrates that
the Phase 2 belief layer can be wired to everyday ingestion and
querying without writing custom Python.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from patha.belief import (
    BeliefLayer,
    BeliefStore,
    DirectAnswerer,
    StubContradictionDetector,
)
from patha.integrated import IntegratedPatha


DEFAULT_DATA_DIR = Path.home() / ".patha"


def _default_session() -> str:
    """Session id = today's date (YYYY-MM-DD). Sessions bucket by day."""
    return datetime.now().strftime("%Y-%m-%d")


# ─── Layer construction ────────────────────────────────────────────

def _build_integrated(data_dir: Path) -> IntegratedPatha:
    """Construct an IntegratedPatha with persistent state.

    No NLI model is loaded by default — the CLI is meant to be fast.
    Use the Python API to wire a real detector when you need
    contradiction handling. The CLI still supports lookup queries,
    validity filtering, and non-destructive supersession via the
    StubContradictionDetector heuristic (which fires on clear
    asymmetric-negation pairs).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    store_path = data_dir / "beliefs.jsonl"
    layer = BeliefLayer(
        store=BeliefStore(persistence_path=store_path),
        detector=StubContradictionDetector(),
    )
    answerer = DirectAnswerer(layer.store)
    return IntegratedPatha(
        phase1_retrieve=None,  # CLI uses belief store directly
        belief_layer=layer,
        direct_answerer=answerer,
    )


# ─── Commands ──────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> int:
    patha = _build_integrated(args.data_dir)
    session = args.session or _default_session()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 1
        lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    else:
        lines = [" ".join(args.text)] if args.text else []

    if not lines:
        print("error: nothing to ingest (pass text or --file)", file=sys.stderr)
        return 1

    for i, line in enumerate(lines):
        ev = patha.ingest(
            proposition=line,
            asserted_at=datetime.now(),
            asserted_in_session=session,
            source_proposition_id=f"cli-{session}-{i}",
        )
        marker = {
            "added": "+",
            "reinforced": "~",
            "superseded": "!",
        }.get(ev.action, "?")
        print(f"{marker} [{ev.action}] {line}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    patha = _build_integrated(args.data_dir)
    question = " ".join(args.text)
    if not question:
        print("error: no question given", file=sys.stderr)
        return 1

    response = patha.query(
        question,
        at_time=datetime.now(),
        include_history=args.history,
    )

    print(f"[strategy: {response.strategy}]")
    if response.strategy == "direct_answer":
        print()
        print(response.answer)
    else:
        print(
            f"(would send {response.tokens_in} tokens to an LLM — no LLM "
            "wired in CLI mode; use the Python API to generate a final answer)"
        )
        print()
        print(response.prompt)
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    patha = _build_integrated(args.data_dir)
    term = " ".join(args.text).lower()
    if not term:
        print("error: no search term", file=sys.stderr)
        return 1

    matching = [
        b for b in patha.belief_layer.store.all()
        if term in b.proposition.lower()
    ]
    if not matching:
        print(f"no beliefs mention {term!r}")
        return 0

    matching.sort(key=lambda b: b.asserted_at)
    for b in matching:
        status = b.status.value
        date = b.asserted_at.strftime("%Y-%m-%d")
        pramana = b.pramana.value
        print(
            f"[{date}] [{status:11s}] [{pramana:10s}] conf={b.confidence:.2f} "
            f"{b.proposition}"
        )
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    patha = _build_integrated(args.data_dir)
    store = patha.belief_layer.store
    print(f"Data dir:      {args.data_dir}")
    print(f"Total beliefs: {len(store)}")
    print(f"  current:     {len(store.current())}")
    print(f"  superseded:  {len(store.superseded())}")
    print(f"  disputed:    {len(store.disputed())}")
    print(f"  archived:    {len(store.archived())}")
    return 0


# ─── Main ──────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="patha",
        description="Minimal CLI for the Patha belief layer",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Where to persist belief state (default: {DEFAULT_DATA_DIR})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser(
        "ingest", help="Add a proposition (or a file of propositions)"
    )
    p_ingest.add_argument(
        "text", nargs="*", help="Proposition text (or use --file)"
    )
    p_ingest.add_argument(
        "--file", help="Ingest one proposition per line from this file",
    )
    p_ingest.add_argument(
        "--session",
        help="Session id to attribute this ingest to (default: today's date)",
    )
    p_ingest.set_defaults(fn=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Query the belief state")
    p_ask.add_argument("text", nargs="+", help="Question text")
    p_ask.add_argument(
        "--history", action="store_true",
        help="Include supersession lineage in the response",
    )
    p_ask.set_defaults(fn=cmd_ask)

    p_hist = sub.add_parser(
        "history", help="Show all beliefs mentioning a term",
    )
    p_hist.add_argument("text", nargs="+", help="Search term")
    p_hist.set_defaults(fn=cmd_history)

    p_stats = sub.add_parser("stats", help="Show store statistics")
    p_stats.set_defaults(fn=cmd_stats)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
