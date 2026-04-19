"""Patha CLI — command-line interface to the belief-layer memory system.

Commands:

    patha verify                # env + import check (do this first)
    patha demo                  # 10-sec demo with no model downloads
    patha ingest "I love sushi"
    patha ingest --file notes.txt
    patha ask "what do I currently believe about sushi?"
    patha history "sushi"
    patha stats
    patha mcp                   # run the MCP server (stdio)
    patha viewer                # launch the Streamlit viewer

State persists to ~/.patha/ by default. Override with --data-dir or
with the PATHA_STORE_PATH environment variable.

Detector selection (--detector):
    stub            — heuristic, no model download [default for CLI ops
                       because it's instant]
    full-stack-v7   — production detector: NLI + adhyāsa + numerical +
                       sequential (downloads ~1.7 GB on first run)
    nli, adhyasa-nli, full-stack — intermediate stacks for ablations

Run `patha --help` or `patha <cmd> --help` for full docs.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from patha.belief import (
    AVAILABLE_DETECTORS,
    BeliefLayer,
    BeliefStore,
    DirectAnswerer,
    describe_detector,
    make_detector,
)
from patha.integrated import IntegratedPatha


DEFAULT_DATA_DIR = Path(
    os.environ.get("PATHA_STORE_PATH", str(Path.home() / ".patha"))
)

MIN_PYTHON = (3, 11)


def _check_python_version() -> str | None:
    """Return an error message if Python version is too old, else None."""
    if sys.version_info < MIN_PYTHON:
        return (
            f"error: Patha requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+, "
            f"you have {sys.version_info[0]}.{sys.version_info[1]}.\n"
            f"       Install a newer Python (e.g. `uv python install 3.11`) "
            f"and rerun."
        )
    return None


def _default_session() -> str:
    """Session id = today's date (YYYY-MM-DD). Sessions bucket by day."""
    return datetime.now().strftime("%Y-%m-%d")


# ─── Layer construction ────────────────────────────────────────────

def _build_integrated(data_dir: Path, detector_name: str = "stub") -> IntegratedPatha:
    """Construct an IntegratedPatha with persistent state + named detector."""
    data_dir.mkdir(parents=True, exist_ok=True)
    store_path = data_dir / "beliefs.jsonl"
    layer = BeliefLayer(
        store=BeliefStore(persistence_path=store_path),
        detector=make_detector(detector_name),
    )
    answerer = DirectAnswerer(layer.store)
    return IntegratedPatha(
        phase1_retrieve=None,  # CLI uses belief store directly
        belief_layer=layer,
        direct_answerer=answerer,
    )


# ─── Commands ──────────────────────────────────────────────────────

def cmd_verify(args: argparse.Namespace) -> int:
    """Print environment + import sanity check. Run this first."""
    err = _check_python_version()
    if err:
        print(err, file=sys.stderr)
        return 1

    print("Patha environment check")
    print("─" * 40)
    print(f"Python version:   {sys.version_info.major}.{sys.version_info.minor}."
          f"{sys.version_info.micro} ✓")
    print(f"Store directory:  {args.data_dir}")
    print(f"  exists:         {args.data_dir.exists()}")

    # Try importing every major module
    import_checks = [
        ("patha.belief.layer", "BeliefLayer"),
        ("patha.belief.store", "BeliefStore"),
        ("patha.belief.contradiction", "StubContradictionDetector"),
        ("patha.belief.plasticity", "LongTermPotentiation"),
        ("patha.belief.sequential_detector", "SequentialEventDetector"),
        ("patha.belief.counterfactual", "reingest_order_sensitivity"),
        ("patha.integrated", "IntegratedPatha"),
    ]
    for mod, attr in import_checks:
        try:
            m = __import__(mod, fromlist=[attr])
            getattr(m, attr)
            print(f"  {mod:40s} ✓")
        except Exception as e:
            print(f"  {mod:40s} ✗  ({e})")
            return 1

    print()
    print("Available detectors:")
    for name in AVAILABLE_DETECTORS:
        print(f"  {name:18s} {describe_detector(name)}")

    # Try instantiating the requested detector (optionally preload weights)
    print()
    print(f"Selected detector: {args.detector}")
    try:
        det = make_detector(args.detector)
        print(f"  instantiated:     {type(det).__name__} ✓")
        if args.preload and args.detector != "stub":
            print("  preloading model weights (this can take several minutes)…")
            # Run a tiny inference to force lazy-loading
            _ = det.detect("hello", "world")
            print("  preload:          done ✓")
    except Exception as e:
        print(f"  instantiation:    ✗  ({e})")
        return 1

    print()
    print("All checks passed. You're good to go.")
    print()
    print("Try:")
    print("  patha demo                     # 10-second demo")
    print("  patha ingest 'I love sushi'")
    print("  patha ask 'what do I believe?'")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the end-to-end belief-layer demo (stub detector, no downloads)."""
    from patha.demo import demo
    demo(use_nli=args.nli)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    """Run the Patha MCP server on stdio."""
    try:
        from patha.mcp_server import main as mcp_main
    except ImportError as e:
        print(
            f"error: MCP server requires the 'mcp' extra.\n"
            f"       Install with: uv pip install 'patha[mcp]'\n"
            f"       ({e})",
            file=sys.stderr,
        )
        return 1
    return mcp_main()


def cmd_viewer(args: argparse.Namespace) -> int:
    """Launch the Streamlit belief viewer."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "error: Streamlit viewer requires the 'viewer' extra.\n"
            "       Install with: uv pip install 'patha[viewer]'",
            file=sys.stderr,
        )
        return 1
    import subprocess
    app_path = Path(__file__).parent / "viewer" / "app.py"
    cmd = ["streamlit", "run", str(app_path)]
    if args.data_dir:
        cmd.extend(["--", "--data-dir", str(args.data_dir)])
    return subprocess.call(cmd)


def cmd_ingest(args: argparse.Namespace) -> int:
    patha = _build_integrated(args.data_dir, args.detector)
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
    patha = _build_integrated(args.data_dir, args.detector)
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
    patha = _build_integrated(args.data_dir, args.detector)
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
    patha = _build_integrated(args.data_dir, args.detector)
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
    # Python version guard runs before argparse so errors are helpful.
    err = _check_python_version()
    if err:
        print(err, file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(
        prog="patha",
        description="Patha — local-first AI memory with supersession and plasticity",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Where to persist belief state (default: {DEFAULT_DATA_DIR}; "
             "overridable via PATHA_STORE_PATH env var)",
    )
    parser.add_argument(
        "--detector",
        choices=AVAILABLE_DETECTORS,
        default="stub",
        help="Contradiction detector (default: stub). Use 'full-stack-v7' "
             "for production behavior; downloads ~1.7 GB on first run.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # verify
    p_verify = sub.add_parser(
        "verify", help="Check environment + imports (run this first)",
    )
    p_verify.add_argument(
        "--preload", action="store_true",
        help="Also preload the selected detector's model weights "
             "(triggers the ~1.7 GB NLI download for full-stack-v7).",
    )
    p_verify.set_defaults(fn=cmd_verify)

    # demo
    p_demo = sub.add_parser(
        "demo", help="Run the end-to-end belief-layer demo (no downloads)",
    )
    p_demo.add_argument(
        "--nli", action="store_true",
        help="Use real NLI model (downloads ~1.7 GB on first run)",
    )
    p_demo.set_defaults(fn=cmd_demo)

    # mcp
    p_mcp = sub.add_parser(
        "mcp", help="Run the Patha MCP server (stdio transport)",
    )
    p_mcp.set_defaults(fn=cmd_mcp)

    # viewer
    p_viewer = sub.add_parser(
        "viewer", help="Launch the Streamlit belief viewer in a browser",
    )
    p_viewer.set_defaults(fn=cmd_viewer)

    # ingest
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

    # ask
    p_ask = sub.add_parser("ask", help="Query the belief state")
    p_ask.add_argument("text", nargs="+", help="Question text")
    p_ask.add_argument(
        "--history", action="store_true",
        help="Include supersession lineage in the response",
    )
    p_ask.set_defaults(fn=cmd_ask)

    # history
    p_hist = sub.add_parser(
        "history", help="Show all beliefs mentioning a term",
    )
    p_hist.add_argument("text", nargs="+", help="Search term")
    p_hist.set_defaults(fn=cmd_history)

    # stats
    p_stats = sub.add_parser("stats", help="Show store statistics")
    p_stats.set_defaults(fn=cmd_stats)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
