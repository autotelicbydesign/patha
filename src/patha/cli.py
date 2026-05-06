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
import patha as _patha_module  # re-imported to access Memory without circular


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

def _build_memory(data_dir: Path, detector_name: str = "stub"):
    """Construct the public Memory API with persistent state.

    Wires the same auto-enabled ganita synthesis layer + regex karaṇa
    extractor that `import patha; patha.Memory()` ships, so CLI usage
    matches the documented Python API. Phase 1 is disabled by default
    in CLI mode because the CLI is for one-fact-at-a-time interaction;
    set PATHA_PHASE1=on to opt in.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    store_path = data_dir / "beliefs.jsonl"
    enable_phase1 = os.environ.get("PATHA_PHASE1", "off").lower() == "on"
    return _patha_module.Memory(
        path=store_path,
        detector=detector_name,
        enable_phase1=enable_phase1,
        enable_ganita=True,
    )


def _build_integrated(data_dir: Path, detector_name: str = "stub") -> IntegratedPatha:
    """Lower-level IntegratedPatha. Retained for `cmd_history` / `cmd_stats`
    which need direct access to the belief store. Prefer `_build_memory` for
    user-facing query/ingest commands."""
    data_dir.mkdir(parents=True, exist_ok=True)
    store_path = data_dir / "beliefs.jsonl"
    layer = BeliefLayer(
        store=BeliefStore(persistence_path=store_path),
        detector=make_detector(detector_name),
    )
    answerer = DirectAnswerer(layer.store)
    return IntegratedPatha(
        phase1_retrieve=None,
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
        ("patha.belief.ganita", "GanitaIndex"),
        ("patha.belief.karana", "OllamaKaranaExtractor"),
        ("patha.importers", "import_obsidian_vault"),
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

    # Probe Ollama if it's reachable — gives users a clear yes/no on
    # whether Innovation #2 (karaṇa LLM extractor) is wired up.
    print()
    print("Optional services:")
    _probe_ollama()

    print()
    print("All checks passed. You're good to go.")
    print()
    print("Try:")
    print("  patha demo                     # 10-second demo")
    print("  patha ingest 'I love sushi'")
    print("  patha ask 'what do I believe?'")
    print("  patha import obsidian-vault ~/MyVault    # bring existing notes")
    return 0


def _probe_ollama() -> None:
    """Light reachability check for Ollama. Used by `patha verify` to
    let users know whether the karaṇa LLM extractor will work."""
    import json as _json
    import os as _os
    import urllib.error as _urle
    import urllib.request as _urlr

    host = _os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        req = _urlr.Request(f"{host.rstrip('/')}/api/tags", method="GET")
        with _urlr.urlopen(req, timeout=2.0) as r:
            body = r.read()
        data = _json.loads(body)
        models = data.get("models", [])
        names = [m.get("name", "?") for m in models]
        if not models:
            print(f"  ollama at {host}: reachable, no models pulled "
                  f"(`ollama pull qwen2.5:7b-instruct` to enable karaṇa)")
        else:
            shown = ", ".join(names[:3]) + (
                f" (+{len(names)-3} more)" if len(names) > 3 else ""
            )
            print(f"  ollama at {host}: ✓ ({shown})")
            print(f"    set PATHA_KARANA=ollama to enable LLM extraction at ingest")
    except (_urle.URLError, OSError, _json.JSONDecodeError):
        print(f"  ollama at {host}: not reachable "
              f"(karaṇa falls back to regex; that's fine)")


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


def cmd_install_mcp(args: argparse.Namespace) -> int:
    """Merge Patha into the Claude Desktop / Claude Code MCP config."""
    from patha.install import install
    karana_mode = getattr(args, "karana_mode", None)
    if karana_mode == "default":
        karana_mode = None
    hebbian = getattr(args, "hebbian", None)
    if hebbian == "default":
        hebbian = None
    elif hebbian == "on":
        hebbian = True
    elif hebbian == "off":
        hebbian = False
    # Resolve which detector to bake into the generated MCP config:
    #   1. Explicit --install-detector wins (the dedicated subcommand flag)
    #   2. Otherwise, take the global --detector / PATHA_DETECTOR env value
    #      (so `patha --detector full-stack-v8 install-mcp` and
    #      `PATHA_DETECTOR=full-stack-v8 patha install-mcp` both Just Work,
    #      which they did NOT in v0.10.7 — that was a real bug)
    install_detector = args.install_detector
    if install_detector == "stub" and args.detector != "stub":
        install_detector = args.detector
    return install(
        client=args.client,
        use_uvx=args.uvx,
        store_path=args.store_path,
        detector=install_detector,
        yes=args.yes,
        dry_run=args.dry_run,
        karana_mode=karana_mode,
        hebbian_expansion=hebbian,
    )


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
    memory = _build_memory(args.data_dir, args.detector)
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
        ev = memory.remember(
            line,
            asserted_at=datetime.now(),
            session_id=session,
            source_id=f"cli-{session}-{i}",
        )
        action = ev["action"] if isinstance(ev, dict) else getattr(ev, "action", "?")
        marker = {
            "added": "+",
            "reinforced": "~",
            "superseded": "!",
        }.get(action, "?")
        print(f"{marker} [{action}] {line}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    memory = _build_memory(args.data_dir, args.detector)
    question = " ".join(args.text)
    if not question:
        print("error: no question given", file=sys.stderr)
        return 1

    rec = memory.recall(question, include_history=args.history)

    print(f"[strategy: {rec.strategy}]")
    print(f"[tokens at recall: {rec.tokens}]")
    print()

    # Synthesis intent — gaṇita produced a deterministic answer with zero LLM tokens.
    if rec.ganita is not None:
        print(f"answer: {rec.ganita.value} {rec.ganita.unit or ''}".rstrip())
        print(f"  via:  {rec.ganita.operator} over {len(rec.ganita.contributing_belief_ids)} belief(s)")
        if rec.ganita.explanation:
            print(f"  why:  {rec.ganita.explanation}")
        return 0

    # Direct-answer (belief layer found a single clear answer)
    if rec.strategy == "direct_answer" and rec.answer:
        print(f"answer: {rec.answer}")
        return 0

    # Structured summary — pipe to your LLM. The CLI doesn't call an LLM.
    print("summary:")
    print(rec.summary or "(no relevant beliefs)")
    print()
    print("(this is the structured summary Patha emits; pipe it into")
    print(" your LLM as system context to get a natural-language answer.")
    print(" Use the Python API: `memory.recall(...).summary`.)")
    return 0


def cmd_shell(args: argparse.Namespace) -> int:
    """Interactive REPL — type sentences to remember, prefix `?` to ask.

    Removes the `patha ingest "..."` boilerplate. Type a statement to
    remember it. Prefix with `?` to query. Type `exit` or hit Ctrl-D
    to quit.
    """
    memory = _build_memory(args.data_dir, args.detector)
    print("Patha shell. Type a sentence to remember, prefix `?` to ask.")
    print("Examples:")
    print("    patha> I am vegetarian")
    print("    patha> ? what do I eat?")
    print("Type `exit` or press Ctrl-D to quit.")
    print()
    n_remembered = 0
    n_asked = 0
    while True:
        try:
            line = input("patha> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit", ":q"):
            break
        # Question — prefix with ? or ends with ?
        is_question = line.startswith("?") or line.rstrip().endswith("?")
        if is_question:
            question = line.lstrip("?").strip() or line.strip()
            try:
                rec = memory.recall(question)
            except Exception as e:
                print(f"  error: {e}")
                continue
            print(f"  [strategy: {rec.strategy}] [tokens: {rec.tokens}]")
            if rec.ganita is not None:
                unit = rec.ganita.unit or ""
                print(f"  → {rec.ganita.value} {unit}".rstrip())
                if rec.ganita.explanation:
                    print(f"     {rec.ganita.explanation}")
            elif rec.answer:
                print(f"  → {rec.answer}")
            else:
                summary = rec.summary or "(no relevant beliefs)"
                # Indent each line of the summary so the structure is obvious.
                for ln in summary.splitlines():
                    print(f"     {ln}")
            n_asked += 1
        else:
            try:
                ev = memory.remember(line)
            except Exception as e:
                print(f"  error: {e}")
                continue
            action = ev["action"] if isinstance(ev, dict) else getattr(ev, "action", "?")
            marker = {
                "added": "+",
                "reinforced": "~",
                "superseded": "!",
            }.get(action, "?")
            print(f"  {marker} [{action}]")
            n_remembered += 1
        print()
    if n_remembered or n_asked:
        print(f"(session: {n_remembered} remembered, {n_asked} asked)")
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


def cmd_import(args: argparse.Namespace) -> int:
    """Import files / folders / Obsidian vaults into the belief store."""
    import patha as patha_pkg
    from patha.importers import (
        import_file, import_folder, import_obsidian_vault, ImportStats,
    )

    target = Path(args.path).expanduser().resolve()
    if not target.exists():
        print(f"error: {target} does not exist", file=sys.stderr)
        return 1

    # Use the developer-API Memory, which routes through Phase 1 +
    # Phase 2 + gaṇita. Persists to the same store the rest of the
    # CLI uses (data_dir/beliefs.jsonl).
    args.data_dir.mkdir(parents=True, exist_ok=True)
    store_path = args.data_dir / "beliefs.jsonl"
    memory = patha_pkg.Memory(
        path=store_path,
        detector=args.detector,
        # Phase 1 is built lazily — fine for one-shot import.
        enable_phase1=False,
    )

    print(f"Importing {target} → {store_path}")
    if args.kind == "obsidian-vault":
        if target.is_dir():
            stats = import_obsidian_vault(target, memory)
        else:
            print(f"error: {target} is not a directory", file=sys.stderr)
            return 1
    elif args.kind == "folder":
        stats = import_folder(target, memory, obsidian=args.obsidian)
    elif args.kind == "file":
        stats = ImportStats()
        import_file(target, memory, obsidian=args.obsidian, stats=stats)
    elif args.kind == "claude-export":
        from patha.importers import import_claude_export
        stats = import_claude_export(
            target, memory,
            sentence_split=not args.whole_messages,
            verbose=args.verbose,
        )
    else:
        print(f"error: unknown import kind: {args.kind}", file=sys.stderr)
        return 1

    print(f"  files seen:           {stats.files_seen}")
    print(f"  files imported:       {stats.files_imported}")
    print(f"  files skipped:        {stats.files_skipped}")
    print(f"  beliefs added:        {stats.beliefs_added}")
    print(f"  beliefs reinforced:   {stats.beliefs_reinforced}")
    print(f"  beliefs superseded:   {stats.beliefs_superseded}")
    if stats.files_skipped and stats.files_skipped <= 5:
        for p in stats.skipped_paths[:5]:
            print(f"    skipped: {p}")
    elif stats.files_skipped:
        print(f"    (use --verbose to list skipped files)")
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
    from patha import __version__ as _patha_version
    parser.add_argument(
        "--version", action="version",
        version=f"patha {_patha_version}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Where to persist belief state (default: {DEFAULT_DATA_DIR}; "
             "overridable via PATHA_STORE_PATH env var)",
    )
    _default_detector = os.environ.get("PATHA_DETECTOR", "stub")
    if _default_detector not in AVAILABLE_DETECTORS:
        # Bad env-var value would otherwise crash with an opaque argparse
        # error. Fall back to stub and warn.
        print(
            f"warn: PATHA_DETECTOR={_default_detector!r} is not a known "
            f"detector ({', '.join(AVAILABLE_DETECTORS)}); falling back to 'stub'.",
            file=sys.stderr,
        )
        _default_detector = "stub"
    parser.add_argument(
        "--detector",
        choices=AVAILABLE_DETECTORS,
        default=_default_detector,
        help=f"Contradiction detector (default: {_default_detector}; "
             "set via --detector or PATHA_DETECTOR env var). Use "
             "'full-stack-v7' or 'full-stack-v8' for production behavior; "
             "downloads ~1.7 GB on first run.",
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

    # install-mcp — set up Claude Desktop / Claude Code config
    p_install = sub.add_parser(
        "install-mcp",
        help="Configure Claude Desktop (or Claude Code) to use Patha as an "
             "MCP server. Detects OS, finds the config file, merges safely.",
    )
    from patha.install import CLIENTS
    p_install.add_argument(
        "--client", choices=list(CLIENTS.keys()), default="claude-desktop",
        help="Which client to configure (default: claude-desktop).",
    )
    p_install.add_argument(
        "--uvx", action="store_true",
        help="Use `uvx patha-memory` instead of local checkout (pypi only).",
    )
    p_install.add_argument(
        "--store-path", type=Path, default=None,
        help="Where to persist the belief store (default: ~/.patha).",
    )
    p_install.add_argument(
        "--install-detector", default="stub",
        choices=AVAILABLE_DETECTORS,
        help="Contradiction detector to write into the config "
             "(default: stub — instant, no downloads).",
    )
    p_install.add_argument(
        "-y", "--yes", action="store_true",
        help="Don't prompt before writing the config.",
    )
    p_install.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be written, don't actually write.",
    )
    p_install.add_argument(
        "--karana-mode",
        choices=["default", "regex", "ollama", "off"],
        default="default",
        help="Bake PATHA_KARANA into the generated config. 'default' "
             "leaves it unset (server uses 'regex'). Use 'ollama' to "
             "enable Vedic karaṇa LLM ingest-time extraction; requires "
             "Ollama running with a small model pulled.",
    )
    p_install.add_argument(
        "--hebbian",
        choices=["default", "on", "off"],
        default="default",
        help="Bake PATHA_HEBBIAN into the generated config. 'default' "
             "leaves it unset (server uses 'on'). Use 'off' for "
             "ablation studies of Hebbian-cluster-aware retrieval.",
    )
    p_install.set_defaults(fn=cmd_install_mcp)

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

    # shell — interactive REPL
    p_shell = sub.add_parser(
        "shell",
        help="Interactive REPL — type sentences to remember, "
        "prefix `?` (or end with `?`) to ask questions. No more "
        "`patha ingest \"...\"` boilerplate.",
    )
    p_shell.set_defaults(fn=cmd_shell)

    # history
    p_hist = sub.add_parser(
        "history", help="Show all beliefs mentioning a term",
    )
    p_hist.add_argument("text", nargs="+", help="Search term")
    p_hist.set_defaults(fn=cmd_history)

    # stats
    p_stats = sub.add_parser("stats", help="Show store statistics")
    p_stats.set_defaults(fn=cmd_stats)

    # import
    p_import = sub.add_parser(
        "import",
        help="Import a file, folder, or Obsidian vault into the store. "
             "Each Markdown / text file becomes one or more beliefs.",
    )
    p_import.add_argument(
        "kind",
        choices=["file", "folder", "obsidian-vault", "claude-export"],
        help="What to import: a single file, a recursive folder, an "
             "Obsidian vault (frontmatter + wikilinks aware), or a Claude "
             "conversation export (.zip from claude.ai → Settings → "
             "Privacy → Export data).",
    )
    p_import.add_argument(
        "path",
        help="Path to the file / folder / vault / export-zip to import.",
    )
    p_import.add_argument(
        "--obsidian", action="store_true",
        help="When importing a folder/file, treat as Obsidian (parse "
             "YAML frontmatter, extract wikilink + tag entity hints). "
             "Implied by `obsidian-vault`.",
    )
    p_import.add_argument(
        "--whole-messages", action="store_true",
        help="(claude-export only) Ingest each user message as a single "
             "belief instead of sentence-splitting. Default is to split.",
    )
    p_import.add_argument(
        "--verbose", action="store_true",
        help="Print one line per imported conversation / file.",
    )
    p_import.set_defaults(fn=cmd_import)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
