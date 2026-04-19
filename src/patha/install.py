"""`patha install-mcp` — set up the Claude Desktop (or Claude Code) MCP
config to point at this checkout of Patha.

Handles:
  - Detecting the right config path for the current OS
  - Finding the absolute path of this Patha checkout
  - Safely merging into existing MCP config (preserves other servers)
  - Backing up the old config before writing
  - Choosing between `uvx patha-memory` (once on pypi) and
    `uv run --project <path> patha-mcp` (local checkout)

Never overwrites other `mcpServers` entries; only replaces the `patha` one.
Never writes without asking for confirmation.

Portable across macOS, Linux, and Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path


# ─── Config-file locations ──────────────────────────────────────────

def _claude_desktop_config_path() -> Path:
    """Return the standard Claude Desktop config path for this OS."""
    system = platform.system()
    if system == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
        return (
            Path.home()
            / "AppData"
            / "Roaming"
            / "Claude"
            / "claude_desktop_config.json"
        )
    # Linux and BSDs — Claude Desktop doesn't have an official Linux build
    # but users commonly symlink here.
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def _claude_code_config_path() -> Path:
    """Return the Claude Code config path (same on every OS)."""
    return Path.home() / ".claude" / "config.json"


# Keyed by client name. Values are (label, lazy-path-fn) so tests and
# callers can patch the individual path functions without racing
# against module-import time.
CLIENTS: dict[str, tuple[str, "callable[[], Path]"]] = {
    "claude-desktop": ("Claude Desktop", _claude_desktop_config_path),
    "claude-code": ("Claude Code", _claude_code_config_path),
}


# ─── Patha path discovery ───────────────────────────────────────────

def _patha_checkout_path() -> Path:
    """Walk up from this file to find the project root (pyproject.toml)."""
    p = Path(__file__).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Fallback: assume current working directory
    return Path.cwd()


# ─── Config construction ────────────────────────────────────────────

def build_patha_server_entry(
    *,
    use_uvx: bool,
    project_path: Path,
    store_path: Path,
    detector: str,
) -> dict:
    """Return the JSON dict for the `patha` entry under mcpServers."""
    if use_uvx:
        return {
            "command": "uvx",
            "args": ["patha-memory", "patha-mcp"],
            "env": {
                "PATHA_STORE_PATH": str(store_path),
                "PATHA_DETECTOR": detector,
            },
        }
    return {
        "command": "uv",
        "args": ["run", "--project", str(project_path), "patha-mcp"],
        "env": {
            "PATHA_STORE_PATH": str(store_path),
            "PATHA_DETECTOR": detector,
        },
    }


def merge_into_config(
    existing: dict | None, patha_entry: dict,
) -> dict:
    """Return a config dict with patha merged under mcpServers, preserving
    any other existing servers."""
    config = dict(existing) if existing else {}
    servers = dict(config.get("mcpServers") or {})
    servers["patha"] = patha_entry
    config["mcpServers"] = servers
    return config


# ─── File I/O with backup ───────────────────────────────────────────

def _backup(path: Path) -> Path | None:
    """Copy path to path.bak.<timestamp> if it exists. Return backup path."""
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, backup)
    return backup


def _read_json_lenient(path: Path) -> dict | None:
    """Read JSON; return None if missing, raise on malformed."""
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    return json.loads(text)


# ─── The main install flow ──────────────────────────────────────────

def install(
    *,
    client: str = "claude-desktop",
    use_uvx: bool = False,
    store_path: Path | None = None,
    detector: str = "stub",
    yes: bool = False,
    dry_run: bool = False,
) -> int:
    """Perform the install. Returns 0 on success, non-zero on error."""
    if client not in CLIENTS:
        print(
            f"error: unknown client {client!r}; choose from {list(CLIENTS.keys())}",
            file=sys.stderr,
        )
        return 2
    label, _ = CLIENTS[client]
    # Resolve the path by looking up the function on this module at call
    # time, so test patches like `patch("patha.install._claude_desktop_config_path")`
    # take effect. (Storing the function reference in CLIENTS at import
    # time would freeze it to the original, pre-patch version.)
    import patha.install as _self
    path_fn_name = (
        "_claude_desktop_config_path" if client == "claude-desktop"
        else "_claude_code_config_path"
    )
    config_path = getattr(_self, path_fn_name)()
    project_path = _patha_checkout_path()
    store = store_path or (Path.home() / ".patha")

    print(f"Configuring {label} MCP integration for Patha")
    print(f"  Config file:   {config_path}")
    print(f"  Patha path:    {project_path}")
    print(f"  Store path:    {store}")
    print(f"  Detector:      {detector}")
    print(f"  Install via:   {'uvx patha-memory' if use_uvx else 'uv run (local checkout)'}")
    print()

    # Read existing config if any
    try:
        existing = _read_json_lenient(config_path)
    except json.JSONDecodeError as e:
        print(
            f"error: {config_path} exists but isn't valid JSON: {e}\n"
            f"       Fix the file manually, then rerun.",
            file=sys.stderr,
        )
        return 1

    other_servers = list((existing or {}).get("mcpServers", {}).keys())
    other_servers = [s for s in other_servers if s != "patha"]
    if other_servers:
        print(f"  Existing MCP servers (will be preserved): {', '.join(other_servers)}")
    if existing and "patha" in (existing.get("mcpServers") or {}):
        print("  Note: an existing 'patha' entry will be replaced.")
    print()

    entry = build_patha_server_entry(
        use_uvx=use_uvx,
        project_path=project_path,
        store_path=store,
        detector=detector,
    )
    new_config = merge_into_config(existing, entry)

    print("New 'patha' server entry:")
    for line in json.dumps(entry, indent=2).split("\n"):
        print(f"  {line}")
    print()

    if dry_run:
        print("(dry run — not writing)")
        return 0

    if not yes:
        resp = input(f"Write {config_path}? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("aborted.")
            return 0

    # Backup + write
    config_path.parent.mkdir(parents=True, exist_ok=True)
    backup = _backup(config_path)
    if backup:
        print(f"  backed up existing config → {backup}")
    config_path.write_text(json.dumps(new_config, indent=2) + "\n")
    print(f"  wrote {config_path} ✓")
    print()
    print(f"Done. Next step: fully quit and reopen {label}, then look for the")
    print("'patha' entry in your tools panel / MCP servers list.")
    if detector != "stub":
        print()
        print("Note: you set detector =", repr(detector) + ".")
        print("On first ingest, Patha will download ~1.7 GB of NLI model weights.")
        print("To pre-download now:")
        print(f"  PATHA_DETECTOR={detector} uv run patha verify --preload")
    return 0


# ─── CLI entry (used by `patha install-mcp`) ────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="patha install-mcp",
        description=(
            "Install Patha as an MCP server for Claude Desktop or Claude Code. "
            "Detects OS, finds the config file, and merges the entry safely "
            "(existing MCP servers are preserved)."
        ),
    )
    ap.add_argument(
        "--client",
        choices=list(CLIENTS.keys()),
        default="claude-desktop",
        help="Which client to configure (default: claude-desktop).",
    )
    ap.add_argument(
        "--uvx", action="store_true",
        help="Use `uvx patha-memory` instead of a local checkout path. "
             "Requires Patha to be published on pypi.",
    )
    ap.add_argument(
        "--store-path", type=Path, default=None,
        help="Where to persist the belief store (default: ~/.patha).",
    )
    ap.add_argument(
        "--detector", default="stub",
        choices=["stub", "nli", "adhyasa-nli", "full-stack", "full-stack-v7"],
        help="Contradiction detector (default: stub — instant startup, no "
             "model downloads).",
    )
    ap.add_argument(
        "-y", "--yes", action="store_true",
        help="Don't prompt before writing the config.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be written, don't actually write.",
    )
    args = ap.parse_args(argv)

    return install(
        client=args.client,
        use_uvx=args.uvx,
        store_path=args.store_path,
        detector=args.detector,
        yes=args.yes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
