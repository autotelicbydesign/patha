"""End-to-end MCP server test: drive it over stdio with real JSON-RPC
messages, exercising the full protocol handshake + each tool.

This is the closest thing to "verify it works in Claude Desktop" that
we can do without actually driving Claude Desktop.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _send(proc: subprocess.Popen, msg: dict) -> None:
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()


def _recv(proc: subprocess.Popen) -> dict:
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"server closed stdout. stderr: {proc.stderr.read()}")
    return json.loads(line)


@pytest.mark.slow
def test_mcp_full_roundtrip():
    """Spawn patha-mcp, do MCP initialize, call every tool, verify responses."""
    store_dir = tempfile.mkdtemp(prefix="patha-mcp-e2e-")
    env = os.environ.copy()
    env["PATHA_STORE_PATH"] = store_dir
    env["PATHA_DETECTOR"] = "stub"

    proc = subprocess.Popen(
        [sys.executable, "-m", "patha.mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, bufsize=1,
    )
    try:
        # Step 1: initialize
        _send(proc, {
            "jsonrpc": "2.0", "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.1"},
            },
        })
        resp = _recv(proc)
        assert resp["result"]["serverInfo"]["name"] == "patha"
        assert "tools" in resp["result"]["capabilities"]

        # Step 2: initialized notification (no response)
        _send(proc, {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })

        # Step 3: list tools
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        resp = _recv(proc)
        tool_names = {t["name"] for t in resp["result"]["tools"]}
        assert tool_names == {
            "patha_ingest", "patha_query", "patha_history", "patha_stats",
        }

        # Step 4: ingest
        _send(proc, {
            "jsonrpc": "2.0", "id": 3,
            "method": "tools/call",
            "params": {
                "name": "patha_ingest",
                "arguments": {"proposition": "I love sushi"},
            },
        })
        resp = _recv(proc)
        # tools/call returns a list of content items; the structured output
        # is in the first text item as JSON.
        content = resp["result"]["content"]
        payload = json.loads(content[0]["text"])
        assert payload["action"] == "added"

        # Step 5: another ingest
        _send(proc, {
            "jsonrpc": "2.0", "id": 4,
            "method": "tools/call",
            "params": {
                "name": "patha_ingest",
                "arguments": {
                    "proposition": "I avoid raw fish on my doctor's advice",
                },
            },
        })
        _recv(proc)  # ingest response, ignore details

        # Step 6: query
        _send(proc, {
            "jsonrpc": "2.0", "id": 5,
            "method": "tools/call",
            "params": {
                "name": "patha_query",
                "arguments": {"question": "what do I eat?"},
            },
        })
        resp = _recv(proc)
        payload = json.loads(resp["result"]["content"][0]["text"])
        assert "current" in payload
        assert payload["strategy"] in ("direct_answer", "structured", "raw")

        # Step 7: history
        _send(proc, {
            "jsonrpc": "2.0", "id": 6,
            "method": "tools/call",
            "params": {
                "name": "patha_history",
                "arguments": {"term": "sushi"},
            },
        })
        resp = _recv(proc)
        payload = json.loads(resp["result"]["content"][0]["text"])
        assert any("sushi" in m["proposition"].lower() for m in payload["matches"])

        # Step 8: stats
        _send(proc, {
            "jsonrpc": "2.0", "id": 7,
            "method": "tools/call",
            "params": {"name": "patha_stats", "arguments": {}},
        })
        resp = _recv(proc)
        payload = json.loads(resp["result"]["content"][0]["text"])
        assert payload["total_beliefs"] == 2
        assert payload["detector"] == "stub"

        # Step 9: verify persistence on disk
        store_file = Path(store_dir) / "beliefs.jsonl"
        assert store_file.exists()
        assert store_file.stat().st_size > 0

        # Step 10: list resources (v0.9 adds these for auto-context)
        _send(proc, {
            "jsonrpc": "2.0", "id": 8, "method": "resources/list",
        })
        resp = _recv(proc)
        resource_uris = {r["uri"] for r in resp["result"]["resources"]}
        assert "patha://beliefs/current" in resource_uris
        assert "patha://beliefs/all" in resource_uris
        assert "patha://stats" in resource_uris

        # Step 11: read the current-beliefs resource
        _send(proc, {
            "jsonrpc": "2.0", "id": 9,
            "method": "resources/read",
            "params": {"uri": "patha://beliefs/current"},
        })
        resp = _recv(proc)
        text = resp["result"]["contents"][0]["text"]
        # We ingested 2 beliefs above; both should be present
        assert "sushi" in text.lower()
        assert "raw fish" in text.lower()

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        import shutil
        shutil.rmtree(store_dir, ignore_errors=True)
