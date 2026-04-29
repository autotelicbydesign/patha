"""Tests for `patha install-mcp` — the Claude Desktop/Code config helper.

Cover:
  - OS-specific config path detection (macOS, Windows, Linux)
  - Safe merging into existing configs (other MCP servers preserved)
  - Backup creation before overwriting
  - dry-run doesn't write
  - uvx vs local-checkout entry shapes
  - Malformed existing config → clear error, no write
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from patha.install import (
    CLIENTS,
    _claude_code_config_path,
    _claude_desktop_config_path,
    _patha_checkout_path,
    build_patha_server_entry,
    install,
    merge_into_config,
)


class TestConfigPathDiscovery:
    def test_macos_path(self):
        with patch("platform.system", return_value="Darwin"):
            p = _claude_desktop_config_path()
            assert "Library/Application Support/Claude" in str(p)
            assert p.name == "claude_desktop_config.json"

    def test_windows_path_with_appdata(self):
        with patch("platform.system", return_value="Windows"), \
             patch.dict("os.environ", {"APPDATA": "C:/Users/X/AppData/Roaming"}, clear=False):
            p = _claude_desktop_config_path()
            assert "Claude" in str(p)
            assert p.name == "claude_desktop_config.json"

    def test_windows_path_without_appdata(self, monkeypatch):
        monkeypatch.delenv("APPDATA", raising=False)
        with patch("platform.system", return_value="Windows"):
            p = _claude_desktop_config_path()
            assert "AppData" in str(p) or "Claude" in str(p)

    def test_linux_path_with_xdg(self, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", "/home/user/.config")
        with patch("platform.system", return_value="Linux"):
            p = _claude_desktop_config_path()
            assert str(p).startswith("/home/user/.config/Claude")

    def test_linux_path_without_xdg(self, monkeypatch):
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        with patch("platform.system", return_value="Linux"):
            p = _claude_desktop_config_path()
            assert ".config/Claude" in str(p)

    def test_claude_code_path_consistent(self):
        """Claude Code uses ~/.claude/config.json on every OS."""
        p = _claude_code_config_path()
        assert p.name == "config.json"
        assert ".claude" in str(p)

    def test_both_clients_registered(self):
        assert "claude-desktop" in CLIENTS
        assert "claude-code" in CLIENTS


class TestBuildEntry:
    def test_uvx_entry(self):
        entry = build_patha_server_entry(
            use_uvx=True,
            project_path=Path("/any/path"),
            store_path=Path("/home/user/.patha"),
            detector="stub",
        )
        assert entry["command"] == "uvx"
        assert entry["args"] == ["patha-memory", "patha-mcp"]
        assert entry["env"]["PATHA_STORE_PATH"] == "/home/user/.patha"
        assert entry["env"]["PATHA_DETECTOR"] == "stub"

    def test_local_checkout_entry(self):
        entry = build_patha_server_entry(
            use_uvx=False,
            project_path=Path("/Users/stefi/patha"),
            store_path=Path("/Users/stefi/.patha"),
            detector="full-stack-v7",
        )
        assert entry["command"] == "uv"
        assert "--project" in entry["args"]
        assert "/Users/stefi/patha" in entry["args"]
        assert entry["args"][-1] == "patha-mcp"
        assert entry["env"]["PATHA_DETECTOR"] == "full-stack-v7"

    def test_default_omits_innovation_env_vars(self):
        """By default, PATHA_KARANA and PATHA_HEBBIAN aren't baked in;
        the server uses its own defaults."""
        entry = build_patha_server_entry(
            use_uvx=False,
            project_path=Path("/p"),
            store_path=Path("/s"),
            detector="stub",
        )
        assert "PATHA_KARANA" not in entry["env"]
        assert "PATHA_HEBBIAN" not in entry["env"]

    def test_explicit_karana_ollama_baked_in(self):
        entry = build_patha_server_entry(
            use_uvx=False,
            project_path=Path("/p"),
            store_path=Path("/s"),
            detector="stub",
            karana_mode="ollama",
        )
        assert entry["env"]["PATHA_KARANA"] == "ollama"

    def test_explicit_hebbian_off_baked_in(self):
        entry = build_patha_server_entry(
            use_uvx=False,
            project_path=Path("/p"),
            store_path=Path("/s"),
            detector="stub",
            hebbian_expansion=False,
        )
        assert entry["env"]["PATHA_HEBBIAN"] == "off"

    def test_explicit_hebbian_on_baked_in(self):
        entry = build_patha_server_entry(
            use_uvx=False,
            project_path=Path("/p"),
            store_path=Path("/s"),
            detector="stub",
            hebbian_expansion=True,
        )
        assert entry["env"]["PATHA_HEBBIAN"] == "on"


class TestMergeIntoConfig:
    def test_empty_config(self):
        patha_entry = {"command": "uv", "args": ["run", "patha-mcp"]}
        result = merge_into_config(None, patha_entry)
        assert result == {"mcpServers": {"patha": patha_entry}}

    def test_preserves_other_servers(self):
        existing = {
            "mcpServers": {
                "github": {"command": "gh-mcp"},
                "slack": {"command": "slack-mcp"},
            }
        }
        patha_entry = {"command": "uv", "args": ["run", "patha-mcp"]}
        result = merge_into_config(existing, patha_entry)
        servers = result["mcpServers"]
        assert "github" in servers
        assert "slack" in servers
        assert servers["patha"] == patha_entry

    def test_replaces_existing_patha(self):
        existing = {
            "mcpServers": {
                "patha": {"command": "OLD_COMMAND"},
            }
        }
        new_entry = {"command": "NEW_COMMAND"}
        result = merge_into_config(existing, new_entry)
        assert result["mcpServers"]["patha"]["command"] == "NEW_COMMAND"

    def test_preserves_non_mcp_keys(self):
        """Top-level keys other than mcpServers must survive the merge."""
        existing = {
            "theme": "dark",
            "mcpServers": {},
            "customField": {"a": 1},
        }
        result = merge_into_config(existing, {"command": "patha-mcp"})
        assert result["theme"] == "dark"
        assert result["customField"] == {"a": 1}


class TestInstallFlow:
    def test_dry_run_does_not_write(self, tmp_path):
        config_path = tmp_path / "nonexistent.json"
        with patch("patha.install._claude_desktop_config_path", return_value=config_path):
            exit_code = install(
                client="claude-desktop",
                store_path=tmp_path / ".patha",
                detector="stub",
                yes=True,
                dry_run=True,
            )
        assert exit_code == 0
        assert not config_path.exists()

    def test_creates_config_if_absent(self, tmp_path):
        config_path = tmp_path / "claude_desktop_config.json"
        with patch("patha.install._claude_desktop_config_path", return_value=config_path):
            exit_code = install(
                client="claude-desktop",
                store_path=tmp_path / ".patha",
                detector="stub",
                yes=True,
            )
        assert exit_code == 0
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert "patha" in data["mcpServers"]

    def test_merges_without_destroying(self, tmp_path):
        config_path = tmp_path / "claude_desktop_config.json"
        config_path.write_text(json.dumps({
            "mcpServers": {"github": {"command": "gh-mcp"}},
            "theme": "dark",
        }))
        with patch("patha.install._claude_desktop_config_path", return_value=config_path):
            install(
                client="claude-desktop",
                store_path=tmp_path / ".patha",
                detector="stub",
                yes=True,
            )
        data = json.loads(config_path.read_text())
        assert "github" in data["mcpServers"]
        assert "patha" in data["mcpServers"]
        assert data["theme"] == "dark"

    def test_creates_backup(self, tmp_path):
        config_path = tmp_path / "claude_desktop_config.json"
        original = {"mcpServers": {"x": {"command": "y"}}}
        config_path.write_text(json.dumps(original))
        with patch("patha.install._claude_desktop_config_path", return_value=config_path):
            install(
                client="claude-desktop",
                store_path=tmp_path / ".patha",
                detector="stub",
                yes=True,
            )
        backups = list(tmp_path.glob("*.bak.*"))
        assert len(backups) == 1
        restored = json.loads(backups[0].read_text())
        assert restored == original

    def test_malformed_json_errors_without_writing(self, tmp_path, capsys):
        config_path = tmp_path / "claude_desktop_config.json"
        config_path.write_text("{this is not json")
        with patch("patha.install._claude_desktop_config_path", return_value=config_path):
            exit_code = install(
                client="claude-desktop",
                store_path=tmp_path / ".patha",
                detector="stub",
                yes=True,
            )
        assert exit_code != 0
        # File should be unchanged (we didn't overwrite)
        assert config_path.read_text() == "{this is not json"

    def test_unknown_client_errors(self, tmp_path):
        exit_code = install(
            client="not-a-real-client",
            store_path=tmp_path / ".patha",
            detector="stub",
            yes=True,
        )
        assert exit_code != 0


class TestPathaCheckoutDiscovery:
    def test_finds_pyproject_in_ancestors(self):
        p = _patha_checkout_path()
        assert (p / "pyproject.toml").exists()
