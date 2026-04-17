"""Tests for the patha CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from patha.cli import main


@pytest.fixture
def cli_env(tmp_path: Path) -> Path:
    return tmp_path / "data"


class TestIngest:
    def test_ingest_single_line(self, cli_env: Path, capsys) -> None:
        rc = main([
            "--data-dir", str(cli_env),
            "ingest", "I love sushi",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "+" in out and "added" in out

    def test_ingest_persists_across_invocations(
        self, cli_env: Path, capsys
    ) -> None:
        main(["--data-dir", str(cli_env), "ingest", "I love sushi"])
        main(["--data-dir", str(cli_env), "ingest", "I moved to Sofia"])
        rc = main(["--data-dir", str(cli_env), "stats"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Total beliefs: 2" in out

    def test_ingest_file(self, cli_env: Path, tmp_path: Path, capsys) -> None:
        p = tmp_path / "notes.txt"
        p.write_text("I like coffee\nI moved to Sofia\n\nI work at Patha\n")
        rc = main([
            "--data-dir", str(cli_env),
            "ingest", "--file", str(p),
        ])
        assert rc == 0
        rc = main(["--data-dir", str(cli_env), "stats"])
        out = capsys.readouterr().out
        assert "Total beliefs: 3" in out

    def test_ingest_nothing_errors(self, cli_env: Path, capsys) -> None:
        rc = main(["--data-dir", str(cli_env), "ingest"])
        assert rc != 0
        err = capsys.readouterr().err
        assert "nothing to ingest" in err


class TestAsk:
    def test_ask_non_lookup_returns_structured(
        self, cli_env: Path, capsys
    ) -> None:
        main(["--data-dir", str(cli_env), "ingest", "I live in Sofia"])
        rc = main([
            "--data-dir", str(cli_env),
            "ask", "tell me about my life",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "strategy" in out

    def test_ask_lookup_returns_direct(
        self, cli_env: Path, capsys
    ) -> None:
        main(["--data-dir", str(cli_env), "ingest", "I live in Sofia"])
        rc = main([
            "--data-dir", str(cli_env),
            "ask", "Where do I currently live?",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "direct_answer" in out
        assert "Sofia" in out


class TestHistory:
    def test_history_shows_matching_beliefs(
        self, cli_env: Path, capsys
    ) -> None:
        main(["--data-dir", str(cli_env), "ingest", "I love sushi"])
        main(["--data-dir", str(cli_env), "ingest", "I love ramen"])
        main(["--data-dir", str(cli_env), "ingest", "the weather is nice"])
        capsys.readouterr()  # clear accumulated ingest output
        rc = main([
            "--data-dir", str(cli_env),
            "history", "love",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "sushi" in out
        assert "ramen" in out
        assert "weather" not in out

    def test_history_no_match(self, cli_env: Path, capsys) -> None:
        main(["--data-dir", str(cli_env), "ingest", "I love sushi"])
        rc = main([
            "--data-dir", str(cli_env),
            "history", "kayaking",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "no beliefs" in out


class TestStats:
    def test_stats_on_empty(self, cli_env: Path, capsys) -> None:
        rc = main(["--data-dir", str(cli_env), "stats"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Total beliefs: 0" in out
