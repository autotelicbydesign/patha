"""Tests for the filesystem importers (Innovation #3).

Patha gets pre-existing writing — Obsidian vaults, plain Markdown
folders, scattered .txt files — into the belief store. We test:

  1. Single-file import (text + Markdown).
  2. Frontmatter parsing (YAML happy path; minimal regex fallback).
  3. Long-file splitting (H1/H2 boundaries → multiple beliefs).
  4. Recursive folder walk (skips hidden dirs and non-text files).
  5. Wikilink + tag hint extraction (used by gaṇita extractor downstream).
  6. CLI wiring (`patha import obsidian-vault <path>`).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

import patha
from patha.cli import main as cli_main
from patha.importers import (
    ImportStats,
    _parse_frontmatter,
    _split_frontmatter,
    _split_into_propositions,
    extract_entity_hints_from_obsidian,
    import_file,
    import_folder,
    import_obsidian_vault,
)


# ─── Frontmatter parsing ──────────────────────────────────────────────


class TestFrontmatter:
    def test_no_frontmatter(self) -> None:
        fm, body = _split_frontmatter("# Hello\n\nbody text")
        assert fm == {}
        assert body == "# Hello\n\nbody text"

    def test_yaml_frontmatter(self) -> None:
        text = "---\ntitle: My Note\ndate: 2024-01-15\n---\n# Body\n\ncontent"
        fm, body = _split_frontmatter(text)
        assert fm["title"] == "My Note"
        # date may be string or datetime depending on yaml availability
        assert "2024" in str(fm["date"])
        assert body.startswith("# Body")

    def test_regex_fallback(self) -> None:
        # Pure regex, no YAML required.
        fm = _parse_frontmatter('title: "Quoted Title"\ntags: foo, bar')
        assert fm["title"] == "Quoted Title"


# ─── Splitting ───────────────────────────────────────────────────────


class TestSplitting:
    def test_short_text_one_proposition(self) -> None:
        out = _split_into_propositions("just a single short paragraph")
        assert out == ["just a single short paragraph"]

    def test_long_text_splits_on_headings(self) -> None:
        text = (
            "# First chapter\n" + "x" * 1100 + "\n\n"
            + "# Second chapter\n" + "y" * 1100
        )
        out = _split_into_propositions(text, max_chars=2000)
        assert len(out) == 2
        assert out[0].startswith("# First chapter")
        assert out[1].startswith("# Second chapter")

    def test_empty_text(self) -> None:
        assert _split_into_propositions("") == []
        assert _split_into_propositions("   \n\n  \n") == []


# ─── Single-file ingest ──────────────────────────────────────────────


class TestImportFile:
    def test_plain_markdown(self, tmp_path: Path) -> None:
        f = tmp_path / "note.md"
        f.write_text("# Lisbon\n\nI moved to Lisbon last month.")
        mem = patha.Memory(
            path=tmp_path / "store.jsonl", enable_phase1=False,
        )
        stats = import_file(f, mem)
        assert stats.files_imported == 1
        assert stats.beliefs_added == 1
        assert mem.stats()["total"] == 1

    def test_obsidian_with_frontmatter(self, tmp_path: Path) -> None:
        f = tmp_path / "rent.md"
        f.write_text(
            "---\ndate: 2024-01-15\n---\n"
            "Apartment rent is 1500 EUR per month."
        )
        mem = patha.Memory(
            path=tmp_path / "store.jsonl", enable_phase1=False,
        )
        stats = import_file(f, mem, obsidian=True)
        assert stats.beliefs_added == 1
        # asserted_at should match the frontmatter date
        beliefs = mem._patha.belief_layer.store.all()
        assert beliefs[0].asserted_at.year == 2024
        assert beliefs[0].asserted_at.month == 1

    def test_skips_non_text_files(self, tmp_path: Path) -> None:
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG fake")
        mem = patha.Memory(path=tmp_path / "store.jsonl", enable_phase1=False)
        stats = import_file(f, mem)
        assert stats.files_imported == 0
        assert stats.files_skipped == 1


# ─── Folder walk ─────────────────────────────────────────────────────


class TestImportFolder:
    def test_recursive_walk(self, tmp_path: Path) -> None:
        (tmp_path / "vault").mkdir()
        (tmp_path / "vault" / "top.md").write_text("Top level note.")
        (tmp_path / "vault" / "subdir").mkdir()
        (tmp_path / "vault" / "subdir" / "nested.md").write_text(
            "Nested note."
        )
        # Hidden dir should be skipped entirely
        (tmp_path / "vault" / ".obsidian").mkdir()
        (tmp_path / "vault" / ".obsidian" / "config").write_text(
            "shouldn't be ingested"
        )
        mem = patha.Memory(
            path=tmp_path / "store.jsonl", enable_phase1=False,
        )
        stats = import_folder(tmp_path / "vault", mem)
        assert stats.files_imported == 2
        assert stats.beliefs_added == 2

    def test_obsidian_wrapper(self, tmp_path: Path) -> None:
        (tmp_path / "v").mkdir()
        (tmp_path / "v" / "a.md").write_text(
            "---\ndate: 2024-03-10\n---\nMy [[bike]] expense was $50."
        )
        mem = patha.Memory(
            path=tmp_path / "store.jsonl", enable_phase1=False,
        )
        stats = import_obsidian_vault(tmp_path / "v", mem)
        assert stats.files_imported == 1
        beliefs = mem._patha.belief_layer.store.all()
        assert beliefs[0].asserted_at.year == 2024


# ─── Wikilink / tag extraction ───────────────────────────────────────


class TestEntityHints:
    def test_wikilinks(self) -> None:
        out = extract_entity_hints_from_obsidian(
            "I rode my [[bike]] to [[work|the office]] yesterday."
        )
        assert "bike" in out
        assert "work" in out

    def test_tags(self) -> None:
        out = extract_entity_hints_from_obsidian(
            "Today I did some #yoga and #cycling."
        )
        assert "yoga" in out
        assert "cycling" in out

    def test_dedup_and_order(self) -> None:
        out = extract_entity_hints_from_obsidian(
            "[[bike]] [[BIKE]] #bike yet [[helmet]]"
        )
        # Lowercase + dedup
        assert out.count("bike") == 1
        # Order of first appearance preserved
        assert out.index("bike") < out.index("helmet")


# ─── CLI wiring ──────────────────────────────────────────────────────


class TestCliImport:
    def test_import_file_via_cli(
        self, tmp_path: Path, capsys
    ) -> None:
        f = tmp_path / "note.md"
        f.write_text("My favourite color is blue.")
        rc = cli_main([
            "--data-dir", str(tmp_path / "store"),
            "import", "file", str(f),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "files imported" in out
        # The store file should now exist
        assert (tmp_path / "store" / "beliefs.jsonl").exists()

    def test_obsidian_vault_via_cli(
        self, tmp_path: Path, capsys
    ) -> None:
        v = tmp_path / "vault"
        v.mkdir()
        (v / "a.md").write_text(
            "---\ndate: 2024-05-01\n---\nI am vegetarian."
        )
        (v / "b.md").write_text("I live in [[Lisbon]].")
        rc = cli_main([
            "--data-dir", str(tmp_path / "store"),
            "import", "obsidian-vault", str(v),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "files imported:       2" in out

    def test_nonexistent_path_errors(self, tmp_path: Path, capsys) -> None:
        rc = cli_main([
            "--data-dir", str(tmp_path / "store"),
            "import", "file", str(tmp_path / "nope.md"),
        ])
        assert rc == 1
