# -*- coding: utf-8 -*-
"""Which files count as a note, and how they are read.

These tests exist because of issue #1: the indexer walked `rglob('*.md')`, which
descends into dot-directories. On a real Obsidian vault synced with Syncthing that
pulled in `.stversions/` backups, `*.sync-conflict-*` copies, `.obsidian/` plugin
cache and `.git/` internals — and a stale copy of a note is semantically almost
identical to the live one, so it quietly ate a retrieval slot with outdated content.

No model download, no network: `iter_notes` and `read_note` are pure file work.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from index_notes import iter_notes, read_note  # noqa: E402


def _vault(tmp_path: Path) -> Path:
    """A directory shaped like a real synced Obsidian vault."""
    (tmp_path / "real.md").write_text("real note", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.md").write_text("nested real note", encoding="utf-8")

    (tmp_path / ".stversions").mkdir()
    (tmp_path / ".stversions" / "real~20260701.md").write_text("OLD", encoding="utf-8")
    (tmp_path / "real.sync-conflict-20260801-ABC.md").write_text("CONFLICT", encoding="utf-8")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "cache.md").write_text("plugin junk", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "COMMIT_EDITMSG.md").write_text("git junk", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not markdown", encoding="utf-8")
    return tmp_path


def test_only_real_notes_are_indexed(tmp_path):
    got = {p.name for p in iter_notes(_vault(tmp_path))}
    assert got == {"real.md", "nested.md"}


def test_stversions_backup_is_skipped(tmp_path):
    assert not any(".stversions" in p.parts for p in iter_notes(_vault(tmp_path)))


def test_sync_conflict_copy_is_skipped(tmp_path):
    assert not any("sync-conflict" in p.name for p in iter_notes(_vault(tmp_path)))


def test_dot_directories_are_skipped(tmp_path):
    found = iter_notes(_vault(tmp_path))
    assert not any(part.startswith(".") for p in found for part in p.parts[:-1])


def test_exclusions_are_overridable(tmp_path, monkeypatch):
    """Someone will legitimately want a hidden folder indexed — env must allow it."""
    v = _vault(tmp_path)
    monkeypatch.setenv("BRAIN_INDEX_HIDDEN", "1")
    assert any(".obsidian" in p.parts for p in iter_notes(v))


def test_bom_does_not_break_frontmatter(tmp_path):
    """utf-8 + errors='ignore' keeps the BOM, so the first line is not '---'
    and every frontmatter parser downstream sees a note with no frontmatter."""
    f = tmp_path / "bom.md"
    f.write_bytes(b"\xef\xbb\xbf---\ntitle: x\ndate: 2026-08-01\n---\nbody text")
    assert read_note(f).startswith("---")


def test_unreadable_file_is_skipped_not_fatal(tmp_path):
    (tmp_path / "weird.md").write_bytes(b"\xff\xfe\x00binary junk")
    assert read_note(tmp_path / "weird.md") is not None
