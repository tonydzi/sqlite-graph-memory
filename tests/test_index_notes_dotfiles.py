# -*- coding: utf-8 -*-
"""Dot-FILES are skipped the same way dot-DIRECTORIES are (issue #25).

`iter_notes` filtered hidden *directories* by checking `rel.parts[:-1]` — every
path component except the last. The last component is the filename, so a note
called `.draft.md` or `.trash.md` sitting in the vault root (or in any visible
folder) was indexed as if it were a real note, while the same file one level
down inside `.drafts/` was correctly skipped. Reproduced 2026-10-13 with
`.secret.md` next to `ok.md`: `iter_notes()` returned both.

RED-FIRST: restore `rel.parts[:-1]` on the filter line and
`test_dot_file_in_root_is_skipped` fails with `.draft.md` in the result.

Both halves are pinned on purpose — the directory case already worked and must
keep working, the file case is the fix — and `BRAIN_INDEX_HIDDEN=1` must still
let both through, because someone legitimately keeps notes in a hidden folder.

No model download, no network: pure file work.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sqlite_graph_memory.index_notes import iter_notes  # noqa: E402


def _vault(tmp_path: Path) -> Path:
    (tmp_path / "ok.md").write_text("visible note", encoding="utf-8")
    (tmp_path / ".draft.md").write_text("hidden file in root", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / ".trash.md").write_text("hidden file in visible dir", encoding="utf-8")
    (tmp_path / ".drafts").mkdir()
    (tmp_path / ".drafts" / "inside.md").write_text("visible name, hidden dir", encoding="utf-8")
    return tmp_path


def _names(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in iter_notes(root)}


def test_dot_file_in_root_is_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAIN_INDEX_HIDDEN", raising=False)
    assert ".draft.md" not in _names(_vault(tmp_path))


def test_dot_file_in_visible_dir_is_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAIN_INDEX_HIDDEN", raising=False)
    assert "sub/.trash.md" not in _names(_vault(tmp_path))


def test_dot_dir_still_skipped_and_visible_note_still_kept(tmp_path, monkeypatch):
    """The pre-existing behaviour must survive the fix."""
    monkeypatch.delenv("BRAIN_INDEX_HIDDEN", raising=False)
    names = _names(_vault(tmp_path))
    assert ".drafts/inside.md" not in names
    assert "ok.md" in names


def test_hidden_flag_admits_dot_files_and_dot_dirs(tmp_path, monkeypatch):
    """Opt-in must cover both kinds of hidden path, not just directories."""
    monkeypatch.setenv("BRAIN_INDEX_HIDDEN", "1")
    names = _names(_vault(tmp_path))
    assert {"ok.md", ".draft.md", "sub/.trash.md", ".drafts/inside.md"} <= names
