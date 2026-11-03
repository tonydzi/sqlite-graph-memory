# -*- coding: utf-8 -*-
"""Incremental re-embedding (issue #17): only new or changed notes get embedded.

A manifest (`_brain_e5_manifest.json`) records each note's size/mtime/hash plus a
fingerprint of the settings that change chunk content. Unchanged notes keep their
existing `.npy`/`.pkl` rows; changed/new notes are re-chunked and re-embedded;
deleted notes lose their rows. `--full` (or a corrupt/missing/stale manifest)
forces a full rebuild.

A counting fake encoder stands in for the real embedding model, so these tests
run with no model download and no network.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sqlite_graph_memory import index_notes, onnx_backend  # noqa: E402
from sqlite_graph_memory.index_notes import build_index  # noqa: E402
from sqlite_graph_memory.brain_ask import load_index  # noqa: E402


class CountingFakeEncoder:
    """Records every text it is asked to embed; never loads a real model."""

    def __init__(self):
        self.seen = []

    def encode(self, texts, **_kwargs):
        self.seen.extend(texts)
        return [[float(len(t))] for t in texts]


BODY_A = "Alpha note body long enough to form exactly one chunk of text here."
BODY_B = "Bravo note body long enough to form exactly one chunk of text here."
BODY_C = "Charlie note body long enough to form exactly one chunk of text here."
BODY_B_EDITED = "Bravo note body has been edited with new content for this test."


def _write_notes(notes_dir: Path):
    (notes_dir / "a.md").write_text(BODY_A, encoding="utf-8")
    (notes_dir / "b.md").write_text(BODY_B, encoding="utf-8")
    (notes_dir / "c.md").write_text(BODY_C, encoding="utf-8")


def test_first_build_embeds_every_note(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_notes(notes_dir)
    out_dir = tmp_path / "index"

    fake = CountingFakeEncoder()
    result = build_index(notes_dir, out_dir=out_dir, encoder=fake)

    assert len(fake.seen) == 3
    assert result["chunks"] == 3
    assert result["files"] == 3
    assert (out_dir / "_brain_e5_manifest.json").exists()


def test_unchanged_rerun_embeds_nothing(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_notes(notes_dir)
    out_dir = tmp_path / "index"
    build_index(notes_dir, out_dir=out_dir, encoder=CountingFakeEncoder())

    fake = CountingFakeEncoder()
    result = build_index(notes_dir, out_dir=out_dir, encoder=fake)

    assert fake.seen == []
    assert result["chunks"] == 3


def test_unchanged_rerun_uses_size_mtime_fast_path_without_rehashing(tmp_path, monkeypatch):
    """Unchanged files with matching size and mtime skip _file_hash disk reads;
    if mtime changes without content change, _file_hash runs and still skips re-embedding."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_notes(notes_dir)
    out_dir = tmp_path / "index"
    build_index(notes_dir, out_dir=out_dir, encoder=CountingFakeEncoder())

    hashed = []
    real_hash = index_notes._file_hash

    def spy_hash(path):
        hashed.append(Path(path).name)
        return real_hash(path)

    monkeypatch.setattr(index_notes, "_file_hash", spy_hash)

    fake = CountingFakeEncoder()
    result = build_index(notes_dir, out_dir=out_dir, encoder=fake)
    assert hashed == []
    assert fake.seen == []
    assert result["chunks"] == 3

    # Touch b.md (advance mtime without changing content): only b.md is hashed, 0 re-embedded
    b_path = notes_dir / "b.md"
    st = b_path.stat()
    os.utime(b_path, (st.st_atime, st.st_mtime + 10.0))
    fake2 = CountingFakeEncoder()
    result2 = build_index(notes_dir, out_dir=out_dir, encoder=fake2)
    assert hashed == ["b.md"]
    assert fake2.seen == []
    assert result2["chunks"] == 3


def test_editing_one_note_reembeds_only_that_note(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_notes(notes_dir)
    out_dir = tmp_path / "index"
    build_index(notes_dir, out_dir=out_dir, encoder=CountingFakeEncoder())

    (notes_dir / "b.md").write_text(BODY_B_EDITED, encoding="utf-8")
    fake = CountingFakeEncoder()
    result = build_index(notes_dir, out_dir=out_dir, encoder=fake)

    assert len(fake.seen) == 1
    assert "edited" in fake.seen[0]
    assert result["chunks"] == 3


def test_deleting_one_note_drops_its_rows_and_stays_aligned(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_notes(notes_dir)
    out_dir = tmp_path / "index"
    build_index(notes_dir, out_dir=out_dir, encoder=CountingFakeEncoder())

    (notes_dir / "c.md").unlink()
    fake = CountingFakeEncoder()
    result = build_index(notes_dir, out_dir=out_dir, encoder=fake)

    assert fake.seen == []
    assert result["chunks"] == 2

    emb, meta = load_index(out_dir / "_brain_e5.npy", out_dir / "_brain_e5_meta.pkl")
    assert emb.shape[0] == len(meta) == 2
    assert not any(m["path"].endswith("c.md") for m in meta)


def test_full_flag_forces_complete_rebuild(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_notes(notes_dir)
    out_dir = tmp_path / "index"
    build_index(notes_dir, out_dir=out_dir, encoder=CountingFakeEncoder())

    fake = CountingFakeEncoder()
    result = build_index(notes_dir, out_dir=out_dir, encoder=fake, full=True)

    assert len(fake.seen) == 3
    assert result["chunks"] == 3


def test_full_cli_flag_is_parsed_and_forwarded(tmp_path, monkeypatch):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_notes(notes_dir)

    calls = []

    def fake_build_index(nd, out_dir=None, encoder=None, full=False):
        calls.append(full)
        return {"chunks": 3, "files": 3, "new_texts": 0}

    monkeypatch.setattr(index_notes, "build_index", fake_build_index)
    monkeypatch.setattr(sys, "argv", ["index_notes.py", str(notes_dir), "--full"])
    monkeypatch.setenv("SGM_BACKEND", "torch")
    index_notes.main()

    assert calls == [True]


def test_default_encoder_uses_onnx_backend_when_forced(tmp_path, monkeypatch):
    """When encoder=None and SGM_BACKEND=onnx, build_index loads the ONNX encoder
    via brain_ask.load_encoder() without touching sentence_transformers."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_notes(notes_dir)
    out_dir = tmp_path / "index"

    fake_onnx_enc = CountingFakeEncoder()
    monkeypatch.setenv("SGM_BACKEND", "onnx")
    monkeypatch.setattr(onnx_backend, "Encoder", lambda: fake_onnx_enc)

    result = build_index(notes_dir, out_dir=out_dir)

    assert len(fake_onnx_enc.seen) == 3
    assert result["chunks"] == 3
    assert result["files"] == 3


def test_changed_setting_triggers_automatic_full_rebuild(tmp_path, monkeypatch):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_notes(notes_dir)
    out_dir = tmp_path / "index"
    build_index(notes_dir, out_dir=out_dir, encoder=CountingFakeEncoder())

    monkeypatch.setattr(index_notes, "SNIPPET_CHARS", 300)
    fake = CountingFakeEncoder()
    result = build_index(notes_dir, out_dir=out_dir, encoder=fake)

    assert len(fake.seen) == 3
    assert result["chunks"] == 3


def test_missing_manifest_forces_full_rebuild(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_notes(notes_dir)
    out_dir = tmp_path / "index"
    build_index(notes_dir, out_dir=out_dir, encoder=CountingFakeEncoder())

    (out_dir / "_brain_e5_manifest.json").unlink()
    fake = CountingFakeEncoder()
    result = build_index(notes_dir, out_dir=out_dir, encoder=fake)

    assert len(fake.seen) == 3
    assert result["chunks"] == 3


def test_corrupt_manifest_forces_full_rebuild(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_notes(notes_dir)
    out_dir = tmp_path / "index"
    build_index(notes_dir, out_dir=out_dir, encoder=CountingFakeEncoder())

    (out_dir / "_brain_e5_manifest.json").write_text("{not json", encoding="utf-8")
    fake = CountingFakeEncoder()
    result = build_index(notes_dir, out_dir=out_dir, encoder=fake)

    assert len(fake.seen) == 3
    assert result["chunks"] == 3


def test_manifest_with_non_dict_files_forces_full_rebuild(tmp_path):
    """R-Gate (PR #1): a syntactically valid manifest whose 'files' is a list, not a
    dict, must not reach old_files.get(...) and raise AttributeError — it should be
    treated as corrupt, like a missing or unparseable manifest."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_notes(notes_dir)
    out_dir = tmp_path / "index"
    build_index(notes_dir, out_dir=out_dir, encoder=CountingFakeEncoder())

    manifest_path = out_dir / "_brain_e5_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fake = CountingFakeEncoder()
    result = build_index(notes_dir, out_dir=out_dir, encoder=fake)

    assert len(fake.seen) == 3
    assert result["chunks"] == 3


def test_manifest_with_non_dict_file_record_forces_full_rebuild(tmp_path):
    """R-Gate (PR #1 re-review): 'files' is a dict, but one of its own per-file
    records is a list, not a dict. old_files.get(path, {}).get('hash') must not
    reach that record and raise AttributeError — it should be treated as corrupt."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_notes(notes_dir)
    out_dir = tmp_path / "index"
    build_index(notes_dir, out_dir=out_dir, encoder=CountingFakeEncoder())

    manifest_path = out_dir / "_brain_e5_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    some_path = next(iter(manifest["files"]))
    manifest["files"][some_path] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fake = CountingFakeEncoder()
    result = build_index(notes_dir, out_dir=out_dir, encoder=fake)

    assert len(fake.seen) == 3
    assert result["chunks"] == 3


def test_npy_meta_row_mismatch_forces_full_rebuild(tmp_path):
    """R-Gate (PR #1): a readable .npy with fewer rows than _brain_e5_meta.pkl must not
    reach the boolean-mask slice and raise IndexError — it should be treated as corrupt."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_notes(notes_dir)
    out_dir = tmp_path / "index"
    build_index(notes_dir, out_dir=out_dir, encoder=CountingFakeEncoder())

    emb_path = out_dir / "_brain_e5.npy"
    short_emb = np.load(emb_path)[:1]          # fewer rows than _brain_e5_meta.pkl's 3 entries
    np.save(emb_path, short_emb)

    fake = CountingFakeEncoder()
    result = build_index(notes_dir, out_dir=out_dir, encoder=fake)

    assert len(fake.seen) == 3
    assert result["chunks"] == 3
