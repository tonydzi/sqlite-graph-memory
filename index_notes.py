# -*- coding: utf-8 -*-
"""index_notes.py — minimal embedding indexer for a folder of markdown notes.

Builds the dense index that brain_ask.py retrieves from:
  index/_brain_e5.npy       float32 matrix, one row per chunk (L2-normalized)
  index/_brain_e5_meta.pkl  list of dicts: {path, title, snippet, date}

Chunking is deliberately simple (fixed-size windows over the note body) — the
cross-encoder reranker downstream forgives coarse chunk boundaries.

USAGE:
  python index_notes.py <notes_dir>
Config (env, optional):
  BRAIN_INDEX_DIR     output dir (default: ./index)
  BRAIN_INDEX_HIDDEN  1 = also index dot-directories (default: skip them,
                      so .stversions/.obsidian/.git and Syncthing conflict
                      copies never enter the index)
"""
import os, re, sys, pickle
from pathlib import Path
import numpy as np

E5_MODEL = 'intfloat/multilingual-e5-base'
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
SNIPPET_CHARS = 600

FM_RX = re.compile(r'^---\r?\n(.*?)\r?\n---\r?\n', re.S)


def chunks_of(text):
    step = CHUNK_CHARS - CHUNK_OVERLAP
    for start in range(0, max(len(text), 1), step):
        c = text[start:start + CHUNK_CHARS].strip()
        if len(c) >= 40:
            yield c
        if start + CHUNK_CHARS >= len(text):
            break


def fm_date(fm):
    m = re.search(r'(?m)^date[^:]*:\s*"?(\d{4}-\d{2}-\d{2})', fm)
    return m.group(1) if m else ''


# A synced Obsidian vault is full of markdown that is not a note. `.stversions/`
# holds Syncthing's old revisions, `*.sync-conflict-*` are its conflict copies,
# `.obsidian/` is plugin cache and `.git/` is repository internals. Indexing those
# is worse than noise: a stale revision of a note is semantically almost identical
# to the live one, so it sits next to it in the top-K and answers with old content.
CONFLICT_MARKER = 'sync-conflict'


def iter_notes(notes_dir):
    """Files that count as a note under ``notes_dir``.

    Skips dot-directories and Syncthing conflict copies. Set ``BRAIN_INDEX_HIDDEN=1``
    to index hidden folders too — someone will legitimately keep notes in one.
    """
    root = Path(notes_dir)
    include_hidden = os.getenv('BRAIN_INDEX_HIDDEN', '') not in ('', '0', 'false', 'no')
    out = []
    for p in root.rglob('*.md'):
        rel = p.relative_to(root)
        if not include_hidden and any(part.startswith('.') for part in rel.parts[:-1]):
            continue
        if CONFLICT_MARKER in p.name:
            continue
        out.append(p)
    return sorted(out)


def read_note(path):
    """Text of one note, or None if it cannot be read.

    ``utf-8-sig`` strips the BOM. Reading BOM'd files as plain utf-8 leaves a
    leading U+FEFF, so the first line is not ``---`` and the frontmatter regex
    below silently sees a note as having none — losing its date and title.
    """
    try:
        return Path(path).read_text(encoding='utf-8-sig', errors='replace')
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    notes_dir = Path(sys.argv[1])
    out_dir = Path(os.getenv('BRAIN_INDEX_DIR', 'index'))
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, texts = [], []
    for p in iter_notes(notes_dir):
        t = read_note(p)
        if t is None:
            continue
        m = FM_RX.match(t)
        fm = m.group(1) if m else ''
        body = t[m.end():] if m else t
        date = fm_date(fm)
        title = p.stem
        for c in chunks_of(body):
            meta.append({'path': str(p), 'title': title,
                         'snippet': c[:SNIPPET_CHARS], 'date': date})
            texts.append('passage: ' + c)     # e5 expects the passage: prefix
    if not texts:
        print('no indexable .md notes found under', notes_dir); return

    from sentence_transformers import SentenceTransformer
    try:
        import torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        dev = 'cpu'
    enc = SentenceTransformer(E5_MODEL, device=dev)
    emb = enc.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                     batch_size=64, show_progress_bar=True).astype('float32')

    np.save(out_dir / '_brain_e5.npy', emb)
    (out_dir / '_brain_e5_meta.pkl').write_bytes(pickle.dumps(meta))
    files = len({m['path'] for m in meta})
    print('indexed %d chunks from %d files -> %s' % (len(meta), files, out_dir))


if __name__ == '__main__':
    main()
