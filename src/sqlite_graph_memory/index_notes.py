# -*- coding: utf-8 -*-
"""index_notes.py — minimal embedding indexer for a folder of markdown notes.

Builds the dense index that brain_ask.py retrieves from:
  index/_brain_e5.npy            float32 matrix, one row per chunk (L2-normalized)
  index/_brain_e5_meta.pkl       list of dicts: {path, title, snippet, date}
  index/_brain_e5_manifest.json  {path: {size, mtime, hash}} + a settings fingerprint

Chunking is deliberately simple (fixed-size windows over the note body) — the
cross-encoder reranker downstream forgives coarse chunk boundaries.

Re-running only re-embeds notes that are new or changed since the manifest was
written (size + mtime fast path, content sha256 truth); unchanged notes keep
their existing rows, and deleted notes lose theirs. Pass --full (or delete/corrupt
the manifest, or change one of the chunking settings below) to force a complete
rebuild.

USAGE:
  python index_notes.py <notes_dir> [--full]
Config (env, optional):
  BRAIN_INDEX_DIR     output dir (default: ./index)
  BRAIN_INDEX_HIDDEN  1 = also index dot-files and dot-directories (default:
                      skip them, so .draft.md, .stversions/.obsidian/.git and
                      Syncthing conflict copies never enter the index)
"""
import argparse, hashlib, json, os, re, sys, pickle
from pathlib import Path
import numpy as np

E5_MODEL = 'intfloat/multilingual-e5-base'
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
SNIPPET_CHARS = 600

FM_RX = re.compile(r'^---\r?\n(.*?)\r?\n---\r?\n', re.S)

try:
    from sqlite_graph_memory import brain_ask as _ba
except ImportError:                           # run as a loose script next to brain_ask.py
    import brain_ask as _ba

_EMBEDDINGS_HINT = _ba._EMBEDDINGS_HINT


def _require_sentence_transformers():
    """Import sentence_transformers, or exit with instructions.

    Raises SystemExit (not ImportError) so the CLI prints one actionable line
    rather than a traceback through site-packages.
    """
    try:
        import sentence_transformers  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name not in ("sentence_transformers", "torch"):
            raise
        raise SystemExit(_EMBEDDINGS_HINT) from exc
    return sentence_transformers


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

    Skips dot-directories, dot-files (``.draft.md``) and Syncthing conflict copies.
    Set ``BRAIN_INDEX_HIDDEN=1`` to index hidden paths too — someone will legitimately keep notes in one.
    """
    root = Path(notes_dir)
    include_hidden = os.getenv('BRAIN_INDEX_HIDDEN', '') not in ('', '0', 'false', 'no')
    out = []
    for p in root.rglob('*.md'):
        rel = p.relative_to(root)
        if not include_hidden and any(part.startswith('.') for part in rel.parts):
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


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _settings_fingerprint():
    return {'E5_MODEL': E5_MODEL, 'CHUNK_CHARS': CHUNK_CHARS,
            'CHUNK_OVERLAP': CHUNK_OVERLAP, 'SNIPPET_CHARS': SNIPPET_CHARS}


def _chunk_note(p):
    """-> (meta dicts, 'passage: '-prefixed texts) for one note's chunks."""
    t = read_note(p)
    if t is None:
        return [], []
    m = FM_RX.match(t)
    fm = m.group(1) if m else ''
    body = t[m.end():] if m else t
    date = fm_date(fm)
    title = p.stem
    meta, texts = [], []
    for c in chunks_of(body):
        meta.append({'path': str(p), 'title': title,
                     'snippet': c[:SNIPPET_CHARS], 'date': date})
        texts.append('passage: ' + c)     # e5 expects the passage: prefix
    return meta, texts


def _embed(encoder, texts):
    """encoder.encode(texts) if it has one, else encoder(texts); lazily loads
    the default model via brain_ask.load_encoder() only when encoder is None,
    and only when texts is non-empty (so an unchanged re-run never touches
    the embedding stack)."""
    if encoder is None:
        encoder = _ba.load_encoder()
        return np.asarray(
            encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                           batch_size=64, show_progress_bar=True),
            dtype='float32',
        )
    result = encoder.encode(texts) if hasattr(encoder, 'encode') else encoder(texts)
    return np.asarray(result, dtype='float32')


def build_index(notes_dir, out_dir=None, encoder=None, full=False):
    """Index ``notes_dir`` into ``out_dir`` (default ``./index``).

    Re-embeds only notes that are new or changed since the manifest was last
    written; unchanged notes keep their rows, deleted notes lose theirs.
    ``full=True`` (or --full on the CLI) forces a complete rebuild, as does a
    missing/corrupt manifest/index or a changed chunking setting.
    ``encoder``, when given, is called as ``encoder.encode(texts)`` or
    ``encoder(texts)`` — inject a fake in tests so no model ever loads.
    Returns {'chunks': int, 'files': int, 'new_texts': int}.
    """
    notes_dir = Path(notes_dir)
    out_dir = Path(out_dir) if out_dir is not None else Path('index')
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / '_brain_e5.npy'
    meta_path = out_dir / '_brain_e5_meta.pkl'
    manifest_path = out_dir / '_brain_e5_manifest.json'
    settings = _settings_fingerprint()

    notes = iter_notes(notes_dir)

    do_full = full
    old_meta, old_emb, old_files = [], None, {}
    if not do_full:
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), dict):
                raise ValueError('manifest is not {settings, files: {...}}')
            if manifest.get('settings') != settings:
                do_full = True
            else:
                old_files = manifest['files']
                if not all(isinstance(v, dict) for v in old_files.values()):
                    raise ValueError('manifest file record is not a dict')
                old_meta = pickle.loads(meta_path.read_bytes())
                old_emb = np.load(emb_path)
                if (not isinstance(old_meta, list) or old_emb.ndim != 2
                        or old_emb.shape[0] != len(old_meta)):
                    raise ValueError('index/meta row count mismatch')
        except Exception:
            do_full = True

    current_files = {}
    for p in notes:
        key = str(p)
        st = p.stat()
        prev = old_files.get(key) if not do_full else None
        if (isinstance(prev, dict)
                and prev.get('size') == st.st_size
                and prev.get('mtime') == st.st_mtime
                and isinstance(prev.get('hash'), str)
                and prev.get('hash')):
            file_hash = prev['hash']
        else:
            file_hash = _file_hash(p)
        current_files[key] = {'size': st.st_size, 'mtime': st.st_mtime, 'hash': file_hash}

    if do_full:
        old_meta, old_emb, old_files = [], None, {}
        changed_or_new = set(current_files)
    else:
        unchanged = {path for path, info in current_files.items()
                     if old_files.get(path, {}).get('hash') == info['hash']}
        changed_or_new = set(current_files) - unchanged
        kept_mask = [m['path'] in unchanged for m in old_meta]
        old_meta = [m for m, keep in zip(old_meta, kept_mask) if keep]
        old_emb = old_emb[np.array(kept_mask, dtype=bool)] if old_emb is not None else None

    new_meta, new_texts = [], []
    for p in notes:
        if str(p) in changed_or_new:
            m, t = _chunk_note(p)
            new_meta.extend(m)
            new_texts.extend(t)

    new_emb = _embed(encoder, new_texts) if new_texts else None

    meta = old_meta + new_meta
    if old_emb is not None and new_emb is not None:
        emb = np.vstack([old_emb, new_emb]).astype('float32')
    elif new_emb is not None:
        emb = new_emb.astype('float32')
    elif old_emb is not None:
        emb = old_emb.astype('float32')
    else:
        emb = np.zeros((0, 0), dtype='float32')

    np.save(emb_path, emb)
    meta_path.write_bytes(pickle.dumps(meta))
    manifest_path.write_text(json.dumps({'settings': settings, 'files': current_files}, indent=2),
                             encoding='utf-8')

    files = len({m['path'] for m in meta})
    return {'chunks': len(meta), 'files': files, 'new_texts': len(new_texts)}


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('notes_dir', nargs='?')
    ap.add_argument('--full', action='store_true',
                    help='force a complete rebuild, ignoring the manifest')
    args = ap.parse_args()
    if not args.notes_dir:
        print(__doc__); return
    notes_dir = Path(args.notes_dir)
    out_dir = Path(os.getenv('BRAIN_INDEX_DIR', 'index'))

    result = build_index(notes_dir, out_dir=out_dir, full=args.full)
    if result['chunks'] == 0:
        print('no indexable .md notes found under', notes_dir); return
    if _ba.backend() == 'onnx':
        # Fetch the reranker too: otherwise the first recall from an MCP client is also
        # a ~470 MB download and runs into the client's tool timeout.
        try:
            _ba._onnx().prefetch()
        except Exception as exc:              # offline: the first recall fetches it instead
            print('reranker not prefetched (%s); the first recall will fetch it' % type(exc).__name__)
    print('indexed %d chunks from %d files -> %s' % (result['chunks'], result['files'], out_dir))


if __name__ == '__main__':
    main()
