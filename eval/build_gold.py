# -*- coding: utf-8 -*-
"""build_gold.py — turn the wikilinks you already wrote into a retrieval test set.

The premise: in a hand-curated markdown vault the relevance labels already exist.
A note's own title should retrieve that note. Its first paragraph should retrieve it too.
The notes it `[[links]]` to are the neighbours a human decided are related. Nobody has to
label anything, and no LLM is called — this script is pure file work plus the index meta.

Four question classes, each probing a different part of the pipeline:

    title     query = the note's title          -> gold = that note       (name lookup)
    body      query = its first paragraph       -> gold = that note       (theme lookup)
    bridge    query = its first paragraph       -> gold = its [[link]] TARGETS (the graph)
    temporal  query = an outdated note's title  -> gold = its replacement (see below)

`temporal` needs one convention: a note that has been superseded carries
`superseded_by: "[[newer-note]]"` in its frontmatter. If your vault has no such field the
class simply comes out empty and `run_eval.py` says so instead of pretending.

USAGE
    python eval/build_gold.py <notes_dir> [--out FILE] [--n-per-class 60] [--seed 42]
    python eval/build_gold.py ~/vault --folders 02-Decisions,03-Insights,06-Concepts

Config (env): BRAIN_INDEX_DIR — the index built by index_notes.py (default: ./index).
    Questions are only built from notes that are IN that index: a gold answer that was
    never indexed is unreachable by construction and would measure the indexer, not recall.

FREEZE IT. Write the file once, keep the filename (it carries the date), and reuse it for
every later comparison. A gold set rebuilt between two runs makes the two numbers
incomparable, which is the quiet way an eval starts lying to you.
"""
import argparse
import datetime
import json
import os
import pickle
import random
import re
import sys
from collections import Counter
from pathlib import Path

WIKILINK_RX = re.compile(r'\[\[([^\]\|#]+)(?:[|#][^\]]*)?\]\]')
CYRILLIC_RX = re.compile(r'[а-яё]', re.I)

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding='utf-8')
    except Exception: pass


def lang_of(s):
    """Rough language tag, so a run can report per-language recall separately.

    Not linguistics: it splits a bilingual vault into two buckets, because a multilingual
    embedder can be strong in one language and weak in the other and a single average hides it.
    """
    letters = [c for c in s if c.isalpha()]
    if not letters: return 'na'
    return 'ru' if sum(1 for c in letters if CYRILLIC_RX.match(c)) / len(letters) > 0.4 else 'en'


def split_frontmatter(text):
    m = re.match(r'^---\r?\n(.*?)\r?\n---\r?\n?(.*)$', text, re.S)
    return (m.group(1), m.group(2)) if m else ('', text)


def fm_get(fm, key):
    m = re.search(r'(?m)^' + re.escape(key) + r'\s*:\s*"?(.+?)"?\s*$', fm)
    return m.group(1).strip() if m else None


def strip_markup(s):
    s = re.sub(r'\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]', lambda m: m.group(2) or m.group(1), s)
    return re.sub(r'\s+', ' ', re.sub(r'[*_`>#]+', '', s)).strip()


def note_title(fm, body, path):
    """Frontmatter title, else the first H1, else the filename."""
    t = fm_get(fm, 'title') or None
    if not t:
        h = re.search(r'(?m)^#\s+(.+)$', body)
        t = h.group(1).strip() if h else Path(path).stem
    return strip_markup(t)[:200]


def first_paragraph(body, minlen=100, maxlen=400):
    """First real prose paragraph: no headings, tables, code fences or blockquotes.

    minlen exists because a two-word paragraph is not a question — it is a fragment that
    would score as a retrieval failure no matter how good the index is.
    """
    body = re.sub(r'(?m)^\s*>.*$', '', body)
    for para in re.split(r'\n\s*\n', body):
        p = para.strip()
        if not p or p.startswith(('#', '|', '```')): continue
        p = strip_markup(p)
        if len(p) >= minlen: return p[:maxlen]
    return None


def indexed_stems(index_dir):
    """Filenames (no extension, lowercased) present in the embedding index."""
    meta_path = Path(index_dir) / '_brain_e5_meta.pkl'
    if not meta_path.exists():
        sys.exit('no index meta at %s — run index_notes.py first' % meta_path)
    meta = pickle.loads(meta_path.read_bytes())
    return {Path(m['path']).stem.lower() for m in meta}


def read_notes(notes_dir, stems, folders=None):
    """Every indexed note, parsed into the fields the four classes need."""
    root = Path(notes_dir)
    paths = []
    for p in sorted(root.rglob('*.md')):
        rel = p.relative_to(root)
        if any(part.startswith('.') for part in rel.parts[:-1]): continue
        if 'sync-conflict' in p.name: continue
        if folders and (not rel.parts[:-1] or rel.parts[0] not in folders): continue
        if p.stem.lower() not in stems: continue
        paths.append(p)
    notes = []
    for p in paths:
        try: text = p.read_text(encoding='utf-8-sig', errors='ignore')
        except Exception: continue
        fm, body = split_frontmatter(text)
        links = sorted({t.strip().lower() for t in WIKILINK_RX.findall(body)}
                       & stems - {p.stem.lower()})
        sup = fm_get(fm, 'superseded_by')
        sup_m = WIKILINK_RX.findall(sup) if sup else []
        sup = sup_m[0].strip().lower() if sup_m else (sup.strip().lower() if sup else None)
        notes.append({'stem': p.stem.lower(), 'folder': (p.relative_to(root).parts[:-1] or ('',))[0],
                      'title': note_title(fm, body, p), 'para': first_paragraph(body),
                      'links': links, 'sup': sup if sup in stems else None})
    return notes


def build_gold(notes, n_per_class, seed):
    """-> list of question dicts. Sampling is seeded, so the same vault gives the same set."""
    rnd = random.Random(seed)
    notes = list(notes)
    rnd.shuffle(notes)
    pools = {'title':    [n for n in notes if len(n['title']) >= 8],
             'body':     [n for n in notes if n['para']],
             'bridge':   [n for n in notes if n['para'] and len(n['links']) >= 2],
             'temporal': [n for n in notes if n['sup']]}
    gold = []
    for cls in ('title', 'body', 'bridge', 'temporal'):
        for n in pools[cls][:n_per_class]:
            q = n['para'] if cls in ('body', 'bridge') else n['title']
            g = n['links'][:8] if cls == 'bridge' else ([n['sup']] if cls == 'temporal' else [n['stem']])
            gold.append({'id': '%s-%03d' % (cls, sum(1 for x in gold if x['cls'] == cls) + 1),
                         'cls': cls, 'query': q, 'gold': g, 'lang': lang_of(q),
                         'folder': n['folder'], 'src': n['stem']})
    return gold


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('notes_dir', help='the markdown folder you indexed')
    ap.add_argument('--out', help='output .jsonl (default: eval/gold-<today>.jsonl)')
    ap.add_argument('--n-per-class', type=int, default=60,
                    help='questions per class; 60 is enough to see a 0.05 move, not a 0.01 one')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folders', help='comma-separated top-level folders to sample from '
                                      '(default: the whole vault)')
    a = ap.parse_args()

    stems = indexed_stems(os.getenv('BRAIN_INDEX_DIR', 'index'))
    folders = set(f.strip() for f in a.folders.split(',')) if a.folders else None
    notes = read_notes(a.notes_dir, stems, folders)
    if not notes:
        sys.exit('no indexed notes found under %s (folders=%s)' % (a.notes_dir, folders or 'all'))
    gold = build_gold(notes, a.n_per_class, a.seed)

    out = Path(a.out) if a.out else Path(__file__).parent / ('gold-%s.jsonl' % datetime.date.today())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(json.dumps(g, ensure_ascii=False) for g in gold) + '\n', encoding='utf-8')

    counts = Counter(g['cls'] for g in gold)
    print('%d questions from %d indexed notes -> %s' % (len(gold), len(notes), out))
    print('  classes %s · languages %s' % (dict(counts), dict(Counter(g['lang'] for g in gold))))
    for cls in ('title', 'body', 'bridge', 'temporal'):
        if counts[cls] < 20:
            print('  ! %s: only %d questions — too few to judge a change by; treat as a smoke test'
                  % (cls, counts[cls]))
    print('  freeze this file: comparing two runs on two different gold sets proves nothing')


if __name__ == '__main__':
    main()
