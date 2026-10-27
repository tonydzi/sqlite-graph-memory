# -*- coding: utf-8 -*-
"""Plain SQLite FTS5 over whole notes: the second no-model reference row. Stdlib only.

Why it is on the board next to bm25.py: FTS5 is what most "SQLite memory" projects ship
(an FTS table, `ORDER BY rank`). bm25.py is a hand-written BM25; this row is the library
version with SQLite's own bm25() ranking and its default unicode61 tokenizer, so a reader
can see whether the two agree.

Answer floor and abstention are the same as the other retrieval-only rows: the sentence of
the top-3 notes that shares the most content words with the question, and abstain when no
sentence shares any. There is no entity check, so an absent question is answered whenever
any of its words occurs in the vault, which on this vault is every time. That is the honest result for FTS5 as shipped.
"""
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extract import best_sentence  # noqa: E402

SYSTEMS = {'sqlite-fts5': {}}
TERM_RX = re.compile(r"\w+")   # unicode-aware, like the unicode61 tokenizer


def fts_query(question):
    """FTS5 MATCH syntax: every word quoted, joined by OR. Quotes make '?' and apostrophes safe.
    Returns '' for a question with no word characters; the caller abstains instead of sending it."""
    return ' OR '.join('"%s"' % t for t in TERM_RX.findall(question))


class Adapter:
    def __init__(self):
        self.raw = {}

    def info(self):
        return {'version': 'reference', 'url': 'https://www.sqlite.org/fts5.html',
                'config': {'sqlite': sqlite3.sqlite_version, 'tokenizer': 'unicode61 (FTS5 default)', 'rank': 'bm25() (FTS5 default)',
                           'unit': 'whole note', 'query': 'every question word quoted, OR-joined',
                           'answer': 'extractive: best-overlap sentence of top-3 notes',
                           'abstain': 'no hit, or no sentence shares a content word'}}

    def setup(self, vault_dir, work_dir):
        self.raw = {p.stem.lower(): p.read_text(encoding='utf-8') for p in sorted(Path(vault_dir).glob('*.md'))}
        self.con = sqlite3.connect(str(Path(work_dir) / 'fts5.db'))
        self.con.execute('CREATE VIRTUAL TABLE notes USING fts5(id UNINDEXED, body)')
        self.con.executemany('INSERT INTO notes(id, body) VALUES (?, ?)', sorted(self.raw.items()))
        self.con.commit()

    def ask(self, question):
        q = fts_query(question)
        if not q:
            return {'retrieved': [], 'answer': None, 'citations': []}
        rows = self.con.execute('SELECT id FROM notes WHERE notes MATCH ? ORDER BY rank LIMIT 20', (q,)).fetchall()
        ranked = [r[0] for r in rows]
        nid, sent, ov = best_sentence(question, ranked, self.raw.get)
        if not sent:
            return {'retrieved': ranked, 'answer': None, 'citations': []}
        return {'retrieved': ranked, 'answer': sent, 'citations': [{'note': nid, 'quote': sent}]}


def make(**kwargs):
    return Adapter(**kwargs)
