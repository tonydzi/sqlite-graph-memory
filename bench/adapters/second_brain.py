# -*- coding: utf-8 -*-
"""stancsz/second-brain: a local-first knowledge graph for agents, SQLite FTS5 index over
markdown "Concepts". https://github.com/stancsz/second-brain (MIT).

How it is run here: the project is not on PyPI, so setup() clones the repository at a
pinned commit into work_dir and imports scripts/brain.py from it (stdlib only, no key, no
model). Every vault note becomes one Concept: title = note id, content = the note text.
Questions go to SecondBrain.search(), which is `concepts_fts MATCH ? ORDER BY rank`.

One thing the adapter does that the system does not: search() passes the query string to
FTS5 unchanged, and a natural question ("Which tool does X like to use most?") is an FTS5
syntax error at the "?" (also at apostrophes). The adapter therefore quotes every word and
joins them with OR. That is the smallest change that lets the system answer at all, and it
is recorded in config so the row is read with it.

Answer floor and abstention are the shared ones: best-overlap sentence of the top-3 notes;
abstain when nothing is retrieved or no sentence shares a content word with the question.
"""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extract import best_sentence  # noqa: E402

SYSTEMS = {'second-brain': {}}
REPO = 'https://github.com/stancsz/second-brain.git'
COMMIT = 'b17e6d5e9a1b4268a79fd742e9e03f51149b59d5'   # main, 2026-10-05
TERM_RX = re.compile(r"\w+")   # unicode-aware, like the unicode61 tokenizer


def fts_query(question):
    return ' OR '.join('"%s"' % t for t in TERM_RX.findall(question))


class Adapter:
    def __init__(self):
        self.raw = {}

    def info(self):
        return {'version': COMMIT[:12], 'url': 'https://github.com/stancsz/second-brain',
                'config': {'install': 'git clone at the pinned commit, scripts/brain.py imported',
                           'unit': 'one Concept per note (title = note id)', 'search': 'SecondBrain.search(limit=20)',
                           'query': 'every question word quoted, OR-joined (raw questions are an FTS5 syntax error)',
                           'answer': 'extractive: best-overlap sentence of top-3 notes',
                           'abstain': 'no hit, or no sentence shares a content word'}}

    def setup(self, vault_dir, work_dir):
        work_dir = Path(work_dir)
        src = work_dir / 'second-brain'
        subprocess.run(['git', 'clone', '-q', REPO, str(src)], check=True)
        subprocess.run(['git', '-C', str(src), 'checkout', '-q', COMMIT], check=True)
        spec = importlib.util.spec_from_file_location('second_brain_core', src / 'scripts' / 'brain.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.brain = mod.SecondBrain(work_dir / 'brain.db')
        self.raw = {p.stem.lower(): p.read_text(encoding='utf-8') for p in sorted(Path(vault_dir).glob('*.md'))}
        for nid, text in self.raw.items():
            self.brain.add(nid, text, collection='vault')

    def ask(self, question):
        q = fts_query(question)
        if not q:
            return {'retrieved': [], 'answer': None, 'citations': []}
        hits = self.brain.search(q, limit=20)
        ranked = [h['title'] for h in hits if h.get('title') in self.raw]
        nid, sent, ov = best_sentence(question, ranked, self.raw.get)
        if not sent:
            return {'retrieved': ranked, 'answer': None, 'citations': []}
        return {'retrieved': ranked, 'answer': sent, 'citations': [{'note': nid, 'quote': sent}]}


def make(**kwargs):
    return Adapter(**kwargs)
