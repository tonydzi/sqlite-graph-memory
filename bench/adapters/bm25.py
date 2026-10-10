# -*- coding: utf-8 -*-
"""BM25 over whole notes — the no-model reference row. Stdlib only, runs in about a second.

Why it is on the board: if a memory system with embeddings, a graph and a reranker does
not beat plain keyword search on a vault this small, that is the finding.

Abstention rule, fixed before the first run and not tuned on the questions: if the
question names something capitalized (a person, a project, a tool) that occurs in no note
at all, the vault cannot know about it, so abstain. This catches invented entities and is
blind, by design, to a real entity with a missing attribute.
"""
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extract import best_sentence, tokens  # noqa: E402

SYSTEMS = {'bm25': {}}
K1, B = 1.5, 0.75
CAP_RX = re.compile(r"\b[A-Z][a-zA-Z'-]+")
QWORDS = {'In', 'What', 'Which', 'Who', 'When', 'Where', 'How', 'Under', 'Project'}


class Adapter:
    def __init__(self):
        self.docs = {}

    def info(self):
        return {'version': 'reference', 'url': 'https://en.wikipedia.org/wiki/Okapi_BM25',
                'config': {'k1': K1, 'b': B, 'unit': 'whole note',
                           'answer': 'extractive: best-overlap sentence of top-3 notes',
                           'abstain': 'a capitalized question word that occurs in no note'}}

    def setup(self, vault_dir, work_dir):
        self.raw = {p.stem.lower(): p.read_text(encoding='utf-8') for p in sorted(Path(vault_dir).glob('*.md'))}
        self.tf = {k: Counter(re.findall(r'[a-z0-9]+', v.lower())) for k, v in self.raw.items()}
        self.len = {k: sum(c.values()) for k, c in self.tf.items()}
        self.avg = sum(self.len.values()) / len(self.len)
        df = Counter()
        for c in self.tf.values():
            df.update(c.keys())
        n = len(self.tf)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.lower_blob = '\n'.join(self.raw.values()).lower()

    def ask(self, question):
        terms = re.findall(r'[a-z0-9]+', question.lower())
        scores = []
        for k, c in self.tf.items():
            s = 0.0
            for t in terms:
                f = c.get(t)
                if f:
                    s += self.idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * self.len[k] / self.avg))
            scores.append((s, k))
        scores.sort(key=lambda x: (-x[0], x[1]))
        ranked = [k for s, k in scores if s > 0][:20]
        unknown = [w for w in CAP_RX.findall(question)
                   if w not in QWORDS and re.sub(r"'s$", '', w.lower()) not in self.lower_blob]
        nid, sent, ov = best_sentence(question, ranked, self.raw.get)
        if unknown or not sent:
            return {'retrieved': ranked, 'answer': None, 'citations': []}
        return {'retrieved': ranked, 'answer': sent, 'citations': [{'note': nid, 'quote': sent}]}


def make(**kwargs):
    return Adapter(**kwargs)
