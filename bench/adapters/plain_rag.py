# -*- coding: utf-8 -*-
"""Plain dense chunk RAG — the no-graph, no-reranker bi-encoder reference row.

Uses the exact same markdown chunker (`sqlite_graph_memory.index_notes`, MAX_CHARS=1200,
OVERLAP=200) and `intfloat/multilingual-e5-base` bi-encoder as `sqlite-graph-memory`, plus
the shared extractive floor in `_extract.py`, without `expand_1hop` and without cross-encoder
reranking. Comparing this row against `sqlite-graph-memory-vector` and `sqlite-graph-memory`
isolates what the cross-encoder reranker and the 1-hop `[[wikilink]]` graph each contribute:
`plain-rag` beats `sqlite-graph-memory` on `paraphrase` (R@5 1.00 vs 0.50), while
`sqlite-graph-memory` beats it on `bridge` (R@5 1.00 vs 0.87) and on abstention (0.45 vs 0.18).

No cosine threshold is applied (there is no tuned cutoff): the 0.18 (`4/22`) "abstains when
absent" rate comes solely from `_extract.best_sentence` returning `(None, None, 0)` when no
sentence in the top-3 retrieved notes shares a non-stopword token (`_extract.STOP`) with the
question (`q071`, `q074`, `q075`, `q076` in `absent_entity`).

Runs on CPU with `CUDA_VISIBLE_DEVICES` cleared so numbers produced locally and reproduced in
CI come from identical arithmetic.
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extract import best_sentence  # noqa: E402

SYSTEMS = {'plain-rag': {}}


class Adapter:
    """Dense chunk retrieval via e5-base cosine similarity only (no graph, no reranker)."""

    def __init__(self, topk=None):
        self._topk = topk

    def info(self):
        from sqlite_graph_memory import brain_ask as ba
        return {
            'version': 'reference',
            'url': 'https://huggingface.co/intfloat/multilingual-e5-base',
            'config': {
                'embedder': ba.E5_MODEL,
                'reranker': None,
                'graph_hops': 0,
                'topk_retrieve': self._topk or ba.TOPK_RETRIEVE,
                'unit': 'markdown chunks via sqlite_graph_memory.index_notes (MAX_CHARS=1200)',
                'device': 'cpu',
                'answer': 'extractive: _extract.best_sentence of top-3 notes',
                'abstain': 'no sentence shares a content word with the question (no reranker threshold or entity gate)',
            },
        }

    def setup(self, vault_dir, work_dir):
        idx = Path(work_dir) / 'index'
        env = dict(os.environ, BRAIN_INDEX_DIR=str(idx), CUDA_VISIBLE_DEVICES='',
                   PYTHONPATH=str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''))
        subprocess.run([sys.executable, '-m', 'sqlite_graph_memory.index_notes', str(vault_dir)],
                       check=True, env=env, stdout=subprocess.DEVNULL)
        from sqlite_graph_memory import brain_ask as ba
        self.ba = ba
        self.emb, self.meta = ba.load_index(idx / '_brain_e5.npy', idx / '_brain_e5_meta.pkl')
        self.enc = ba.load_encoder('cpu')
        self.raw = {p.stem.lower(): p.read_text(encoding='utf-8') for p in sorted(Path(vault_dir).glob('*.md'))}

    def ask(self, question):
        ba = self.ba
        sims = ba.query_sims(self.enc, self.emb, question)
        cand = ba.retrieve_candidates(sims, self.meta, topk=self._topk or ba.TOPK_RETRIEVE)
        ids, seen = [], set()
        for i in cand:
            nid = Path(self.meta[i]['path']).stem.lower()
            if nid not in seen:
                seen.add(nid)
                ids.append(nid)
        top = float(sims[cand[0]]) if cand else 0.0
        nid, sent, _ = best_sentence(question, ids, self.raw.get)
        if not sent:
            return {'retrieved': ids, 'answer': None, 'citations': [], 'top_score': round(top, 4)}
        return {'retrieved': ids, 'answer': sent, 'citations': [{'note': nid, 'quote': sent}],
                'top_score': round(top, 4)}


def make(**kwargs):
    return Adapter(**kwargs)
