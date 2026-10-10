# -*- coding: utf-8 -*-
"""sqlite-graph-memory itself, through the same functions the agent calls.

Two rows on the board, one per retrieval mode, so the graph has to earn its column:

    sqlite-graph-memory          dense retrieve -> 1-hop [[wikilinks]] -> rerank, with the
                                 entity gate on (what `sgm-ask --graph` does)
    sqlite-graph-memory-vector   dense retrieve -> rerank (what `sgm-ask` does by default)

The library retrieves; it does not answer. Answers and citations come from the shared
extractive floor in _extract.py, and abstention from one rule fixed before the first run:
if the reranker's best score is below ABSTAIN_BELOW, say nothing. 0.0 is the cross-encoder's
own decision boundary, not a number tuned on these questions.

Runs on CPU on purpose (CUDA_VISIBLE_DEVICES is cleared), so a number produced on a laptop
GPU and one reproduced on a CI runner come from the same arithmetic.
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

SYSTEMS = {'sqlite-graph-memory': {'mode': 'graph'},
           'sqlite-graph-memory-vector': {'mode': 'vector'}}
ABSTAIN_BELOW = 0.0


class Adapter:
    def __init__(self, mode):
        assert mode in ('graph', 'vector')
        self.mode = mode

    def info(self):
        from sqlite_graph_memory import brain_ask as ba
        try:
            from importlib.metadata import version
            v = version('sqlite-graph-memory')
        except Exception:
            v = 'source checkout'
        txt = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
        for line in txt.splitlines():
            if line.startswith('version'):
                v = line.split('=', 1)[1].strip().strip('"')
                break
        return {'version': v, 'url': 'https://github.com/tonydzi/sqlite-graph-memory',
                'config': {'mode': self.mode, 'entity_gate': self.mode == 'graph',
                           'embedder': ba.E5_MODEL, 'reranker': ba.RERANK_MODEL,
                           'topk_retrieve': ba.TOPK_RETRIEVE, 'topn': ba.TOPN,
                           'graph_hops': 1, 'graph_seeds': ba.GHOPS, 'graph_max_neighbours': ba.GMAX,
                           'device': 'cpu', 'answer': 'extractive: best-overlap sentence of top-3 notes',
                           'abstain': 'top reranker score < %s' % ABSTAIN_BELOW}}

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
        self.ce = ba.load_reranker('cpu')
        self.by_base = ba.index_by_basename(self.meta)
        self.raw = {p.stem.lower(): p.read_text(encoding='utf-8') for p in Path(vault_dir).glob('*.md')}
        self.graph_added = 0

    def ask(self, question):
        ba = self.ba
        sims = ba.query_sims(self.enc, self.emb, question)
        cand = ba.retrieve_candidates(sims, self.meta)
        if self.mode == 'graph' and not ba.looks_like_entity(question):
            added = ba.expand_1hop(sims, self.meta, cand, self.by_base)
            self.graph_added += len(added)
            cand = cand + added
        ranked = ba.rerank_candidates(self.ce, question, cand, self.meta)
        ids, seen = [], set()
        for i, _ in ranked:
            nid = Path(self.meta[i]['path']).stem.lower()
            if nid not in seen:
                seen.add(nid); ids.append(nid)
        top = float(ranked[0][1]) if ranked else float('-inf')
        nid, sent, _ = best_sentence(question, ids, self.raw.get)
        if top < ABSTAIN_BELOW or not sent:
            return {'retrieved': ids, 'answer': None, 'citations': [], 'top_score': round(top, 3)}
        return {'retrieved': ids, 'answer': sent, 'citations': [{'note': nid, 'quote': sent}],
                'top_score': round(top, 3)}


def make(**kwargs):
    return Adapter(**kwargs)
