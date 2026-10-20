# -*- coding: utf-8 -*-
"""The PageRank lane: what it moves, what it must never move, and that it fails closed.

The protection test is the one that matters. A hub note (everything links to it) and its
neighbours get the most PageRank mass; without the top-5 guard, RRF lifts ten of them over the
reranker's rank-1 hit and pushes it to position 11. Remove the guard and
`test_hub_cannot_push_reranker_rank1_out_of_top5` goes red.

No model download, no network: real wikilink files, a hand-made similarity vector, and a fake
cross-encoder.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'eval'))

from sqlite_graph_memory import brain_ask as ba  # noqa: E402
from sqlite_graph_memory import ppr  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_cache():
    ppr._CACHE.clear()
    yield
    ppr._CACHE.clear()


def _vault(tmp_path, links):
    """links: {stem: [targets]} -> meta list with one chunk per note, in dict order."""
    meta = []
    for stem, tgts in links.items():
        p = tmp_path / ('%s.md' % stem)
        p.write_text('# %s\n\n' % stem + ' '.join('[[%s]]' % t for t in tgts) + '\n',
                     encoding='utf-8')
        meta.append({'path': str(p), 'title': stem, 'snippet': stem, 'sha': stem})
    return meta


def _hub_vault(tmp_path):
    """15 seed notes all link to `hub` and to c00..c09; hub links to the cluster too.
    `answer` is the reranker's rank-1 but only dense rank 16, so it is not a PPR seed.
    20 filler notes link nowhere."""
    cluster = ['c%02d' % i for i in range(10)]
    links = {}
    for i in range(15):
        links['seed%02d' % i] = ['hub'] + cluster
    links['answer'] = []
    links['hub'] = cluster
    for c in cluster:
        links[c] = ['hub']
    for i in range(20):
        links['filler%02d' % i] = []
    meta = _vault(tmp_path, links)
    stems = [m['title'] for m in meta]
    sims = np.zeros(len(meta))
    for k, st in enumerate(stems):           # dense: seeds first, then answer, then the rest
        sims[k] = 0.9 - 0.01 * k if st.startswith('seed') else 0.3
    sims[stems.index('answer')] = 0.5
    base_order = sorted(range(len(meta)), key=lambda i: -sims[i])
    # The reranker: answer first, then fillers, then the hub cluster at the tail.
    rr = (['answer'] + ['filler%02d' % i for i in range(20)] + ['seed%02d' % i for i in range(15)]
          + cluster + ['hub'])
    ranked_full = [(stems.index(s), float(len(rr) - r)) for r, s in enumerate(rr)]
    return meta, sims, base_order, ranked_full, stems


# ------------------------------------------------------------------ the guard
def test_hub_cannot_push_reranker_rank1_out_of_top5(tmp_path):
    meta, sims, base_order, ranked_full, stems = _hub_vault(tmp_path)
    lane = ppr.ppr_file_order(meta, sims, base_order)
    assert lane[0] == 'hub'                  # the hub really is the PPR favourite
    assert 'answer' not in lane              # and rank-1 gets no PPR help
    out = [meta[i]['title'] for i, _ in ppr.fuse_after_rerank(ranked_full, meta, lane, 12)]
    rr_top5 = [meta[i]['title'] for i, _ in ranked_full[:5]]
    assert out[0] == 'answer'
    assert out[:5] == rr_top5                # the head is left exactly as the reranker had it


def test_without_the_guard_the_hub_cluster_does_displace_rank1(tmp_path):
    # The scenario above is a real threat, not a strawman: head=0 shows what the guard stops.
    meta, sims, base_order, ranked_full, _ = _hub_vault(tmp_path)
    lane = ppr.ppr_file_order(meta, sims, base_order)
    out = [meta[i]['title'] for i, _ in ppr.fuse_after_rerank(ranked_full, meta, lane, 12, head=0)]
    assert 'answer' not in out[:5]


def test_ppr_lifts_a_tail_neighbour_into_the_top12(tmp_path):
    meta, sims, base_order, ranked_full, _ = _hub_vault(tmp_path)
    before = [meta[i]['title'] for i, _ in ranked_full[:12]]
    lane = ppr.ppr_file_order(meta, sims, base_order)
    after = [meta[i]['title'] for i, _ in ppr.fuse_after_rerank(ranked_full, meta, lane, 12)]
    assert 'hub' not in before and 'hub' in after
    assert len(after) == 12 and len(set(after)) == 12


def test_fusion_never_brings_files_from_outside_the_pool():
    meta = [{'path': '/v/%s.md' % s} for s in 'abcdef']
    ranked = [(i, 1.0) for i in range(6)]
    out = ppr.fuse_after_rerank(ranked, meta, ['zzz', 'f', 'e'], 12, head=2)
    assert sorted(i for i, _ in out) == list(range(6))
    assert [i for i, _ in out][:2] == [0, 1]


# ------------------------------------------------------------------ off by default
@pytest.mark.parametrize('env,want', [({}, False), ({'SGM_PPR': ''}, False),
                                      ({'SGM_PPR': '0'}, False), ({'SGM_PPR': '1'}, True),
                                      ({'SGM_PPR': 'on'}, True)])
def test_lane_is_off_unless_asked(env, want):
    assert ppr.enabled(env) is want
    assert ba.ppr_enabled(env) is want


# ------------------------------------------------------------------ crash-safety
def test_graph_without_edges_returns_the_reranker_order(tmp_path):
    meta = _vault(tmp_path, {'a': [], 'b': [], 'c': []})
    lane = ppr.ppr_file_order(meta, np.array([0.9, 0.5, 0.1]), [0, 1, 2])
    assert lane == []
    ranked = [(2, 3.0), (0, 2.0), (1, 1.0)]
    assert ppr.fuse_after_rerank(ranked, meta, lane, 12) == ranked


def test_broken_meta_degrades_to_empty_lane(capsys):
    assert ppr.ppr_file_order([{'no_path': 1}], np.array([1.0]), [0]) == []
    assert 'reranker order unchanged' in capsys.readouterr().err


def test_broken_fusion_returns_reranker_top_n(capsys):
    ranked = [(0, 2.0), (1, 1.0)]
    assert ppr.fuse_after_rerank(ranked, [{'nope': 1}], ['x'], 1) == [(0, 2.0)]
    assert 'reranker order' in capsys.readouterr().err


def test_brain_ask_hooks_survive_a_missing_module(monkeypatch, capsys):
    def boom():
        raise ImportError('no ppr module')
    monkeypatch.setattr(ba, '_ppr', boom)
    ranked = [(i, float(-i)) for i in range(20)]
    assert ba.ppr_lane([], np.array([]), []) == []
    assert ba.fuse_ppr(ranked, [], ['x'], 12) == ranked[:12]
    assert ba.ppr_enabled({'SGM_PPR': '1'}) is False
    assert 'fell back' in capsys.readouterr().err


# ------------------------------------------------------------------ the graph mode is untouched
class _FakeCE:
    def predict(self, pairs):
        return [len(t) % 7 + 0.001 * k for k, (_, t) in enumerate(pairs)]


def test_full_rerank_cut_at_12_equals_the_graph_mode():
    meta = [{'title': 't%d' % i * (i % 5 + 1), 'snippet': 's'} for i in range(40)]
    cand = list(range(40))
    full = ba.rerank_candidates(_FakeCE(), 'q', cand, meta, topn=None)
    assert len(full) == 40
    assert full[:12] == ba.rerank_candidates(_FakeCE(), 'q', cand, meta)


def test_summarize_reports_ppr_as_its_own_mode():
    from run_eval import summarize
    r = {'recall': 1.0, 'mrr': 1.0, 'ndcg': 1.0}
    rows = [{'cls': 'bridge', 'lang': 'en', 'res': {'vector': r, 'graph': r, 'ppr': r}}]
    assert ('bridge', 'ppr') in summarize(rows)
    rows[0]['res'].pop('ppr')
    assert ('bridge', 'ppr') not in summarize(rows)
