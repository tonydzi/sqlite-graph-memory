"""The cross-encoder rerank is opt-in: default recall orders by dense similarity.

Why: on long queries the mmarco cross-encoder lowered recall (DolphinBench Riley slice,
1000 sessions: R@5 0.61 dense vs 0.48 reranked), so 0.3.7 turned it off by default.
"""
import numpy as np
import pytest

from sqlite_graph_memory import brain_ask as ba


def test_rerank_is_off_by_default():
    assert ba.rerank_enabled(env={}, argv=['some question']) is False


@pytest.mark.parametrize('env,argv', [({'SGM_RERANK': '1'}, []), ({}, ['--rerank', 'q'])])
def test_rerank_opt_in(env, argv):
    assert ba.rerank_enabled(env=env, argv=argv) is True


def test_rerank_zero_or_other_values_stay_off():
    assert ba.rerank_enabled(env={'SGM_RERANK': '0'}, argv=[]) is False
    assert ba.rerank_enabled(env={'SGM_RERANK': 'yes'}, argv=[]) is False


def test_no_cross_encoder_keeps_dense_order():
    sims = np.array([0.1, 0.9, 0.5, 0.7])
    meta = [{'title': 't%d' % i, 'snippet': 's'} for i in range(4)]
    out = ba.rerank_candidates(None, 'q', [0, 1, 2, 3], meta, topn=3, sims=sims)
    assert [i for i, _ in out] == [1, 3, 2]


def test_no_cross_encoder_needs_sims():
    with pytest.raises(ValueError):
        ba.rerank_candidates(None, 'q', [0], [{'title': 't', 'snippet': 's'}])


def test_cross_encoder_still_used_when_given():
    class FakeCE:
        def predict(self, pairs):
            return [len(p[1]) for p in pairs]          # longer snippet wins
    meta = [{'title': 'a', 'snippet': 'x'}, {'title': 'b', 'snippet': 'xxxx'}]
    out = ba.rerank_candidates(FakeCE(), 'q', [0, 1], meta, sims=np.array([0.9, 0.1]))
    assert [i for i, _ in out] == [1, 0]


def test_main_does_not_load_reranker_by_default(monkeypatch):
    called = []
    monkeypatch.setattr(ba, 'load_reranker', lambda *a, **k: called.append(1))
    monkeypatch.delenv('SGM_RERANK', raising=False)
    assert ba.rerank_enabled(argv=['plain question']) is False
    ce = ba.load_reranker() if ba.rerank_enabled(argv=['plain question']) else None
    assert ce is None and called == []
