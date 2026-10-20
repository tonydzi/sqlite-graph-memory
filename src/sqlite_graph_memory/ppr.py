# -*- coding: utf-8 -*-
"""ppr.py: personalized PageRank over the [[wikilink]] graph, fused with the reranker by RRF.

WHY IT EXISTS. On the home vault's frozen gold set (2026-09-16, 60 bridge questions) 97% of
the correct bridge answers were already in the candidate pool, and Recall@12 in that diagnosis
run was 0.225 (the README's 0.251 is the graph-mode baseline of the final measurement):
the cross-encoder dropped them, because the relation is structural, not textual. Adding more
candidates (more hops, larger GMAX) cannot fix that. This module gives the graph a vote in the
ORDERING instead: PPR seeded by the dense hits yields a structural file order, and that order
is fused with the reranker's order by Reciprocal Rank Fusion. Ranks, not scores, because
cosine, cross-encoder logits and PageRank mass are on incomparable scales.

THE HEAD IS PROTECTED. The first PPR_HEAD (5) positions of the reranker are never touched. When
a query names one thing, the right answer is already first, and any shuffling drops it (home
vault: without the protection title nDCG@12 fell by 0.076; with head=5 it rose by 0.006 at the
same bridge gain). PPR only reorders candidates already in the pool; it never brings new files,
because a file outside the pool has no chunk and no score to show.

OFF BY DEFAULT. Turn it on with SGM_PPR=1 (or `sgm-ask --ppr`). The home pipeline runs it on by
default; the package does not, because the measurement behind it was taken on one vault.

CRASH-SAFETY, same contract as the graph hop: no graph, no edges, a PPR failure or a fusion
failure each return the reranker's order unchanged and write one line to stderr. Worst case you
get the old result, never an empty or broken recall.

COST on the home vault (7.8k files, ~40k edges): building the graph 0.5 s once per process
(cached in memory by an index fingerprint), PPR itself ~3 ms per query.

Ported from the maintainer's home module brain_graph.py (2026-09-16) for issue #51.
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

PPR_ALPHA = 0.85     # damping: share of mass that walks the edges. High on purpose: the gain is
                     # in moving AWAY from the hits to their neighbours (0.5 gave half the bridge gain)
PPR_ITERS = 20       # power iteration; converges in <10 on a graph this size
PPR_SEED_TOPK = 15   # how many best dense FILES seed the personalization (same as GHOPS)
PPR_TOPM = 10        # how many PPR files enter the fusion. More is WORSE: a long structural tail
                     # dilutes the top (topm=30 gave bridge +0.053, topm=10 gave +0.103)
PPR_HEAD = 5         # reranker positions the fusion never touches
RRF_K = 60           # RRF constant (Cormack et al. 2009)

_CACHE = {}          # index fingerprint -> Graph, in process memory only


def enabled(env=None):
    """True when SGM_PPR is set to 1/on/yes/true. Default: off."""
    env = os.environ if env is None else env
    return str(env.get('SGM_PPR', '')).strip().lower() in ('1', 'on', 'yes', 'true')


def _links_in(path):
    # One wikilink parser for the whole package: the one the graph hop uses.
    try:
        from sqlite_graph_memory.brain_ask import _links_in as f
    except ImportError:                       # run as a loose script next to brain_ask.py
        from brain_ask import _links_in as f
    return f(path)


class Graph:
    """File-level wikilink graph in CSR form: node = lowercase file stem."""

    def __init__(self, stems, idx, indptr, indices):
        self.stems = stems
        self.idx = idx
        self.indptr = indptr
        self.indices = indices
        self.n_edges = int(len(indices))

    def ppr(self, seed, alpha=None, iters=None):
        """Personalized PageRank: p = alpha * A^T p + (1 - alpha) * seed.

        Dangling nodes (no outgoing links) return their mass to the seed instead of spreading
        it over the whole graph, otherwise the unlinked half of a vault would wash out the
        personalization.
        """
        alpha = PPR_ALPHA if alpha is None else alpha
        iters = PPR_ITERS if iters is None else iters
        n = len(self.stems)
        s = np.asarray(seed, dtype='float64')
        tot = s.sum()
        if n == 0 or tot <= 0:
            return np.zeros(n, dtype='float64')
        s = s / tot
        if self.n_edges == 0:
            return s                          # no edges: PPR is the personalization itself
        p = s.copy()
        deg = np.diff(self.indptr)
        live = deg > 0
        src = np.repeat(np.arange(n), deg)
        for _ in range(iters):
            contrib = np.zeros(n, dtype='float64')
            contrib[live] = p[live] / deg[live]
            nxt = np.zeros(n, dtype='float64')
            np.add.at(nxt, self.indices, contrib[src])
            dangling = float(p[~live].sum())
            p = alpha * nxt + (alpha * dangling + (1.0 - alpha)) * s
        return p


def _fingerprint(meta):
    """sha256 over (path, sha) of every chunk, so the cache expires when the index changes."""
    h = hashlib.sha256()
    for m in meta:
        h.update(str(m.get('path', '')).encode('utf-8', 'replace'))
        h.update(b'\x1f')
        h.update(str(m.get('sha', '')).encode('utf-8', 'replace'))
        h.update(b'\x1e')
    return h.hexdigest()


def build(meta):
    """Build the file graph from the notes' outgoing [[wikilinks]] (self-links and dups dropped).

    A node is a file NAME, because that is what a wikilink addresses: a/index.md and
    b/index.md are one node, and the outgoing links of both files become its edges.
    """
    try:
        fp = _fingerprint(meta)
    except Exception:
        fp = None
    if fp and fp in _CACHE:
        return _CACHE[fp]
    stems, paths, idx = [], [], {}
    for m in meta:
        st = Path(m['path']).stem.lower()
        if st not in idx:
            idx[st] = len(stems)
            stems.append(st)
            paths.append([])
        if m['path'] not in paths[idx[st]]:
            paths[idx[st]].append(m['path'])
    indptr = np.zeros(len(stems) + 1, dtype='int64')
    indices = []
    for u, files in enumerate(paths):
        seen = set()
        for p in files:
            for tgt in _links_in(p):
                v = idx.get(tgt.strip().lower())
                if v is None or v == u or v in seen:
                    continue
                seen.add(v)
                indices.append(v)
        indptr[u + 1] = len(indices)
    g = Graph(stems, idx, indptr, np.asarray(indices, dtype='int64'))
    if fp:
        _CACHE[fp] = g
    return g


def ppr_file_order(meta, sims, base_order, graph=None, topm=None, seed_topk=None):
    """File stems ordered by personalized PageRank, seeded by the dense hits in base_order.

    Seed weight = best chunk cosine + 1, so it is non-negative (PageRank needs a non-negative
    teleport, cosine lives in [-1, 1]). Returns [] on any failure; the caller keeps its order.
    """
    topm = PPR_TOPM if topm is None else topm
    seed_topk = PPR_SEED_TOPK if seed_topk is None else seed_topk
    try:
        g = build(meta) if graph is None else graph
        if not g.stems or g.n_edges == 0:
            return []
        seed = np.zeros(len(g.stems), dtype='float64')
        seen, used = set(), 0
        for i in base_order:
            st = Path(meta[i]['path']).stem.lower()
            if st in seen:
                continue
            seen.add(st)
            u = g.idx.get(st)
            if u is None:
                continue
            seed[u] = max(seed[u], float(sims[i]) + 1.0)   # seed starts at 0: never negative
            used += 1
            if used >= seed_topk:
                break
        if seed.sum() <= 0:
            return []
        p = g.ppr(seed)
        top = np.argsort(-p, kind='stable')[:max(1, topm)]
        return [g.stems[int(u)] for u in top if p[int(u)] > 0]
    except Exception as e:
        sys.stderr.write('PPR lane failed (%s) -> reranker order unchanged\n' % str(e)[:140])
        return []


def rrf_fuse(*ranked_lists, k=RRF_K):
    """Reciprocal Rank Fusion: score(d) = sum 1 / (k + rank_d + 1). Ties break by item value."""
    score, order = {}, []
    for lst in ranked_lists:
        for rank, i in enumerate(lst):
            if i not in score:
                score[i] = 0.0
                order.append(i)
            score[i] += 1.0 / (k + rank + 1)
    order.sort(key=lambda i: (-score[i], i))
    return order


def fuse_after_rerank(ranked_idx, meta, ppr_stems, topn, head=None):
    """Pipeline hook: reorder [(chunk index, score)] from the reranker, then cut to topn.

    `ranked_idx` must be the FULL reranked list, not the top-12: the point is to lift the right
    neighbour out of the tail, which a fusion that only sees the top-12 cannot do. Empty PPR
    returns the first topn exactly as given.

    Every item is kept: two notes with the same file name in different folders are one graph
    node (wikilinks address notes by name) but stay two results, so the output is never
    shorter than the input. Positions, not names, go into RRF, so a tie breaks toward the
    reranker's earlier position instead of the alphabetical order of file names.
    """
    if not ppr_stems or not ranked_idx:
        return list(ranked_idx)[:topn]
    try:
        items = list(ranked_idx)
        head = PPR_HEAD if head is None else head
        stems = [Path(meta[i]['path']).stem.lower() for i, _ in items]
        kept = set(stems[:head])
        first = {}
        for p in range(head, len(items)):
            first.setdefault(stems[p], p - head)
        ppr_pos = [first[s] for s in ppr_stems if s in first and s not in kept]
        if not ppr_pos:
            return items[:topn]
        fused = rrf_fuse(list(range(len(items) - head)), ppr_pos)
        return (items[:head] + [items[head + p] for p in fused])[:topn]
    except Exception as e:
        sys.stderr.write('PPR fusion failed (%s) -> reranker order\n' % str(e)[:140])
        return list(ranked_idx)[:topn]
