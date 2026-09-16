# -*- coding: utf-8 -*-
"""run_eval.py — score the retrieval pipeline on a frozen gold set. Recall@12 / MRR / nDCG@12.

It imports `brain_ask.py` and calls the SAME functions the agent calls — load_index,
retrieve_candidates, expand_1hop, rerank_candidates — so what you measure is what runs.
An eval that re-implements the pipeline measures the re-implementation.

Every question is scored twice, in both retrieval modes, on the same candidates budget:

    vector   dense retrieve -> rerank            (what you get without the graph)
    graph    dense -> 1-hop [[wikilinks]] -> rerank

The difference between those two columns is the only honest answer to "does the graph
actually help?" — and on our own vault the answer differed by class (see README).

USAGE
    python eval/run_eval.py --gold eval/gold-2026-09-16.jsonl
    python eval/run_eval.py --gold ... --limit 20        # smoke run
    python eval/run_eval.py --selftest                   # metrics only, no model, no index

Config (env): BRAIN_INDEX_DIR (index to score), TURNSTATE_DB (where --sqlite appends a
`gold_eval` row per class/mode so runs accumulate into a history you can diff).

RED-TEST HOOK: with EVAL_MUTANT=1 the nDCG function returns a perfect 1.000 for every
question. Your own test suite should FAIL under that flag. A test that stays green when the
metric is broken never proved anything about the metric.

THRESHOLDS below are engineering defaults from one real vault, not a published benchmark.
Judge a CHANGE by the delta on YOUR numbers, not by whether you beat ours.

LIMITS, stated up front: wikilink gold is weak supervision (a linked neighbour is not
always the note that answers the question); `bridge` is built FROM links, so any method
that walks links is flattered there; one run scores one version of one index; latency is
recorded, never judged.
"""
import argparse
import datetime
import json
import math
import os
import platform
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOPN = 12
THRESHOLDS = {'title': 0.90, 'body': 0.80, 'bridge': 0.70, 'temporal': 0.60}
SMALL_N = 20

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding='utf-8')
    except Exception: pass


# ------------------------------------------------------------------ metrics (pure functions)
def recall_at(ranked, gold, k=TOPN):
    """Share of gold notes present in the top-k."""
    g = set(gold)
    return len(g & set(ranked[:k])) / len(g) if g else 0.0


def mrr_at(ranked, gold, k=TOPN):
    """1/rank of the first gold hit — how far the reader has to scroll."""
    g = set(gold)
    for i, r in enumerate(ranked[:k]):
        if r in g: return 1.0 / (i + 1)
    return 0.0


def ndcg_at(ranked, gold, k=TOPN):
    """Rank-weighted hit rate, 1.0 when every gold note sits at the top."""
    if os.environ.get('EVAL_MUTANT') == '1':
        return 1.0                      # deliberate break; see RED-TEST HOOK above
    g = set(gold)
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(ranked[:k]) if r in g)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(g), k)))
    return dcg / idcg if idcg else 0.0


def selftest():
    """Check the metrics against hand-computed values. No model, no index, no network."""
    ok = True

    def chk(name, got, want):
        nonlocal ok
        if abs(got - want) > 1e-6:
            ok = False; print('FAIL %s: got %.4f want %.4f' % (name, got, want))
        else:
            print('ok   %s = %.4f' % (name, got))

    chk('recall hit', recall_at(['a', 'b', 'c'], ['b']), 1.0)
    chk('recall miss', recall_at(['a', 'b', 'c'], ['z']), 0.0)
    chk('recall half', recall_at(['a', 'b'], ['a', 'z']), 0.5)
    chk('mrr rank 1', mrr_at(['b', 'a'], ['b']), 1.0)
    chk('mrr rank 2', mrr_at(['a', 'b'], ['b']), 0.5)
    chk('ndcg perfect', ndcg_at(['b'], ['b']), 1.0)
    chk('ndcg rank 2', ndcg_at(['a', 'b'], ['b']), 1 / math.log2(3))
    chk('ndcg miss', ndcg_at(['a'], ['b']), 0.0)
    if ndcg_at(['b', 'a'], ['b']) > ndcg_at(['a', 'b'], ['b']):
        print('ok   ndcg rewards a better rank')
    else:
        ok = False; print('FAIL ndcg ignores rank order')
    print('SELFTEST', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


# ------------------------------------------------------------------ runner
def score_gold(gold, progress=True):
    """Run every question through both modes. -> (rows, config dict)."""
    import brain_ask as ba

    dev = ba.pick_device()
    emb, meta = ba.load_index()
    enc = ba.load_encoder(dev)
    ce = ba.load_reranker(dev)
    by_base = ba.index_by_basename(meta)
    stem_of = lambda i: Path(meta[i]['path']).stem.lower()

    rows, t0 = [], time.time()
    for k, g in enumerate(gold):
        tq = time.time()
        sims = ba.query_sims(enc, emb, g['query'])
        base = ba.retrieve_candidates(sims, meta)
        added = ba.expand_1hop(sims, meta, base, by_base)
        res = {}
        for mode, cand in (('vector', base), ('graph', base + added)):
            ranked = [stem_of(i) for i, _ in ba.rerank_candidates(ce, g['query'], cand, meta)]
            res[mode] = {'ranked': ranked,
                         'recall': recall_at(ranked, g['gold']),
                         'mrr': mrr_at(ranked, g['gold']),
                         'ndcg': ndcg_at(ranked, g['gold'])}
        rows.append({**g, 'entity_gate': ba.looks_like_entity(g['query']),
                     'cand_vector': len(base), 'cand_graph_added': len(added),
                     'sec': round(time.time() - tq, 2), 'res': res})
        if progress and (k + 1) % 20 == 0:
            sys.stderr.write('  %d/%d  %.0fs\n' % (k + 1, len(gold), time.time() - t0))
    cfg = {'index_chunks': len(emb), 'embedder': ba.E5_MODEL, 'reranker': ba.RERANK_MODEL,
           'device': str(dev), 'topk': ba.TOPK_RETRIEVE, 'topn': TOPN,
           'sec_per_q': round((time.time() - t0) / max(len(rows), 1), 2)}
    return rows, cfg


def summarize(rows):
    """-> {(class or lang, mode): {n, recall, mrr, ndcg}}."""
    out = {}
    groups = [(cls, [r for r in rows if r['cls'] == cls]) for cls in sorted({r['cls'] for r in rows})]
    groups += [('lang:' + lg, [r for r in rows if r['lang'] == lg]) for lg in ('ru', 'en')]
    for name, rs in groups:
        if not rs: continue
        for mode in ('vector', 'graph'):
            out[(name, mode)] = {'n': len(rs),
                                 'recall': sum(r['res'][mode]['recall'] for r in rs) / len(rs),
                                 'mrr': sum(r['res'][mode]['mrr'] for r in rs) / len(rs),
                                 'ndcg': sum(r['res'][mode]['ndcg'] for r in rs) / len(rs)}
    return out


def print_table(summary, cfg, gold_file):
    print('\nRETRIEVAL EVAL · gold=%s · index=%d chunks · %s + %s · %s · %.2fs/question'
          % (Path(gold_file).name, cfg['index_chunks'], cfg['embedder'].split('/')[-1],
             cfg['reranker'].split('/')[-1], cfg['device'], cfg['sec_per_q']))
    print('%-11s %-7s %4s  %-9s %-6s %-7s  %s'
          % ('class', 'mode', 'n', 'Recall@12', 'MRR', 'nDCG@12', 'threshold'))
    for (name, mode), s in summary.items():
        th = THRESHOLDS.get(name)
        flag = '' if th is None else ('pass >=%.2f' % th if s['recall'] >= th else 'BELOW %.2f' % th)
        if s['n'] < SMALL_N: flag += '  (n<%d: noise)' % SMALL_N
        print('%-11s %-7s %4d  %-9.3f %-6.3f %-7.3f  %s'
              % (name, mode, s['n'], s['recall'], s['mrr'], s['ndcg'], flag))
    for name in ('bridge', 'body', 'title', 'temporal'):
        if (name, 'graph') in summary:
            d = summary[(name, 'graph')]['ndcg'] - summary[(name, 'vector')]['ndcg']
            print('graph effect on %-9s nDCG@12 %+.3f' % (name, d))


def write_sqlite(summary, cfg, gold_file, scores_file):
    """Append one row per class/mode, so a later run can be diffed against this one."""
    db = os.getenv('TURNSTATE_DB', 'turnstate.db')
    try:
        con = sqlite3.connect(str(db), timeout=5.0)
        con.execute("""CREATE TABLE IF NOT EXISTS gold_eval(
            ts TEXT, gold_file TEXT, scores_file TEXT, cls TEXT, mode TEXT, n INTEGER,
            recall12 REAL, mrr REAL, ndcg12 REAL, embedder TEXT, reranker TEXT,
            index_chunks INTEGER, node TEXT)""")
        ts = datetime.datetime.now().isoformat(timespec='seconds')
        for (name, mode), s in summary.items():
            con.execute('INSERT INTO gold_eval VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                        (ts, Path(gold_file).name, Path(scores_file).name, name, mode, s['n'],
                         s['recall'], s['mrr'], s['ndcg'], cfg['embedder'], cfg['reranker'],
                         cfg['index_chunks'], platform.node()))
        con.commit(); con.close()
        return str(db)
    except Exception as e:
        # platform.node(), not os.uname(): os.uname does not exist on Windows, and the
        # AttributeError inside this try is exactly how an eval ends up printing a table
        # while silently writing no history at all.
        sys.stderr.write('sqlite gold_eval not written: %s\n' % e)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gold', help='gold .jsonl from build_gold.py')
    ap.add_argument('--limit', type=int, help='score only the first N questions (smoke run)')
    ap.add_argument('--out', help='scores .jsonl (default: eval/scores-<timestamp>.jsonl)')
    ap.add_argument('--sqlite', action='store_true', help='also append to gold_eval in TURNSTATE_DB')
    ap.add_argument('--no-persist', action='store_true', help='print the table, write nothing')
    ap.add_argument('--selftest', action='store_true', help='check the metrics and exit')
    a = ap.parse_args()

    if a.selftest: return selftest()
    if not a.gold:
        here = sorted(Path(__file__).parent.glob('gold-*.jsonl'))
        if not here: return ap.error('--gold is required (run build_gold.py first)')
        a.gold = str(here[-1])
        print('no --gold given, using the newest: %s' % a.gold)

    gold = [json.loads(l) for l in Path(a.gold).read_text(encoding='utf-8').splitlines() if l.strip()]
    if a.limit: gold = gold[:a.limit]
    if not gold: return ap.error('gold file %s is empty' % a.gold)

    rows, cfg = score_gold(gold)
    summary = summarize(rows)
    print_table(summary, cfg, a.gold)
    if os.environ.get('EVAL_MUTANT') == '1':
        print('\n!! EVAL_MUTANT=1 — nDCG is deliberately broken; these numbers are fiction')

    if a.no_persist:
        print('nothing written (--no-persist)')
        return 0
    out = Path(a.out) if a.out else (Path(__file__).parent /
                                     ('scores-%s.jsonl' % datetime.datetime.now().strftime('%Y%m%d-%H%M')))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')
    print('per-question scores -> %s' % out)
    if a.sqlite:
        db = write_sqlite(summary, cfg, a.gold, out)
        if db: print('history row appended -> %s (table gold_eval)' % db)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
