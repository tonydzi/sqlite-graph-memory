# -*- coding: utf-8 -*-
"""run_ablation.py — sweep retrieval-pipeline parameters over one gold file in one run.

Scores every grid cell on the SAME gold set in a single pass, reusing the exact
pipeline functions eval/run_eval.py and the agent both call (brain_ask.py) — an
ablation runner that re-implements retrieval measures the re-implementation.

Axes swept per cell:
    hops          0 (vector-only) or 1 (+ 1-hop [[wikilink]] graph expansion)
    ghops         how many of the top dense hits seed the graph hop (brain_ask.GHOPS)
    gmax          max graph-expanded neighbours admitted (brain_ask.GMAX)
    topk          dense-retrieve candidate pool size (brain_ask.TOPK_RETRIEVE)
    entity_gate   on/off: suppress the hop on entity/name-shaped queries (looks_like_entity)

Per cell/class, the summary also reports:
    gate_rate         share of queries where this cell's entity_gate fired (query-shape
                      only — computed even on hops=0 cells, where it has no effect)
    hop_applied_rate  share of queries where expand_1hop actually admitted >=1 new
                      candidate. A metric can be unchanged between an entity-gate-on and
                      an entity-gate-off cell for two very different reasons — the gate
                      firing, or the hop firing but adding nothing the reranker needed
                      (issue #93) — and only hop_applied_rate tells those apart.
    rank_churn        share of queries where this cell's top-12 differs at all from the
                      0-hop baseline's, for the same query. A single-note gold pins
                      rank #1 regardless of what fills ranks 2..12, so delta_ndcg can
                      read 0.0 while the hop still reordered the rest of the page.

The dense embedding and its similarity scores are computed ONCE per query (the model
load and the encode() call are the expensive parts) and reused across every cell: a
cell's dense pool is a prefix of the max-topk pool, sliced rather than re-retrieved.
Only the cross-encoder rerank, which depends on each cell's candidate set, runs per
(query, cell).

USAGE
    python eval/run_ablation.py --list-grid                      # cells only, no model/index
    python eval/run_ablation.py --gold eval/gold-2026-09-16.jsonl
    python eval/run_ablation.py --gold ... --grid default --out eval/ablation-run1

Config (env): BRAIN_INDEX_DIR (index to score), same as run_eval.py / brain_ask.py.
"""
import argparse
import csv
import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlite_graph_memory import brain_ask as ba  # noqa: E402
from run_eval import SMALL_N, mrr_at, ndcg_at, recall_at  # noqa: E402


def _default_cell():
    return {'hops': 1, 'ghops': ba.GHOPS, 'gmax': ba.GMAX, 'topk': ba.TOPK_RETRIEVE,
            'entity_gate': True}


def _fill_cell(partial, index):
    cell = _default_cell()
    cell.update(partial)
    if not cell.get('name'):
        cell['name'] = 'cell-%d' % index
    if cell['hops'] not in (0, 1):
        raise ValueError("cell %r: hops must be 0 or 1 (score_grid only runs the 1-hop "
                         "expand_1hop), got %r" % (cell['name'], cell['hops']))
    return cell


DEFAULT_GRID = [
    _fill_cell({'name': '0-hop', 'hops': 0}, 0),
    _fill_cell({'name': '1-hop'}, 1),
    _fill_cell({'name': 'gmax-10', 'gmax': 10}, 2),
    _fill_cell({'name': 'gmax-20', 'gmax': 20}, 3),
    _fill_cell({'name': 'gmax-80', 'gmax': 80}, 4),
    _fill_cell({'name': 'topk-20', 'topk': 20}, 5),
    _fill_cell({'name': 'topk-80', 'topk': 80}, 6),
    _fill_cell({'name': 'ghops-5', 'ghops': 5}, 7),
    _fill_cell({'name': 'ghops-30', 'ghops': 30}, 8),
    _fill_cell({'name': 'entity-gate-off', 'entity_gate': False}, 9),
]
GRID_PRESETS = {'default': DEFAULT_GRID}

_BOOL_KEYS = {'entity_gate'}
_STR_KEYS = {'name'}


def parse_grid(spec):
    """'default' | a path or literal JSON array of cell dicts | ';'-separated key=val cells."""
    spec = (spec or 'default').strip()
    if spec in GRID_PRESETS:
        return GRID_PRESETS[spec]

    path = Path(spec)
    if path.exists():
        spec = path.read_text(encoding='utf-8')

    if spec.lstrip().startswith('['):
        return [_fill_cell(c, i) for i, c in enumerate(json.loads(spec))]

    cells = []
    for i, part in enumerate(spec.split(';')):
        partial = {}
        for pair in part.split(','):
            if not pair.strip():
                continue
            k, _, v = pair.partition('=')
            k, v = k.strip(), v.strip()
            if k in _BOOL_KEYS:
                partial[k] = v.lower() not in ('0', 'false', 'no', '')
            elif k in _STR_KEYS:
                partial[k] = v
            else:
                partial[k] = int(v)
        cells.append(_fill_cell(partial, i))
    return cells


def print_grid(grid):
    print('%-18s %4s %5s %4s %4s  %s' % ('cell', 'hops', 'ghops', 'gmax', 'topk', 'entity_gate'))
    for c in grid:
        print('%-18s %4d %5d %4d %4d  %s'
              % (c['name'], c['hops'], c['ghops'], c['gmax'], c['topk'], c['entity_gate']))


def load_gold(path):
    """Same validation as run_eval.py's --gold handling: a typo reads like a message."""
    gold_path = Path(path)
    if not gold_path.exists():
        sys.exit('no gold file at %s — build one with `python eval/build_gold.py <notes_dir>`'
                 % gold_path)
    try:
        lines = gold_path.read_text(encoding='utf-8').splitlines()
        gold = [json.loads(l) for l in lines if l.strip()]
    except json.JSONDecodeError as e:
        sys.exit('%s is not a gold file (expected one JSON object per line): %s' % (gold_path, e))
    if not gold:
        sys.exit('gold file %s is empty' % gold_path)
    return gold


def score_grid(gold, grid, emb, meta, enc, ce, by_base=None):
    """Score every grid cell against gold.
    -> [{cell, cls, recall, mrr, ndcg, gated, hop_applied, rank_changed}, ...].

    `gated` is whether this cell's entity_gate suppressed the hop for this query,
    independent of `hops` (it reflects what the gate alone would have done).
    `hop_applied` is whether expand_1hop actually admitted >=1 new candidate for
    this (query, cell) — the thing a metric delta can't show when the gold note
    was already in `base` (issue #93: gate-on and gate-off score identically
    because admission, not suppression, is what moved).
    `rank_changed` is whether this cell's top-12 ranking differs at all from the
    0-hop baseline cell's, for this same query — a single-note gold pins rank #1
    regardless of what fills ranks 2..12, so recall/mrr/ndcg can read unchanged
    (delta_ndcg == 0.0) while the hop still reordered everything below it.

    One dense pass (query_sims + the max-topk retrieve) per query; only the rerank,
    which depends on each cell's own candidate set, runs once per (query, cell).
    """
    by_base = by_base or ba.index_by_basename(meta)
    stem_of = lambda i: Path(meta[i]['path']).stem.lower()
    max_topk = max(c['topk'] for c in grid)
    baseline_name = next((c['name'] for c in grid if c['hops'] == 0), grid[0]['name'])

    rows = []
    for g in gold:
        sims = ba.query_sims(enc, emb, g['query'])
        order_max = ba.retrieve_candidates(sims, meta, topk=max_topk)
        is_entity = ba.looks_like_entity(g['query'])
        cell_results = []
        for cell in grid:
            base = order_max[:cell['topk']]
            gated = cell['entity_gate'] and is_entity
            added = (ba.expand_1hop(sims, meta, base, by_base, ghops=cell['ghops'], gmax=cell['gmax'])
                     if cell['hops'] and base and not gated else [])
            scored = ba.rerank_candidates(ce, g['query'], base + added, meta, sims=sims)
            ranked = [stem_of(i) for i, _ in scored]
            cell_results.append((cell, ranked, gated, added))
        base_ranked = next((r for c, r, _, _ in cell_results if c['name'] == baseline_name), None)
        for cell, ranked, gated, added in cell_results:
            rows.append({'cell': cell['name'], 'cls': g['cls'],
                         'recall': recall_at(ranked, g['gold']),
                         'mrr': mrr_at(ranked, g['gold']),
                         'ndcg': ndcg_at(ranked, g['gold']),
                         'gated': gated, 'hop_applied': bool(added),
                         'rank_changed': base_ranked is not None and ranked != base_ranked})
    return rows


def summarize_grid(rows, grid, classes):
    """-> ordered rows of {cell, class, n, recall, mrr, ndcg, delta_ndcg, gate_rate,
    hop_applied_rate, rank_churn}, one per (cell, class) plus (cell, 'overall');
    delta_ndcg is vs the 0-hop baseline cell (the grid's first hops=0 cell, else its
    first cell) for the same class. gate_rate/hop_applied_rate/rank_churn default to
    0.0 for rows lacking those keys (older score_grid output, or hand-built rows in a
    caller's own tests)."""
    baseline = next((c['name'] for c in grid if c['hops'] == 0), grid[0]['name'])
    by_cell_cls = {}
    for r in rows:
        by_cell_cls.setdefault((r['cell'], r['cls']), []).append(r)

    def agg(rs):
        n = len(rs)
        return {'n': n, 'recall': sum(r['recall'] for r in rs) / n,
                'mrr': sum(r['mrr'] for r in rs) / n, 'ndcg': sum(r['ndcg'] for r in rs) / n,
                'gate_rate': sum(r.get('gated', False) for r in rs) / n,
                'hop_applied_rate': sum(r.get('hop_applied', False) for r in rs) / n,
                'rank_churn': sum(r.get('rank_changed', False) for r in rs) / n}

    out = []
    for cell in grid:
        for cls in list(classes) + ['overall']:
            rs = ([r for r in rows if r['cell'] == cell['name']] if cls == 'overall'
                  else by_cell_cls.get((cell['name'], cls), []))
            if not rs:
                continue
            row = agg(rs)
            base_rs = ([r for r in rows if r['cell'] == baseline] if cls == 'overall'
                       else by_cell_cls.get((baseline, cls), []))
            row['delta_ndcg'] = row['ndcg'] - agg(base_rs)['ndcg'] if base_rs else None
            out.append({'cell': cell['name'], 'class': cls, **row})
    return out


def format_markdown(summary):
    lines = ['| cell | class | n | Recall@12 | MRR | nDCG@12 | delta_nDCG@12 | gate_rate | hop_applied_rate | rank_churn |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in summary:
        d = '' if r['delta_ndcg'] is None else '%+.3f' % r['delta_ndcg']
        lines.append('| %s | %s | %d | %.3f | %.3f | %.3f | %s | %.3f | %.3f | %.3f |'
                     % (r['cell'], r['class'], r['n'], r['recall'], r['mrr'], r['ndcg'], d,
                        r['gate_rate'], r['hop_applied_rate'], r['rank_churn']))
    return '\n'.join(lines)


def write_csv(summary, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['cell', 'class', 'n', 'Recall@12', 'MRR', 'nDCG@12', 'delta_nDCG@12',
                    'gate_rate', 'hop_applied_rate', 'rank_churn'])
        for r in summary:
            w.writerow([r['cell'], r['class'], r['n'], round(r['recall'], 4), round(r['mrr'], 4),
                       round(r['ndcg'], 4), '' if r['delta_ndcg'] is None else round(r['delta_ndcg'], 4),
                       round(r['gate_rate'], 4), round(r['hop_applied_rate'], 4),
                       round(r['rank_churn'], 4)])


def low_n_warnings(summary, small_n=SMALL_N):
    return ['WARNING: cell=%s class=%s has n=%d (< %d) — treat as noise'
           % (r['cell'], r['class'], r['n'], small_n)
           for r in summary if r['n'] < small_n]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--gold', help='gold .jsonl from build_gold.py')
    ap.add_argument('--grid', default='default',
                    help="'default', a JSON array of cells, a path to one, or "
                         "';'-separated key=val cells (default: %(default)s)")
    ap.add_argument('--list-grid', action='store_true',
                    help='print the named grid cells and exit (no model/index load)')
    ap.add_argument('--out', help='output prefix for <prefix>.csv/.md (default: eval/ablation-<ts>)')
    ap.add_argument('--csv', help='CSV path (overrides the --out-derived one)')
    ap.add_argument('--md', help='Markdown path (overrides the --out-derived one)')
    ap.add_argument('--rerank', action='store_true',
                    help='score with the cross-encoder rerank (off by default; same as SGM_RERANK=1)')
    a = ap.parse_args()
    if a.rerank:
        os.environ['SGM_RERANK'] = '1'

    grid = parse_grid(a.grid)
    if a.list_grid:
        print_grid(grid)
        return 0
    if not a.gold:
        return ap.error('--gold is required (run build_gold.py first)')

    gold = load_gold(a.gold)
    classes = sorted({g['cls'] for g in gold})

    dev = ba.pick_device()
    emb, meta = ba.load_index()
    enc = ba.load_encoder(dev)
    ce = ba.load_reranker(dev) if ba.rerank_enabled() else None
    by_base = ba.index_by_basename(meta)

    rows = score_grid(gold, grid, emb, meta, enc, ce, by_base)
    summary = summarize_grid(rows, grid, classes)

    md = format_markdown(summary)
    print(md)
    for w in low_n_warnings(summary):
        print(w)

    prefix = a.out or str(Path(__file__).parent / ('ablation-%s' % datetime.datetime.now().strftime('%Y%m%d-%H%M')))
    csv_path = Path(a.csv) if a.csv else Path(prefix + '.csv')
    md_path = Path(a.md) if a.md else Path(prefix + '.md')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(summary, csv_path)
    md_path.write_text(md + '\n', encoding='utf-8')
    print('\nwritten -> %s, %s' % (csv_path, md_path))
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
