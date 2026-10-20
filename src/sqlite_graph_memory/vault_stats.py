#!/usr/bin/env python3
"""sgm-stats -- four health numbers for a folder of linked markdown notes.

    sgm-stats <notes_dir> [--json] [--today YYYY-MM-DD] [--threshold-* N ...]

Measures the graph the retriever actually walks: the same note filter as the indexer
(``iter_notes``: no dot-directories, no sync-conflict copies) and the same wikilink parser
and target resolution as ``brain_ask`` (``_links_in``, basename match, case-insensitive).
A second parser would report on a graph recall never sees.

  orphan rate      pages with no inbound link          ok <5%   warn 5-15%   fail >15%
  avg degree       resolved outgoing links per page    ok 3-8   warn 2-3 / 8-10   fail <2 or >10
  main component   share of pages in the largest       ok >=80%  warn below
                   weakly connected component
  stale concepts   concept pages whose `updated`       warn >50%
                   (or `date`) is older than 90 days

Links to pages that do not exist and links from a page to itself are not edges.
A concept page is one with ``type: concept`` in its frontmatter or one that lives in a
folder named ``concepts``.

Exit code: 0 all ok, 1 any warn, 2 any fail -- so it can sit in cron as a watchdog.
Zero model calls, zero network, stdlib + the two sibling modules.
"""
import argparse
import datetime
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from sqlite_graph_memory.brain_ask import _links_in
from sqlite_graph_memory.index_notes import FM_RX, iter_notes, read_note

DEFAULTS = {
    'orphan_warn': 5.0, 'orphan_fail': 15.0,
    'degree_ok_min': 3.0, 'degree_ok_max': 8.0,
    'degree_warn_min': 2.0, 'degree_warn_max': 10.0,
    'component_ok': 80.0,
    'stale_days': 90, 'stale_warn': 50.0,
}
RANK = {'ok': 0, 'warn': 1, 'fail': 2, 'n/a': 0}

TYPE_RX = re.compile(r'(?m)^type:\s*"?([^"\n]+?)"?\s*$')
UPDATED_RX = re.compile(r'(?m)^updated:\s*"?(\d{4}-\d{2}-\d{2})')
DATE_RX = re.compile(r'(?m)^date:\s*"?(\d{4}-\d{2}-\d{2})')


def build_graph(notes_dir):
    """(pages, edges): pages = {stem_lower: Path}, edges = {(src, dst)} between real pages."""
    pages = {}
    for p in iter_notes(notes_dir):
        pages.setdefault(p.stem.lower(), p)
    edges = set()
    for src, path in pages.items():
        for tgt in _links_in(path):
            dst = tgt.lower()
            if dst in pages and dst != src:
                edges.add((src, dst))
    return pages, edges


def largest_component(pages, edges):
    """Size of the largest weakly connected component (union-find)."""
    parent = {p: p for p in pages}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    sizes = defaultdict(int)
    for p in pages:
        sizes[find(p)] += 1
    return max(sizes.values()) if sizes else 0, len(sizes)


def concept_dates(pages):
    """[(stem, date or None)] for every concept page."""
    out = []
    for stem, path in pages.items():
        text = read_note(path) or ''
        m = FM_RX.match(text)
        fm = m.group(1) if m else ''
        t = TYPE_RX.search(fm)
        is_concept = (t and t.group(1).strip().lower() == 'concept') or \
            any(part.lower() == 'concepts' for part in path.parent.parts)
        if not is_concept:
            continue
        d = UPDATED_RX.search(fm) or DATE_RX.search(fm)
        out.append((stem, datetime.date.fromisoformat(d.group(1)) if d else None))
    return out


def verdict_orphan(v, th):
    return 'fail' if v > th['orphan_fail'] else 'warn' if v >= th['orphan_warn'] else 'ok'


def verdict_degree(v, th):
    if th['degree_ok_min'] <= v <= th['degree_ok_max']:
        return 'ok'
    if th['degree_warn_min'] <= v <= th['degree_warn_max']:
        return 'warn'
    return 'fail'


def compute(notes_dir, today=None, th=None):
    th = {**DEFAULTS, **(th or {})}
    today = today or datetime.date.today()
    pages, edges = build_graph(notes_dir)
    n = len(pages)
    if n == 0:
        raise SystemExit(f"sgm-stats: no notes under {notes_dir}")

    indeg = defaultdict(int)
    for _, dst in edges:
        indeg[dst] += 1
    orphans = sorted(p for p in pages if indeg[p] == 0)
    orphan_pct = 100.0 * len(orphans) / n
    degree = len(edges) / n
    main, n_comp = largest_component(pages, edges)
    main_pct = 100.0 * main / n

    concepts = concept_dates(pages)
    dated = [(s, d) for s, d in concepts if d is not None]
    cutoff = today - datetime.timedelta(days=int(th['stale_days']))
    stale = sorted(s for s, d in dated if d < cutoff)
    if dated:
        stale_pct = 100.0 * len(stale) / len(dated)
        stale_v = 'warn' if stale_pct > th['stale_warn'] else 'ok'
    else:
        stale_pct, stale_v = None, 'n/a'

    metrics = [
        {'name': 'orphan_rate', 'value': round(orphan_pct, 1), 'unit': '%',
         'verdict': verdict_orphan(orphan_pct, th),
         'detail': f"{len(orphans)} of {n} pages have no inbound link",
         'examples': orphans[:5]},
        {'name': 'avg_degree', 'value': round(degree, 2), 'unit': 'links/page',
         'verdict': verdict_degree(degree, th),
         'detail': f"{len(edges)} resolved links over {n} pages"},
        {'name': 'main_component', 'value': round(main_pct, 1), 'unit': '%',
         'verdict': 'ok' if main_pct >= th['component_ok'] else 'warn',
         'detail': f"{main} of {n} pages; {n_comp} component{'' if n_comp == 1 else 's'}"},
        {'name': 'stale_concepts',
         'value': None if stale_pct is None else round(stale_pct, 1), 'unit': '%',
         'verdict': stale_v,
         'detail': (f"{len(stale)} of {len(dated)} dated concept pages older than "
                    f"{int(th['stale_days'])} days (as of {today.isoformat()})"
                    + (f"; {len(concepts) - len(dated)} undated" if len(dated) < len(concepts) else '')
                    if concepts else "no concept pages"),
         'examples': stale[:5]},
    ]
    worst = max(RANK[m['verdict']] for m in metrics)
    return {'notes_dir': str(notes_dir), 'pages': n, 'metrics': metrics,
            'verdict': ['ok', 'warn', 'fail'][worst], 'exit_code': worst}


def render(report):
    lines = [f"{report['pages']} pages in {report['notes_dir']}", '']
    for m in report['metrics']:
        val = 'n/a' if m['value'] is None else f"{m['value']}{'%' if m['unit'] == '%' else ''}"
        lines.append(f"{m['verdict'].upper():<5} {m['name']:<15} {val:>7}   {m['detail']}")
    lines += ['', f"verdict: {report['verdict']}"]
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='sgm-stats', description=__doc__.split('\n\n')[0])
    ap.add_argument('notes_dir')
    ap.add_argument('--json', action='store_true', help='machine-readable output')
    ap.add_argument('--today', type=datetime.date.fromisoformat,
                    help='date to measure staleness against (YYYY-MM-DD; default: today)')
    for key, val in DEFAULTS.items():
        ap.add_argument('--threshold-' + key.replace('_', '-'), dest=key, type=float,
                        default=val, help=f'default {val}')
    a = ap.parse_args(argv)
    if not Path(a.notes_dir).is_dir():
        ap.error(f"not a directory: {a.notes_dir}")
    th = {k: getattr(a, k) for k in DEFAULTS}
    report = compute(a.notes_dir, today=a.today, th=th)
    print(json.dumps(report, indent=2, ensure_ascii=False) if a.json else render(report))
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
