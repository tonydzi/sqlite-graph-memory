# -*- coding: utf-8 -*-
"""run.py — run one memory system over the bench and write its result file.

    python bench/run.py --system bm25                          # ~1 s, no model
    python bench/run.py --system sqlite-graph-memory           # needs the [embeddings] extra
    python bench/run.py --system sqlite-graph-memory-vector
    python bench/run.py --system my-system --limit 5           # smoke run, writes nothing
    python bench/run.py --list                                 # every system the adapters expose

A system is found by name in the SYSTEMS dict of any bench/adapters/*.py (start from
template.py). The result goes to bench/results/<name>.json, then:

    python bench/leaderboard.py      # regenerate LEADERBOARD.md
    python bench/validate.py         # what CI runs on your pull request
"""
import argparse
import datetime
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_core as bc  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding='utf-8')
    except Exception: pass


def adapter_modules():
    for p in sorted((HERE / 'adapters').glob('*.py')):
        if p.name.startswith('_'):
            continue
        spec = importlib.util.spec_from_file_location('bench_adapter_' + p.stem, p)
        yield p, spec


def find_system(name):
    for p, spec in adapter_modules():
        src = p.read_text(encoding='utf-8')
        if repr(name) not in src and '"%s"' % name not in src:
            continue                      # avoid importing heavy adapters we do not need
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if name in getattr(mod, 'SYSTEMS', {}):
            return p, mod, mod.SYSTEMS[name]
    sys.exit('no adapter exposes a system named %r — see `python bench/run.py --list`' % name)


def print_table(m):
    o = m['overall']
    print('\n%-18s %5s %6s %6s %7s %8s %8s %8s %7s' % ('class', 'n', 'R@5', 'R@10', 'ans', 'cite_vb', 'faithful', 'abstain', 'f_abst'))
    rows = list(m['by_class'].items()) + [('OVERALL', o)]
    f = lambda x: '   -  ' if x is None else '%.3f' % x
    for cls, s in rows:
        print('%-18s %5d %6s %6s %7s %8s %8s %8s %7s' % (cls, s['n'], f(s['recall@5']), f(s['recall@10']),
              f(s['answer_acc']), f(s['citation_verbatim']), f(s['citation_faithful']),
              f(s['abstain_correct']), f(s['false_abstain'])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--system', help='system name from an adapter SYSTEMS dict')
    ap.add_argument('--limit', type=int, help='first N questions only; nothing is written')
    ap.add_argument('--out', help='result path (default bench/results/<system>.json)')
    ap.add_argument('--submitted-by', default='tonydzi', help='your GitHub handle')
    ap.add_argument('--notes', default='', help='honest caveats, shown with the result')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()

    if a.list:
        for p, spec in adapter_modules():
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                print('%-28s %s' % (p.name, ', '.join(getattr(mod, 'SYSTEMS', {}))))
            except Exception as e:
                print('%-28s (does not import: %s)' % (p.name, e))
        return 0
    if not a.system:
        ap.error('--system is required')

    path, mod, kwargs = find_system(a.system)
    questions = bc.load_questions()
    if a.limit:
        questions = questions[:a.limit]
    adapter = mod.make(**kwargs)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as work:
        adapter.setup(bc.VAULT, Path(work))
        t_setup = time.time() - t0
        per_q = []
        for k, q in enumerate(questions):
            tq = time.time()
            out = adapter.ask(q['question']) or {}
            row = {'id': q['id'], 'retrieved': [str(x) for x in (out.get('retrieved') or [])][:bc.MAX_RETRIEVED],
                   'answer': out.get('answer'), 'citations': (out.get('citations') or [])[:bc.MAX_CITATIONS],
                   'sec': round(time.time() - tq, 3)}
            for extra in ('top_score',):
                if extra in out:
                    row[extra] = out[extra]
            per_q.append(row)
            if (k + 1) % 25 == 0:
                sys.stderr.write('  %d/%d\n' % (k + 1, len(questions)))
    vault = bc.load_vault()
    metrics = bc.score(questions, per_q, vault)
    print_table(metrics)
    if a.limit:
        print('\n--limit given: smoke run, nothing written')
        return 0

    info = adapter.info()
    rel = path.relative_to(HERE.parent).as_posix()
    res = {
        'schema': bc.SCHEMA, 'bench_version': bc.BENCH_VERSION,
        'system': {'name': a.system, 'version': info['version'], 'url': info['url'],
                   'adapter': rel, 'config': info.get('config', {})},
        'submitted_by': a.submitted_by,
        'date': datetime.date.today().isoformat(),
        'vault_sha256': bc.tree_sha256(bc.VAULT),
        'questions_sha256': bc.file_sha256(bc.QUESTIONS),
        'reproduce': {'in_ci': a.system in bc.ci_reproduced(), 'command': 'python bench/run.py --system %s' % a.system},
        'runtime': {'setup_sec': round(t_setup, 1),
                    'sec_per_question': round(sum(r['sec'] for r in per_q) / max(len(per_q), 1), 3),
                    'python': sys.version.split()[0]},
        'notes': a.notes,
        'metrics': metrics,
        'per_question': per_q,
    }
    out = Path(a.out) if a.out else bc.RESULTS / ('%s.json' % a.system)
    if out.exists():
        try:   # keep the caveats a previous run recorded
            old = json.loads(out.read_text(encoding='utf-8'))
            if not a.notes:
                res['notes'] = old.get('notes', '')
        except Exception:
            pass
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('\nresult -> %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
