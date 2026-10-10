# -*- coding: utf-8 -*-
"""validate.py — the CI gate for bench/results/*.json. Stdlib only, no model.

    python bench/validate.py                         # every result file + vault + leaderboard
    python bench/validate.py bench/results/x.json    # one file
    python bench/validate.py --selftest              # prove the gate rejects forgeries
    python bench/validate.py --reproduce bm25 [--tolerance 0.02]
                                                     # rerun a system, compare with its file

What "valid" means:
  1. the vault and questions were not hand-edited (make_vault.py --check),
  2. each result has the right shape and was run against THIS vault and question set
     (hashes), and covers every question exactly once,
  3. every headline number is re-derived from the per-question outputs by the same
     scorer run.py used — a number that does not follow from its outputs is a forgery,
  4. LEADERBOARD.md is exactly what leaderboard.py generates from those files.

What it cannot catch: per-question outputs that were themselves made up. That is why a
result says whether CI reproduced it (`reproduce.in_ci`), and the board shows it.
"""
import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_core as bc  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding='utf-8')
    except Exception: pass


def check_files(files):
    questions, vault = bc.load_questions(), bc.load_vault()
    vs, qs = bc.tree_sha256(bc.VAULT), bc.file_sha256(bc.QUESTIONS)
    bad = 0
    for f in files:
        try:
            res = json.loads(Path(f).read_text(encoding='utf-8'))
        except Exception as e:
            print('FAIL %s: not JSON (%s)' % (f, e)); bad += 1; continue
        errs = bc.check_result(res, questions, vault, vs, qs, filename=f)
        if errs:
            bad += 1
            print('FAIL %s' % f)
            for e in errs[:20]:
                print('     - %s' % e)
            if len(errs) > 20:
                print('     ... and %d more' % (len(errs) - 20))
        else:
            o = res['metrics']['overall']
            print('ok   %s  (R@5 %.3f, abstain %.3f, CI-reproduced: %s)'
                  % (Path(f).name, o['recall@5'] or 0, o['abstain_correct'] or 0, res['reproduce']['in_ci']))
    return bad


def check_leaderboard():
    import leaderboard
    want = leaderboard.render()
    have = (HERE / 'LEADERBOARD.md').read_text(encoding='utf-8') if (HERE / 'LEADERBOARD.md').exists() else ''
    if have.replace('\r\n', '\n') != want:
        print('FAIL LEADERBOARD.md is stale: run `python bench/leaderboard.py` and commit the result')
        return 1
    print('ok   LEADERBOARD.md matches the result files')
    return 0


def selftest():
    """Forge a copy of a real result nine ways; each forgery must be rejected."""
    files = sorted(bc.RESULTS.glob('*.json'))
    if not files:
        print('selftest needs at least one result file'); return 1
    base = json.loads(files[0].read_text(encoding='utf-8'))
    q, v = bc.load_questions(), bc.load_vault()
    vs, qs = bc.tree_sha256(bc.VAULT), bc.file_sha256(bc.QUESTIONS)
    if bc.check_result(base, q, v, vs, qs, filename=files[0]):
        print('FAIL selftest: the untouched file is already invalid'); return 1

    def forge_headline(r):
        v = r['metrics']['overall']['recall@5'] or 0
        r['metrics']['overall']['recall@5'] = round(v - 0.1 if v >= 0.5 else v + 0.1, 4)   # must MOVE

    def forge_class(r):
        c = next(iter(r['metrics']['by_class']))
        r['metrics']['by_class'][c]['abstain_correct'] = 1.0 if r['metrics']['by_class'][c]['abstain_correct'] != 1.0 else 0.0

    def forge_quote(r):
        for o in r['per_question']:
            o['citations'] = [{'note': o['retrieved'][0] if o['retrieved'] else 'home', 'quote': 'A sentence no note contains.'}]
        # leave metrics as they were: faithfulness now drops and the claim no longer matches

    def forge_drop(r):
        r['per_question'] = r['per_question'][1:]

    def forge_vault(r):
        r['vault_sha256'] = '0' * 64

    def forge_ci_claim(r):
        r['reproduce']['in_ci'] = not r['reproduce']['in_ci']

    def forge_extra_class(r):
        r['metrics']['by_class']['made-up'] = dict(r['metrics']['overall'], **{'recall@5': 1.0})

    def forge_epsilon(r):
        v = r['metrics']['overall']['answer_acc'] or 0
        r['metrics']['overall']['answer_acc'] = round(v + 0.0001 if v < 1 else v - 0.0001, 4)

    def forge_padding(r):
        o = r['per_question'][0]
        good = [{'note': 'home', 'quote': 'Entry point of the Larkfield Lab vault.'}] * 4
        o['citations'] = [{'note': 'home', 'quote': 'A sentence no note contains.'}] + good
        # claimed faithfulness unchanged: the fake quote must still be counted against it

    ok = True
    for name, fn in [('inflated headline', forge_headline), ('flipped class metric', forge_class),
                     ('invented quotes', forge_quote), ('dropped question', forge_drop),
                     ('other vault', forge_vault), ('self-awarded CI badge', forge_ci_claim),
                     ('fake quote padded', forge_padding), ('invented class', forge_extra_class),
                     ('+0.0001 nudge', forge_epsilon)]:
        r = copy.deepcopy(base); fn(r)
        errs = bc.check_result(r, q, v, vs, qs, filename=files[0])
        print('%s forgery %-22s -> %s' % ('ok  ' if errs else 'FAIL', name, (errs[0][:90] if errs else 'ACCEPTED')))
        ok &= bool(errs)
    print('SELFTEST', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


def reproduce(system, tol):
    """Rerun `system` from scratch and compare its metrics with the committed file."""
    committed = bc.RESULTS / ('%s.json' % system)
    old = json.loads(committed.read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / ('%s.json' % system)
        subprocess.run([sys.executable, str(HERE / 'run.py'), '--system', system, '--out', str(out)], check=True)
        new = json.loads(out.read_text(encoding='utf-8'))
    worst, diffs = 0.0, []
    for scope, m in [('overall', old['metrics']['overall'])] + [('by_class.' + c, mm) for c, mm in old['metrics']['by_class'].items()]:
        nm = new['metrics']['overall'] if scope == 'overall' else new['metrics']['by_class'][scope.split('.', 1)[1]]
        for k, ov in m.items():
            nv = nm.get(k)
            if ov is None or nv is None or isinstance(ov, int) and k.startswith('n'):
                if ov != nv: diffs.append('%s.%s %r -> %r' % (scope, k, ov, nv)); worst = 1.0
                continue
            d = abs(ov - nv)
            worst = max(worst, d)
            if d > 1e-9: diffs.append('%s.%s %.4f -> %.4f' % (scope, k, ov, nv))
    changed = sum(1 for a, b in zip(old['per_question'], new['per_question']) if a['retrieved'][:10] != b['retrieved'][:10])
    print('reproduce %s: worst metric drift %.4f (tolerance %.4f); %d/%d questions changed their top-10'
          % (system, worst, tol, changed, len(new['per_question'])))
    for d in diffs[:15]:
        print('   ', d)
    if worst > tol:
        print('FAIL the committed numbers for %s do not reproduce' % system); return 1
    print('ok   %s reproduces' % system)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('files', nargs='*')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--reproduce')
    ap.add_argument('--tolerance', type=float, default=0.0)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.reproduce:
        return reproduce(a.reproduce, a.tolerance)
    bad = 0
    import make_vault
    with tempfile.TemporaryDirectory():
        notes, qs = make_vault.build()
    with tempfile.TemporaryDirectory() as td:
        make_vault.write(notes, qs, Path(td) / 'vault', Path(td) / 'q.jsonl')
        if (bc.tree_sha256(Path(td) / 'vault'), bc.file_sha256(Path(td) / 'q.jsonl')) != \
           (bc.tree_sha256(bc.VAULT), bc.file_sha256(bc.QUESTIONS)):
            print('FAIL bench/vault or questions.jsonl differ from make_vault.py output (hand-edited?)'); bad += 1
        else:
            print('ok   vault and questions match the generator')
    files = a.files or sorted(str(p) for p in bc.RESULTS.glob('*.json'))
    if not files:
        print('FAIL no result files in bench/results'); return 1
    bad += check_files(files)
    if not a.files:
        bad += check_leaderboard()
    print('VALID' if not bad else 'INVALID (%d problem(s))' % bad)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
