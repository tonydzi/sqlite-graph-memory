# -*- coding: utf-8 -*-
"""bench_core.py — the scorer and the result-file contract. Stdlib only.

One source of truth for every number on the leaderboard: run.py calls score() to write a
result file, and validate.py calls the same score() to re-derive the numbers from the
per-question outputs inside that file. A result whose headline metrics do not match its
own per-question outputs is rejected, so the only way onto the board is to submit the
outputs, not the averages.

WHAT A SYSTEM RETURNS PER QUESTION (the adapter contract)

    retrieved   ranked list of note ids, best first. A note id is the filename without
                `.md` (`person-ilsa-moravec`); paths and the `.md` suffix are tolerated.
    answer      a string, or null to ABSTAIN ("the vault does not say").
    citations   list of {"note": <id>, "quote": <text>}: the passage the answer rests on.

METRICS (all in [0, 1], higher is better unless named "false_")

    recall@5, recall@10   answerable questions: share of support notes in the top k
    answer_acc            answerable: the answer contains a gold answer string
    citation_verbatim     answerable: some citation (a) quotes >= 15 characters that appear
                          VERBATIM in the cited note, (b) cites a support note, and
                          (c) contains a gold answer. "Right answer, real source, exact words."
    citation_faithful     all citations, all questions: share whose quote really is in the
                          note it cites. A made-up quote lowers this, whatever the answer was.
    abstain_correct       unanswerable questions: share where the system abstained
    false_abstain         answerable questions: share where it abstained anyway (lower = better)

Verbatim means: after turning [[target|label]] into `label`, collapsing runs of whitespace
and stripping surrounding whitespace, the quote is a substring of the note. Case counts.
"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
VAULT = HERE / 'vault'
QUESTIONS = HERE / 'questions.jsonl'
RESULTS = HERE / 'results'
SCHEMA = 'memory-bench/result@1'
BENCH_VERSION = '0.1'
MIN_QUOTE = 15
MAX_RETRIEVED = 50
MAX_CITATIONS = 5
CI_REPRODUCED = HERE / 'ci_reproduced.txt'
TOL = 1e-9   # both sides are rounded to 4 places by aggregate(); any visible change must fail

WIKI_RX = re.compile(r'\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]')


# ------------------------------------------------------------------ hashing
def _lf(b):
    return b.replace(b'\r\n', b'\n')


def file_sha256(path):
    return hashlib.sha256(_lf(Path(path).read_bytes())).hexdigest()


def tree_sha256(folder):
    """Hash of every file under the folder, recursively (relative path + content), CRLF-
    insensitive so a Windows checkout hashes the same as the Linux CI runner. Recursive on
    purpose: the sqlite-graph-memory indexer reads subfolders, so a note smuggled into one
    must change the hash."""
    h = hashlib.sha256()
    root = Path(folder)
    for p in sorted(x for x in root.rglob('*') if x.is_file()):
        h.update(p.relative_to(root).as_posix().encode('utf-8') + b'\0' + _lf(p.read_bytes()) + b'\0')
    return h.hexdigest()


# ------------------------------------------------------------------ loading
def load_questions(path=QUESTIONS):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def normalize(text):
    text = WIKI_RX.sub(lambda m: m.group(2) or m.group(1), text or '')
    return re.sub(r'\s+', ' ', text).strip()


def load_vault(folder=VAULT):
    """-> {note id: normalized text}."""
    return {p.stem.lower(): normalize(p.read_text(encoding='utf-8'))
            for p in sorted(Path(folder).glob('*.md'))}


def note_id(x):
    s = str(x or '').replace('\\', '/').rsplit('/', 1)[-1].strip()
    if s.lower().endswith('.md'):
        s = s[:-3]
    return s.lower()


def contains_answer(text, answers):
    t = (text or '').lower()
    for a in answers:
        if re.search(r'(?<![a-z0-9])' + re.escape(a.lower()) + r'(?![a-z0-9])', t):
            return True
    return False


# ------------------------------------------------------------------ scoring
def score_one(q, out, vault):
    """Per-question verdicts for one system output. Pure function."""
    answerable = bool(q['answers'])
    ans = out.get('answer')
    abstained = ans is None or not str(ans).strip()
    retrieved, seen = [], set()
    for r in out.get('retrieved') or []:
        r = note_id(r)
        if r and r not in seen:
            seen.add(r); retrieved.append(r)
    support = set(q['support'])
    cites, seen_c = [], set()
    for c in (out.get('citations') or [])[:MAX_CITATIONS]:   # duplicates must not pad faithfulness
        key = (note_id(c.get('note')), normalize(c.get('quote')))
        if key not in seen_c:
            seen_c.add(key); cites.append(c)
    faithful = []
    verbatim_hit = False
    for c in cites:
        nid, quote = note_id(c.get('note')), normalize(c.get('quote'))
        ok = bool(quote) and nid in vault and quote in vault[nid]
        faithful.append(ok)
        if (ok and answerable and len(quote) >= MIN_QUOTE and nid in support
                and contains_answer(quote, q['answers'])):
            verbatim_hit = True
    v = {'answerable': answerable, 'abstained': abstained,
         'cites': len(cites), 'cites_faithful': sum(faithful)}
    if answerable:
        v['recall@5'] = len(support & set(retrieved[:5])) / len(support)
        v['recall@10'] = len(support & set(retrieved[:10])) / len(support)
        v['answer_ok'] = (not abstained) and contains_answer(str(ans), q['answers'])
        v['citation_verbatim'] = verbatim_hit
    return v


def _mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def aggregate(verdicts):
    ans = [v for v in verdicts if v['answerable']]
    una = [v for v in verdicts if not v['answerable']]
    total_c = sum(v['cites'] for v in verdicts)
    return {
        'n': len(verdicts), 'n_answerable': len(ans), 'n_unanswerable': len(una),
        'recall@5': _mean(v['recall@5'] for v in ans),
        'recall@10': _mean(v['recall@10'] for v in ans),
        'answer_acc': _mean(float(v['answer_ok']) for v in ans),
        'citation_verbatim': _mean(float(v['citation_verbatim']) for v in ans),
        'citation_faithful': round(sum(v['cites_faithful'] for v in verdicts) / total_c, 4) if total_c else None,
        'abstain_correct': _mean(float(v['abstained']) for v in una),
        'false_abstain': _mean(float(v['abstained']) for v in ans),
    }


def score(questions, per_question, vault):
    """-> {'overall': {...}, 'by_class': {cls: {...}}}."""
    outs = {o['id']: o for o in per_question}
    rows = [(q['cls'], score_one(q, outs.get(q['id'], {}), vault)) for q in questions]
    by = {}
    for cls, v in rows:
        by.setdefault(cls, []).append(v)
    return {'overall': aggregate([v for _, v in rows]),
            'by_class': {c: aggregate(vs) for c, vs in sorted(by.items())}}


# ------------------------------------------------------------------ result-file contract
def ci_reproduced():
    """Systems the bench workflow reruns from scratch. Maintainer-owned: a pull request that
    adds a name here is visible in review and must also be runnable without keys."""
    if not CI_REPRODUCED.exists():
        return set()
    return {l.split('#')[0].strip() for l in CI_REPRODUCED.read_text(encoding='utf-8').splitlines()} - {''}


def check_result(res, questions, vault, vault_sha, questions_sha, filename=None):
    """-> list of error strings; empty = valid. Re-derives every metric from per_question."""
    errs = []
    if not isinstance(res, dict):
        return ['the result must be a JSON object']
    for k in ('system', 'reproduce', 'metrics'):
        if not isinstance(res.get(k), dict):
            return ['%s must be a JSON object' % k]

    def need(cond, msg):
        if not cond: errs.append(msg)

    need(res.get('schema') == SCHEMA, 'schema must be %r' % SCHEMA)
    sysd = res.get('system') or {}
    need(isinstance(sysd.get('name'), str) and re.fullmatch(r'[a-z0-9][a-z0-9._-]*', sysd.get('name') or ''),
         'system.name must be a lowercase slug')
    if filename and sysd.get('name'):
        need(Path(filename).stem == sysd['name'], 'file name must be <system.name>.json')
    for k in ('version', 'url'):
        need(isinstance(sysd.get(k), str) and sysd.get(k), 'system.%s is required' % k)
    need(isinstance(res.get('submitted_by'), str) and res.get('submitted_by'), 'submitted_by is required')
    need(isinstance(res.get('date'), str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', res.get('date') or ''), 'date must be YYYY-MM-DD')
    need(res.get('vault_sha256') == vault_sha, 'vault_sha256 does not match bench/vault (stale result or edited vault)')
    need(res.get('questions_sha256') == questions_sha, 'questions_sha256 does not match bench/questions.jsonl')
    rep = res.get('reproduce') or {}
    need(isinstance(rep.get('in_ci'), bool), 'reproduce.in_ci must be true or false')
    if isinstance(rep.get('in_ci'), bool) and sysd.get('name'):
        need(rep['in_ci'] == (sysd['name'] in ci_reproduced()),
             'reproduce.in_ci is decided by bench/ci_reproduced.txt (the list CI reruns), not by the submitter')
    need(isinstance(rep.get('command'), str) and rep.get('command'), 'reproduce.command is required')
    pq = res.get('per_question')
    need(isinstance(pq, list), 'per_question must be a list')
    if not isinstance(pq, list):
        return errs
    ids = [o.get('id') for o in pq if isinstance(o, dict) and isinstance(o.get('id'), str)]
    want = [q['id'] for q in questions]
    need(len(ids) == len(pq), 'every per_question entry must be an object with a string id')
    need(sorted(ids) == sorted(want), 'per_question must cover every question id exactly once '
         '(missing %d, extra %d)' % (len(set(want) - set(ids)), len(set(ids) - set(want))))
    for o in pq:
        if not isinstance(o, dict): continue
        r = o.get('retrieved')
        if not (isinstance(r, list) and all(isinstance(x, str) for x in r) and len(r) <= MAX_RETRIEVED):
            errs.append('%s: retrieved must be a list of <= %d note ids' % (o.get('id'), MAX_RETRIEVED))
        if not (o.get('answer') is None or isinstance(o.get('answer'), str)):
            errs.append('%s: answer must be a string or null' % o.get('id'))
        c = o.get('citations', [])
        if isinstance(c, list) and len(c) > MAX_CITATIONS:
            errs.append('%s: at most %d citations per question' % (o.get('id'), MAX_CITATIONS))
        if not (isinstance(c, list) and all(isinstance(x, dict) and isinstance(x.get('note'), str)
                                            and isinstance(x.get('quote'), str) for x in c)):
            errs.append('%s: citations must be a list of {note, quote} strings' % o.get('id'))
    if errs:
        return errs
    got = score(questions, pq, vault)
    claimed = res.get('metrics') or {}
    # Claims must have exactly the computed shape: an invented class or an extra key is a
    # number nobody derived, even if nothing renders it today.
    if set(claimed) != {'overall', 'by_class'}:
        errs.append('metrics must have exactly the keys overall and by_class')
    if not isinstance(claimed.get('overall'), dict) or set(claimed.get('overall') or {}) != set(got['overall']):
        errs.append('metrics.overall must have exactly the keys %s' % sorted(got['overall']))
    cb = claimed.get('by_class') if isinstance(claimed.get('by_class'), dict) else {}
    if set(cb) != set(got['by_class']):
        errs.append('metrics.by_class classes %s, the question set has %s' % (sorted(cb), sorted(got['by_class'])))
    for cls, m in cb.items():
        if cls in got['by_class'] and (not isinstance(m, dict) or set(m) != set(got['by_class'][cls])):
            errs.append('metrics.by_class.%s has the wrong keys' % cls)
    for scope in ('overall',):
        for k, v in got[scope].items():
            cv = (claimed.get(scope) or {}).get(k, 'missing')
            if not _same(cv, v):
                errs.append('metrics.%s.%s claims %r, the per-question outputs give %r' % (scope, k, cv, v))
    for cls, m in got['by_class'].items():
        for k, v in m.items():
            cv = ((claimed.get('by_class') or {}).get(cls) or {}).get(k, 'missing')
            if not _same(cv, v):
                errs.append('metrics.by_class.%s.%s claims %r, the per-question outputs give %r' % (cls, k, cv, v))
    return errs


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or not isinstance(a, (int, float)):
        return False
    return abs(a - b) <= TOL
