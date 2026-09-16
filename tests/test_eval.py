# -*- coding: utf-8 -*-
"""What the eval measures, and proof that the measurement itself can fail.

Two things are tested here:

1. the metrics (`recall_at` / `mrr_at` / `ndcg_at`) against hand-computed values;
2. the gold builder — that questions come out of the right notes, that `[[links]]` become
   the `bridge` answers, that `superseded_by` becomes `temporal`, and that a note missing
   from the index never becomes a gold answer (it would be unreachable by construction,
   so it would measure the indexer, not retrieval).

RED-FIRST: run the suite with EVAL_MUTANT=1 and it MUST fail — that flag makes nDCG
return a perfect 1.000 for everything. A green suite under a broken metric would mean
these tests never touched the metric at all.

No model download, no network: everything here is file work and arithmetic.
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval'))

from build_gold import build_gold, first_paragraph, lang_of, read_notes  # noqa: E402
from run_eval import mrr_at, ndcg_at, recall_at  # noqa: E402

PARA = ('This paragraph is long enough to be treated as a real question by the gold builder, '
        'which skips fragments shorter than a hundred characters.')


# ------------------------------------------------------------------ metrics
def test_recall_counts_gold_notes_in_top_k():
    assert recall_at(['a', 'b', 'c'], ['b']) == 1.0
    assert recall_at(['a', 'b', 'c'], ['z']) == 0.0
    assert recall_at(['a', 'b'], ['a', 'z']) == 0.5


def test_recall_respects_the_cutoff():
    ranked = ['x'] * 12 + ['gold']
    assert recall_at(ranked, ['gold'], k=12) == 0.0
    assert recall_at(ranked, ['gold'], k=13) == 1.0


def test_mrr_is_one_over_the_first_hit_rank():
    assert mrr_at(['b', 'a'], ['b']) == 1.0
    assert mrr_at(['a', 'b'], ['b']) == 0.5
    assert mrr_at(['a', 'b'], ['z']) == 0.0


def test_ndcg_is_perfect_only_when_gold_is_first():
    assert ndcg_at(['b'], ['b']) == 1.0
    assert ndcg_at(['b', 'a'], ['b']) > ndcg_at(['a', 'b'], ['b'])


def test_ndcg_is_zero_when_nothing_relevant_was_retrieved():
    # This is the assertion EVAL_MUTANT=1 breaks: the mutant returns 1.0 here.
    assert ndcg_at(['a', 'b', 'c'], ['z']) == 0.0


def test_lang_split_separates_a_bilingual_vault():
    assert lang_of('как измерить recall') == 'ru'
    assert lang_of('how to measure recall') == 'en'


def test_first_paragraph_skips_headings_tables_and_short_fragments():
    body = '# Heading\n\nshort\n\n| a | b |\n\n' + PARA
    assert first_paragraph(body) == PARA


# ------------------------------------------------------------------ gold builder
def _vault(tmp_path):
    """A tiny vault: two linked notes, one superseded note, one note left out of the index."""
    (tmp_path / 'alpha.md').write_text(
        '---\ntitle: "Alpha the first note"\n---\n\n' + PARA +
        '\n\nSee [[beta]] and [[gamma]] for the rest.\n', encoding='utf-8')
    (tmp_path / 'beta.md').write_text('---\ntitle: "Beta the second note"\n---\n\n' + PARA + '\n',
                                      encoding='utf-8')
    (tmp_path / 'gamma.md').write_text('# Gamma the third note\n\n' + PARA + '\n', encoding='utf-8')
    (tmp_path / 'old-rule.md').write_text(
        '---\ntitle: "Old rule about backups"\nsuperseded_by: "[[beta]]"\n---\n\n' + PARA + '\n',
        encoding='utf-8')
    (tmp_path / 'not-indexed.md').write_text('---\ntitle: "Never indexed"\n---\n\n' + PARA + '\n',
                                             encoding='utf-8')
    return tmp_path


INDEXED = {'alpha', 'beta', 'gamma', 'old-rule'}     # note: 'not-indexed' deliberately absent


def test_every_class_is_built_from_the_right_note(tmp_path):
    gold = build_gold(read_notes(_vault(tmp_path), INDEXED), n_per_class=10, seed=1)
    by_cls = {}
    for g in gold:
        by_cls.setdefault(g['cls'], []).append(g)

    title_alpha = next(g for g in by_cls['title'] if g['src'] == 'alpha')
    assert title_alpha['query'] == 'Alpha the first note'
    assert title_alpha['gold'] == ['alpha']

    body_alpha = next(g for g in by_cls['body'] if g['src'] == 'alpha')
    assert body_alpha['query'] == PARA and body_alpha['gold'] == ['alpha']

    bridge_alpha = next(g for g in by_cls['bridge'] if g['src'] == 'alpha')
    assert bridge_alpha['gold'] == ['beta', 'gamma']       # the wikilinks ARE the answer

    temporal = next(iter(by_cls['temporal']))
    assert temporal['src'] == 'old-rule' and temporal['gold'] == ['beta']


def test_a_note_outside_the_index_never_becomes_a_question_or_an_answer(tmp_path):
    vault = _vault(tmp_path)
    (vault / 'alpha.md').write_text(
        (vault / 'alpha.md').read_text(encoding='utf-8') + '\nAlso [[not-indexed]].\n',
        encoding='utf-8')
    gold = build_gold(read_notes(vault, INDEXED), n_per_class=10, seed=1)
    assert all(g['src'] != 'not-indexed' for g in gold)
    assert all('not-indexed' not in g['gold'] for g in gold)


def test_bridge_needs_at_least_two_links(tmp_path):
    vault = _vault(tmp_path)
    (vault / 'beta.md').write_text(
        (vault / 'beta.md').read_text(encoding='utf-8') + '\nOnly [[gamma]].\n', encoding='utf-8')
    gold = build_gold(read_notes(vault, INDEXED), n_per_class=10, seed=1)
    assert {g['src'] for g in gold if g['cls'] == 'bridge'} == {'alpha'}


def test_the_same_seed_gives_the_same_question_set(tmp_path):
    notes = read_notes(_vault(tmp_path), INDEXED)
    a = build_gold(notes, n_per_class=2, seed=7)
    b = build_gold(notes, n_per_class=2, seed=7)
    assert [g['query'] for g in a] == [g['query'] for g in b]


# ------------------------------------------------------------------ the eval CLI itself
def test_build_gold_cli_writes_a_frozen_file(tmp_path):
    """End to end over the CLI, with a stand-in index meta — no embeddings needed."""
    vault = _vault(tmp_path)
    index = tmp_path / 'index'
    index.mkdir()
    (index / '_brain_e5_meta.pkl').write_bytes(pickle.dumps(
        [{'path': str(vault / (s + '.md')), 'title': s, 'snippet': PARA, 'date': ''}
         for s in sorted(INDEXED)]))
    out = tmp_path / 'gold.jsonl'
    env = {**dict(__import__('os').environ), 'BRAIN_INDEX_DIR': str(index)}
    r = subprocess.run([sys.executable, str(ROOT / 'eval' / 'build_gold.py'), str(vault),
                        '--out', str(out)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in out.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert {r_['cls'] for r_ in rows} == {'title', 'body', 'bridge', 'temporal'}
    assert all(r_['gold'] for r_ in rows)


def test_selftest_exits_zero():
    r = subprocess.run([sys.executable, str(ROOT / 'eval' / 'run_eval.py'), '--selftest'],
                       capture_output=True, text=True)
    assert r.returncode == 0 and 'SELFTEST PASS' in r.stdout
