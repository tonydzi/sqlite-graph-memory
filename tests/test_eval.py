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

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'eval'))

from build_gold import (build_gold, first_paragraph, indexed_stems,  # noqa: E402
                        lang_of, read_notes)
from run_eval import mrr_at, ndcg_at, recall_at, SMALL_N  # noqa: E402
import run_ablation  # noqa: E402
from run_ablation import (DEFAULT_GRID, format_markdown, low_n_warnings,  # noqa: E402
                          parse_grid, score_grid, summarize_grid, write_csv)

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


def test_ndcg_never_exceeds_one_when_a_name_is_retrieved_twice():
    """Found by an external review panel, reproduced at nDCG = 1.63 on live code.

    Wikilinks address notes by filename, so two notes sharing a filename put the same name in
    the ranking twice. DCG used to count both while IDCG counted the gold set once.
    """
    assert ndcg_at(['foo', 'foo'], ['foo']) == 1.0
    assert ndcg_at(['a', 'foo', 'foo'], ['foo']) <= 1.0
    assert ndcg_at(['foo', 'foo', 'bar'], ['foo', 'bar']) <= 1.0


def test_a_duplicate_hit_is_worth_no_more_than_a_single_one():
    assert ndcg_at(['x', 'foo', 'foo'], ['foo']) == ndcg_at(['x', 'foo', 'zzz'], ['foo'])


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


def test_a_bridge_question_records_how_many_links_were_cut(tmp_path):
    """A low bridge score must be readable: cut gold, or genuinely missed notes?"""
    vault = _vault(tmp_path)
    many = ' '.join('[[n%02d]]' % i for i in range(12))
    (vault / 'alpha.md').write_text('---\ntitle: "Alpha"\n---\n\n' + PARA + '\n\n' + many + '\n',
                                    encoding='utf-8')
    stems = INDEXED | {'n%02d' % i for i in range(12)}
    gold = build_gold(read_notes(vault, stems), n_per_class=10, seed=1)
    bridge = next(g for g in gold if g['cls'] == 'bridge' and g['src'] == 'alpha')
    assert len(bridge['gold']) == 8
    assert bridge['gold_truncated'] == 4


def test_gold_skips_hidden_notes_exactly_like_the_indexer(tmp_path, monkeypatch):
    """build_gold walks the vault with the indexer's own filter (issue #25).

    A root-level `.draft.md` is never indexed, so it must never become a question either,
    even when a stale index still lists its stem. `BRAIN_INDEX_HIDDEN=1` lets it into the
    index, so it must let it into the gold set too: one rule, one place.
    """
    vault = _vault(tmp_path)
    (vault / '.draft.md').write_text('---\ntitle: "Hidden draft"\n---\n\n' + PARA + '\n',
                                     encoding='utf-8')
    stems = INDEXED | {'.draft'}

    monkeypatch.delenv('BRAIN_INDEX_HIDDEN', raising=False)
    assert '.draft' not in {n['stem'] for n in read_notes(vault, stems)}

    monkeypatch.setenv('BRAIN_INDEX_HIDDEN', '1')
    assert '.draft' in {n['stem'] for n in read_notes(vault, stems)}


def test_a_filename_living_in_two_folders_is_reported_as_ambiguous(tmp_path):
    """Found by the review panel: `[[foo]]` names both files, so neither can be gold."""
    vault = _vault(tmp_path)
    (vault / 'sub').mkdir()
    (vault / 'sub' / 'beta.md').write_text('---\ntitle: "Another Beta"\n---\n\n' + PARA + '\n',
                                           encoding='utf-8')
    index = tmp_path / 'index'
    index.mkdir()
    paths = [vault / 'alpha.md', vault / 'beta.md', vault / 'sub' / 'beta.md', vault / 'gamma.md']
    (index / '_brain_e5_meta.pkl').write_bytes(pickle.dumps(
        [{'path': str(p), 'title': p.stem, 'snippet': PARA, 'date': ''} for p in paths]))
    all_stems, ambiguous = indexed_stems(index)
    assert ambiguous == {'beta'}
    assert 'alpha' in all_stems

    out = tmp_path / 'gold.jsonl'
    env = {**dict(__import__('os').environ), 'BRAIN_INDEX_DIR': str(index)}
    r = subprocess.run([sys.executable, str(ROOT / 'eval' / 'build_gold.py'), str(vault),
                        '--out', str(out)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert 'more than one folder' in r.stdout, r.stdout
    rows = [json.loads(l) for l in out.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert all('beta' not in row['gold'] for row in rows)


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


def _run_eval(*args, **kw):
    return subprocess.run([sys.executable, str(ROOT / 'eval' / 'run_eval.py'), *args],
                          capture_output=True, text=True, **kw)


def test_a_wrong_gold_path_is_a_message_not_a_stack_trace(tmp_path):
    """Pointing --gold at nothing is a typo. It should read like one."""
    r = _run_eval('--gold', str(tmp_path / 'nope.jsonl'))
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert 'Traceback' not in out, out
    assert 'build_gold.py' in out, out       # tells you which command makes the missing file


def test_a_gold_file_that_is_not_jsonl_says_so(tmp_path):
    bad = tmp_path / 'bad.jsonl'
    bad.write_text('this is not json\n', encoding='utf-8')
    r = _run_eval('--gold', str(bad))
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert 'Traceback' not in out, out
    assert 'not a gold file' in out, out


def test_an_empty_gold_file_is_refused(tmp_path):
    empty = tmp_path / 'empty.jsonl'
    empty.write_text('', encoding='utf-8')
    r = _run_eval('--gold', str(empty))
    assert r.returncode != 0
    assert 'empty' in (r.stdout + r.stderr)


def test_selftest_exits_zero():
    r = subprocess.run([sys.executable, str(ROOT / 'eval' / 'run_eval.py'), '--selftest'],
                       capture_output=True, text=True)
    assert r.returncode == 0 and 'SELFTEST PASS' in r.stdout


# ------------------------------------------------------------------ run_ablation.py (issue #19)
import numpy as np  # noqa: E402


class _FakeEncoder:
    """Ignores the query text; always returns the same unit vector (dim 2)."""

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        return np.array([[1.0, 0.0]] * len(texts), dtype='float32')


class _FakeReranker:
    """Scores a candidate by what its title names: friend > home > everything else.

    Models 'friend' being the true answer, visible to the reranker only once some
    candidate-selection step (graph expansion) actually puts it in the pool.
    """

    def predict(self, pairs):
        def score(text):
            if 'friend' in text:
                return 100.0
            if 'home' in text:
                return 50.0
            return 10.0
        return [score(text) for _, text in pairs]


def _linked_vault(tmp_path):
    """home.md links to friend.md and distant.md; stranger.md links to nothing.

    Dense similarity (via _FakeEncoder, same vector for every query) ranks home
    highest, stranger second, friend and distant lowest — so at topk=1 only home
    is in the dense pool, and friend/distant can only arrive via graph expansion.
    """
    v = tmp_path / 'vault'
    v.mkdir()
    (v / 'home.md').write_text('See [[friend]] and [[distant]] for more.', encoding='utf-8')
    (v / 'friend.md').write_text('Friend note body.', encoding='utf-8')
    (v / 'distant.md').write_text('Distant note body.', encoding='utf-8')
    (v / 'stranger.md').write_text('Unrelated stranger note.', encoding='utf-8')
    meta = [
        {'path': str(v / 'home.md'), 'title': 'home', 'snippet': 'home snippet', 'date': ''},
        {'path': str(v / 'stranger.md'), 'title': 'stranger', 'snippet': 'stranger snippet', 'date': ''},
        {'path': str(v / 'friend.md'), 'title': 'friend', 'snippet': 'friend snippet', 'date': ''},
        {'path': str(v / 'distant.md'), 'title': 'distant', 'snippet': 'distant snippet', 'date': ''},
    ]
    # emb row i dotted with the fake encoder's [1, 0] query vector must reproduce
    # the home > stranger > {friend, distant} similarity order claimed above.
    emb = np.array([[10.0, 0.0], [5.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype='float32')
    return meta, emb


def test_list_grid_prints_without_loading_any_model_or_index(tmp_path, monkeypatch):
    monkeypatch.setenv('BRAIN_INDEX_DIR', str(tmp_path / 'does-not-exist'))
    r = subprocess.run([sys.executable, str(ROOT / 'eval' / 'run_ablation.py'), '--list-grid'],
                       capture_output=True, text=True, env={**dict(__import__('os').environ),
                       'BRAIN_INDEX_DIR': str(tmp_path / 'does-not-exist')})
    assert r.returncode == 0, r.stderr
    assert '0-hop' in r.stdout


def test_default_grid_covers_a_0hop_baseline_and_varies_each_axis():
    grid = parse_grid('default')
    names = {c['name'] for c in grid}
    assert any(c['hops'] == 0 for c in grid)
    assert len({c['gmax'] for c in grid}) > 1
    assert len({c['topk'] for c in grid}) > 1
    assert any(c['entity_gate'] is False for c in grid)
    assert len(names) == len(grid)          # names are unique


def test_parse_grid_accepts_a_keyval_spec():
    grid = parse_grid('hops=0,topk=10;hops=1,gmax=5,name=mycell')
    assert grid[0]['hops'] == 0 and grid[0]['topk'] == 10
    assert grid[1]['hops'] == 1 and grid[1]['gmax'] == 5 and grid[1]['name'] == 'mycell'


def test_hop_expansion_recovers_a_note_the_dense_pool_missed(tmp_path):
    meta, emb = _linked_vault(tmp_path)
    gold = [{'query': 'where do I keep in touch', 'cls': 'theme', 'gold': ['friend']}]
    grid = parse_grid('hops=0,topk=1,name=0-hop;hops=1,topk=1,gmax=10,name=1-hop')

    rows = score_grid(gold, grid, emb, meta, _FakeEncoder(), _FakeReranker())
    by_cell = {r['cell']: r for r in rows}

    assert by_cell['0-hop']['recall'] == 0.0     # friend never entered the top-1 dense pool
    assert by_cell['1-hop']['recall'] == 1.0     # graph hop pulled it in, reranker surfaced it


def test_gmax_bounds_how_many_linked_neighbours_enter_the_pool(tmp_path):
    meta, emb = _linked_vault(tmp_path)
    gold = [{'query': 'q', 'cls': 'theme', 'gold': ['distant']}]
    grid = parse_grid('hops=1,topk=1,gmax=1,name=tight;hops=1,topk=1,gmax=10,name=loose')

    rows = score_grid(gold, grid, emb, meta, _FakeEncoder(), _FakeReranker())
    by_cell = {r['cell']: r for r in rows}

    assert by_cell['loose']['recall'] == 1.0     # gmax=10: room for both [[friend]] and [[distant]]
    assert by_cell['tight']['recall'] == 0.0     # gmax=1: only the first link (friend) fits


def test_entity_gate_suppresses_the_hop_only_when_enabled(tmp_path):
    meta, emb = _linked_vault(tmp_path)
    gold = [{'query': 'Obsidian', 'cls': 'theme', 'gold': ['friend']}]   # a known ENTITY_TOOLS name
    grid = parse_grid('hops=1,topk=1,gmax=10,entity_gate=1,name=gated;'
                      'hops=1,topk=1,gmax=10,entity_gate=0,name=ungated')

    rows = score_grid(gold, grid, emb, meta, _FakeEncoder(), _FakeReranker())
    by_cell = {r['cell']: r for r in rows}

    assert by_cell['gated']['recall'] == 0.0     # entity-like query: hop suppressed
    assert by_cell['ungated']['recall'] == 1.0   # gate disabled for this cell: hop runs


def test_markdown_and_csv_report_delta_vs_the_0hop_baseline(tmp_path):
    rows = [
        {'cell': '0-hop', 'cls': 'theme', 'recall': 0.0, 'mrr': 0.0, 'ndcg': 0.0},
        {'cell': '1-hop', 'cls': 'theme', 'recall': 1.0, 'mrr': 1.0, 'ndcg': 1.0},
    ]
    grid = [{'name': '0-hop', 'hops': 0}, {'name': '1-hop', 'hops': 1}]
    summary = summarize_grid(rows, grid, ['theme'])

    overall_1hop = next(r for r in summary if r['cell'] == '1-hop' and r['class'] == 'overall')
    assert overall_1hop['delta_ndcg'] == 1.0
    baseline_row = next(r for r in summary if r['cell'] == '0-hop' and r['class'] == 'theme')
    assert baseline_row['delta_ndcg'] == 0.0

    md = format_markdown(summary)
    assert '| 1-hop | theme |' in md and '+1.000' in md

    out = tmp_path / 'ablation.csv'
    write_csv(summary, out)
    body = out.read_text(encoding='utf-8')
    assert 'delta_nDCG@12' in body.splitlines()[0]
    assert '1-hop,theme' in body.replace(' ', '')


def test_low_n_warning_names_the_thin_cell_and_class():
    summary = [{'cell': '0-hop', 'class': 'temporal', 'n': 3, 'recall': 1.0, 'mrr': 1.0,
               'ndcg': 1.0, 'delta_ndcg': None},
              {'cell': '0-hop', 'class': 'overall', 'n': 25, 'recall': 1.0, 'mrr': 1.0,
               'ndcg': 1.0, 'delta_ndcg': None}]
    warnings = low_n_warnings(summary)
    assert len(warnings) == 1
    assert 'temporal' in warnings[0] and '0-hop' in warnings[0] and str(SMALL_N) in warnings[0]


def test_default_grid_varies_ghops_too(tmp_path):
    """R-Gate (PR #4): the default sweep must vary all 5 axes, ghops included —
    it previously left ghops fixed at 15 across every cell."""
    assert len({c['ghops'] for c in DEFAULT_GRID}) > 1


def test_parse_grid_rejects_unsupported_hop_counts():
    """R-Gate (PR #4): score_grid only ever runs the 1-hop expand_1hop, so a cell
    claiming hops=2 would be silently scored as a 1-hop cell and mislabeled."""
    with pytest.raises(ValueError):
        parse_grid('hops=2')
    with pytest.raises(ValueError):
        parse_grid('[{"name": "two-hop", "hops": 2}]')
