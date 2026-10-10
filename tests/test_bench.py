# -*- coding: utf-8 -*-
"""The bench's scorer and gate, without any model: the parts every submission relies on."""
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import bench_core as bc  # noqa: E402
import make_vault  # noqa: E402


def test_generator_matches_committed_vault():
    notes, qs = make_vault.build()
    assert len(notes) >= 200 and len(qs) >= 50
    assert sum(1 for q in qs if not q['answers']) >= 10
    assert len({q['question'] for q in qs}) == len(qs), 'duplicate question text'
    vault = bc.load_vault()
    for q in qs:
        for e in q['evidence']:
            assert any(e in vault[s] for s in q['support']), q['id']


def test_scorer_on_hand_made_outputs():
    q = {'id': 'x', 'cls': 'fact', 'answers': ['2011'], 'support': ['person-ilsa-thorsby'], 'evidence': []}
    vault = bc.load_vault()
    quote = 'Ilsa Thorsby joined the lab in 2011.'
    assert quote in vault['person-ilsa-thorsby']
    good = {'id': 'x', 'retrieved': ['home', 'person-ilsa-thorsby.md'], 'answer': quote,
            'citations': [{'note': 'person-ilsa-thorsby', 'quote': quote}]}
    v = bc.score_one(q, good, vault)
    assert v['recall@5'] == 1.0 and v['answer_ok'] and v['citation_verbatim'] and v['cites_faithful'] == 1
    fake = dict(good, citations=[{'note': 'person-ilsa-thorsby', 'quote': 'Ilsa Thorsby joined the lab in 2011 and left.'}])
    v = bc.score_one(q, fake, vault)
    assert not v['citation_verbatim'] and v['cites_faithful'] == 0
    absent = {'id': 'y', 'cls': 'absent_entity', 'answers': [], 'support': [], 'evidence': []}
    assert bc.score_one(absent, {'answer': None}, vault)['abstained']
    assert not bc.score_one(absent, {'answer': 'unknown'}, vault)['abstained']


def test_answer_match_is_word_bounded():
    assert bc.contains_answer('version 4.2.1 ships', ['4.2.1'])
    assert not bc.contains_answer('in 20117', ['2011'])
    assert bc.contains_answer('Rust.', ['Rust'])


def test_every_committed_result_is_valid():
    out = subprocess.run([sys.executable, str(ROOT / 'bench' / 'validate.py')], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_gate_rejects_forgeries():
    out = subprocess.run([sys.executable, str(ROOT / 'bench' / 'validate.py'), '--selftest'],
                         capture_output=True, text=True)
    assert out.returncode == 0 and 'SELFTEST PASS' in out.stdout, out.stdout


def test_inflated_number_is_caught():
    f = sorted(bc.RESULTS.glob('*.json'))[0]
    r = json.loads(f.read_text(encoding='utf-8'))
    r2 = copy.deepcopy(r)
    r2['metrics']['overall']['answer_acc'] = round((r['metrics']['overall']['answer_acc'] or 0) + 0.05, 4)
    errs = bc.check_result(r2, bc.load_questions(), bc.load_vault(), bc.tree_sha256(bc.VAULT),
                           bc.file_sha256(bc.QUESTIONS), filename=f)
    assert any('answer_acc' in e for e in errs)
