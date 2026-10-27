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


def test_adapter_named_after_installed_package_is_refused(tmp_path):
    """An adapter file named like an installed package (adapters/json.py) shadows it; run.py must say so."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('bench_run_mod', ROOT / 'bench' / 'run.py')
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    clash = tmp_path / 'json.py'
    clash.write_text("SYSTEMS = {'x': {}}\n")
    assert run.shadowed_package(clash) == 'json'
    assert run.shadowed_package(tmp_path / 'no_such_pkg_zz9.py') is None
    # the adapter's own folder never counts: a module found only there is the adapter itself
    own = tmp_path / 'zz_only_here_9.py'
    own.write_text("SYSTEMS = {'y': {}}\n")
    sys.path.insert(0, str(tmp_path))
    try:
        assert run.shadowed_package(own) is None
    finally:
        sys.path.remove(str(tmp_path))
    assert all(run.shadowed_package(p) is None for p in (ROOT / 'bench' / 'adapters').glob('*.py'))


def test_ci_needs_picks_only_marked_systems(tmp_path):
    """The workflow chooses the interpreter per system from `needs:` notes; the badge ignores them."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('bench_ci_needs', ROOT / 'bench' / 'ci_needs.py')
    cn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cn)
    f = tmp_path / 'list.txt'
    f.write_text('# header, needs: ollama is only prose here\n'
                 'bm25\n'
                 'fidelis   # needs: ollama\n'
                 'both      # needs: torch, ollama\n'
                 'other     # needs: torch\n'
                 'prose     # not a needs: ollama line\n', encoding='utf-8')
    assert cn.needing('ollama', f) == ['fidelis', 'both']
    assert cn.needing('torch', f) == ['both', 'other']
    assert cn.needing('gpu', f) == []
    # the live list: the two Ollama systems are marked, and both carry the badge
    live = set(cn.needing('ollama'))
    assert live == {'fidelis', 'mem0-local'}
    assert live <= bc.ci_reproduced()
