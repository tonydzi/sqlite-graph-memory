"""`--rerank` is accepted by the eval scripts (argparse used to reject it with exit 2)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


def test_run_eval_accepts_rerank_flag():
    r = _run('eval/run_eval.py', '--rerank', '--selftest')
    assert r.returncode == 0, r.stderr


def test_run_ablation_accepts_rerank_flag():
    r = _run('eval/run_ablation.py', '--rerank', '--list-grid')
    assert r.returncode == 0, r.stderr
