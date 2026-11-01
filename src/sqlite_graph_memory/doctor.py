"""sgm-doctor: check this machine in plain words, one command, nothing downloaded.

Passport
  in:   environment (BRAIN_INDEX_DIR, HF_HOME), the installed packages
  out:  one line per check, ok / warn / fail, then a verdict; exit 1 if anything failed
  who:  a person who just ran `pip install sqlite-graph-memory` and got an error
  rail: stdlib only. torch is imported inside a try block, because importing it is the check.
"""
import os
import platform
import sqlite3
import sys
from pathlib import Path

from sqlite_graph_memory import brain_ask

OK, WARN, FAIL = 'ok', 'warn', 'fail'
MARK = {OK: '[ok]  ', WARN: '[warn]', FAIL: '[FAIL]'}
FEEDBACK = brain_ask.FEEDBACK_URL


def check_python(version=None):
    v = version or sys.version_info[:3]
    if v < (3, 9):
        return FAIL, 'Python %d.%d.%d is too old.' % v, 'Install Python 3.9 or newer.'
    return OK, 'Python %d.%d.%d.' % v, ''


def check_numpy():
    try:
        import numpy
    except ModuleNotFoundError:
        return FAIL, 'numpy is not installed.', 'pip install sqlite-graph-memory'
    return OK, 'numpy %s.' % numpy.__version__, ''


def _intel_mac(system=None, machine=None):
    return (system or platform.system()) == 'Darwin' and (machine or platform.machine()) == 'x86_64'


def check_embeddings(system=None, machine=None):
    """The heavy half. Without it the graph and the SQLite parts work, indexing and recall do not."""
    try:
        import numpy
        np_major = int(numpy.__version__.split('.')[0])
    except Exception:
        np_major = 0
    try:
        import torch
        import sentence_transformers
    except ModuleNotFoundError as exc:
        return (WARN, 'The embeddings extra is not installed (%s missing).' % exc.name,
                'Only needed to index and recall. Install it with: pip install "sqlite-graph-memory[embeddings]"')
    except Exception as exc:  # torch dies at import on a numpy it was not built for
        why = '%s: %s' % (type(exc).__name__, str(exc).splitlines()[0] if str(exc) else '')
        if _intel_mac(system, machine) and np_major >= 2:
            return (FAIL, 'torch cannot start on this Intel Mac with NumPy 2 (%s).' % why,
                    'pip install "numpy<2"')
        return FAIL, 'torch or sentence-transformers fails to import (%s).' % why, \
            'Reinstall in a fresh virtualenv: pip install --force-reinstall "sqlite-graph-memory[embeddings]"'
    if _intel_mac(system, machine) and np_major >= 2:
        return (WARN, 'Intel Mac with NumPy 2: torch imported, but this pairing is known to crash while indexing.',
                'pip install "numpy<2"')
    return OK, 'torch %s and sentence-transformers %s import cleanly.' % (
        torch.__version__, sentence_transformers.__version__), ''


def _hf_cache_dirs():
    roots = [os.getenv('HF_HUB_CACHE'), os.getenv('SENTENCE_TRANSFORMERS_HOME')]
    hf_home = os.getenv('HF_HOME') or str(Path.home() / '.cache' / 'huggingface')
    roots.append(str(Path(hf_home) / 'hub'))
    return [Path(r) for r in roots if r]


def check_models():
    """Look for the two model folders in the local cache. Never downloads anything."""
    missing = []
    for name in (brain_ask.E5_MODEL, brain_ask.RERANK_MODEL):
        folder = 'models--' + name.replace('/', '--')
        if not any((d / folder).exists() for d in _hf_cache_dirs()):
            missing.append(name)
    if missing:
        return (WARN, 'Not downloaded yet: %s.' % ', '.join(missing),
                'They download on the first index or recall (about 1 GB, needs internet once).')
    return OK, 'Both models are already in the local cache.', ''


def check_index(index_dir=None):
    d = Path(index_dir) if index_dir else brain_ask.INDEX_DIR
    emb, meta = d / '_brain_e5.npy', d / '_brain_e5_meta.pkl'
    if not emb.exists() and not meta.exists():
        return (WARN, 'No index in %s.' % d.resolve(),
                'Run: sgm-index /path/to/your/notes   (set BRAIN_INDEX_DIR to keep the index elsewhere)')
    if emb.exists() != meta.exists():
        left = meta.name if emb.exists() else emb.name
        return FAIL, 'The index is half there: %s is missing.' % left, 'Rebuild it: sgm-index /path/to/your/notes'
    return OK, 'Index found in %s (%.1f MB).' % (d.resolve(), emb.stat().st_size / 1e6), ''


def check_sqlite():
    return OK, 'SQLite %s.' % sqlite3.sqlite_version, ''


def run_checks():
    return [('Python', check_python()), ('SQLite', check_sqlite()), ('numpy', check_numpy()),
            ('Embeddings', check_embeddings()), ('Models', check_models()), ('Index', check_index())]


def render(results):
    lines = []
    for name, (status, what, fix) in results:
        lines.append('%s %-10s %s' % (MARK[status], name, what))
        if fix:
            lines.append('       -> %s' % fix)
    fails = sum(1 for _, r in results if r[0] == FAIL)
    warns = sum(1 for _, r in results if r[0] == WARN)
    if fails:
        lines.append('\n%d problem(s) to fix first. Still stuck? Tell us: %s' % (fails, FEEDBACK))
    elif warns:
        lines.append('\nNothing is broken; %d step(s) above are still to do.' % warns)
    else:
        lines.append('\nAll good.')
    return '\n'.join(lines), fails


def main():
    text, fails = render(run_checks())
    print(text)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
