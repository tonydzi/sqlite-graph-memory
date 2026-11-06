# -*- coding: utf-8 -*-
"""MemPalace (mempalace on PyPI) with its defaults: local ChromaDB, no API key, no LLM.
https://github.com/milla-jovovich/mempalace (MIT).

- setup: copies the vault into work_dir/vault and calls `mempalace.miner.mine()` in its
  default "projects" mode, the same path as `mempalace mine <dir>`. Palace on disk in work_dir.
- search: `mempalace.searcher.search_memories(question, palace_path, n_results=20)`, default
  candidate strategy ("vector" + its own hybrid rerank). Hits are mapped back to note ids
  through the drawer's source_file.
- No LLM is used anywhere (init --llm and AAAK compression are off).
Answer floor and abstention are the shared ones (best-overlap sentence of top-3 notes;
abstain when nothing is retrieved or no sentence overlaps).
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extract import best_sentence  # noqa: E402

SYSTEMS = {'mempalace': {}}
N_RESULTS = 20


class Adapter:
    def __init__(self):
        self.raw = {}
        self.palace = None

    def info(self):
        import mempalace
        ver = getattr(mempalace, '__version__', None)
        if not ver:
            from importlib.metadata import version
            ver = version('mempalace')
        return {'version': ver, 'url': 'https://github.com/milla-jovovich/mempalace',
                'config': {'ingest': 'miner.mine(), default projects mode, no LLM',
                           'store': 'chromadb (default backend and embedder), on disk',
                           'search': 'searcher.search_memories(n_results=%d), default strategy' % N_RESULTS,
                           'answer': 'extractive: best-overlap sentence of top-3 notes',
                           'abstain': 'no hit, or no sentence shares a content word'}}

    def setup(self, vault_dir, work_dir):
        from mempalace.miner import mine
        src = Path(work_dir) / 'vault'
        shutil.copytree(vault_dir, src)
        self.palace = str(Path(work_dir) / 'palace')
        self.raw = {p.stem.lower(): p.read_text(encoding='utf-8') for p in sorted(src.glob('*.md'))}
        mine(str(src), self.palace, wing_override='bench', respect_gitignore=False)

    def ask(self, question):
        from mempalace.searcher import search_memories
        res = search_memories(question, self.palace, n_results=N_RESULTS)
        hits = res.get('results', []) if isinstance(res, dict) else []
        ranked, seen = [], set()
        for h in hits:
            src = h.get('source_file') or (h.get('metadata') or {}).get('source_file') or ''
            nid = Path(str(src)).stem.lower()
            if nid in self.raw and nid not in seen:
                seen.add(nid)
                ranked.append(nid)
        nid, sent, _ = best_sentence(question, ranked, self.raw.get)
        out = {'retrieved': ranked, 'answer': None, 'citations': []}
        if sent:
            out.update(answer=sent, citations=[{'note': nid, 'quote': sent}])
        return out


def make(**kwargs):
    return Adapter(**kwargs)
