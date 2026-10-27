# -*- coding: utf-8 -*-
"""mem0 (mem0ai on PyPI) in local mode: no API key, no cloud.
https://github.com/mem0ai/mem0 (Apache-2.0).

What "local mode" means here, every piece of it chosen by the adapter and listed in config:
- embedder: Ollama `nomic-embed-text` on localhost (mem0's own ollama provider);
- vector store: Qdrant embedded, on disk in work_dir;
- llm: mem0 needs an LLM entry to construct Memory; it is set to a local Ollama model and
  is never called, because every note is added with `infer=False` (raw text stored as one
  memory, no fact extraction). With extraction on, mem0 would rewrite the notes into
  short facts and could not be scored on verbatim quotes.
- search: `Memory.search(question, filters={'user_id': 'bench'}, limit=20)`, hits mapped
  back to note ids through metadata.

Needs a running Ollama with the embedding model pulled, so CI does not rerun this row.
Answer floor and abstention are the shared ones (best-overlap sentence of top-3 notes;
abstain when nothing is retrieved or no sentence overlaps).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extract import best_sentence  # noqa: E402

SYSTEMS = {'mem0-local': {}}
EMBED_MODEL = 'nomic-embed-text'
LLM_MODEL = 'gemma4:latest'        # constructed, never called (infer=False everywhere)
OLLAMA = 'http://localhost:11434'
USER = 'bench'


class Adapter:
    def __init__(self):
        self.raw = {}

    def info(self):
        import mem0
        return {'version': getattr(mem0, '__version__', '?'), 'url': 'https://github.com/mem0ai/mem0',
                'config': {'embedder': 'ollama/' + EMBED_MODEL, 'vector_store': 'qdrant (embedded, on disk)',
                           'llm': 'ollama/' + LLM_MODEL + ', never called', 'add': 'infer=False, one memory per note',
                           'search': 'Memory.search(limit=20), note id from metadata',
                           'answer': 'extractive: best-overlap sentence of top-3 notes',
                           'abstain': 'no hit, or no sentence shares a content word'}}

    def setup(self, vault_dir, work_dir):
        from mem0 import Memory
        cfg = {'embedder': {'provider': 'ollama', 'config': {'model': EMBED_MODEL, 'ollama_base_url': OLLAMA}},
               'llm': {'provider': 'ollama', 'config': {'model': LLM_MODEL, 'ollama_base_url': OLLAMA}},
               'vector_store': {'provider': 'qdrant', 'config': {'path': str(Path(work_dir) / 'qdrant'),
                                                                 'collection_name': 'bench', 'on_disk': True,
                                                                 'embedding_model_dims': 768}}}
        self.mem = Memory.from_config(cfg)
        self.raw = {p.stem.lower(): p.read_text(encoding='utf-8') for p in sorted(Path(vault_dir).glob('*.md'))}
        for nid, text in self.raw.items():
            self.mem.add(text, user_id=USER, infer=False, metadata={'note': nid})

    def ask(self, question):
        res = self.mem.search(question, filters={'user_id': USER}, limit=20)
        hits = res.get('results', []) if isinstance(res, dict) else list(res or [])   # 1.x returned a list
        ranked, seen = [], set()
        for h in hits:
            nid = (h.get('metadata') or {}).get('note')
            if nid in self.raw and nid not in seen:
                seen.add(nid)
                ranked.append(nid)
        nid, sent, ov = best_sentence(question, ranked, self.raw.get)
        top = hits[0].get('score') if hits else None
        out = {'retrieved': ranked, 'answer': None, 'citations': []}
        if top is not None:
            out['top_score'] = round(float(top), 4)
        if sent:
            out.update(answer=sent, citations=[{'note': nid, 'quote': sent}])
        return out


def make(**kwargs):
    return Adapter(**kwargs)
