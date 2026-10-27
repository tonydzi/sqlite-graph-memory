# -*- coding: utf-8 -*-
"""Fidelis Memory (fidelis-memory on PyPI) by Hermes Labs: zero-LLM agent memory,
BM25 + dense + reciprocal-rank fusion over a mem0 store. https://github.com/hermes-labs-ai/fidelis
(Apache-2.0).

Run as the package documents it: `Memory.from_config(fidelis.config.mem0_config(cfg))`
(chroma + Ollama embeddings), then `recall_hybrid(memory, question, user_id, cfg,
tier="zero_llm")`. The `hybrid` extra (bm25s) is installed so the BM25 leg is on; without
it fidelis silently runs dense-only and tags the method `_nobm25`. The method tag of every
call is checked and the run aborts if the BM25 leg is missing.

Note on what "hybrid" means in this pipeline: Stage 1 pulls its candidate pool from mem0's
dense search (top 20 per sub-query), then runs BM25 and dense ranking over that pool and
fuses them. A note the dense search never returns cannot be rescued by the BM25 leg.

Ingestion uses fidelis's verbatim write path (what its /store endpoint does: the text is
stored as-is, no extraction LLM), one memory per note, `infer=False`. The LLM model in the
config is constructed by mem0 and never called at the zero_llm tier.

Speed shim, results unchanged: on every query fidelis re-embeds its whole candidate pool
(up to ~100 note texts) through Ollama with a hard-coded 30 s timeout, and on a loaded CPU
box that times out and fidelis falls back to unranked candidates. The adapter therefore puts
a small caching proxy in front of Ollama (`ollama_url` points at it): each distinct
(model, text) is embedded once by the real Ollama and the same vector is served again
afterwards. Retrieval logic, models, prefixes and fusion are fidelis's own and untouched.
The one difference from a direct call: a cached vector was computed in a different batch,
which llama.cpp may round differently in the last float digits.

Needs a running Ollama with `nomic-embed-text`, so CI does not rerun this row.
Answer floor and abstention are the shared ones: best-overlap sentence of the top-3 notes;
abstain when nothing is recalled or no sentence overlaps. Fidelis has no abstention rule of
its own at this tier, and the absent_* columns show what that costs.
"""
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extract import best_sentence  # noqa: E402

SYSTEMS = {'fidelis': {}}   # file is not named fidelis.py: that would shadow the package
EMBED_MODEL = 'nomic-embed-text'
LLM_MODEL = 'gemma4:latest'        # constructed by mem0, never called at tier zero_llm
USER = 'bench'
OLLAMA = 'http://localhost:11434'


def start_embed_cache(upstream=OLLAMA):
    """Local HTTP proxy for Ollama that memoizes /api/embed per (model, text). -> (url, server)."""
    cache, lock = {}, threading.Lock()

    def forward(path, body, method):
        req = urllib.request.Request(upstream + path, data=body, method=method,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, r.read()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, data):
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _proxy(self, method):
            body = self.rfile.read(int(self.headers.get('Content-Length') or 0)) if method == 'POST' else None
            try:
                if self.path == '/api/embed' and body:
                    req = json.loads(body)
                    texts = req['input'] if isinstance(req['input'], list) else [req['input']]
                    opts = json.dumps({k: v for k, v in req.items() if k != 'input'}, sort_keys=True)
                    keys = [(opts, t) for t in texts]   # model and every other request field are in the key
                    with lock:
                        miss = list(dict.fromkeys(k for k in keys if k not in cache))
                    if miss:
                        _, raw = forward('/api/embed', json.dumps(dict(req, input=[t for _, t in miss])).encode(), 'POST')
                        vecs = json.loads(raw)['embeddings']
                        if len(vecs) != len(miss):
                            raise ValueError('upstream returned %d vectors for %d texts' % (len(vecs), len(miss)))
                        with lock:
                            cache.update(zip(miss, vecs))
                    with lock:
                        out = {'model': req['model'], 'embeddings': [cache[k] for k in keys]}
                    return self._send(200, json.dumps(out).encode())
                code, data = forward(self.path, body, method)
                return self._send(code, data)
            except urllib.error.HTTPError as e:
                return self._send(e.code, e.read())
            except Exception as e:   # Ollama down, bad payload: an error fidelis sees, never a guess
                return self._send(502, json.dumps({'error': 'embed cache: %s' % e}).encode())

        def do_GET(self):
            self._proxy('GET')

        def do_POST(self):
            self._proxy('POST')

    srv = ThreadingHTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return 'http://127.0.0.1:%d' % srv.server_address[1], srv


class Adapter:
    def __init__(self):
        self.raw = {}
        self.by_text = {}

    def info(self):
        import fidelis
        return {'version': getattr(fidelis, '__version__', '?'), 'url': 'https://github.com/hermes-labs-ai/fidelis',
                'config': {'tier': 'zero_llm', 'retrieval': 'recall_hybrid: BM25 (bm25s) + dense + RRF, limit=20',
                           'embedder': 'ollama/' + EMBED_MODEL, 'store': 'mem0 + chroma (fidelis mem0_config)',
                           'llm': 'ollama/' + LLM_MODEL + ', never called', 'add': 'verbatim, infer=False, one memory per note',
                           'ollama': 'behind a memoizing /api/embed proxy (same vectors, embedded once)',
                           'answer': 'extractive: best-overlap sentence of top-3 notes',
                           'abstain': 'no hit, or no sentence shares a content word'}}

    def setup(self, vault_dir, work_dir):
        from mem0 import Memory
        from fidelis.config import load, mem0_config
        from fidelis.recall_hybrid import _bm25_available
        if not _bm25_available():
            sys.exit('fidelis adapter: install fidelis-memory[hybrid] (bm25s), otherwise the BM25 leg is silently off')
        self.proxy_url, self.proxy = start_embed_cache()
        cfg = load()
        cfg.update(ollama_url=self.proxy_url, store_path=str(Path(work_dir) / 'store'), collection='bench', user_id=USER,
                   embed_model=EMBED_MODEL, llm_model=LLM_MODEL, recall_limit=20)
        self.cfg = cfg
        self.mem = Memory.from_config(mem0_config(cfg))
        self.raw = {p.stem.lower(): p.read_text(encoding='utf-8') for p in sorted(Path(vault_dir).glob('*.md'))}
        for nid, text in self.raw.items():
            self.mem.add(text, user_id=USER, infer=False, metadata={'note': nid})
            key = ' '.join(text.split())
            if key in self.by_text:
                sys.exit('fidelis adapter: notes %s and %s have identical text; hits could not be told apart' % (self.by_text[key], nid))
            self.by_text[key] = nid
        self._prewarm()

    def _prewarm(self):
        """Fill the embed cache with every stored memory, as fidelis will send it (doc prefix,
        2000-char cut), so no query waits on a cold batch under fidelis's 30 s timeout."""
        from fidelis.recall_hybrid import _DOC_PREFIX
        got = self.mem.get_all(filters={'user_id': USER}, top_k=10000)
        stored = [m['memory'] for m in (got.get('results', []) if isinstance(got, dict) else got)]
        if len(stored) != len(self.raw):
            sys.exit('fidelis adapter: mem0 holds %d memories for %d notes' % (len(stored), len(self.raw)))
        texts = [(_DOC_PREFIX + t)[:2000] for t in stored]
        for i in range(0, len(texts), 16):
            body = json.dumps({'model': EMBED_MODEL, 'input': texts[i:i + 16]}).encode()
            req = urllib.request.Request(self.proxy_url + '/api/embed', data=body,
                                         headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=900) as r:
                assert len(json.loads(r.read())['embeddings']) == len(texts[i:i + 16])

    def ask(self, question):
        from fidelis.recall_hybrid import recall_hybrid
        hits, method = recall_hybrid(self.mem, question, user_id=USER, cfg=self.cfg, limit=20, tier='zero_llm')
        if '_nobm25' in method or 'fallback' in method or method == 'no_candidates':
            sys.exit('fidelis adapter: degraded method %r (Ollama or bm25s missing); refusing to score a crippled run' % method)
        ranked, seen = [], set()
        for h in hits:
            nid = self.by_text.get(' '.join(h.get('text', '').split()))
            if nid is None:
                sys.exit('fidelis adapter: a recalled memory maps to no vault note; the store changed the text')
            if nid not in seen:
                seen.add(nid)
                ranked.append(nid)
        nid, sent, ov = best_sentence(question, ranked, self.raw.get)
        out = {'retrieved': ranked, 'answer': None, 'citations': []}
        if hits:
            out['top_score'] = round(float(hits[0].get('score', 0.0)), 4)
        if sent:
            out.update(answer=sent, citations=[{'note': nid, 'quote': sent}])
        return out


def make(**kwargs):
    return Adapter(**kwargs)
