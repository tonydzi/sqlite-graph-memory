# -*- coding: utf-8 -*-
"""tangleclaw-style — in-adapter SQLite FTS5 + BM25 + 1-hop [[wikilink]] + frontmatter lineage reference row.
Reference recipe from https://github.com/Jason-Vaughan/TangleClaw (MIT).

System identity and scope:
TangleClaw is a multi-session agent orchestrator whose markdown continuity capsules and notes track
document supersession via YAML frontmatter (`status: superseded`, `superseded_by: "[[...]]"`) and
cross-note associations via `[[wikilinks]]`. Because TangleClaw itself is a Node.js session host
rather than a standalone Python retrieval package, this row is named `tangleclaw-style` and
implements that markdown retrieval recipe directly in Python + SQLite on top of
`sqlite_graph_memory`'s own graph/fusion primitives (`brain_ask._links_in` and `ppr.rrf_fuse`) so CI
can reproduce the row from scratch without an external daemon or API key (`version: "reference"`):

1. Lexical + FTS5 Reciprocal Rank Fusion:
   Indexes each markdown note into SQLite FTS5 (`notes_fts`) and scores whole-note Okapi BM25
   (`k1=1.5, b=0.75`), fusing the two rankings via `sqlite_graph_memory.ppr.rrf_fuse`.
2. 1-hop outgoing `[[wikilink]]` expansion:
   Parses outgoing wikilinks with `sqlite_graph_memory.brain_ask._links_in` into a SQLite `edges`
   table and includes the best-scoring 1-hop neighbour of each top-3 seed note in the RRF fusion.
3. Frontmatter `superseded_by` lineage resolution:
   Stores YAML frontmatter `status` and `superseded_by` in SQLite (`notes_meta`). Unless the query
   contains a general past-time marker (`originally`, `original`, `earlier`, `initial`, `previous`),
   any retrieved note with `status: superseded` promotes its active `superseded_by` target ahead of
   the superseded note.
4. Shared extractive floor and capitalized-entity abstention gate:
   Uses `_extract.best_sentence` (top-3 notes) for verbatim answers and citations, and abstains when
   a capitalized entity word or multi-word proper name (`CAP_SEQ_RX`) in the question does not appear
   in the vault. `QWORDS` strips only general English sentence-initial question words and
   prepositions (no domain nouns like `Project`, `Team`, `Office`, or `Lab`). Checking multi-word
   proper names in addition to single tokens catches chimeric two-word names whose individual tokens
   appear on different people in the vault (`10/10` on `absent_entity` without domain-noun stopwords).
   Like `bm25` and `sqlite-fts5`, this keyless lexical pipeline has no embedding model or LLM
   reader, so it scores 0.00 on zero-shared-word `paraphrase` queries and 0.00 on `absent_attribute`.
"""
import math
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extract import best_sentence  # noqa: E402
from sqlite_graph_memory import brain_ask as ba, ppr  # noqa: E402

SYSTEMS = {'tangleclaw-style': {}}
K1, B = 1.5, 0.75
GHOPS = 3
FM_RX = re.compile(r'^---\r?\n(.*?)\r?\n---\r?\n', re.S)
WIKILINK_TARGET_RX = re.compile(r'\[\[([^\]\|#]+)')
CAP_SEQ_RX = re.compile(r"\b[A-Z][a-zA-Z'-]*(?:\s+[A-Z][a-zA-Z'-]*)*")
TERM_RX = re.compile(r'\w+')
QWORDS = {
    'In', 'What', 'Which', 'Who', 'When', 'Where', 'Why', 'How',
    'Under', 'On', 'At', 'To', 'From', 'By', 'For',
}
PAST_CUES = {'originally', 'original', 'earlier', 'initial', 'previous'}


def parse_frontmatter(raw_text):
    """Extract top-of-file YAML frontmatter key:value lines as a lowercase dict."""
    m = FM_RX.match(raw_text or '')
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            out[k.strip().lower()] = v.strip().strip('"').strip("'")
    return out


def fts_query(question):
    """Quote every word token and join with OR for safe SQLite FTS5 MATCH queries."""
    return ' OR '.join('"%s"' % t for t in TERM_RX.findall(question))


class Adapter:
    def __init__(self, **kwargs):
        self.raw = {}
        self.con = None

    def info(self):
        return {
            'version': 'reference',
            'url': 'https://www.sqlite.org/fts5.html',
            'config': {
                'implementation': 'in-adapter Python/SQLite reference pipeline for TangleClaw markdown continuity (no external daemon)',
                'reference_repo': 'https://github.com/Jason-Vaughan/TangleClaw',
                'sqlite': sqlite3.sqlite_version,
                'retrieval': 'BM25 (k1=1.5, b=0.75) + SQLite FTS5 + 1-hop wikilinks (brain_ask._links_in), fused by ppr.rrf_fuse',
                'graph_hops': 1,
                'graph_seeds': GHOPS,
                'lineage': 'frontmatter status=superseded promotes superseded_by target unless query has past-time cue',
                'answer': 'extractive: _extract.best_sentence of top-3 notes',
                'abstain': 'capitalized entity word or multi-word proper name absent from vault, or no overlapping sentence',
            },
        }

    def setup(self, vault_dir, work_dir):
        self.raw = {}
        self.con = sqlite3.connect(str(Path(work_dir) / 'tangleclaw.db'))
        self.con.execute('CREATE VIRTUAL TABLE notes_fts USING fts5(id UNINDEXED, body)')
        self.con.execute('CREATE TABLE notes_meta(id TEXT PRIMARY KEY, status TEXT, superseded_by TEXT)')
        self.con.execute('CREATE TABLE edges(src TEXT, dst TEXT)')

        for p in sorted(Path(vault_dir).glob('*.md')):
            nid = p.stem.lower()
            text = p.read_text(encoding='utf-8')
            self.raw[nid] = text
            fm = parse_frontmatter(text)
            sup_m = WIKILINK_TARGET_RX.search(fm.get('superseded_by', ''))
            sup_id = sup_m.group(1).strip().lower() if sup_m else ''
            self.con.execute('INSERT INTO notes_fts(id, body) VALUES (?, ?)', (nid, text))
            self.con.execute(
                'INSERT INTO notes_meta(id, status, superseded_by) VALUES (?, ?, ?)',
                (nid, fm.get('status', '').lower(), sup_id),
            )
            for tgt in ba._links_in(p):
                self.con.execute('INSERT INTO edges(src, dst) VALUES (?, ?)', (nid, tgt.lower()))
        self.con.commit()

        self.tf = {k: Counter(re.findall(r'[a-z0-9]+', v.lower())) for k, v in self.raw.items()}
        self.len = {k: sum(c.values()) for k, c in self.tf.items()}
        self.avg = sum(self.len.values()) / max(len(self.len), 1)
        df = Counter()
        for c in self.tf.values():
            df.update(c.keys())
        n = len(self.tf)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.lower_blob = '\n'.join(self.raw.values()).lower()
        self.meta_map = {
            r[0]: (r[1], r[2])
            for r in self.con.execute('SELECT id, status, superseded_by FROM notes_meta')
        }
        self.out_edges = {}
        for src, dst in self.con.execute('SELECT src, dst FROM edges'):
            if dst in self.raw:
                self.out_edges.setdefault(src, []).append(dst)

    def _has_unknown_entity(self, question):
        for m in CAP_SEQ_RX.finditer(question):
            words = [w for w in m.group(0).split() if w not in QWORDS]
            if not words:
                continue
            phrase = re.sub(r"'s$", '', ' '.join(words).lower())
            if len(words) >= 2 and phrase not in self.lower_blob:
                return True
            for w in words:
                clean = re.sub(r"'s$", '', w.lower())
                if clean not in self.lower_blob:
                    return True
        return False

    def ask(self, question):
        terms = re.findall(r'[a-z0-9]+', question.lower())
        bm25_scores = {}
        for k, c in self.tf.items():
            s = 0.0
            for t in terms:
                f = c.get(t)
                if f:
                    s += self.idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * self.len[k] / self.avg))
            bm25_scores[k] = s
        bm25_ranked = [k for k, s in sorted(bm25_scores.items(), key=lambda x: (-x[1], x[0])) if s > 0][:20]

        q = fts_query(question)
        fts_ranked = []
        if q:
            rows = self.con.execute(
                'SELECT id FROM notes_fts WHERE notes_fts MATCH ? ORDER BY rank LIMIT 20', (q,)
            ).fetchall()
            fts_ranked = [r[0] for r in rows]

        graph_ranked = []
        seen_g = set()
        for idx, nid in enumerate(bm25_ranked):
            if nid not in seen_g:
                seen_g.add(nid)
                graph_ranked.append(nid)
            if idx < GHOPS:
                nbrs = sorted(self.out_edges.get(nid, []), key=lambda x: (-bm25_scores.get(x, 0.0), x))
                for nb in nbrs[:1]:
                    if bm25_scores.get(nb, 0.0) > 0 and nb not in seen_g:
                        seen_g.add(nb)
                        graph_ranked.append(nb)

        fused = ppr.rrf_fuse(bm25_ranked, fts_ranked, graph_ranked) if (bm25_ranked or fts_ranked) else []

        is_past = bool(set(terms) & PAST_CUES)
        if not is_past:
            resolved, seen_res = [], set()
            for nid in fused:
                status, sup_by = self.meta_map.get(nid, ('', ''))
                if status == 'superseded' and sup_by in self.raw and sup_by not in seen_res:
                    seen_res.add(sup_by)
                    resolved.append(sup_by)
                if nid not in seen_res:
                    seen_res.add(nid)
                    resolved.append(nid)
            ranked = resolved[:20]
        else:
            ranked = fused[:20]

        top = float(bm25_scores.get(ranked[0], 0.0)) if ranked else 0.0
        nid, sent, _ = best_sentence(question, ranked, self.raw.get)
        if self._has_unknown_entity(question) or not sent:
            return {'retrieved': ranked, 'answer': None, 'citations': [], 'top_score': round(top, 4)}
        return {
            'retrieved': ranked,
            'answer': sent,
            'citations': [{'note': nid, 'quote': sent}],
            'top_score': round(top, 4),
        }


def make(**kwargs):
    return Adapter(**kwargs)
