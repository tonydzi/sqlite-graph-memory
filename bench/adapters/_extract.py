# -*- coding: utf-8 -*-
"""Shared helper for retrieval-only systems: pick one sentence from the top notes as the
answer, and cite it word for word.

sqlite-graph-memory and BM25 retrieve notes; they do not write answers. To score them on
answers and citations at all, both use this deliberately dumb extractor: split the top
notes into sentences, keep the one that shares the most content words with the question.
It is a floor, not a reader. A system with an LLM reader on top should beat it, and the
gap is the point of having the column.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_core import normalize  # noqa: E402

STOP = set('''a an the of in on at to for from by with and or is are was were be been
what which who whom whose when where why how did does do in into than that this these those
it its their them they he she his her our we you your i me my most more first last current
originally set get got under start begin working work lab team tool project like use
'''.split())
TOKEN_RX = re.compile(r"[a-z0-9][a-z0-9.,'-]*[a-z0-9]|[a-z0-9]")
FM_RX = re.compile(r'^---\r?\n.*?\r?\n---\r?\n', re.S)


def tokens(s):
    return {re.sub(r"'s$", '', t).strip(".,") for t in TOKEN_RX.findall(s.lower())} - STOP - {''}


def sentences(raw_note):
    """Sentences of a note body, each already normalized, so a quote of one is verbatim."""
    body = FM_RX.sub('', raw_note, count=1)
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        for s in re.split(r'(?<=[.!?])\s+(?=[A-Z\[-])', normalize(line)):
            if s.strip():
                out.append(s.strip())
    return out


def best_sentence(question, ranked_ids, read_note, depth=3):
    """-> (note id, sentence, overlap) from the top `depth` notes, or (None, None, 0)."""
    qt = tokens(question)
    best = (None, None, 0)
    for nid in ranked_ids[:depth]:
        for s in sentences(read_note(nid)):
            ov = len(qt & tokens(s))
            if ov > best[2]:
                best = (nid, s, ov)
    return best
