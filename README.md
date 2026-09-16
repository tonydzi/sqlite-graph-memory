# sqlite-graph-memory

**Graph RAG on SQLite for AI agents — a working pilot, not a framework.**

The extracted memory layer of a personal "second brain" agent setup: three small Python
scripts — `index_notes.py`, `brain_ask.py`, `turnstate_hook.py` — that give an LLM agent
*associative* recall over a folder of markdown notes.
SQLite is the only database ([`schema.sql`](schema.sql)) and `[[wikilinks]]` are the graph.

Status: **pilot**, and the word is load-bearing: it runs daily in one real setup (a ~100k-note
Obsidian vault driven by Claude Code), but it is deliberately minimal and makes no attempt
to be general — see [What's intentionally missing](#whats-intentionally-missing), which is
where the uneven test coverage is described.
Published as the companion code for an upcoming write-up on lightweight Graph RAG for agents;
the one design note that is already written is [`docs/bitemporal.md`](docs/bitemporal.md).

## Why

Most Graph RAG stacks assume a graph database, an ETL pipeline, and an entity-extraction pass —
HippoRAG ([arXiv:2405.14831](https://arxiv.org/abs/2405.14831)) derives its KG that way, and
Graphiti/Zep maintain a temporal one.
For a single-user agent over a markdown knowledge base, all three are overkill:

- The **graph already exists** — Obsidian/Logseq/plain-markdown users hand-curate their
  edges as `[[wikilinks]]`, and `brain_ask.py` just parses them.
  No extraction pass, and edge quality beats anything a model would mine.
- **SQLite is enough** — the only things worth persisting are derived state (a per-turn
  ledger) and telemetry, and both are defined in [`schema.sql`](schema.sql).
  One file, stdlib driver, zero ops.
- The expensive part of RAG quality is a **reranker**, not graph infrastructure — see the
  rerank step in `brain_ask.py`.

So the pilot's bet: *vector retrieval for entry points, hand-curated wikilinks for
association, a cross-encoder to keep serendipity honest, SQLite for everything that must
persist.* All four steps live in one file, `brain_ask.py`, in that order.

## Architecture

Three layers, three failure-isolated components:

```
markdown notes (with [[wikilinks]])
        │
        ├── index_notes.py      chunk + embed (e5-base) → .npy/.pkl index
        │
        ├── brain_ask.py        recall pipeline:
        │       1. dense retrieve  top-60 chunks, dedup by file
        │       2. --graph         +1-hop wikilink neighbours of top-15 hits (max 40)
        │       3. rerank          cross-encoder → top-12
        │       4. --ab            run vector-only AND vector+graph, diff,
        │                          log the delta to SQLite (ab_recall)
        │
        └── turnstate_hook.py   agent Stop-hook: after every assistant turn,
                                parse the transcript tail and append one row
                                (ask, summary, files, tools, commands, decisions)
                                to SQLite (turns). 0 LLM tokens, pure stdlib.
                                turnstate_show.py = read-only viewer.
```

[`schema.sql`](schema.sql) documents both tables. The graph is **not** materialized in SQL:
`brain_ask.py` parses edges from the notes at query time, bounded to 1 hop / 40 neighbours.
That keeps the graph permanently in sync with the notes at zero maintenance cost; materialize
an edge table only when hop depth or corpus size demands it.

### Design choices that survived contact with reality

- **Graph expansion is candidate generation, not ranking** (`brain_ask.py`). Neighbours go
  into the same reranker pool as vector hits, so an irrelevant linked note gets buried.
  That is what makes 1-hop expansion safe to leave on.
- **Entity gate** (`looks_like_entity` in `brain_ask.py`). A/B telemetry showed graph
  expansion helps *theme* queries ("approaches to agent memory") but hurts *name/tool*
  lookups ("NotebookLM"): a name's card links to everything, so the hop pulls in noise.
  The heuristic in `brain_ask.py` is deliberately conservative, and switches the graph off
  only for name-shaped queries.
- **Crash-safety as a policy.** The graph hop (`brain_ask.py`) and the Stop-hook
  (`turnstate_hook.py`) are both wrapped, so any failure degrades to the previous behavior:
  vector-only recall, no ledger row.
  A memory layer must never make the agent worse than having no memory layer.
- **Measure, don't believe.** `--ab` runs both pipelines on every real query and logs how
  many notes the graph promoted into the top-N, to `ab_recall` in [`schema.sql`](schema.sql).
  Whether "associative memory" earns its keep is a weekly SQL query, not a vibe.

### What's intentionally missing

- No entity lane. The production setup has an extra retrieval lane over people/project
  cards; it is too entangled with personal data to publish.
- No incremental indexing and no packaging. There IS an eval suite now — see
  [Measure before you swap](#measure-before-you-swap--eval) — but it scores retrieval on
  wikilink-derived gold, which is weak supervision and says so out loud. Test coverage is uneven:
  `index_notes.py`, `brain_ask.py` and `mcp_server.py` have unit tests, `turnstate_hook.py`
  has none. Run `pytest -q` for the current count rather than trusting a number written in
  prose here — that is how this file ended up claiming "no tests" while seven were passing.
  The status line above says **pilot** on purpose.

## Quickstart

```bash
pip install -r requirements.txt

# 1. index a folder of markdown notes
python index_notes.py /path/to/notes

# 2. ask, three ways
python brain_ask.py "how do I think about agent memory"
python brain_ask.py --graph "how do I think about agent memory"
python brain_ask.py --ab    "how do I think about agent memory"   # logs the diff to SQLite

# 3. inspect the A/B telemetry
sqlite3 turnstate.db "select ts, query, new_in_top, promoted_via_graph from ab_recall order by id desc limit 10"
```

To enable the per-turn ledger in Claude Code, register `turnstate_hook.py` as a Stop hook
(the exact block is `examples/claude-code-stop-hook.json`), then:

```bash
python turnstate_show.py --stats
```

Configuration is env-only and every variable is listed in `.env.example`; everything
defaults to the current directory. GPU is used automatically if a CUDA torch build is
present, and CPU works fine for small corpora.

### What `--ab` looks like

Output of `brain_ask.py` with `--ab`, on a toy 4-note corpus (real runs use a ~100k-note vault):

```
A/B RECALL: how should agent memory work
(Direct=4 vector / Associative=+0 graph; Associative promoted 0 new notes into top-12, 0 of them via graph)

--- DIRECT memory (vector) ---
 1. [  4.11] agent-memory
 2. [ -6.50] graph-rag
 3. [ -8.66] sqlite-ledger
 4. [ -9.73] cooking

--- ASSOCIATIVE memory (vector+graph) ---
 1. [  4.11] agent-memory
 2. [ -6.50] graph-rag
 3. [ -8.66] sqlite-ledger
 4. [ -9.73] cooking
```

On a corpus this small the graph adds nothing — every note is already in the candidate
pool. The interesting deltas appear at scale, and that is exactly what the `ab_recall`
table in [`schema.sql`](schema.sql) accumulates evidence for.

## Measure before you swap — `eval/`

Every "obvious" upgrade to a RAG pipeline — a newer embedder, a stronger reranker, hybrid
BM25, a different chunker — is a hypothesis. Without a number on *your* corpus, swapping one
in is superstition with a changelog entry. `eval/` is the smallest honest ruler we could
build, and it needs no labelling budget and no LLM calls: **the wikilinks you already wrote
are the relevance labels.**

```bash
python eval/run_eval.py --selftest                       # metrics only: no model, no index
python eval/build_gold.py /path/to/notes --n-per-class 60   # -> eval/gold-<today>.jsonl (freeze it)
python eval/run_eval.py --gold eval/gold-<today>.jsonl --sqlite
```

Four question classes, each probing a different part of the pipeline: `title` (a note's own
title should find it), `body` (its first paragraph should find it), `bridge` (its first
paragraph should surface the notes it `[[links]]` to — this is what the graph is *for*), and
`temporal` (an outdated note's title should surface its replacement, via a
`superseded_by: "[[newer-note]]"` frontmatter field). Every question is scored in both modes,
vector-only and vector+graph, so "does the graph help?" stops being a matter of taste.

`run_eval.py` imports `brain_ask.py` and calls the same functions the agent calls —
`load_index`, `retrieve_candidates`, `expand_1hop`, `rerank_candidates` — so what you measure
is what runs. An eval that re-implements the pipeline measures the re-implementation.
`--sqlite` appends one row per class and mode to a `gold_eval` table, which is how a run six
weeks from now can be diffed against today's.

### What it cost us to find out

Numbers from the home vault (~24k chunks, RU+EN, 2026-09-16), one frozen gold set of 184
questions (60 title / 60 body / 60 bridge / 4 temporal), vector+graph mode:

| class | Recall@12 | nDCG@12 |
|---|---|---|
| title | 0.933 -> **0.950** | 0.890 -> **0.895** |
| body | 0.733 -> **0.800** | 0.600 -> **0.623** |
| bridge | 0.251 -> **0.451** | 0.176 -> **0.279** |
| temporal (n=4 — a smoke test, not a result) | 1.000 -> 1.000 | 0.531 -> 0.531 |

Those title and temporal nDCG figures are **corrected**, and the correction is the most
useful thing on this page. An external review panel reading this module found that `ndcg_at`
credited a gold note twice when the same filename appeared twice in the ranking — and since
wikilinks address notes by filename, that happens whenever two folders hold a `foo.md`.
Reproduced immediately: `ndcg_at(['foo','foo'], ['foo'])` returned **1.63**, above the
maximum the metric can have. It affected 4 of our 184 questions, which inflated title nDCG by
0.031 and temporal by 0.108. Recall@12 and MRR were never affected (one is set-based, the
other stops at the first hit), and because the same four questions inflated both runs, every
before/after delta survived to the fourth decimal: title +0.0056 became +0.0055. The merge
and revert decisions below therefore stand — but the absolute levels we would have quoted
were wrong, and an eval nobody attacks keeps being wrong politely.

Two changes earned that. Cleaning duplicate snapshot copies out of the index (27 095 ->
24 216 chunks) moved the metric by exactly 0.0000 — it buys nothing on the ruler and stops
stale copies eating retrieval slots, which is worth knowing precisely *because* the number is
zero. The change that did move it gave the graph a vote in the **ordering** rather than in the
candidate set: personalized PageRank over the wikilink graph, fused with the reranker by RRF,
with the reranker's top 5 positions protected. The root cause was found by measuring instead
of guessing — 97% of the correct `bridge` answers were *already in the candidate pool* and the
cross-encoder was dropping them, because the relation is structural, not textual. Widening
retrieval, the obvious fix, would have done nothing at all.

Seven other changes were rejected by the same ruler, under one rule fixed in advance:
**a regression worse than 0.01 nDCG@12 on any class reverts the change.**

| tried | verdict |
|---|---|
| BGE-M3 embedder | +0.005…0.007 on title/body/bridge, but temporal −0.011 and Russian −0.013 (English alone +0.080) -> **rejected** |
| Qwen3-Embedding-0.6B | body Recall 0.767 -> 0.683; reindex 2863 s vs 333 s -> **rejected** |
| bge-reranker-v2-m3 | −0.011…−0.018 nDCG on every class except English (+0.041) -> **rejected** |
| Qwen3-Reranker-0.6B | title −0.111 nDCG at 12x the cost -> **rejected** |
| BM25 + RRF hybrid | temporal −0.035 -> **merged switched off** |
| heading-path prefix in every chunk | body −0.023, temporal −0.035 -> **reverted** |
| `sqlite-vec` (vec0) | full scan 11.99 s vs 0.14 s for a plain SQLite table, and its ANN missed 1.2 hits of 60 -> vectors live in SQLite, **the extension does not** |
| bi-temporal demotion of superseded notes | temporal +0.130 nDCG on a 41-question temporal set, but title −0.012 -> **merged behind a flag, default off** |

None of those verdicts is a claim about the models. They are claims about *this corpus with
this ruler*, which is the only kind of claim a swap decision actually needs — and the reason
the module is here rather than in a blog post is that your corpus will disagree with ours.

### What this eval cannot tell you

Wikilink gold is weak supervision: a linked neighbour is not always the note that answers the
question. `bridge` is built *from* links, so any method that walks links is flattered there —
it measures "can the pipeline follow a connection", not "is this the right connection". One
run scores one version of one index, and rebuilding the gold set between two runs makes the
two numbers incomparable, which is the quiet way an eval starts lying. On 60 questions a
±0.02 nDCG difference is inside the noise: we learned that by watching two candidate freezes
of the same pipeline disagree on 33 questions out of 184.

The metric can be broken on purpose. `EVAL_MUTANT=1` makes nDCG return a perfect 1.000 for
everything, and `pytest tests/test_eval.py` **must go red** under it. A suite that stays green
while the metric is broken never tested the metric.

## Roadmap

**Now — [v0.1.1](https://github.com/tonydzi/sqlite-graph-memory/releases).**
The pilot as it runs daily: retrieval pipeline, per-turn ledger and A/B telemetry, all in
`brain_ask.py` and `turnstate_hook.py`, plus the bi-temporal design note. Known defects live in
the [tracker](https://github.com/tonydzi/sqlite-graph-memory/issues) rather than in footnotes
here, and this file deliberately names no issue numbers: they get closed, the prose around them
does not notice, and that drift is the defect this paragraph used to carry twice over.

**Next:**

- **v0.2**: a public benchmark. Half of it now exists as
  [`eval/`](#measure-before-you-swap--eval): a gold builder that needs no hand labelling, four
  query classes, both retrieval modes scored side by side, and the numbers from nine real
  changes on one 24k-chunk vault. What is still missing is the *public* half — a shareable
  synthetic mini-vault (200–500 notes with real wikilinks) so the numbers can be reproduced by
  someone who is not us — and the full ablation matrix (hops × seed caps × neighbour caps ×
  rerank pool × gating policy). The interesting question was never "does graph help" but
  *for which query classes*; `eval/` answers that per class instead of per intuition.
- **Bi-temporal edges** — design note [`docs/bitemporal.md`](docs/bitemporal.md). When you
  materialize the graph instead of parsing it at query time, give every edge a validity
  window (`valid_from` / `valid_to` / `observed_at`) so a rebuild *closes* superseded facts
  instead of deleting them: history is kept, recall prefers the present, and `as_of` queries
  can reconstruct the past. Non-destructive aging by confidence decay. The day-1 A/B result
  — context size ~unchanged, retrieved candidates ~60% fresher by mean age — is written up
  with its caveats in `docs/bitemporal.md`: the volume win is longitudinal, so that is one
  day on one corpus, not a benchmark.
- A write-up on the pattern ("Graph RAG without graph extraction") is in progress.

Every noticeable change ships as a new release, so the
[release feed](https://github.com/tonydzi/sqlite-graph-memory/releases) — not
the commit graph — is where you can see whether "pilot" has stopped being the right word.

## Models

- Embeddings: `intfloat/multilingual-e5-base` (multilingual; the home corpus is RU+EN)
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

Both are set as constants (`E5_MODEL`, `RERANK_MODEL`) in `brain_ask.py` and small enough
to run on a laptop GPU; swap freely, since nothing in the pipeline depends on these
specific models.

## Cite this work

If this repo shows up in your research, cite it via [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button). Academic identity: Anton Dzyatkovsky publishes as **Anton Dziatkovskii** ([ORCID 0000-0001-7408-3054](https://orcid.org/0000-0001-7408-3054)).

## AI contributors

This project is built by a human + AI team, and the git log says so: Claude
writes most of the code, Codex and Grok review it, Gemini feeds the research.
Each is credited on a commit **only if its output changed that commit's
content** — no decorative credits. Lab-wide policy, one source for every repo:
[AI-CONTRIBUTORS.md](https://github.com/tonydzi/.github/blob/main/AI-CONTRIBUTORS.md).

## Cite this work

The paper about this system — retrieval design, pilot telemetry, and the query-class-stratified evaluation protocol — is published: [DOI 10.5281/zenodo.22639718](https://doi.org/10.5281/zenodo.22639718) (preprint, CC BY 4.0). Machine-readable citation: [CITATION.cff](CITATION.cff).

## License

MIT

<!-- CONTACT-FOOTER -->
## Contact

Questions, war stories, or you want to run this on your own fleet:

- 👤 Author: **Anton Dziatkovskii** — Telegram [@tonydzi](https://t.me/tonydzi) · WhatsApp [+1 341 222 9178](https://wa.me/13412229178) · X [@Tony_Stef_](https://x.com/Tony_Stef_)
- 📣 Channels: [@ClawRus](https://t.me/ClawRus) (RU) · [@ClawEng](https://t.me/ClawEng) (EN)
- 🌐 [palo-alto.ai](https://palo-alto.ai) · [Palo Alto AI Research Lab](https://github.com/tonydzi)
- 🧪 **Engineers: want to test-drive this setup?** Message me — I hand out free starter seeds to engineers who test and report back.

## Contributors welcome — and there is a queue

Issues labelled [`accepted`](https://github.com/tonydzi/sqlite-graph-memory/issues?q=is%3Aissue+is%3Aopen+label%3Aaccepted)
are scoped, free to take, and nobody is on them. Comment **"claiming this"** — no permission needed —
and it is yours for 7 days. New here? Start with
[`good first issue`](https://github.com/tonydzi/sqlite-graph-memory/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

**You keep the copyright to your code.** No CLA, no assignment, ever — your contribution goes in
under this repo's existing license, the same terms as ours. We answer every issue and PR within
48 hours, including "no, and here is why"; our silence is our bug, so ping the thread.

Full deal: [CONTRIBUTING.md](https://github.com/tonydzi/.github/blob/main/CONTRIBUTING.md)

---

<!--ecosystem-map:start-->

## 🧩 One piece of a working system

This repository is one piece lifted out of a live operation: one non-technical founder, an AI
cofounder, and a fleet of machines that reach consensus with each other and wake the human only
for money or the irreversible. It was extracted after it survived production, not written as a
demo — and it runs on its own: nothing here phones home to the rest.

**See how the whole thing fits together → [SYSTEM.md](https://github.com/tonydzi/tonydzi/blob/main/SYSTEM.md)**

Its closest neighbours in the **memory** layer: [`second-brain-starter-kit`](https://github.com/tonydzi/second-brain-starter-kit) · [`voice2brain`](https://github.com/tonydzi/voice2brain) · [`compact-canon`](https://github.com/tonydzi/compact-canon)

<!--ecosystem-map:end-->
