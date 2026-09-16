# AGENTS.md — working in this repo

Written for AI coding agents, and equally readable by a human contributor. Short on purpose.

## What this repo is

The extracted memory layer of a personal second-brain agent setup: vector retrieval for entry
points, hand-curated `[[wikilinks]]` for association, a cross-encoder rerank, and SQLite for the
only two things worth persisting — a per-turn ledger and A/B telemetry.

**Status: pilot.** It runs daily in one real setup and it is deliberately minimal. It is not a
framework and should not grow into one. The interesting claim here is what it *does not* need: no
graph database, no ETL, no entity-extraction pass.

## Stack and layout

- **Python + SQLite (stdlib driver).** `requirements.txt` covers the embedding/rerank models only.
- `index_notes.py` — chunk + embed a markdown folder (e5-base) into a `.npy`/`.pkl` index.
- `brain_ask.py` — the recall pipeline: dense retrieve → optional `--graph` 1-hop wikilink
  expansion → cross-encoder rerank → top-12. `--ab` runs vector-only *and* vector+graph, diffs
  them, and logs the delta to SQLite.
- `turnstate_hook.py` — agent Stop-hook; appends one row per assistant turn. Zero tokens, pure
  stdlib. `turnstate_show.py` is the read-only viewer.
- `eval/build_gold.py` + `eval/run_eval.py` — the retrieval ruler: a gold set built
  from the vault's own `[[wikilinks]]` (four question classes), scored in both modes
  as Recall@12 / MRR / nDCG@12. `run_eval.py` imports `brain_ask.py` rather than
  re-implementing it, so the eval and the agent cannot drift apart.
- `schema.sql` — documents both tables (`turns`, `ab_recall`); `run_eval.py --sqlite`
  adds a third, `gold_eval`, one row per class and mode per run.
- `examples/claude-code-stop-hook.json` — how the hook gets wired.

## How to verify a change

Start with `pytest -q` — the suite needs no model download and no network. Then run the thing
you changed and paste the output:

```bash
pytest -q                                      # file rules, entity gate, link parsing, eval metrics
python index_notes.py <folder-of-markdown>     # build an index over a small sample
python brain_ask.py "<question>" --graph       # recall, with graph expansion
python brain_ask.py "<question>" --ab          # both arms + the logged delta
python turnstate_show.py                       # what the ledger captured
```

**If you touch retrieval, numbers are not optional.** Build a gold set once, keep the file, and
show the table before and after your change:

```bash
python eval/build_gold.py <folder-of-markdown> --n-per-class 60
python eval/run_eval.py --gold eval/gold-<date>.jsonl        # before
python eval/run_eval.py --gold eval/gold-<date>.jsonl        # after
```

Same gold file both times: rebuilding it between runs makes the two tables incomparable. A
regression worse than 0.01 nDCG@12 on any class is a reason to revert, not to explain. And if you
touch `eval/` itself, prove the ruler still bites: `EVAL_MUTANT=1 pytest tests/test_eval.py` must
FAIL.

Use a **synthetic** notes folder — five or six files with a couple of `[[wikilinks]]` between them
is enough to show a behaviour change. Never paste real notes into an issue or PR.

If you touch retrieval, `--ab` is the honest way to show your change helps: it produces a measured
delta, not an opinion.

## Conventions

- **The graph is not materialized in SQL.** Edges are parsed from the notes at query time, bounded
  to 1 hop and 40 neighbours. That keeps the graph in sync with the notes at zero maintenance cost.
  Materializing an edge table is a real option — but only when hop depth or corpus size demands it,
  and with the measurement that shows it.
- **The turn ledger costs zero tokens.** Anything that puts a model call in the hook path is
  rejected on principle: it runs after *every* turn.
- Stdlib and boring SQL. One file, no ORM, no migrations framework.
- The indexer must ignore what is not a note: editor state, sync-conflict copies, versioned
  backups. Silently indexing five copies of one file is a bug we have already had.

## Boundaries — what needs a human

- **Turning the pilot into a framework** — plugin systems, abstraction layers, a config format.
  Open an issue; the answer is usually "not yet, and here is why".
- **Changing the embedding or reranker model.** It invalidates every existing index; needs a
  measured before/after and a note in the README.
- **Schema changes** to `turns` or `ab_recall` — existing data lives in those tables.

## The deal

Your copyright stays yours, there is no CLA, and issues labelled `accepted` are free to take —
comment "claiming this". Full terms:
[CONTRIBUTING.md](https://github.com/tonydzi/.github/blob/main/CONTRIBUTING.md).

If an AI wrote your change, say so in the PR and confirm you ran it. Welcome here — we do it daily.
Unread generated code is the one thing that gets closed on sight.
