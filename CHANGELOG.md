# Changelog

What shipped, in plain words. Entries below v0.1.3 were written on 2026-09-05 from the tags and
release notes that already existed — the file itself did not exist until then, so those three are
recorded after the fact rather than backdated to look contemporaneous.

## v0.2.0 — 2026-09-20

**An eval, and the seven upgrades it talked us out of.** `eval/build_gold.py` turns the vault's own
`[[wikilinks]]` into a retrieval test set — no hand labelling, no LLM calls — and `eval/run_eval.py`
scores it as Recall@12 / MRR / nDCG@12 in both retrieval modes, vector-only and vector+graph.

- **The eval imports the pipeline instead of copying it.** `brain_ask.py` keeps exactly the
  behaviour it had (proven byte-for-byte on nine runs over a 10 918-chunk index), but its stages are
  now importable functions — `load_index`, `query_sims`, `retrieve_candidates`, `index_by_basename`,
  `expand_1hop`, `rerank_candidates` — and `main()` is a CLI over them. An eval that re-implements
  the pipeline measures the re-implementation.
- **Four question classes**, because "does the graph help?" has four different answers: `title`,
  `body`, `bridge` (the note's own links are the gold answers) and `temporal` (a `superseded_by`
  frontmatter field points at the replacement).
- **The README now carries real numbers** from the home vault: what two changes improved (bridge
  Recall@12 0.251 -> 0.451) and what seven changes that "obviously should have helped" measurably did
  not — BGE-M3, two rerankers, a BM25 hybrid, a heading-path chunker, `sqlite-vec`, and more.
- **The ruler is testable too.** `EVAL_MUTANT=1` breaks nDCG deliberately, and `tests/test_eval.py`
  must go red under it: a suite that stays green while the metric is broken never tested the metric.

- **An external review panel broke the ruler before anyone else could.** Two engines
  independently found that `ndcg_at` credited the same gold note twice when one filename
  appeared twice in the ranking — which, since wikilinks address notes by filename, happens as
  soon as two folders hold a `foo.md`. `ndcg_at(['foo','foo'], ['foo'])` returned 1.63. Fixed,
  with the published title/temporal numbers corrected downward (0.921 -> 0.890 and 0.638 ->
  0.531); the before/after deltas were unaffected, because the same four questions of 184
  inflated both runs. Also from that review: ambiguous filenames are now excluded from gold
  and reported, a truncated `bridge` gold set records how many links were cut (9 of 30
  questions on our own vault), a run where the graph never fired says so instead of printing
  "the graph adds nothing", and an `EVAL_MUTANT=1` run refuses to write history at all.

**The MCP server got a runtime, examples, and three fixes to how it reports nothing.**
`examples/` now carries a runnable MCP server and its own README, so the recall pipeline is
reachable from an agent harness without reading the source first
([#5](https://github.com/tonydzi/sqlite-graph-memory/issues/5)).

- **The answer file was shared between calls.** Two recalls in flight wrote the same path, so one
  could return the other's text. It is scoped per call now, and `tests/test_mcp_server.py` pins the
  filename so the race can actually fail a test instead of passing on timing.
- **Empty is not an error, and an error is not empty.** An empty answer file used to fall back to
  whatever was on stdout, which meant a log line could come back as a recall result; the server
  returns `isError` instead
  ([#9](https://github.com/tonydzi/sqlite-graph-memory/issues/9)). A file it could not read is now
  reported separately from a recall that genuinely found nothing
  ([#13](https://github.com/tonydzi/sqlite-graph-memory/issues/13)).
- **The paper is out.** A JOSS draft (`paper.md` and its bibliography) with a readiness checklist,
  and `CITATION.cff` plus a cite section pointing at
  [10.5281/zenodo.22639718](https://doi.org/10.5281/zenodo.22639718).
- `brain_ask.py`'s `looks_like_entity` and `_links_in` are covered by tests, and the README
  describes its test coverage by shape instead of by a count that rots.

## v0.1.3 — 2026-09-05

Two ways the indexer quietly answered with the wrong text. Both were found on a real synced vault,
not in review, and both now have tests that run with no model download and no network.

- **Not everything ending in `.md` is a note.** The walk was a plain `rglob('*.md')`, which descends
  into dot-directories. On an Obsidian vault synced with Syncthing that pulled in `.stversions/`
  (Syncthing's old revisions), `*.sync-conflict-*` copies, `.obsidian/` plugin cache and `.git/`
  internals. This is worse than noise: a stale revision of a note is semantically almost identical
  to the live one, so it lands next to it in the top-K and answers the question with outdated
  content. Now skipped. `BRAIN_INDEX_HIDDEN=1` opts back in, because someone will legitimately
  keep notes in a hidden folder.
- **A byte-order mark ate the frontmatter.** Notes were read as plain `utf-8`, so a BOM'd file kept
  a leading U+FEFF and its first line was no longer `---`. The frontmatter regex then saw a note
  with no frontmatter and dropped its date and title, silently. Reading as `utf-8-sig` fixes it.
- Closes [#1](https://github.com/tonydzi/sqlite-graph-memory/issues/1) and
  [#2](https://github.com/tonydzi/sqlite-graph-memory/issues/2) (the smoke test that needs no model).

The two behaviours are now `iter_notes` and `read_note`, which is what made them testable at all —
`tests/test_index_notes.py`, seven cases shaped like a real synced vault.

## v0.1.2 — 2026-08-25

The bilingual regex, explained.

## v0.1.1 — 2026-08-04

The design note, and everything a stranger needs to run the pilot.

## v0.1.0 — 2026-08-04

The pilot as first published on 3 July 2026. Tagged a month later; the code did not change in
between, the tag was simply never cut.
